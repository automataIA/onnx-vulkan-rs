//! Shared shaders and dispatch layouts for `Resize` (2D spatial).

pub const BINDINGS: u32 = 2;
/// 10 u32 fields + 2 f32 in the push constant struct.
pub const PUSH_BYTES: u32 = 48;

/// ONNX coordinate transformation mode as integer in push constants.
pub const COORD_HALF_PIXEL: u32 = 0;
pub const COORD_ASYMMETRIC: u32 = 1;
pub const COORD_ALIGN_CORNERS: u32 = 2;
pub const COORD_PYTORCH_HALF_PIXEL: u32 = 3;

/// `mode`: 0 = nearest, 1 = linear (bilinear), 2 = cubic (bicubic 4×4).
pub const MODE_NEAREST: u32 = 0;
pub const MODE_LINEAR: u32 = 1;
pub const MODE_CUBIC: u32 = 2;

/// Nearest rounding mode as integer in push constants.
pub const NEAREST_ROUND_PREFER_FLOOR: u32 = 0;
pub const NEAREST_ROUND_PREFER_CEIL: u32 = 1;
pub const NEAREST_FLOOR: u32 = 2;
pub const NEAREST_CEIL: u32 = 3;

/// 2D nearest/bilinear resize on [N, C, H, W]: one thread per output
/// element. Only spatial dimensions are resized, as in
/// real graphs (YOLO/FPN neck upsample).
pub const RESIZE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
struct Push {
    total: u32, nc: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    coord: u32, mode: u32, nearest: u32, cubic_a: f32,
    scale_h: f32, scale_w: f32,
}
var<immediate> pc: Push;

/// Source coordinate corresponding to `o`, per ONNX mode.
fn src_coord(o: u32, scale: f32, len_in: u32, len_out: u32) -> f32 {
    let ox = f32(o);
    if (pc.coord == 1u) {          // asymmetric
        return ox / scale;
    }
    if (pc.coord == 2u) {          // align_corners
        if (len_out <= 1u) { return 0.0; }
        return ox * f32(len_in - 1u) / f32(len_out - 1u);
    }
    if (pc.coord == 3u) {          // pytorch_half_pixel
        if (len_out <= 1u) { return -0.5; }
        return (ox + 0.5) / scale - 0.5;
    }
    return (ox + 0.5) / scale - 0.5;  // half_pixel
}

fn round_nearest(v: f32) -> f32 {
    if (pc.nearest == 2u) { return floor(v); }
    if (pc.nearest == 3u) { return ceil(v); }
    if (pc.nearest == 1u) { return floor(v + 0.5); }
    return ceil(v - 0.5);          // round_prefer_floor
}

fn clampi(v: i32, hi: u32) -> u32 {
    return u32(clamp(v, 0, i32(hi) - 1));
}

/// ONNX cubic weights for fractional offset `t`, in neighbor order
/// `floor-1, floor, floor+1, floor+2`. Keys kernel with
/// coefficient `A = cubic_coeff_a`; four weights sum to 1.
/// `exclude_outside = 0`, no renormalization needed.
fn cubic_weights(t: f32) -> vec4<f32> {
    let a = pc.cubic_a;
    let u = 1.0 - t;
    return vec4<f32>(
        ((a * (t + 1.0) - 5.0 * a) * (t + 1.0) + 8.0 * a) * (t + 1.0) - 4.0 * a,
        ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0,
        ((a + 2.0) * u - (a + 3.0)) * u * u + 1.0,
        ((a * (u + 1.0) - 5.0 * a) * (u + 1.0) + 8.0 * a) * (u + 1.0) - 4.0 * a,
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.total) { return; }
    let ow = o % pc.w_out;
    let t1 = o / pc.w_out;
    let oh = t1 % pc.h_out;
    let plane = (t1 / pc.h_out) * pc.h_in * pc.w_in;

    let fy = src_coord(oh, pc.scale_h, pc.h_in, pc.h_out);
    let fx = src_coord(ow, pc.scale_w, pc.w_in, pc.w_out);

    if (pc.mode == 0u) {
        let iy = clampi(i32(round_nearest(fy)), pc.h_in);
        let ix = clampi(i32(round_nearest(fx)), pc.w_in);
        out[o] = x[plane + iy * pc.w_in + ix];
        return;
    }
    if (pc.mode == 2u) {
        // separable 4×4, out-of-range neighbors replicated at the border (the
        // ONNX reference `_get_neighbor` pads in edge mode)
        let by = i32(floor(fy));
        let bx = i32(floor(fx));
        let wy4 = cubic_weights(fy - floor(fy));
        let wx4 = cubic_weights(fx - floor(fx));
        var acc = 0.0;
        for (var r = 0; r < 4; r = r + 1) {
            let row = plane + clampi(by + r - 1, pc.h_in) * pc.w_in;
            var line = 0.0;
            for (var s = 0; s < 4; s = s + 1) {
                line = line + wx4[s] * x[row + clampi(bx + s - 1, pc.w_in)];
            }
            acc = acc + wy4[r] * line;
        }
        out[o] = acc;
        return;
    }
    let y0 = floor(fy);
    let x0 = floor(fx);
    let wy = fy - y0;
    let wx = fx - x0;
    let iy0 = clampi(i32(y0), pc.h_in);
    let iy1 = clampi(i32(y0) + 1, pc.h_in);
    let ix0 = clampi(i32(x0), pc.w_in);
    let ix1 = clampi(i32(x0) + 1, pc.w_in);
    let v00 = x[plane + iy0 * pc.w_in + ix0];
    let v01 = x[plane + iy0 * pc.w_in + ix1];
    let v10 = x[plane + iy1 * pc.w_in + ix0];
    let v11 = x[plane + iy1 * pc.w_in + ix1];
    out[o] = mix(mix(v00, v01, wx), mix(v10, v11, wx), wy);
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn source_compiles() {
        vk_compute::compile_wgsl(super::RESIZE).expect("shader Resize valido");
    }
}
