//! Shared shaders and dispatch layouts for `GridSample` (2D spatial).
//!
//! Deformable attention sampling: `X` is [N, C, H, W], `grid` is
//! [N, H_out, W_out, 2] with normalized coordinates in [-1, 1] (x, y order), and
//! output is [N, C, H_out, W_out]. Only `mode = bilinear`: other modes
//! are rejected upstream, not approximated.

pub const BINDINGS: u32 = 3;
/// 9 u32 fields in the push constant struct.
pub const PUSH_BYTES: u32 = 36;

/// `padding_mode`, as integer in push constants. `reflection` is not implemented.
pub const PAD_ZEROS: u32 = 0;
pub const PAD_BORDER: u32 = 1;

/// One thread per output element.
pub const GRID_SAMPLE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> grid: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
struct Push {
    total: u32, c: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    align: u32, padding: u32, stride_y: u32,
}
var<immediate> pc: Push;

/// From normalized coordinate in [-1, 1] to source tensor index.
fn unnormalize(v: f32, len: u32) -> f32 {
    if (pc.align == 1u) {
        // the endpoints land on the centers of the first and last element
        return (v + 1.0) * f32(len - 1u) * 0.5;
    }
    return ((v + 1.0) * f32(len) - 1.0) * 0.5;
}

/// Element of `X` with selected border policy. Out of bounds:
/// `zeros` returns 0, `border` clamps to the last valid element.
fn sample(plane: u32, iy: i32, ix: i32) -> f32 {
    var yy = iy;
    var xx = ix;
    if (pc.padding == 1u) {
        yy = clamp(iy, 0, i32(pc.h_in) - 1);
        xx = clamp(ix, 0, i32(pc.w_in) - 1);
    } else if (iy < 0 || iy >= i32(pc.h_in) || ix < 0 || ix >= i32(pc.w_in)) {
        return 0.0;
    }
    return x[plane + u32(yy) * pc.w_in + u32(xx)];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.y * pc.stride_y + gid.x;
    if (o >= pc.total) { return; }
    let ow = o % pc.w_out;
    let t1 = o / pc.w_out;
    let oh = t1 % pc.h_out;
    let t2 = t1 / pc.h_out;
    let ch = t2 % pc.c;
    let bn = t2 / pc.c;

    // grid is [N, H_out, W_out, 2], order (x, y): does not depend on the channel
    let g = ((bn * pc.h_out + oh) * pc.w_out + ow) * 2u;
    let fx = unnormalize(grid[g], pc.w_in);
    let fy = unnormalize(grid[g + 1u], pc.h_in);

    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let tx = fx - floor(fx);
    let ty = fy - floor(fy);

    let plane = (bn * pc.c + ch) * pc.h_in * pc.w_in;
    let v00 = sample(plane, y0, x0);
    let v01 = sample(plane, y0, x0 + 1);
    let v10 = sample(plane, y0 + 1, x0);
    let v11 = sample(plane, y0 + 1, x0 + 1);
    let top = mix(v00, v01, tx);
    let bot = mix(v10, v11, tx);
    out[o] = mix(top, bot, ty);
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn source_compiles() {
        vk_compute::compile_wgsl(super::GRID_SAMPLE).expect("valid GridSample shader");
    }
}
