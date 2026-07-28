//! Shared shaders and dispatch layouts for `ConvInteger`.

pub const BINDINGS: u32 = 5;
/// 19 u32 fields in the push constant struct.
pub const PUSH_BYTES: u32 = 76;

/// Direct int8→i32 quantized 1D/2D Conv (1D normalized to 2D with W=1):
/// handles groups (depthwise), stride, pad, and dilation; per-tensor zero points
/// are read from buffer without readback. ONNX allows both `uint8` and `int8` for X and W
/// independently: `x_signed`/`w_signed` select sign extension
/// of the bytes and zero points.
pub const CONV_INTEGER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<u32>;   // u8 packed
@group(0) @binding(1) var<storage, read> w: array<u32>;   // u8 packed
@group(0) @binding(2) var<storage, read> azp: array<u32>;
@group(0) @binding(3) var<storage, read> wzp: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<i32>;
struct Push {
    total: u32, c_in: u32, c_out: u32, group: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    phb: u32, pwb: u32, dh: u32, dw: u32, gsi: u32,
    x_signed: u32, w_signed: u32,
}
var<immediate> pc: Push;

/// Byte as signed integer if `signed`, else unsigned.
fn as_signed(raw: u32, signed: u32) -> i32 {
    let v = i32(raw & 0xffu);
    if (signed != 0u && v > 127) { return v - 256; }
    return v;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.total) { return; }
    let ow = o % pc.w_out;
    let t1 = o / pc.w_out;
    let oh = t1 % pc.h_out;
    let t2 = t1 / pc.h_out;
    let m = t2 % pc.c_out;
    let bn = t2 / pc.c_out;
    let gso = pc.c_out / pc.group;
    let g = m / gso;
    let a_zp = as_signed(azp[0], pc.x_signed);
    let w_zp = as_signed(wzp[0], pc.w_signed);
    var acc = 0i;
    for (var cg = 0u; cg < pc.gsi; cg = cg + 1u) {
        let ic = g * pc.gsi + cg;
        for (var r = 0u; r < pc.kh; r = r + 1u) {
            let ih = i32(oh) * i32(pc.sh) - i32(pc.phb) + i32(r) * i32(pc.dh);
            if (ih < 0 || ih >= i32(pc.h_in)) { continue; }
            for (var s = 0u; s < pc.kw; s = s + 1u) {
                let iw = i32(ow) * i32(pc.sw) - i32(pc.pwb) + i32(s) * i32(pc.dw);
                if (iw < 0 || iw >= i32(pc.w_in)) { continue; }
                let xidx = ((bn * pc.c_in + ic) * pc.h_in + u32(ih)) * pc.w_in + u32(iw);
                let widx = ((m * pc.gsi + cg) * pc.kh + r) * pc.kw + s;
                let xv = as_signed(x[xidx >> 2u] >> ((xidx & 3u) * 8u), pc.x_signed);
                let wv = as_signed(w[widx >> 2u] >> ((widx & 3u) * 8u), pc.w_signed);
                acc = acc + (xv - a_zp) * (wv - w_zp);
            }
        }
    }
    out[o] = acc;
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn source_compiles() {
        vk_compute::compile_wgsl(super::CONV_INTEGER).expect("shader ConvInteger valido");
    }
}
