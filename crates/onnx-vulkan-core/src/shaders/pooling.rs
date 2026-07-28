//! Shared shaders and dispatch layouts for spatial pooling.
//!
//! Single source covers `MaxPool`, `AveragePool`, and `GlobalAveragePool`: the
//! three ops differ in initial value, accumulation, and final step,
//! which are the placeholders `INIT`/`ACC`/`FIN`. Global pooling is when
//! window covers the entire map.

pub const BINDINGS: u32 = 2;
/// 15 u32 fields in the push constant struct.
pub const PUSH_BYTES: u32 = 60;

/// Template: one thread per output element, `kh×kw` window per channel.
/// Out-of-bound input edges do not contribute (`count` tracks valid elements
/// only, which is ONNX semantics with `count_include_pad = 0`).
pub const POOL: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
struct Push {
    total: u32, c: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    phb: u32, pwb: u32, dh: u32, dw: u32, pad_count: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.total) { return; }
    let ow = o % pc.w_out;
    let t1 = o / pc.w_out;
    let oh = t1 % pc.h_out;
    let t2 = t1 / pc.h_out;
    let ch = t2 % pc.c;
    let bn = t2 / pc.c;
    let plane = (bn * pc.c + ch) * pc.h_in * pc.w_in;

    var acc = INIT;
    var count = 0u;
    for (var r = 0u; r < pc.kh; r = r + 1u) {
        let ih = i32(oh) * i32(pc.sh) - i32(pc.phb) + i32(r) * i32(pc.dh);
        if (ih < 0 || ih >= i32(pc.h_in)) { continue; }
        for (var s = 0u; s < pc.kw; s = s + 1u) {
            let iw = i32(ow) * i32(pc.sw) - i32(pc.pwb) + i32(s) * i32(pc.dw);
            if (iw < 0 || iw >= i32(pc.w_in)) { continue; }
            let v = x[plane + u32(ih) * pc.w_in + u32(iw)];
            acc = ACC;
            count = count + 1u;
        }
    }
    // with count_include_pad the mean divides by the whole window
    let n = select(count, pc.kh * pc.kw, pc.pad_count != 0u);
    out[o] = FIN;
}
"#;

/// Maximum over the window.
pub const MAX_INIT: &str = "-3.4028235e38";
pub const MAX_ACC: &str = "max(acc, v)";
pub const MAX_FIN: &str = "acc";

/// Average over the window.
pub const AVG_INIT: &str = "0.0";
pub const AVG_ACC: &str = "acc + v";
pub const AVG_FIN: &str = "acc / f32(max(n, 1u))";

/// Complete source for a variant.
pub fn source(init: &str, acc: &str, fin: &str) -> String {
    POOL.replace("INIT", init).replace("ACC", acc).replace(
        // `FIN` must be replaced last: it appears after the other two
        "FIN", fin,
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn sources_compile() {
        for (init, acc, fin) in [
            (super::MAX_INIT, super::MAX_ACC, super::MAX_FIN),
            (super::AVG_INIT, super::AVG_ACC, super::AVG_FIN),
        ] {
            let src = super::source(init, acc, fin);
            vk_compute::compile_wgsl(&src).expect("shader di pooling valido");
        }
    }
}
