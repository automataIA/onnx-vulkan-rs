//! Shared shaders and dispatch layouts for single-axis reduction.
//!
//! Single source covers `ReduceMean`, `ReduceSum`, and `ReduceMax`: the three ops
//! differ in initial value, accumulation, and final step, which are the
//! three placeholders `INIT`/`ACC`/`FIN`, as in `pooling`.
//!
//! Layout matches `Softmax`: reduced axis has `c` elements separated
//! by `inner`, and each thread produces an output element. With last axis
//! `inner = 1` rows are contiguous.

pub const BINDINGS: u32 = 2;
/// 4 u32 fields in the push constant struct.
pub const PUSH_BYTES: u32 = 16;

/// Template: one thread per output element, reduction over `c` elements.
pub const REDUCE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
// `stride_y` = invocations per grid row (workgroup on x × 256): used to
// unroll the 2D grid, needed because `rows` often exceeds the 65535
// workgroups-per-axis limit.
struct Push { c: u32, inner: u32, rows: u32, stride_y: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.y * pc.stride_y + gid.x;
    if (r >= pc.rows) { return; }
    // row r = (outer, j): element k at outer*c*inner + k*inner + j
    let j = r % pc.inner;
    let outer = r / pc.inner;
    let base = outer * pc.c * pc.inner + j;

    var acc = INIT;
    for (var k = 0u; k < pc.c; k = k + 1u) {
        let v = x[base + k * pc.inner];
        acc = ACC;
    }
    out[r] = FIN;
}
"#;

/// Sum along the axis.
pub const SUM_INIT: &str = "0.0";
pub const SUM_ACC: &str = "acc + v";
pub const SUM_FIN: &str = "acc";

/// Mean along the axis.
pub const MEAN_INIT: &str = "0.0";
pub const MEAN_ACC: &str = "acc + v";
pub const MEAN_FIN: &str = "acc / f32(max(pc.c, 1u))";

/// Maximum along the axis.
pub const MAX_INIT: &str = "-3.4028235e38";
pub const MAX_ACC: &str = "max(acc, v)";
pub const MAX_FIN: &str = "acc";

/// Minimum along the axis.
pub const MIN_INIT: &str = "3.4028235e38";
pub const MIN_ACC: &str = "min(acc, v)";
pub const MIN_FIN: &str = "acc";

/// Complete source for a variant.
pub fn source(init: &str, acc: &str, fin: &str) -> String {
    REDUCE.replace("INIT", init).replace("ACC", acc).replace(
        // `FIN` must be replaced last: it appears after the other two
        "FIN", fin,
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn sources_compile() {
        for (init, acc, fin) in [
            (super::SUM_INIT, super::SUM_ACC, super::SUM_FIN),
            (super::MEAN_INIT, super::MEAN_ACC, super::MEAN_FIN),
            (super::MAX_INIT, super::MAX_ACC, super::MAX_FIN),
            (super::MIN_INIT, super::MIN_ACC, super::MIN_FIN),
        ] {
            let src = super::source(init, acc, fin);
            vk_compute::compile_wgsl(&src).expect("shader di riduzione valido");
        }
    }
}
