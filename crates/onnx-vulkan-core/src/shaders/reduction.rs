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

/// `ArgMax`, first stage: a workgroup per (row, split) writes the best value it
/// saw and where it saw it.
///
/// The value reductions above give one thread an entire axis, which is the right
/// shape when `rows` is what fills the grid. An LLM's `argmax` is the opposite
/// extreme — one row of 151936 logits — so here the axis is split across
/// workgroups and reduced in a tree, the same trick `matmul_fp32::gemv_split`
/// uses, for the same reason: nothing else fabricates parallelism at `rows = 1`.
///
/// `0xffffffff` marks "no candidate": a lane that scanned nothing, or scanned
/// only `-inf`. It loses every tie-break, and a row where every candidate is
/// invalid answers 0 — which is also the first-occurrence answer ONNX wants for
/// a row of equal values.
pub const ARGMAX_PARTIAL: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> best: array<f32>;
@group(0) @binding(2) var<storage, read_write> best_idx: array<u32>;
struct Push { c: u32, inner: u32, rows: u32, splits: u32, gx: u32 }
var<immediate> pc: Push;

var<workgroup> sv: array<f32, 256>;
var<workgroup> si: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let wg = wid.y * pc.gx + wid.x;
    if (wg >= pc.rows * pc.splits) { return; }
    let split = wg % pc.splits;
    let r = wg / pc.splits;
    // row r = (outer, j): element k at outer*c*inner + k*inner + j
    let j = r % pc.inner;
    let outer = r / pc.inner;
    let base = outer * pc.c * pc.inner + j;

    let per = (pc.c + pc.splits - 1u) / pc.splits;
    let start = split * per;
    let end = min(start + per, pc.c);

    var bv = -3.4028235e38;
    var bi = 0xffffffffu;
    var k = start + lid.x;
    while (k < end) {
        let v = x[base + k * pc.inner];
        // strictly greater: a lane walks k upwards, so the first occurrence wins
        if (v > bv) { bv = v; bi = k; }
        k = k + 256u;
    }
    sv[lid.x] = bv;
    si[lid.x] = bi;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) {
            let o = lid.x + stride;
            if (sv[o] > sv[lid.x] || (sv[o] == sv[lid.x] && si[o] < si[lid.x])) {
                sv[lid.x] = sv[o];
                si[lid.x] = si[o];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) {
        best[wg] = sv[0];
        best_idx[wg] = si[0];
    }
}
"#;

/// `ArgMax`, second stage: one workgroup per row picks among that row's
/// `splits` candidates and writes the index as an int64 pair.
///
/// It runs even at `splits == 1`, where it is a copy: the alternative is a
/// second write-out path in the first stage, and the dispatch costs less than
/// two kernels that can disagree.
pub const ARGMAX_FINAL: &str = r#"
@group(0) @binding(0) var<storage, read> best: array<f32>;
@group(0) @binding(1) var<storage, read> best_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
struct Push { rows: u32, splits: u32, gx: u32 }
var<immediate> pc: Push;

var<workgroup> sv: array<f32, 256>;
var<workgroup> si: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wid.y * pc.gx + wid.x;
    if (r >= pc.rows) { return; }
    let base = r * pc.splits;

    var bv = -3.4028235e38;
    var bi = 0xffffffffu;
    var s = lid.x;
    while (s < pc.splits) {
        let v = best[base + s];
        let i = best_idx[base + s];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
        s = s + 256u;
    }
    sv[lid.x] = bv;
    si[lid.x] = bi;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) {
            let o = lid.x + stride;
            if (sv[o] > sv[lid.x] || (sv[o] == sv[lid.x] && si[o] < si[lid.x])) {
                sv[lid.x] = sv[o];
                si[lid.x] = si[o];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) {
        var idx = si[0];
        if (idx == 0xffffffffu) { idx = 0u; }
        // int64 output, little endian: the high word of an index is always 0
        out[2u * r] = idx;
        out[2u * r + 1u] = 0u;
    }
}
"#;

/// Both `ArgMax` stages take three bindings; the push constants differ.
pub const ARGMAX_BINDINGS: u32 = 3;
pub const ARGMAX_PARTIAL_PUSH_BYTES: u32 = 20;
pub const ARGMAX_FINAL_PUSH_BYTES: u32 = 12;
/// Workgroups the split aims for, the same target `matmul_fp32::gemv_split`
/// picked and calibrated on the same GPU.
pub const ARGMAX_TARGET_WGS: usize = 192;
/// Elements a split is not worth going below: 256 threads with 4 each.
pub const ARGMAX_MIN_SPLIT: usize = 1024;

/// How many ways to split the axis so the grid is worth launching.
pub fn splits(rows: usize, c: usize) -> usize {
    if rows >= ARGMAX_TARGET_WGS {
        return 1;
    }
    (ARGMAX_TARGET_WGS / rows.max(1))
        .min(c.div_ceil(ARGMAX_MIN_SPLIT))
        .max(1)
}

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
        for src in [super::ARGMAX_PARTIAL, super::ARGMAX_FINAL] {
            vk_compute::compile_wgsl(src).expect("valid argmax shader");
        }
    }

    /// The split has to stay inside what the shader assumes: at least one, and
    /// never so many that a split is emptier than a single wave of threads.
    #[test]
    fn the_split_is_bounded_on_both_sides() {
        assert_eq!(super::splits(1, 151936), 149, "one long row: fill the grid");
        assert_eq!(super::splits(1, 1000), 1, "a classifier row is one workgroup");
        assert_eq!(super::splits(4096, 8), 1, "many rows already fill it");
        assert_eq!(super::splits(0, 0), 1);
    }
}
