//! Shared shaders and dispatch layouts for f32 `MatMul`.
//!
//! Two geometries, picked per shape by [`prefer_blocked`]. The blocked one is
//! several times faster per useful FLOP but computes a whole 64×64 tile even
//! for one row of output, so a graph whose matrices are thin — a transformer
//! run at sequence length 1, say — is faster on the narrow one.

pub const TILE_SIZE: u32 = 64;
pub const SMALL_TILE_SIZE: u32 = 16;
pub const BINDINGS: u32 = 3;
pub const PUSH_BYTES: u32 = 112;

/// Whether the register-blocked kernel is worth its padding on `m`×`n`.
///
/// Both kernels round the output up to their tile, so the work each one really
/// does is that padded area. The blocked kernel measured ~5× the throughput of
/// the narrow one on tile-filling shapes; requiring it to stay within 3× the
/// padded area keeps a margin and still sends every shape that fills a tile —
/// including `n = 64`, one tile column — down the fast path.
pub fn prefer_blocked(m: usize, n: usize) -> bool {
    let padded = |t: usize| m.div_ceil(t) * t * n.div_ceil(t) * t;
    padded(TILE_SIZE as usize) <= 3 * padded(SMALL_TILE_SIZE as usize)
}

/// Register-blocked f32 MatMul with shared memory + ONNX batch broadcasting.
///
/// A: `[ba..., M, K]`, B: `[bb..., K, N]` → out: `[broadcast(ba,bb)..., M, N]`.
/// `wid.z` indexes the batch; batch strides (in matrix units, 0 on broadcast
/// dimensions) arrive in push constants. At the borders (M/N/K) zeros are
/// loaded → null contribution.
///
/// A 256-thread workgroup stages a 64×64 output tile and each thread keeps a
/// 4×4 micro-tile in registers: 8 shared reads feed 16 FMAs, where the previous
/// one-output-per-thread kernel spent 2 reads per FMA. On the shapes ORT leaves
/// as `MatMul` in sam3 ViT-H — the attention products, everything biased having
/// been fused into `Gemm` — that measured 1.2 → 6.2 TFLOP/s.
///
/// The geometry was chosen by measurement (`vk-compute --example
/// matmul_tiling`), not by copying `Gemm`: two of those four shapes have N=64,
/// a single tile column. Parallelism still comes out fine because the batch
/// dimension carries it (1296 workgroups against 46 SMs), and 64 divides every
/// M, N and K involved, so nothing is wasted on partial tiles. A 128×64 tile
/// (8×4 micro-tile) measured 3% faster but divides none of them.
pub const MATMUL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Push {
    m: u32, k: u32, n: u32, rank: u32,
    os0: vec4<u32>, os1: vec4<u32>,
    as0: vec4<u32>, as1: vec4<u32>,
    bs0: vec4<u32>, bs1: vec4<u32>,
}
var<immediate> pc: Push;

fn dim(v0: vec4<u32>, v1: vec4<u32>, d: u32) -> u32 {
    if (d < 4u) { return v0[d]; }
    return v1[d - 4u];
}

const TILE = 64u;   // output block per workgroup
const KSTEP = 16u;  // slice of K held in shared memory per iteration
var<workgroup> as_tile: array<f32, 1024>;
var<workgroup> bs_tile: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.y * 16u + lid.x;    // 0..255
    let row0 = wid.y * TILE;
    let col0 = wid.x * TILE;

    // batch decomposition → matrix offset for A and B
    var rem = wid.z;
    var ia = 0u;
    var ib = 0u;
    for (var d = 0u; d < pc.rank; d = d + 1u) {
        let os = dim(pc.os0, pc.os1, d);
        let c = rem / os;
        rem = rem % os;
        ia = ia + c * dim(pc.as0, pc.as1, d);
        ib = ib + c * dim(pc.bs0, pc.bs1, d);
    }
    let a_base = ia * pc.m * pc.k;
    let b_base = ib * pc.k * pc.n;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let ntiles = (pc.k + KSTEP - 1u) / KSTEP;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        let k0 = t * KSTEP;
        // --- stage A: 64×16 values, 4 per thread
        for (var s = 0u; s < 4u; s = s + 1u) {
            let l = tid + s * 256u;
            let gr = row0 + l / KSTEP;
            let gk = k0 + l % KSTEP;
            var v = 0.0;
            if (gr < pc.m && gk < pc.k) { v = a[a_base + gr * pc.k + gk]; }
            as_tile[l] = v;
        }
        // --- stage B: 16×64 values, 4 per thread
        for (var s = 0u; s < 4u; s = s + 1u) {
            let l = tid + s * 256u;
            let gk = k0 + l / TILE;
            let gc = col0 + l % TILE;
            var v = 0.0;
            if (gk < pc.k && gc < pc.n) { v = b[b_base + gk * pc.n + gc]; }
            bs_tile[l] = v;
        }
        workgroupBarrier();
        // --- 4 scalars of A + 4 of B per 16 FMAs
        let arow = lid.y * 4u;
        let bcol = lid.x * 4u;
        for (var kk = 0u; kk < KSTEP; kk = kk + 1u) {
            let bo = kk * TILE + bcol;
            let bvec = vec4<f32>(bs_tile[bo], bs_tile[bo + 1u], bs_tile[bo + 2u], bs_tile[bo + 3u]);
            acc0 = fma(vec4<f32>(as_tile[(arow + 0u) * KSTEP + kk]), bvec, acc0);
            acc1 = fma(vec4<f32>(as_tile[(arow + 1u) * KSTEP + kk]), bvec, acc1);
            acc2 = fma(vec4<f32>(as_tile[(arow + 2u) * KSTEP + kk]), bvec, acc2);
            acc3 = fma(vec4<f32>(as_tile[(arow + 3u) * KSTEP + kk]), bvec, acc3);
        }
        workgroupBarrier();
    }

    // --- write the 4×4 micro-tile
    for (var i = 0u; i < 4u; i = i + 1u) {
        let row = row0 + lid.y * 4u + i;
        if (row >= pc.m) { continue; }
        var accv = acc0;
        if (i == 1u) { accv = acc1; }
        if (i == 2u) { accv = acc2; }
        if (i == 3u) { accv = acc3; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            let col = col0 + lid.x * 4u + j;
            if (col >= pc.n) { continue; }
            out[(wid.z * pc.m + row) * pc.n + col] = accv[j];
        }
    }
}
"#;

/// The same MatMul on a 16×16 tile, one output per thread.
///
/// Kept for the shapes [`prefer_blocked`] rejects: it is the slower kernel per
/// FLOP, but it rounds the output up to 16 instead of 64, which is what matters
/// when a matrix is only a few rows tall.
pub const MATMUL_SMALL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Push {
    m: u32, k: u32, n: u32, rank: u32,
    os0: vec4<u32>, os1: vec4<u32>,
    as0: vec4<u32>, as1: vec4<u32>,
    bs0: vec4<u32>, bs1: vec4<u32>,
}
var<immediate> pc: Push;

fn dim(v0: vec4<u32>, v1: vec4<u32>, d: u32) -> u32 {
    if (d < 4u) { return v0[d]; }
    return v1[d - 4u];
}

const TILE = 16u;
var<workgroup> as_tile: array<f32, 256>;
var<workgroup> bs_tile: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let ty = lid.y;
    let tx = lid.x;
    let row = wid.y * TILE + ty;
    let col = wid.x * TILE + tx;

    // batch decomposition → matrix offset for A and B
    var rem = wid.z;
    var ia = 0u;
    var ib = 0u;
    for (var d = 0u; d < pc.rank; d = d + 1u) {
        let os = dim(pc.os0, pc.os1, d);
        let c = rem / os;
        rem = rem % os;
        ia = ia + c * dim(pc.as0, pc.as1, d);
        ib = ib + c * dim(pc.bs0, pc.bs1, d);
    }
    let a_base = ia * pc.m * pc.k;
    let b_base = ib * pc.k * pc.n;

    var acc = 0.0;
    let ntiles = (pc.k + TILE - 1u) / TILE;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        let ak = t * TILE + tx;
        if (row < pc.m && ak < pc.k) {
            as_tile[ty * TILE + tx] = a[a_base + row * pc.k + ak];
        } else {
            as_tile[ty * TILE + tx] = 0.0;
        }
        let bk = t * TILE + ty;
        if (bk < pc.k && col < pc.n) {
            bs_tile[ty * TILE + tx] = b[b_base + bk * pc.n + col];
        } else {
            bs_tile[ty * TILE + tx] = 0.0;
        }
        workgroupBarrier();
        for (var i = 0u; i < TILE; i = i + 1u) {
            acc = acc + as_tile[ty * TILE + i] * bs_tile[i * TILE + tx];
        }
        workgroupBarrier();
    }
    if (row < pc.m && col < pc.n) {
        out[(wid.z * pc.m + row) * pc.n + col] = acc;
    }
}
"#;

// ------------------------------------------------------------------ GEMV

/// Columns handled by one GEMV workgroup, of its 256 threads.
pub const GEMV_COLS: u32 = 32;
/// Workgroups the split is sized to launch; see [`gemv_split`].
pub const GEMV_TARGET_WGS: u32 = 192;
/// Ceiling on the split, so the partials buffer stays `32 * N` floats.
pub const GEMV_MAX_SPLIT: u32 = 32;
/// Smallest slice of K a split may leave a workgroup: 8 lanes × 8 iterations.
/// roberta's `K = 768` at its measured-best split of 8 sits exactly at 96, so a
/// looser bound would clamp the shape this was calibrated on.
const GEMV_MIN_K_PER_SPLIT: usize = 64;
/// Below this there is nothing to split; the shape stays on `MATMUL_SMALL`.
const GEMV_MIN_K: usize = 128;
pub const GEMV_BINDINGS: u32 = 3;
pub const GEMV_REDUCE_BINDINGS: u32 = 2;
pub const GEMV_PUSH_BYTES: u32 = 16;

/// How many ways to split `K` for a matrix-vector product, or `None` to leave
/// the shape to [`MATMUL_SMALL`].
///
/// At `M = 1` neither tiled kernel has enough output to fill the machine:
/// `MATMUL_SMALL` launches `ceil(N/16)` workgroups — 48 for `N = 768`, about one
/// per SM on a 4070 — and throws away 15 of every 16 threads. Thinning the tile
/// alone makes it worse, not better: a 1×256 kernel wastes nothing and measured
/// **0.42×**, because the workgroup count fell to 3. The parallelism has to be
/// fabricated along `K`, which no output element supplies, at the cost of a
/// second pass over the partials.
///
/// The regime is bandwidth-bound (roberta at `seq_len = 1`: 170 MFLOP against
/// 340 MB of weight reads), so the target is a grid that saturates the memory
/// system. Measured across roberta's three real geometries
/// (`--example gemv`), throughput peaks at **~192 workgroups** on all of them
/// and falls off on both sides — 768 for `N = 768` is 2.6× where 192 is 3.7×.
/// So the split is whatever brings `ceil(N / GEMV_COLS)` up to that:
///
/// | shape | base WGs | split | vs `MatMul16` |
/// |---|---|---|---|
/// | k768 × n768 | 24 | 8 | 3.72× |
/// | k768 × n3072 | 96 | 2 | 4.68× |
/// | k3072 × n768 | 24 | 8 | 8.92× |
///
/// `GEMV_TARGET_WGS` is calibrated to this GPU's 46 SMs, like `conv::WG_FLOOR`;
/// re-run the example on a different target. Small shapes are refused rather
/// than split thin: below `GEMV_MIN_K_PER_SPLIT` per workgroup the reduction
/// costs more than the loop it shortens.
pub fn gemv_split(m: usize, k: usize, n: usize, batch: usize) -> Option<u32> {
    // No batch-stride decomposition in the kernel: at M = 1 the batched shapes
    // are attention's per-head products, which are far too small to want this.
    if m != 1 || batch > 1 || k < GEMV_MIN_K || n < GEMV_COLS as usize {
        return None;
    }
    let base = (n as u32).div_ceil(GEMV_COLS);
    let split = GEMV_TARGET_WGS
        .div_ceil(base)
        .min(GEMV_MAX_SPLIT)
        .min((k / GEMV_MIN_K_PER_SPLIT) as u32)
        .max(1);
    Some(split)
}

/// `[1, K] × [K, N]`, with `K` split across `pc.split` workgroups.
///
/// One workgroup covers `GEMV_COLS` columns of one slice of `K`; its 256
/// threads are `256 / GEMV_COLS` lanes per column, strided over the slice and
/// tree-reduced in shared memory. `col = col0 + tid % COLS` on purpose:
/// consecutive threads take consecutive columns, so a lane's `b[k * n + col]`
/// is coalesced across the row — the only access that matters at 340 MB of
/// weights. The `a` vector is broadcast, every column wanting the same `a[k]`,
/// and is left to the cache rather than staged: it is ≤ 12 KB.
///
/// Writes `[split, N]` partials, which [`GEMV_REDUCE`] sums when `split > 1`.
/// With `split == 1` that is exactly the output, and no second pass runs.
pub const GEMV: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Push { k: u32, n: u32, split: u32, pad: u32 }
var<immediate> pc: Push;

const COLS = 32u;
const LANES = 8u;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let col = wid.x * COLS + tid % COLS;
    let lane = tid / COLS;

    // this workgroup's slice of K
    let kper = (pc.k + pc.split - 1u) / pc.split;
    let kstart = wid.y * kper;
    var kend = kstart + kper;
    if (kend > pc.k) { kend = pc.k; }

    var acc = 0.0;
    if (col < pc.n) {
        for (var k = kstart + lane; k < kend; k = k + LANES) {
            acc = fma(a[k], b[k * pc.n + col], acc);
        }
    }
    red[tid] = acc;
    workgroupBarrier();

    // tree-reduce the LANES partials of each column; lane 0 holds the result
    for (var s = LANES / 2u; s > 0u; s = s / 2u) {
        if (lane < s) { red[tid] = red[tid] + red[tid + s * COLS]; }
        workgroupBarrier();
    }
    if (lane == 0u && col < pc.n) {
        out[wid.y * pc.n + col] = red[tid];
    }
}
"#;

/// Sums the `split` partial vectors [`GEMV`] leaves. Dispatched only when
/// `split > 1`.
pub const GEMV_REDUCE: &str = r#"
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Push { k: u32, n: u32, split: u32, pad: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= pc.n) { return; }
    var acc = 0.0;
    for (var s = 0u; s < pc.split; s = s + 1u) {
        acc = acc + partials[s * pc.n + col];
    }
    out[col] = acc;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_compiles() {
        vk_compute::compile_wgsl(MATMUL).expect("shader MatMul f32 valido");
        vk_compute::compile_wgsl(MATMUL_SMALL).expect("shader MatMul f32 16×16 valido");
        vk_compute::compile_wgsl(GEMV).expect("shader GEMV valido");
        vk_compute::compile_wgsl(GEMV_REDUCE).expect("shader GEMV reduce valido");
    }

    #[test]
    fn gemv_split_targets_the_measured_workgroup_count() {
        // The three geometries roberta runs at seq_len = 1, and the split each
        // one measured fastest at.
        assert_eq!(gemv_split(1, 768, 768, 1), Some(8));
        assert_eq!(gemv_split(1, 768, 3072, 1), Some(2));
        assert_eq!(gemv_split(1, 3072, 768, 1), Some(8));
        // Already past the target on its own: no split, no second pass.
        assert_eq!(gemv_split(1, 768, 8192, 1), Some(1));
        // Refused: not a vector, batched, or too small to reduce.
        assert_eq!(gemv_split(128, 768, 768, 1), None);
        assert_eq!(gemv_split(1, 768, 768, 12), None);
        assert_eq!(gemv_split(1, 64, 768, 1), None);
        assert_eq!(gemv_split(1, 768, 1, 1), None);
        // K bounds the split: 256 rows of K feed 4 workgroups, not 8.
        assert_eq!(gemv_split(1, 256, 768, 1), Some(4));
    }

    #[test]
    fn blocked_is_chosen_exactly_when_the_tile_is_worth_its_padding() {
        // sam3 ViT-H attention: fills the tile, including the n = 64 column.
        assert!(prefer_blocked(576, 576));
        assert!(prefer_blocked(576, 64));
        assert!(prefer_blocked(5184, 64));
        // roberta at sequence length 1: a 64-row tile for one row of output.
        assert!(!prefer_blocked(1, 768));
        assert!(!prefer_blocked(1, 1));
        // A single tile is exactly break-even at 4×, so it stays narrow.
        assert!(!prefer_blocked(16, 16));
        assert!(prefer_blocked(64, 64));
    }
}
