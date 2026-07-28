//! Shared shaders and dispatch layouts for `Gemm`.
//!
//! ## Why a 64×64 tile with a 4×4 micro-tile per thread
//!
//! The obvious tiled GEMM — a 16×16 workgroup where each thread owns **one**
//! output element — is bound by shared memory, not by arithmetic: each FMA
//! needs two shared reads, so the ALUs idle waiting on LDS. Measured on sam3
//! ViT-H (128 dispatches, 4.61 TFLOP): **1.26 TFLOP/s on a card that does ~29**,
//! 4.3% of peak.
//!
//! Here each thread owns a 4×4 block of the output instead. One pass of the
//! inner loop reads 4 scalars of `A` and one `vec4` of `B` — 5 shared reads —
//! and issues 16 FMAs. The ratio goes from 0.5 to 3.2 FMA per read, and the
//! accumulators stay in registers because nothing indexes them dynamically.
//!
//! The tile is 64×64 because that is 16×16 threads times the 4×4 micro-tile,
//! and because every dimension a transformer produces is a multiple of it
//! (ViT-H: M = 5184, K and N ∈ {1024, 3072, 4736}) — the fast path never pays
//! for the bounds checks it still has to carry for the general case.
//!
//! `B` is staged as `vec4` so a whole micro-tile row arrives in one read; `A`
//! stays scalar because its four values belong to four *different* rows of the
//! tile and are not contiguous.

/// Output block per workgroup, along both M and N.
pub const TILE_SIZE: u32 = 64;
/// K slice staged in shared memory per iteration.
pub const K_STEP: u32 = 16;
/// Output elements per thread, along both M and N.
pub const MICRO: u32 = 4;
pub const BINDINGS: u32 = 4;
/// 8 four-byte fields in push constant struct.
pub const PUSH_BYTES: u32 = 32;

/// `Y = alpha · A' · B' + beta · C`, with optionally transposed `A'`/`B'` and `C`
/// broadcastable over rows and/or columns (ONNX `Gemm`, always 2D).
///
/// Transposition is resolved in the indexing of the staging loads, so no
/// transposed matrix is ever materialized.
pub const GEMM: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Push {
    m: u32, k: u32, n: u32, flags: u32,   // bit0 transA, bit1 transB, bit2 has_c
    alpha: f32, beta: f32,
    c_rows: u32, c_cols: u32,             // C dimensions before broadcast
}
var<immediate> pc: Push;

const TILE = 64u;   // output tile per workgroup
const KSTEP = 16u;  // slice of K kept in shared per iteration

// A: [TILE][KSTEP] scalar — the 4 rows of a micro-tile are not contiguous
var<workgroup> as_tile: array<f32, 1024>;
// B: [KSTEP][TILE/4] vec4 — one micro-tile row in a single read
var<workgroup> bs_tile: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let trans_a = (pc.flags & 1u) != 0u;
    let trans_b = (pc.flags & 2u) != 0u;
    let row0 = wid.y * TILE;          // first row of the tile
    let col0 = wid.x * TILE;          // first column of the tile
    let tid = lid.y * 16u + lid.x;    // 0..255

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let ntiles = (pc.k + KSTEP - 1u) / KSTEP;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        let k0 = t * KSTEP;

        // --- stage A: 1024 values, 4 per thread (r = row in the tile, kk in K)
        for (var s = 0u; s < 4u; s = s + 1u) {
            let l = tid + s * 256u;
            let r = l / KSTEP;
            let kk = l % KSTEP;
            let gr = row0 + r;
            let gk = k0 + kk;
            var v = 0.0;
            if (gr < pc.m && gk < pc.k) {
                // A is [M,K] or [K,M] if transposed
                v = a[select(gr * pc.k + gk, gk * pc.m + gr, trans_a)];
            }
            as_tile[l] = v;
        }

        // --- stage B: 256 vec4, one per thread (kk in K, 4 contiguous columns)
        let bk = tid / 16u;
        let bc = (tid % 16u) * 4u;
        let gk = k0 + bk;
        var bv = vec4<f32>(0.0);
        if (gk < pc.k) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let gc = col0 + bc + j;
                if (gc < pc.n) {
                    // B is [K,N] or [N,K] if transposed
                    bv[j] = b[select(gk * pc.n + gc, gc * pc.k + gk, trans_b)];
                }
            }
        }
        bs_tile[tid] = bv;
        workgroupBarrier();

        // --- 4 scalars of A + 1 vec4 of B for 16 FMAs
        let arow = lid.y * 4u;
        for (var kk = 0u; kk < KSTEP; kk = kk + 1u) {
            let bvec = bs_tile[kk * 16u + lid.x];
            acc0 = fma(vec4<f32>(as_tile[(arow + 0u) * KSTEP + kk]), bvec, acc0);
            acc1 = fma(vec4<f32>(as_tile[(arow + 1u) * KSTEP + kk]), bvec, acc1);
            acc2 = fma(vec4<f32>(as_tile[(arow + 2u) * KSTEP + kk]), bvec, acc2);
            acc3 = fma(vec4<f32>(as_tile[(arow + 3u) * KSTEP + kk]), bvec, acc3);
        }
        workgroupBarrier();
    }

    // --- 4×4 micro-tile write, with alpha/beta and C broadcast
    let has_c = (pc.flags & 4u) != 0u;
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
            var value = pc.alpha * accv[j];
            if (has_c) {
                let cr = select(0u, row, pc.c_rows > 1u);
                let cc = select(0u, col, pc.c_cols > 1u);
                value = value + pc.beta * c[cr * pc.c_cols + cc];
            }
            out[row * pc.n + col] = value;
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn source_compiles() {
        vk_compute::compile_wgsl(super::GEMM).expect("valid Gemm shader");
    }

    /// The staging loops must cover both tiles exactly, with no element written
    /// twice and none left stale from the previous K slice: a gap would read
    /// whatever the last iteration put there and the error would depend on K.
    #[test]
    fn staging_covers_both_tiles_exactly() {
        let (tile, kstep) = (super::TILE_SIZE, super::K_STEP);
        let threads = 256;

        let mut a_hits = vec![0u32; (tile * kstep) as usize];
        let mut b_hits = vec![0u32; (kstep * tile / 4) as usize];
        for tid in 0..threads {
            for s in 0..4 {
                a_hits[(tid + s * threads) as usize] += 1;
            }
            b_hits[tid as usize] += 1;
        }
        assert!(a_hits.iter().all(|&h| h == 1), "tile A: {a_hits:?}");
        assert!(b_hits.iter().all(|&h| h == 1), "tile B: {b_hits:?}");

        // and the 16×16 threads must own the 64×64 output block exactly
        assert_eq!(threads * super::MICRO * super::MICRO, tile * tile);
    }
}
