//! `MatMulNBits` (com.microsoft): matmul against a block-quantized weight.
//!
//! `A [.., M, K] × dequant(B) [K, N] → [.., M, N]`, where `B` is stored
//! transposed and quantized in blocks of `block_size` along `K`:
//!
//! | tensor | shape | meaning |
//! |---|---|---|
//! | `B` | `[N, n_blocks, block_size·bits/8]` u8 | two 4-bit weights per byte |
//! | `scales` | `[N, n_blocks]` f32 | one scale per block |
//! | `zero_points` | `[N, ceil(n_blocks/2)]` u8 | one 4-bit zero point per block |
//!
//! with `n_blocks = ceil(K / block_size)` and
//! `B[n][k] = (nibble(n, k) - zp(n, k / block_size)) · scale(n, k / block_size)`.
//!
//! Both packed tensors put element `2i` in the **low** nibble of byte `i`. For
//! `B` that makes a `u32` word hold eight consecutive `k`, weight `i` of the
//! word at bit `4i` — no byte extraction, one shift. The zero points are not
//! so kind: their row stride is `ceil(n_blocks/2)` bytes, 18 for gemma3, so a
//! row is not `u32`-aligned and the index has to be computed in bytes.
//!
//! This is the correctness kernel: one workgroup per output element, threads
//! strided along `K`, tree-reduced. It reads the weight column once per output
//! **row**, so a prefill of `M` tokens moves `M` times the weight — fine at
//! `M = 1`, which is the decode step and the whole point of the op, and the
//! reason a tier-1 example comes before any of this is tuned.

/// `a`, `quant`, `scales`, `zero_points`, `out`.
pub const BINDINGS: u32 = 5;
pub const PUSH_BYTES: u32 = 32;

/// Threads per workgroup, i.e. ways the `K` loop is split.
///
/// One workgroup covers one output element, so this is also the reduction
/// width. 64 rather than 256 because `K/8` is the number of words to go
/// round: at gemma3's `K = 1152` that is 144, so 256 threads would leave
/// nearly half of them with nothing to do on the second pass.
pub const WG: u32 = 64;

/// Load/dequantization form explored by the `MatMulNBits` tuner.
///
/// The numeric codes preserve the existing example and Python protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CandidateLoad {
    /// One `vec4<u32>` block with a single chained accumulator.
    Block = 0,
    /// One `u32` word per loop iteration.
    Word = 1,
    /// One `vec4<u32>` block with four independent accumulators.
    BlockFourAccumulators = 2,
    /// One `vec4<u32>` load with per-word address calculations.
    Quad = 4,
}

impl CandidateLoad {
    pub const fn code(self) -> u32 {
        self as u32
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::Word => "vec1",
            Self::BlockFourAccumulators => "block4",
            Self::Quad => "vec4",
        }
    }

    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::Block),
            1 => Some(Self::Word),
            2 => Some(Self::BlockFourAccumulators),
            4 => Some(Self::Quad),
            _ => None,
        }
    }

    const fn units_per_column(self, words: usize) -> usize {
        match self {
            Self::Word => words,
            Self::Block | Self::BlockFourAccumulators | Self::Quad => words / 4,
        }
    }
}

/// Reduction implementation explored by the tuner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CandidateReduction {
    WorkgroupTree,
    SubgroupShuffle,
}

impl CandidateReduction {
    pub const fn code(self) -> u32 {
        match self {
            Self::WorkgroupTree => 0,
            Self::SubgroupShuffle => 1,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::WorkgroupTree => "tree",
            Self::SubgroupShuffle => "sub",
        }
    }

    pub const fn uses_subgroup(self) -> bool {
        matches!(self, Self::SubgroupShuffle)
    }
}

/// Fully specified candidate metadata shared by enumeration and orchestration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CandidateConfig {
    pub lanes: u32,
    pub load: CandidateLoad,
    pub workgroup_size: u32,
    pub reduction: CandidateReduction,
}

impl CandidateConfig {
    pub const fn new(
        lanes: u32,
        load: CandidateLoad,
        workgroup_size: u32,
        reduction: CandidateReduction,
    ) -> Self {
        Self {
            lanes,
            load,
            workgroup_size,
            reduction,
        }
    }

    /// Static geometry/capability filter used before shader compilation.
    pub fn is_viable(self, words_per_column: usize, subgroup_size: u32) -> bool {
        if self.lanes == 0 || self.workgroup_size == 0 {
            return false;
        }
        let shape_ok = self.lanes as usize <= self.load.units_per_column(words_per_column)
            && self.lanes <= self.workgroup_size
            && self.workgroup_size.is_multiple_of(self.lanes);
        let subgroup_ok = !self.reduction.uses_subgroup()
            || (subgroup_size > 0
                && self.lanes > 1
                && self.lanes <= subgroup_size
                && self.workgroup_size.is_multiple_of(subgroup_size));
        shape_ok && subgroup_ok
    }
}

pub const CANDIDATE_LANES: &[u32] = &[1, 2, 4, 8, 16, 32, 64, 128, 256];
pub const CANDIDATE_LOADS: &[CandidateLoad] = &[
    CandidateLoad::Block,
    CandidateLoad::Word,
    CandidateLoad::BlockFourAccumulators,
    CandidateLoad::Quad,
];
pub const CANDIDATE_WORKGROUP_SIZES: &[u32] = &[64, 128, 256, 512, 1024];
pub const CANDIDATE_REDUCTIONS: &[CandidateReduction] = &[
    CandidateReduction::WorkgroupTree,
    CandidateReduction::SubgroupShuffle,
];

/// The original 18 configurations printed by the example in its default mode.
pub const DEFAULT_CANDIDATES: &[CandidateConfig] = &[
    CandidateConfig::new(
        256,
        CandidateLoad::Word,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        128,
        CandidateLoad::Word,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        64,
        CandidateLoad::Word,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        32,
        CandidateLoad::Word,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        16,
        CandidateLoad::Word,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        8,
        CandidateLoad::Word,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        64,
        CandidateLoad::Quad,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        32,
        CandidateLoad::Quad,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        16,
        CandidateLoad::Quad,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        8,
        CandidateLoad::Quad,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        64,
        CandidateLoad::Block,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        32,
        CandidateLoad::Block,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        16,
        CandidateLoad::Block,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        8,
        CandidateLoad::Block,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        64,
        CandidateLoad::BlockFourAccumulators,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        32,
        CandidateLoad::BlockFourAccumulators,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        16,
        CandidateLoad::BlockFourAccumulators,
        256,
        CandidateReduction::WorkgroupTree,
    ),
    CandidateConfig::new(
        8,
        CandidateLoad::BlockFourAccumulators,
        256,
        CandidateReduction::WorkgroupTree,
    ),
];

/// Complete Cartesian search space, excluding configurations where lanes
/// already exceed the workgroup. Geometry-specific filtering remains in
/// [`CandidateConfig::is_viable`].
pub fn candidate_space() -> Vec<CandidateConfig> {
    let mut candidates = Vec::new();
    for &workgroup_size in CANDIDATE_WORKGROUP_SIZES {
        for &lanes in CANDIDATE_LANES {
            for &load in CANDIDATE_LOADS {
                for &reduction in CANDIDATE_REDUCTIONS {
                    if lanes <= workgroup_size {
                        candidates.push(CandidateConfig::new(
                            lanes,
                            load,
                            workgroup_size,
                            reduction,
                        ));
                    }
                }
            }
        }
    }
    candidates
}

/// `out[m][n] = Σ_k a[m][k] · (nibble(n,k) − zp) · scale`.
///
/// The `K` loop walks `u32` words of the weight column: thread `t` takes word
/// `t`, `t + 64`, …, so consecutive threads read consecutive words and the
/// weight — the only tensor big enough to matter, 151 MB for gemma3's lm_head
/// — is read fully coalesced. `a` is re-read by every column and left to the
/// cache: it is `K` floats, 4.6 KB.
///
/// The scale multiply is hoisted out of the eight weights of a word and, when
/// a word sits inside one block, out of the block: `Σ (w−z)·a` scaled once,
/// not eight times. `block_size` is a multiple of 8 in every export, so a word
/// never straddles two blocks.
pub const MATMUL_NBITS: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> quant: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> zero_points: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

struct Push {
    k: u32, n: u32, n_blocks: u32, blob_words: u32,
    block_size: u32, zp_row_bytes: u32, gx: u32, pad: u32,
}
var<immediate> pc: Push;

const WG = 64u;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    // the column index is folded over x and y: N reaches 262144 on the
    // unembedding, past any guaranteed limit on a single grid dimension
    let col = wid.y * pc.gx + wid.x;
    let row = wid.z;
    var acc = 0.0;
    // uniform across the workgroup, so the barriers below are still reached
    // by every thread of every live workgroup
    if (col < pc.n) {
        let words = pc.n_blocks * pc.blob_words;
        let q_base = col * words;
        let a_base = row * pc.k;
        for (var w = tid; w < words; w = w + WG) {
            let block = w / pc.blob_words;
            let k0 = block * pc.block_size + (w % pc.blob_words) * 8u;
            let word = quant[q_base + w];
            // zero point: nibble `block` of a byte-packed row of unaligned stride
            let zoff = col * pc.zp_row_bytes + block / 2u;
            let z = f32((zero_points[zoff / 4u] >> (8u * (zoff % 4u) + 4u * (block % 2u))) & 15u);
            var part = 0.0;
            for (var i = 0u; i < 8u; i = i + 1u) {
                let k = k0 + i;
                if (k >= pc.k) { break; }
                part = fma(f32((word >> (4u * i)) & 15u) - z, a[a_base + k], part);
            }
            acc = fma(part, scales[col * pc.n_blocks + block], acc);
        }
    }
    red[tid] = acc;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s / 2u) {
        if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u && col < pc.n) {
        out[row * pc.n + col] = red[0];
    }
}
"#;

/// Columns wide enough for [`wide_source`] to be worth its extra pipeline.
///
/// Below this the op runs at the dispatch floor — 0.010–0.014 ms for the whole
/// node, 60 GB/s of a 0.6 MB weight — and no variant measured on the 4070 moves
/// it by more than the noise, while the wide-column kernel measured **0.91×** at
/// `K = 6912, N = 1152`. The sweep over all 11 census geometries agrees and
/// sharpens it: at `N ≤ 2048` the best configuration is 1.07×–1.17× over the
/// shipped kernel and it is a *different* one on every geometry, which is a
/// threshold, not a kernel. Above it the same configuration wins everywhere and
/// wins by 1.32×–1.43×.
///
/// Calibrated on an RTX 4070 with `example matmulnbits`; on another device it is
/// an unmeasured constant.
pub const WIDE_MIN_N: usize = 4096;

/// Threads sharing one output column in [`wide_source`]. 4 of 256, so a
/// workgroup covers 64 columns.
///
/// **4, not 8 and not 16, and the difference is not small.** This kernel used to
/// be two — a word-loading form at 16 lanes for `4096 ≤ N < 65536` and this one
/// at 8 lanes above it — and both were picked from a variant list whose lane
/// column started at 8. Sweeping the same list down to 1 (`example matmulnbits
/// --sweep`, all 11 census geometries) makes this form at 4 lanes the best
/// configuration at **every** `N ≥ 4096`, which is why the word form is gone:
///
/// | geometry | was | at 4 lanes | |
/// |---|---|---|---|
/// | gemma3 `ffn-in` `N=6912` | 184 GB/s | 243 | 1.32× |
/// | qwen `ffn-in` `N=11008` | 270 | 378 | 1.42× |
/// | gemma3 `lm_head` `N=262144` | 291 | 401 | 1.43× |
/// | qwen `lm_head` `N=151936` | 275–292 | 281–289 | flat |
///
/// It is the mechanism this file already documents — a lane needs *enough blocks
/// to iterate over* — read one step further down than anyone had measured. The
/// ceiling that `cronologia.md` 2026-07-30 called exhausted at 291 GB/s (58% of
/// the card's 504) is **413 GB/s (82%)** at 4 lanes.
///
/// The workgroup size stays 256 deliberately: it is worth 3–17% more on three
/// geometries and its optimum walks 64 → 512 → 1024 with no predicate that fits
/// (`docs/autotuning.md` §2). One calibrated constant, not two.
pub const WIDE_LANES: u32 = 4;

/// Production kernel selected for a concrete `MatMulNBits` invocation.
///
/// The wide-column tactic is calibrated for one-row decoder work only. Keeping
/// this decision beside the shader constants lets the interpreter and the
/// autotuning audit exercise exactly the same routing predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Row,
    Wide { lanes: u32 },
}

/// Select the currently shipped route without compiling or initializing Vulkan.
pub const fn route(rows: usize, n: usize) -> Route {
    if rows == 1 && n >= WIDE_MIN_N {
        Route::Wide { lanes: WIDE_LANES }
    } else {
        Route::Row
    }
}

/// The wide-column kernel: one **block** per lane iteration, read as a
/// `vec4<u32>`, with four independent accumulators.
///
/// A block is `blob_words = 4` words = 16 bytes = 32 weights, and it is the unit
/// the format is built around: one scale and one zero point cover it, so both
/// are loaded once instead of four times, and the 32 nibbles need no bound check
/// because `K` is a multiple of `block_size`.
///
/// **The four accumulators are the point, not the wide load.** The same kernel
/// with a single `part` chained through all 32 fma measured 1.41× where this one
/// measures 1.46×, and a 16-byte load with the per-word address arithmetic left
/// in measured 1.38× — the three are within a few percent of each other and all
/// collapse to 0.37× at `lanes = 32`. What decides this kernel is that a lane
/// gets **enough blocks to iterate over**; at 8 lanes over `n_blocks = 36..64`
/// that is 4–8 iterations, and at 32 lanes it is one, with nothing to hide the
/// dependency chain behind.
///
/// Only claimed above [`WIDE_MIN_N`], where the column count keeps the grid
/// full: at 4 lanes `N / 64` is 4096 workgroups on gemma3's head and 18 on a
/// 1152-wide projection, which is why the same kernel loses badly there.
pub fn wide_source(lanes: u32) -> String {
    let cols = 256 / lanes;
    format!(
        r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> quant: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> zero_points: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

struct Push {{
    k: u32, n: u32, n_blocks: u32, blob_words: u32,
    block_size: u32, zp_row_bytes: u32, gx: u32, pad: u32,
}}
var<immediate> pc: Push;

const LANES = {lanes}u;
const COLS = {cols}u;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {{
    let col = (wid.y * pc.gx + wid.x) * COLS + tid / LANES;
    let lane = tid % LANES;
    let row = wid.z;
    var acc = 0.0;
    if (col < pc.n) {{
        let q_base = col * pc.n_blocks;
        let a_base = row * pc.k;
        for (var block = lane; block < pc.n_blocks; block = block + LANES) {{
            let quad = quant[q_base + block];
            let zoff = col * pc.zp_row_bytes + block / 2u;
            let z = f32((zero_points[zoff / 4u] >> (8u * (zoff % 4u) + 4u * (block % 2u))) & 15u);
            let k0 = a_base + block * pc.block_size;
            var p0 = 0.0; var p1 = 0.0; var p2 = 0.0; var p3 = 0.0;
            for (var i = 0u; i < 8u; i = i + 1u) {{
                let sh = 4u * i;
                p0 = fma(f32((quad.x >> sh) & 15u) - z, a[k0 + i], p0);
                p1 = fma(f32((quad.y >> sh) & 15u) - z, a[k0 + 8u + i], p1);
                p2 = fma(f32((quad.z >> sh) & 15u) - z, a[k0 + 16u + i], p2);
                p3 = fma(f32((quad.w >> sh) & 15u) - z, a[k0 + 24u + i], p3);
            }}
            acc = fma((p0 + p1) + (p2 + p3), scales[q_base + block], acc);
        }}
    }}
    red[tid] = acc;
    workgroupBarrier();
    for (var s = LANES / 2u; s > 0u; s = s / 2u) {{
        if (lane < s) {{ red[tid] = red[tid] + red[tid + s]; }}
        workgroupBarrier();
    }}
    if (lane == 0u && col < pc.n) {{ out[row * pc.n + col] = red[tid]; }}
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn source_compiles() {
        vk_compute::compile_wgsl(MATMUL_NBITS).expect("shader MatMulNBits valid");
        vk_compute::compile_wgsl(&wide_source(WIDE_LANES)).expect("wide variant valid");
    }

    #[test]
    fn production_route_is_decode_only_and_inclusive_at_threshold() {
        assert_eq!(route(1, WIDE_MIN_N - 1), Route::Row);
        assert_eq!(route(1, WIDE_MIN_N), Route::Wide { lanes: WIDE_LANES });
        assert_eq!(route(2, WIDE_MIN_N), Route::Row);
        assert_eq!(route(0, WIDE_MIN_N), Route::Row);
    }

    #[test]
    fn candidate_space_is_complete_and_unique() {
        let candidates = candidate_space();
        let unique = candidates.iter().copied().collect::<HashSet<_>>();

        assert_eq!(candidates.len(), 336);
        assert_eq!(unique.len(), candidates.len());
        assert_eq!(
            candidates.first(),
            Some(&CandidateConfig::new(
                1,
                CandidateLoad::Block,
                64,
                CandidateReduction::WorkgroupTree,
            ))
        );
        assert_eq!(
            candidates.last(),
            Some(&CandidateConfig::new(
                256,
                CandidateLoad::Quad,
                1024,
                CandidateReduction::SubgroupShuffle,
            ))
        );
    }

    #[test]
    fn default_candidates_preserve_the_legacy_order() {
        let legacy = DEFAULT_CANDIDATES
            .iter()
            .map(|candidate| (candidate.lanes, candidate.load.code()))
            .collect::<Vec<_>>();

        assert_eq!(
            legacy,
            [
                (256, 1),
                (128, 1),
                (64, 1),
                (32, 1),
                (16, 1),
                (8, 1),
                (64, 4),
                (32, 4),
                (16, 4),
                (8, 4),
                (64, 0),
                (32, 0),
                (16, 0),
                (8, 0),
                (64, 2),
                (32, 2),
                (16, 2),
                (8, 2),
            ]
        );
        assert!(DEFAULT_CANDIDATES.iter().all(|candidate| {
            candidate.workgroup_size == 256
                && candidate.reduction == CandidateReduction::WorkgroupTree
        }));
    }

    #[test]
    fn viability_covers_load_and_subgroup_constraints() {
        let config = |lanes, load, reduction| CandidateConfig::new(lanes, load, 256, reduction);

        assert!(
            config(64, CandidateLoad::Word, CandidateReduction::WorkgroupTree).is_viable(144, 32)
        );
        assert!(
            !config(64, CandidateLoad::Block, CandidateReduction::WorkgroupTree).is_viable(144, 32)
        );
        assert!(
            config(
                32,
                CandidateLoad::Block,
                CandidateReduction::SubgroupShuffle
            )
            .is_viable(144, 32)
        );
        assert!(
            !config(64, CandidateLoad::Word, CandidateReduction::SubgroupShuffle)
                .is_viable(144, 32)
        );
        assert!(
            !config(1, CandidateLoad::Word, CandidateReduction::SubgroupShuffle).is_viable(144, 32)
        );
        assert!(
            !CandidateConfig::new(
                0,
                CandidateLoad::Word,
                256,
                CandidateReduction::WorkgroupTree,
            )
            .is_viable(144, 32)
        );
    }
}
