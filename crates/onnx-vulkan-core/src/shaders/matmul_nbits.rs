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

/// Columns wide enough for the decode kernel to be worth its extra pipeline.
///
/// Below this the op runs at the dispatch floor — 0.010–0.014 ms for the whole
/// node, 60 GB/s of a 0.6 MB weight — and no variant measured on the 4070 moves
/// it by more than the noise, while the wide-column kernel measured **0.91×** at
/// `K = 6912, N = 1152`. Wide columns are the opposite: `K = 1152, N = 6912`
/// gains 1.37× and `K = 2048, N = 11008` 1.28×.
///
/// Calibrated on an RTX 4070 with `example matmulnbits`; on another device it is
/// an unmeasured constant.
pub const DECODE_MIN_N: usize = 4096;

/// Columns above which the unembedding form wins instead (`N = 151936` and
/// `262144` in the matrix): 1.46× weighted, 1.60× on gemma3's, at 291 GB/s of
/// the card's 504.
pub const HEAD_MIN_N: usize = 65536;

/// Threads sharing one output column in [`decode_source`]. 16 of 256, so a
/// workgroup covers 16 columns.
pub const DECODE_LANES: u32 = 16;

/// Threads sharing one output column in [`head_source`].
pub const HEAD_LANES: u32 = 8;

/// The decode kernel: 256 threads, `256 / lanes` columns per workgroup, `lanes`
/// threads walking one column's words.
///
/// Same arithmetic as [`MATMUL_NBITS`] and same push constants; what changes is
/// the shape. The shipped kernel gives a whole 64-thread workgroup to one output
/// element, so at `K = 1152` each thread loads 2.25 words and then pays a 6-deep
/// tree reduction — the reduction, not the loading, is what it spends its time
/// on. Sharing a workgroup between 16 columns cuts the reduction to 4 levels and
/// quadruples the work each thread does before it.
///
/// `lane` varies fastest inside a column on purpose: `B` is
/// `[N, n_blocks, blob]`, i.e. K-contiguous *within* a column, so the coalesced
/// mapping is consecutive threads on consecutive words of one column. This is
/// transposed with respect to `shaders::matmul_fp32`'s `gemv_split`, where `B`
/// was `[K, N]` — and the transposition is measured, not assumed: the other
/// mapping reads 576 bytes apart per lane.
pub fn decode_source(lanes: u32) -> String {
    let cols = 256 / lanes;
    format!(
        r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> quant: array<u32>;
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
        let words = pc.n_blocks * pc.blob_words;
        let q_base = col * words;
        let a_base = row * pc.k;
        for (var w = lane; w < words; w = w + LANES) {{
            let block = w / pc.blob_words;
            let k0 = block * pc.block_size + (w % pc.blob_words) * 8u;
            let word = quant[q_base + w];
            let zoff = col * pc.zp_row_bytes + block / 2u;
            let z = f32((zero_points[zoff / 4u] >> (8u * (zoff % 4u) + 4u * (block % 2u))) & 15u);
            var part = 0.0;
            for (var i = 0u; i < 8u; i = i + 1u) {{
                let k = k0 + i;
                if (k >= pc.k) {{ break; }}
                part = fma(f32((word >> (4u * i)) & 15u) - z, a[a_base + k], part);
            }}
            acc = fma(part, scales[col * pc.n_blocks + block], acc);
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

/// The unembedding kernel: one **block** per lane iteration, read as a
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
/// Only claimed above [`HEAD_MIN_N`], where the column count keeps the grid full
/// at 8 lanes: `N / 32` is 8192 workgroups on gemma3's head and 36 on a 1152-wide
/// projection, which is why the same kernel loses badly there.
pub fn head_source(lanes: u32) -> String {
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

    #[test]
    fn source_compiles() {
        vk_compute::compile_wgsl(MATMUL_NBITS).expect("shader MatMulNBits valid");
        vk_compute::compile_wgsl(&decode_source(DECODE_LANES)).expect("decode variant valid");
        vk_compute::compile_wgsl(&head_source(HEAD_LANES)).expect("head variant valid");
    }
}
