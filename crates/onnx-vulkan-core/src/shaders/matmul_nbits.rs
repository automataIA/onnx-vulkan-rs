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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_compiles() {
        vk_compute::compile_wgsl(MATMUL_NBITS).expect("shader MatMulNBits valid");
    }
}
