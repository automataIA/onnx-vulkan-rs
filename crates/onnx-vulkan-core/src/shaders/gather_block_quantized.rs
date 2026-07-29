//! `GatherBlockQuantized` (com.microsoft): gather rows of a block-quantized
//! table and dequantize them on the way out.
//!
//! This is `Gather` fused with the dequantization of `MatMulNBits`, and it
//! uses the same 4-bit packing: element `2i` lives in the **low** nibble of
//! byte `i`, and so does block `2i` of the zero points. Only the axis roles
//! differ — the gathered axis is 0 and the quantized axis is the last one, so
//! a row of the table is one contiguous run of blocks:
//!
//! | tensor | shape |
//! |---|---|
//! | `data` | `[rows, cols/2]` u8, i.e. `cols` 4-bit elements per row |
//! | `indices` | any shape, folded to `idx_count` |
//! | `scales` | `[rows, n_blocks]` f32 |
//! | `zero_points` | `[rows, ceil(n_blocks/2)]` u8 |
//! | output | `indices.shape ++ [cols]` f32 |
//!
//! In gemma3-1b this is the embedding table: 262144 × 1152 int4, gathered one
//! token at a time. One thread per output element is enough — a decode step
//! reads 1152 of the 302 million elements, and the dispatch is pure latency.

pub const BINDINGS: u32 = 5;
pub const PUSH_BYTES: u32 = 32;

/// `out[j, c] = (unpack(data[idx[j], c]) - zp) · scale`.
pub const GATHER_BLOCK_QUANTIZED: &str = r#"
@group(0) @binding(0) var<storage, read> quant: array<u32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> zero_points: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

struct Push {
    count: u32, cols: u32, row_bytes: u32, n_blocks: u32,
    block_size: u32, zp_row_bytes: u32, gx: u32, pad: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = (wid.y * pc.gx + wid.x) * 256u + lid.x;
    if (i >= pc.count) { return; }
    let c = i % pc.cols;
    let row = u32(indices[i / pc.cols]);
    let block = c / pc.block_size;

    let qoff = row * pc.row_bytes + c / 2u;
    let q = f32((quant[qoff / 4u] >> (8u * (qoff % 4u) + 4u * (c % 2u))) & 15u);
    let zoff = row * pc.zp_row_bytes + block / 2u;
    let z = f32((zero_points[zoff / 4u] >> (8u * (zoff % 4u) + 4u * (block % 2u))) & 15u);

    out[i] = (q - z) * scales[row * pc.n_blocks + block];
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn source_compiles() {
        vk_compute::compile_wgsl(super::GATHER_BLOCK_QUANTIZED).expect("valid shader");
    }
}
