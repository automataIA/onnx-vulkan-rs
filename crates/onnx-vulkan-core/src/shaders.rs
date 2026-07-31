//! WGSL sources shared by the standalone executor and the backend adapters.

pub mod attention;
pub mod conv;
pub mod conv_integer;
pub mod conv_transpose;
pub mod dynamic_quantize;
pub mod elementwise;
pub mod gather_block_quantized;
pub mod gemm;
pub mod grid_sample;
pub mod matmul_fp32;
pub mod matmul_integer;
pub mod matmul_nbits;
pub mod movement;
pub mod normalization;
pub mod pooling;
pub mod qlinear;
pub mod quantize_linear;
pub mod reduction;
pub mod resize;

/// Serializes `MAX_RANK` `u32` strides (two `vec4<u32>`) into the push
/// constants, zero-padding beyond the effective rank. Shared layout for
/// elementwise, movement and matmul.
pub fn push_vec4s(push: &mut Vec<u8>, values: &[u32]) {
    for d in 0..elementwise::MAX_RANK {
        push.extend_from_slice(&values.get(d).copied().unwrap_or(0).to_le_bytes());
    }
}
