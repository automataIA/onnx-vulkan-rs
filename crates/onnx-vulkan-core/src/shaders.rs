//! Sorgenti WGSL condivise da executor standalone e adattatori backend.

pub mod conv;
pub mod conv_integer;
pub mod conv_transpose;
pub mod dynamic_quantize;
pub mod elementwise;
pub mod gemm;
pub mod grid_sample;
pub mod matmul_fp32;
pub mod matmul_integer;
pub mod movement;
pub mod normalization;
pub mod pooling;
pub mod quantize_linear;
pub mod reduction;
pub mod resize;

/// Serializza `MAX_RANK` stride `u32` (due `vec4<u32>`) nei push constant,
/// zero-padding beyond effective rank. Shared layout for elementwise,
/// movement e matmul.
pub fn push_vec4s(push: &mut Vec<u8>, values: &[u32]) {
    for d in 0..elementwise::MAX_RANK {
        push.extend_from_slice(&values.get(d).copied().unwrap_or(0).to_le_bytes());
    }
}
