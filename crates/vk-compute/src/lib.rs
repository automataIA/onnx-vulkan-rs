//! Pure Vulkan compute runtime (no dependency on ONNX Runtime).
//!
//! Components: [`VkContext`] (instance/device/queue via `ash`), buffers with
//! `gpu-allocator`, WGSL→SPIR-V compilation via `naga`, compute pipelines and
//! synchronous dispatch with staging upload/readback.

mod buffer;
mod context;
mod descriptor;
mod pipeline;
mod shader;
pub mod stats;
mod stream;

pub use buffer::GpuBuffer;
pub use context::{CoopMatU8, VkContext};
pub use pipeline::{BufferSlice, ComputePipeline};
pub use shader::compile_wgsl;
