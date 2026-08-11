//! Pure Vulkan compute runtime (no dependency on ONNX Runtime).
//!
//! Components: [`VkContext`] (instance/device/queue via `ash`), buffers with
//! `gpu-allocator`, WGSL→SPIR-V compilation via `naga`, compute pipelines and
//! synchronous dispatch with staging upload/readback.

mod buffer;
mod capture;
mod context;
mod descriptor;
mod pipeline;
mod shader;
pub mod stats;
mod stream;

pub use buffer::GpuBuffer;
pub use capture::{CopyOp, DispatchOp, StreamOp, UploadOp};
pub use context::{ComputeLimits, CoopMatU8, DeviceFingerprint, VkContext};
pub use pipeline::{BufferSlice, ComputePipeline};
pub use shader::compile_wgsl;
