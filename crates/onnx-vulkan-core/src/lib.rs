//! Types and algorithms independent of the ONNX frontend and host runtime.
//!
//! This crate is the reusable core of `onnx-vulkan-rs`: independent of
//! ONNX Runtime and any specific FFI API.

pub mod cache;
pub mod device;
mod error;
pub mod execution;
pub mod executor;
pub mod fusion;
pub mod graph;
pub mod host_ops;
pub mod interp;
pub mod rewrite;
pub mod shaders;
pub mod shape;

pub use cache::KernelCache;
pub use device::{DeviceBuffer, DeviceTensor, Tensor};
pub use error::{Error, Result};
pub use execution::{ExecutionEnv, device_storage_bytes};
pub use executor::{Executor, Outputs};
pub use fusion::convex_groups;
pub use graph::{
    AttrValue, ElementType, GraphIr, InitializerIr, NodeIr, constant_outputs, elem_size,
    fold_constant_params, storage_len,
};
pub use host_ops::HostTensor;
pub use interp::{execute, is_implemented, is_implemented_node};
pub use rewrite::{fold_constants, fuse_layernorm, prune_dead_initializers, prune_dead_nodes};
pub use shape::{Broadcast, broadcast, element_count};
