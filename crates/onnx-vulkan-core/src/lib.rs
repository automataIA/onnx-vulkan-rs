//! Types and algorithms independent of the ONNX frontend and host runtime.
//!
//! This crate is the reusable core of `onnx-vulkan-rs`: independent of
//! ONNX Runtime and any specific FFI API.

pub mod cache;
pub mod comparison;
pub mod device;
mod error;
pub mod execution;
pub mod executor;
pub mod fusion;
pub mod graph;
pub mod host_ops;
pub mod interp;
pub mod plan;
pub mod rewrite;
pub mod shaders;
pub mod shape;
pub mod tuning;
pub mod work;

pub use cache::{KernelCache, PackedWeightInfo, PipelineKey};
pub use device::{DeviceBuffer, DeviceTensor, Tensor};
pub use error::{Error, Result};
pub use execution::{ExecutionEnv, device_storage_bytes};
pub use executor::{Executor, Outputs, prepare_graph};
pub use fusion::convex_groups;
pub use graph::{
    AttrValue, ElementType, GraphIr, InitializerIr, NodeIr, constant_outputs, elem_size,
    fold_constant_params, graph_digest, storage_len,
};
pub use host_ops::HostTensor;
pub use interp::{
    MAX_OPSET, argmax_of, execute, is_implemented, is_implemented_node, unsupported_dtype,
    unsupported_quantization,
};
pub use interp::{execute_host_nodes, execute_traced};
pub use plan::{StepPlan, StepPlanStats, StepTrace, host_bytes};
pub use rewrite::{fold_constants, fuse_layernorm, prune_dead_initializers, prune_dead_nodes};
pub use shape::{Broadcast, broadcast, element_count};
