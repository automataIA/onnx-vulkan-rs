//! ONNX frontend: `.onnx` file → `GraphIr`, without ONNX Runtime.
//!
//! This is the piece that makes the engine standalone. Until now the IR came
//! from `OrtGraph`, so running a model meant having ORT in the process; here
//! the same IR is built by reading the protobuf directly.

mod convert;
pub mod shape;

/// Types generated from `proto/onnx.proto` by the build script.
pub mod proto {
    // generated code: style lints have no place where they would be correct
    #![allow(clippy::all, clippy::pedantic)]
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use convert::{Error, Result, declared_types, model_to_ir, read_tensor_proto};
pub use shape::{Conflict, Dim, TensorType, Types};

use prost::Message;
use std::path::Path;

/// A loaded model: the graph the engine runs, plus the type of every value.
#[derive(Debug)]
pub struct Model {
    pub graph: onnx_vulkan_core::GraphIr,
    pub types: Types,
    /// Places where the file's declared type and the inferred one disagree.
    ///
    /// Not fatal by construction: it is the caller who knows whether a
    /// mismatched shape is a broken export or a limit of our inference.
    pub conflicts: Vec<Conflict>,
}

impl Model {
    /// Type of a graph input, for a caller that has to build one.
    pub fn input_type(&self, name: &str) -> Option<&TensorType> {
        self.graph
            .inputs
            .iter()
            .any(|i| i == name)
            .then(|| self.types.get(name))
            .flatten()
    }
}

/// Loads an `.onnx` file and converts it into the core's IR.
///
/// External weights are resolved relative to the model's own directory, which
/// is what the ONNX spec prescribes and the only interpretation that keeps a
/// downloaded model self-contained.
pub fn load(path: impl AsRef<Path>) -> Result<Model> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|e| Error::Malformed(format!("lettura di {}: {e}", path.display())))?;
    load_from_bytes(&bytes, path.parent())
}

/// Same as [`load`], for a model already in memory.
///
/// `base_dir` is where external weights are looked up; `None` rejects a model
/// that uses them instead of guessing.
pub fn load_from_bytes(bytes: &[u8], base_dir: Option<&Path>) -> Result<Model> {
    let model = proto::ModelProto::decode(bytes)
        .map_err(|e| Error::Malformed(format!("decodifica protobuf: {e}")))?;
    let graph = model_to_ir(&model, base_dir)?;
    let declared = model.graph.as_ref().map(declared_types).unwrap_or_default();
    let (types, conflicts) = shape::infer(&graph, &declared);
    Ok(Model {
        graph,
        types,
        conflicts,
    })
}
