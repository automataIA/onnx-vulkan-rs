//! `ModelProto` → `GraphIr`.
//!
//! The conversion is deliberately literal: it copies what the file says and
//! refuses what it cannot represent, without inventing defaults. The one
//! transformation it does apply is the canonicalization of constant parameters
//! passed as inputs, because that is what decides op coverage — and coverage
//! must not depend on which frontend built the IR.
//!
//! Shape inference stays out of here.

use crate::proto;
use onnx_vulkan_core::{AttrValue, GraphIr, InitializerIr, NodeIr};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The file is not a readable ONNX model.
    Malformed(String),
    /// The model is well-formed but uses something this frontend does not read.
    Unsupported(String),
    /// External weights that could not be read from disk.
    ExternalData(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(message) => write!(formatter, "malformed ONNX model: {message}"),
            Self::Unsupported(message) => {
                write!(formatter, "unsupported ONNX model: {message}")
            }
            Self::ExternalData(message) => write!(formatter, "unreadable external weights: {message}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Converts a parsed model into the core's IR.
///
/// `base_dir` is the directory the model was loaded from: external weights
/// declare a path relative to it. Pass `None` to reject a model that uses them
/// rather than guessing where they are.
pub fn model_to_ir(model: &proto::ModelProto, base_dir: Option<&Path>) -> Result<GraphIr> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| Error::Malformed("no graph in the model".into()))?;

    // An op is identified by (domain, version): the version is not on the
    // node but in the opset the model imports for that domain.
    let opsets: HashMap<&str, i64> = model
        .opset_import
        .iter()
        .map(|o| (o.domain(), o.version()))
        .collect();

    let mut initializers = HashMap::with_capacity(graph.initializer.len());
    for tensor in &graph.initializer {
        let name = tensor.name().to_string();
        if name.is_empty() {
            return Err(Error::Malformed("unnamed initializer".into()));
        }
        initializers.insert(name, tensor_to_initializer(tensor, base_dir)?);
    }
    if !graph.sparse_initializer.is_empty() {
        return Err(Error::Unsupported(format!(
            "{} sparse initializers",
            graph.sparse_initializer.len()
        )));
    }

    let mut nodes = Vec::with_capacity(graph.node.len());
    for node in &graph.node {
        nodes.push(node_to_ir(node, &opsets, base_dir)?);
    }

    // Same normalization the EP applies in `GetCapability`: `axes` and the
    // other parameters passed as constant inputs become attributes again.
    // Without it, the same graph would be covered for one path and not the
    // other — and coverage is what decides whether a model runs or fails.
    let mut constants = onnx_vulkan_core::constant_outputs(&nodes);
    constants.extend(initializers.clone());
    for node in &mut nodes {
        onnx_vulkan_core::fold_constant_params(node, &constants);
    }

    // Initializers also appear among the graph inputs (IR version ≥ 4 makes
    // them overridable). For the core, inputs are what the caller must provide,
    // so the constants must be removed.
    let inputs = graph
        .input
        .iter()
        .map(|v| v.name().to_string())
        .filter(|name| !initializers.contains_key(name))
        .collect();
    let outputs = graph.output.iter().map(|v| v.name().to_string()).collect();

    Ok(GraphIr {
        nodes,
        initializers,
        inputs,
        outputs,
    })
}

/// Reads a serialized `TensorProto`, the format of the `.pb` files in the ONNX
/// model zoo's `test_data_set_*` directories.
///
/// Returns the tensor's own name alongside it: in those files the name is what
/// binds the tensor to a graph input or output, and losing it would leave the
/// caller guessing by position.
pub fn read_tensor_proto(bytes: &[u8]) -> Result<(String, InitializerIr)> {
    use prost::Message;
    let tensor = proto::TensorProto::decode(bytes)
        .map_err(|e| Error::Malformed(format!("TensorProto decode: {e}")))?;
    let name = tensor.name().to_string();
    Ok((name, tensor_to_initializer(&tensor, None)?))
}

/// Types the file **declares**, from `graph.input`, `graph.output` and
/// `value_info`.
///
/// How much is there varies wildly between exporters — measured on our matrix,
/// from 0% of the intermediates (mobilenetv2, yolov8n) to 100% (rfdetr). That
/// spread is exactly why inference has to exist and why it must also accept
/// what it finds.
pub fn declared_types(graph: &proto::GraphProto) -> HashMap<String, crate::shape::TensorType> {
    graph
        .input
        .iter()
        .chain(&graph.output)
        .chain(&graph.value_info)
        .filter_map(|info| Some((info.name().to_string(), value_info_type(info)?)))
        .collect()
}

fn value_info_type(info: &proto::ValueInfoProto) -> Option<crate::shape::TensorType> {
    use crate::shape::{Dim, TensorType};
    use proto::type_proto::Value;

    let Value::TensorType(tensor) = info.r#type.as_ref()?.value.as_ref()? else {
        // sequence, map and optional do not have a single shape: outside the
        // core's tensor model
        return None;
    };
    let dtype = tensor.elem_type;
    let shape = tensor.shape.as_ref().map(|s| {
        s.dim
            .iter()
            .map(|d| match &d.value {
                Some(proto::tensor_shape_proto::dimension::Value::DimValue(n)) => Dim::Fixed(*n),
                Some(proto::tensor_shape_proto::dimension::Value::DimParam(p)) => {
                    Dim::Symbol(p.clone())
                }
                None => Dim::Unknown,
            })
            .collect()
    });
    Some(TensorType { dtype, shape })
}

fn node_to_ir(
    node: &proto::NodeProto,
    opsets: &HashMap<&str, i64>,
    base_dir: Option<&Path>,
) -> Result<NodeIr> {
    let domain = node.domain().to_string();
    let version = opsets.get(domain.as_str()).copied().ok_or_else(|| {
        Error::Unsupported(format!(
            "node '{}' ({}): domain '{}' not imported by the model",
            node.name(),
            node.op_type(),
            domain
        ))
    })?;

    let mut attrs = HashMap::with_capacity(node.attribute.len());
    for attribute in &node.attribute {
        let name = attribute.name().to_string();
        if let Some(value) = attribute_to_ir(attribute, base_dir)? {
            attrs.insert(name, value);
        }
    }

    Ok(NodeIr {
        domain,
        op: node.op_type().to_string(),
        since_version: version as i32,
        name: node.name().to_string(),
        inputs: node.input.clone(),
        outputs: node.output.clone(),
        attrs,
    })
}

/// `None` when the attribute carries no value (ONNX allows a name-only
/// attribute as a reference inside a function body).
fn attribute_to_ir(
    attribute: &proto::AttributeProto,
    base_dir: Option<&Path>,
) -> Result<Option<AttrValue>> {
    use proto::attribute_proto::AttributeType;

    let kind = attribute.r#type();
    let described =
        |what: &str| Error::Unsupported(format!("attribute '{}' of type {what}", attribute.name()));

    Ok(Some(match kind {
        AttributeType::Int => AttrValue::Int(attribute.i()),
        AttributeType::Ints => AttrValue::Ints(attribute.ints.clone()),
        AttributeType::Float => AttrValue::Float(attribute.f()),
        AttributeType::Floats => AttrValue::Floats(attribute.floats.clone()),
        AttributeType::String => AttrValue::String(
            String::from_utf8(attribute.s().to_vec())
                .map_err(|_| described("non-UTF-8 string"))?,
        ),
        AttributeType::Tensor => {
            let tensor = attribute
                .t
                .as_ref()
                .ok_or_else(|| described("tensor without value"))?;
            AttrValue::Tensor(tensor_to_initializer(tensor, base_dir)?)
        }
        AttributeType::Undefined => return Ok(None),
        // subgraphs (If/Loop/Scan), multiple strings, sparse tensors and
        // type-proto: the interpreter does not run them, so the node carrying
        // them must be rejected outright, not silently emptied
        other => return Err(described(&format!("{other:?}"))),
    }))
}

/// `TensorProto` → raw little-endian bytes.
///
/// ONNX stores weights in three mutually exclusive ways: `raw_data` (already
/// the layout we want), the typed repeated fields, or an external file. The
/// typed fields are the annoying case — ONNX packs several small dtypes into
/// `int32_data`, so the destination width comes from `data_type`, never from
/// the field the values arrived in.
fn tensor_to_initializer(
    tensor: &proto::TensorProto,
    base_dir: Option<&Path>,
) -> Result<InitializerIr> {
    use proto::tensor_proto::DataType;

    let dtype = tensor.data_type();
    let shape = tensor.dims.clone();
    let name = tensor.name();

    if tensor.data_location() == proto::tensor_proto::DataLocation::External {
        return Ok(InitializerIr {
            dtype,
            shape,
            data: read_external(tensor, base_dir)?,
        });
    }

    if !tensor.raw_data().is_empty() {
        return Ok(InitializerIr {
            dtype,
            shape,
            data: tensor.raw_data().to_vec(),
        });
    }

    let kind = DataType::try_from(dtype)
        .map_err(|_| Error::Malformed(format!("tensor '{name}': dtype {dtype} unknown")))?;
    let data = match kind {
        DataType::Float => le_bytes(&tensor.float_data, |v| v.to_le_bytes().to_vec()),
        DataType::Double => le_bytes(&tensor.double_data, |v| v.to_le_bytes().to_vec()),
        DataType::Int64 => le_bytes(&tensor.int64_data, |v| v.to_le_bytes().to_vec()),
        DataType::Uint64 | DataType::Uint32 => {
            let width = if kind == DataType::Uint32 { 4 } else { 8 };
            le_bytes(&tensor.uint64_data, |v| v.to_le_bytes()[..width].to_vec())
        }
        // int32_data is the shared container for narrow dtypes: the value must
        // be truncated to the dtype's true width, not the field's width
        DataType::Int32 => le_bytes(&tensor.int32_data, |v| v.to_le_bytes().to_vec()),
        DataType::Int16 | DataType::Uint16 | DataType::Float16 | DataType::Bfloat16 => {
            le_bytes(&tensor.int32_data, |v| v.to_le_bytes()[..2].to_vec())
        }
        DataType::Int8 | DataType::Uint8 | DataType::Bool => {
            le_bytes(&tensor.int32_data, |v| vec![v.to_le_bytes()[0]])
        }
        other => {
            return Err(Error::Unsupported(format!(
                "tensor '{name}': dtype {other:?} without a readable data field"
            )));
        }
    };

    Ok(InitializerIr { dtype, shape, data })
}

fn le_bytes<T: Copy>(values: &[T], encode: impl Fn(T) -> Vec<u8>) -> Vec<u8> {
    values.iter().flat_map(|v| encode(*v)).collect()
}

/// Reads weights stored outside the model file (`data_location = EXTERNAL`).
///
/// Models above 2 GB have no choice — the protobuf limit forces it — but the
/// form is common well below that too, and rejecting it would leave out most
/// large models.
fn read_external(tensor: &proto::TensorProto, base_dir: Option<&Path>) -> Result<Vec<u8>> {
    let name = tensor.name();
    let entry = |key: &str| {
        tensor
            .external_data
            .iter()
            .find(|kv| kv.key() == key)
            .map(|kv| kv.value().to_string())
    };
    let number = |key: &str| -> Result<Option<u64>> {
        entry(key)
            .map(|v| {
                v.parse::<u64>().map_err(|_| {
                    Error::ExternalData(format!("tensor '{name}': '{key}' = '{v}' not numeric"))
                })
            })
            .transpose()
    };

    let location = entry("location").ok_or_else(|| {
        Error::ExternalData(format!("tensor '{name}': missing 'location' key"))
    })?;
    // the path is relative to the model directory, and must stay so: a `..`
    // in a downloaded file would read outside the model tree
    if Path::new(&location)
        .components()
        .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(Error::ExternalData(format!(
            "tensor '{name}': external path '{location}' not relative"
        )));
    }
    let base = base_dir.ok_or_else(|| {
        Error::ExternalData(format!(
            "tensor '{name}': weights in '{location}', but the model directory is not known"
        ))
    })?;

    let path = base.join(&location);
    let bytes = std::fs::read(&path)
        .map_err(|e| Error::ExternalData(format!("reading {}: {e}", path.display())))?;
    let offset = number("offset")?.unwrap_or(0) as usize;
    let length = number("length")?
        .map(|n| n as usize)
        .unwrap_or_else(|| bytes.len().saturating_sub(offset));
    let end = offset.checked_add(length).ok_or_else(|| {
        Error::ExternalData(format!("tensor '{name}': offset+length overflow"))
    })?;
    if end > bytes.len() {
        return Err(Error::ExternalData(format!(
            "tensor '{name}': requested bytes {offset}..{end} of a file of {}",
            bytes.len()
        )));
    }
    Ok(bytes[offset..end].to_vec())
}
