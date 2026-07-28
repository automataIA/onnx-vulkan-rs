//! Owned intermediate representation of an ONNX graph.

use crate::host_ops::HostTensor;
use std::collections::HashMap;

/// Elementary `TensorProto` type according to ONNX numeric codes.
///
/// The enum explicitly indicates these codes belong to the ONNX format and not
/// to ONNX Runtime. Sub-byte types do not have integer byte widths and
/// will be handled by the frontend alongside their packing format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum ElementType {
    Undefined = 0,
    Float32 = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Float64 = 11,
    Uint32 = 12,
    Uint64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    Bfloat16 = 16,
    Float8E4M3Fn = 17,
    Float8E4M3Fnuz = 18,
    Float8E5M2 = 19,
    Float8E5M2Fnuz = 20,
    Uint4 = 21,
    Int4 = 22,
    Float4E2M1 = 23,
    Uint2 = 24,
    Int2 = 25,
    Float8E8M0 = 26,
}

impl ElementType {
    /// Logical bit width of a fixed-size element.
    pub const fn bit_width(self) -> Option<usize> {
        match self {
            Self::Float32 | Self::Int32 | Self::Uint32 => Some(32),
            Self::Uint8
            | Self::Int8
            | Self::Bool
            | Self::Float8E4M3Fn
            | Self::Float8E4M3Fnuz
            | Self::Float8E5M2
            | Self::Float8E5M2Fnuz
            | Self::Float8E8M0 => Some(8),
            Self::Uint16 | Self::Int16 | Self::Float16 | Self::Bfloat16 => Some(16),
            Self::Int64 | Self::Float64 | Self::Uint64 | Self::Complex64 => Some(64),
            Self::Complex128 => Some(128),
            Self::Uint4 | Self::Int4 | Self::Float4E2M1 => Some(4),
            Self::Uint2 | Self::Int2 => Some(2),
            Self::Undefined | Self::String => None,
        }
    }

    /// Byte width of a fixed-size element.
    ///
    /// Returns `None` for sub-byte types: use [`storage_len`] when
    /// packed buffer size is needed.
    pub const fn byte_width(self) -> Option<usize> {
        match self.bit_width() {
            Some(bits) if bits >= 8 => Some(bits / 8),
            _ => None,
        }
    }
}

impl TryFrom<i32> for ElementType {
    type Error = UnknownElementType;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Undefined),
            1 => Ok(Self::Float32),
            2 => Ok(Self::Uint8),
            3 => Ok(Self::Int8),
            4 => Ok(Self::Uint16),
            5 => Ok(Self::Int16),
            6 => Ok(Self::Int32),
            7 => Ok(Self::Int64),
            8 => Ok(Self::String),
            9 => Ok(Self::Bool),
            10 => Ok(Self::Float16),
            11 => Ok(Self::Float64),
            12 => Ok(Self::Uint32),
            13 => Ok(Self::Uint64),
            14 => Ok(Self::Complex64),
            15 => Ok(Self::Complex128),
            16 => Ok(Self::Bfloat16),
            17 => Ok(Self::Float8E4M3Fn),
            18 => Ok(Self::Float8E4M3Fnuz),
            19 => Ok(Self::Float8E5M2),
            20 => Ok(Self::Float8E5M2Fnuz),
            21 => Ok(Self::Uint4),
            22 => Ok(Self::Int4),
            23 => Ok(Self::Float4E2M1),
            24 => Ok(Self::Uint2),
            25 => Ok(Self::Int2),
            26 => Ok(Self::Float8E8M0),
            code => Err(UnknownElementType(code)),
        }
    }
}

/// ONNX tensor code unrecognized by core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownElementType(pub i32);

impl std::fmt::Display for UnknownElementType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "unknown ONNX tensor element type {}", self.0)
    }
}

impl std::error::Error for UnknownElementType {}

/// Bytes per element, or zero for unknown/variable-size types.
///
/// Keeps the signature used by the existing interpreter for now. The frontend
/// must reject unsupported types before execution.
pub fn elem_size(dtype: i32) -> usize {
    ElementType::try_from(dtype)
        .ok()
        .and_then(ElementType::byte_width)
        .unwrap_or(0)
}

/// Packed buffer size for `element_count` ONNX elements.
pub fn storage_len(dtype: i32, element_count: usize) -> Option<usize> {
    let bits = ElementType::try_from(dtype).ok()?.bit_width()?;
    element_count
        .checked_mul(bits)?
        .checked_add(7)
        .map(|n| n / 8)
}

/// Value of an ONNX attribute.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Int(i64),
    Ints(Vec<i64>),
    Float(f32),
    Floats(Vec<f32>),
    String(String),
    Tensor(InitializerIr),
}

impl AttrValue {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_tensor(&self) -> Option<&InitializerIr> {
        match self {
            Self::Tensor(tensor) => Some(tensor),
            _ => None,
        }
    }

    pub fn as_ints(&self) -> Option<&[i64]> {
        match self {
            Self::Ints(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float(value) => Some(*value),
            _ => None,
        }
    }
}

/// Owned ONNX node, independent of the source format.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeIr {
    /// ONNX domain; empty string for the standard `ai.onnx` operators.
    pub domain: String,
    pub op: String,
    /// Operator schema version resolved from the model's opset.
    pub since_version: i32,
    pub name: String,
    /// Names of input values; empty string = missing optional input.
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, AttrValue>,
}

/// Constant tensor copied into host memory.
#[derive(Debug, Clone, PartialEq)]
pub struct InitializerIr {
    /// Numeric `TensorProto.DataType` code from the ONNX standard.
    pub dtype: i32,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
}

/// Complete IR of an ONNX graph or subgraph.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GraphIr {
    /// Nodes in stable topological order.
    pub nodes: Vec<NodeIr>,
    pub initializers: HashMap<String, InitializerIr>,
    /// External graph inputs, excluding constant initializers.
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// Ops whose `axes` migrated from attribute to input, with the input index.
///
/// ONNX moved `axes` to inputs at different times: `ReduceSum` from
/// opset 13, the other reductions from opset 18. The two forms describe the
/// same node.
const AXES_AS_INPUT: [(&str, usize); 3] = [("ReduceSum", 1), ("ReduceMean", 1), ("ReduceMax", 1)];

/// Values produced by `Constant` nodes, indexed by output name.
///
/// They are as constant as initializers: ORT usually folds them already, but
/// not when optimizations are disabled. Merging them with initializers makes
/// canonicalization independent of the optimization level.
pub fn constant_outputs(nodes: &[NodeIr]) -> HashMap<String, InitializerIr> {
    nodes
        .iter()
        .filter(|n| n.op == "Constant" && n.domain.is_empty())
        .filter_map(|n| {
            let tensor = n.attrs.get("value")?.as_tensor()?;
            Some((n.outputs.first()?.clone(), tensor.clone()))
        })
        .collect()
}

/// Promotes parameters passed as **constant input** to an attribute.
///
/// This is needed because the capability check sees one node at a time and
/// cannot resolve the value of an input: without this normalization a
/// `ReduceSum` with axes in an initializer would be rejected, splitting the
/// block, even though it is identical to the attribute form.
///
/// Must be applied **before** the support check and before execution, on the
/// same IR, so the two decisions cannot diverge. Unaffected nodes stay
/// unchanged.
pub fn fold_constant_params(node: &mut NodeIr, initializers: &HashMap<String, InitializerIr>) {
    let Some(&(_, index)) = AXES_AS_INPUT.iter().find(|(op, _)| *op == node.op) else {
        return;
    };
    if node.attrs.contains_key("axes") {
        return;
    }
    let Some(name) = node.inputs.get(index).filter(|n| !n.is_empty()) else {
        return;
    };
    let Some(init) = initializers.get(name) else {
        return;
    };
    let Ok(axes) = HostTensor::new(init.dtype, init.shape.clone(), init.data.clone()).to_i64()
    else {
        return;
    };
    node.attrs.insert("axes".to_string(), AttrValue::Ints(axes));
    node.inputs.truncate(index);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_type_codes_and_widths_match_onnx() {
        assert_eq!(ElementType::try_from(1), Ok(ElementType::Float32));
        assert_eq!(ElementType::try_from(10), Ok(ElementType::Float16));
        assert_eq!(ElementType::try_from(999), Err(UnknownElementType(999)));
        assert_eq!(elem_size(ElementType::Int8 as i32), 1);
        assert_eq!(elem_size(ElementType::Float32 as i32), 4);
        assert_eq!(elem_size(ElementType::Complex128 as i32), 16);
        assert_eq!(elem_size(ElementType::String as i32), 0);
        assert_eq!(ElementType::Int4.byte_width(), None);
        assert_eq!(storage_len(ElementType::Int4 as i32, 3), Some(2));
        assert_eq!(storage_len(ElementType::Uint2 as i32, 5), Some(2));
    }

    #[test]
    fn graph_ir_is_owned_and_frontend_independent() {
        let initializer = InitializerIr {
            dtype: ElementType::Float32 as i32,
            shape: vec![2],
            data: vec![0; 8],
        };
        let graph = GraphIr {
            nodes: vec![NodeIr {
                domain: String::new(),
                op: "Add".into(),
                since_version: 14,
                name: "add".into(),
                inputs: vec!["x".into(), "bias".into()],
                outputs: vec!["y".into()],
                attrs: HashMap::new(),
            }],
            initializers: HashMap::from([("bias".into(), initializer)]),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
        };

        assert_eq!(graph.nodes[0].op, "Add");
        assert_eq!(graph.initializers["bias"].data.len(), 8);
    }
}
