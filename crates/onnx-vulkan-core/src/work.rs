//! Analytic FLOPs and bytes per node — the two inputs of a Roofline.
//!
//! The profiler measures milliseconds; it cannot measure work. Work is a
//! property of the *graph*, derivable from the shapes and dtypes the
//! interpreter already has, so it is computed here once per node and handed to
//! `vk_compute::stats`, which divides it by the time it measured.
//!
//! Two deliberate choices about what the numbers mean:
//!
//! - **`bytes` is compulsory traffic**, i.e. each distinct input tensor read
//!   once plus each output written once. It is *not* the traffic the kernel
//!   actually issues: an implicit-GEMM `Conv` re-reads its input tile per
//!   output tile, and a split-K kernel writes partials it later re-reads. The
//!   compulsory model is the standard Roofline denominator — it answers "how
//!   close is this to the best any implementation could do", which is the
//!   question an optimization loop asks. A kernel above 100% of peak bandwidth
//!   is therefore impossible; a kernel far below it may still be bandwidth-bound
//!   because it is *wasting* bandwidth, and that is the finding, not an error.
//! - **`flops` is exact for the matmul family and approximate elsewhere.**
//!   Only `MatMul`/`Gemm`/`Conv` and their integer variants have an
//!   unambiguous count; for everything else the count is per-element and the
//!   useful column is `gb_s`, not `gflops`.
//!
//! Ops not modelled here return `None` and simply carry no Roofline fields.

use crate::execution::device_storage_bytes;
use crate::graph::{AttrValue, NodeIr};

/// Work of one node execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Work {
    pub flops: u64,
    pub bytes: u64,
}

/// Shapes and dtypes of a node's tensors, as the caller can supply them.
///
/// A trait rather than `&Env` so the model stays testable without a Vulkan
/// device: the only thing it needs from the environment is metadata.
pub trait TensorMeta {
    fn shape(&self, name: &str) -> Option<Vec<i64>>;
    fn dtype(&self, name: &str) -> Option<i32>;
}

fn elems(shape: &[i64]) -> u64 {
    shape.iter().map(|d| (*d).max(0) as u64).product()
}

/// Compulsory traffic: every distinct input read once, every output written once.
fn traffic(node: &NodeIr, meta: &impl TensorMeta) -> u64 {
    let mut seen: Vec<&str> = Vec::new();
    let mut total = 0u64;
    for name in node.inputs.iter().chain(node.outputs.iter()) {
        if name.is_empty() || seen.contains(&name.as_str()) {
            continue;
        }
        seen.push(name.as_str());
        let (Some(shape), Some(dtype)) = (meta.shape(name), meta.dtype(name)) else {
            continue;
        };
        let count = elems(&shape) as usize;
        total += device_storage_bytes(dtype, count).unwrap_or(0);
    }
    total
}

/// `2·|out|·K` — one multiply and one add per contracted element.
fn matmul_flops(node: &NodeIr, meta: &impl TensorMeta) -> Option<u64> {
    let a = meta.shape(node.inputs.first()?)?;
    let out = meta.shape(node.outputs.first()?)?;
    // K is A's last dimension for MatMul/MatMulInteger; Gemm with transA=1
    // contracts over the first instead
    let trans_a = node
        .attrs
        .get("transA")
        .and_then(AttrValue::as_i64)
        .unwrap_or(0);
    let k = if trans_a == 1 {
        *a.first()?
    } else {
        *a.last()?
    };
    Some(2 * elems(&out) * k.max(0) as u64)
}

/// `2·|out|·(C_in/group)·∏kernel`.
fn conv_flops(node: &NodeIr, meta: &impl TensorMeta) -> Option<u64> {
    let w = meta.shape(node.inputs.get(1)?)?;
    let out = meta.shape(node.outputs.first()?)?;
    // weight is [C_out, C_in/group, *kernel] for Conv and
    // [C_in, C_out/group, *kernel] for ConvTranspose: in both cases everything
    // but the first dimension is contracted per output element
    Some(2 * elems(&out) * elems(w.get(1..)?))
}

/// `None` for an op whose work this model does not claim to know.
pub fn node_work(node: &NodeIr, meta: &impl TensorMeta) -> Option<Work> {
    let bytes = traffic(node, meta);
    let out_elems = || meta.shape(node.outputs.first()?).map(|s| elems(&s));
    let in_elems = || meta.shape(node.inputs.first()?).map(|s| elems(&s));
    let flops = match node.op.as_str() {
        "MatMul" | "MatMulInteger" | "Gemm" => matmul_flops(node, meta)?,
        "Conv" | "ConvInteger" | "ConvTranspose" => conv_flops(node, meta)?,
        // per-element counts, approximate by construction (see the module doc):
        // an `exp` is not one flop and a `Div` is not one either, but the
        // column that matters for these is `gb_s`
        "Softmax" | "LogSoftmax" => 3 * out_elems()?,
        "LayerNormalization"
        | "InstanceNormalization"
        | "BatchNormalization"
        | "SkipLayerNormalization" => 5 * out_elems()?,
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceL2"
        | "GlobalAveragePool" | "MaxPool" | "AveragePool" => in_elems()?,
        "Add"
        | "Sub"
        | "Mul"
        | "Div"
        | "Pow"
        | "Sqrt"
        | "Exp"
        | "Log"
        | "Erf"
        | "Sigmoid"
        | "Relu"
        | "LeakyRelu"
        | "Clip"
        | "Tanh"
        | "Gelu"
        | "Neg"
        | "Abs"
        | "Min"
        | "Max"
        | "Where"
        | "Equal"
        | "Greater"
        | "Less"
        | "QuantizeLinear"
        | "DequantizeLinear"
        | "DynamicQuantizeLinear" => out_elems()?,
        // pure movement: no arithmetic, but the traffic is real
        "Reshape" | "Squeeze" | "Unsqueeze" | "Transpose" | "Concat" | "Slice" | "Gather"
        | "Cast" | "Identity" | "Pad" | "Expand" | "Split" | "Resize" | "Tile" => 0,
        _ => return None,
    };
    Some(Work { flops, bytes })
}

impl TensorMeta for crate::execution::ExecutionEnv<'_, '_> {
    fn shape(&self, name: &str) -> Option<Vec<i64>> {
        self.shape_of(name).ok()
    }
    fn dtype(&self, name: &str) -> Option<i32> {
        self.dtype_of(name).ok()
    }
}

/// Charges the node's work to the kernel that ran it. No-op unless the
/// profiler is on: without milliseconds the work has nothing to divide into.
pub fn record(node: &NodeIr, meta: &impl TensorMeta) {
    if !vk_compute::stats::enabled() {
        return;
    }
    if let Some(work) = node_work(node, meta) {
        vk_compute::stats::record_work(vk_compute::stats::primary_op(), work.flops, work.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct Meta(HashMap<&'static str, (Vec<i64>, i32)>);
    impl TensorMeta for Meta {
        fn shape(&self, name: &str) -> Option<Vec<i64>> {
            self.0.get(name).map(|(s, _)| s.clone())
        }
        fn dtype(&self, name: &str) -> Option<i32> {
            self.0.get(name).map(|(_, d)| *d)
        }
    }

    fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeIr {
        NodeIr {
            domain: String::new(),
            op: op.into(),
            since_version: 13,
            name: String::new(),
            inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
            outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
            attrs: HashMap::new(),
        }
    }

    /// roberta's GEMV, the geometry `CLAUDE.md` quotes as 170 MFLOP / 340 MB.
    #[test]
    fn gemv_matches_the_documented_figures() {
        const FLOAT: i32 = 1;
        let meta = Meta(HashMap::from([
            ("x", (vec![1, 1, 768], FLOAT)),
            ("w", (vec![768, 3072], FLOAT)),
            ("y", (vec![1, 1, 3072], FLOAT)),
        ]));
        let w = node_work(&node("MatMul", &["x", "w"], &["y"]), &meta).unwrap();
        assert_eq!(w.flops, 2 * 3072 * 768);
        // the weight dominates: 768·3072·4 B
        assert_eq!(w.bytes, 768 * 3072 * 4 + 768 * 4 + 3072 * 4);
    }

    #[test]
    fn conv_contracts_over_the_whole_filter() {
        const FLOAT: i32 = 1;
        let meta = Meta(HashMap::from([
            ("x", (vec![1, 64, 56, 56], FLOAT)),
            ("w", (vec![64, 64, 3, 3], FLOAT)),
            ("y", (vec![1, 64, 56, 56], FLOAT)),
        ]));
        let w = node_work(&node("Conv", &["x", "w"], &["y"]), &meta).unwrap();
        assert_eq!(w.flops, 2 * 64 * 56 * 56 * 64 * 3 * 3);
    }

    #[test]
    fn unmodelled_ops_carry_no_work() {
        let meta = Meta(HashMap::new());
        assert!(node_work(&node("NonMaxSuppression", &["x"], &["y"]), &meta).is_none());
    }
}
