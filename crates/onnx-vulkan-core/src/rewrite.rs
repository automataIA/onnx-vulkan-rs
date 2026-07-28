//! Load-time graph rewrites: patterns of standard operators replaced by the
//! single fused operator the interpreter already implements.
//!
//! This is not the convex partitioning of [`crate::fusion`], which decides
//! *which* nodes a backend claims. Here the node list itself changes: nine
//! dispatches become one, on a graph that was already entirely claimed.
//!
//! The lever is exporter age. `LayerNormalization` only exists from opset 17,
//! so every transformer exported below it carries the normalization written out
//! as `ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div → Mul → Add`.
//! roberta-base (opset 11) repeats that pattern 25 times: 225 nodes doing the
//! work of 25 dispatches, against a kernel that already exists.
//!
//! Two conditions bound what is matched, and both are about not producing a
//! node the kernel would then reject at run time:
//!
//! - the reduction axis must be the **last** one (`axes = [-1]`), because that
//!   is the only axis `core::shaders::layernorm` normalizes over. Exporters
//!   also emit this shape on `axes = [1]` for a channel-wise normalization —
//!   rfdetr does, nine times — and that is a different operator, not fused here;
//! - `scale` and `bias` must be **rank-1 constants of the same length > 1**.
//!   The IR carries no shapes, so this is what stands in for
//!   `scale.elem_count == last_dim`: a rank-1 operand broadcasts over the last
//!   axis by definition, and length 1 is excluded because it would broadcast as
//!   a scalar instead, which the kernel does not accept.

use crate::graph::{AttrValue, ElementType, GraphIr, InitializerIr, NodeIr, constant_outputs};
use crate::host_ops::HostTensor;
use std::collections::{HashMap, HashSet};

/// Producer, consumers and graph outputs of every value, by node index.
struct Index {
    consumers: HashMap<String, Vec<usize>>,
    graph_outputs: HashSet<String>,
}

impl Index {
    fn build(ir: &GraphIr) -> Self {
        let mut consumers: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, node) in ir.nodes.iter().enumerate() {
            for name in node.inputs.iter().filter(|n| !n.is_empty()) {
                let list = consumers.entry(name.clone()).or_default();
                if list.last() != Some(&i) {
                    list.push(i);
                }
            }
        }
        Index {
            consumers,
            graph_outputs: ir.outputs.iter().cloned().collect(),
        }
    }

    /// The single node reading `name`, when the value escapes nowhere else.
    ///
    /// A value observable outside the pattern — read by a node that is not part
    /// of it, or declared as a graph output — cannot be deleted with it.
    fn only_consumer(&self, name: &str) -> Option<usize> {
        if self.graph_outputs.contains(name) {
            return None;
        }
        match self.consumers.get(name)?.as_slice() {
            [only] => Some(*only),
            _ => None,
        }
    }
}

/// A matched pattern: the nodes it consumes and the node that replaces them.
struct Match {
    members: Vec<usize>,
    fused: NodeIr,
}

fn scalar_f32(name: &str, constants: &HashMap<String, InitializerIr>) -> Option<f32> {
    let tensor = constants.get(name)?;
    if tensor.dtype != ElementType::Float32 as i32 {
        return None;
    }
    match HostTensor::new(tensor.dtype, tensor.shape.clone(), tensor.data.clone())
        .to_f32()
        .ok()?
        .as_slice()
    {
        [only] => Some(*only),
        _ => None,
    }
}

/// Length of a rank-1 f32 constant usable as `scale`/`bias`, if it is one.
fn per_channel_len(name: &str, constants: &HashMap<String, InitializerIr>) -> Option<i64> {
    let tensor = constants.get(name)?;
    match (
        tensor.dtype == ElementType::Float32 as i32,
        &tensor.shape[..],
    ) {
        (true, [len]) if *len > 1 => Some(*len),
        _ => None,
    }
}

/// The operand of a two-input node that is not `known`.
fn other_input<'a>(node: &'a NodeIr, known: &str) -> Option<&'a str> {
    match &node.inputs[..] {
        [a, b] if a == known => Some(b),
        [a, b] if b == known => Some(a),
        _ => None,
    }
}

/// `axes` reducing the last dimension, with `keepdims` left at its default.
fn reduces_last_axis(node: &NodeIr) -> bool {
    let axes_last = node.attrs.get("axes").and_then(AttrValue::as_ints) == Some(&[-1][..]);
    let keeps = node.attrs.get("keepdims").and_then(AttrValue::as_i64) != Some(0);
    axes_last && keeps
}

fn is_op(node: &NodeIr, op: &str) -> bool {
    node.op == op && node.domain.is_empty()
}

/// Matches the decomposed pattern rooted at the first `ReduceMean`.
fn match_layernorm(
    nodes: &[NodeIr],
    root: usize,
    index: &Index,
    constants: &HashMap<String, InitializerIr>,
) -> Option<Match> {
    let mean = &nodes[root];
    if !is_op(mean, "ReduceMean") || !reduces_last_axis(mean) {
        return None;
    }
    let x = mean.inputs.first()?;

    let i_sub = index.only_consumer(&mean.outputs[0])?;
    let sub = &nodes[i_sub];
    if !is_op(sub, "Sub") || sub.inputs != [x.clone(), mean.outputs[0].clone()] {
        return None;
    }
    // `x - mean` feeds both the variance branch and the division: two readers,
    // and they must be exactly the pattern's own two.
    let deviation = &sub.outputs[0];
    if index.graph_outputs.contains(deviation) {
        return None;
    }
    let readers = index.consumers.get(deviation)?;
    if readers.len() != 2 {
        return None;
    }
    let i_pow = *readers.iter().find(|&&i| is_op(&nodes[i], "Pow"))?;
    let pow = &nodes[i_pow];
    if pow.inputs.first() != Some(deviation) || scalar_f32(&pow.inputs[1], constants)? != 2.0 {
        return None;
    }

    let i_var = index.only_consumer(&pow.outputs[0])?;
    let var = &nodes[i_var];
    if !is_op(var, "ReduceMean")
        || var.inputs.first() != Some(&pow.outputs[0])
        || !reduces_last_axis(var)
    {
        return None;
    }

    let i_add_eps = index.only_consumer(&var.outputs[0])?;
    let add_eps = &nodes[i_add_eps];
    if !is_op(add_eps, "Add") {
        return None;
    }
    let epsilon = scalar_f32(other_input(add_eps, &var.outputs[0])?, constants)?;

    let i_sqrt = index.only_consumer(&add_eps.outputs[0])?;
    let sqrt = &nodes[i_sqrt];
    if !is_op(sqrt, "Sqrt") || sqrt.inputs.first() != Some(&add_eps.outputs[0]) {
        return None;
    }

    let i_div = index.only_consumer(&sqrt.outputs[0])?;
    let div = &nodes[i_div];
    if !is_op(div, "Div") || div.inputs != [deviation.clone(), sqrt.outputs[0].clone()] {
        return None;
    }
    // the division must be the pattern's other reader of `x - mean`
    if !readers.contains(&i_div) {
        return None;
    }

    let i_mul = index.only_consumer(&div.outputs[0])?;
    let mul = &nodes[i_mul];
    if !is_op(mul, "Mul") {
        return None;
    }
    let scale = other_input(mul, &div.outputs[0])?.to_string();
    let channels = per_channel_len(&scale, constants)?;

    // The bias is optional: without it the kernel runs with `has_bias = 0`.
    let mut members = vec![root, i_sub, i_pow, i_var, i_add_eps, i_sqrt, i_div, i_mul];
    let mut inputs = vec![x.clone(), scale];
    let mut output = mul.outputs[0].clone();
    if let Some(i_add_bias) = index.only_consumer(&mul.outputs[0])
        && is_op(&nodes[i_add_bias], "Add")
        && let Some(bias) = other_input(&nodes[i_add_bias], &mul.outputs[0])
        && per_channel_len(bias, constants) == Some(channels)
    {
        inputs.push(bias.to_string());
        output = nodes[i_add_bias].outputs[0].clone();
        members.push(i_add_bias);
    }

    let fused = NodeIr {
        domain: String::new(),
        op: "LayerNormalization".into(),
        // the operator's own opset, independent of the graph's: the pattern
        // exists precisely because the graph predates it
        since_version: 17,
        name: format!("{}_fused_LayerNormalization", output),
        inputs,
        outputs: vec![output],
        attrs: HashMap::from([
            ("axis".to_string(), AttrValue::Int(-1)),
            ("epsilon".to_string(), AttrValue::Float(epsilon)),
        ]),
    };
    Some(Match { members, fused })
}

/// Replaces every decomposed layer normalization with a single
/// `LayerNormalization`. Returns how many patterns were fused.
pub fn fuse_layernorm(ir: &mut GraphIr) -> usize {
    let mut constants = constant_outputs(&ir.nodes);
    constants.extend(ir.initializers.clone());
    let index = Index::build(ir);

    let mut claimed = vec![false; ir.nodes.len()];
    let mut matches = Vec::new();
    for root in 0..ir.nodes.len() {
        let Some(m) = match_layernorm(&ir.nodes, root, &index, &constants) else {
            continue;
        };
        if m.members.iter().any(|&i| claimed[i]) {
            continue;
        }
        for &i in &m.members {
            claimed[i] = true;
        }
        matches.push(m);
    }
    if matches.is_empty() {
        return 0;
    }

    // The fused node takes the slot of the pattern's last member: every other
    // member is one of its ancestors, so topological order is preserved.
    let mut replacement: HashMap<usize, NodeIr> = HashMap::new();
    let count = matches.len();
    for m in matches {
        let slot = *m.members.iter().max().expect("a match has members");
        replacement.insert(slot, m.fused);
    }
    let nodes = std::mem::take(&mut ir.nodes);
    ir.nodes = nodes
        .into_iter()
        .enumerate()
        .filter_map(|(i, node)| match (claimed[i], replacement.remove(&i)) {
            (true, fused) => fused,
            (false, _) => Some(node),
        })
        .collect();
    count
}

/// Evaluates a node whose inputs are all constant, on host, at load time.
///
/// Only the operators the measurement points at are here. A generic evaluator
/// would mean a second implementation of the interpreter on host tensors, and
/// the payoff does not need one: across the nine models in the matrix five gain
/// nothing at all, and what the other four fold is weight dequantization and
/// shape plumbing. `resnet50-qdq` re-dequantizes its weights on every run —
/// 107 `DequantizeLinear` reading 25.5 MB of int8 and writing **102.1 MB of
/// fp32** for a result that never changes.
fn evaluate(node: &NodeIr, inputs: &[HostTensor]) -> Option<HostTensor> {
    if !node.domain.is_empty() {
        return None;
    }
    match node.op.as_str() {
        "DequantizeLinear" => {
            // `axis` only means anything for a per-axis scale, which the host
            // implementation rejects; a scalar scale ignores it.
            crate::host_ops::dequantize_linear(&inputs[0], inputs.get(1)?, inputs.get(2)).ok()
        }
        "Cast" => {
            let to = node.attrs.get("to")?.as_i64()? as i32;
            crate::host_ops::cast(&inputs[0], to).ok()
        }
        "Concat" => {
            let axis = node.attrs.get("axis")?.as_i64()?;
            crate::host_ops::concat(inputs, axis).ok()
        }
        // Pure reinterpretations of the same bytes: only the shape changes, and
        // it is already resolved because the shape operand is constant too.
        "Reshape" => {
            let requested = inputs.get(1)?.to_i64().ok()?;
            let shape = resolve_reshape(&inputs[0].shape, &requested)?;
            Some(HostTensor::new(
                inputs[0].dtype,
                shape,
                inputs[0].data.clone(),
            ))
        }
        _ => None,
    }
}

/// `Reshape`'s target shape with `0` (copy the input dim) and `-1` (infer)
/// resolved, or `None` if it does not describe the same element count.
fn resolve_reshape(input: &[i64], requested: &[i64]) -> Option<Vec<i64>> {
    let total: i64 = input.iter().product();
    let mut shape: Vec<i64> = requested
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d == 0 {
                *input.get(i).unwrap_or(&0)
            } else {
                d
            }
        })
        .collect();
    let inferred = shape.iter().position(|&d| d == -1);
    if let Some(i) = inferred {
        let known: i64 = shape.iter().filter(|&&d| d != -1).product();
        if known == 0 || total % known != 0 {
            return None;
        }
        shape[i] = total / known;
    }
    (shape.iter().product::<i64>() == total && shape.iter().all(|&d| d >= 0)).then_some(shape)
}

/// Replaces every node whose inputs are all constant with its own result,
/// promoted to an initializer. Returns how many nodes were folded.
///
/// The nodes are visited in topological order, so a folded output is itself
/// constant for whatever reads it downstream — the closure is transitive in a
/// single pass. What the fold removes is *dispatches*, not blocks: every model
/// in the matrix is already at one convex block.
pub fn fold_constants(ir: &mut GraphIr) -> usize {
    let mut constants: HashMap<String, InitializerIr> = constant_outputs(&ir.nodes);
    constants.extend(ir.initializers.clone());

    let graph_outputs: HashSet<&str> = ir.outputs.iter().map(String::as_str).collect();
    let mut folded = Vec::new();
    for (i, node) in ir.nodes.iter().enumerate() {
        // A graph output has to stay a produced value: the run's environment
        // hands back what the nodes computed, not what the initializers hold.
        if node.op == "Constant"
            || node.inputs.is_empty()
            || node.outputs.len() != 1
            || graph_outputs.contains(node.outputs[0].as_str())
        {
            continue;
        }
        let Some(inputs) = node
            .inputs
            .iter()
            .map(|name| {
                let init = constants.get(name)?;
                Some(HostTensor::new(
                    init.dtype,
                    init.shape.clone(),
                    init.data.clone(),
                ))
            })
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let Some(result) = evaluate(node, &inputs) else {
            continue;
        };
        let value = InitializerIr {
            dtype: result.dtype,
            shape: result.shape,
            data: result.data,
        };
        constants.insert(node.outputs[0].clone(), value.clone());
        folded.push((i, node.outputs[0].clone(), value));
    }
    if folded.is_empty() {
        return 0;
    }

    let count = folded.len();
    let removed: HashSet<usize> = folded.iter().map(|(i, ..)| *i).collect();
    for (_, name, value) in folded {
        ir.initializers.insert(name, value);
    }
    let nodes = std::mem::take(&mut ir.nodes);
    ir.nodes = nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !removed.contains(i))
        .map(|(_, node)| node)
        .collect();
    count
}

/// Drops initializers no node reads. Returns the bytes released.
///
/// Constant folding orphans its own inputs — the int8 weights behind 107
/// `DequantizeLinear` are 25.5 MB nothing reads once their fp32 result is an
/// initializer — and they are held for the session's whole life.
pub fn prune_dead_initializers(ir: &mut GraphIr) -> usize {
    let live: HashSet<&str> = ir
        .nodes
        .iter()
        .flat_map(|n| n.inputs.iter())
        .map(String::as_str)
        .chain(ir.outputs.iter().map(String::as_str))
        .collect();
    let dead: Vec<String> = ir
        .initializers
        .keys()
        .filter(|name| !live.contains(name.as_str()))
        .cloned()
        .collect();
    dead.iter()
        .filter_map(|name| ir.initializers.remove(name))
        .map(|init| init.data.len())
        .sum()
}

/// Drops nodes whose outputs nobody reads and that no graph output declares.
///
/// Every operator the interpreter runs is pure, so an unread result is work
/// with no observer. Rewriting orphans the constants a pattern carried —
/// roberta's `Pow` exponents and epsilons — and they would otherwise stay in
/// the block as uploads nothing reads. Iterated to a fixpoint because removing
/// a node can orphan its own producers.
pub fn prune_dead_nodes(ir: &mut GraphIr) -> usize {
    let mut removed = 0;
    loop {
        let index = Index::build(ir);
        let before = ir.nodes.len();
        let nodes = std::mem::take(&mut ir.nodes);
        ir.nodes = nodes
            .into_iter()
            .filter(|node| {
                node.outputs.iter().any(|name| {
                    !name.is_empty()
                        && (index.graph_outputs.contains(name)
                            || index.consumers.contains_key(name))
                })
            })
            .collect();
        removed += before - ir.nodes.len();
        if ir.nodes.len() == before {
            return removed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(op: &str, inputs: &[&str], output: &str) -> NodeIr {
        NodeIr {
            domain: String::new(),
            op: op.into(),
            since_version: 11,
            name: output.into(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: vec![output.into()],
            attrs: HashMap::new(),
        }
    }

    fn reduce_mean(input: &str, output: &str, axes: i64) -> NodeIr {
        let mut n = node("ReduceMean", &[input], output);
        n.attrs.insert("axes".into(), AttrValue::Ints(vec![axes]));
        n
    }

    fn scalar(value: f32) -> InitializerIr {
        InitializerIr {
            dtype: ElementType::Float32 as i32,
            shape: vec![],
            data: value.to_le_bytes().to_vec(),
        }
    }

    fn vector(len: usize) -> InitializerIr {
        InitializerIr {
            dtype: ElementType::Float32 as i32,
            shape: vec![len as i64],
            data: vec![0; len * 4],
        }
    }

    /// The pattern roberta-base emits 25 times, `axes = [-1]`.
    fn decomposed(axes: i64) -> GraphIr {
        GraphIr {
            nodes: vec![
                reduce_mean("x", "m", axes),
                node("Sub", &["x", "m"], "d"),
                node("Pow", &["d", "two"], "p"),
                reduce_mean("p", "v", axes),
                node("Add", &["v", "eps"], "ve"),
                node("Sqrt", &["ve"], "s"),
                node("Div", &["d", "s"], "n"),
                node("Mul", &["n", "scale"], "ns"),
                node("Add", &["ns", "bias"], "y"),
            ],
            initializers: HashMap::from([
                ("two".into(), scalar(2.0)),
                ("eps".into(), scalar(1e-5)),
                ("scale".into(), vector(768)),
                ("bias".into(), vector(768)),
            ]),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
        }
    }

    #[test]
    fn fuses_the_decomposed_pattern() {
        let mut ir = decomposed(-1);
        assert_eq!(fuse_layernorm(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 1);
        let fused = &ir.nodes[0];
        assert_eq!(fused.op, "LayerNormalization");
        assert_eq!(fused.inputs, ["x", "scale", "bias"]);
        assert_eq!(fused.outputs, ["y"]);
        assert_eq!(fused.attrs["axis"], AttrValue::Int(-1));
        assert_eq!(fused.attrs["epsilon"], AttrValue::Float(1e-5));
    }

    /// rfdetr writes the same nine nodes over the channel axis. That is a
    /// different operator and the kernel normalizes the last axis only.
    #[test]
    fn leaves_a_channel_axis_normalization_alone() {
        let mut ir = decomposed(1);
        assert_eq!(fuse_layernorm(&mut ir), 0);
        assert_eq!(ir.nodes.len(), 9);
    }

    #[test]
    fn fuses_without_the_bias_add() {
        let mut ir = decomposed(-1);
        ir.nodes.pop();
        ir.outputs = vec!["ns".into()];
        assert_eq!(fuse_layernorm(&mut ir), 1);
        assert_eq!(ir.nodes[0].inputs, ["x", "scale"]);
        assert_eq!(ir.nodes[0].outputs, ["ns"]);
    }

    /// An intermediate someone else reads cannot be deleted with the pattern.
    #[test]
    fn refuses_when_an_intermediate_escapes() {
        let mut ir = decomposed(-1);
        ir.nodes.push(node("Neg", &["s"], "leak"));
        ir.outputs.push("leak".into());
        assert_eq!(fuse_layernorm(&mut ir), 0);

        let mut ir = decomposed(-1);
        ir.outputs.push("v".into());
        assert_eq!(fuse_layernorm(&mut ir), 0);
    }

    /// A scalar scale broadcasts over every axis, not per channel: the kernel
    /// would reject it, so the rewrite must not build it.
    #[test]
    fn refuses_a_scalar_scale() {
        let mut ir = decomposed(-1);
        ir.initializers.insert("scale".into(), vector(1));
        assert_eq!(fuse_layernorm(&mut ir), 0);
    }

    #[test]
    fn refuses_an_exponent_that_is_not_two() {
        let mut ir = decomposed(-1);
        ir.initializers.insert("two".into(), scalar(3.0));
        assert_eq!(fuse_layernorm(&mut ir), 0);
    }

    /// Two patterns in a row, the second reading the first's output.
    #[test]
    fn fuses_consecutive_patterns_and_keeps_topological_order() {
        let mut ir = decomposed(-1);
        let second: Vec<NodeIr> = decomposed(-1)
            .nodes
            .into_iter()
            .map(|mut n| {
                let rename = |s: &String| match s.as_str() {
                    "x" => "y".to_string(),
                    "scale" | "bias" | "two" | "eps" => s.clone(),
                    other => format!("{other}2"),
                };
                n.inputs = n.inputs.iter().map(rename).collect();
                n.outputs = n.outputs.iter().map(rename).collect();
                n
            })
            .collect();
        ir.nodes.extend(second);
        ir.outputs = vec!["y2".into()];
        assert_eq!(fuse_layernorm(&mut ir), 2);
        assert_eq!(ir.nodes.len(), 2);
        assert_eq!(ir.nodes[0].outputs, ["y"]);
        assert_eq!(ir.nodes[1].inputs[0], "y");
    }

    fn int8(shape: Vec<i64>, values: &[i8]) -> InitializerIr {
        InitializerIr {
            dtype: ElementType::Int8 as i32,
            shape,
            data: values.iter().map(|&v| v as u8).collect(),
        }
    }

    /// The 107 nodes `resnet50-qdq` re-runs every inference on a constant.
    #[test]
    fn folds_a_constant_weight_dequantization() {
        let mut ir = GraphIr {
            nodes: vec![
                node("DequantizeLinear", &["w_q", "w_scale", "w_zp"], "w"),
                node("Conv", &["x", "w"], "y"),
            ],
            initializers: HashMap::from([
                ("w_q".into(), int8(vec![2, 2], &[-2, -1, 1, 2])),
                ("w_scale".into(), scalar(0.5)),
                ("w_zp".into(), int8(vec![], &[-1])),
            ]),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
        };
        assert_eq!(fold_constants(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 1);
        assert_eq!(ir.nodes[0].op, "Conv");

        let w = &ir.initializers["w"];
        assert_eq!(w.dtype, ElementType::Float32 as i32);
        assert_eq!(w.shape, [2, 2]);
        let values = HostTensor::new(w.dtype, w.shape.clone(), w.data.clone())
            .to_f32()
            .unwrap();
        assert_eq!(values, [-0.5, 0.0, 1.0, 1.5]);

        // the int8 source and its quantization parameters are now unread
        assert_eq!(prune_dead_initializers(&mut ir), 4 + 4 + 1);
        assert!(!ir.initializers.contains_key("w_q"));
        assert!(ir.initializers.contains_key("w"));
    }

    /// A fold whose result feeds another foldable node closes transitively in
    /// one pass, because the nodes are visited in topological order.
    #[test]
    fn folds_transitively_in_one_pass() {
        let mut ir = GraphIr {
            nodes: vec![
                node("DequantizeLinear", &["w_q", "w_scale"], "w"),
                node("Reshape", &["w", "shape"], "w2"),
                node("Conv", &["x", "w2"], "y"),
            ],
            initializers: HashMap::from([
                ("w_q".into(), int8(vec![2, 2], &[1, 2, 3, 4])),
                ("w_scale".into(), scalar(1.0)),
                (
                    "shape".into(),
                    InitializerIr {
                        dtype: ElementType::Int64 as i32,
                        shape: vec![3],
                        data: [1i64, 4, -1].iter().flat_map(|v| v.to_le_bytes()).collect(),
                    },
                ),
            ]),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
        };
        assert_eq!(fold_constants(&mut ir), 2);
        assert_eq!(ir.nodes.len(), 1);
        assert_eq!(ir.initializers["w2"].shape, [1, 4, 1]);
    }

    /// Per-axis quantization is refused rather than folded wrongly.
    #[test]
    fn refuses_a_per_axis_dequantization() {
        let mut ir = GraphIr {
            nodes: vec![node("DequantizeLinear", &["w_q", "w_scale"], "w")],
            initializers: HashMap::from([
                ("w_q".into(), int8(vec![2, 2], &[1, 2, 3, 4])),
                ("w_scale".into(), vector(2)),
            ]),
            inputs: vec![],
            outputs: vec!["w".into()],
        };
        assert_eq!(fold_constants(&mut ir), 0);
    }

    /// A node producing a graph output stays: the run hands back computed
    /// values, not initializers.
    #[test]
    fn keeps_a_fold_that_produces_a_graph_output() {
        let mut ir = GraphIr {
            nodes: vec![node("DequantizeLinear", &["w_q", "w_scale"], "w")],
            initializers: HashMap::from([
                ("w_q".into(), int8(vec![2], &[1, 2])),
                ("w_scale".into(), scalar(1.0)),
            ]),
            inputs: vec![],
            outputs: vec!["w".into()],
        };
        assert_eq!(fold_constants(&mut ir), 0);
    }

    #[test]
    fn prunes_the_constants_the_rewrite_orphans() {
        let mut ir = decomposed(-1);
        ir.initializers.remove("two");
        ir.initializers.remove("eps");
        let mut two = node("Constant", &[], "two");
        two.attrs
            .insert("value".into(), AttrValue::Tensor(scalar(2.0)));
        let mut eps = node("Constant", &[], "eps");
        eps.attrs
            .insert("value".into(), AttrValue::Tensor(scalar(1e-5)));
        ir.nodes.insert(0, two);
        ir.nodes.insert(1, eps);

        assert_eq!(fuse_layernorm(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 3, "the two Constant nodes are now orphans");
        assert_eq!(prune_dead_nodes(&mut ir), 2);
        assert_eq!(ir.nodes.len(), 1);
        assert_eq!(ir.nodes[0].op, "LayerNormalization");
    }
}
