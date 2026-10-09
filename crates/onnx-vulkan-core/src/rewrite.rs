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
    producers: HashMap<String, usize>,
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
            producers: producers(ir),
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
        opset: 17,
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
    replace_matches(ir, |nodes, root, index| {
        match_layernorm(nodes, root, index, &constants)
    })
}

/// Domain of the operators that exist only after a rewrite. No exporter writes
/// it, so a node carrying it was produced here.
pub const REWRITE_DOMAIN: &str = "onnx_vulkan.rewrite";

/// Matches `MatMul(a, b) → Softmax(last axis) → MatMul(p, v)` rooted at the
/// first `MatMul`, when neither the scores nor the probabilities escape.
///
/// The Softmax axis must be the last one by attribute: `-1`, or absent from
/// opset 13 on, where it is the default. Below 13 the default is `1` with the
/// input coerced to 2D, which is the last axis only at rank 2, and the IR has
/// no ranks to tell.
///
/// The head transposes around it are folded in when only the pattern reads
/// them, as multi-head attention exports write them: `v` made `[B, H, n, DV]`
/// by `Transpose(·, [0, 2, 1, 3])` from `[B, n, H, DV]` (attribute `v_bshd`),
/// and the output made `[B, m, H, DV]` by the same perm (`out_bshd`). Both are
/// pure data movement, so the kernel only reads and writes elsewhere.
///
/// Two more steps are folded in when only the pattern reads them:
///
/// - an `Add(scores, bias)` before the Softmax, a position bias or an
///   additive mask: `bias` becomes the fourth input, broadcast as the `Add`
///   broadcasts it;
/// - `Where(IsNaN(p), 0, p)` after it, the cleanup exporters write for rows
///   whose every key is masked to `-inf` (attribute `nan_to_zero`). A
///   probability is NaN only when its whole row is, so the kernel writes a
///   zero output row wherever the softmax denominator is not positive.
fn match_attention(
    nodes: &[NodeIr],
    root: usize,
    index: &Index,
    producers: &HashMap<String, usize>,
    constants: &HashMap<String, InitializerIr>,
) -> Option<Match> {
    let qk = &nodes[root];
    if !is_op(qk, "MatMul") || qk.inputs.len() != 2 {
        return None;
    }
    let mut members = vec![root];
    let mut s = index.only_consumer(qk.outputs.first()?)?;
    let mut bias = None;
    if is_op(&nodes[s], "Add") {
        let add = &nodes[s];
        let other = other_input(add, &qk.outputs[0])?;
        if other == qk.outputs[0] {
            return None;
        }
        bias = Some(other.to_owned());
        members.push(s);
        s = index.only_consumer(add.outputs.first()?)?;
    }
    let softmax = &nodes[s];
    let axis = softmax.attrs.get("axis").and_then(AttrValue::as_i64);
    let last_axis = axis == Some(-1) || (axis.is_none() && softmax.opset >= 13);
    if !is_op(softmax, "Softmax") || !last_axis {
        return None;
    }
    members.push(s);
    let probs = softmax.outputs.first()?;
    let mut attrs = HashMap::new();
    let (probs, pv_index) = match nan_cleanup(nodes, probs, index, constants) {
        Some((is_nan, cleaned)) => {
            members.extend([is_nan, cleaned]);
            attrs.insert("nan_to_zero".to_owned(), AttrValue::Int(1));
            let cleaned = &nodes[cleaned].outputs[0];
            (cleaned, index.only_consumer(cleaned)?)
        }
        None => (probs, index.only_consumer(probs)?),
    };
    let pv = &nodes[pv_index];
    // the probabilities must be the left operand: `v · p` is another product
    if !is_op(pv, "MatMul") || pv.inputs.len() != 2 || pv.inputs[0] != *probs {
        return None;
    }
    members.push(pv_index);
    let heads_swap = |node: &NodeIr| {
        is_op(node, "Transpose")
            && node.inputs.len() == 1
            && node.attrs.get("perm").and_then(AttrValue::as_ints) == Some(&[0, 2, 1, 3][..])
    };

    // set only when a step is folded: a plain attention stays the node it
    // always was, and so does the identity of plans built on it
    let mut v = pv.inputs[1].clone();
    if let Some(&t) = producers.get(&v)
        && heads_swap(&nodes[t])
        && index.only_consumer(&v) == Some(pv_index)
    {
        v = nodes[t].inputs[0].clone();
        attrs.insert("v_bshd".to_owned(), AttrValue::Int(1));
        members.push(t);
    }
    let mut outputs = pv.outputs.clone();
    if let Some(t) = index.only_consumer(pv.outputs.first()?)
        && heads_swap(&nodes[t])
    {
        outputs = nodes[t].outputs.clone();
        attrs.insert("out_bshd".to_owned(), AttrValue::Int(1));
        members.push(t);
    }
    Some(Match {
        members,
        fused: NodeIr {
            domain: REWRITE_DOMAIN.to_owned(),
            op: "FlashAttention".to_owned(),
            // the model's `ai.onnx` opset, which the fallback's Softmax needs
            opset: softmax.opset,
            name: pv.name.clone(),
            inputs: [qk.inputs[0].clone(), qk.inputs[1].clone(), v]
                .into_iter()
                .chain(bias)
                .collect(),
            outputs,
            attrs,
        },
    })
}

/// `(IsNaN, Where)` when the only readers of `probs` are `IsNaN(probs)` and
/// `Where(that, 0, probs)`, and `IsNaN`'s only reader is the `Where`.
fn nan_cleanup(
    nodes: &[NodeIr],
    probs: &str,
    index: &Index,
    constants: &HashMap<String, InitializerIr>,
) -> Option<(usize, usize)> {
    if index.graph_outputs.contains(probs) {
        return None;
    }
    let [x, y] = index.consumers.get(probs)?.as_slice() else {
        return None;
    };
    let (is_nan, cleaned) = if is_op(&nodes[*x], "IsNaN") {
        (*x, *y)
    } else {
        (*y, *x)
    };
    let (test, select) = (&nodes[is_nan], &nodes[cleaned]);
    let zero = |name: &str| scalar_f32(name, constants).map(f32::to_bits) == Some(0);
    let matches = is_op(test, "IsNaN")
        && test.inputs == [probs]
        && is_op(select, "Where")
        && select.inputs.len() == 3
        && select.inputs[0] == test.outputs[0]
        && zero(&select.inputs[1])
        && select.inputs[2] == probs
        && index.only_consumer(&test.outputs[0]) == Some(cleaned);
    matches.then_some((is_nan, cleaned))
}

/// Replaces every `MatMul → Softmax → MatMul` attention with one
/// `FlashAttention` that never materializes the scores. Returns how many
/// patterns were fused.
pub fn fuse_attention(ir: &mut GraphIr) -> usize {
    let producers = producers(ir);
    // only scalars are looked up: no copy of the weights
    let mut constants = constant_outputs(&ir.nodes);
    constants.extend(
        ir.initializers
            .iter()
            .filter(|(_, tensor)| tensor.data.len() == 4)
            .map(|(name, tensor)| (name.clone(), tensor.clone())),
    );
    replace_matches(ir, |nodes, root, index| {
        match_attention(nodes, root, index, &producers, &constants)
    })
}

/// The index of the node producing each value.
fn producers(ir: &GraphIr) -> HashMap<String, usize> {
    ir.nodes
        .iter()
        .enumerate()
        .flat_map(|(i, node)| node.outputs.iter().map(move |out| (out.clone(), i)))
        .collect()
}

/// Matches `MatMulInteger → Cast(to = FLOAT) → Mul(·, scale) → Add(·, bias)`
/// rooted at the `MatMulInteger`, when neither intermediate escapes.
///
/// This is how dynamic int8 quantization writes every linear layer: the i32
/// product, dequantized by the product of the two scales, plus the float bias.
/// The fused node keeps the four original operands of `MatMulInteger` and
/// appends `scale` and `bias`; both multiplications and the addition are
/// commutative in IEEE arithmetic, so their operand order is not recorded.
///
/// Shapes are unknown here. Whether `scale` and `bias` broadcast along the
/// columns only is checked at run time, where anything else runs the four
/// original steps.
///
/// `ConvInteger` is dequantized the same way, with the bias one value per
/// output channel; `op` is the integer node, `fused_op` the replacement, which
/// keeps the root's attributes (the convolution geometry).
///
/// A `MatMulInteger` whose `Add` feeds only a `Relu` takes it too, recorded as
/// the attribute `relu = 1`: SAM 3's detector has 57 such feed-forward layers,
/// and each separate `Relu` read and wrote the whole activation again.
///
/// A fused `MatMulInteger` whose result a `DynamicQuantizeLinear` reads gets
/// `range = 1`: the kernel can fold the range into the quantization's slot as
/// it stores the values, sparing the quantization a pass over them.
///
/// When `scale` is itself a `Mul` read by nothing else — the activation scale
/// times the weight scale, one tiny dispatch ahead of every linear layer — that
/// `Mul` goes too: its two operands become inputs 4 and 6, in their order, and
/// the kernel forms the product. A path that cannot runs the `Mul` first.
fn match_integer_epilogue(
    nodes: &[NodeIr],
    root: usize,
    index: &Index,
    op: &str,
    fused_op: &str,
) -> Option<Match> {
    let mmi = &nodes[root];
    if !is_op(mmi, op) || mmi.inputs.len() != 4 || mmi.inputs.iter().any(String::is_empty) {
        return None;
    }
    let c = index.only_consumer(mmi.outputs.first()?)?;
    let cast = &nodes[c];
    let to = cast.attrs.get("to").and_then(AttrValue::as_i64);
    if !is_op(cast, "Cast") || to != Some(ElementType::Float32 as i64) {
        return None;
    }
    let m = index.only_consumer(cast.outputs.first()?)?;
    let mul = &nodes[m];
    let scale = other_input(mul, &cast.outputs[0])?;
    // `Mul(x, x)` names the removed value twice
    if !is_op(mul, "Mul") || scale == cast.outputs[0] {
        return None;
    }
    let a = index.only_consumer(mul.outputs.first()?)?;
    let add = &nodes[a];
    let bias = other_input(add, &mul.outputs[0])?;
    if !is_op(add, "Add") || bias == mul.outputs[0] {
        return None;
    }
    let mut inputs = mmi.inputs.clone();
    inputs.extend([scale.to_owned(), bias.to_owned()]);
    let mut members = vec![root, c, m, a];
    if let Some(&p) = index.producers.get(scale)
        && let product = &nodes[p]
        && is_op(product, "Mul")
        && product.inputs.len() == 2
        && !product.inputs.iter().any(String::is_empty)
        && index.only_consumer(scale) == Some(m)
    {
        inputs[4] = product.inputs[0].clone();
        inputs.push(product.inputs[1].clone());
        members.push(p);
    }
    let mut attrs = mmi.attrs.clone();
    let mut last = add;
    if op == "MatMulInteger"
        && let Some(r) = index.only_consumer(add.outputs.first()?)
        && is_op(&nodes[r], "Relu")
    {
        members.push(r);
        attrs.insert("relu".to_owned(), AttrValue::Int(1));
        last = &nodes[r];
    }
    let out = last.outputs.first()?;
    let quantized = index.consumers.get(out).is_some_and(|readers| {
        readers.iter().any(|&q| {
            is_op(&nodes[q], "DynamicQuantizeLinear") && nodes[q].inputs.first() == Some(out)
        })
    });
    if op == "MatMulInteger" && quantized {
        attrs.insert("range".to_owned(), AttrValue::Int(1));
    }
    Some(Match {
        members,
        fused: NodeIr {
            domain: REWRITE_DOMAIN.to_owned(),
            op: fused_op.to_owned(),
            opset: add.opset,
            name: last.name.clone(),
            inputs,
            outputs: last.outputs.clone(),
            attrs,
        },
    })
}

/// Replaces every dequantizing `MatMulInteger → Cast → Mul → Add` with one
/// `MatMulIntegerScaleBias`. Returns how many patterns were fused.
pub fn fuse_matmul_integer_epilogue(ir: &mut GraphIr) -> usize {
    replace_matches(ir, |nodes, root, index| {
        match_integer_epilogue(
            nodes,
            root,
            index,
            "MatMulInteger",
            "MatMulIntegerScaleBias",
        )
    })
}

/// Replaces every dequantizing `ConvInteger → Cast → Mul → Add` with one
/// `ConvIntegerScaleBias`. Returns how many patterns were fused.
pub fn fuse_conv_integer_epilogue(ir: &mut GraphIr) -> usize {
    replace_matches(ir, |nodes, root, index| {
        match_integer_epilogue(nodes, root, index, "ConvInteger", "ConvIntegerScaleBias")
    })
}

/// Matches `Mul(x, scale) → Add(·, bias) → Relu` rooted at the `Mul`, with
/// `scale` and `bias` constants and neither intermediate escaping: the affine
/// a convolutional network applies after a normalization whose own affine is
/// the identity. SAM 3's pixel decoder writes `GroupNorm` that way, as
/// `InstanceNormalization` over the groups followed by a scale and a shift per
/// channel. The fused node keeps `x, scale, bias`; IEEE multiplication and
/// addition are commutative, so the operand order is not recorded.
///
/// Shapes are unknown here. Whether `scale` and `bias` broadcast per tensor or
/// per channel is checked at run time, where anything else runs the three
/// original steps.
fn match_mul_add_relu(
    nodes: &[NodeIr],
    root: usize,
    index: &Index,
    constants: &HashSet<String>,
) -> Option<Match> {
    let mul = &nodes[root];
    if !is_op(mul, "Mul") || mul.inputs.len() != 2 {
        return None;
    }
    let (x, scale) = match (
        constants.contains(&mul.inputs[0]),
        constants.contains(&mul.inputs[1]),
    ) {
        (false, true) => (&mul.inputs[0], &mul.inputs[1]),
        (true, false) => (&mul.inputs[1], &mul.inputs[0]),
        _ => return None,
    };
    let a = index.only_consumer(mul.outputs.first()?)?;
    let add = &nodes[a];
    let bias = other_input(add, &mul.outputs[0])?;
    if !is_op(add, "Add") || !constants.contains(bias) {
        return None;
    }
    let r = index.only_consumer(add.outputs.first()?)?;
    let relu = &nodes[r];
    if !is_op(relu, "Relu") {
        return None;
    }
    Some(Match {
        members: vec![root, a, r],
        fused: NodeIr {
            domain: REWRITE_DOMAIN.to_owned(),
            op: "MulAddRelu".to_owned(),
            opset: relu.opset,
            name: relu.name.clone(),
            inputs: vec![x.clone(), scale.clone(), bias.to_owned()],
            outputs: relu.outputs.clone(),
            attrs: HashMap::new(),
        },
    })
}

/// Replaces every `Mul → Add → Relu` by constants with one `MulAddRelu`.
/// Returns how many patterns were fused.
pub fn fuse_mul_add_relu(ir: &mut GraphIr) -> usize {
    let constants: HashSet<String> = constant_outputs(&ir.nodes)
        .into_keys()
        .chain(ir.initializers.keys().cloned())
        .collect();
    replace_matches(ir, |nodes, root, index| {
        match_mul_add_relu(nodes, root, index, &constants)
    })
}

/// Matches the exact GELU as exporters write it below opset 20,
/// `Div(x, c1) → Erf → Add(·, c2) → Mul(x, ·) → Mul(·, c3)`, or with the two
/// products the other way round, `… → Mul(·, c3) → Mul(x, ·)` (the SAM 3
/// vision encoder), rooted at the `Div`, with `c1`, `c2`, `c3` scalar f32
/// constants and no intermediate escaping. `x` itself may have other readers.
///
/// The fused node is not the standard `Gelu`: that one multiplies by `1/√2`
/// where this divides by `c1`, and rounds in another order. It keeps the
/// constants as inputs and replays the five steps in their own order, so the
/// result does not change by a bit; `factor_first = 1` records the second
/// order.
fn match_decomposed_gelu(
    nodes: &[NodeIr],
    root: usize,
    index: &Index,
    constants: &HashMap<String, InitializerIr>,
) -> Option<Match> {
    let div = &nodes[root];
    if !is_op(div, "Div") || div.inputs.len() != 2 {
        return None;
    }
    let (x, c1) = (&div.inputs[0], &div.inputs[1]);
    scalar_f32(c1, constants)?;
    let e = index.only_consumer(div.outputs.first()?)?;
    let erf = &nodes[e];
    if !is_op(erf, "Erf") {
        return None;
    }
    let a = index.only_consumer(erf.outputs.first()?)?;
    let add = &nodes[a];
    let c2 = other_input(add, &erf.outputs[0])?;
    if !is_op(add, "Add") {
        return None;
    }
    scalar_f32(c2, constants)?;
    let g = index.only_consumer(add.outputs.first()?)?;
    let first = &nodes[g];
    let first_other = other_input(first, &add.outputs[0])?;
    if !is_op(first, "Mul") {
        return None;
    }
    let h = index.only_consumer(first.outputs.first()?)?;
    let last = &nodes[h];
    let last_other = other_input(last, &first.outputs[0])?;
    if !is_op(last, "Mul") {
        return None;
    }
    let (c3, factor_first) = if first_other == x {
        (last_other, false)
    } else if last_other == x {
        (first_other, true)
    } else {
        return None;
    };
    scalar_f32(c3, constants)?;
    let mut attrs = HashMap::new();
    if factor_first {
        attrs.insert("factor_first".to_owned(), AttrValue::Int(1));
    }
    Some(Match {
        members: vec![root, e, a, g, h],
        fused: NodeIr {
            domain: REWRITE_DOMAIN.to_owned(),
            op: "DecomposedGelu".to_owned(),
            opset: last.opset,
            name: last.name.clone(),
            inputs: vec![x.clone(), c1.clone(), c2.to_owned(), c3.to_owned()],
            outputs: last.outputs.clone(),
            attrs,
        },
    })
}

/// Replaces every decomposed exact GELU with one `DecomposedGelu`. Returns how
/// many patterns were fused.
pub fn fuse_decomposed_gelu(ir: &mut GraphIr) -> usize {
    // only scalars are looked up: no copy of the weights
    let mut constants = constant_outputs(&ir.nodes);
    constants.extend(
        ir.initializers
            .iter()
            .filter(|(_, tensor)| tensor.data.len() == 4)
            .map(|(name, tensor)| (name.clone(), tensor.clone())),
    );
    replace_matches(ir, |nodes, root, index| {
        match_decomposed_gelu(nodes, root, index, &constants)
    })
}

/// Matches a `DecomposedGelu` whose only reader is a `DynamicQuantizeLinear`,
/// rooted at the GELU. The fused node keeps the GELU's inputs and attributes
/// and the quantization's outputs: the f32 GELU, the largest tensor of a
/// transformer MLP, is never stored (`interp::decomposed_gelu_quantize`).
fn match_gelu_quantize(nodes: &[NodeIr], root: usize, index: &Index) -> Option<Match> {
    let gelu = &nodes[root];
    if gelu.domain != REWRITE_DOMAIN || gelu.op != "DecomposedGelu" {
        return None;
    }
    let q = index.only_consumer(gelu.outputs.first()?)?;
    let quantize = &nodes[q];
    if !is_op(quantize, "DynamicQuantizeLinear") || quantize.inputs.len() != 1 {
        return None;
    }
    Some(Match {
        members: vec![root, q],
        fused: NodeIr {
            domain: REWRITE_DOMAIN.to_owned(),
            op: "DecomposedGeluQuantize".to_owned(),
            opset: quantize.opset,
            name: quantize.name.clone(),
            inputs: gelu.inputs.clone(),
            outputs: quantize.outputs.clone(),
            attrs: gelu.attrs.clone(),
        },
    })
}

/// Replaces every `DecomposedGelu → DynamicQuantizeLinear` with one
/// `DecomposedGeluQuantize`; runs after [`fuse_decomposed_gelu`]. Returns how
/// many patterns were fused.
pub fn fuse_gelu_quantize(ir: &mut GraphIr) -> usize {
    replace_matches(ir, match_gelu_quantize)
}

/// The values of an integer constant, if `name` is one.
fn ints(name: &str, constants: &HashMap<String, InitializerIr>) -> Option<Vec<i64>> {
    let tensor = constants.get(name)?;
    if tensor.dtype != ElementType::Int64 as i32 && tensor.dtype != ElementType::Int32 as i32 {
        return None;
    }
    HostTensor::new(tensor.dtype, tensor.shape.clone(), tensor.data.clone())
        .to_i64()
        .ok()
}

/// A last-axis attribute or operand of a rank-5 value: `-1`, or `4`.
fn last_of_rank5(axis: Option<i64>) -> bool {
    matches!(axis, Some(-1 | 4))
}

/// The single node producing `name`, if `op` is its type.
fn producer<'a>(
    nodes: &'a [NodeIr],
    producers: &HashMap<String, usize>,
    name: &str,
    op: &str,
) -> Option<(usize, &'a NodeIr)> {
    let i = *producers.get(name)?;
    is_op(&nodes[i], op).then_some((i, &nodes[i]))
}

/// `Reshape` that does not let a `0` in the shape mean a zero-sized dimension.
fn plain_reshape(node: &NodeIr) -> bool {
    is_op(node, "Reshape")
        && node.inputs.len() == 2
        && node.attrs.get("allowzero").and_then(AttrValue::as_i64) != Some(1)
}

/// `Shape(of)` over the whole shape: no `end`, and `start` absent or `0`.
/// ORT fills in defaults when it resolves a graph, so the EP sees `start = 0`
/// on a node the file wrote without it.
fn whole_shape(node: &NodeIr, of: &str) -> bool {
    is_op(node, "Shape")
        && node.inputs.len() == 1
        && node.inputs[0] == of
        && node
            .attrs
            .get("start")
            .is_none_or(|a| a.as_i64() == Some(0))
        && !node.attrs.contains_key("end")
}

/// The nodes computing `Unsqueeze(Gather(Shape(t), axis), [0])`, the 1-element
/// shape `[t.shape[axis]]`, rooted at the value `name`.
fn shape_dim(
    nodes: &[NodeIr],
    producers: &HashMap<String, usize>,
    constants: &HashMap<String, InitializerIr>,
    name: &str,
    t: &str,
    axis: i64,
) -> Option<[usize; 3]> {
    let (u, unsqueeze) = producer(nodes, producers, name, "Unsqueeze")?;
    if unsqueeze.inputs.len() != 2 || ints(&unsqueeze.inputs[1], constants)? != [0] {
        return None;
    }
    let (g, gather) = producer(nodes, producers, &unsqueeze.inputs[0], "Gather")?;
    let gather_axis = gather.attrs.get("axis").and_then(AttrValue::as_i64);
    if gather.inputs.len() != 2 || gather_axis.unwrap_or(0) != 0 {
        return None;
    }
    // a scalar index: a `[1]` index would make the Unsqueeze produce rank 2
    let index = constants.get(&gather.inputs[1])?;
    if !index.shape.is_empty() || ints(&gather.inputs[1], constants)? != [axis] {
        return None;
    }
    let (s, shape) = producer(nodes, producers, &gather.inputs[0], "Shape")?;
    whole_shape(shape, t).then_some([u, g, s])
}

/// `Squeeze` or `Unsqueeze` (`op`) of the last axis of a rank-5 value.
fn last_axis_op(node: &NodeIr, op: &str, constants: &HashMap<String, InitializerIr>) -> bool {
    let axes = match node.inputs.get(1) {
        Some(axes) => ints(axes, constants),
        None => node
            .attrs
            .get("axes")
            .and_then(AttrValue::as_ints)
            .map(<[i64]>::to_vec),
    };
    is_op(node, op) && matches!(axes.as_deref(), Some([-1 | 4]))
}

/// The swap of every pair from its two halves `x0` and `x1` (rank 5, last
/// axis of size 1):
///
/// ```text
/// swapped = Concat(Unsqueeze(Neg(Squeeze(x1))), Unsqueeze(Squeeze(x0)), axis = -1)
/// ```
///
/// Returns its nodes, the `Concat` last.
fn swap_pairs(
    nodes: &[NodeIr],
    index: &Index,
    constants: &HashMap<String, InitializerIr>,
    x0: &str,
    x1: &str,
) -> Option<Vec<usize>> {
    let sq0 = index.only_consumer(x0)?;
    let sq1 = index.only_consumer(x1)?;
    if !last_axis_op(&nodes[sq0], "Squeeze", constants)
        || !last_axis_op(&nodes[sq1], "Squeeze", constants)
    {
        return None;
    }
    let ng = index.only_consumer(nodes[sq1].outputs.first()?)?;
    if !is_op(&nodes[ng], "Neg") {
        return None;
    }
    let u1 = index.only_consumer(nodes[ng].outputs.first()?)?;
    let u0 = index.only_consumer(nodes[sq0].outputs.first()?)?;
    if !last_axis_op(&nodes[u1], "Unsqueeze", constants)
        || !last_axis_op(&nodes[u0], "Unsqueeze", constants)
    {
        return None;
    }
    let sw = index.only_consumer(nodes[u1].outputs.first()?)?;
    let swap = &nodes[sw];
    if !is_op(swap, "Concat")
        || !last_of_rank5(swap.attrs.get("axis").and_then(AttrValue::as_i64))
        || swap.inputs != [nodes[u1].outputs[0].clone(), nodes[u0].outputs[0].clone()]
        || swap.outputs.is_empty()
    {
        return None;
    }
    Some(vec![sq0, sq1, ng, u1, u0, sw])
}

/// `t` to `rot` as TorchScript exports write it, rooted at the first `Reshape`:
/// shapes computed from `Shape`, the halves taken by a `Split`.
///
/// ```text
/// pairs   = Reshape(t, [t.0, t.1, t.2, -1, 2])          the shape from Shape(t)
/// x0, x1  = Split(pairs, [1, 1], axis = -1)
/// swapped = swap_pairs(x0, x1)
/// rot     = Reshape(swapped, [swapped.shape[:-2], -1])  the shape from Shape(swapped)
/// ```
///
/// Returns the nodes and the second `Reshape`.
fn traced_rotation(
    nodes: &[NodeIr],
    root: usize,
    index: &Index,
    producers: &HashMap<String, usize>,
    constants: &HashMap<String, InitializerIr>,
) -> Option<(Vec<usize>, usize)> {
    let pairs = &nodes[root];
    if !plain_reshape(pairs) {
        return None;
    }
    let t = pairs.inputs[0].as_str();
    let mut members = vec![root];

    // the shape [t.0, t.1, t.2, -1, 2]
    let (c, concat) = producer(nodes, producers, &pairs.inputs[1], "Concat")?;
    if concat.attrs.get("axis").and_then(AttrValue::as_i64) != Some(0) || concat.inputs.len() != 5 {
        return None;
    }
    members.push(c);
    for (axis, name) in concat.inputs[..3].iter().enumerate() {
        members.extend(shape_dim(
            nodes,
            producers,
            constants,
            name,
            t,
            axis as i64,
        )?);
    }
    if ints(&concat.inputs[3], constants)? != [-1] || ints(&concat.inputs[4], constants)? != [2] {
        return None;
    }

    // swap and negate within each pair
    let sp = index.only_consumer(pairs.outputs.first()?)?;
    let split = &nodes[sp];
    let sizes = match split.attrs.get("split").and_then(AttrValue::as_ints) {
        Some(sizes) => sizes.to_vec(),
        None => ints(split.inputs.get(1)?, constants)?,
    };
    if !is_op(split, "Split")
        || !last_of_rank5(split.attrs.get("axis").and_then(AttrValue::as_i64))
        || split.outputs.len() != 2
        || sizes != [1, 1]
    {
        return None;
    }
    let swap = swap_pairs(
        nodes,
        index,
        constants,
        &split.outputs[0],
        &split.outputs[1],
    )?;
    let swapped = &nodes[*swap.last()?].outputs[0];
    members.push(sp);
    members.extend(swap);

    // back to [B, H, S, D]: the shape is [swapped.shape[:-2], -1]
    let rt = *index
        .consumers
        .get(swapped)?
        .iter()
        .find(|&&i| nodes[i].inputs[0] == *swapped && plain_reshape(&nodes[i]))?;
    let rot = &nodes[rt];
    let (c, concat) = producer(nodes, producers, &rot.inputs[1], "Concat")?;
    if concat.attrs.get("axis").and_then(AttrValue::as_i64) != Some(0)
        || concat.inputs.len() != 2
        || ints(&concat.inputs[1], constants)? != [-1]
    {
        return None;
    }
    let (sl, slice) = producer(nodes, producers, &concat.inputs[0], "Slice")?;
    let operand = |k: usize| slice.inputs.get(k).and_then(|name| ints(name, constants));
    let steps_one = slice
        .inputs
        .get(4)
        .is_none_or(|_| operand(4) == Some(vec![1]));
    if slice.inputs.len() < 4
        || operand(1)? != [0]
        || operand(2)? != [-2]
        || operand(3)? != [0]
        || !steps_one
    {
        return None;
    }
    let (sh, shape) = producer(nodes, producers, &slice.inputs[0], "Shape")?;
    if !whole_shape(shape, swapped) {
        return None;
    }
    members.extend([rt, c, sl, sh]);
    Some((members, rt))
}

/// `Slice(pairs, [start], [start + 1], [-1])`: one half of every pair, as a
/// rank-5 value with a last axis of size 1.
fn pair_half(node: &NodeIr, start: i64, constants: &HashMap<String, InitializerIr>) -> bool {
    let operand = |k: usize| node.inputs.get(k).and_then(|name| ints(name, constants));
    is_op(node, "Slice")
        && (4..=5).contains(&node.inputs.len())
        && operand(1) == Some(vec![start])
        && operand(2) == Some(vec![start + 1])
        && matches!(operand(3).as_deref(), Some([-1 | 4]))
        && node
            .inputs
            .get(4)
            .is_none_or(|_| operand(4) == Some(vec![1]))
}

/// `t` to `rot` as `torch.onnx.export(dynamo=True)` writes it, rooted at the
/// first `Reshape`: constant shapes, the halves taken by two `Slice`s.
///
/// ```text
/// pairs   = Reshape(t, [B, H, S, -1, 2])
/// x0, x1  = Slice(pairs, [0], [1], [-1]), Slice(pairs, [1], [2], [-1])
/// swapped = swap_pairs(x0, x1)
/// rot     = Reshape(swapped, [B, H, S, D])
/// ```
///
/// Both shapes are positive apart from the `-1`, so `allowzero` (which the
/// export sets) changes nothing. Returns the nodes, the second `Reshape` and
/// `[B, H, S, D]`, which the fused node checks against `t` at run time.
fn exported_rotation(
    nodes: &[NodeIr],
    root: usize,
    index: &Index,
    constants: &HashMap<String, InitializerIr>,
) -> Option<(Vec<usize>, usize, [i64; 4])> {
    let pairs = &nodes[root];
    if !is_op(pairs, "Reshape") || pairs.inputs.len() != 2 {
        return None;
    }
    let [b, h, s, -1, 2] = ints(&pairs.inputs[1], constants)?[..] else {
        return None;
    };
    if b <= 0 || h <= 0 || s <= 0 {
        return None;
    }
    let pairs_out = pairs.outputs.first()?;
    if index.graph_outputs.contains(pairs_out) {
        return None;
    }
    let halves = index.consumers.get(pairs_out)?;
    let half = |start: i64| {
        halves
            .iter()
            .copied()
            .find(|&i| nodes[i].inputs[0] == *pairs_out && pair_half(&nodes[i], start, constants))
    };
    let (h0, h1) = (half(0)?, half(1)?);
    if halves.len() != 2 {
        return None;
    }
    let swap = swap_pairs(
        nodes,
        index,
        constants,
        nodes[h0].outputs.first()?,
        nodes[h1].outputs.first()?,
    )?;
    let swapped = &nodes[*swap.last()?].outputs[0];

    let rt = index.only_consumer(swapped)?;
    let rot = &nodes[rt];
    if !is_op(rot, "Reshape") || rot.inputs.len() != 2 || rot.inputs[0] != *swapped {
        return None;
    }
    let [rb, rh, rs, d] = ints(&rot.inputs[1], constants)?[..] else {
        return None;
    };
    if [rb, rh, rs] != [b, h, s] || d <= 0 || d % 2 != 0 {
        return None;
    }
    let mut members = vec![root, h0, h1];
    members.extend(swap);
    members.push(rt);
    Some((members, rt, [b, h, s, d]))
}

/// `Transpose(y, [0, 1, 3, 2])` of a `[B, H, S, D]` value `y` as the dynamo
/// export writes it, on three dimensions:
///
/// ```text
/// Reshape(y, [B·H, S, D])     or [-1, S, D]
/// Transpose(_, [0, 2, 1])
/// Reshape(_, [B, H, D, S])
/// ```
///
/// Returns the three nodes, from the only reader of `y`.
fn rank3_transpose(
    nodes: &[NodeIr],
    index: &Index,
    constants: &HashMap<String, InitializerIr>,
    y: &str,
    [b, h, s, d]: [i64; 4],
) -> Option<[usize; 3]> {
    let reshape = |i: usize, from: &str, targets: &[[i64; 3]]| {
        let node = &nodes[i];
        is_op(node, "Reshape")
            && node.inputs.len() == 2
            && node.inputs[0] == from
            && ints(&node.inputs[1], constants)
                .is_some_and(|target| targets.iter().any(|t| target == t))
    };
    let to3 = index.only_consumer(y)?;
    if !reshape(to3, y, &[[b * h, s, d], [-1, s, d]]) {
        return None;
    }
    let tr = index.only_consumer(nodes[to3].outputs.first()?)?;
    let perm = nodes[tr].attrs.get("perm").and_then(AttrValue::as_ints);
    if !is_op(&nodes[tr], "Transpose") || perm != Some(&[0, 2, 1][..]) {
        return None;
    }
    let to4 = index.only_consumer(nodes[tr].outputs.first()?)?;
    let node = &nodes[to4];
    let back = is_op(node, "Reshape")
        && node.inputs.len() == 2
        && node.inputs[0] == nodes[tr].outputs[0]
        && ints(&node.inputs[1], constants) == Some(vec![b, h, d, s]);
    back.then_some([to3, tr, to4])
}

/// Matches the interleaved rotary embedding as exporters write it out without
/// a `RotaryEmbedding` operator, rooted at the first `Reshape`:
///
/// ```text
/// rot = rotation(t)        traced_rotation or exported_rotation
/// y   = Mul(t, cos) + Mul(rot, sin)
/// ```
///
/// `rot[2i] = -t[2i+1]` and `rot[2i+1] = t[2i]`: everything between `t` and
/// `rot` only moves values, and the Neg is exact. SAM 3 writes it 64 times in
/// each vision encoder, once for each of Q and K in every layer: the int8
/// model in the traced form, the fp32 dynamo export in the other.
///
/// Three neighbours fold in when nothing else reads the value between them:
/// a `Transpose(x, [0, 2, 1, 3])` producing `t` (the fused node then reads
/// `[B, S, H, D]`), a `Transpose(y, [0, 1, 3, 2])` after it (K before
/// `Q · Kᵀ`; the dynamo export writes it on three dimensions, see
/// `rank3_transpose`), and a `Mul` by a scalar constant after that
/// (attention's scale, which SAM 3 applies to both Q and K). The fused node replays the same
/// multiplications and the addition with the same roundings; that `cos` and
/// `sin` broadcast as `[S, D]` is checked at run time.
fn match_interleaved_rope(
    nodes: &[NodeIr],
    root: usize,
    index: &Index,
    producers: &HashMap<String, usize>,
    constants: &HashMap<String, InitializerIr>,
) -> Option<Match> {
    let t = nodes[root].inputs.first()?.as_str();
    let (mut members, rt, dims) = match traced_rotation(nodes, root, index, producers, constants) {
        Some((members, rt)) => (members, rt, None),
        None => {
            let (members, rt, dims) = exported_rotation(nodes, root, index, constants)?;
            (members, rt, Some(dims))
        }
    };
    let rot = &nodes[rt];

    // y = t · cos + rot · sin
    let ms = index.only_consumer(rot.outputs.first()?)?;
    let mul_sin = &nodes[ms];
    let sin = other_input(mul_sin, &rot.outputs[0])?;
    let ad = index.only_consumer(mul_sin.outputs.first()?)?;
    let add = &nodes[ad];
    if !is_op(mul_sin, "Mul") || !is_op(add, "Add") {
        return None;
    }
    let direct = other_input(add, &mul_sin.outputs[0])?;
    let (mc, mul_cos) = producer(nodes, producers, direct, "Mul")?;
    let cos = other_input(mul_cos, t)?;
    if cos == t || sin == rot.outputs[0] {
        return None;
    }
    members.extend([ms, ad, mc]);

    // the Transpose producing t, when only the pattern reads t
    let mut x = t;
    let mut bshd = 0;
    if let Some((tr, transpose)) = producer(nodes, producers, t, "Transpose") {
        let perm = transpose.attrs.get("perm").and_then(AttrValue::as_ints);
        let readers = index.consumers.get(t).map_or(&[][..], Vec::as_slice);
        if perm == Some(&[0, 2, 1, 3][..])
            && !index.graph_outputs.contains(t)
            && readers.iter().all(|i| members.contains(i))
        {
            x = &transpose.inputs[0];
            bshd = 1;
            members.push(tr);
        }
    }

    // the Transpose and the scale after it
    let mut last = ad;
    let mut transpose_output = 0;
    if let Some(tr) = index.only_consumer(add.outputs.first()?) {
        let perm = nodes[tr].attrs.get("perm").and_then(AttrValue::as_ints);
        if is_op(&nodes[tr], "Transpose") && perm == Some(&[0, 1, 3, 2][..]) {
            transpose_output = 1;
            last = tr;
            members.push(tr);
        } else if let Some(dims) = dims
            && let Some(steps) = rank3_transpose(nodes, index, constants, &add.outputs[0], dims)
        {
            transpose_output = 1;
            last = steps[2];
            members.extend(steps);
        }
    }
    let mut scale = None;
    if let Some(m) = index.only_consumer(nodes[last].outputs.first()?) {
        let value = &nodes[last].outputs[0];
        if is_op(&nodes[m], "Mul")
            && let Some(factor) = other_input(&nodes[m], value)
            && factor != value
            && scalar_f32(factor, constants).is_some()
        {
            scale = Some(factor.to_owned());
            last = m;
            members.push(m);
        }
    }

    // nothing computed inside may escape: the shape plumbing fans out (one
    // Shape read by three Gathers), so this is checked over the whole set
    members.sort_unstable();
    members.dedup();
    let escapes = members.iter().filter(|&&i| i != last).any(|&i| {
        nodes[i].outputs.iter().any(|out| {
            index.graph_outputs.contains(out)
                || index
                    .consumers
                    .get(out)
                    .is_some_and(|readers| readers.iter().any(|r| !members.contains(r)))
        })
    });
    if escapes {
        return None;
    }

    let end = &nodes[last];
    let mut inputs = vec![x.to_owned(), cos.to_owned(), sin.to_owned()];
    inputs.extend(scale);
    Some(Match {
        members,
        fused: NodeIr {
            domain: REWRITE_DOMAIN.to_owned(),
            op: "InterleavedRope".to_owned(),
            opset: end.opset,
            name: end.name.clone(),
            inputs,
            outputs: end.outputs.clone(),
            attrs: HashMap::from_iter(
                [
                    ("bshd".to_owned(), AttrValue::Int(bshd)),
                    (
                        "transpose_output".to_owned(),
                        AttrValue::Int(transpose_output),
                    ),
                ]
                .into_iter()
                .chain(dims.map(|dims| ("shape".to_owned(), AttrValue::Ints(dims.to_vec())))),
            ),
        },
    })
}

/// Replaces every written-out interleaved rotary embedding with one
/// `InterleavedRope`. Returns how many patterns were fused.
pub fn fuse_interleaved_rope(ir: &mut GraphIr) -> usize {
    // only shape operands and scalars are looked up: no copy of the weights
    let small = |tensor: &InitializerIr| tensor.data.len() <= 64;
    let mut constants = constant_outputs(&ir.nodes);
    constants.retain(|_, tensor| small(tensor));
    constants.extend(
        ir.initializers
            .iter()
            .filter(|(_, tensor)| small(tensor))
            .map(|(name, tensor)| (name.clone(), tensor.clone())),
    );
    let producers = producers(ir);
    replace_matches(ir, |nodes, root, index| {
        match_interleaved_rope(nodes, root, index, &producers, &constants)
    })
}

/// The `Mul` weights of a packed byte, first element in the most significant
/// bit as `numpy.packbits` orders them.
const PACK_WEIGHTS: [f32; 8] = [128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0];

/// Matches `numpy.packbits(x > 0, axis = -1)` written in standard operators,
/// rooted at the `Reshape`, when no intermediate escapes:
///
/// ```text
/// Reshape(x, [..., W/8, 8]) → Greater(·, 0) → Cast(to = FLOAT)
///   → Mul(·, [128, 64, …, 1]) → ReduceSum(last axis, keepdims = 0)
///   → Cast(to = UINT8)
/// ```
///
/// This is how `scripts/pack_mask_bits.py` returns a mask decoder's logits
/// when the caller only thresholds them at zero: the bits at a 32nd of the
/// f32 bytes. Every step is exact — a sum of distinct powers of two below 256
/// is an integer f32 holds, in any order — so the fused node is the same
/// function. It keeps `x` and the reshape target; its output is the target's
/// shape without the last axis.
fn match_pack_positive_bits(
    nodes: &[NodeIr],
    root: usize,
    index: &Index,
    constants: &HashMap<String, InitializerIr>,
) -> Option<Match> {
    let reshape = &nodes[root];
    if !plain_reshape(reshape) {
        return None;
    }
    let target = ints(&reshape.inputs[1], constants)?;
    if target.last() != Some(&8) {
        return None;
    }
    let last = target.len() as i64 - 1;

    let g = index.only_consumer(reshape.outputs.first()?)?;
    let greater = &nodes[g];
    if !is_op(greater, "Greater")
        || greater.inputs.len() != 2
        || greater.inputs[0] != reshape.outputs[0]
        || scalar_f32(&greater.inputs[1], constants)? != 0.0
    {
        return None;
    }
    let c = index.only_consumer(greater.outputs.first()?)?;
    let cast = &nodes[c];
    if !is_op(cast, "Cast")
        || cast.attrs.get("to").and_then(AttrValue::as_i64) != Some(ElementType::Float32 as i64)
    {
        return None;
    }
    let m = index.only_consumer(cast.outputs.first()?)?;
    let mul = &nodes[m];
    let weights = constants.get(other_input(mul, &cast.outputs[0])?)?;
    // rank 1, so it broadcasts along the last axis only
    let values = HostTensor::new(weights.dtype, weights.shape.clone(), weights.data.clone());
    if !is_op(mul, "Mul")
        || weights.dtype != ElementType::Float32 as i32
        || weights.shape != [8]
        || values.to_f32().ok()? != PACK_WEIGHTS
    {
        return None;
    }
    let r = index.only_consumer(mul.outputs.first()?)?;
    let reduce = &nodes[r];
    // both frontends fold an opset-18 `axes` input into the attribute
    let axes = reduce.attrs.get("axes").and_then(AttrValue::as_ints)?;
    if !is_op(reduce, "ReduceSum")
        || reduce.inputs.len() != 1
        || !matches!(axes, [axis] if *axis == -1 || *axis == last)
        || reduce.attrs.get("keepdims").and_then(AttrValue::as_i64) != Some(0)
    {
        return None;
    }
    let u = index.only_consumer(reduce.outputs.first()?)?;
    let out = &nodes[u];
    if !is_op(out, "Cast")
        || out.attrs.get("to").and_then(AttrValue::as_i64) != Some(ElementType::Uint8 as i64)
    {
        return None;
    }
    Some(Match {
        members: vec![root, g, c, m, r, u],
        fused: NodeIr {
            domain: REWRITE_DOMAIN.to_owned(),
            op: "PackPositiveBits".to_owned(),
            opset: out.opset,
            name: out.name.clone(),
            inputs: reshape.inputs.clone(),
            outputs: out.outputs.clone(),
            attrs: HashMap::new(),
        },
    })
}

/// Replaces every `numpy.packbits(x > 0, axis = -1)` written in standard
/// operators with one `PackPositiveBits`. Returns how many were fused.
pub fn fuse_pack_positive_bits(ir: &mut GraphIr) -> usize {
    // only shape operands, scalars and the eight weights are looked up
    let small = |tensor: &InitializerIr| tensor.data.len() <= 64;
    let mut constants = constant_outputs(&ir.nodes);
    constants.retain(|_, tensor| small(tensor));
    constants.extend(
        ir.initializers
            .iter()
            .filter(|(_, tensor)| small(tensor))
            .map(|(name, tensor)| (name.clone(), tensor.clone())),
    );
    replace_matches(ir, |nodes, root, index| {
        match_pack_positive_bits(nodes, root, index, &constants)
    })
}

/// Matches `Add(a, b) → Reshape → Transpose → Pad(mode = constant)` rooted at
/// the `Add`, when no intermediate escapes and the pads are not given `axes`.
///
/// This is how SAM 3's detector builds the box relative-position bias of each
/// decoder layer: `[1, 200, 72, 1, 8] + [1, 200, 1, 72, 8]` broadcast to a
/// 33 MB product, merged to 5184 keys, moved heads first and padded with the
/// presence token's zero row — four passes over the big tensor for a sum of
/// two 0.5 MB ones. The fused node keeps `a`, `b`, the reshape target, the
/// pads and the optional constant value as inputs and the permutation as its
/// attribute; it writes every element once with the same f32 addition, so
/// nothing changes by a bit. Shapes are unknown here: whether the reshape
/// only merges axes is checked at run time, where anything else runs the four
/// original steps.
fn match_add_transpose_pad(nodes: &[NodeIr], root: usize, index: &Index) -> Option<Match> {
    let add = &nodes[root];
    if !is_op(add, "Add") || add.inputs.len() != 2 || add.inputs.iter().any(String::is_empty) {
        return None;
    }
    let sum = add.outputs.first()?;
    let r = index.only_consumer(sum)?;
    let reshape = &nodes[r];
    if !is_op(reshape, "Reshape")
        || reshape.inputs.len() != 2
        || reshape.inputs[0] != *sum
        || reshape.inputs[1] == *sum
    {
        return None;
    }
    let reshaped = reshape.outputs.first()?;
    let t = index.only_consumer(reshaped)?;
    let transpose = &nodes[t];
    if !is_op(transpose, "Transpose") || transpose.inputs.len() != 1 {
        return None;
    }
    let p = index.only_consumer(transpose.outputs.first()?)?;
    let pad = &nodes[p];
    let mode = pad.attrs.get("mode").map(|mode| match mode {
        AttrValue::String(mode) => mode.as_str(),
        _ => "",
    });
    if !is_op(pad, "Pad")
        || !(2..=3).contains(&pad.inputs.len())
        || pad.inputs[0] != transpose.outputs[0]
        || pad.inputs[1..].contains(&transpose.outputs[0])
        || pad.inputs[1].is_empty()
        || mode.is_some_and(|mode| mode != "constant")
    {
        return None;
    }
    let mut inputs = add.inputs.clone();
    inputs.extend(pad.inputs[1..].iter().cloned());
    inputs.insert(2, reshape.inputs[1].clone());
    let attrs = transpose
        .attrs
        .get("perm")
        .map(|perm| HashMap::from([("perm".to_owned(), perm.clone())]))
        .unwrap_or_default();
    Some(Match {
        members: vec![root, r, t, p],
        fused: NodeIr {
            domain: REWRITE_DOMAIN.to_owned(),
            op: "AddTransposePad".to_owned(),
            opset: pad.opset,
            name: pad.name.clone(),
            inputs,
            outputs: pad.outputs.clone(),
            attrs,
        },
    })
}

/// Replaces every `Add → Reshape → Transpose → Pad` with one
/// `AddTransposePad`. Returns how many were fused.
pub fn fuse_add_transpose_pad(ir: &mut GraphIr) -> usize {
    replace_matches(ir, match_add_transpose_pad)
}

/// Matches `Add(a, b) → LayerNormalization` rooted at the normalization, when
/// it asks for no statistics output.
///
/// The residual `Add` before a transformer's normalization is a full pass over
/// the activations that the normalization then reads back. Unlike the other
/// patterns here the sum may escape: a pre-norm block reads it again as the
/// next residual. It then becomes the fused node's second output, written by
/// the same pass, and every other reader must come after the normalization,
/// whose slot the fused node takes. Nothing changes numerically: the kernel
/// adds the same two `f32` the `Add` would have.
fn match_add_layernorm(nodes: &[NodeIr], root: usize, index: &Index) -> Option<Match> {
    let norm = &nodes[root];
    if !is_op(norm, "LayerNormalization")
        || !(2..=3).contains(&norm.inputs.len())
        || norm.outputs.iter().skip(1).any(|o| !o.is_empty())
    {
        return None;
    }
    let sum = norm.inputs.first()?;
    let add = &nodes[*index.producers.get(sum)?];
    if !is_op(add, "Add")
        || add.inputs.len() != 2
        || add.inputs.iter().any(String::is_empty)
        || norm.inputs[1..].contains(sum)
    {
        return None;
    }
    let readers = index.consumers.get(sum)?;
    if readers.iter().any(|&reader| reader < root) {
        return None;
    }
    let mut outputs = vec![norm.outputs[0].clone()];
    if readers.len() > 1 || index.graph_outputs.contains(sum) {
        outputs.push(sum.clone());
    }
    let mut inputs = add.inputs.clone();
    inputs.extend(norm.inputs[1..].iter().cloned());
    Some(Match {
        members: vec![index.producers[sum], root],
        fused: NodeIr {
            domain: REWRITE_DOMAIN.to_owned(),
            op: "AddLayerNormalization".to_owned(),
            opset: norm.opset,
            name: norm.name.clone(),
            inputs,
            outputs,
            attrs: norm.attrs.clone(),
        },
    })
}

/// Replaces every `Add → LayerNormalization` with one
/// `AddLayerNormalization`. Returns how many were fused.
pub fn fuse_add_layernorm(ir: &mut GraphIr) -> usize {
    replace_matches(ir, match_add_layernorm)
}

/// Runs `matcher` from every node, keeps the matches that overlap no earlier
/// one, and splices their fused nodes in. Returns how many were fused.
fn replace_matches(
    ir: &mut GraphIr,
    matcher: impl Fn(&[NodeIr], usize, &Index) -> Option<Match>,
) -> usize {
    let index = Index::build(ir);

    let mut claimed = vec![false; ir.nodes.len()];
    let mut matches = Vec::new();
    for root in 0..ir.nodes.len() {
        let Some(m) = matcher(&ir.nodes, root, &index) else {
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
        "Transpose" => {
            let rank = inputs[0].shape.len();
            let perm: Vec<usize> = match node.attrs.get("perm") {
                Some(attr) => attr
                    .as_ints()?
                    .iter()
                    .map(|&p| usize::try_from(p).ok())
                    .collect::<Option<_>>()?,
                None => (0..rank).rev().collect(),
            };
            let mut seen = vec![false; rank];
            if perm.len() != rank
                || perm
                    .iter()
                    .any(|&p| p >= rank || std::mem::replace(&mut seen[p], true))
            {
                return None;
            }
            let out_shape: Vec<i64> = perm.iter().map(|&p| inputs[0].shape[p]).collect();
            crate::interp::host_transpose(&inputs[0], &perm, &out_shape).ok()
        }
        // SAM 3's vision encoder tiles its position table 4×4 and crops it,
        // `Slice(Slice(Tile(Transpose(Reshape(pos)))))`: on host every run, the
        // `Tile` alone ~185 ms of a ~690 ms int8 run.
        "Tile" => crate::host_ops::tile(&inputs[0], &inputs.get(1)?.to_i64().ok()?).ok(),
        // Opset ≥ 10 only: the attribute form of opset 1 has one input.
        "Slice" if inputs.len() >= 3 => {
            let operand = |i: usize| inputs.get(i).map(HostTensor::to_i64).transpose().ok();
            let (axes, steps) = (operand(3)?, operand(4)?);
            let (shape, starts, steps) = crate::host_ops::slice_params(
                &inputs[0].shape,
                &operand(1)??,
                &operand(2)??,
                axes.as_deref(),
                steps.as_deref(),
            )
            .ok()?;
            crate::host_ops::slice(&inputs[0], &shape, &starts, &steps).ok()
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
    let mut readers: HashMap<&str, usize> = HashMap::new();
    for name in ir.nodes.iter().flat_map(|n| n.inputs.iter()) {
        *readers.entry(name.as_str()).or_default() += 1;
    }
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
        // A permuted, tiled or sliced copy of a constant something else also
        // reads would keep both layouts resident (a tied embedding read by
        // `Gather` and, transposed, by a `MatMul`). Folded only when this node
        // is the sole reader of its data, so that pruning the source makes the
        // fold free in memory; the small index operands may be shared.
        if matches!(node.op.as_str(), "Transpose" | "Tile" | "Slice")
            && node.inputs.first().is_some_and(|name| {
                readers.get(name.as_str()) != Some(&1) || graph_outputs.contains(name.as_str())
            })
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
            opset: 11,
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
        };
        assert_eq!(fold_constants(&mut ir), 2);
        assert_eq!(ir.nodes.len(), 1);
        assert_eq!(ir.initializers["w2"].shape, [1, 4, 1]);
    }

    fn floats(shape: Vec<i64>, values: impl IntoIterator<Item = f32>) -> InitializerIr {
        InitializerIr {
            dtype: ElementType::Float32 as i32,
            shape,
            data: values.into_iter().flat_map(f32::to_le_bytes).collect(),
        }
    }

    /// SAM 3's detector permutes a reshaped position table every run:
    /// `Transpose(Reshape(pos))`, all constant, is now an initializer.
    #[test]
    fn folds_a_constant_transpose_read_once() {
        let mut transpose = node("Transpose", &["pos2"], "pos_t");
        transpose
            .attrs
            .insert("perm".into(), AttrValue::Ints(vec![0, 2, 1]));
        let mut ir = GraphIr {
            nodes: vec![
                node("Reshape", &["pos", "shape"], "pos2"),
                transpose,
                node("Add", &["x", "pos_t"], "y"),
            ],
            initializers: HashMap::from([
                ("pos".into(), floats(vec![6], (0..6).map(|v| v as f32))),
                (
                    "shape".into(),
                    InitializerIr {
                        dtype: ElementType::Int64 as i32,
                        shape: vec![3],
                        data: [1i64, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect(),
                    },
                ),
            ]),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            ..Default::default()
        };
        assert_eq!(fold_constants(&mut ir), 2);
        assert_eq!(ir.nodes.len(), 1);
        // [[0 1 2] [3 4 5]] → [[0 3] [1 4] [2 5]]
        let expected = floats(vec![1, 3, 2], [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(ir.initializers["pos_t"].shape, expected.shape);
        assert_eq!(ir.initializers["pos_t"].data, expected.data);
        // no layout kept twice: the source is unread once pruned
        assert_eq!(prune_dead_initializers(&mut ir), 24 + 24 + 24);
        assert!(!ir.initializers.contains_key("pos"));
    }

    /// No `perm` reverses the axes; a source another node also reads, or an
    /// invalid permutation, is left to the interpreter.
    #[test]
    fn folds_a_transpose_only_when_it_is_free_and_valid() {
        let mut ir = GraphIr {
            nodes: vec![
                node("Transpose", &["w"], "w_t"),
                node("Add", &["x", "w_t"], "y"),
            ],
            initializers: HashMap::from([(
                "w".into(),
                floats(vec![2, 3], (0..6).map(|v| v as f32)),
            )]),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            ..Default::default()
        };
        assert_eq!(fold_constants(&mut ir), 1);
        let expected = floats(vec![3, 2], [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(ir.initializers["w_t"].shape, expected.shape);
        assert_eq!(ir.initializers["w_t"].data, expected.data);

        let shared = |perm: Vec<i64>, also_read: bool| {
            let mut transpose = node("Transpose", &["w"], "w_t");
            transpose.attrs.insert("perm".into(), AttrValue::Ints(perm));
            let mut nodes = vec![transpose, node("Add", &["x", "w_t"], "y")];
            if also_read {
                nodes.push(node("Mul", &["x", "w"], "z"));
            }
            GraphIr {
                nodes,
                initializers: HashMap::from([("w".into(), floats(vec![2, 3], [0.0; 6]))]),
                inputs: vec!["x".into()],
                outputs: vec!["y".into(), "z".into()],
                ..Default::default()
            }
        };
        assert_eq!(fold_constants(&mut shared(vec![1, 0], true)), 0);
        assert_eq!(fold_constants(&mut shared(vec![1, 1], false)), 0);
        assert_eq!(fold_constants(&mut shared(vec![0, 2], false)), 0);
        assert_eq!(fold_constants(&mut shared(vec![0], false)), 0);
        assert_eq!(fold_constants(&mut shared(vec![1, 0], false)), 1);
    }

    fn longs(values: &[i64]) -> InitializerIr {
        InitializerIr {
            dtype: ElementType::Int64 as i32,
            shape: vec![values.len() as i64],
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    /// SAM 3's vision encoder tiles a constant position table and crops it,
    /// `Slice(Slice(Tile(pos)))`, with the index operands shared by both
    /// slices: the chain becomes one initializer, and its sources are unread.
    #[test]
    fn folds_a_constant_tile_and_its_slices() {
        let mut ir = GraphIr {
            nodes: vec![
                node("Tile", &["pos", "repeats"], "tiled"),
                node("Slice", &["tiled", "zero", "three", "axis0"], "rows"),
                node("Slice", &["rows", "zero", "three", "axis1"], "crop"),
                node("Add", &["x", "crop"], "y"),
            ],
            initializers: HashMap::from([
                ("pos".into(), floats(vec![2, 2], [0.0, 1.0, 2.0, 3.0])),
                ("repeats".into(), longs(&[2, 2])),
                ("zero".into(), longs(&[0])),
                ("three".into(), longs(&[3])),
                ("axis0".into(), longs(&[0])),
                ("axis1".into(), longs(&[1])),
            ]),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            ..Default::default()
        };
        assert_eq!(fold_constants(&mut ir), 3);
        assert_eq!(ir.nodes.len(), 1);
        // [[0 1] [2 3]] tiled 2×2 is 4×4; its top-left 3×3:
        let expected = floats(vec![3, 3], [0.0, 1.0, 0.0, 2.0, 3.0, 2.0, 0.0, 1.0, 0.0]);
        assert_eq!(ir.initializers["crop"].shape, expected.shape);
        assert_eq!(ir.initializers["crop"].data, expected.data);
        prune_dead_initializers(&mut ir);
        let mut left: Vec<_> = ir.initializers.keys().map(String::as_str).collect();
        left.sort_unstable();
        assert_eq!(left, ["crop"]);
    }

    /// A tile of a constant another node also reads would keep both resident.
    #[test]
    fn leaves_a_tile_of_a_shared_constant_to_the_interpreter() {
        let mut ir = GraphIr {
            nodes: vec![
                node("Tile", &["pos", "repeats"], "tiled"),
                node("Add", &["x", "tiled"], "y"),
                node("Mul", &["x", "pos"], "z"),
            ],
            initializers: HashMap::from([
                ("pos".into(), floats(vec![2], [0.0, 1.0])),
                ("repeats".into(), longs(&[2])),
            ]),
            inputs: vec!["x".into()],
            outputs: vec!["y".into(), "z".into()],
            ..Default::default()
        };
        assert_eq!(fold_constants(&mut ir), 0);
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
            ..Default::default()
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
            ..Default::default()
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

    /// The linear layer of a dynamically quantized model: the scale is the
    /// product of the activation and weight scales, the bias an initializer.
    fn dequantizing_matmul() -> GraphIr {
        let mut cast = node("Cast", &["acc"], "accf");
        cast.attrs
            .insert("to".into(), AttrValue::Int(ElementType::Float32 as i64));
        GraphIr {
            nodes: vec![
                node("Mul", &["a_scale", "w_scale"], "scale"),
                node("MatMulInteger", &["a", "w", "a_zp", "w_zp"], "acc"),
                cast,
                node("Mul", &["accf", "scale"], "y0"),
                node("Add", &["bias", "y0"], "y"),
            ],
            initializers: HashMap::from([("bias".into(), vector(64))]),
            inputs: vec!["a".into()],
            outputs: vec!["y".into()],
            ..Default::default()
        }
    }

    #[test]
    fn fuses_the_dequantizing_epilogue() {
        let mut ir = dequantizing_matmul();
        assert_eq!(fuse_matmul_integer_epilogue(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 1, "the scale product goes too");
        let fused = &ir.nodes[0];
        assert_eq!(fused.domain, REWRITE_DOMAIN);
        assert_eq!(fused.op, "MatMulIntegerScaleBias");
        assert_eq!(
            fused.inputs,
            ["a", "w", "a_zp", "w_zp", "a_scale", "bias", "w_scale"]
        );
        assert_eq!(fused.outputs, ["y"]);
    }

    #[test]
    fn keeps_a_scale_product_read_elsewhere() {
        let mut ir = dequantizing_matmul();
        ir.outputs.push("scale".into());
        assert_eq!(fuse_matmul_integer_epilogue(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 2);
        assert_eq!(ir.nodes[0].op, "Mul");
        assert_eq!(
            ir.nodes[1].inputs,
            ["a", "w", "a_zp", "w_zp", "scale", "bias"]
        );
    }

    #[test]
    fn fuses_a_following_relu_into_the_epilogue() {
        let mut ir = dequantizing_matmul();
        ir.nodes.push(node("Relu", &["y"], "z"));
        ir.outputs = vec!["z".into()];
        assert_eq!(fuse_matmul_integer_epilogue(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 1);
        let fused = &ir.nodes[0];
        assert_eq!(fused.op, "MatMulIntegerScaleBias");
        assert_eq!(fused.outputs, ["z"]);
        assert_eq!(fused.attrs.get("relu"), Some(&AttrValue::Int(1)));

        // the biased value escapes: the Relu stays a node of its own
        let mut ir = dequantizing_matmul();
        ir.nodes.push(node("Relu", &["y"], "z"));
        ir.outputs = vec!["y".into(), "z".into()];
        assert_eq!(fuse_matmul_integer_epilogue(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 2);
        assert_eq!(ir.nodes[0].outputs, ["y"]);
        assert_eq!(ir.nodes[0].attrs.get("relu"), None);
    }

    #[test]
    fn fuses_the_dequantizing_convolution_with_its_geometry() {
        let mut ir = dequantizing_matmul();
        ir.nodes[1].op = "ConvInteger".into();
        ir.nodes[1]
            .attrs
            .insert("pads".into(), AttrValue::Ints(vec![1, 1, 1, 1]));
        assert_eq!(fuse_matmul_integer_epilogue(&mut ir), 0);
        assert_eq!(fuse_conv_integer_epilogue(&mut ir), 1);
        let fused = &ir.nodes[0];
        assert_eq!(fused.domain, REWRITE_DOMAIN);
        assert_eq!(fused.op, "ConvIntegerScaleBias");
        assert_eq!(
            fused.inputs,
            ["a", "w", "a_zp", "w_zp", "a_scale", "bias", "w_scale"]
        );
        assert_eq!(fused.outputs, ["y"]);
        assert_eq!(
            fused.attrs.get("pads"),
            Some(&AttrValue::Ints(vec![1, 1, 1, 1]))
        );
    }

    /// The affine and activation after SAM 3's pixel decoder `GroupNorm`.
    fn channel_affine_relu() -> GraphIr {
        GraphIr {
            nodes: vec![
                node("InstanceNormalization", &["g", "ones", "zeros"], "x"),
                node("Mul", &["gamma", "x"], "scaled"),
                node("Add", &["scaled", "beta"], "shifted"),
                node("Relu", &["shifted"], "y"),
            ],
            initializers: HashMap::from([
                ("gamma".into(), vector(256)),
                ("beta".into(), vector(256)),
            ]),
            inputs: vec!["g".into()],
            outputs: vec!["y".into()],
            ..Default::default()
        }
    }

    #[test]
    fn fuses_the_affine_and_relu_by_constants() {
        let mut ir = channel_affine_relu();
        assert_eq!(fuse_mul_add_relu(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 2);
        let fused = &ir.nodes[1];
        assert_eq!(fused.domain, REWRITE_DOMAIN);
        assert_eq!(fused.op, "MulAddRelu");
        assert_eq!(fused.inputs, ["x", "gamma", "beta"]);
        assert_eq!(fused.outputs, ["y"]);
    }

    #[test]
    fn leaves_an_affine_by_computed_values_or_escaping_alone() {
        // a scale computed at run time cannot tell which operand is `x`
        let mut ir = channel_affine_relu();
        ir.initializers.remove("gamma");
        assert_eq!(fuse_mul_add_relu(&mut ir), 0);

        let mut ir = channel_affine_relu();
        ir.initializers.remove("beta");
        assert_eq!(fuse_mul_add_relu(&mut ir), 0);

        let mut ir = channel_affine_relu();
        ir.outputs.push("shifted".into());
        assert_eq!(fuse_mul_add_relu(&mut ir), 0);

        let mut ir = channel_affine_relu();
        ir.nodes[3].op = "Sigmoid".into();
        assert_eq!(fuse_mul_add_relu(&mut ir), 0);
        assert_eq!(ir.nodes.len(), 4);
    }

    #[test]
    fn leaves_a_cast_to_another_type_alone() {
        let mut ir = dequantizing_matmul();
        ir.nodes[2]
            .attrs
            .insert("to".into(), AttrValue::Int(ElementType::Float16 as i64));
        assert_eq!(fuse_matmul_integer_epilogue(&mut ir), 0);
        assert_eq!(ir.nodes.len(), 5);
    }

    #[test]
    fn keeps_an_epilogue_whose_intermediate_escapes() {
        for leaked in ["acc", "accf", "y0"] {
            let mut ir = dequantizing_matmul();
            ir.outputs.push(leaked.into());
            assert_eq!(fuse_matmul_integer_epilogue(&mut ir), 0, "{leaked}");
        }
    }

    /// `Mul(x, x)` would leave the fused node reading the value it removed.
    #[test]
    fn refuses_a_square_instead_of_a_scale() {
        let mut ir = dequantizing_matmul();
        ir.nodes[3].inputs = vec!["accf".into(), "accf".into()];
        assert_eq!(fuse_matmul_integer_epilogue(&mut ir), 0);
    }

    /// The exact GELU of a pre-opset-20 transformer MLP.
    fn decomposed_gelu() -> GraphIr {
        GraphIr {
            nodes: vec![
                node("Div", &["x", "sqrt2"], "xs"),
                node("Erf", &["xs"], "e"),
                node("Add", &["e", "one"], "e1"),
                node("Mul", &["x", "e1"], "g"),
                node("Mul", &["g", "half"], "y"),
            ],
            initializers: HashMap::from([
                ("sqrt2".into(), scalar(std::f32::consts::SQRT_2)),
                ("one".into(), scalar(1.0)),
                ("half".into(), scalar(0.5)),
            ]),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            ..Default::default()
        }
    }

    #[test]
    fn fuses_the_decomposed_gelu() {
        let mut ir = decomposed_gelu();
        assert_eq!(fuse_decomposed_gelu(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 1);
        let fused = &ir.nodes[0];
        assert_eq!(fused.domain, REWRITE_DOMAIN);
        assert_eq!(fused.op, "DecomposedGelu");
        assert_eq!(fused.inputs, ["x", "sqrt2", "one", "half"]);
        assert_eq!(fused.outputs, ["y"]);
    }

    #[test]
    fn keeps_a_gelu_whose_intermediate_escapes() {
        for leaked in ["xs", "e", "e1", "g"] {
            let mut ir = decomposed_gelu();
            ir.outputs.push(leaked.into());
            assert_eq!(fuse_decomposed_gelu(&mut ir), 0, "{leaked}");
        }
    }

    #[test]
    fn refuses_a_gelu_without_scalar_constants() {
        let mut ir = decomposed_gelu();
        ir.initializers.insert("one".into(), vector(8));
        assert_eq!(fuse_decomposed_gelu(&mut ir), 0);
        let mut ir = decomposed_gelu();
        ir.initializers.remove("half");
        ir.inputs.push("half".into());
        assert_eq!(fuse_decomposed_gelu(&mut ir), 0);
    }

    /// The gate multiplies by something other than the `Div`'s input.
    #[test]
    fn refuses_a_gate_on_another_tensor() {
        let mut ir = decomposed_gelu();
        ir.nodes[3].inputs = vec!["z".into(), "e1".into()];
        assert_eq!(fuse_decomposed_gelu(&mut ir), 0);
    }

    /// The SAM 3 vision encoder's order: `(erf + 1) · 0.5` first, then `x ·`.
    fn factor_first_gelu() -> GraphIr {
        let mut ir = decomposed_gelu();
        ir.nodes[3] = node("Mul", &["half", "e1"], "g");
        ir.nodes[4] = node("Mul", &["x", "g"], "y");
        ir
    }

    #[test]
    fn fuses_the_gelu_that_scales_before_the_gate() {
        let mut ir = factor_first_gelu();
        assert_eq!(fuse_decomposed_gelu(&mut ir), 1);
        let fused = &ir.nodes[0];
        assert_eq!(fused.op, "DecomposedGelu");
        assert_eq!(fused.inputs, ["x", "sqrt2", "one", "half"]);
        assert_eq!(fused.outputs, ["y"]);
        assert_eq!(fused.attrs["factor_first"].as_i64(), Some(1));
        let mut ir = decomposed_gelu();
        fuse_decomposed_gelu(&mut ir);
        assert!(!ir.nodes[0].attrs.contains_key("factor_first"));
    }

    #[test]
    fn fuses_a_gelu_into_its_quantization() {
        for factor_first in [false, true] {
            let mut ir = if factor_first {
                factor_first_gelu()
            } else {
                decomposed_gelu()
            };
            ir.nodes.push(NodeIr {
                outputs: vec!["q".into(), "scale".into(), "zp".into()],
                ..node("DynamicQuantizeLinear", &["y"], "q")
            });
            ir.outputs = vec!["q".into(), "scale".into(), "zp".into()];
            assert_eq!(fuse_decomposed_gelu(&mut ir), 1);
            assert_eq!(fuse_gelu_quantize(&mut ir), 1);
            assert_eq!(ir.nodes.len(), 1);
            let fused = &ir.nodes[0];
            assert_eq!(fused.domain, REWRITE_DOMAIN);
            assert_eq!(fused.op, "DecomposedGeluQuantize");
            assert_eq!(fused.inputs, ["x", "sqrt2", "one", "half"]);
            assert_eq!(fused.outputs, ["q", "scale", "zp"]);
            assert_eq!(fused.attrs.contains_key("factor_first"), factor_first);
        }
    }

    /// The f32 GELU has another reader, or is a graph output: it must exist.
    #[test]
    fn keeps_a_quantized_gelu_that_is_read_elsewhere() {
        for leak in [false, true] {
            let mut ir = decomposed_gelu();
            ir.nodes.push(node("DynamicQuantizeLinear", &["y"], "q"));
            ir.outputs = vec!["q".into()];
            if leak {
                ir.outputs.push("y".into());
            } else {
                ir.nodes.push(node("Relu", &["y"], "r"));
                ir.outputs.push("r".into());
            }
            assert_eq!(fuse_decomposed_gelu(&mut ir), 1);
            assert_eq!(fuse_gelu_quantize(&mut ir), 0, "leak {leak}");
        }
    }

    /// Neither product multiplies by the `Div`'s input.
    #[test]
    fn refuses_a_factor_first_gelu_gated_by_another_tensor() {
        let mut ir = factor_first_gelu();
        ir.nodes[4].inputs = vec!["z".into(), "g".into()];
        assert_eq!(fuse_decomposed_gelu(&mut ir), 0);
        let mut ir = factor_first_gelu();
        ir.initializers.insert("half".into(), vector(8));
        assert_eq!(fuse_decomposed_gelu(&mut ir), 0);
    }

    fn int64(shape: Vec<i64>, values: &[i64]) -> InitializerIr {
        InitializerIr {
            dtype: ElementType::Int64 as i32,
            shape,
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    fn with_attr(mut node: NodeIr, key: &str, value: AttrValue) -> NodeIr {
        node.attrs.insert(key.into(), value);
        node
    }

    /// SAM 3's written-out interleaved rotary embedding: `t` is the
    /// `Transpose` of `x` when `bshd`, K's `Transpose` and the attention scale
    /// follow when asked. The output is always `y`.
    fn interleaved_rope(bshd: bool, transpose: bool, scale: bool) -> GraphIr {
        let mut nodes = Vec::new();
        if bshd {
            let perm = AttrValue::Ints(vec![0, 2, 1, 3]);
            nodes.push(with_attr(node("Transpose", &["x"], "t"), "perm", perm));
        }
        let axis = |n: NodeIr, a: i64| with_attr(n, "axis", AttrValue::Int(a));
        nodes.extend([
            node("Mul", &["t", "cos"], "direct"),
            node("Shape", &["t"], "shp"),
            axis(node("Gather", &["shp", "i0"], "g0"), 0),
            axis(node("Gather", &["shp", "i1"], "g1"), 0),
            axis(node("Gather", &["shp", "i2"], "g2"), 0),
            node("Unsqueeze", &["g0", "zero"], "d0"),
            node("Unsqueeze", &["g1", "zero"], "d1"),
            node("Unsqueeze", &["g2", "zero"], "d2"),
            axis(
                node(
                    "Concat",
                    &["d0", "d1", "d2", "minus1", "two"],
                    "pairs_shape",
                ),
                0,
            ),
            node("Reshape", &["t", "pairs_shape"], "pairs"),
        ]);
        let mut split = axis(node("Split", &["pairs", "halves"], "x0"), -1);
        split.outputs.push("x1".into());
        let last = if transpose || scale { "sum" } else { "y" };
        nodes.extend([
            split,
            node("Squeeze", &["x0", "minus1"], "even"),
            node("Squeeze", &["x1", "minus1"], "odd"),
            node("Neg", &["odd"], "neg"),
            node("Unsqueeze", &["neg", "minus1"], "negu"),
            node("Unsqueeze", &["even", "minus1"], "evenu"),
            axis(node("Concat", &["negu", "evenu"], "swapped"), -1),
            node("Shape", &["swapped"], "sshp"),
            node("Slice", &["sshp", "zero", "minus2", "zero"], "lead"),
            axis(node("Concat", &["lead", "minus1"], "rot_shape"), 0),
            node("Reshape", &["swapped", "rot_shape"], "rot"),
            node("Mul", &["rot", "sin"], "rotated"),
            node("Add", &["direct", "rotated"], last),
        ]);
        if transpose {
            let to = if scale { "kt" } else { "y" };
            let perm = AttrValue::Ints(vec![0, 1, 3, 2]);
            nodes.push(with_attr(node("Transpose", &["sum"], to), "perm", perm));
        }
        if scale {
            let from = if transpose { "kt" } else { "sum" };
            nodes.push(node("Mul", &[from, "scale"], "y"));
        }
        GraphIr {
            nodes,
            initializers: HashMap::from([
                ("i0".into(), int64(vec![], &[0])),
                ("i1".into(), int64(vec![], &[1])),
                ("i2".into(), int64(vec![], &[2])),
                ("zero".into(), int64(vec![1], &[0])),
                ("minus1".into(), int64(vec![1], &[-1])),
                ("minus2".into(), int64(vec![1], &[-2])),
                ("two".into(), int64(vec![1], &[2])),
                ("halves".into(), int64(vec![2], &[1, 1])),
                ("scale".into(), scalar(0.25)),
            ]),
            inputs: vec![
                if bshd { "x" } else { "t" }.into(),
                "cos".into(),
                "sin".into(),
            ],
            outputs: vec!["y".into()],
            ..Default::default()
        }
    }

    fn flag(node: &NodeIr, name: &str) -> Option<i64> {
        node.attrs.get(name).and_then(AttrValue::as_i64)
    }

    #[test]
    fn fuses_the_rope_of_q_with_its_transpose_and_scale() {
        let mut ir = interleaved_rope(true, false, true);
        assert_eq!(fuse_interleaved_rope(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 1, "{:?}", ir.nodes);
        let fused = &ir.nodes[0];
        assert_eq!(
            (fused.domain.as_str(), fused.op.as_str()),
            (REWRITE_DOMAIN, "InterleavedRope")
        );
        assert_eq!(fused.inputs, ["x", "cos", "sin", "scale"]);
        assert_eq!(fused.outputs, ["y"]);
        assert_eq!(flag(fused, "bshd"), Some(1));
        assert_eq!(flag(fused, "transpose_output"), Some(0));
    }

    #[test]
    fn fuses_the_rope_of_k_with_the_transpose_after_it() {
        let mut ir = interleaved_rope(true, true, true);
        assert_eq!(fuse_interleaved_rope(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 1);
        assert_eq!(flag(&ir.nodes[0], "transpose_output"), Some(1));
        assert_eq!(ir.nodes[0].outputs, ["y"]);
    }

    #[test]
    fn fuses_a_bare_rope() {
        let mut ir = interleaved_rope(false, false, false);
        assert_eq!(fuse_interleaved_rope(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 1);
        assert_eq!(ir.nodes[0].inputs, ["t", "cos", "sin"]);
        assert_eq!(flag(&ir.nodes[0], "bshd"), Some(0));
    }

    /// A `t` read outside the pattern keeps its Transpose; the rope still fuses.
    #[test]
    fn keeps_the_transpose_of_a_shared_input() {
        let mut ir = interleaved_rope(true, false, true);
        ir.outputs.push("t".into());
        assert_eq!(fuse_interleaved_rope(&mut ir), 1);
        let ops: Vec<&str> = ir.nodes.iter().map(|n| n.op.as_str()).collect();
        assert_eq!(ops, ["Transpose", "InterleavedRope"]);
        assert_eq!(ir.nodes[1].inputs[0], "t");
        assert_eq!(flag(&ir.nodes[1], "bshd"), Some(0));
    }

    /// An escaping sum keeps the tail outside: the Transpose and the scale stay.
    #[test]
    fn leaves_the_tail_of_an_escaping_sum() {
        let mut ir = interleaved_rope(true, true, true);
        ir.outputs.push("sum".into());
        assert_eq!(fuse_interleaved_rope(&mut ir), 1);
        let ops: Vec<&str> = ir.nodes.iter().map(|n| n.op.as_str()).collect();
        assert_eq!(ops, ["InterleavedRope", "Transpose", "Mul"]);
        assert_eq!(ir.nodes[0].outputs, ["sum"]);
    }

    #[test]
    fn keeps_a_rope_whose_intermediate_escapes() {
        for leaked in [
            "direct",
            "shp",
            "d1",
            "pairs_shape",
            "pairs",
            "x1",
            "even",
            "neg",
            "evenu",
            "swapped",
            "lead",
            "rot",
            "rotated",
        ] {
            let mut ir = interleaved_rope(true, true, true);
            ir.outputs.push(leaked.into());
            assert_eq!(fuse_interleaved_rope(&mut ir), 0, "{leaked}");
        }
    }

    /// What ORT hands the EP: the default `start = 0` written out on `Shape`.
    #[test]
    fn fuses_with_the_default_shape_start_written_out() {
        let mut ir = interleaved_rope(true, true, true);
        for shape in ir.nodes.iter_mut().filter(|n| n.op == "Shape") {
            shape.attrs.insert("start".into(), AttrValue::Int(0));
        }
        assert_eq!(fuse_interleaved_rope(&mut ir), 1);
        // a window of the shape is something else
        let mut ir = interleaved_rope(true, true, true);
        let shape = ir.nodes.iter_mut().find(|n| n.outputs == ["shp"]).unwrap();
        shape.attrs.insert("start".into(), AttrValue::Int(1));
        assert_eq!(fuse_interleaved_rope(&mut ir), 0);
    }

    /// `[x0, -x1]` rotates the other way: another operator.
    #[test]
    fn refuses_the_opposite_rotation() {
        let mut ir = interleaved_rope(false, false, false);
        let concat = ir
            .nodes
            .iter_mut()
            .find(|n| n.outputs == ["swapped"])
            .unwrap();
        concat.inputs.reverse();
        assert_eq!(fuse_interleaved_rope(&mut ir), 0);
    }

    #[test]
    fn refuses_pairs_that_are_not_pairs() {
        let mut ir = interleaved_rope(false, false, false);
        ir.initializers.insert("two".into(), int64(vec![1], &[4]));
        assert_eq!(fuse_interleaved_rope(&mut ir), 0);
        // the shape of another tensor than the one being rotated
        let mut ir = interleaved_rope(false, false, false);
        ir.nodes
            .iter_mut()
            .find(|n| n.outputs == ["shp"])
            .unwrap()
            .inputs = vec!["cos".into()];
        assert_eq!(fuse_interleaved_rope(&mut ir), 0);
        // dimension 2 read twice
        let mut ir = interleaved_rope(false, false, false);
        ir.nodes
            .iter_mut()
            .find(|n| n.outputs == ["g1"])
            .unwrap()
            .inputs[1] = "i2".into();
        assert_eq!(fuse_interleaved_rope(&mut ir), 0);
    }

    /// The scale must be a scalar constant to fold; otherwise it stays a `Mul`.
    #[test]
    fn leaves_a_tensor_scale_outside() {
        let mut ir = interleaved_rope(true, false, true);
        ir.initializers.insert("scale".into(), vector(8));
        assert_eq!(fuse_interleaved_rope(&mut ir), 1);
        let ops: Vec<&str> = ir.nodes.iter().map(|n| n.op.as_str()).collect();
        assert_eq!(ops, ["InterleavedRope", "Mul"]);
    }

    /// The same rotation as SAM 3's fp32 dynamo export writes it, on
    /// `[B, H, S, D] = [2, 3, 4, 8]`: constant shapes with `allowzero = 1`,
    /// two `Slice`s, and K's transpose on three dimensions.
    fn exported_rope(bshd: bool, transpose: bool, scale: bool) -> GraphIr {
        let mut nodes = Vec::new();
        if bshd {
            let perm = AttrValue::Ints(vec![0, 2, 1, 3]);
            nodes.push(with_attr(node("Transpose", &["x"], "t"), "perm", perm));
        }
        let allowzero = |n: NodeIr| with_attr(n, "allowzero", AttrValue::Int(1));
        let last = if transpose || scale { "sum" } else { "y" };
        let slice = |inputs: &[&str], output: &str| node("Slice", inputs, output);
        nodes.extend([
            node("Mul", &["t", "cos"], "direct"),
            allowzero(node("Reshape", &["t", "pairs_dims"], "pairs")),
            slice(&["pairs", "zero", "one", "minus1"], "x0"),
            slice(&["pairs", "one", "two", "minus1"], "x1"),
            node("Squeeze", &["x0", "minus1"], "even"),
            node("Squeeze", &["x1", "minus1"], "odd"),
            node("Neg", &["odd"], "neg"),
            node("Unsqueeze", &["neg", "minus1"], "negu"),
            node("Unsqueeze", &["even", "minus1"], "evenu"),
            with_attr(
                node("Concat", &["negu", "evenu"], "swapped"),
                "axis",
                AttrValue::Int(-1),
            ),
            allowzero(node("Reshape", &["swapped", "rot_dims"], "rot")),
            node("Mul", &["rot", "sin"], "rotated"),
            node("Add", &["direct", "rotated"], last),
        ]);
        if transpose {
            let to = if scale { "kt" } else { "y" };
            let perm = AttrValue::Ints(vec![0, 2, 1]);
            nodes.extend([
                node("Reshape", &["sum", "k3_dims"], "k3"),
                with_attr(node("Transpose", &["k3"], "k3t"), "perm", perm),
                node("Reshape", &["k3t", "k4_dims"], to),
            ]);
        }
        if scale {
            let from = if transpose { "kt" } else { "sum" };
            nodes.push(node("Mul", &[from, "scale"], "y"));
        }
        GraphIr {
            nodes,
            initializers: HashMap::from([
                ("zero".into(), int64(vec![1], &[0])),
                ("one".into(), int64(vec![1], &[1])),
                ("two".into(), int64(vec![1], &[2])),
                ("minus1".into(), int64(vec![1], &[-1])),
                ("pairs_dims".into(), int64(vec![5], &[2, 3, 4, -1, 2])),
                ("rot_dims".into(), int64(vec![4], &[2, 3, 4, 8])),
                ("k3_dims".into(), int64(vec![3], &[-1, 4, 8])),
                ("k4_dims".into(), int64(vec![4], &[2, 3, 8, 4])),
                ("scale".into(), scalar(0.25)),
            ]),
            inputs: vec![
                if bshd { "x" } else { "t" }.into(),
                "cos".into(),
                "sin".into(),
            ],
            outputs: vec!["y".into()],
            ..Default::default()
        }
    }

    fn ints_attr<'a>(node: &'a NodeIr, name: &str) -> Option<&'a [i64]> {
        node.attrs.get(name).and_then(AttrValue::as_ints)
    }

    #[test]
    fn fuses_the_exported_rope_of_q() {
        let mut ir = exported_rope(true, false, true);
        assert_eq!(fuse_interleaved_rope(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 1, "{:?}", ir.nodes);
        let fused = &ir.nodes[0];
        assert_eq!(fused.op, "InterleavedRope");
        assert_eq!(fused.inputs, ["x", "cos", "sin", "scale"]);
        assert_eq!(flag(fused, "bshd"), Some(1));
        assert_eq!(flag(fused, "transpose_output"), Some(0));
        assert_eq!(ints_attr(fused, "shape"), Some(&[2, 3, 4, 8][..]));
    }

    /// K's `Reshape [-1, S, D] → Transpose [0, 2, 1] → Reshape [B, H, D, S]`
    /// is its `[0, 1, 3, 2]`: it folds, and so does the scale after it.
    #[test]
    fn fuses_the_exported_rope_of_k_with_its_rank3_transpose() {
        let mut ir = exported_rope(true, true, true);
        assert_eq!(fuse_interleaved_rope(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 1, "{:?}", ir.nodes);
        assert_eq!(flag(&ir.nodes[0], "transpose_output"), Some(1));
        assert_eq!(ir.nodes[0].inputs.len(), 4);
        assert_eq!(ir.nodes[0].outputs, ["y"]);
        // [B·H, S, D] instead of [-1, S, D] is the same reshape
        let mut ir = exported_rope(true, true, false);
        ir.initializers
            .insert("k3_dims".into(), int64(vec![3], &[6, 4, 8]));
        assert_eq!(fuse_interleaved_rope(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 1);
    }

    /// A rank-3 round trip that is not K's transpose stays outside.
    #[test]
    fn leaves_another_rank3_round_trip_outside() {
        for (key, shape, values) in [
            ("k4_dims", vec![4], &[2, 3, 4, 8][..]),
            ("k3_dims", vec![3], &[-1, 8, 4][..]),
        ] {
            let mut ir = exported_rope(true, true, true);
            ir.initializers.insert(key.into(), int64(shape, values));
            assert_eq!(fuse_interleaved_rope(&mut ir), 1, "{key}");
            let ops: Vec<&str> = ir.nodes.iter().map(|n| n.op.as_str()).collect();
            assert_eq!(
                ops,
                ["InterleavedRope", "Reshape", "Transpose", "Reshape", "Mul"],
                "{key}"
            );
            assert_eq!(flag(&ir.nodes[0], "transpose_output"), Some(0));
        }
    }

    #[test]
    fn refuses_exported_shapes_that_disagree() {
        // the rotation restores another [B, H, S]
        let mut ir = exported_rope(false, false, false);
        ir.initializers
            .insert("rot_dims".into(), int64(vec![4], &[2, 3, 2, 16]));
        assert_eq!(fuse_interleaved_rope(&mut ir), 0);
        // a 0 that allowzero would keep as a zero-sized dimension
        let mut ir = exported_rope(false, false, false);
        ir.initializers
            .insert("pairs_dims".into(), int64(vec![5], &[0, 3, 4, -1, 2]));
        assert_eq!(fuse_interleaved_rope(&mut ir), 0);
        // an odd D
        let mut ir = exported_rope(false, false, false);
        ir.initializers
            .insert("rot_dims".into(), int64(vec![4], &[2, 3, 4, 7]));
        assert_eq!(fuse_interleaved_rope(&mut ir), 0);
    }

    /// The halves taken the other way round, or a slice that is not a half.
    #[test]
    fn refuses_exported_halves_that_are_not_halves() {
        let mut ir = exported_rope(false, false, false);
        for n in ir.nodes.iter_mut().filter(|n| n.op == "Slice") {
            n.outputs[0] = if n.outputs[0] == "x0" { "x1" } else { "x0" }.into();
        }
        assert_eq!(fuse_interleaved_rope(&mut ir), 0);
        let mut ir = exported_rope(false, false, false);
        let half = ir.nodes.iter_mut().find(|n| n.outputs == ["x1"]).unwrap();
        half.inputs[2] = "minus1".into();
        assert_eq!(fuse_interleaved_rope(&mut ir), 0);
        // a slice of another axis
        let mut ir = exported_rope(false, false, false);
        let half = ir.nodes.iter_mut().find(|n| n.outputs == ["x0"]).unwrap();
        half.inputs[3] = "zero".into();
        assert_eq!(fuse_interleaved_rope(&mut ir), 0);
    }

    #[test]
    fn keeps_an_exported_rope_whose_intermediate_escapes() {
        for leaked in [
            "direct", "pairs", "x0", "x1", "even", "neg", "evenu", "swapped", "rot", "rotated",
        ] {
            let mut ir = exported_rope(true, true, true);
            ir.outputs.push(leaked.into());
            assert_eq!(fuse_interleaved_rope(&mut ir), 0, "{leaked}");
        }
        // an escaping step of K's transpose keeps the tail outside
        for leaked in ["k3", "k3t"] {
            let mut ir = exported_rope(true, true, true);
            ir.outputs.push(leaked.into());
            assert_eq!(fuse_interleaved_rope(&mut ir), 1, "{leaked}");
            assert_eq!(ir.nodes[0].outputs, ["sum"], "{leaked}");
        }
    }

    /// A pre-norm block: the residual sum is normalized and read again by
    /// the next residual `Add`.
    fn residual_layernorm() -> GraphIr {
        GraphIr {
            nodes: vec![
                node("Add", &["x", "r"], "sum"),
                node("LayerNormalization", &["sum", "gamma", "beta"], "y"),
                node("Add", &["sum", "y"], "next"),
            ],
            initializers: HashMap::from([
                ("gamma".into(), vector(256)),
                ("beta".into(), vector(256)),
            ]),
            inputs: vec!["x".into(), "r".into()],
            outputs: vec!["next".into()],
            ..Default::default()
        }
    }

    #[test]
    fn fuses_the_residual_add_into_the_layernorm() {
        let mut ir = residual_layernorm();
        assert_eq!(fuse_add_layernorm(&mut ir), 1);
        assert_eq!(ir.nodes.len(), 2);
        let fused = &ir.nodes[0];
        assert_eq!(fused.domain, REWRITE_DOMAIN);
        assert_eq!(fused.op, "AddLayerNormalization");
        assert_eq!(fused.inputs, ["x", "r", "gamma", "beta"]);
        // the sum still has a reader, so it is the second output
        assert_eq!(fused.outputs, ["y", "sum"]);

        // read by nothing else, the sum is not written at all
        let mut ir = residual_layernorm();
        ir.nodes[2].inputs[0] = "x".into();
        assert_eq!(fuse_add_layernorm(&mut ir), 1);
        assert_eq!(ir.nodes[0].outputs, ["y"]);

        // a graph output is a reader too
        let mut ir = residual_layernorm();
        ir.nodes[2].inputs[0] = "x".into();
        ir.outputs.push("sum".into());
        assert_eq!(fuse_add_layernorm(&mut ir), 1);
        assert_eq!(ir.nodes[0].outputs, ["y", "sum"]);
    }

    #[test]
    fn leaves_a_sum_read_before_the_layernorm_alone() {
        // the fused node takes the normalization's slot, after this reader
        let mut ir = residual_layernorm();
        ir.nodes.insert(1, node("Relu", &["sum"], "early"));
        assert_eq!(fuse_add_layernorm(&mut ir), 0);

        // a statistics output is not produced by the kernel
        let mut ir = residual_layernorm();
        ir.nodes[1].outputs.push("mean".into());
        assert_eq!(fuse_add_layernorm(&mut ir), 0);
        assert_eq!(ir.nodes.len(), 3);
    }
}
