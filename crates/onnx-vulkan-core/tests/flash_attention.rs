//! `MatMul → Softmax → MatMul` attention fused into `FlashAttention` at load
//! time: which graphs the rewrite takes, and that the fused kernel and its
//! unfused fallback both answer like the three nodes they replace.

use onnx_vulkan_core::rewrite::{REWRITE_DOMAIN, fuse_attention};
use onnx_vulkan_core::shaders::flash_attention::key_tile;
use onnx_vulkan_core::{AttrValue, Executor, GraphIr, HostTensor, InitializerIr, NodeIr, Tensor};
use vk_compute::VkContext;

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: 18,
        name: format!("{op}_{}", outputs[0]),
        inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
        outputs: outputs.iter().map(|name| (*name).to_string()).collect(),
        attrs: Default::default(),
    }
}

/// `y = softmax(q · kt) · v`, the shape SAM 3's detector encoder exports.
fn attention_graph() -> GraphIr {
    GraphIr {
        inputs: vec!["q".into(), "kt".into(), "v".into()],
        outputs: vec!["y".into()],
        nodes: vec![
            node("MatMul", &["q", "kt"], &["scores"]),
            node("Softmax", &["scores"], &["probs"]),
            node("MatMul", &["probs", "v"], &["y"]),
        ],
        ..Default::default()
    }
}

#[test]
fn rewrite_fuses_attention_into_one_node() {
    let mut graph = attention_graph();
    assert_eq!(fuse_attention(&mut graph), 1);
    let [fused] = &graph.nodes[..] else {
        panic!("one node expected, got {:?}", graph.nodes);
    };
    assert_eq!(fused.domain, REWRITE_DOMAIN);
    assert_eq!(fused.op, "FlashAttention");
    assert_eq!(fused.inputs, ["q", "kt", "v"]);
    assert_eq!(fused.outputs, ["y"]);
}

#[test]
fn rewrite_leaves_attention_whose_intermediates_escape() {
    // the scores are a graph output
    let mut graph = attention_graph();
    graph.outputs.push("scores".into());
    assert_eq!(fuse_attention(&mut graph), 0);

    // the probabilities have a second reader
    let mut graph = attention_graph();
    graph.nodes.push(node("Relu", &["probs"], &["extra"]));
    graph.outputs.push("extra".into());
    assert_eq!(fuse_attention(&mut graph), 0);

    // `v · p` is not the product the kernel computes
    let mut graph = attention_graph();
    graph.nodes[2].inputs = vec!["v".into(), "probs".into()];
    assert_eq!(fuse_attention(&mut graph), 0);

    // below opset 13 the default axis is 1, not the last one
    let mut graph = attention_graph();
    graph.nodes[1].opset = 11;
    assert_eq!(fuse_attention(&mut graph), 0);
    graph.nodes[1]
        .attrs
        .insert("axis".into(), AttrValue::Int(-1));
    assert_eq!(fuse_attention(&mut graph), 1);

    // a softmax over another axis
    let mut graph = attention_graph();
    graph.nodes[1]
        .attrs
        .insert("axis".into(), AttrValue::Int(1));
    assert_eq!(fuse_attention(&mut graph), 0);
}

/// Deterministic values in `[-scale, scale)`.
fn values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// The reference in f64, broadcasting the batch dimensions as `MatMul` does.
fn reference(q: &HostTensor, kt: &HostTensor, v: &HostTensor) -> Vec<f64> {
    reference_biased(q, kt, v, None, false)
}

/// The reference with `bias` added to the scores, broadcast as `Add` does,
/// and with `nan_to_zero` the probabilities of a row whose denominator is zero
/// (every key at `-inf`) replaced by zeros, as `Where(IsNaN(p), 0, p)` does.
fn reference_biased(
    q: &HostTensor,
    kt: &HostTensor,
    v: &HostTensor,
    bias: Option<&HostTensor>,
    nan_to_zero: bool,
) -> Vec<f64> {
    let (qv, kv, vv) = (
        q.to_f32().unwrap(),
        kt.to_f32().unwrap(),
        v.to_f32().unwrap(),
    );
    let rank = q.shape.len();
    let (m, d) = (q.shape[rank - 2] as usize, q.shape[rank - 1] as usize);
    let n = kt.shape[rank - 1] as usize;
    let dv = v.shape[rank - 1] as usize;
    // the bias with leading ones up to the scores' rank
    let bias_shape: Vec<i64> = bias.map_or(vec![1; rank], |b| {
        let mut shape = vec![1; rank - b.shape.len()];
        shape.extend(&b.shape);
        shape
    });
    let bias_values = bias.map(|b| b.to_f32().unwrap());
    let batch: Vec<usize> = (0..rank - 2)
        .map(|i| {
            q.shape[i]
                .max(kt.shape[i])
                .max(v.shape[i])
                .max(bias_shape[i]) as usize
        })
        .collect();
    // offset of batch index `flat` in a tensor whose batch dims may be 1
    let base = |shape: &[i64], mut flat: usize, matrix: usize| {
        let mut offset = 0;
        let mut stride = matrix;
        for i in (0..rank - 2).rev() {
            let index = flat % batch[i];
            flat /= batch[i];
            if shape[i] != 1 {
                offset += index * stride;
            }
            stride *= shape[i] as usize;
        }
        offset
    };
    let (bias_m, bias_n) = (bias_shape[rank - 2] as usize, bias_shape[rank - 1] as usize);
    let mut out = Vec::new();
    for bh in 0..batch.iter().product() {
        let (qb, kb, vb, bb) = (
            base(&q.shape, bh, m * d),
            base(&kt.shape, bh, d * n),
            base(&v.shape, bh, n * dv),
            base(&bias_shape, bh, bias_m * bias_n),
        );
        for row in 0..m {
            let scores: Vec<f64> = (0..n)
                .map(|j| {
                    let added = bias_values.as_ref().map_or(0.0, |b| {
                        f64::from(b[bb + (row % bias_m) * bias_n + j % bias_n])
                    });
                    (0..d)
                        .map(|k| f64::from(qv[qb + row * d + k]) * f64::from(kv[kb + k * n + j]))
                        .sum::<f64>()
                        + added
                })
                .collect();
            let max = scores.iter().cloned().fold(f64::MIN, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for c in 0..dv {
                let acc: f64 = (0..n)
                    .map(|j| exps[j] * f64::from(vv[vb + j * dv + c]))
                    .sum();
                out.push(if nan_to_zero && sum == 0.0 {
                    0.0
                } else {
                    acc / sum
                });
            }
        }
    }
    out
}

/// Runs the attention graph through `Executor`, whose load-time rewrite fuses
/// it, and checks the result against the f64 reference. `kernel` says whether
/// the fused shader must run, with the key tile this device fits, or the
/// three-step fallback: both answer alike, so only the compiled pipelines tell
/// them apart.
fn check(q_shape: Vec<i64>, kt_shape: Vec<i64>, v_shape: Vec<i64>, kernel: bool) {
    let numel = |shape: &[i64]| shape.iter().product::<i64>() as usize;
    let q = HostTensor::from_f32(q_shape.clone(), &values(numel(&q_shape), 1, 1.0));
    // keys that grow along the sequence move the running maximum from tile to
    // tile, which is the rescaling the online softmax must get right
    let n = *kt_shape.last().unwrap() as usize;
    let keys: Vec<f32> = values(numel(&kt_shape), 2, 1.0)
        .iter()
        .enumerate()
        .map(|(i, x)| x + (i % n) as f32 * 0.02)
        .collect();
    let kt = HostTensor::from_f32(kt_shape.clone(), &keys);
    let v = HostTensor::from_f32(v_shape.clone(), &values(numel(&v_shape), 3, 2.0));
    let expected = reference(&q, &kt, &v);

    let context = VkContext::new().expect("Vulkan context");
    let executor = Executor::new(&context, attention_graph()).expect("executor");
    assert_eq!(
        executor
            .graph()
            .nodes
            .iter()
            .map(|n| n.op.as_str())
            .collect::<Vec<_>>(),
        ["FlashAttention"]
    );
    let outputs = executor
        .run(vec![
            ("q", Tensor::Host(q)),
            ("kt", Tensor::Host(kt)),
            ("v", Tensor::Host(v)),
        ])
        .expect("run");
    let y = outputs.host("y").expect("output");
    let mut out_shape = q_shape.clone();
    for (i, dim) in out_shape.iter_mut().enumerate().take(q_shape.len() - 2) {
        *dim = (*dim).max(kt_shape[i]).max(v_shape[i]);
    }
    *out_shape.last_mut().unwrap() = *v_shape.last().unwrap();
    assert_eq!(y.shape, out_shape);
    let got = y.to_f32().unwrap();
    assert_eq!(got.len(), expected.len());
    let worst = got
        .iter()
        .zip(&expected)
        .map(|(g, e)| (f64::from(*g) - e).abs())
        .fold(0.0, f64::max);
    assert!(worst < 2e-5, "max |Δ| {worst:e}");
    outputs.finish();

    let compiled: Vec<String> = executor
        .cache()
        .pipeline_keys()
        .iter()
        .filter(|key| key.operation() == "FlashAttention")
        .filter_map(|key| key.tactic_id().map(str::to_owned))
        .collect();
    let expected: Vec<String> = if kernel {
        let (d, dv) = (q_shape[q_shape.len() - 1], v_shape[v_shape.len() - 1]);
        let (d, dv) = (d as u32, dv as u32);
        let bc = key_tile(d, dv, context.compute_limits().max_shared_memory_bytes)
            .expect("the device fits the smallest key tile");
        vec![format!("d{d}_dv{dv}_bc{bc}")]
    } else {
        Vec::new()
    };
    assert_eq!(compiled, expected);
}

#[test]
fn fused_kernel_matches_reference_on_partial_tiles() {
    // neither 100 queries nor 77 keys is a multiple of the 64-wide tiles
    check(
        vec![2, 3, 100, 32],
        vec![2, 3, 32, 77],
        vec![2, 3, 77, 32],
        true,
    );
}

#[test]
fn fused_kernel_with_sam3_vision_heads() {
    // d = dv = 64, SAM 3's vision encoder: 64-key tiles exceed 48 KiB, so on
    // the RTX 4070 this runs 32-key tiles, two full and a tail of 13
    check(
        vec![2, 2, 100, 64],
        vec![2, 2, 64, 77],
        vec![2, 2, 77, 64],
        true,
    );
}

#[test]
fn fused_kernel_with_odd_head_sizes() {
    check(vec![1, 130, 5], vec![1, 5, 200], vec![1, 200, 16], true);
    check(vec![70, 64], vec![64, 3], vec![3, 64], true);
}

#[test]
fn shapes_outside_the_kernel_fall_back_to_three_steps() {
    // broadcast batch
    check(vec![2, 3, 10, 8], vec![1, 3, 8, 9], vec![2, 1, 9, 8], false);
    // head larger than the registers the shader holds
    check(
        vec![1, 2, 20, 80],
        vec![1, 2, 80, 30],
        vec![1, 2, 30, 16],
        false,
    );
    // value head not a whole number of vec4s per thread
    check(
        vec![1, 2, 20, 16],
        vec![1, 2, 16, 30],
        vec![1, 2, 30, 8],
        false,
    );
}

/// Multi-head attention as exporters write it around the three nodes: values
/// `[B, n, H, DV]` made `[B, H, n, DV]`, and the result made `[B, m, H, DV]`,
/// both by `Transpose(·, [0, 2, 1, 3])`.
fn multi_head_graph() -> GraphIr {
    let heads_swap = |input: &str, output: &str| {
        let mut n = node("Transpose", &[input], &[output]);
        n.attrs
            .insert("perm".into(), AttrValue::Ints(vec![0, 2, 1, 3]));
        n
    };
    GraphIr {
        inputs: vec!["q".into(), "kt".into(), "v_bshd".into()],
        outputs: vec!["y".into()],
        nodes: vec![
            heads_swap("v_bshd", "v"),
            node("MatMul", &["q", "kt"], &["scores"]),
            node("Softmax", &["scores"], &["probs"]),
            node("MatMul", &["probs", "v"], &["heads"]),
            heads_swap("heads", "y"),
        ],
        ..Default::default()
    }
}

fn flag(node: &NodeIr, key: &str) -> Option<i64> {
    node.attrs.get(key).and_then(AttrValue::as_i64)
}

#[test]
fn rewrite_folds_the_head_transposes() {
    let mut graph = multi_head_graph();
    assert_eq!(fuse_attention(&mut graph), 1);
    let [fused] = &graph.nodes[..] else {
        panic!("one node expected, got {:?}", graph.nodes);
    };
    assert_eq!(fused.inputs, ["q", "kt", "v_bshd"]);
    assert_eq!(fused.outputs, ["y"]);
    assert_eq!(flag(fused, "v_bshd"), Some(1));
    assert_eq!(flag(fused, "out_bshd"), Some(1));

    // a plain attention carries no layout attribute at all
    let mut graph = attention_graph();
    fuse_attention(&mut graph);
    assert!(graph.nodes[0].attrs.is_empty());
}

#[test]
fn rewrite_keeps_head_transposes_seen_elsewhere() {
    // the transposed values are a graph output, so is the attention's result
    let mut graph = multi_head_graph();
    graph.outputs.extend(["v".into(), "heads".into()]);
    assert_eq!(fuse_attention(&mut graph), 1);
    let ops: Vec<&str> = graph.nodes.iter().map(|n| n.op.as_str()).collect();
    assert_eq!(ops, ["Transpose", "FlashAttention", "Transpose"]);
    let fused = &graph.nodes[1];
    assert_eq!(fused.inputs, ["q", "kt", "v"]);
    assert_eq!(fused.outputs, ["heads"]);
    assert!(fused.attrs.is_empty());

    // another reader of the transposed values
    let mut graph = multi_head_graph();
    graph.nodes.push(node("Relu", &["v"], &["extra"]));
    graph.outputs.push("extra".into());
    assert_eq!(fuse_attention(&mut graph), 1);
    let fused = graph
        .nodes
        .iter()
        .find(|n| n.op == "FlashAttention")
        .unwrap();
    assert_eq!(fused.inputs[2], "v");
    assert_eq!(flag(fused, "v_bshd"), None);
    assert_eq!(flag(fused, "out_bshd"), Some(1));

    // another permutation is another layout
    let mut graph = multi_head_graph();
    for transpose in graph.nodes.iter_mut().filter(|n| n.op == "Transpose") {
        transpose
            .attrs
            .insert("perm".into(), AttrValue::Ints(vec![0, 1, 3, 2]));
    }
    assert_eq!(fuse_attention(&mut graph), 1);
    let fused = graph
        .nodes
        .iter()
        .find(|n| n.op == "FlashAttention")
        .unwrap();
    assert!(fused.attrs.is_empty());
}

/// Bits of `y` and whether the fused shader ran.
fn run_multi_head(
    context: &VkContext,
    graph: GraphIr,
    feeds: &[(&str, &HostTensor)],
) -> (Vec<i64>, Vec<u32>, bool) {
    let executor = Executor::new(context, graph).expect("executor");
    let outputs = executor
        .run(
            feeds
                .iter()
                .map(|(name, t)| (*name, Tensor::Host((*t).clone())))
                .collect(),
        )
        .expect("run");
    let y = outputs.host("y").expect("output");
    let shape = y.shape.clone();
    let bits = y.to_f32().unwrap().iter().map(|x| x.to_bits()).collect();
    outputs.finish();
    let kernel = executor
        .cache()
        .pipeline_keys()
        .iter()
        .any(|key| key.operation() == "FlashAttention");
    (shape, bits, kernel)
}

/// Folded transposes against the same graph with both transposes kept, bit
/// for bit: the arithmetic is the same, only the addresses differ.
fn check_multi_head(b: i64, h: i64, m: i64, n: i64, d: i64, dv: i64, kernel: bool) {
    let numel = |shape: &[i64]| shape.iter().product::<i64>() as usize;
    let q_shape = [b, h, m, d];
    let kt_shape = [b, h, d, n];
    let v_shape = [b, n, h, dv];
    let q = HostTensor::from_f32(q_shape.to_vec(), &values(numel(&q_shape), 11, 1.0));
    let kt = HostTensor::from_f32(kt_shape.to_vec(), &values(numel(&kt_shape), 12, 1.0));
    let v = HostTensor::from_f32(v_shape.to_vec(), &values(numel(&v_shape), 13, 2.0));
    let feeds = [("q", &q), ("kt", &kt), ("v_bshd", &v)];
    let context = VkContext::new().expect("Vulkan context");

    let mut reference = multi_head_graph();
    reference.outputs.extend(["v".into(), "heads".into()]);
    let (want_shape, want, want_kernel) = run_multi_head(&context, reference, &feeds);
    let (shape, got, got_kernel) = run_multi_head(&context, multi_head_graph(), &feeds);
    assert_eq!(shape, [b, m, h, dv]);
    assert_eq!(shape, want_shape);
    assert_eq!((got_kernel, want_kernel), (kernel, kernel));
    let differing = got.iter().zip(&want).filter(|(g, w)| g != w).count();
    assert_eq!(differing, 0, "{differing} of {} elements differ", got.len());
}

#[test]
fn folded_head_transposes_match_the_explicit_ones() {
    // SAM 3's window attention heads, 1.2M outputs
    check_multi_head(2, 16, 576, 576, 64, 64, true);
    // partial query and key tiles, three heads
    check_multi_head(2, 3, 100, 77, 32, 32, true);
}

#[test]
fn folded_head_transposes_fall_back_to_the_original_steps() {
    // a value head the shader does not take: the fallback runs the transposes
    check_multi_head(1, 2, 20, 30, 16, 8, false);
}

/// `y = softmax(q · kt + bias) · v`, with the cleanup exporters write after a
/// masked softmax, `Where(IsNaN(p), 0, p)`, when `nan_cleanup`.
fn biased_graph(nan_cleanup: bool) -> GraphIr {
    let mut nodes = vec![
        node("MatMul", &["q", "kt"], &["scores"]),
        node("Add", &["scores", "bias"], &["biased"]),
        node("Softmax", &["biased"], &["probs"]),
    ];
    if nan_cleanup {
        nodes.push(node("IsNaN", &["probs"], &["is_nan"]));
        nodes.push(node("Where", &["is_nan", "zero", "probs"], &["cleaned"]));
        nodes.push(node("MatMul", &["cleaned", "v"], &["y"]));
    } else {
        nodes.push(node("MatMul", &["probs", "v"], &["y"]));
    }
    let zero = InitializerIr {
        dtype: 1,
        shape: vec![],
        data: 0f32.to_le_bytes().to_vec(),
    };
    GraphIr {
        inputs: vec!["q".into(), "kt".into(), "v".into(), "bias".into()],
        outputs: vec!["y".into()],
        nodes,
        initializers: [("zero".to_owned(), zero)].into(),
        ..Default::default()
    }
}

#[test]
fn rewrite_folds_the_bias_and_the_nan_cleanup() {
    for nan_cleanup in [false, true] {
        let mut graph = biased_graph(nan_cleanup);
        assert_eq!(fuse_attention(&mut graph), 1);
        let [fused] = &graph.nodes[..] else {
            panic!("one node expected, got {:?}", graph.nodes);
        };
        assert_eq!(fused.inputs, ["q", "kt", "v", "bias"]);
        assert_eq!(fused.outputs, ["y"]);
        let expected = nan_cleanup.then_some(1);
        assert_eq!(flag(fused, "nan_to_zero"), expected);
    }

    // the bias on the left of the Add
    let mut graph = biased_graph(false);
    graph.nodes[1].inputs = vec!["bias".into(), "scores".into()];
    assert_eq!(fuse_attention(&mut graph), 1);
    assert_eq!(graph.nodes[0].inputs[3], "bias");
}

#[test]
fn rewrite_leaves_bias_and_cleanup_it_cannot_fold() {
    // the biased scores are a graph output
    let mut graph = biased_graph(false);
    graph.outputs.push("biased".into());
    assert_eq!(fuse_attention(&mut graph), 0);

    // the scores added to themselves
    let mut graph = biased_graph(false);
    graph.nodes[1].inputs = vec!["scores".into(), "scores".into()];
    assert_eq!(fuse_attention(&mut graph), 0);

    // a cleanup that writes something other than zero
    let mut graph = biased_graph(true);
    graph.initializers.get_mut("zero").unwrap().data = 1f32.to_le_bytes().to_vec();
    assert_eq!(fuse_attention(&mut graph), 0);

    // -0.0 is not the zero `Where` writes: the bits must match too
    let mut graph = biased_graph(true);
    graph.initializers.get_mut("zero").unwrap().data = (-0f32).to_le_bytes().to_vec();
    assert_eq!(fuse_attention(&mut graph), 0);

    // the NaN mask has another reader
    let mut graph = biased_graph(true);
    graph.outputs.push("is_nan".into());
    assert_eq!(fuse_attention(&mut graph), 0);

    // the cleanup's operands swapped: `Where(IsNaN(p), p, 0)`
    let mut graph = biased_graph(true);
    graph.nodes[4].inputs = vec!["is_nan".into(), "probs".into(), "zero".into()];
    assert_eq!(fuse_attention(&mut graph), 0);
}

/// `biased_graph(nan_cleanup)` fused against the f64 reference, and the same
/// graph kept unfused (the biased scores made an output) against it too: the
/// reference is what the original nodes compute. `kernel` says whether the
/// fused shader must run or the fallback. `masked` sets rows 0 and 5 of the
/// bias to `-inf` and every third key to `-3.4e38`, as an additive mask does.
fn check_biased(shapes: [Vec<i64>; 4], nan_cleanup: bool, masked: bool, kernel: bool) {
    let [q_shape, kt_shape, v_shape, bias_shape] = shapes;
    let numel = |shape: &[i64]| shape.iter().product::<i64>() as usize;
    let rank = bias_shape.len();
    let (bias_m, n) = if rank >= 2 {
        (bias_shape[rank - 2] as usize, bias_shape[rank - 1] as usize)
    } else {
        (1, bias_shape[0] as usize)
    };
    let mut bias = values(numel(&bias_shape), 4, 2.0);
    if masked {
        for (i, x) in bias.iter_mut().enumerate() {
            let (row, key) = ((i / n) % bias_m, i % n);
            if row == 0 || row == 5 {
                *x = f32::NEG_INFINITY;
            } else if key % 3 == 0 {
                *x = f32::MIN;
            }
        }
    }
    let tensors = [
        HostTensor::from_f32(q_shape.clone(), &values(numel(&q_shape), 1, 1.0)),
        HostTensor::from_f32(kt_shape.clone(), &values(numel(&kt_shape), 2, 1.0)),
        HostTensor::from_f32(v_shape.clone(), &values(numel(&v_shape), 3, 2.0)),
        HostTensor::from_f32(bias_shape.clone(), &bias),
    ];
    let expected = reference_biased(
        &tensors[0],
        &tensors[1],
        &tensors[2],
        Some(&tensors[3]),
        nan_cleanup,
    );
    let context = VkContext::new().expect("Vulkan context");
    for fused in [true, false] {
        let mut graph = biased_graph(nan_cleanup);
        if !fused {
            graph.outputs.push("biased".into());
        }
        let executor = Executor::new(&context, graph).expect("executor");
        let outputs = executor
            .run(
                ["q", "kt", "v", "bias"]
                    .into_iter()
                    .zip(&tensors)
                    .map(|(name, t)| (name, Tensor::Host(t.clone())))
                    .collect(),
            )
            .expect("run");
        let got = outputs.host("y").expect("output").to_f32().unwrap();
        outputs.finish();
        assert_eq!(got.len(), expected.len());
        // `f64::max` skips NaN: a NaN output would pass the bound below
        let non_finite = got.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(non_finite, 0, "fused {fused}: non-finite outputs");
        let worst = got
            .iter()
            .zip(&expected)
            .map(|(g, e)| (f64::from(*g) - e).abs())
            .fold(0.0, f64::max);
        assert!(worst < 2e-5, "fused {fused}: max |Δ| {worst:e}");
        let compiled: Vec<String> = executor
            .cache()
            .pipeline_keys()
            .iter()
            .filter(|key| key.operation() == "FlashAttention")
            .filter_map(|key| key.tactic_id().map(str::to_owned))
            .collect();
        assert_eq!(compiled.len(), usize::from(fused && kernel), "{compiled:?}");
        assert!(compiled.iter().all(|tactic| tactic.ends_with("_bias")));
    }
}

#[test]
fn fused_kernel_adds_a_full_bias() {
    // a position bias per head and query, SAM 3's detector query attention
    check_biased(
        [
            vec![2, 3, 100, 32],
            vec![2, 3, 32, 77],
            vec![2, 3, 77, 32],
            vec![2, 3, 100, 77],
        ],
        false,
        false,
        true,
    );
}

#[test]
fn fused_kernel_broadcasts_a_masking_bias_and_zeroes_masked_rows() {
    // one mask for every head, as SAM 3's text cross-attention writes it, with
    // two rows masked entirely: NaN probabilities the cleanup turns to zeros
    check_biased(
        [
            vec![2, 3, 100, 32],
            vec![2, 3, 32, 77],
            vec![2, 3, 77, 32],
            vec![1, 1, 100, 77],
        ],
        true,
        true,
        true,
    );
    // one bias row for every query and head, rank 1
    check_biased(
        [
            vec![2, 3, 100, 32],
            vec![2, 3, 32, 77],
            vec![2, 3, 77, 32],
            vec![77],
        ],
        false,
        false,
        true,
    );
}

#[test]
fn bias_the_kernel_cannot_address_falls_back() {
    // batch dimensions before the heads broadcast in part: [2, 2] against
    // [2, 1], which the kernel's single outer stride cannot express
    check_biased(
        [
            vec![2, 2, 3, 20, 16],
            vec![2, 2, 3, 16, 30],
            vec![2, 2, 3, 30, 16],
            vec![2, 1, 1, 20, 30],
        ],
        true,
        true,
        false,
    );
}
