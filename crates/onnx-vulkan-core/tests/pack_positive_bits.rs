//! `numpy.packbits(x > 0, axis = -1)` written in standard operators, fused
//! into `PackPositiveBits` at load time, must answer **bit for bit** like the
//! nodes it replaces and like `numpy.packbits` itself — on the GPU and on the
//! host path. The reduction axes are the attribute: both frontends fold an
//! opset-18 `axes` input into it before the rewrite runs.
//!
//! The reference is the same graph with the `Greater` output declared as a
//! graph output: an intermediate that escapes cannot be fused, so the rewrite
//! leaves that graph alone and it runs the original kernels.

use onnx_vulkan_core::host_ops::{FLOAT, INT64};
use onnx_vulkan_core::{AttrValue, Executor, GraphIr, HostTensor, InitializerIr, NodeIr, Tensor};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: 18,
        name: format!("{op}_{}", outputs[0]),
        inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
        outputs: outputs.iter().map(|name| (*name).to_string()).collect(),
        attrs: attrs
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect(),
    }
}

fn int64(values: &[i64]) -> InitializerIr {
    InitializerIr {
        dtype: INT64,
        shape: vec![values.len() as i64],
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn float(shape: Vec<i64>, values: &[f32]) -> InitializerIr {
    InitializerIr {
        dtype: FLOAT,
        shape,
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// The tail `scripts/pack_mask_bits.py` appends, over `x` of `shape`.
fn pack_graph(shape: &[i64], axis: i64) -> GraphIr {
    let (last, lead) = shape.split_last().expect("rank ≥ 1");
    let mut grouped = lead.to_vec();
    grouped.extend([last / 8, 8]);
    let initializers = HashMap::from([
        ("grouped".to_string(), int64(&grouped)),
        ("zero".to_string(), float(vec![], &[0.0])),
        (
            "weights".to_string(),
            float(vec![8], &[128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0]),
        ),
    ]);
    GraphIr {
        nodes: vec![
            node("Reshape", &["x", "grouped"], &["groups"], &[]),
            node("Greater", &["groups", "zero"], &["positive"], &[]),
            node(
                "Cast",
                &["positive"],
                &["ones"],
                &[("to", AttrValue::Int(i64::from(FLOAT)))],
            ),
            node("Mul", &["ones", "weights"], &["weighted"], &[]),
            node(
                "ReduceSum",
                &["weighted"],
                &["sum"],
                &[
                    ("axes", AttrValue::Ints(vec![axis])),
                    ("keepdims", AttrValue::Int(0)),
                ],
            ),
            node(
                "Cast",
                &["sum"],
                &["bits"],
                &[("to", AttrValue::Int(2))], // UINT8
            ),
        ],
        initializers,
        inputs: vec!["x".into()],
        outputs: vec!["bits".into()],
        ..Default::default()
    }
}

struct Run {
    bytes: Vec<u8>,
    shape: Vec<i64>,
    ops: Vec<String>,
    fused_kernel: bool,
}

fn run(context: &VkContext, graph: GraphIr, shape: &[i64], x: &[f32]) -> Run {
    let executor = Executor::new(context, graph).expect("executor");
    let ops = executor
        .graph()
        .nodes
        .iter()
        .map(|n| n.op.clone())
        .collect();
    let feed = HostTensor::from_f32(shape.to_vec(), x);
    let outputs = executor.run(vec![("x", Tensor::Host(feed))]).expect("run");
    let bits = outputs.host("bits").expect("output");
    assert_eq!(bits.dtype, 2, "u8 output");
    let (bytes, shape) = (bits.data.clone(), bits.shape.clone());
    outputs.finish();
    let fused_kernel = executor
        .cache()
        .pipeline_keys()
        .iter()
        .any(|key| key.operation() == "PackPositiveBits");
    Run {
        bytes,
        shape,
        ops,
        fused_kernel,
    }
}

/// Arbitrary bit patterns (both signs, ±0, subnormals, ±inf, NaNs) mixed with
/// activations of a realistic magnitude, with both zeros planted.
fn activations(n: usize) -> Vec<f32> {
    let mut state = n as u64;
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 32) as u32;
            match i % 7 {
                0 => 0.0,
                1 => -0.0,
                2 | 3 => f32::from_bits(bits),
                _ => (bits as f32 / u32::MAX as f32 * 2.0 - 1.0) * 6.0,
            }
        })
        .collect()
}

/// `numpy.packbits(x > 0, axis = -1)` for a last axis that is a multiple of 8.
fn packbits(x: &[f32]) -> Vec<u8> {
    x.chunks_exact(8)
        .map(|group| {
            group
                .iter()
                .enumerate()
                .fold(0, |byte, (k, &v)| byte | (u8::from(v > 0.0) << (7 - k)))
        })
        .collect()
}

/// `axis` is the reduction axis as written: `-1` or the rank of `shape`.
fn check(shape: &[i64], axis: i64, kernel: bool) {
    let x = activations(shape.iter().product::<i64>() as usize);
    let context = VkContext::new().expect("Vulkan context");

    let mut reference = pack_graph(shape, axis);
    reference.outputs.push("positive".into());
    let expected = run(&context, reference, shape, &x);
    assert!(expected.ops.iter().any(|op| op == "Greater"));
    assert!(!expected.fused_kernel);

    let got = run(&context, pack_graph(shape, axis), shape, &x);
    assert_eq!(got.ops, ["PackPositiveBits"], "{shape:?}");
    assert_eq!(got.fused_kernel, kernel, "{shape:?}");
    let mut packed = shape.to_vec();
    *packed.last_mut().unwrap() /= 8;
    assert_eq!(got.shape, packed);
    assert_eq!(expected.shape, packed);
    assert_eq!(got.bytes, expected.bytes, "fused vs unfused, {shape:?}");
    assert_eq!(
        got.bytes,
        packbits(&x),
        "fused vs numpy.packbits, {shape:?}"
    );
}

#[test]
fn on_the_gpu_the_bytes_are_those_of_the_nodes_and_of_packbits() {
    // 1440 bytes, whole words
    check(&[1, 3, 40, 96], -1, true);
    // 625 bytes: the last word is partial; the axis written positive
    check(&[1, 1, 5, 1000], 4, true);
}

#[test]
fn a_small_host_input_is_packed_on_the_host_alike() {
    // 240 floats, below the upload threshold
    check(&[1, 2, 3, 40], -1, false);
}

#[test]
fn other_weights_or_a_kept_axis_are_left_alone() {
    let context = VkContext::new().expect("Vulkan context");
    let mut reversed = pack_graph(&[1, 2, 3, 40], -1);
    reversed.initializers.insert(
        "weights".into(),
        float(vec![8], &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]),
    );
    let mut kept = pack_graph(&[1, 2, 3, 40], -1);
    kept.nodes[4]
        .attrs
        .insert("keepdims".into(), AttrValue::Int(1));
    for graph in [reversed, kept] {
        let executor = Executor::new(&context, graph).expect("executor");
        assert!(
            executor
                .graph()
                .nodes
                .iter()
                .all(|n| n.op != "PackPositiveBits"),
            "not numpy.packbits: must not be fused"
        );
    }
}
