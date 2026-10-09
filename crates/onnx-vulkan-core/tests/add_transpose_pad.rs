//! `Add → Reshape → Transpose → Pad`, fused into `AddTransposePad` at load
//! time, must answer **bit for bit** like the four nodes it replaces — the
//! one-pass kernel when the shapes allow it, the original steps when not.
//!
//! The reference is the same graph with the sum declared as a graph output:
//! an intermediate that escapes cannot be fused, so the rewrite leaves that
//! graph alone and it runs the original kernels.

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

struct Case {
    a: Vec<i64>,
    b: Vec<i64>,
    target: Vec<i64>,
    perm: Vec<i64>,
    pads: Vec<i64>,
    value: Option<f32>,
}

fn graph(case: &Case) -> GraphIr {
    let mut initializers = HashMap::from([
        ("target".to_string(), int64(&case.target)),
        ("pads".to_string(), int64(&case.pads)),
    ]);
    let mut pad_inputs = vec!["transposed", "pads"];
    if let Some(value) = case.value {
        initializers.insert(
            "value".to_string(),
            InitializerIr {
                dtype: FLOAT,
                shape: vec![],
                data: value.to_le_bytes().to_vec(),
            },
        );
        pad_inputs.push("value");
    }
    GraphIr {
        nodes: vec![
            node("Add", &["a", "b"], &["sum"], &[]),
            node(
                "Reshape",
                &["sum", "target"],
                &["reshaped"],
                &[("allowzero", AttrValue::Int(1))],
            ),
            node(
                "Transpose",
                &["reshaped"],
                &["transposed"],
                &[("perm", AttrValue::Ints(case.perm.clone()))],
            ),
            node(
                "Pad",
                &pad_inputs,
                &["y"],
                &[("mode", AttrValue::String("constant".into()))],
            ),
        ],
        initializers,
        inputs: vec!["a".into(), "b".into()],
        outputs: vec!["y".into()],
        ..Default::default()
    }
}

struct Run {
    bits: Vec<u32>,
    shape: Vec<i64>,
    ops: Vec<String>,
    fused_kernel: bool,
}

fn run(context: &VkContext, graph: GraphIr, feeds: &[(&str, &[i64], &[f32])]) -> Run {
    let executor = Executor::new(context, graph).expect("executor");
    let ops = executor
        .graph()
        .nodes
        .iter()
        .map(|n| n.op.clone())
        .collect();
    let feeds = feeds
        .iter()
        .map(|(name, shape, x)| (*name, Tensor::Host(HostTensor::from_f32(shape.to_vec(), x))))
        .collect();
    let outputs = executor.run(feeds).expect("run");
    let y = outputs.host("y").expect("output");
    let bits = y
        .to_f32()
        .expect("f32")
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let shape = y.shape.clone();
    outputs.finish();
    let fused_kernel = executor
        .cache()
        .pipeline_keys()
        .iter()
        .any(|key| key.operation() == "AddTransposePad");
    Run {
        bits,
        shape,
        ops,
        fused_kernel,
    }
}

/// Arbitrary bit patterns (both signs, ±0, subnormals, ±inf, NaNs) mixed with
/// activations of a realistic magnitude.
fn activations(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 32) as u32;
            match i % 5 {
                0 => f32::from_bits(bits),
                _ => (bits as f32 / u32::MAX as f32 * 2.0 - 1.0) * 6.0,
            }
        })
        .collect()
}

fn check(case: &Case, kernel: bool) {
    let a = activations(case.a.iter().product::<i64>() as usize, 3);
    let b = activations(case.b.iter().product::<i64>() as usize, 7);
    let feeds: [(&str, &[i64], &[f32]); 2] = [("a", &case.a, &a), ("b", &case.b, &b)];
    let context = VkContext::new().expect("Vulkan context");

    let mut reference = graph(case);
    reference.outputs.push("sum".into());
    let expected = run(&context, reference, &feeds);
    assert_eq!(expected.ops, ["Add", "Reshape", "Transpose", "Pad"]);
    assert!(!expected.fused_kernel);

    let got = run(&context, graph(case), &feeds);
    assert_eq!(got.ops, ["AddTransposePad"]);
    assert_eq!(got.fused_kernel, kernel, "{:?}", case.pads);
    assert_eq!(got.shape, expected.shape);
    assert!(
        got.bits == expected.bits,
        "fused vs unfused, {:?}",
        case.pads
    );
}

/// SAM 3's box relative-position bias at a tenth of the queries and a sixth
/// of the grid: rows plus columns, merged to keys, heads first, one zero row.
fn sam3_bias(pads: Vec<i64>, value: Option<f32>) -> Case {
    Case {
        a: vec![1, 20, 12, 1, 8],
        b: vec![1, 20, 1, 12, 8],
        target: vec![1, 20, 144, 8],
        perm: vec![0, 3, 1, 2],
        pads,
        value,
    }
}

#[test]
fn the_relative_position_bias_is_written_in_one_pass_alike() {
    check(&sam3_bias(vec![0, 0, 1, 0, 0, 0, 0, 0], None), true);
}

#[test]
fn any_pad_value_and_both_ends_on_unmerged_axes_alike() {
    check(&sam3_bias(vec![0, 2, 1, 0, 0, 1, 3, 0], Some(-2.5)), true);
}

#[test]
fn a_pad_on_a_merged_axis_runs_the_original_steps_alike() {
    check(&sam3_bias(vec![0, 0, 0, 1, 0, 0, 0, 0], None), false);
}
