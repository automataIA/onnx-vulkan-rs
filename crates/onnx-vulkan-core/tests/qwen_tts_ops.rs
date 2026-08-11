//! Numeric coverage for the ONNX operators added for Qwen3-TTS codecs.

use onnx_vulkan_core::host_ops::{self, CmpOp, HostTensor};
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, KernelCache, NodeIr, Tensor, execute, is_implemented_node,
};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(op: &str, inputs: &[&str], attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: 18,
        name: format!("{op}_qwen_test"),
        inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
        outputs: vec!["out".to_string()],
        attrs: attrs
            .iter()
            .map(|(name, value)| ((*name).to_string(), value.clone()))
            .collect(),
    }
}

fn run_unary(op: NodeIr, shape: Vec<i64>, values: &[f32]) -> (Vec<i64>, Vec<f32>) {
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let graph = GraphIr {
        nodes: vec![op],
        inputs: vec!["x".into()],
        outputs: vec!["out".into()],
        ..Default::default()
    };
    let mut env = ExecutionEnv::new(&cache, &graph.initializers);
    env.set("x", Tensor::Host(HostTensor::from_f32(shape, values)));
    execute(&graph, &mut env).expect("graph execution");
    let output = env.host("out").expect("host output").clone();
    env.finish();
    let values = output.to_f32().expect("f32 output");
    (output.shape, values)
}

#[test]
fn gather_nd_copies_slices_and_normalizes_negative_indices() {
    let data = HostTensor::from_f32(
        vec![2, 3, 2],
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    );
    let indices = HostTensor::from_i64(vec![3, 2], &[0, 1, 1, 2, -1, 0]);
    let output = host_ops::gather_nd(&data, &indices).expect("GatherND");
    assert_eq!(output.shape, vec![3, 2]);
    assert_eq!(
        output.to_f32().expect("f32 output"),
        vec![2.0, 3.0, 10.0, 11.0, 6.0, 7.0]
    );
}

#[test]
fn isnan_and_less_or_equal_produce_boolean_tensors() {
    let values = HostTensor::from_f32(vec![4], &[f32::NAN, 1.0, f32::INFINITY, -2.0]);
    let nan = host_ops::is_nan(&values).expect("IsNaN");
    assert_eq!(nan.data, vec![1, 0, 0, 0]);

    let left = HostTensor::from_f32(vec![2, 1], &[1.0, 3.0]);
    let right = HostTensor::from_f32(vec![2], &[1.0, 2.0]);
    let compared = host_ops::compare(&left, &right, CmpOp::LessOrEqual).expect("LessOrEqual");
    assert_eq!(compared.shape, vec![2, 2]);
    assert_eq!(compared.data, vec![1, 1, 0, 0]);
}

#[test]
fn elu_honors_alpha_on_the_gpu() {
    let op = node("Elu", &["x"], &[("alpha", AttrValue::Float(0.5))]);
    let (_, got) = run_unary(op, vec![4], &[-2.0, -0.0, 1.0, 3.0]);
    let want = [0.5 * ((-2.0f32).exp() - 1.0), -0.0, 1.0, 3.0];
    for (actual, expected) in got.iter().zip(want) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
}

#[test]
fn argmin_uses_first_index_for_ties_on_the_gpu() {
    let mut values = vec![3.0f32; 4096];
    values[700] = -4.0;
    values[3000] = -4.0;
    let op = node(
        "ArgMin",
        &["x"],
        &[("axis", AttrValue::Int(0)), ("keepdims", AttrValue::Int(0))],
    );
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let graph = GraphIr {
        nodes: vec![op],
        inputs: vec!["x".into()],
        outputs: vec!["out".into()],
        initializers: HashMap::new(),
        ..Default::default()
    };
    let mut env = ExecutionEnv::new(&cache, &graph.initializers);
    env.set("x", Tensor::Host(HostTensor::from_f32(vec![4096], &values)));
    execute(&graph, &mut env).expect("ArgMin execution");
    assert_eq!(
        env.host("out").expect("output").to_i64().expect("indices"),
        vec![700]
    );
    env.finish();
}

#[test]
fn unsupported_gather_nd_batching_and_last_tie_break_are_refused() {
    let batched = node(
        "GatherND",
        &["x", "indices"],
        &[("batch_dims", AttrValue::Int(1))],
    );
    assert!(!is_implemented_node(&batched));
    let last = node(
        "ArgMin",
        &["x"],
        &[("select_last_index", AttrValue::Int(1))],
    );
    assert!(!is_implemented_node(&last));
}
