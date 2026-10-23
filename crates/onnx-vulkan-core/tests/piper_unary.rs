//! Regressions for Piper's unary operators and their shader specializations.

use onnx_vulkan_core::host_ops::HostTensor;
use onnx_vulkan_core::{AttrValue, ExecutionEnv, GraphIr, KernelCache, NodeIr, Tensor, execute};
use std::collections::HashMap;
use vk_compute::VkContext;

fn unary(op: &str, attrs: HashMap<String, AttrValue>) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.into(),
        opset: 17,
        name: format!("{op}_test"),
        inputs: vec!["x".into()],
        outputs: vec!["out".into()],
        attrs,
    }
}

fn run_unary(op: NodeIr, values: &[f32]) -> Vec<f32> {
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let graph = GraphIr {
        nodes: vec![op],
        inputs: vec!["x".into()],
        outputs: vec!["out".into()],
        ..Default::default()
    };
    let mut env = ExecutionEnv::new(&cache, &graph.initializers);
    env.set(
        "x",
        Tensor::Host(HostTensor::from_f32(vec![values.len() as i64], values)),
    );
    execute(&graph, &mut env).expect("unary graph execution");
    let result = env
        .host("out")
        .expect("unary output")
        .to_f32()
        .expect("f32 output");
    env.finish();
    result
}

fn leaky_relu(name: &str, output: &str, alpha: f32) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: "LeakyRelu".into(),
        opset: 17,
        name: name.into(),
        inputs: vec!["x".into()],
        outputs: vec![output.into()],
        attrs: HashMap::from([("alpha".into(), AttrValue::Float(alpha))]),
    }
}

#[test]
fn ceil_handles_fractional_values_on_both_sides_of_zero() {
    let got = run_unary(unary("Ceil", HashMap::new()), &[-1.8, -0.2, 0.2, 1.8]);
    assert_eq!(got, [-1.0, 0.0, 1.0, 2.0]);
}

#[test]
fn softplus_extreme_inputs_remain_finite() {
    let got = run_unary(unary("Softplus", HashMap::new()), &[-1000.0, 1000.0]);
    assert!(got.iter().all(|value| value.is_finite()), "{got:?}");
    assert_eq!(got[0], 0.0);
    assert_eq!(got[1], 1000.0);
}

#[test]
fn softplus_matches_stable_reference_for_moderate_inputs() {
    let input = [-3.0, -1.0, 0.0, 1.0, 3.0];
    let got = run_unary(unary("Softplus", HashMap::new()), &input);
    for (x, actual) in input.into_iter().zip(got) {
        let expected = x.max(0.0) + (-x.abs()).exp().ln_1p();
        assert!(
            (actual - expected).abs() < 2e-6,
            "softplus({x}) = {actual}, expected {expected}"
        );
    }
}

#[test]
fn softplus_preserves_tiny_negative_tail() {
    let got = run_unary(unary("Softplus", HashMap::new()), &[-20.0]);
    let expected = (-20.0f32).exp();
    assert!(got[0] > 0.0, "tiny tail was rounded away: {}", got[0]);
    assert!(
        (got[0] - expected).abs() < expected * 0.02,
        "softplus(-20) = {}, expected about {expected}",
        got[0]
    );
}

#[test]
fn leaky_relu_alpha_variants_use_distinct_pipelines_in_one_session() {
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let graph = GraphIr {
        nodes: vec![
            leaky_relu("leaky_relu_alpha_01", "out_01", 0.1),
            leaky_relu("leaky_relu_alpha_001", "out_001", 0.01),
        ],
        inputs: vec!["x".into()],
        outputs: vec!["out_01".into(), "out_001".into()],
        ..Default::default()
    };
    let mut env = ExecutionEnv::new(&cache, &graph.initializers);
    env.set(
        "x",
        Tensor::Host(HostTensor::from_f32(vec![2], &[-2.0, 3.0])),
    );

    execute(&graph, &mut env).expect("graph execution");

    let out_01 = env
        .host("out_01")
        .expect("alpha=0.1 output")
        .to_f32()
        .expect("f32 output");
    let out_001 = env
        .host("out_001")
        .expect("alpha=0.01 output")
        .to_f32()
        .expect("f32 output");
    assert_eq!(out_01, [-0.2, 3.0]);
    assert_eq!(out_001, [-0.02, 3.0]);
    assert_eq!(cache.builds().0, 2);
    env.finish();
}
