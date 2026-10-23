//! GPU distribution and stream-contract checks for RandomNormalLike.

use onnx_vulkan_core::host_ops::{FLOAT, HostTensor, INT64};
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, KernelCache, NodeIr, Tensor, execute, is_implemented_node,
};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(name: &str, output: &str, attrs: HashMap<String, AttrValue>) -> NodeIr {
    let mut attrs = attrs;
    attrs.entry("dtype".into()).or_insert(AttrValue::Int(1));
    NodeIr {
        domain: String::new(),
        op: "RandomNormalLike".into(),
        opset: 22,
        name: name.into(),
        inputs: vec!["shape_source".into()],
        outputs: vec![output.into()],
        attrs,
    }
}

fn run(graph: &GraphIr, shape: Vec<i64>, cache: &KernelCache<'_>) -> Vec<f32> {
    let mut env = ExecutionEnv::new(cache, &graph.initializers);
    let input_bytes = shape.iter().product::<i64>() as usize * 8;
    env.set(
        "shape_source",
        Tensor::Host(HostTensor::new(INT64, shape, vec![0; input_bytes])),
    );
    execute(graph, &mut env).expect("RandomNormalLike graph execution");
    let output = env.host(&graph.outputs[0]).expect("download random output");
    assert_eq!(output.dtype, FLOAT);
    output.to_f32().expect("float32 output")
}

fn graph(attrs: HashMap<String, AttrValue>) -> GraphIr {
    GraphIr {
        nodes: vec![node("noise", "noise_out", attrs)],
        inputs: vec!["shape_source".into()],
        outputs: vec!["noise_out".into()],
        ..Default::default()
    }
}

fn sized_graph(attrs: HashMap<String, AttrValue>) -> GraphIr {
    graph(attrs)
}

fn capability_node(attrs: HashMap<String, AttrValue>) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: "RandomNormalLike".into(),
        opset: 21,
        name: "noise".into(),
        inputs: vec!["shape_source".into()],
        outputs: vec!["noise_out".into()],
        attrs,
    }
}

#[test]
fn capability_requires_well_typed_finite_random_attributes() {
    assert!(is_implemented_node(&capability_node(HashMap::new())));
    assert!(is_implemented_node(&capability_node(HashMap::from([(
        "dtype".into(),
        AttrValue::Int(1),
    )]))));

    for malformed in [
        HashMap::from([("dtype".into(), AttrValue::Float(1.0))]),
        HashMap::from([("dtype".into(), AttrValue::Int(6))]),
        HashMap::from([("mean".into(), AttrValue::Int(0))]),
        HashMap::from([("mean".into(), AttrValue::Float(f32::NAN))]),
        HashMap::from([("mean".into(), AttrValue::Float(f32::INFINITY))]),
        HashMap::from([("scale".into(), AttrValue::Int(1))]),
        HashMap::from([("scale".into(), AttrValue::Float(f32::NEG_INFINITY))]),
        HashMap::from([("scale".into(), AttrValue::Float(-0.1))]),
        HashMap::from([("seed".into(), AttrValue::String("1".into()))]),
        HashMap::from([("seed".into(), AttrValue::Float(f32::NAN))]),
    ] {
        assert!(
            !is_implemented_node(&capability_node(malformed)),
            "malformed RandomNormalLike attributes must not be claimed"
        );
    }

    let mut no_input = capability_node(HashMap::new());
    no_input.inputs.clear();
    assert!(!is_implemented_node(&no_input));
    let mut empty_input = capability_node(HashMap::new());
    empty_input.inputs[0].clear();
    assert!(!is_implemented_node(&empty_input));
    let mut no_output = capability_node(HashMap::new());
    no_output.outputs.clear();
    assert!(!is_implemented_node(&no_output));
    let mut empty_output = capability_node(HashMap::new());
    empty_output.outputs[0].clear();
    assert!(!is_implemented_node(&empty_output));
}

#[test]
fn explicit_seed_repeats_across_sessions_and_advances_within_one_session() {
    let context = VkContext::new().expect("Vulkan context");
    let attrs = HashMap::from([("seed".to_owned(), AttrValue::Float(17.0))]);
    let graph = sized_graph(attrs);
    let cache_a = KernelCache::new(&context);
    let first = run(&graph, vec![4096], &cache_a);
    let second = run(&graph, vec![4096], &cache_a);
    assert_ne!(
        first, second,
        "a node's random stream advances per invocation"
    );

    let cache_b = KernelCache::new(&context);
    assert_eq!(first, run(&graph, vec![4096], &cache_b));
}

#[test]
fn distinct_nodes_have_independent_streams_and_distribution_is_normal() {
    let context = VkContext::new().expect("Vulkan context");
    let attrs = HashMap::from([("seed".to_owned(), AttrValue::Float(7.0))]);
    let mut first_node = node("a", "a_out", attrs.clone());
    let mut second_node = node("b", "b_out", attrs);
    first_node.inputs[0] = "shape_source".into();
    second_node.inputs[0] = "shape_source".into();
    let graph = GraphIr {
        nodes: vec![first_node, second_node],
        inputs: vec!["shape_source".into()],
        outputs: vec!["a_out".into(), "b_out".into()],
        ..Default::default()
    };
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &graph.initializers);
    env.set(
        "shape_source",
        Tensor::Host(HostTensor::new(INT64, vec![131072], vec![0; 131072 * 8])),
    );
    execute(&graph, &mut env).expect("graph execution");
    let a = env.host("a_out").expect("a output").to_f32().expect("f32");
    let b = env.host("b_out").expect("b output").to_f32().expect("f32");
    assert_ne!(a, b, "same-shaped nodes use separate random streams");
    for samples in [&a, &b] {
        let mean = samples.iter().map(|x| f64::from(*x)).sum::<f64>() / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|x| (f64::from(*x) - mean).powi(2))
            .sum::<f64>()
            / samples.len() as f64;
        assert!(mean.abs() < 0.025, "mean {mean}");
        assert!((0.96..=1.04).contains(&variance), "variance {variance}");
        assert!(samples.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn scalar_empty_and_zero_scale_are_supported() {
    let context = VkContext::new().expect("Vulkan context");
    let scalar_graph = sized_graph(HashMap::from([
        ("seed".to_owned(), AttrValue::Float(3.0)),
        ("mean".to_owned(), AttrValue::Float(2.5)),
        ("scale".to_owned(), AttrValue::Float(0.0)),
    ]));
    let cache = KernelCache::new(&context);
    let scalar = run(&scalar_graph, vec![], &cache);
    assert_eq!(scalar, vec![2.5]);

    let empty_graph = sized_graph(HashMap::new());
    let empty_cache = KernelCache::new(&context);
    assert!(run(&empty_graph, vec![0], &empty_cache).is_empty());
}
