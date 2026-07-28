//! The core's public execution API: one graph, one set of session resources.
//!
//! `Executor` is what the ORT plugin uses too — the point of the type is that
//! there is no second, private path. These tests exercise it without ORT.

use onnx_vulkan_core::host_ops::{FLOAT, HostTensor, INT64};
use onnx_vulkan_core::{
    Error, Executor, GraphIr, InitializerIr, NodeIr, Tensor, is_implemented_node,
};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        since_version: 13,
        name: format!("{op}_0"),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: HashMap::new(),
    }
}

/// `x` [2,4] → Mul by a broadcast initializer [4] → Relu, plus a
/// host-resolved `Shape`: both interpreter paths in one graph.
fn graph() -> GraphIr {
    let mut initializers = HashMap::new();
    initializers.insert(
        "w".to_string(),
        InitializerIr {
            dtype: FLOAT,
            shape: vec![4],
            data: [0.5f32, 2.0, 1.0, -1.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        },
    );
    GraphIr {
        nodes: vec![
            node("Shape", &["x"], &["x_shape"]),
            node("Mul", &["x", "w"], &["prod"]),
            node("Relu", &["prod"], &["out"]),
        ],
        initializers,
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string(), "x_shape".to_string()],
    }
}

const X: [f32; 8] = [1.0, -2.0, 3.0, -4.0, 5.0, 6.0, -7.0, 8.0];
const WANT: [f32; 8] = [0.5, 0.0, 3.0, 4.0, 2.5, 12.0, 0.0, 0.0];

fn input() -> Tensor<'static> {
    Tensor::Host(HostTensor::from_f32(vec![2, 4], &X))
}

#[test]
fn runs_a_graph_and_reads_both_device_and_host_outputs() {
    let context = VkContext::new().expect("Vulkan context");
    let executor = Executor::new(&context, graph()).expect("supported graph");

    let outputs = executor.run(vec![("x", input())]).expect("execution");
    assert!(outputs.on_device("out"), "the GPU branch stays in VRAM");
    assert_eq!(outputs.shape_of("out").expect("shape"), vec![2, 4]);
    assert_eq!(outputs.dtype_of("out").expect("dtype"), FLOAT);
    assert_eq!(
        outputs
            .host("out")
            .expect("download")
            .to_f32()
            .expect("f32"),
        WANT.to_vec()
    );

    // `Shape` is resolved on the host: it must not end up in VRAM
    assert!(!outputs.on_device("x_shape"));
    let shape = outputs.host("x_shape").expect("host shape");
    assert_eq!(shape.dtype, INT64);
    assert_eq!(shape.to_i64().expect("i64"), vec![2, 4]);
    outputs.finish();
}

/// The reason the executor owns the cache: the second run does not
/// recompile pipelines. This is the property on which weight residency rests.
#[test]
fn a_second_run_compiles_no_new_pipeline() {
    let context = VkContext::new().expect("Vulkan context");
    let executor = Executor::new(&context, graph()).expect("supported graph");

    executor
        .run(vec![("x", input())])
        .expect("first execution")
        .finish();
    let after_first = executor.cache().builds();
    assert!(
        after_first.0 > 0,
        "the first run compiles at least one pipeline"
    );

    executor
        .run(vec![("x", input())])
        .expect("second execution")
        .finish();
    assert_eq!(
        executor.cache().builds(),
        after_first,
        "the second run must not compile or pack anything"
    );
}

/// A model either runs entirely on the GPU or fails hard, and the useful
/// moment to fail is at load time — not mid-inference.
#[test]
fn an_unimplemented_node_is_rejected_at_construction() {
    let context = VkContext::new().expect("Vulkan context");
    let mut ir = graph();
    let bad = node("NonEsisteQuestaOp", &["out"], &["worse"]);
    assert!(!is_implemented_node(&bad), "test premise");
    ir.nodes.push(bad);

    match Executor::new(&context, ir) {
        Err(Error::Unsupported(message)) => {
            assert!(
                message.contains("NonEsisteQuestaOp"),
                "the error must say which node: {message}"
            );
        }
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => panic!("the unsupported graph was accepted"),
    }
}

/// An unsupported form of a known op must also be rejected at load time:
/// the name in the list is not enough.
#[test]
fn an_unsupported_form_of_a_known_op_is_rejected_too() {
    let context = VkContext::new().expect("Vulkan context");
    let mut ir = graph();
    let mut reduce = node("ReduceSum", &["out"], &["reduced"]);
    // axes as unresolved input: not claimable
    reduce.inputs.push("axes".to_string());
    ir.nodes.push(reduce);

    assert!(matches!(
        Executor::new(&context, ir),
        Err(Error::Unsupported(_))
    ));
}
