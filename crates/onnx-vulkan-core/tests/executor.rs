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
        opset: 13,
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
        ..Default::default()
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
    // `Shape` is resolved on the host: it must not end up in VRAM. Reading it
    // together with the device output exercises mixed-residency batching and
    // preserves the caller's requested order.
    assert!(!outputs.on_device("x_shape"));
    let mut downloaded = outputs
        .host_many(&["x_shape", "out"])
        .expect("batched mixed-residency readback");
    let output = downloaded.pop().expect("device output");
    let shape = downloaded.pop().expect("host shape");
    assert_eq!(output.to_f32().expect("f32"), WANT.to_vec());
    assert_eq!(shape.dtype, INT64);
    assert_eq!(shape.to_i64().expect("i64"), vec![2, 4]);

    // The batched path populates the same small-download cache as `host`.
    assert_eq!(
        outputs.host("out").expect("cached output").data,
        output.data
    );
    outputs.finish();
}

/// The caller-owned readback: a second run overwrites the same tensors in
/// place, keeping the output buffer's allocation, whatever they held before.
#[test]
fn reads_outputs_into_tensors_kept_across_runs() {
    let context = VkContext::new().expect("Vulkan context");
    let executor = Executor::new(&context, graph()).expect("supported graph");
    // stale contents of another dtype and shape
    let mut kept = vec![HostTensor::from_f32(vec![3], &[9.0; 3]); 2];

    let outputs = executor.run(vec![("x", input())]).expect("first execution");
    outputs
        .host_many_into(&["out", "x_shape"], &mut kept)
        .expect("readback into kept tensors");
    assert_eq!(kept[0], outputs.host("out").expect("owned readback"));
    assert_eq!(kept[0].to_f32().expect("f32"), WANT.to_vec());
    assert_eq!(
        (kept[1].dtype, kept[1].to_i64().expect("i64")),
        (INT64, vec![2, 4])
    );
    let wrong = outputs.host_many_into(&["out"], &mut kept);
    assert!(wrong.is_err(), "one destination per name");
    outputs.finish();

    let buffer = kept[0].data.as_ptr();
    let negated: Vec<f32> = X.iter().map(|v| -v).collect();
    let outputs = executor
        .run(vec![(
            "x",
            Tensor::Host(HostTensor::from_f32(vec![2, 4], &negated)),
        )])
        .expect("second execution");
    outputs
        .host_many_into(&["out", "x_shape"], &mut kept)
        .expect("second readback");
    assert_eq!(
        kept[0].to_f32().expect("f32"),
        [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0]
    );
    assert_eq!(kept[0].data.as_ptr(), buffer, "allocation reused");
    outputs.finish();
}

/// A one-shot run leaves its buffers pooled within the context's budget, and
/// the next run of the same graph is served them: nothing allocated, nothing
/// freed.
#[test]
fn a_second_run_is_served_the_buffers_the_first_one_pooled() {
    let context = VkContext::new().expect("Vulkan context");
    let executor = Executor::new(&context, graph()).expect("supported graph");
    let pooled_ids = || {
        let pooled = context.take_storage_pool();
        let mut ids: Vec<u64> = pooled.iter().map(|buffer| buffer.id()).collect();
        for buffer in pooled {
            context.recycle_storage_buffer(buffer);
        }
        ids.sort_unstable();
        ids
    };

    executor
        .run(vec![("x", input())])
        .expect("first execution")
        .finish();
    let first = pooled_ids();
    assert!(
        !first.is_empty(),
        "the run keeps its buffers for the next one"
    );

    let outputs = executor
        .run(vec![("x", input())])
        .expect("second execution");
    assert_eq!(outputs.host("out").expect("out").to_f32().unwrap(), WANT);
    outputs.finish();
    assert_eq!(pooled_ids(), first, "the same buffers, none new");
}

/// `Reshape` takes its input's buffer when it is the input's last reader, and
/// copies it when the graph still needs the input: same output, one buffer
/// fewer in the first case.
#[test]
fn a_reshape_takes_the_buffer_of_an_input_that_dies_there() {
    let context = VkContext::new().expect("Vulkan context");
    let mut ir = graph();
    ir.nodes = vec![
        node("Mul", &["x", "w"], &["prod"]),
        node("Relu", &["prod"], &["act"]),
        node("Reshape", &["act", "flat"], &["out"]),
    ];
    ir.outputs = vec!["out".to_string()];
    ir.initializers.insert(
        "flat".to_string(),
        InitializerIr {
            dtype: INT64,
            shape: vec![1],
            data: 8i64.to_le_bytes().to_vec(),
        },
    );
    // storage buffers the run asked for, allocated or recycled
    let buffers_requested = |ir: GraphIr| {
        let executor = Executor::new(&context, ir).expect("supported graph");
        let before = vk_compute::stats::allocs() + vk_compute::stats::pool_hits();
        let outputs = executor.run(vec![("x", input())]).expect("execution");
        assert!(outputs.on_device("out"));
        assert_eq!(outputs.shape_of("out").expect("shape"), vec![8]);
        assert_eq!(outputs.host("out").expect("out").to_f32().unwrap(), WANT);
        outputs.finish();
        vk_compute::stats::allocs() + vk_compute::stats::pool_hits() - before
    };

    let moved = buffers_requested(ir.clone());
    ir.outputs.push("act".to_string());
    let copied = buffers_requested(ir);
    assert_eq!(
        copied,
        moved + 1,
        "the live input is copied, the dead one is not"
    );
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
