//! `Add → LayerNormalization`, fused into `AddLayerNormalization` at load
//! time, must answer **bit for bit** like the two nodes it replaces — the
//! one-pass kernel when the operands share a shape, the original steps when
//! not — and so must the sum when the graph reads it again.
//!
//! The reference is the same graph with the sum also read by an `Identity`
//! placed before the normalization: the fused node would take the
//! normalization's slot, after that reader, so the rewrite leaves that graph
//! alone and it runs the original kernels.

use onnx_vulkan_core::host_ops::FLOAT;
use onnx_vulkan_core::{AttrValue, Executor, GraphIr, HostTensor, InitializerIr, NodeIr, Tensor};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(op: &str, inputs: &[&str], output: &str, attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: 18,
        name: format!("{op}_{output}"),
        inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
        outputs: vec![output.to_string()],
        attrs: attrs
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect(),
    }
}

/// Realistic activations from a fixed seed, in `[-6, 6)`.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * 6.0
        })
        .collect()
}

fn vector(values: &[f32]) -> InitializerIr {
    InitializerIr {
        dtype: FLOAT,
        shape: vec![values.len() as i64],
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

struct Case {
    a: Vec<i64>,
    b: Vec<i64>,
    bias: bool,
    /// the next residual `Add` reads the sum again
    residual: bool,
}

fn graph(case: &Case, probe: bool) -> GraphIr {
    let c = *case.a.last().unwrap() as usize;
    let mut initializers = HashMap::from([("gamma".to_string(), vector(&values(c, 11)))]);
    let mut norm_inputs = vec!["sum", "gamma"];
    if case.bias {
        initializers.insert("beta".to_string(), vector(&values(c, 13)));
        norm_inputs.push("beta");
    }
    let mut nodes = vec![node("Add", &["a", "b"], "sum", &[])];
    let mut outputs = vec!["y".to_string()];
    if probe {
        nodes.push(node("Identity", &["sum"], "probe", &[]));
        outputs.push("probe".into());
    }
    nodes.push(node(
        "LayerNormalization",
        &norm_inputs,
        "y",
        &[
            ("axis", AttrValue::Int(-1)),
            ("epsilon", AttrValue::Float(1e-5)),
        ],
    ));
    if case.residual {
        nodes.push(node("Add", &["sum", "y"], "next", &[]));
        outputs.push("next".into());
    }
    GraphIr {
        nodes,
        initializers,
        inputs: vec!["a".into(), "b".into()],
        outputs,
        ..Default::default()
    }
}

struct Run {
    /// `y`, then `next` when the case has it
    bits: Vec<Vec<u32>>,
    ops: Vec<String>,
    fused_kernel: bool,
}

fn run(context: &VkContext, graph: GraphIr, case: &Case) -> Run {
    let executor = Executor::new(context, graph).expect("executor");
    let ops = executor
        .graph()
        .nodes
        .iter()
        .map(|n| n.op.clone())
        .collect();
    let a = values(case.a.iter().product::<i64>() as usize, 3);
    let b = values(case.b.iter().product::<i64>() as usize, 7);
    let feeds = vec![
        ("a", Tensor::Host(HostTensor::from_f32(case.a.clone(), &a))),
        ("b", Tensor::Host(HostTensor::from_f32(case.b.clone(), &b))),
    ];
    let outputs = executor.run(feeds).expect("run");
    let names: &[&str] = if case.residual {
        &["y", "next"]
    } else {
        &["y"]
    };
    let bits = names
        .iter()
        .map(|name| {
            let tensor = outputs.host(name).expect("output");
            assert_eq!(tensor.shape, case.a, "{name}");
            tensor
                .to_f32()
                .expect("f32")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect();
    outputs.finish();
    let fused_kernel = executor
        .cache()
        .pipeline_keys()
        .iter()
        .any(|key| key.operation() == "AddLayerNormalization");
    Run {
        bits,
        ops,
        fused_kernel,
    }
}

fn check(case: &Case, kernel: bool) {
    let context = VkContext::new().expect("Vulkan context");

    let expected = run(&context, graph(case, true), case);
    assert_eq!(expected.ops[..3], ["Add", "Identity", "LayerNormalization"]);
    assert!(!expected.fused_kernel);

    let got = run(&context, graph(case, false), case);
    assert_eq!(got.ops[0], "AddLayerNormalization");
    assert_eq!(got.fused_kernel, kernel, "{:?} + {:?}", case.a, case.b);
    assert!(
        got.bits == expected.bits,
        "fused vs unfused, {:?} + {:?}",
        case.a,
        case.b
    );
}

/// SAM 3's detector at a few rows: a residual sum normalized and then read
/// again, rows wider than one workgroup pass.
#[test]
fn a_residual_read_again_is_normalized_and_written_in_one_pass_alike() {
    let case = Case {
        a: vec![2, 37, 320],
        b: vec![2, 37, 320],
        bias: true,
        residual: true,
    };
    check(&case, true);
}

#[test]
fn a_sum_read_only_by_the_normalization_and_no_bias_alike() {
    let case = Case {
        a: vec![1, 21, 256],
        b: vec![1, 21, 256],
        bias: false,
        residual: false,
    };
    check(&case, true);
}

#[test]
fn a_broadcast_add_runs_the_original_steps_alike() {
    let case = Case {
        a: vec![1, 21, 256],
        b: vec![256],
        bias: true,
        residual: true,
    };
    check(&case, false);
}
