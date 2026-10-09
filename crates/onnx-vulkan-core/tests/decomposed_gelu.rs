//! The exact GELU written out as `Div → Erf → Add → Mul → Mul`, fused into
//! `DecomposedGelu` at load time, must answer **bit for bit** like the five
//! nodes it replaces.
//!
//! The reference is the same graph with the `Erf` output declared as a graph
//! output: an intermediate that escapes cannot be fused, so the rewrite leaves
//! that graph alone and it runs the five original kernels on the same device.

use onnx_vulkan_core::{Executor, GraphIr, HostTensor, InitializerIr, NodeIr, Tensor};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(op: &str, inputs: &[&str], output: &str) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: 17,
        name: format!("{op}_{output}"),
        inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
        outputs: vec![output.to_string()],
        attrs: HashMap::new(),
    }
}

fn scalar(value: f32) -> InitializerIr {
    InitializerIr {
        dtype: onnx_vulkan_core::host_ops::FLOAT,
        shape: vec![],
        data: value.to_le_bytes().to_vec(),
    }
}

/// `y = (x · (erf(x / √2) + 1)) · 0.5`, or with `factor_first`
/// `y = x · ((erf(x / √2) + 1) · 0.5)` as the SAM 3 vision encoder writes it.
fn gelu_graph(factor_first: bool) -> GraphIr {
    let products = if factor_first {
        [
            node("Mul", &["half", "e1"], "g"),
            node("Mul", &["x", "g"], "y"),
        ]
    } else {
        [
            node("Mul", &["x", "e1"], "g"),
            node("Mul", &["g", "half"], "y"),
        ]
    };
    GraphIr {
        nodes: [
            node("Div", &["x", "sqrt2"], "xs"),
            node("Erf", &["xs"], "e"),
            node("Add", &["e", "one"], "e1"),
        ]
        .into_iter()
        .chain(products)
        .collect(),
        initializers: HashMap::from([
            // SAM 3 stores 1.4142135381698608, which is this f32
            ("sqrt2".into(), scalar(std::f32::consts::SQRT_2)),
            ("one".into(), scalar(1.0)),
            ("half".into(), scalar(0.5)),
        ]),
        inputs: vec!["x".into()],
        outputs: vec!["y".into()],
        ..Default::default()
    }
}

struct Run {
    bits: Vec<u32>,
    ops: Vec<String>,
    fused_kernel: bool,
}

fn run(context: &VkContext, graph: GraphIr, x: &[f32]) -> Run {
    let executor = Executor::new(context, graph).expect("executor");
    let ops = executor
        .graph()
        .nodes
        .iter()
        .map(|n| n.op.clone())
        .collect();
    let feed = HostTensor::from_f32(vec![x.len() as i64], x);
    let outputs = executor.run(vec![("x", Tensor::Host(feed))]).expect("run");
    let y = outputs.host("y").expect("output");
    let bits = y.to_f32().unwrap().iter().map(|v| v.to_bits()).collect();
    outputs.finish();
    let fused_kernel = executor
        .cache()
        .pipeline_keys()
        .iter()
        .any(|key| key.operation() == "DecomposedGelu");
    Run {
        bits,
        ops,
        fused_kernel,
    }
}

/// Fused against unfused on `x`, bit for bit; `kernel` says whether the fused
/// shader must have run or the five-step replay.
fn check(x: &[f32], kernel: bool, factor_first: bool) {
    let context = VkContext::new().expect("Vulkan context");

    let mut reference = gelu_graph(factor_first);
    reference.outputs.push("e".into());
    let expected = run(&context, reference, x);
    assert!(expected.ops.iter().any(|op| op == "Erf"));
    assert!(!expected.fused_kernel);

    let got = run(&context, gelu_graph(factor_first), x);
    assert_eq!(got.ops, ["DecomposedGelu"]);
    assert_eq!(got.fused_kernel, kernel);
    let differing: Vec<(f32, u32, u32)> = x
        .iter()
        .zip(got.bits.iter().zip(&expected.bits))
        .filter(|(_, (g, e))| g != e)
        .map(|(x, (g, e))| (*x, *g, *e))
        .collect();
    assert!(
        differing.is_empty(),
        "{} of {} elements differ, first (x, got, expected): {:?}",
        differing.len(),
        x.len(),
        &differing[..differing.len().min(5)]
    );
}

#[test]
fn fused_gelu_matches_the_five_kernels() {
    // every 1024th f32 bit pattern -- both signs, subnormals, ±inf, NaNs --
    // plus a dense sweep of the range where erf is neither 0 nor ±1
    let mut x: Vec<f32> = (0..1u64 << 22)
        .map(|i| f32::from_bits((i << 10) as u32))
        .collect();
    let dense = 1 << 21;
    x.extend((0..dense).map(|i| (i as f32 / dense as f32 * 2.0 - 1.0) * 8.0));
    for factor_first in [false, true] {
        check(&x, true, factor_first);
    }
}

#[test]
fn small_host_input_replays_the_five_steps() {
    // a few host values: the `Div` runs on host, so the fused shader must not
    for factor_first in [false, true] {
        check(&[-3.0, -0.5, 0.0, 0.25, 2.0, 7.5], false, factor_first);
    }
}

/// `DynamicQuantizeLinear` of the GELU, as every SAM 3 MLP feeds its second
/// int8 matmul.
fn quantized_gelu_graph(factor_first: bool) -> GraphIr {
    let mut graph = gelu_graph(factor_first);
    graph.nodes.push(NodeIr {
        outputs: vec!["q".into(), "scale".into(), "zp".into()],
        ..node("DynamicQuantizeLinear", &["y"], "q")
    });
    graph.outputs = vec!["q".into(), "scale".into(), "zp".into()];
    graph
}

/// The bytes of `q`, `scale` and `zp`, the graph's ops after the rewrites and
/// whether the fused quantizing kernel ran.
fn run_quantized(
    context: &VkContext,
    graph: GraphIr,
    x: &[f32],
) -> (Vec<Vec<u8>>, Vec<String>, bool) {
    let executor = Executor::new(context, graph).expect("executor");
    let ops = executor
        .graph()
        .nodes
        .iter()
        .map(|n| n.op.clone())
        .collect();
    let feed = HostTensor::from_f32(vec![x.len() as i64], x);
    let outputs = executor.run(vec![("x", Tensor::Host(feed))]).expect("run");
    let bytes = ["q", "scale", "zp"]
        .map(|name| outputs.host(name).expect("output").data)
        .to_vec();
    outputs.finish();
    let fused_kernel = executor
        .cache()
        .pipeline_keys()
        .iter()
        .any(|key| key.operation() == "DecomposedGeluQuantize");
    (bytes, ops, fused_kernel)
}

/// Fused GELU quantization against the five kernels and the three
/// `DynamicQuantizeLinear` passes, byte for byte.
fn check_quantized(x: &[f32], kernel: bool, factor_first: bool) {
    let context = VkContext::new().expect("Vulkan context");

    // the escaping `Erf` output keeps the GELU, and so its quantization, apart
    let mut reference = quantized_gelu_graph(factor_first);
    reference.outputs.push("e".into());
    let (expected, ops, fused) = run_quantized(&context, reference, x);
    assert!(ops.iter().any(|op| op == "Erf"));
    assert!(ops.iter().any(|op| op == "DynamicQuantizeLinear"));
    assert!(!fused);

    let (got, ops, fused) = run_quantized(&context, quantized_gelu_graph(factor_first), x);
    assert_eq!(ops, ["DecomposedGeluQuantize"]);
    assert_eq!(fused, kernel);
    assert_eq!(got[1], expected[1], "scale");
    assert_eq!(got[2], expected[2], "zero point");
    let differing = got[0]
        .iter()
        .zip(&expected[0])
        .filter(|(g, e)| g != e)
        .count();
    assert_eq!(differing, 0, "{differing} of {} bytes differ", x.len());
    assert_eq!(got[0].len(), x.len());
}

#[test]
fn quantized_gelu_matches_the_unfused_nodes() {
    // a dense sweep where erf is neither 0 nor ±1, an odd length so the last
    // packed word is partial, and one outlier that sets the range: as in an
    // MLP activation, most values then share a few quantization steps
    let dense = (1 << 21) + 3;
    let mut x: Vec<f32> = (0..dense)
        .map(|i| (i as f32 / dense as f32 * 2.0 - 1.0) * 8.0)
        .collect();
    for factor_first in [false, true] {
        check_quantized(&x, true, factor_first);
    }
    x[12345] = 113.25;
    for factor_first in [false, true] {
        check_quantized(&x, true, factor_first);
    }
    // all negative: the GELU is ≤ 0 and the range is pinned at 0 from above
    let negative: Vec<f32> = x.iter().map(|v| -v.abs()).collect();
    check_quantized(&negative, true, true);
}

#[test]
fn small_host_input_replays_the_quantized_gelu() {
    for factor_first in [false, true] {
        check_quantized(&[-3.0, -0.5, 0.0, 0.25, 2.0, 7.5], false, factor_first);
    }
}
