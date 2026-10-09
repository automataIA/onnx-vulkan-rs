//! `MatMulInteger` with a K that is not a multiple of four — SAM 3's box and
//! point projections have K = 2 — runs on the device when A is there, and must
//! give the exact sums the host gives.
//!
//! A comes out of a quantizer inside the graph (so it is a device tensor, as
//! in SAM 3's detector) and is also a graph output: the expected sums are
//! computed here from the very bytes and zero point the matmul read.

use onnx_vulkan_core::host_ops::{FLOAT, INT8, UINT8};
use onnx_vulkan_core::{AttrValue, Executor, GraphIr, HostTensor, InitializerIr, NodeIr, Tensor};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: 17,
        name: format!("{op}_{}", outputs[0]),
        inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
        outputs: outputs.iter().map(|name| (*name).to_string()).collect(),
        attrs: HashMap::<String, AttrValue>::new(),
    }
}

fn bytes(dtype: i32, shape: Vec<i64>, data: Vec<u8>) -> InitializerIr {
    InitializerIr { dtype, shape, data }
}

/// Deterministic bytes (a 64-bit LCG's high bits).
fn pseudo(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 56) as u8
        })
        .collect()
}

/// `x [M, K]` quantized to A — by `DynamicQuantizeLinear` (u8) or by a
/// `QuantizeLinear` with an int8 zero point (i8) — then `A × w`.
fn graph(k: usize, n: usize, a_signed: bool, b_signed: bool) -> GraphIr {
    let w_dtype = if b_signed { INT8 } else { UINT8 };
    let mut initializers = HashMap::from([
        (
            "w".into(),
            bytes(w_dtype, vec![k as i64, n as i64], pseudo(k * n, 7)),
        ),
        (
            "w_zp".into(),
            bytes(w_dtype, vec![], vec![if b_signed { 0xfd } else { 131 }]),
        ),
    ]);
    let mut nodes = Vec::new();
    if a_signed {
        initializers.insert(
            "scale".into(),
            bytes(FLOAT, vec![], 0.02f32.to_le_bytes().to_vec()),
        );
        initializers.insert("zp".into(), bytes(INT8, vec![], vec![0x05]));
        nodes.push(node("QuantizeLinear", &["x", "scale", "zp"], &["a"]));
        nodes.push(node("Identity", &["zp"], &["a_zp"]));
    } else {
        nodes.push(node(
            "DynamicQuantizeLinear",
            &["x"],
            &["a", "a_scale", "a_zp"],
        ));
    }
    nodes.push(node("MatMulInteger", &["a", "w", "a_zp", "w_zp"], &["y"]));
    GraphIr {
        nodes,
        initializers,
        inputs: vec!["x".into()],
        outputs: vec!["y".into(), "a".into(), "a_zp".into()],
        ..Default::default()
    }
}

fn signed_codes(tensor: &HostTensor) -> Vec<i64> {
    tensor.to_i64().expect("integer tensor")
}

fn check(m: usize, k: usize, n: usize, a_signed: bool, b_signed: bool) {
    let context = VkContext::new().expect("Vulkan context");
    let x: Vec<f32> = pseudo(m * k, (m * k) as u64)
        .iter()
        .map(|&v| (f32::from(v) - 100.0) * 0.03)
        .collect();
    let g = graph(k, n, a_signed, b_signed);
    let w = signed_codes(&HostTensor::new(
        g.initializers["w"].dtype,
        g.initializers["w"].shape.clone(),
        g.initializers["w"].data.clone(),
    ));
    let w_zp = signed_codes(&HostTensor::new(
        g.initializers["w_zp"].dtype,
        vec![],
        g.initializers["w_zp"].data.clone(),
    ))[0];
    let executor = Executor::new(&context, g).expect("executor");
    let outputs = executor
        .run(vec![(
            "x",
            Tensor::Host(HostTensor::from_f32(vec![m as i64, k as i64], &x)),
        )])
        .expect("run");
    let a = signed_codes(&outputs.host("a").expect("a"));
    let a_zp = signed_codes(&outputs.host("a_zp").expect("a_zp"))[0];
    let y = outputs.host("y").expect("y");
    assert_eq!(y.shape, [m as i64, n as i64]);
    let got = signed_codes(&y);
    outputs.finish();
    let expected: Vec<i64> = (0..m * n)
        .map(|i| {
            let (row, col) = (i / n, i % n);
            (0..k)
                .map(|j| (a[row * k + j] - a_zp) * (w[j * n + col] - w_zp))
                .sum()
        })
        .collect();
    let differing = got.iter().zip(&expected).filter(|(g, e)| g != e).count();
    assert_eq!(
        differing,
        0,
        "m={m} k={k} n={n} a_signed={a_signed} b_signed={b_signed}: {differing} of {} differ",
        got.len()
    );
    let ran = executor
        .cache()
        .pipeline_keys()
        .iter()
        .any(|key| key.operation() == "MMI_matmul_any_k");
    assert!(ran, "m={m} k={k} n={n}: not on the device kernel");
}

#[test]
fn any_k_matches_the_exact_sums() {
    // SAM 3's box projection: 200 queries × 72 points, K = 2, N = 256
    check(14_400, 2, 256, false, true);
    for k in [1, 3, 5, 7] {
        for (a_signed, b_signed) in [(false, false), (false, true), (true, false), (true, true)] {
            check(1_000, k, 37, a_signed, b_signed);
        }
    }
}

/// A host A (a graph input here) stays on the host, as before.
#[test]
fn host_input_stays_on_the_host() {
    let context = VkContext::new().expect("Vulkan context");
    let (m, k, n) = (3, 2, 8);
    let a = pseudo(m * k, 11);
    let w = pseudo(k * n, 7);
    let ir = GraphIr {
        nodes: vec![node("MatMulInteger", &["a", "w", "a_zp", "w_zp"], &["y"])],
        initializers: HashMap::from([
            (
                "w".into(),
                bytes(UINT8, vec![k as i64, n as i64], w.clone()),
            ),
            ("a_zp".into(), bytes(UINT8, vec![], vec![128])),
            ("w_zp".into(), bytes(UINT8, vec![], vec![131])),
        ]),
        inputs: vec!["a".into()],
        outputs: vec!["y".into()],
        ..Default::default()
    };
    let executor = Executor::new(&context, ir).expect("executor");
    let outputs = executor
        .run(vec![(
            "a",
            Tensor::Host(HostTensor::new(UINT8, vec![m as i64, k as i64], a.clone())),
        )])
        .expect("run");
    let got = signed_codes(&outputs.host("y").expect("y"));
    outputs.finish();
    let expected: Vec<i64> = (0..m * n)
        .map(|i| {
            let (row, col) = (i / n, i % n);
            (0..k)
                .map(|j| (i64::from(a[row * k + j]) - 128) * (i64::from(w[j * n + col]) - 131))
                .sum()
        })
        .collect();
    assert_eq!(got, expected);
    assert!(
        !executor
            .cache()
            .pipeline_keys()
            .iter()
            .any(|key| key.operation() == "MMI_matmul_any_k")
    );
}
