//! `SkipSimplifiedLayerNormalization` (com.microsoft) against a scalar
//! reference, and against the two nodes it replaces.
//!
//! The op is `Add` + `SimplifiedLayerNormalization` fused, so the strongest
//! oracle available is the unfused pair running in the same engine: if the two
//! agree, the fusion is faithful whatever the reference implementation of RMS
//! normalization rounds to. The scalar reference is kept as well, because two
//! agreeing paths could still share a wrong idea of what the op computes.
//!
//! The fourth output is the reason the fusion is worth having — 71 of qwen2.5-
//! VL's 72 nodes consume `x + skip` as the next layer's residual — so it is
//! asserted, not just the normalized one.

use onnx_vulkan_core::host_ops::HostTensor;
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, KernelCache, NodeIr, Tensor, execute, is_implemented_node,
};
use std::collections::HashMap;
use vk_compute::VkContext;

const EPS: f32 = 1e-6;

fn ramp(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 31 + seed * 13) % 17) as f32 - 8.0) / 8.0)
        .collect()
}

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeIr {
    NodeIr {
        domain: "com.microsoft".into(),
        op: op.into(),
        opset: 1,
        name: op.into(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: HashMap::from([("epsilon".to_string(), AttrValue::Float(EPS))]),
    }
}

/// Runs a graph over host inputs and brings the named outputs back.
fn run(ir: &GraphIr, host: &[(&str, Vec<i64>, &[f32])], outputs: &[&str]) -> Vec<Vec<f32>> {
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    for (name, shape, data) in host {
        env.set(
            name,
            Tensor::Host(HostTensor::from_f32(shape.clone(), data)),
        );
    }
    execute(ir, &mut env).expect("graph execution");
    let got = outputs
        .iter()
        .map(|name| {
            env.host(name)
                .expect("output on host")
                .to_f32()
                .expect("f32 output")
        })
        .collect();
    env.finish();
    got
}

/// RMS normalization of `x + skip`, longhand.
fn reference(x: &[f32], skip: &[f32], gamma: &[f32], c: usize) -> (Vec<f32>, Vec<f32>) {
    let sum: Vec<f32> = x.iter().zip(skip).map(|(a, b)| a + b).collect();
    let out = sum
        .chunks(c)
        .flat_map(|row| {
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / c as f32;
            let inv = 1.0 / (mean_sq + EPS).sqrt();
            row.iter()
                .zip(gamma)
                .map(move |(v, g)| v * inv * g)
                .collect::<Vec<_>>()
        })
        .collect();
    (out, sum)
}

/// qwen2.5-VL's shape, shrunk: the residual sum is consumed downstream, so both
/// outputs are checked.
#[test]
fn skip_form_matches_the_unfused_add_and_rms_norm() {
    let (rows, c) = (3usize, 16usize);
    let x = ramp(rows * c, 1);
    let skip = ramp(rows * c, 2);
    let gamma = ramp(c, 3);
    let shape = vec![1, rows as i64, c as i64];
    let host: Vec<(&str, Vec<i64>, &[f32])> = vec![
        ("x", shape.clone(), &x),
        ("skip", shape.clone(), &skip),
        ("gamma", vec![c as i64], &gamma),
    ];

    let fused = node(
        "SkipSimplifiedLayerNormalization",
        &["x", "skip", "gamma"],
        // `mean` and `inv_std_var` stay empty: the kernel does not produce them
        &["out", "", "", "sum"],
    );
    assert!(is_implemented_node(&fused), "the node must be claimable");
    let fused_ir = GraphIr {
        nodes: vec![fused],
        initializers: HashMap::new(),
        inputs: Vec::new(),
        outputs: vec!["out".into(), "sum".into()],
        ..Default::default()
    };
    let got = run(&fused_ir, &host, &["out", "sum"]);

    // the same computation as two nodes, which is what the export would emit
    // without the contrib op
    let unfused_ir = GraphIr {
        nodes: vec![
            NodeIr {
                domain: String::new(),
                op: "Add".into(),
                opset: 14,
                name: "add".into(),
                inputs: vec!["x".into(), "skip".into()],
                outputs: vec!["sum".into()],
                attrs: HashMap::new(),
            },
            node("SimplifiedLayerNormalization", &["sum", "gamma"], &["out"]),
        ],
        initializers: HashMap::new(),
        inputs: Vec::new(),
        outputs: vec!["out".into(), "sum".into()],
        ..Default::default()
    };
    let unfused = run(&unfused_ir, &host, &["out", "sum"]);

    let (expected_out, expected_sum) = reference(&x, &skip, &gamma, c);
    for (name, expected, actual) in [
        ("output", &expected_out, &got[0]),
        ("sum", &expected_sum, &got[1]),
    ] {
        assert_eq!(expected.len(), actual.len(), "{name}: length");
        for (i, (e, a)) in expected.iter().zip(actual).enumerate() {
            assert!((e - a).abs() <= 1e-5, "{name}[{i}] = {a}, reference {e}");
        }
    }
    for (i, (fused, split)) in got[0].iter().zip(&unfused[0]).enumerate() {
        assert!(
            (fused - split).abs() <= 1e-6,
            "fused[{i}] = {fused}, unfused {split}"
        );
    }
}

/// Without the fourth output the sum must not be written anywhere, and the
/// normalized output is unchanged — the case the last layer of the decoder is.
#[test]
fn the_residual_sum_is_optional() {
    let (rows, c) = (2usize, 8usize);
    let x = ramp(rows * c, 4);
    let skip = ramp(rows * c, 5);
    let gamma = ramp(c, 6);
    let shape = vec![1, rows as i64, c as i64];
    let host: Vec<(&str, Vec<i64>, &[f32])> = vec![
        ("x", shape.clone(), &x),
        ("skip", shape.clone(), &skip),
        ("gamma", vec![c as i64], &gamma),
    ];
    let only_out = node(
        "SkipSimplifiedLayerNormalization",
        &["x", "skip", "gamma"],
        &["out"],
    );
    assert!(is_implemented_node(&only_out), "the node must be claimable");
    let ir = GraphIr {
        nodes: vec![only_out],
        initializers: HashMap::new(),
        inputs: Vec::new(),
        outputs: vec!["out".into()],
        ..Default::default()
    };
    let got = run(&ir, &host, &["out"]);
    let (expected, _) = reference(&x, &skip, &gamma, c);
    for (i, (e, a)) in expected.iter().zip(&got[0]).enumerate() {
        assert!((e - a).abs() <= 1e-5, "out[{i}] = {a}, reference {e}");
    }
}

/// `beta` and `bias` are not implemented, and `mean`/`inv_std_var` are not
/// produced: a node asking for either must be refused at capability time rather
/// than run with a term missing.
#[test]
fn unimplemented_inputs_and_outputs_are_refused() {
    let with_bias = node(
        "SkipSimplifiedLayerNormalization",
        &["x", "skip", "gamma", "beta"],
        &["out"],
    );
    assert!(!is_implemented_node(&with_bias));
    let with_mean = node(
        "SkipSimplifiedLayerNormalization",
        &["x", "skip", "gamma"],
        &["out", "mean"],
    );
    assert!(!is_implemented_node(&with_mean));
}
