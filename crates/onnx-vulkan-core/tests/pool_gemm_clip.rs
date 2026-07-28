//! Ops added to remove CPU↔GPU boundaries from vision graphs:
//! `MaxPool`, `AveragePool`, `GlobalAveragePool`, `Gemm`, `Clip`, `Erf`.
//! Each test compares the kernel against a CPU reference written here.

use onnx_vulkan_core::host_ops::{FLOAT, HostTensor};
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, InitializerIr, KernelCache, NodeIr, Tensor, execute,
};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        since_version: 13,
        name: format!("{op}_0"),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: attrs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect(),
    }
}

fn f32_init(shape: Vec<i64>, values: &[f32]) -> InitializerIr {
    InitializerIr {
        dtype: FLOAT,
        shape,
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn graph(nodes: Vec<NodeIr>, initializers: Vec<(&str, InitializerIr)>, output: &str) -> GraphIr {
    GraphIr {
        nodes,
        initializers: initializers
            .into_iter()
            .map(|(name, init)| (name.to_string(), init))
            .collect::<HashMap<_, _>>(),
        inputs: Vec::new(),
        outputs: vec![output.to_string()],
    }
}

fn pseudo(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        })
        .collect()
}

fn run(ir: &GraphIr, inputs: &[(&str, HostTensor)], output: &str) -> Vec<f32> {
    let context = VkContext::new().expect("contesto Vulkan");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    for (name, tensor) in inputs {
        env.set(name, Tensor::Host(tensor.clone()));
    }
    execute(ir, &mut env).expect("graph execution");
    let out = env.host(output).expect("output on host").clone();
    env.finish();
    out.to_f32().expect("output f32")
}

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "lunghezze diverse");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5 + 1e-4 * w.abs(),
            "element {i}: {g} != {w}"
        );
    }
}

/// CPU reference for 2D pooling on a [1, C, H, W] tensor.
#[allow(clippy::too_many_arguments)]
fn pool_ref(
    x: &[f32],
    c: usize,
    h: usize,
    w: usize,
    k: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    max: bool,
) -> Vec<f32> {
    let h_out = (h + 2 * pad.0 - k.0) / stride.0 + 1;
    let w_out = (w + 2 * pad.1 - k.1) / stride.1 + 1;
    let mut out = vec![0.0; c * h_out * w_out];
    for ch in 0..c {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let mut acc = if max { f32::NEG_INFINITY } else { 0.0 };
                let mut count = 0.0;
                for r in 0..k.0 {
                    let ih = (oh * stride.0 + r) as isize - pad.0 as isize;
                    if ih < 0 || ih >= h as isize {
                        continue;
                    }
                    for s in 0..k.1 {
                        let iw = (ow * stride.1 + s) as isize - pad.1 as isize;
                        if iw < 0 || iw >= w as isize {
                            continue;
                        }
                        let v = x[(ch * h + ih as usize) * w + iw as usize];
                        acc = if max { acc.max(v) } else { acc + v };
                        count += 1.0;
                    }
                }
                out[(ch * h_out + oh) * w_out + ow] = if max { acc } else { acc / count };
            }
        }
    }
    out
}

#[test]
fn maxpool_matches_cpu_reference() {
    let (c, h, w) = (3usize, 7usize, 5usize);
    let x = pseudo(c * h * w, 7);
    let ir = graph(
        vec![node(
            "MaxPool",
            &["x"],
            &["y"],
            &[
                ("kernel_shape", AttrValue::Ints(vec![3, 3])),
                ("strides", AttrValue::Ints(vec![2, 1])),
                ("pads", AttrValue::Ints(vec![1, 1, 1, 1])),
            ],
        )],
        vec![],
        "y",
    );
    let got = run(
        &ir,
        &[(
            "x",
            HostTensor::from_f32(vec![1, c as i64, h as i64, w as i64], &x),
        )],
        "y",
    );
    let want = pool_ref(&x, c, h, w, (3, 3), (2, 1), (1, 1), true);
    assert_close(&got, &want);
}

#[test]
fn averagepool_excludes_padding_like_onnx() {
    let (c, h, w) = (2usize, 5usize, 5usize);
    let x = pseudo(c * h * w, 11);
    let ir = graph(
        vec![node(
            "AveragePool",
            &["x"],
            &["y"],
            &[
                ("kernel_shape", AttrValue::Ints(vec![3, 3])),
                ("strides", AttrValue::Ints(vec![2, 2])),
                ("pads", AttrValue::Ints(vec![1, 1, 1, 1])),
            ],
        )],
        vec![],
        "y",
    );
    let got = run(
        &ir,
        &[(
            "x",
            HostTensor::from_f32(vec![1, c as i64, h as i64, w as i64], &x),
        )],
        "y",
    );
    let want = pool_ref(&x, c, h, w, (3, 3), (2, 2), (1, 1), false);
    assert_close(&got, &want);
}

#[test]
fn global_average_pool_reduces_the_whole_map() {
    let (c, h, w) = (4usize, 6usize, 3usize);
    let x = pseudo(c * h * w, 13);
    let ir = graph(
        vec![node("GlobalAveragePool", &["x"], &["y"], &[])],
        vec![],
        "y",
    );
    let got = run(
        &ir,
        &[(
            "x",
            HostTensor::from_f32(vec![1, c as i64, h as i64, w as i64], &x),
        )],
        "y",
    );
    let want: Vec<f32> = (0..c)
        .map(|ch| x[ch * h * w..(ch + 1) * h * w].iter().sum::<f32>() / (h * w) as f32)
        .collect();
    assert_close(&got, &want);
}

#[test]
fn gemm_with_transpose_alpha_beta_and_bias() {
    let (m, k, n) = (5usize, 4usize, 3usize);
    // A transposed: [K, M]; B straight: [K, N]; C per row: [N]
    let a = pseudo(k * m, 17);
    let b = pseudo(k * n, 19);
    let c = pseudo(n, 23);
    let (alpha, beta) = (0.5f32, 2.0f32);
    let ir = graph(
        vec![node(
            "Gemm",
            &["a", "b", "c"],
            &["y"],
            &[
                ("transA", AttrValue::Int(1)),
                ("alpha", AttrValue::Float(alpha)),
                ("beta", AttrValue::Float(beta)),
            ],
        )],
        vec![
            ("b", f32_init(vec![k as i64, n as i64], &b)),
            ("c", f32_init(vec![n as i64], &c)),
        ],
        "y",
    );
    let got = run(
        &ir,
        &[("a", HostTensor::from_f32(vec![k as i64, m as i64], &a))],
        "y",
    );
    let mut want = vec![0.0f32; m * n];
    for (i, w) in want.iter_mut().enumerate() {
        let (row, col) = (i / n, i % n);
        let dot: f32 = (0..k).map(|t| a[t * m + row] * b[t * n + col]).sum();
        *w = alpha * dot + beta * c[col];
    }
    assert_close(&got, &want);
}

#[test]
fn clip_takes_limits_from_inputs_and_attributes() {
    let x = pseudo(64, 29);
    // opset ≥11: min/max as scalar inputs
    let ir = graph(
        vec![node("Clip", &["x", "lo", "hi"], &["y"], &[])],
        vec![
            ("lo", f32_init(vec![], &[-0.25])),
            ("hi", f32_init(vec![], &[0.5])),
        ],
        "y",
    );
    let got = run(&ir, &[("x", HostTensor::from_f32(vec![64], &x))], "y");
    let want: Vec<f32> = x.iter().map(|v| v.clamp(-0.25, 0.5)).collect();
    assert_close(&got, &want);

    // opset 6: limits as attributes, and only one side constrained
    let ir = graph(
        vec![node(
            "Clip",
            &["x"],
            &["y"],
            &[("min", AttrValue::Float(0.0))],
        )],
        vec![],
        "y",
    );
    let got = run(&ir, &[("x", HostTensor::from_f32(vec![64], &x))], "y");
    let want: Vec<f32> = x.iter().map(|v| v.max(0.0)).collect();
    assert_close(&got, &want);
}

#[test]
fn erf_and_friends_match_the_cpu_math() {
    let x: Vec<f32> = (0..48).map(|i| (i as f32 - 24.0) / 8.0).collect();
    for (op, reference) in [
        ("Erf", (|v: f32| erf_ref(v)) as fn(f32) -> f32),
        ("Neg", |v| -v),
        ("Exp", f32::exp),
        ("Sqrt", |v: f32| v.abs().sqrt()),
        ("Tanh", f32::tanh),
    ] {
        // Sqrt receives positive values: the test uses |x| for that case
        let input: Vec<f32> = if op == "Sqrt" {
            x.iter().map(|v| v.abs()).collect()
        } else {
            x.clone()
        };
        let ir = graph(vec![node(op, &["x"], &["y"], &[])], vec![], "y");
        let got = run(
            &ir,
            &[("x", HostTensor::from_f32(vec![input.len() as i64], &input))],
            "y",
        );
        let want: Vec<f32> = input.iter().map(|v| reference(*v)).collect();
        assert_close(&got, &want);
    }
}

/// `Gelu`'s two `approximate` modes are checked against values computed in
/// double precision, not against the shader's own formula: the exact branch
/// leans on the same `erf` approximation as `Erf`, so comparing it with itself
/// would prove only the plumbing.
#[test]
fn gelu_matches_the_double_precision_reference() {
    let x = [-4.0f32, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0];
    for (approximate, want) in [
        (
            "none",
            [
                -0.000_126_684_97_f32,
                -0.045_500_264,
                -0.158_655_26,
                -0.154_268_77,
                0.0,
                0.345_731_23,
                0.841_344_8,
                1.954_499_7,
                3.999_873_4,
            ],
        ),
        (
            "tanh",
            [
                -7.024_595e-5,
                -0.045_402_307,
                -0.158_808_01,
                -0.154_286,
                0.0,
                0.345_714,
                0.841_192,
                1.954_597_7,
                3.999_929_7,
            ],
        ),
    ] {
        let ir = graph(
            vec![node(
                "Gelu",
                &["x"],
                &["y"],
                &[("approximate", AttrValue::String(approximate.into()))],
            )],
            vec![],
            "y",
        );
        let got = run(&ir, &[("x", HostTensor::from_f32(vec![9], &x))], "y");
        assert_close(&got, &want);
    }

    // no attribute means "none"
    let ir = graph(vec![node("Gelu", &["x"], &["y"], &[])], vec![], "y");
    let got = run(&ir, &[("x", HostTensor::from_f32(vec![9], &x))], "y");
    assert!(
        (got[8] - 3.999_873_4).abs() < 1e-5,
        "default = none: {got:?}"
    );
}

/// The kernel stages a 64×64 tile and gives each thread a 4×4 micro-tile, so
/// the shapes that matter are the ones that cross a tile boundary and the ones
/// that do not divide it. The test above is 5×4×3: a single workgroup entirely
/// inside the bounds checks, which never exercises the tiling at all.
#[test]
fn gemm_tiles_and_edges_match_the_cpu_reference() {
    // (M, K, N): one exact multiple of the tile, then edges on each dimension
    for (m, k, n) in [(128usize, 64usize, 128usize), (130, 70, 67), (70, 130, 3)] {
        for (trans_a, trans_b) in [(false, false), (true, false), (false, true), (true, true)] {
            let a = pseudo(m * k, 31);
            let b = pseudo(k * n, 37);
            let c = pseudo(n, 41);
            let (alpha, beta) = (0.75f32, -1.5f32);
            let a_shape = if trans_a {
                vec![k as i64, m as i64]
            } else {
                vec![m as i64, k as i64]
            };
            let b_shape = if trans_b {
                vec![n as i64, k as i64]
            } else {
                vec![k as i64, n as i64]
            };
            let ir = graph(
                vec![node(
                    "Gemm",
                    &["a", "b", "c"],
                    &["y"],
                    &[
                        ("transA", AttrValue::Int(i64::from(trans_a))),
                        ("transB", AttrValue::Int(i64::from(trans_b))),
                        ("alpha", AttrValue::Float(alpha)),
                        ("beta", AttrValue::Float(beta)),
                    ],
                )],
                vec![
                    ("b", f32_init(b_shape, &b)),
                    ("c", f32_init(vec![n as i64], &c)),
                ],
                "y",
            );
            let got = run(&ir, &[("a", HostTensor::from_f32(a_shape, &a))], "y");

            let mut want = vec![0.0f32; m * n];
            for (i, w) in want.iter_mut().enumerate() {
                let (row, col) = (i / n, i % n);
                let dot: f32 = (0..k)
                    .map(|t| {
                        let av = if trans_a {
                            a[t * m + row]
                        } else {
                            a[row * k + t]
                        };
                        let bv = if trans_b {
                            b[col * k + t]
                        } else {
                            b[t * n + col]
                        };
                        av * bv
                    })
                    .sum();
                *w = alpha * dot + beta * c[col];
            }
            assert_close(&got, &want);
        }
    }
}

#[test]
fn pow_handles_negative_bases_with_integer_exponents() {
    // ONNX defines (-2)^2 = 4; `pow` in WGSL is undefined for a negative base.
    // The tensor is intentionally large: below the threshold the computation would stay on the host.
    let x: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) / 512.0).collect();
    let ir = graph(
        vec![node("Pow", &["x", "e"], &["y"], &[])],
        vec![("e", f32_init(vec![], &[2.0]))],
        "y",
    );
    let got = run(
        &ir,
        &[("x", HostTensor::from_f32(vec![x.len() as i64], &x))],
        "y",
    );
    let want: Vec<f32> = x.iter().map(|v| v * v).collect();
    assert_close(&got, &want);

    // odd exponent: the sign is preserved
    let ir = graph(
        vec![node("Pow", &["x", "e"], &["y"], &[])],
        vec![("e", f32_init(vec![], &[3.0]))],
        "y",
    );
    let got = run(
        &ir,
        &[("x", HostTensor::from_f32(vec![x.len() as i64], &x))],
        "y",
    );
    let want: Vec<f32> = x.iter().map(|v| v * v * v).collect();
    assert_close(&got, &want);
}

/// Reference `erf` in double precision (no need for a series/continued
/// fraction: the same approximation evaluated in f64 is enough).
fn erf_ref(x: f32) -> f32 {
    let x = f64::from(x);
    let s = x.signum();
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    (s * (1.0 - poly * (-a * a).exp())) as f32
}
