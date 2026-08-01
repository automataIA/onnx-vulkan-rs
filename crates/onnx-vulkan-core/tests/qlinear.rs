//! The QOperator (static int8) kernels: `QLinearConv`, `QLinearMatMul`,
//! `QLinearAdd`, `QLinearGlobalAveragePool`, and `MaxPool` on quantized input.
//!
//! The reference here is the operator's definition evaluated on the host in
//! **integer** arithmetic — accumulate in i32, requantize once — which is the
//! whole point of the format: it is not "close to" the float computation, it
//! is a different and exactly reproducible one. That is why the assertions are
//! on equality and not on a tolerance.
//!
//! End to end these kernels are checked against the ONNX model zoo's own
//! `test_data_set_0` for resnet50-int8 and mobilenetv2-int8, where the output
//! comes out **bit-exact** (`cronologia.md`, 2026-07-31).

use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, InitializerIr, KernelCache, NodeIr, execute,
};
use std::collections::HashMap;
use vk_compute::VkContext;

const FLOAT: i32 = 1;
const UINT8: i32 = 2;
const INT8: i32 = 3;
const INT32: i32 = 6;

fn node(op: &str, inputs: &[&str], attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: 10,
        name: format!("{op}_0"),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: vec!["out".to_string()],
        attrs: attrs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect(),
    }
}

fn init(dtype: i32, shape: &[i64], data: Vec<u8>) -> InitializerIr {
    InitializerIr {
        dtype,
        shape: shape.to_vec(),
        data,
    }
}

fn f32s(shape: &[i64], values: &[f32]) -> InitializerIr {
    init(
        FLOAT,
        shape,
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

fn i32s(shape: &[i64], values: &[i32]) -> InitializerIr {
    init(
        INT32,
        shape,
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

fn bytes(dtype: i32, shape: &[i64], values: &[i32]) -> InitializerIr {
    init(
        dtype,
        shape,
        values.iter().map(|v| *v as u8).collect::<Vec<u8>>(),
    )
}

/// Runs a single-node graph whose inputs are all initializers.
fn run(node: NodeIr, values: &[(&str, InitializerIr)]) -> Vec<i64> {
    let inputs = node.inputs.clone();
    let ir = GraphIr {
        nodes: vec![node],
        initializers: values
            .iter()
            .map(|(name, t)| ((*name).to_string(), t.clone()))
            .collect::<HashMap<_, _>>(),
        inputs,
        outputs: vec!["out".to_string()],
        ..Default::default()
    };
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    execute(&ir, &mut env).expect("graph execution");
    let out = env.host("out").expect("output on host").to_i64().unwrap();
    env.finish();
    out
}

/// The epilogue as the ONNX reference defines it, on the host.
fn requantize(acc: i32, ratio: f32, y_zp: i32, signed: bool) -> i64 {
    let q = (acc as f32 * ratio).round_ties_even() as i32 + y_zp;
    i64::from(if signed {
        q.clamp(-128, 127)
    } else {
        q.clamp(0, 255)
    })
}

/// A 2×2 convolution over a 3×3 input, two output channels with **per-channel**
/// weight scales — the shape every `QLinearConv` of the two zoo models has.
#[test]
fn a_convolution_accumulates_in_int32_and_requantizes_once() {
    let x: Vec<i32> = vec![130, 120, 140, 128, 200, 60, 10, 250, 128];
    let w: Vec<i32> = vec![1, -2, 3, 4, -5, 6, -7, 8];
    let (x_scale, y_scale) = (0.02f32, 0.05f32);
    let w_scale = [0.01f32, 0.03];
    let (x_zp, y_zp) = (128i32, 10i32);
    let bias = [7i32, -13];

    let out = run(
        node(
            "QLinearConv",
            &[
                "x", "x_scale", "x_zp", "w", "w_scale", "w_zp", "y_scale", "y_zp", "B",
            ],
            &[("kernel_shape", AttrValue::Ints(vec![2, 2]))],
        ),
        &[
            ("x", bytes(UINT8, &[1, 1, 3, 3], &x)),
            ("x_scale", f32s(&[], &[x_scale])),
            ("x_zp", bytes(UINT8, &[], &[x_zp])),
            ("w", bytes(INT8, &[2, 1, 2, 2], &w)),
            ("w_scale", f32s(&[2], &w_scale)),
            ("w_zp", bytes(INT8, &[2], &[0, 0])),
            ("y_scale", f32s(&[], &[y_scale])),
            ("y_zp", bytes(UINT8, &[], &[y_zp])),
            ("B", i32s(&[2], &bias)),
        ],
    );

    // integer reference: 2×2 window, stride 1, no padding → 2×2 output
    let mut want = Vec::new();
    for (c, scale) in w_scale.iter().enumerate() {
        for oh in 0..2usize {
            for ow in 0..2usize {
                let mut acc = bias[c];
                for r in 0..2usize {
                    for s in 0..2usize {
                        let xv = x[(oh + r) * 3 + ow + s] - x_zp;
                        acc += xv * w[c * 4 + r * 2 + s];
                    }
                }
                want.push(requantize(acc, x_scale * scale / y_scale, y_zp, false));
            }
        }
    }
    assert_eq!(out, want);
}

/// The same node without a bias: the schema makes it optional, and the kernel
/// must not read the binding it still has to bind.
#[test]
fn a_convolution_without_bias_is_the_same_computation() {
    let x: Vec<i32> = vec![200, 10, 50, 128];
    let w: Vec<i32> = vec![2, -3, 4, 5];
    let out = run(
        node(
            "QLinearConv",
            &[
                "x", "x_scale", "x_zp", "w", "w_scale", "w_zp", "y_scale", "y_zp",
            ],
            &[("kernel_shape", AttrValue::Ints(vec![2, 2]))],
        ),
        &[
            ("x", bytes(UINT8, &[1, 1, 2, 2], &x)),
            ("x_scale", f32s(&[], &[0.03])),
            ("x_zp", bytes(UINT8, &[], &[100])),
            ("w", bytes(INT8, &[1, 1, 2, 2], &w)),
            ("w_scale", f32s(&[1], &[0.02])),
            ("w_zp", bytes(INT8, &[1], &[0])),
            ("y_scale", f32s(&[], &[0.1])),
            ("y_zp", bytes(UINT8, &[], &[20])),
        ],
    );
    let acc: i32 = x
        .iter()
        .zip(&w)
        .map(|(xv, wv)| (xv - 100) * wv)
        .sum::<i32>();
    assert_eq!(out, vec![requantize(acc, 0.03 * 0.02 / 0.1, 20, false)]);
}

/// `QLinearMatMul` with a per-column scale — mobilenetv2's form. resnet50
/// passes one scalar instead, which is the same shader with `axis_len = 1`.
#[test]
fn a_matmul_scales_per_column() {
    let a: Vec<i32> = vec![130, 120, 140, 128];
    let b: Vec<i32> = vec![1, -2, 3, 4, -5, 6, 7, -8];
    let (a_scale, y_scale) = (0.02f32, 0.05f32);
    let b_scale = [0.01f32, 0.03];
    let (a_zp, y_zp) = (128i32, 10i32);

    let out = run(
        node(
            "QLinearMatMul",
            &[
                "a", "a_scale", "a_zp", "b", "b_scale", "b_zp", "y_scale", "y_zp",
            ],
            &[],
        ),
        &[
            ("a", bytes(UINT8, &[1, 4], &a)),
            ("a_scale", f32s(&[], &[a_scale])),
            ("a_zp", bytes(UINT8, &[], &[a_zp])),
            ("b", bytes(INT8, &[4, 2], &b)),
            ("b_scale", f32s(&[2], &b_scale)),
            ("b_zp", bytes(INT8, &[2], &[0, 0])),
            ("y_scale", f32s(&[], &[y_scale])),
            ("y_zp", bytes(UINT8, &[], &[y_zp])),
        ],
    );

    let want: Vec<i64> = (0..2)
        .map(|col| {
            let acc: i32 = (0..4).map(|k| (a[k] - a_zp) * b[k * 2 + col]).sum();
            requantize(acc, a_scale * b_scale[col] / y_scale, y_zp, false)
        })
        .collect();
    assert_eq!(out, want);
}

/// `QLinearAdd` with the broadcast both models end on: a `[1000]` bias added
/// to `[N, 1000]` logits.
#[test]
fn an_add_broadcasts_and_lands_on_the_output_scale() {
    let a: Vec<i32> = vec![200, 10, 50, 128, 255, 0];
    let b: Vec<i32> = vec![100, 130, 20];
    let (a_scale, b_scale, c_scale) = (0.02f32, 0.05f32, 0.03f32);
    let (a_zp, b_zp, c_zp) = (128i32, 100i32, 30i32);

    let out = run(
        node(
            "QLinearAdd",
            &[
                "A", "A_scale", "A_zp", "B", "B_scale", "B_zp", "C_scale", "C_zp",
            ],
            &[],
        ),
        &[
            ("A", bytes(UINT8, &[2, 3], &a)),
            ("A_scale", f32s(&[], &[a_scale])),
            ("A_zp", bytes(UINT8, &[], &[a_zp])),
            ("B", bytes(UINT8, &[3], &b)),
            ("B_scale", f32s(&[], &[b_scale])),
            ("B_zp", bytes(UINT8, &[], &[b_zp])),
            ("C_scale", f32s(&[], &[c_scale])),
            ("C_zp", bytes(UINT8, &[], &[c_zp])),
        ],
    );

    let want: Vec<i64> = (0..6)
        .map(|i| {
            let sum = a_scale / c_scale * (a[i] - a_zp) as f32
                + b_scale / c_scale * (b[i % 3] - b_zp) as f32;
            i64::from((sum.round_ties_even() as i32 + c_zp).clamp(0, 255))
        })
        .collect();
    assert_eq!(out, want);
}

/// `MaxPool` between two quantized convolutions, which is where resnet50-int8
/// puts one. Quantization is monotone, so this is the maximum of the codes —
/// no scale and no zero point enter the computation.
#[test]
fn max_pooling_stays_in_the_quantized_domain() {
    let x: Vec<i32> = vec![
        10, 200, 30, 40, //
        50, 60, 250, 80, //
        90, 100, 110, 120, //
        130, 140, 150, 160,
    ];
    let out = run(
        node(
            "MaxPool",
            &["x"],
            &[
                ("kernel_shape", AttrValue::Ints(vec![2, 2])),
                ("strides", AttrValue::Ints(vec![2, 2])),
            ],
        ),
        &[("x", bytes(UINT8, &[1, 1, 4, 4], &x))],
    );
    assert_eq!(out, vec![200, 250, 140, 160]);
}

/// `QLinearGlobalAveragePool` is lowered to dequantize → pool → quantize, and
/// the assertion is on the numbers that lowering must produce.
#[test]
fn the_global_average_pool_averages_in_the_real_domain() {
    let x: Vec<i32> = vec![10, 20, 30, 40, 200, 210, 220, 230];
    let (x_scale, y_scale) = (0.02f32, 0.05f32);
    let (x_zp, y_zp) = (15i32, 7i32);
    let out = run(
        node(
            "QLinearGlobalAveragePool",
            &["x", "x_scale", "x_zp", "y_scale", "y_zp"],
            &[],
        ),
        &[
            ("x", bytes(UINT8, &[1, 2, 2, 2], &x)),
            ("x_scale", f32s(&[], &[x_scale])),
            ("x_zp", bytes(UINT8, &[], &[x_zp])),
            ("y_scale", f32s(&[], &[y_scale])),
            ("y_zp", bytes(UINT8, &[], &[y_zp])),
        ],
    );

    let want: Vec<i64> = x
        .chunks(4)
        .map(|plane| {
            let mean = plane
                .iter()
                .map(|v| (v - x_zp) as f32 * x_scale)
                .sum::<f32>()
                / 4.0;
            i64::from(((mean / y_scale).round_ties_even() as i32 + y_zp).clamp(0, 255))
        })
        .collect();
    assert_eq!(out, want);
}
