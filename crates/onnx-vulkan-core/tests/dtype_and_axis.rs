//! Constraints beyond the op name: `MatMulInteger`/`ConvInteger` with `int8`
//! operands (ONNX allows them alongside `uint8`) and `Softmax` over any axis.
//!
//! These are the cases that used to fail at runtime in RF-DETR, SAM 3 and
//! YOLOv8 after the compiling EP had already claimed the nodes.

use onnx_vulkan_core::host_ops::{HostTensor, INT8, UINT8};
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

fn byte_init(dtype: i32, shape: Vec<i64>, data: Vec<u8>) -> InitializerIr {
    InitializerIr { dtype, shape, data }
}

fn run(ir: &GraphIr, inputs: &[(&str, HostTensor)], output: &str) -> HostTensor {
    let context = VkContext::new().expect("contesto Vulkan");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    for (name, tensor) in inputs {
        env.set(name, Tensor::Host(tensor.clone()));
    }
    execute(ir, &mut env).expect("graph execution");
    let out = env.host(output).expect("output on host").clone();
    env.finish();
    out
}

fn as_i32(t: &HostTensor) -> Vec<i32> {
    t.data
        .chunks_exact(4)
        .map(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// `MatMulInteger` with A and B of independent sign. `signed_a`/`signed_b`
/// choose how the bytes are interpreted; the CPU reference does the same.
fn matmul_integer_case(signed_a: bool, signed_b: bool) {
    const M: usize = 3;
    const K: usize = 8;
    const N: usize = 5;

    // identical raw bytes in both cases: only the interpretation changes
    let a_raw: Vec<u8> = (0..M * K).map(|i| ((i * 37) % 256) as u8).collect();
    let b_raw: Vec<u8> = (0..K * N).map(|i| ((i * 53) % 256) as u8).collect();
    let (a_dtype, b_dtype) = (
        if signed_a { INT8 } else { UINT8 },
        if signed_b { INT8 } else { UINT8 },
    );
    let a_zp_raw: u8 = if signed_a { 0xf9 } else { 7 }; // -7 as int8
    let b_zp_raw: u8 = if signed_b { 0x85 } else { 200 }; // 0x85 = -123 as int8

    let val = |raw: u8, signed: bool| -> i32 {
        if signed {
            i32::from(raw as i8)
        } else {
            i32::from(raw)
        }
    };
    let (a_zp, b_zp) = (val(a_zp_raw, signed_a), val(b_zp_raw, signed_b));
    let mut want = vec![0i32; M * N];
    for (row, out) in want.chunks_exact_mut(N).enumerate() {
        for (col, cell) in out.iter_mut().enumerate() {
            *cell = (0..K)
                .map(|k| {
                    (val(a_raw[row * K + k], signed_a) - a_zp)
                        * (val(b_raw[k * N + col], signed_b) - b_zp)
                })
                .sum();
        }
    }

    let mut initializers = HashMap::new();
    initializers.insert(
        "b".to_string(),
        byte_init(b_dtype, vec![K as i64, N as i64], b_raw),
    );
    initializers.insert(
        "a_zp".to_string(),
        byte_init(a_dtype, vec![], vec![a_zp_raw]),
    );
    initializers.insert(
        "b_zp".to_string(),
        byte_init(b_dtype, vec![], vec![b_zp_raw]),
    );
    let ir = GraphIr {
        nodes: vec![node(
            "MatMulInteger",
            &["a", "b", "a_zp", "b_zp"],
            &["out"],
            &[],
        )],
        initializers,
        inputs: vec!["a".to_string()],
        outputs: vec!["out".to_string()],
    };

    let out = run(
        &ir,
        &[(
            "a",
            HostTensor::new(a_dtype, vec![M as i64, K as i64], a_raw),
        )],
        "out",
    );
    assert_eq!(out.shape, vec![M as i64, N as i64]);
    assert_eq!(
        as_i32(&out),
        want,
        "MatMulInteger signed_a={signed_a} signed_b={signed_b}"
    );
}

#[test]
fn matmul_integer_unsigned_both() {
    matmul_integer_case(false, false);
}

#[test]
fn matmul_integer_signed_weights() {
    // the case of real models: uint8 activations, int8 weights
    matmul_integer_case(false, true);
}

#[test]
fn matmul_integer_signed_both() {
    matmul_integer_case(true, true);
}

#[test]
fn conv_integer_with_signed_weights() {
    // 1D, group=1, kernel 3, stride 1, no pad: X uint8, W int8
    const C_IN: usize = 2;
    const C_OUT: usize = 3;
    const L: usize = 6;
    const KS: usize = 3;
    let l_out = L - KS + 1;

    let x_raw: Vec<u8> = (0..C_IN * L).map(|i| ((i * 29) % 256) as u8).collect();
    let w_raw: Vec<u8> = (0..C_OUT * C_IN * KS)
        .map(|i| ((i * 47) % 256) as u8)
        .collect();
    let x_zp: u8 = 120;
    let w_zp_raw: u8 = 0xfb; // -5 as int8

    let w_zp = i32::from(w_zp_raw as i8);
    let mut want = vec![0i32; C_OUT * l_out];
    for m in 0..C_OUT {
        for o in 0..l_out {
            let mut acc = 0i32;
            for c in 0..C_IN {
                for k in 0..KS {
                    let xv = i32::from(x_raw[c * L + o + k]) - i32::from(x_zp);
                    let wv = i32::from(w_raw[(m * C_IN + c) * KS + k] as i8) - w_zp;
                    acc += xv * wv;
                }
            }
            want[m * l_out + o] = acc;
        }
    }

    let mut initializers = HashMap::new();
    initializers.insert(
        "w".to_string(),
        byte_init(
            INT8,
            vec![C_OUT as i64, C_IN as i64, KS as i64],
            w_raw.clone(),
        ),
    );
    initializers.insert("x_zp".to_string(), byte_init(UINT8, vec![], vec![x_zp]));
    initializers.insert("w_zp".to_string(), byte_init(INT8, vec![], vec![w_zp_raw]));
    let ir = GraphIr {
        nodes: vec![node(
            "ConvInteger",
            &["x", "w", "x_zp", "w_zp"],
            &["out"],
            &[("kernel_shape", AttrValue::Ints(vec![KS as i64]))],
        )],
        initializers,
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
    };

    let out = run(
        &ir,
        &[(
            "x",
            HostTensor::new(UINT8, vec![1, C_IN as i64, L as i64], x_raw),
        )],
        "out",
    );
    assert_eq!(out.shape, vec![1, C_OUT as i64, l_out as i64]);
    assert_eq!(as_i32(&out), want, "ConvInteger con pesi int8");
}

/// Reference Softmax along `axis` of a tensor with shape `shape`.
fn softmax_reference(x: &[f32], shape: &[usize], axis: usize) -> Vec<f32> {
    let c = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let rows = x.len() / c;
    let mut out = vec![0.0f32; x.len()];
    for row in 0..rows {
        let base = (row / inner) * c * inner + (row % inner);
        let mut m = f32::NEG_INFINITY;
        for i in 0..c {
            m = m.max(x[base + i * inner]);
        }
        let sum: f32 = (0..c).map(|i| (x[base + i * inner] - m).exp()).sum();
        for i in 0..c {
            out[base + i * inner] = (x[base + i * inner] - m).exp() / sum;
        }
    }
    out
}

fn softmax_case(shape: &[usize], axis: i64) {
    let n: usize = shape.iter().product();
    let mut state = 7u64;
    let x: Vec<f32> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 29) as f32) - 2.0
        })
        .collect();
    let resolved = if axis < 0 {
        (axis + shape.len() as i64) as usize
    } else {
        axis as usize
    };
    let want = softmax_reference(&x, shape, resolved);

    let ir = GraphIr {
        nodes: vec![node(
            "Softmax",
            &["x"],
            &["out"],
            &[("axis", AttrValue::Int(axis))],
        )],
        initializers: HashMap::new(),
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
    };
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let out = run(&ir, &[("x", HostTensor::from_f32(dims.clone(), &x))], "out");
    assert_eq!(out.shape, dims);
    let got = out.to_f32().expect("output f32");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5,
            "Softmax shape={shape:?} axis={axis}: element {i}: {g} vs expected {w}"
        );
    }
}

#[test]
fn softmax_last_axis_stays_correct() {
    softmax_case(&[2, 3, 7], -1);
}

#[test]
fn softmax_middle_axis() {
    // the YOLOv8 case: rank 4, axis 1, rows with stride
    softmax_case(&[2, 4, 3, 5], 1);
}

#[test]
fn softmax_first_axis() {
    softmax_case(&[5, 2, 3], 0);
}

#[test]
fn softmax_more_rows_than_one_grid_dimension() {
    // 40000 rows: beyond the 32768 workgroup limit on the x axis, so the
    // dispatch must fall back to the 2D grid
    softmax_case(&[40000, 4], -1);
}
