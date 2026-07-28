//! Floating-point `Conv` and the `QuantizeLinear`/`DequantizeLinear` pair
//! (QDQ form) run by the core on synthetic `GraphIr`s, compared against a
//! CPU reference implementation written here.

use onnx_vulkan_core::host_ops::{FLOAT, HostTensor, INT8, UINT8};
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

fn byte_init(dtype: i32, shape: Vec<i64>, data: Vec<u8>) -> InitializerIr {
    InitializerIr { dtype, shape, data }
}

/// Reproducible pseudo-random numbers in `[-1, 1)`, with no external dependencies.
fn pseudo(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        })
        .collect()
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

fn to_f32(t: &HostTensor) -> Vec<f32> {
    t.to_f32().expect("output convertibile in f32")
}

/// CPU reference for 2D `Conv` (the 1D case is obtained with `w_in = kw = 1`).
#[allow(clippy::too_many_arguments)]
struct ConvRef {
    n: usize,
    c_in: usize,
    c_out: usize,
    group: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    /// Padding (begin, end) on H and on W: `auto_pad` may make them asymmetric.
    pad: ((usize, usize), (usize, usize)),
    dil: (usize, usize),
}

impl ConvRef {
    fn out_dims(&self) -> (usize, usize) {
        let h = (self.h_in + self.pad.0.0 + self.pad.0.1 - (self.dil.0 * (self.kh - 1) + 1))
            / self.stride.0
            + 1;
        let w = (self.w_in + self.pad.1.0 + self.pad.1.1 - (self.dil.1 * (self.kw - 1) + 1))
            / self.stride.1
            + 1;
        (h, w)
    }

    fn apply(&self, x: &[f32], w: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
        let (h_out, w_out) = self.out_dims();
        let gsi = self.c_in / self.group;
        let gso = self.c_out / self.group;
        let mut out = vec![0.0f32; self.n * self.c_out * h_out * w_out];
        for bn in 0..self.n {
            for m in 0..self.c_out {
                let g = m / gso;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = bias.map_or(0.0, |b| b[m]);
                        for cg in 0..gsi {
                            let ic = g * gsi + cg;
                            for r in 0..self.kh {
                                let ih = (oh * self.stride.0 + r * self.dil.0) as isize
                                    - self.pad.0.0 as isize;
                                if ih < 0 || ih >= self.h_in as isize {
                                    continue;
                                }
                                for s in 0..self.kw {
                                    let iw = (ow * self.stride.1 + s * self.dil.1) as isize
                                        - self.pad.1.0 as isize;
                                    if iw < 0 || iw >= self.w_in as isize {
                                        continue;
                                    }
                                    let xi = ((bn * self.c_in + ic) * self.h_in + ih as usize)
                                        * self.w_in
                                        + iw as usize;
                                    let wi = ((m * gsi + cg) * self.kh + r) * self.kw + s;
                                    acc += x[xi] * w[wi];
                                }
                            }
                        }
                        let oi = ((bn * self.c_out + m) * h_out + oh) * w_out + ow;
                        out[oi] = acc;
                    }
                }
            }
        }
        out
    }
}

fn conv_graph(attrs: &[(&str, AttrValue)], with_bias: bool) -> Vec<NodeIr> {
    let inputs: &[&str] = if with_bias {
        &["x", "w", "b"]
    } else {
        &["x", "w"]
    };
    vec![node("Conv", inputs, &["out"], attrs)]
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lunghezze diverse");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}: element {i}: {g} vs expected {w}"
        );
    }
}

#[test]
fn conv_f32_with_stride_pad_dilation_and_bias() {
    let r = ConvRef {
        n: 1,
        c_in: 3,
        c_out: 4,
        group: 1,
        h_in: 7,
        w_in: 9,
        kh: 3,
        kw: 3,
        stride: (2, 1),
        pad: ((1, 1), (2, 2)),
        dil: (1, 2),
    };
    let x = pseudo(r.n * r.c_in * r.h_in * r.w_in, 11);
    let w = pseudo(r.c_out * r.c_in * r.kh * r.kw, 23);
    let bias = pseudo(r.c_out, 37);
    let want = r.apply(&x, &w, Some(&bias));
    let (h_out, w_out) = r.out_dims();

    let mut initializers = HashMap::new();
    initializers.insert(
        "w".to_string(),
        f32_init(
            vec![r.c_out as i64, r.c_in as i64, r.kh as i64, r.kw as i64],
            &w,
        ),
    );
    initializers.insert("b".to_string(), f32_init(vec![r.c_out as i64], &bias));
    let ir = GraphIr {
        nodes: conv_graph(
            &[
                ("kernel_shape", AttrValue::Ints(vec![3, 3])),
                ("strides", AttrValue::Ints(vec![2, 1])),
                ("pads", AttrValue::Ints(vec![1, 2, 1, 2])),
                ("dilations", AttrValue::Ints(vec![1, 2])),
            ],
            true,
        ),
        initializers,
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
    };

    let out = run(
        &ir,
        &[(
            "x",
            HostTensor::from_f32(
                vec![r.n as i64, r.c_in as i64, r.h_in as i64, r.w_in as i64],
                &x,
            ),
        )],
        "out",
    );
    assert_eq!(
        out.shape,
        vec![r.n as i64, r.c_out as i64, h_out as i64, w_out as i64]
    );
    assert_close(&to_f32(&out), &want, 1e-4, "Conv 2D");
}

/// `K = C_in·KH·KW = 288` with a batch, so `conv::split_k` routes this to the
/// split-K pair: the 64×64 tile over four slices of K plus a reduction. The
/// batch matters — `wid.z` carries the image and the slice together — and so
/// does the bias, which only the reduction may apply.
#[test]
fn conv_f32_split_k_over_a_batch_matches_the_cpu_reference() {
    let r = ConvRef {
        n: 2,
        c_in: 32,
        c_out: 24,
        group: 1,
        h_in: 9,
        w_in: 9,
        kh: 3,
        kw: 3,
        stride: (1, 1),
        pad: ((1, 1), (1, 1)),
        dil: (1, 1),
    };
    assert_eq!(
        onnx_vulkan_core::shaders::conv::split_k(81, r.c_out, r.c_in * r.kh * r.kw),
        Some(4),
        "this geometry is the one under test only if it routes to split-K"
    );
    let x = pseudo(r.n * r.c_in * r.h_in * r.w_in, 13);
    let w = pseudo(r.c_out * r.c_in * r.kh * r.kw, 29);
    let bias = pseudo(r.c_out, 41);
    let want = r.apply(&x, &w, Some(&bias));
    let (h_out, w_out) = r.out_dims();

    let mut initializers = HashMap::new();
    initializers.insert(
        "w".to_string(),
        f32_init(
            vec![r.c_out as i64, r.c_in as i64, r.kh as i64, r.kw as i64],
            &w,
        ),
    );
    initializers.insert("b".to_string(), f32_init(vec![r.c_out as i64], &bias));
    let ir = GraphIr {
        nodes: conv_graph(
            &[
                ("kernel_shape", AttrValue::Ints(vec![3, 3])),
                ("strides", AttrValue::Ints(vec![1, 1])),
                ("pads", AttrValue::Ints(vec![1, 1, 1, 1])),
            ],
            true,
        ),
        initializers,
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
    };

    let out = run(
        &ir,
        &[(
            "x",
            HostTensor::from_f32(
                vec![r.n as i64, r.c_in as i64, r.h_in as i64, r.w_in as i64],
                &x,
            ),
        )],
        "out",
    );
    assert_eq!(
        out.shape,
        vec![r.n as i64, r.c_out as i64, h_out as i64, w_out as i64]
    );
    assert_close(&to_f32(&out), &want, 1e-4, "Conv split-K");
}

#[test]
fn conv_f32_depthwise_without_bias() {
    let r = ConvRef {
        n: 1,
        c_in: 4,
        c_out: 4,
        group: 4,
        h_in: 5,
        w_in: 5,
        kh: 3,
        kw: 3,
        stride: (1, 1),
        pad: ((1, 1), (1, 1)),
        dil: (1, 1),
    };
    let x = pseudo(r.c_in * r.h_in * r.w_in, 5);
    let w = pseudo(r.c_out * r.kh * r.kw, 9); // gsi = 1 (depthwise)
    let want = r.apply(&x, &w, None);

    let mut initializers = HashMap::new();
    initializers.insert(
        "w".to_string(),
        f32_init(vec![r.c_out as i64, 1, r.kh as i64, r.kw as i64], &w),
    );
    let ir = GraphIr {
        nodes: conv_graph(
            &[
                ("group", AttrValue::Int(4)),
                ("kernel_shape", AttrValue::Ints(vec![3, 3])),
                ("pads", AttrValue::Ints(vec![1, 1, 1, 1])),
            ],
            false,
        ),
        initializers,
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
    };

    let out = run(
        &ir,
        &[(
            "x",
            HostTensor::from_f32(vec![1, r.c_in as i64, r.h_in as i64, r.w_in as i64], &x),
        )],
        "out",
    );
    assert_eq!(out.shape, vec![1, 4, 5, 5]);
    assert_close(&to_f32(&out), &want, 1e-4, "Conv depthwise");
}

#[test]
fn conv_f32_1d_with_same_upper_padding() {
    // auto_pad SAME_UPPER: output size is ceil(in / stride), padding
    // distributed with the extra at the end.
    let (c_in, c_out, l_in, k, stride) = (2usize, 3usize, 6usize, 3usize, 2usize);
    let x = pseudo(c_in * l_in, 77);
    let w = pseudo(c_out * c_in * k, 91);

    let l_out = l_in.div_ceil(stride);
    let needed = (l_out - 1) * stride + k - l_in;
    let begin = needed / 2;
    // interesting case: SAME_UPPER produces asymmetric padding (0 at the start, 1 at the end)
    assert_ne!(begin, needed - begin, "caso di test con pad asimmetrico");
    let r = ConvRef {
        n: 1,
        c_in,
        c_out,
        group: 1,
        h_in: l_in,
        w_in: 1,
        kh: k,
        kw: 1,
        stride: (stride, 1),
        pad: ((begin, needed - begin), (0, 0)),
        dil: (1, 1),
    };
    let want = r.apply(&x, &w, None);

    let mut initializers = HashMap::new();
    initializers.insert(
        "w".to_string(),
        f32_init(vec![c_out as i64, c_in as i64, k as i64], &w),
    );
    let ir = GraphIr {
        nodes: conv_graph(
            &[
                ("kernel_shape", AttrValue::Ints(vec![k as i64])),
                ("strides", AttrValue::Ints(vec![stride as i64])),
                ("auto_pad", AttrValue::String("SAME_UPPER".to_string())),
            ],
            false,
        ),
        initializers,
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
    };

    let out = run(
        &ir,
        &[(
            "x",
            HostTensor::from_f32(vec![1, c_in as i64, l_in as i64], &x),
        )],
        "out",
    );
    assert_eq!(out.shape, vec![1, c_out as i64, l_out as i64]);
    assert_close(&to_f32(&out), &want, 1e-4, "Conv 1D SAME_UPPER");
}

/// `QuantizeLinear` → `DequantizeLinear`: the round-trip must stay within
/// half a quantization step, per-tensor and per-axis, for u8 and i8.
fn qdq_roundtrip(dtype: i32, per_axis: bool) {
    let (c, inner) = (3usize, 4usize);
    let n = c * inner;
    let x: Vec<f32> = pseudo(n, 101).iter().map(|v| v * 10.0).collect();
    let scales: Vec<f32> = if per_axis {
        vec![0.05, 0.1, 0.2]
    } else {
        vec![0.1]
    };
    let zero: Vec<u8> = if dtype == UINT8 {
        if per_axis {
            vec![128, 100, 90]
        } else {
            vec![128]
        }
    } else if per_axis {
        vec![0u8, 250, 6] // -0, -6, +6 in two's complement
    } else {
        vec![0]
    };

    let scale_shape = if per_axis { vec![c as i64] } else { vec![] };
    let mut initializers = HashMap::new();
    initializers.insert("scale".to_string(), f32_init(scale_shape.clone(), &scales));
    initializers.insert(
        "zp".to_string(),
        byte_init(dtype, scale_shape.clone(), zero.clone()),
    );

    let axis_attr: &[(&str, AttrValue)] = &[("axis", AttrValue::Int(1))];
    let ir = GraphIr {
        nodes: vec![
            node("QuantizeLinear", &["x", "scale", "zp"], &["q"], axis_attr),
            node(
                "DequantizeLinear",
                &["q", "scale", "zp"],
                &["out"],
                axis_attr,
            ),
        ],
        initializers,
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
    };

    let out = run(
        &ir,
        &[(
            "x",
            HostTensor::from_f32(vec![1, c as i64, inner as i64], &x),
        )],
        "out",
    );
    assert_eq!(out.shape, vec![1, c as i64, inner as i64]);
    let got = to_f32(&out);

    // reference: saturated quantization with the same per-channel step
    let (lo, hi) = if dtype == UINT8 {
        (0.0f32, 255.0f32)
    } else {
        (-128.0, 127.0)
    };
    for i in 0..n {
        let c_idx = if per_axis { i / inner } else { 0 };
        let s = scales[c_idx];
        let z = if dtype == UINT8 {
            f32::from(zero[c_idx])
        } else {
            f32::from(zero[c_idx] as i8)
        };
        let q = ((x[i] / s).round_ties_even() + z).clamp(lo, hi);
        let want = (q - z) * s;
        assert!(
            (got[i] - want).abs() <= 1e-4,
            "dtype {dtype} per_axis {per_axis}: element {i}: {} vs expected {want}",
            got[i]
        );
    }
}

#[test]
fn qdq_roundtrip_u8_per_tensor() {
    qdq_roundtrip(UINT8, false);
}

#[test]
fn qdq_roundtrip_u8_per_axis() {
    qdq_roundtrip(UINT8, true);
}

#[test]
fn qdq_roundtrip_i8_per_tensor() {
    qdq_roundtrip(INT8, false);
}

#[test]
fn qdq_roundtrip_i8_per_axis() {
    qdq_roundtrip(INT8, true);
}

#[test]
fn qdq_conv_matches_float_conv_within_quantization_error() {
    // The real pattern of int8 CNN models: DequantizeLinear on input and
    // weights (per-channel), float Conv, QuantizeLinear on the output.
    let r = ConvRef {
        n: 1,
        c_in: 2,
        c_out: 3,
        group: 1,
        h_in: 5,
        w_in: 5,
        kh: 3,
        kw: 3,
        stride: (1, 1),
        pad: ((1, 1), (1, 1)),
        dil: (1, 1),
    };
    let x_scale = 0.02f32;
    let x_zp = 128u8;
    let w_scales = [0.01f32, 0.02, 0.03];

    let x_q: Vec<u8> = (0..r.c_in * r.h_in * r.w_in)
        .map(|i| ((i * 37) % 256) as u8)
        .collect();
    let w_q: Vec<u8> = (0..r.c_out * r.c_in * r.kh * r.kw)
        .map(|i| (((i * 53) % 200) as i32 - 100) as i8 as u8)
        .collect();

    // expected dequantized values, used by the CPU reference
    let x_f: Vec<f32> = x_q
        .iter()
        .map(|&q| (f32::from(q) - f32::from(x_zp)) * x_scale)
        .collect();
    let per_w = r.c_in * r.kh * r.kw;
    let w_f: Vec<f32> = w_q
        .iter()
        .enumerate()
        .map(|(i, &q)| f32::from(q as i8) * w_scales[i / per_w])
        .collect();
    let want = r.apply(&x_f, &w_f, None);

    let out_scale = 0.05f32;
    let out_zp = 100u8;
    let mut initializers = HashMap::new();
    initializers.insert(
        "w_q".to_string(),
        byte_init(
            INT8,
            vec![r.c_out as i64, r.c_in as i64, r.kh as i64, r.kw as i64],
            w_q,
        ),
    );
    initializers.insert(
        "w_scale".to_string(),
        f32_init(vec![r.c_out as i64], &w_scales),
    );
    initializers.insert(
        "w_zp".to_string(),
        byte_init(INT8, vec![r.c_out as i64], vec![0; r.c_out]),
    );
    initializers.insert("x_scale".to_string(), f32_init(vec![], &[x_scale]));
    initializers.insert("x_zp".to_string(), byte_init(UINT8, vec![], vec![x_zp]));
    initializers.insert("o_scale".to_string(), f32_init(vec![], &[out_scale]));
    initializers.insert("o_zp".to_string(), byte_init(UINT8, vec![], vec![out_zp]));

    let ir = GraphIr {
        nodes: vec![
            node(
                "DequantizeLinear",
                &["x_q", "x_scale", "x_zp"],
                &["x_f"],
                &[],
            ),
            // per-channel quantized weights: axis 0 of the W tensor
            node(
                "DequantizeLinear",
                &["w_q", "w_scale", "w_zp"],
                &["w_f"],
                &[("axis", AttrValue::Int(0))],
            ),
            node(
                "Conv",
                &["x_f", "w_f"],
                &["conv"],
                &[
                    ("kernel_shape", AttrValue::Ints(vec![3, 3])),
                    ("pads", AttrValue::Ints(vec![1, 1, 1, 1])),
                ],
            ),
            node(
                "QuantizeLinear",
                &["conv", "o_scale", "o_zp"],
                &["out"],
                &[],
            ),
        ],
        initializers,
        inputs: vec!["x_q".to_string()],
        outputs: vec!["out".to_string()],
    };

    let out = run(
        &ir,
        &[(
            "x_q",
            HostTensor::new(
                UINT8,
                vec![1, r.c_in as i64, r.h_in as i64, r.w_in as i64],
                x_q,
            ),
        )],
        "out",
    );
    assert_eq!(out.dtype, UINT8);
    assert_eq!(out.shape, vec![1, 3, 5, 5]);

    // the quantized output must stay within one step of the reference float
    // conv, re-quantized with the same scale
    let got: Vec<f32> = out.data.iter().map(|&q| f32::from(q)).collect();
    for (i, &w) in want.iter().enumerate() {
        let want_q = ((w / out_scale).round_ties_even() + f32::from(out_zp)).clamp(0.0, 255.0);
        assert!(
            (got[i] - want_q).abs() <= 1.0,
            "element {i}: {} vs expected {want_q}",
            got[i]
        );
    }
}
