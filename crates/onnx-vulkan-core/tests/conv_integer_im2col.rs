//! `ConvInteger` on the cooperative matrix path must give the exact `i32`
//! sums, whichever im2col feeds it: the HWC one (K ordered `(kh, kw, c)`, the
//! weight permuted to match) when the channels and the plane start on words,
//! the CHW one otherwise. The expected sums are computed here, directly from
//! the definition.

use onnx_vulkan_core::host_ops::{INT8, UINT8};
use onnx_vulkan_core::shaders::matmul_integer::coop_variant;
use onnx_vulkan_core::{AttrValue, Executor, GraphIr, HostTensor, InitializerIr, NodeIr, Tensor};
use std::collections::HashMap;
use vk_compute::VkContext;

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

struct Case {
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    /// top, left, bottom, right
    pads: [usize; 4],
    x_signed: bool,
    w_signed: bool,
}

impl Case {
    fn out_dims(&self) -> (usize, usize) {
        let span = self.dilation * (self.k - 1) + 1;
        let [t, l, b, r] = self.pads;
        (
            (self.h + t + b - span) / self.stride + 1,
            (self.w + l + r - span) / self.stride + 1,
        )
    }

    fn value(bytes: &[u8], i: usize, signed: bool) -> i64 {
        if signed {
            i64::from(bytes[i] as i8)
        } else {
            i64::from(bytes[i])
        }
    }

    fn expected(&self, x: &[u8], x_zp: u8, w: &[u8], w_zp: u8) -> Vec<i64> {
        let (ho, wo) = self.out_dims();
        let xz = Self::value(&[x_zp], 0, self.x_signed);
        let wz = Self::value(&[w_zp], 0, self.w_signed);
        let mut out = vec![0i64; self.c_out * ho * wo];
        for o in 0..self.c_out {
            for oh in 0..ho {
                for ow in 0..wo {
                    let mut acc = 0;
                    for c in 0..self.c_in {
                        for r in 0..self.k {
                            for s in 0..self.k {
                                let ih = (oh * self.stride + r * self.dilation) as i64
                                    - self.pads[0] as i64;
                                let iw = (ow * self.stride + s * self.dilation) as i64
                                    - self.pads[1] as i64;
                                if ih < 0 || iw < 0 || ih >= self.h as i64 || iw >= self.w as i64 {
                                    continue; // padded with the zero point: contributes 0
                                }
                                let xi = (c * self.h + ih as usize) * self.w + iw as usize;
                                let wi = ((o * self.c_in + c) * self.k + r) * self.k + s;
                                acc += (Self::value(x, xi, self.x_signed) - xz)
                                    * (Self::value(w, wi, self.w_signed) - wz);
                            }
                        }
                    }
                    out[(o * ho + oh) * wo + ow] = acc;
                }
            }
        }
        out
    }

    /// Runs the node and returns the pipelines it compiled.
    fn check(&self) -> Vec<String> {
        let context = VkContext::new().expect("Vulkan context");
        let x_dtype = if self.x_signed { INT8 } else { UINT8 };
        let w_dtype = if self.w_signed { INT8 } else { UINT8 };
        let x = pseudo(self.c_in * self.h * self.w, 11);
        let w = pseudo(self.c_out * self.c_in * self.k * self.k, 5);
        let (x_zp, w_zp) = (if self.x_signed { 0xfb } else { 129 }, 0x07);
        let k = self.k as i64;
        let initializers = HashMap::from([
            (
                "w".to_string(),
                InitializerIr {
                    dtype: w_dtype,
                    shape: vec![self.c_out as i64, self.c_in as i64, k, k],
                    data: w.clone(),
                },
            ),
            (
                "x_zp".to_string(),
                InitializerIr {
                    dtype: x_dtype,
                    shape: vec![],
                    data: vec![x_zp],
                },
            ),
            (
                "w_zp".to_string(),
                InitializerIr {
                    dtype: w_dtype,
                    shape: vec![],
                    data: vec![w_zp],
                },
            ),
        ]);
        let [t, l, b, r] = self.pads.map(|p| p as i64);
        let (s, d) = (self.stride as i64, self.dilation as i64);
        let node = NodeIr {
            domain: String::new(),
            op: "ConvInteger".into(),
            opset: 10,
            name: "conv".into(),
            inputs: ["x", "w", "x_zp", "w_zp"].map(String::from).to_vec(),
            outputs: vec!["y".into()],
            attrs: HashMap::from([
                ("pads".to_string(), AttrValue::Ints(vec![t, l, b, r])),
                ("strides".to_string(), AttrValue::Ints(vec![s, s])),
                ("dilations".to_string(), AttrValue::Ints(vec![d, d])),
                ("kernel_shape".to_string(), AttrValue::Ints(vec![k, k])),
            ]),
        };
        let graph = GraphIr {
            nodes: vec![node],
            initializers,
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            ..Default::default()
        };
        let executor = Executor::new(&context, graph).expect("executor");
        let shape = vec![1, self.c_in as i64, self.h as i64, self.w as i64];
        let outputs = executor
            .run(vec![(
                "x",
                Tensor::Host(HostTensor::new(x_dtype, shape, x.clone())),
            )])
            .expect("run");
        let y = outputs.host("y").expect("y");
        let (ho, wo) = self.out_dims();
        assert_eq!(y.shape, [1, self.c_out as i64, ho as i64, wo as i64]);
        let got = y.to_i64().expect("i32 output");
        outputs.finish();
        let want = self.expected(&x, x_zp, &w, w_zp);
        let differing = got.iter().zip(&want).filter(|(g, e)| g != e).count();
        assert_eq!(differing, 0, "{differing} of {} sums differ", want.len());
        let mut keys: Vec<String> = executor
            .cache()
            .pipeline_keys()
            .iter()
            .map(|key| key.operation().to_string())
            .filter(|op| op.starts_with("ConvInteger"))
            .collect();
        keys.sort();
        keys.dedup();
        if coop_variant(&context).is_none() {
            return Vec::new(); // no tensor cores: the implicit ladder ran
        }
        keys
    }
}

fn has(keys: &[String], op: &str) -> bool {
    keys.iter().any(|k| k == op)
}

#[test]
fn hwc_im2col_gives_the_exact_sums_of_a_padded_3x3() {
    // SAM 3's pixel decoder convolution, fewer channels and a smaller image
    let keys = Case {
        c_in: 64,
        c_out: 32,
        h: 20,
        w: 20,
        k: 3,
        stride: 1,
        dilation: 1,
        pads: [1, 1, 1, 1],
        x_signed: false,
        w_signed: false,
    }
    .check();
    if !keys.is_empty() {
        assert!(has(&keys, "ConvInteger_im2col_hwc"), "{keys:?}");
        assert!(has(&keys, "ConvInteger_to_hwc"), "{keys:?}");
        assert!(!has(&keys, "ConvInteger_im2col"), "{keys:?}");
    }
}

#[test]
fn hwc_im2col_with_signed_operands_stride_dilation_and_uneven_pads() {
    let keys = Case {
        c_in: 32,
        c_out: 48,
        h: 18,
        w: 22,
        k: 3,
        stride: 2,
        dilation: 2,
        pads: [2, 0, 1, 3],
        x_signed: true,
        w_signed: true,
    }
    .check();
    if !keys.is_empty() {
        assert!(has(&keys, "ConvInteger_im2col_hwc"), "{keys:?}");
    }
}

#[test]
fn a_plane_off_word_boundaries_keeps_the_chw_im2col() {
    let keys = Case {
        c_in: 32,
        c_out: 16,
        h: 15,
        w: 15,
        k: 3,
        stride: 1,
        dilation: 1,
        pads: [1, 1, 1, 1],
        x_signed: false,
        w_signed: true,
    }
    .check();
    if !keys.is_empty() {
        assert!(has(&keys, "ConvInteger_im2col"), "{keys:?}");
        assert!(!has(&keys, "ConvInteger_im2col_hwc"), "{keys:?}");
    }
}
