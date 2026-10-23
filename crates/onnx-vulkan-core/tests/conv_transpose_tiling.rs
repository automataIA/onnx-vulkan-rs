//! `ConvTranspose` f32 against a CPU reference, on all three dispatch routes.
//!
//! No model in the regression suite contains a `ConvTranspose` at all — only
//! the sam3 family does, and it is not in the matrix — so nothing downstream
//! would catch a wrong phase or polyphase decomposition. This file is the enforcement.
//!
//! Each case is labelled with the route it must take (see
//! [`route`]) and the test asserts
//! that too: a geometry silently falling back would still pass the numerics and
//! quietly stop testing the thing this file exists for.

use onnx_vulkan_core::host_ops::FLOAT;
use onnx_vulkan_core::host_ops::HostTensor;
use onnx_vulkan_core::shaders::conv_transpose::{PhaseGeom, Route, route};
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, InitializerIr, KernelCache, NodeIr, Tensor, execute,
};
use std::collections::HashMap;
use vk_compute::VkContext;

fn pseudo(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    k: (usize, usize),
    stride: (usize, usize),
    pads: [i64; 4],
    dilation: (i64, i64),
    bias: bool,
    /// The route the geometry must take, `W` being an initializer.
    route: Route,
}

impl Case {
    /// `W = 1` cases run as the 1D node they stand for: only 1D nodes take the
    /// polyphase route.
    fn one_d(&self) -> bool {
        self.w_in == 1
            && self.k.1 == 1
            && self.stride.1 == 1
            && self.pads[1] == 0
            && self.pads[3] == 0
            && self.dilation.1 == 1
    }
}

/// Straight transcription of the operator: walk the input, spray each value
/// over its kernel footprint. Deliberately the *other* formulation from the
/// gather kernel under test, so a shared index mistake cannot cancel out.
fn reference(c: &Case, x: &[f32], w: &[f32], b: &[f32], h_out: usize, w_out: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c.c_out * h_out * w_out];
    for m in 0..c.c_out {
        if c.bias {
            for p in 0..h_out * w_out {
                out[m * h_out * w_out + p] = b[m];
            }
        }
    }
    for ic in 0..c.c_in {
        for m in 0..c.c_out {
            for ih in 0..c.h_in {
                for iw in 0..c.w_in {
                    let v = x[(ic * c.h_in + ih) * c.w_in + iw];
                    for r in 0..c.k.0 {
                        for s in 0..c.k.1 {
                            let oh =
                                ih as i64 * c.stride.0 as i64 + r as i64 * c.dilation.0 - c.pads[0];
                            let ow =
                                iw as i64 * c.stride.1 as i64 + s as i64 * c.dilation.1 - c.pads[1];
                            if oh < 0 || ow < 0 || oh >= h_out as i64 || ow >= w_out as i64 {
                                continue;
                            }
                            let wv = w[((ic * c.c_out + m) * c.k.0 + r) * c.k.1 + s];
                            out[(m * h_out + oh as usize) * w_out + ow as usize] += v * wv;
                        }
                    }
                }
            }
        }
    }
    out
}

fn attrs(c: &Case) -> HashMap<String, AttrValue> {
    let mut a = HashMap::new();
    if c.one_d() {
        a.insert(
            "kernel_shape".to_string(),
            AttrValue::Ints(vec![c.k.0 as i64]),
        );
        a.insert(
            "strides".to_string(),
            AttrValue::Ints(vec![c.stride.0 as i64]),
        );
        a.insert(
            "pads".to_string(),
            AttrValue::Ints(vec![c.pads[0], c.pads[2]]),
        );
        a.insert("dilations".to_string(), AttrValue::Ints(vec![c.dilation.0]));
        return a;
    }
    a.insert(
        "kernel_shape".to_string(),
        AttrValue::Ints(vec![c.k.0 as i64, c.k.1 as i64]),
    );
    a.insert(
        "strides".to_string(),
        AttrValue::Ints(vec![c.stride.0 as i64, c.stride.1 as i64]),
    );
    a.insert("pads".to_string(), AttrValue::Ints(c.pads.to_vec()));
    a.insert(
        "dilations".to_string(),
        AttrValue::Ints(vec![c.dilation.0, c.dilation.1]),
    );
    a
}

/// Which operand [`run`] makes partly non-finite.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Poison {
    None,
    /// Two weights of input channel 0: the last tap of output channel 0 (NaN)
    /// and the first tap of output channel 1 (+Inf).
    W,
    /// Three inputs: the first of channel 0 (NaN) and the last of the last
    /// channel (+Inf), at the borders, and the middle of channel 1 (-Inf).
    X,
}

/// Puts `poison`'s non-finite values into `x` or `w`, laid out as `c`.
fn poison_operands(c: &Case, poison: Poison, x: &mut [f32], w: &mut [f32]) {
    match poison {
        Poison::None => {}
        Poison::W => {
            let taps = c.k.0 * c.k.1;
            w[taps - 1] = f32::NAN;
            w[taps] = f32::INFINITY;
        }
        Poison::X => {
            let plane = c.h_in * c.w_in;
            x[0] = f32::NAN;
            x[x.len() - 1] = f32::INFINITY;
            x[plane + plane / 2] = f32::NEG_INFINITY;
        }
    }
}

impl Case {
    fn out_len(&self) -> (usize, usize) {
        let h =
            ((self.h_in - 1) * self.stride.0) as i64 + self.dilation.0 * (self.k.0 as i64 - 1) + 1
                - self.pads[0]
                - self.pads[2];
        let w =
            ((self.w_in - 1) * self.stride.1) as i64 + self.dilation.1 * (self.k.1 as i64 - 1) + 1
                - self.pads[1]
                - self.pads[3];
        (h as usize, w as usize)
    }

    fn x_shape(&self) -> Vec<i64> {
        if self.one_d() {
            vec![1, self.c_in as i64, self.h_in as i64]
        } else {
            vec![1, self.c_in as i64, self.h_in as i64, self.w_in as i64]
        }
    }

    fn w_shape(&self) -> Vec<i64> {
        if self.one_d() {
            vec![self.c_in as i64, self.c_out as i64, self.k.0 as i64]
        } else {
            vec![
                self.c_in as i64,
                self.c_out as i64,
                self.k.0 as i64,
                self.k.1 as i64,
            ]
        }
    }

    /// `X`, `W` and `B`, `poison` making some of them non-finite.
    fn operands(&self, poison: Poison) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut x = pseudo(self.c_in * self.h_in * self.w_in, 7);
        let mut w = pseudo(self.c_in * self.c_out * self.k.0 * self.k.1, 11);
        poison_operands(self, poison, &mut x, &mut w);
        (x, w, pseudo(self.c_out, 13))
    }
}

/// One `ConvTranspose` node `Y = f(X, W[, B])`, `W` and `B` initializers of
/// the given shapes: the phase route caches its packed slices by name and only
/// takes constants.
fn graph(
    attrs: HashMap<String, AttrValue>,
    w: (Vec<i64>, &[f32]),
    b: Option<(Vec<i64>, &[f32])>,
) -> GraphIr {
    let init = |(shape, v): (Vec<i64>, &[f32])| InitializerIr {
        dtype: FLOAT,
        shape,
        data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    let mut inputs = vec!["X".to_string(), "W".to_string()];
    let mut initializers = HashMap::new();
    initializers.insert("W".to_string(), init(w));
    if let Some(b) = b {
        inputs.push("B".to_string());
        initializers.insert("B".to_string(), init(b));
    }
    GraphIr {
        nodes: vec![NodeIr {
            domain: String::new(),
            op: "ConvTranspose".to_string(),
            opset: 11,
            name: "ct".to_string(),
            inputs,
            outputs: vec!["Y".to_string()],
            attrs,
        }],
        initializers,
        inputs: vec!["X".to_string()],
        outputs: vec!["Y".to_string()],
        ..Default::default()
    }
}

/// Runs `ir` on `X`: `Y`, or the error the interpreter reported.
fn execute_graph(ir: &GraphIr, x_shape: Vec<i64>, x: &[f32]) -> onnx_vulkan_core::Result<Vec<f32>> {
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    env.set("X", Tensor::Host(HostTensor::from_f32(x_shape, x)));
    let got = execute(ir, &mut env).map(|()| {
        env.host("Y")
            .expect("output on host")
            .clone()
            .to_f32()
            .expect("output f32")
    });
    env.finish();
    got
}

/// Runs `c` on the GPU and on the CPU reference, `poison` making some of the
/// operands non-finite.
fn run(c: &Case, poison: Poison) -> (Vec<f32>, Vec<f32>) {
    let (x, w, b) = c.operands(poison);
    let ir = graph(
        attrs(c),
        (c.w_shape(), &w),
        c.bias.then(|| (vec![c.c_out as i64], &b[..])),
    );
    let got = execute_graph(&ir, c.x_shape(), &x).expect("ConvTranspose execution");
    let (h_out, w_out) = c.out_len();
    (got, reference(c, &x, &w, &b, h_out, w_out))
}

/// The geometries at the edge of the predicate, each side of it.
const CASES: &[Case] = &[
    // sam3's shape in miniature: stride == kernel, the phases tile the output.
    Case {
        name: "stride == kernel",
        c_in: 96,
        c_out: 80,
        h_in: 20,
        w_in: 20,
        k: (2, 2),
        stride: (2, 2),
        pads: [0; 4],
        dilation: (1, 1),
        bias: false,
        route: Route::Phase,
    },
    // same, with a bias — it rides in as the GEMM's C.
    Case {
        name: "stride == kernel, bias",
        c_in: 96,
        c_out: 80,
        h_in: 20,
        w_in: 20,
        k: (2, 2),
        stride: (2, 2),
        pads: [0; 4],
        dilation: (1, 1),
        bias: true,
        route: Route::Phase,
    },
    // stride > kernel: the phases leave holes only the fill pass writes.
    Case {
        name: "stride > kernel, bias",
        c_in: 96,
        c_out: 80,
        h_in: 20,
        w_in: 20,
        k: (2, 2),
        stride: (3, 4),
        pads: [0; 4],
        dilation: (1, 1),
        bias: true,
        route: Route::Phase,
    },
    // dimensions that divide neither the 64×64 GEMM tile nor a 256-wide group.
    Case {
        name: "shape indivisibile",
        c_in: 70,
        c_out: 67,
        h_in: 41,
        w_in: 17,
        k: (3, 1),
        stride: (3, 1),
        pads: [0; 4],
        dilation: (1, 1),
        bias: true,
        route: Route::Phase,
    },
    // Piper's upsamplers in miniature, 1D: overlapping footprints (kernel =
    // 2·stride) and pads cropping both borders. Lengths divide neither the
    // 64-wide GEMM tile nor a 256-wide group.
    Case {
        name: "piper k16 s8 p4",
        c_in: 96,
        c_out: 80,
        h_in: 140,
        w_in: 1,
        k: (16, 1),
        stride: (8, 1),
        pads: [4, 0, 4, 0],
        dilation: (1, 1),
        bias: true,
        route: Route::Polyphase,
    },
    Case {
        name: "piper k4 s2 p1",
        c_in: 70,
        c_out: 67,
        h_in: 900,
        w_in: 1,
        k: (4, 1),
        stride: (2, 1),
        pads: [1, 0, 1, 0],
        dilation: (1, 1),
        bias: true,
        route: Route::Polyphase,
    },
    // pad wider than the stride: the first q of every phase lands in the pad
    Case {
        name: "1D pad > stride",
        c_in: 96,
        c_out: 80,
        h_in: 600,
        w_in: 1,
        k: (5, 1),
        stride: (2, 1),
        pads: [3, 0, 1, 0],
        dilation: (1, 1),
        bias: false,
        route: Route::Polyphase,
    },
    // stride > kernel with pads: phases past the kernel have no tap, bias only
    Case {
        name: "1D stride > kernel, pads",
        c_in: 96,
        c_out: 80,
        h_in: 300,
        w_in: 1,
        k: (3, 1),
        stride: (5, 1),
        pads: [1, 0, 1, 0],
        dilation: (1, 1),
        bias: true,
        route: Route::Polyphase,
    },
    // stride 1: a single phase holding every tap
    Case {
        name: "1D stride 1",
        c_in: 64,
        c_out: 64,
        h_in: 2100,
        w_in: 1,
        k: (3, 1),
        stride: (1, 1),
        pads: [1, 0, 1, 0],
        dilation: (1, 1),
        bias: true,
        route: Route::Polyphase,
    },
    // --- and the fallbacks, same operator through the gather kernel
    Case {
        name: "non-zero pads",
        c_in: 96,
        c_out: 80,
        h_in: 20,
        w_in: 20,
        k: (2, 2),
        stride: (2, 2),
        pads: [1, 1, 1, 1],
        dilation: (1, 1),
        bias: true,
        route: Route::Gather,
    },
    Case {
        name: "stride < kernel",
        c_in: 96,
        c_out: 80,
        h_in: 20,
        w_in: 20,
        k: (3, 3),
        stride: (2, 2),
        pads: [0; 4],
        dilation: (1, 1),
        bias: true,
        route: Route::Gather,
    },
    Case {
        name: "dilation",
        c_in: 96,
        c_out: 80,
        h_in: 20,
        w_in: 20,
        k: (2, 2),
        stride: (2, 2),
        pads: [0; 4],
        dilation: (2, 2),
        bias: false,
        route: Route::Gather,
    },
    // small enough that the extra dispatches are not worth it
    Case {
        name: "below threshold",
        c_in: 8,
        c_out: 8,
        h_in: 6,
        w_in: 6,
        k: (2, 2),
        stride: (2, 2),
        pads: [0; 4],
        dilation: (1, 1),
        bias: true,
        route: Route::Gather,
    },
];

#[test]
fn every_case_takes_the_route_it_claims() {
    for c in CASES {
        let h_out = ((c.h_in - 1) * c.stride.0) as i64 + c.dilation.0 * (c.k.0 as i64 - 1) + 1
            - c.pads[0]
            - c.pads[2];
        let w_out = ((c.w_in - 1) * c.stride.1) as i64 + c.dilation.1 * (c.k.1 as i64 - 1) + 1
            - c.pads[1]
            - c.pads[3];
        let geom = PhaseGeom {
            batch: 1,
            group: 1,
            kernel: (c.k.0 as i64, c.k.1 as i64),
            stride: (c.stride.0 as i64, c.stride.1 as i64),
            dilation: c.dilation,
            zero_pads: c.pads.iter().all(|&p| p == 0),
            zero_output_padding: true,
            one_d: c.one_d(),
            macs: (c.c_in * c.c_out) as i64 * h_out * w_out,
        };
        assert_eq!(route(&geom, true), c.route, "route of «{}»", c.name);
    }
}

/// `got` equals `want` up to accumulation order: the same NaN class, the same
/// infinity, finite values within the tolerance of the operator's tests.
fn assert_matches(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "«{name}»: lunghezze");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let same = if w.is_nan() {
            g.is_nan()
        } else if w.is_infinite() {
            g == w
        } else {
            (g - w).abs() <= 1e-5 + 1e-4 * w.abs()
        };
        assert!(same, "«{name}» element {i}: {g} != {w}");
    }
}

#[test]
fn conv_transpose_matches_the_cpu_reference_on_every_route() {
    for c in CASES {
        let (got, want) = run(c, Poison::None);
        assert_matches(c.name, &got, &want);
    }
}

/// A tap that falls outside the input contributes nothing, whatever its weight.
/// The CPU reference never visits it, so a kernel that multiplies it by a zero
/// instead turns an infinite or NaN weight into a NaN where ONNX has a finite
/// value: at the borders, where the polyphase GEMM sees padding.
#[test]
fn non_finite_weights_reach_only_the_outputs_their_taps_touch() {
    for c in CASES {
        let (got, want) = run(c, Poison::W);
        assert_matches(c.name, &got, &want);
    }
}

/// The same from the other operand: a non-finite input reaches the outputs its
/// footprint covers and no other, including through the zero weights a kernel
/// pads its taps with and at both borders of the input.
#[test]
fn non_finite_inputs_reach_only_the_outputs_their_footprint_covers() {
    for c in CASES {
        let (got, want) = run(c, Poison::X);
        assert_matches(c.name, &got, &want);
    }
}

/// Invalid attributes and shapes are the model's error: reported as one by the
/// operator's own validation, on every route, before any index is computed
/// from them — never a panic, a division by zero or a fallback.
#[test]
fn invalid_attributes_are_reported_not_run() {
    let poly = CASES.iter().find(|c| c.route == Route::Polyphase).unwrap();
    let gather = CASES.iter().find(|c| c.route == Route::Gather).unwrap();
    for c in [poly, gather] {
        let (x, w, b) = c.operands(Poison::None);
        let bias = || Some((vec![c.c_out as i64], &b[..]));
        let with = |key: &str, edit: &dyn Fn(&mut Vec<i64>)| {
            let mut a = attrs(c);
            let Some(AttrValue::Ints(v)) = a.get_mut(key) else {
                panic!("«{}» has no {key}", c.name)
            };
            edit(v);
            a
        };
        let output_padding = |v: i64| {
            let mut a = attrs(c);
            let n = if c.one_d() { 1 } else { 2 };
            a.insert("output_padding".to_string(), AttrValue::Ints(vec![v; n]));
            a
        };
        let mut w_wider = c.w_shape();
        w_wider[2] += 1;
        let w_long = vec![0.0f32; w.len() / c.k.0 * (c.k.0 + 1)];
        let cases: Vec<(&str, GraphIr, &str)> = vec![
            (
                "stride 0",
                graph(with("strides", &|v| v[0] = 0), (c.w_shape(), &w), bias()),
                "must be positive",
            ),
            (
                "negative pad",
                graph(with("pads", &|v| v[0] = -1), (c.w_shape(), &w), bias()),
                "non-negative",
            ),
            (
                "dilation 0",
                graph(with("dilations", &|v| v[0] = 0), (c.w_shape(), &w), bias()),
                "must be positive",
            ),
            (
                "kernel 0",
                graph(
                    with("kernel_shape", &|v| v[0] = 0),
                    (c.w_shape(), &w),
                    bias(),
                ),
                "must be positive",
            ),
            (
                "negative output_padding",
                graph(output_padding(-1), (c.w_shape(), &w), bias()),
                "non-negative",
            ),
            (
                "W wider than kernel_shape",
                graph(attrs(c), (w_wider, &w_long), bias()),
                "does not match",
            ),
            (
                "B of C_out + 1",
                graph(
                    attrs(c),
                    (c.w_shape(), &w),
                    Some((vec![c.c_out as i64 + 1], &[0.0; 1024][..=c.c_out])),
                ),
                "is not",
            ),
            (
                "output length past i64",
                graph(
                    with("strides", &|v| v[0] = i64::MAX),
                    (c.w_shape(), &w),
                    bias(),
                ),
                "overflows",
            ),
        ];
        for (why, ir, says) in cases {
            let err = execute_graph(&ir, c.x_shape(), &x)
                .expect_err(&format!("«{}» {why}: accepted", c.name))
                .to_string();
            assert!(err.contains(says), "«{}» {why}: {err}", c.name);
        }
    }
}

/// Every output of a geometry the polyphase route accepts is written, by it
/// and by the gather kernel alike, and the two agree up to the order of the
/// sums — NaN for NaN, infinity for infinity — with either operand poisoned.
///
/// Both outputs start as a NaN pattern no arithmetic produces (quiet NaNs
/// propagate quiet): one left over is an output no thread wrote.
#[test]
fn polyphase_writes_every_output_and_matches_the_gather_kernel() {
    use onnx_vulkan_core::shaders::conv_transpose as ct;
    use vk_compute::compile_wgsl;
    const UNWRITTEN: u32 = 0x7fba_d5ed;

    // Beyond the polyphase cases above: lengths and channels off every tile
    // size, a stride wider than kernel and pad, asymmetric pads.
    let extra = [
        Case {
            name: "odd sizes",
            c_in: 3,
            c_out: 5,
            h_in: 37,
            w_in: 1,
            k: (7, 1),
            stride: (3, 1),
            pads: [2, 0, 2, 0],
            dilation: (1, 1),
            bias: true,
            route: Route::Polyphase,
        },
        Case {
            name: "stride > kernel, pad",
            c_in: 2,
            c_out: 70,
            h_in: 50,
            w_in: 1,
            k: (2, 1),
            stride: (5, 1),
            pads: [3, 0, 1, 0],
            dilation: (1, 1),
            bias: false,
            route: Route::Polyphase,
        },
        Case {
            name: "stride 1, off-tile",
            c_in: 17,
            c_out: 65,
            h_in: 129,
            w_in: 1,
            k: (3, 1),
            stride: (1, 1),
            pads: [1, 0, 1, 0],
            dilation: (1, 1),
            bias: true,
            route: Route::Polyphase,
        },
    ];

    let ctx = VkContext::new().expect("Vulkan context");
    let limits = ctx.compute_limits().max_workgroup_count;
    let gather = ctx
        .create_pipeline(
            &compile_wgsl(&ct::direct()).unwrap(),
            ct::BINDINGS,
            ct::PUSH_BYTES,
        )
        .unwrap();
    let poly = ctx
        .create_pipeline(
            &compile_wgsl(ct::POLYPHASE_1D).unwrap(),
            ct::POLYPHASE_BINDINGS,
            ct::POLYPHASE_PUSH_BYTES,
        )
        .unwrap();
    let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
    let push = |f: &[usize]| {
        f.iter()
            .flat_map(|&v| u32::try_from(v).unwrap().to_le_bytes())
            .collect::<Vec<_>>()
    };

    let polyphase_cases = CASES.iter().filter(|c| c.route == Route::Polyphase);
    for c in polyphase_cases.chain(&extra) {
        let (l_out, _) = c.out_len();
        let (pad, s) = (c.pads[0] as usize, c.stride.0);
        let d = ct::PolyphaseDims {
            c_in: c.c_in,
            c_out: c.c_out,
            l_in: c.h_in,
            l_out,
            kernel: c.k.0,
            stride: s,
            pad,
        };
        assert!(
            ct::polyphase_indices_fit(&d) && ct::polyphase_fits_device(&d, limits),
            "«{}» must be a geometry the polyphase route accepts",
            c.name
        );
        let (nq, grid) = (d.nq().unwrap(), d.grid().unwrap());
        let total = c.c_out * l_out;
        let unwritten = vec![UNWRITTEN; total]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();

        let mut identical = [0usize; 3];
        for (slot, poison) in [Poison::None, Poison::W, Poison::X].into_iter().enumerate() {
            let (x, w, b) = c.operands(poison);
            let buf = |v: &[u8]| {
                let buf = ctx.create_storage_buffer(v.len() as u64).unwrap();
                ctx.stream_upload(&buf, v).unwrap();
                buf
            };
            let (x_buf, w_buf, b_buf) = (buf(&bytes(&x)), buf(&bytes(&w)), buf(&bytes(&b)));
            let (out_g, out_p) = (buf(&unwritten), buf(&unwritten));
            ctx.flush().unwrap();
            let has_bias = usize::from(c.bias);
            // 1D as the interpreter runs it: W = 1
            let pg = push(&[
                total, c.c_in, c.c_out, 1, c.h_in, 1, l_out, 1, c.k.0, 1, s, 1, pad, 0, 1, 1,
                c.c_out, has_bias,
            ]);
            ctx.stream_dispatch(
                &gather,
                &[&x_buf, &w_buf, &b_buf, &out_g],
                &pg,
                [total.div_ceil(256) as u32, 1, 1],
            )
            .unwrap();
            let pp = push(&[c.c_in, c.c_out, c.h_in, l_out, c.k.0, s, pad, nq, has_bias]);
            ctx.stream_dispatch(
                &poly,
                &[&w_buf, &x_buf, &b_buf, &out_p],
                &pp,
                grid.map(|n| n as u32),
            )
            .unwrap();
            ctx.flush().unwrap();
            let read = |buf| -> Vec<f32> {
                ctx.stream_download(buf, 4 * total)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let (g, p) = (read(&out_g), read(&out_p));
            for (kernel, out) in [("gather", &g), ("polyphase", &p)] {
                let left = out.iter().filter(|v| v.to_bits() == UNWRITTEN).count();
                assert_eq!(
                    left, 0,
                    "«{}» {poison:?}: {kernel} left outputs unwritten",
                    c.name
                );
            }
            assert_matches(&format!("{} {poison:?}", c.name), &p, &g);
            identical[slot] = g
                .iter()
                .zip(&p)
                .filter(|(a, b)| a.to_bits() == b.to_bits())
                .count();
        }
        println!(
            "«{}»: polyphase bit-identical to gather {}/{total} clean, {}/{total} W poisoned, \
             {}/{total} X poisoned",
            c.name, identical[0], identical[1], identical[2]
        );
    }
}

/// A valid node whose phases outnumber the device's grid z: the polyphase
/// route cannot launch it, so it falls back, and runs correctly, instead of
/// dispatching past the limit (Vulkan's minimum is 65535).
#[test]
fn a_grid_past_the_device_limit_falls_back_and_runs() {
    use onnx_vulkan_core::shaders::conv_transpose as ct;
    let c = Case {
        name: "stride 65536",
        c_in: 16,
        c_out: 16,
        h_in: 2,
        w_in: 1,
        k: (16, 1),
        stride: (65_536, 1),
        pads: [0; 4],
        dilation: (1, 1),
        bias: true,
        route: Route::Polyphase,
    };
    let (l_out, _) = c.out_len();
    let d = ct::PolyphaseDims {
        c_in: c.c_in,
        c_out: c.c_out,
        l_in: c.h_in,
        l_out,
        kernel: c.k.0,
        stride: c.stride.0,
        pad: 0,
    };
    let limits = *VkContext::new().expect("Vulkan context").compute_limits();
    assert!(ct::polyphase_indices_fit(&d));
    println!(
        "grid z limit {}: polyphase {}",
        limits.max_workgroup_count[2],
        if ct::polyphase_fits_device(&d, limits.max_workgroup_count) {
            "launchable"
        } else {
            "refused, gather fallback"
        }
    );
    let (got, want) = run(&c, Poison::None);
    assert_matches(c.name, &got, &want);
}
