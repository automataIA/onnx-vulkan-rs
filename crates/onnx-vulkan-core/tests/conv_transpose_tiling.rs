//! `ConvTranspose` f32 against a CPU reference, on both dispatch routes.
//!
//! No model in the regression suite contains a `ConvTranspose` at all — only
//! the sam3 family does, and it is not in the matrix — so nothing downstream
//! would catch a wrong phase decomposition. This file is the enforcement.
//!
//! Each case is labelled with the route it must take (`phase` when
//! [`phase_gemm_applies`] accepts it, `gather` otherwise) and the test asserts
//! that too: a geometry silently falling back would still pass the numerics and
//! quietly stop testing the thing this file exists for.

use onnx_vulkan_core::host_ops::FLOAT;
use onnx_vulkan_core::host_ops::HostTensor;
use onnx_vulkan_core::shaders::conv_transpose::{PhaseGeom, phase_gemm_applies};
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
    /// Must the geometry take the phase-GEMM route?
    phase: bool,
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

fn run(c: &Case) -> (Vec<f32>, Vec<f32>) {
    let h_out = ((c.h_in - 1) * c.stride.0) as i64 + c.dilation.0 * (c.k.0 as i64 - 1) + 1
        - c.pads[0]
        - c.pads[2];
    let w_out = ((c.w_in - 1) * c.stride.1) as i64 + c.dilation.1 * (c.k.1 as i64 - 1) + 1
        - c.pads[1]
        - c.pads[3];
    let (h_out, w_out) = (h_out as usize, w_out as usize);

    let x = pseudo(c.c_in * c.h_in * c.w_in, 7);
    let w = pseudo(c.c_in * c.c_out * c.k.0 * c.k.1, 11);
    let b = pseudo(c.c_out, 13);

    let mut inputs = vec!["X".to_string(), "W".to_string()];
    if c.bias {
        inputs.push("B".to_string());
    }
    // W (and B) as initializers: the phase route caches its packed slices by
    // name and only takes constants.
    let init = |shape: Vec<i64>, v: &[f32]| InitializerIr {
        dtype: FLOAT,
        shape,
        data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    let mut initializers = HashMap::new();
    initializers.insert(
        "W".to_string(),
        init(
            vec![c.c_in as i64, c.c_out as i64, c.k.0 as i64, c.k.1 as i64],
            &w,
        ),
    );
    if c.bias {
        initializers.insert("B".to_string(), init(vec![c.c_out as i64], &b));
    }
    let ir = GraphIr {
        nodes: vec![NodeIr {
            domain: String::new(),
            op: "ConvTranspose".to_string(),
            since_version: 11,
            name: "ct".to_string(),
            inputs,
            outputs: vec!["Y".to_string()],
            attrs: attrs(c),
        }],
        initializers,
        inputs: vec!["X".to_string()],
        outputs: vec!["Y".to_string()],
    };

    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    env.set(
        "X",
        Tensor::Host(HostTensor::from_f32(
            vec![1, c.c_in as i64, c.h_in as i64, c.w_in as i64],
            &x,
        )),
    );
    execute(&ir, &mut env).expect("ConvTranspose execution");
    let got = env
        .host("Y")
        .expect("output on host")
        .clone()
        .to_f32()
        .expect("output f32");
    env.finish();

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
        phase: true,
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
        phase: true,
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
        phase: true,
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
        phase: true,
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
        phase: false,
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
        phase: false,
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
        phase: false,
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
        phase: false,
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
            macs: (c.c_in * c.c_out) as i64 * h_out * w_out,
        };
        assert_eq!(phase_gemm_applies(&geom), c.phase, "route of «{}»", c.name);
    }
}

#[test]
fn conv_transpose_matches_the_cpu_reference_on_both_routes() {
    for c in CASES {
        let (got, want) = run(c);
        assert_eq!(got.len(), want.len(), "«{}»: lunghezze", c.name);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-5 + 1e-4 * w.abs(),
                "«{}» element {i}: {g} != {w}",
                c.name
            );
        }
    }
}
