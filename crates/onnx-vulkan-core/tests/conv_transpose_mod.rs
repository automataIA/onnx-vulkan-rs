//! `ConvTranspose` (GPU) and `Mod` (host), the two operators that close sam3.
//!
//! The `ConvTranspose` reference here is written in the **scatter** form — walk
//! the input, add each value into its kernel footprint — which is the direct
//! reading of the operator. The kernel inverts that into a gather so it needs
//! no atomics, so the two are genuinely different formulations of the same
//! function and agreeing is evidence rather than a tautology.
//!
//! Covered: the sam3 case (2×2 kernel, stride 2, no pad), a case with stride,
//! pads, dilation and `output_padding` together, grouped, and the forms that
//! `is_implemented_node` must refuse.
//!
//! The first two were additionally checked against `onnx.reference` on the same
//! inputs (sums 8.521232 vs 8.521235 and 1.348969 vs 1.348971, in f32
//! summation order). The grouped one could not be: `onnx.reference`'s own
//! `ConvTranspose` raises on `group > 1`, writing each group's whole output
//! into a single channel slice.

use onnx_vulkan_core::host_ops::HostTensor;
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, InitializerIr, KernelCache, NodeIr, Tensor, execute,
    is_implemented_node,
};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(op: &str, inputs: &[&str], attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        since_version: 11,
        name: format!("{op}_0"),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: vec!["out".to_string()],
        attrs: attrs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect(),
    }
}

fn graph(op: &str, inputs: &[&str], attrs: &[(&str, AttrValue)]) -> GraphIr {
    GraphIr {
        nodes: vec![node(op, inputs, attrs)],
        initializers: HashMap::<String, InitializerIr>::new(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: vec!["out".to_string()],
    }
}

fn run(ir: &GraphIr, values: &[(&str, HostTensor)]) -> (Vec<f32>, Vec<i64>) {
    let context = VkContext::new().expect("contesto Vulkan");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    for (name, t) in values {
        env.set(name, Tensor::Host(t.clone()));
    }
    execute(ir, &mut env).expect("graph execution");
    let out = env.host("out").expect("output on host").clone();
    env.finish();
    (out.to_f32().expect("output f32"), out.shape)
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

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "lunghezze diverse");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5 + 1e-4 * w.abs(),
            "element {i}: {g} != {w}"
        );
    }
}

#[derive(Clone, Copy)]
struct Geom {
    n: usize,
    c_in: usize,
    gso: usize,
    group: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    phb: usize,
    phe: usize,
    pwb: usize,
    pwe: usize,
    dh: usize,
    dw: usize,
    oph: usize,
    opw: usize,
}

impl Geom {
    fn out_hw(&self) -> (usize, usize) {
        (
            (self.h_in - 1) * self.sh + self.dh * (self.kh - 1) + 1 + self.oph
                - self.phb
                - self.phe,
            (self.w_in - 1) * self.sw + self.dw * (self.kw - 1) + 1 + self.opw
                - self.pwb
                - self.pwe,
        )
    }
    fn c_out(&self) -> usize {
        self.gso * self.group
    }
}

/// Scatter reference: `x` [N, C_in, H, W], `w` [C_in, C_out/group, kH, kW].
fn conv_transpose_ref(x: &[f32], w: &[f32], bias: Option<&[f32]>, g: Geom) -> Vec<f32> {
    let (h_out, w_out) = g.out_hw();
    let c_out = g.c_out();
    let gsi = g.c_in / g.group;
    let mut out = vec![0.0f32; g.n * c_out * h_out * w_out];
    for bn in 0..g.n {
        for ic in 0..g.c_in {
            let grp = ic / gsi;
            for ih in 0..g.h_in {
                for iw in 0..g.w_in {
                    let v = x[((bn * g.c_in + ic) * g.h_in + ih) * g.w_in + iw];
                    for mg in 0..g.gso {
                        let m = grp * g.gso + mg;
                        for r in 0..g.kh {
                            let oh = (ih * g.sh + r * g.dh) as i64 - g.phb as i64;
                            if oh < 0 || oh >= h_out as i64 {
                                continue;
                            }
                            for s in 0..g.kw {
                                let ow = (iw * g.sw + s * g.dw) as i64 - g.pwb as i64;
                                if ow < 0 || ow >= w_out as i64 {
                                    continue;
                                }
                                let wv = w[((ic * g.gso + mg) * g.kh + r) * g.kw + s];
                                out[((bn * c_out + m) * h_out + oh as usize) * w_out
                                    + ow as usize] += v * wv;
                            }
                        }
                    }
                }
            }
        }
    }
    if let Some(b) = bias {
        for (i, o) in out.iter_mut().enumerate() {
            *o += b[(i / (h_out * w_out)) % c_out];
        }
    }
    out
}

fn check(g: Geom, attrs: &[(&str, AttrValue)], with_bias: bool) {
    let c_out = g.c_out();
    let x = pseudo(g.n * g.c_in * g.h_in * g.w_in, 7);
    let wt = pseudo(g.c_in * g.gso * g.kh * g.kw, 11);
    let bias = with_bias.then(|| pseudo(c_out, 13));

    let mut inputs = vec!["x", "w"];
    let mut values = vec![
        (
            "x",
            HostTensor::from_f32(
                vec![g.n as i64, g.c_in as i64, g.h_in as i64, g.w_in as i64],
                &x,
            ),
        ),
        (
            "w",
            HostTensor::from_f32(
                vec![g.c_in as i64, g.gso as i64, g.kh as i64, g.kw as i64],
                &wt,
            ),
        ),
    ];
    if let Some(b) = &bias {
        inputs.push("b");
        values.push(("b", HostTensor::from_f32(vec![c_out as i64], b)));
    }

    let ir = graph("ConvTranspose", &inputs, attrs);
    assert!(is_implemented_node(&ir.nodes[0]), "node not claimed");
    let (got, shape) = run(&ir, &values);

    let (h_out, w_out) = g.out_hw();
    assert_eq!(
        shape,
        vec![g.n as i64, c_out as i64, h_out as i64, w_out as i64]
    );
    assert_close(&got, &conv_transpose_ref(&x, &wt, bias.as_deref(), g));
}

const BASE: Geom = Geom {
    n: 1,
    c_in: 3,
    gso: 2,
    group: 1,
    h_in: 4,
    w_in: 5,
    kh: 2,
    kw: 2,
    sh: 2,
    sw: 2,
    phb: 0,
    phe: 0,
    pwb: 0,
    pwe: 0,
    dh: 1,
    dw: 1,
    oph: 0,
    opw: 0,
};

/// The sam3 FPN case: 2×2 kernel, stride 2, no padding, with bias — a clean
/// ×2 upsample where every output element has exactly one contributor.
#[test]
fn stride2_kernel2_matches_the_scatter_reference() {
    check(
        BASE,
        &[
            ("kernel_shape", AttrValue::Ints(vec![2, 2])),
            ("strides", AttrValue::Ints(vec![2, 2])),
            ("pads", AttrValue::Ints(vec![0, 0, 0, 0])),
            ("dilations", AttrValue::Ints(vec![1, 1])),
            ("group", AttrValue::Int(1)),
        ],
        true,
    );
}

/// Stride, asymmetric pads, dilation and `output_padding` at once: this is
/// where the exact-division test in the kernel earns its keep, because most
/// (output, kernel row) pairs have no contributing input.
#[test]
fn strided_padded_dilated_matches_the_scatter_reference() {
    let g = Geom {
        n: 2,
        c_in: 2,
        gso: 3,
        h_in: 5,
        w_in: 4,
        kh: 3,
        kw: 2,
        sh: 3,
        sw: 2,
        phb: 1,
        phe: 2,
        pwb: 1,
        pwe: 0,
        dh: 2,
        dw: 2,
        oph: 1,
        opw: 1,
        ..BASE
    };
    check(
        g,
        &[
            ("kernel_shape", AttrValue::Ints(vec![3, 2])),
            ("strides", AttrValue::Ints(vec![3, 2])),
            // ONNX pads order: [h_begin, w_begin, h_end, w_end]
            ("pads", AttrValue::Ints(vec![1, 1, 2, 0])),
            ("dilations", AttrValue::Ints(vec![2, 2])),
            ("output_padding", AttrValue::Ints(vec![1, 1])),
        ],
        false,
    );
}

/// `group > 1`: each output channel sees only its own slice of input channels.
#[test]
fn grouped_matches_the_scatter_reference() {
    let g = Geom {
        c_in: 4,
        gso: 2,
        group: 2,
        h_in: 3,
        w_in: 3,
        ..BASE
    };
    check(
        g,
        &[
            ("kernel_shape", AttrValue::Ints(vec![2, 2])),
            ("strides", AttrValue::Ints(vec![2, 2])),
            ("group", AttrValue::Int(2)),
        ],
        true,
    );
}

/// `output_shape` and `auto_pad = SAME_*` derive the pads from the requested
/// output; not implemented, so they must not be claimed — claiming a node we
/// cannot run is a runtime error instead of a CPU fallback.
#[test]
fn output_shape_and_same_padding_are_not_claimed() {
    for attrs in [
        vec![("output_shape", AttrValue::Ints(vec![1, 2, 8, 10]))],
        vec![("auto_pad", AttrValue::String("SAME_UPPER".to_string()))],
        vec![("auto_pad", AttrValue::String("SAME_LOWER".to_string()))],
    ] {
        assert!(
            !is_implemented_node(&node("ConvTranspose", &["x", "w"], &attrs)),
            "{attrs:?} must not be claimed"
        );
    }
    assert!(is_implemented_node(&node(
        "ConvTranspose",
        &["x", "w"],
        &[("auto_pad", AttrValue::String("VALID".to_string()))]
    )));
}

// ---- Mod

fn run_mod(a: HostTensor, b: HostTensor, fmod: i64) -> HostTensor {
    let ir = graph("Mod", &["a", "b"], &[("fmod", AttrValue::Int(fmod))]);
    assert!(is_implemented_node(&ir.nodes[0]));
    let context = VkContext::new().expect("contesto Vulkan");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    env.set("a", Tensor::Host(a));
    env.set("b", Tensor::Host(b));
    execute(&ir, &mut env).expect("graph execution");
    let out = env.host("out").expect("output on host").clone();
    env.finish();
    out
}

/// `fmod = 0` on integers: the sign follows the **divisor**, which is where it
/// parts ways with Rust's `%`. Values from the ONNX operator documentation.
#[test]
fn mod_integer_follows_the_divisor_sign() {
    let a = HostTensor::from_i64(vec![6], &[-4, 7, 5, 4, -7, 8]);
    let b = HostTensor::from_i64(vec![6], &[2, -3, 8, -2, 3, 5]);
    let got = run_mod(a, b, 0).to_i64().expect("output int64");
    assert_eq!(got, vec![0, -2, 5, 0, 2, 3]);
}

/// `fmod = 1`: C semantics, sign of the dividend, and the only form legal on
/// floats.
#[test]
fn mod_fmod_follows_the_dividend_sign() {
    let a = HostTensor::from_i64(vec![6], &[-4, 7, 5, 4, -7, 8]);
    let b = HostTensor::from_i64(vec![6], &[2, -3, 8, -2, 3, 5]);
    let got = run_mod(a, b, 1).to_i64().expect("output int64");
    assert_eq!(got, vec![0, 1, 5, 0, -1, 3]);

    let a = HostTensor::from_f32(vec![4], &[-4.3, 7.2, 5.0, -0.5]);
    let b = HostTensor::from_f32(vec![4], &[2.1, -3.4, 8.0, 0.3]);
    let got = run_mod(a, b, 1).to_f32().expect("output f32");
    assert_close(&got, &[-0.1, 0.4, 5.0, -0.2]);
}

/// The sam3 shape: scalar against scalar, and broadcasting against a scalar
/// divisor, which is how it is actually used there.
#[test]
fn mod_broadcasts_against_a_scalar_divisor() {
    let a = HostTensor::from_i64(vec![2, 3], &[10, 11, 12, 13, 14, 15]);
    let b = HostTensor::from_i64(vec![], &[4]);
    let out = run_mod(a, b, 0);
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.to_i64().expect("int64"), vec![2, 3, 0, 1, 2, 3]);
}
