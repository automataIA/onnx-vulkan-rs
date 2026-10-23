//! 1D `Conv` on the wide 128×128 tile ([`wide_1d_route`]) against a CPU
//! reference, and against the blocked 128 kernel it replaces.
//!
//! The route covers Piper's HiFi-GAN resblock convolutions only, so each case
//! states whether it must take it and the test checks the pipeline that ran: a
//! geometry silently falling back would still pass the numerics. Cases off the
//! route each break exactly one condition of the predicate.
//!
//! The CPU reference is f64 on sampled rows and columns (borders, tile edges
//! and a regular stride), to keep debug-build runtimes short; the comparison
//! with the replaced kernel covers every output. Agreement with that kernel is
//! asserted within the usual tolerance; bit identity is only reported, as it is
//! a property of the driver, not of the route. Elements that are NaN on both
//! sides with different payloads are counted apart, as NaN-class equivalent,
//! not as bit-identical.

use onnx_vulkan_core::host_ops::FLOAT;
use onnx_vulkan_core::host_ops::HostTensor;
use onnx_vulkan_core::shaders::conv::{
    self, BLOCKED_K_STEP, BLOCKED_MICRO_TILE_SIZE, Tactic, TacticLimits, Wide1dGeometry,
    tuned_tactic, wide_1d_route,
};
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, InitializerIr, KernelCache, NodeIr, Tensor, execute,
};
use std::collections::HashMap;
use vk_compute::{VkContext, compile_wgsl};

fn pseudo(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    batch: usize,
    c_in: usize,
    c_out: usize,
    group: usize,
    l_in: usize,
    k: usize,
    stride: usize,
    dil: usize,
    pads: (usize, usize),
    bias: bool,
    /// The same conv as a rank-4 `[N, C, L, 1]` node.
    two_d: bool,
    /// Whether the node must run on the wide route.
    wide: bool,
}

impl Case {
    const fn new(name: &'static str, c: usize, l_in: usize, k: usize, dil: usize) -> Self {
        Self {
            name,
            batch: 1,
            c_in: c,
            c_out: c,
            group: 1,
            l_in,
            k,
            stride: 1,
            dil,
            pads: (dil * (k - 1) / 2, dil * (k - 1) / 2),
            bias: true,
            two_d: false,
            wide: true,
        }
    }

    fn l_out(&self) -> usize {
        (self.l_in + self.pads.0 + self.pads.1 - self.dil * (self.k - 1) - 1) / self.stride + 1
    }

    fn gsi(&self) -> usize {
        self.c_in / self.group
    }
}

/// Piper's 18 resblock geometries, at sentence 1 plus 3 frames (L not a
/// multiple of 128), then the route's edges.
fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (c, l) in [(256, 8 * 563), (128, 64 * 563)] {
        for k in [3, 7, 11] {
            for d in [1, 3, 5] {
                out.push(Case::new("piper", c, l, k, d));
            }
        }
    }
    let base = Case::new("", 128, 4501, 3, 1);
    out.extend([
        // rows past C_out in the last tile, L_out not a multiple of 4
        Case {
            name: "C_out 200",
            c_out: 200,
            l_in: 2501,
            ..base
        },
        // C_in a multiple of 16 but not of the tile
        Case {
            name: "C_in 144, dilation 2",
            c_in: 144,
            dil: 2,
            pads: (2, 2),
            ..base
        },
        Case {
            name: "valid padding, no bias",
            k: 7,
            pads: (0, 0),
            bias: false,
            ..base
        },
        // begin pad larger than the dilated kernel span: whole columns of
        // padding on the left, an output longer than the input
        Case {
            name: "begin pad beyond the kernel span",
            dil: 2,
            pads: (9, 0),
            ..base
        },
        Case {
            name: "end pad only",
            k: 5,
            pads: (0, 7),
            ..base
        },
        // ---- off the route, one condition each
        Case {
            name: "C_in 120 (not a multiple of 16)",
            c_in: 120,
            wide: false,
            ..base
        },
        Case {
            name: "stride 2",
            l_in: 9001,
            stride: 2,
            wide: false,
            ..base
        },
        Case {
            name: "batch 2",
            batch: 2,
            wide: false,
            ..base
        },
        Case {
            name: "2D node [N, C, L, 1]",
            two_d: true,
            wide: false,
            ..base
        },
        Case {
            name: "group 2",
            group: 2,
            wide: false,
            ..base
        },
        Case {
            name: "C_out 64 (other tactic)",
            c_out: 64,
            wide: false,
            ..base
        },
    ]);
    out
}

/// The predicate as the interpreter evaluates it, with this device's limits.
fn route(c: &Case) -> bool {
    let ctx = VkContext::new().expect("Vulkan context");
    let limits = TacticLimits::from_vulkan(
        ctx.compute_limits(),
        ctx.compute_limits().max_storage_buffer_bytes,
    );
    let tactic = tuned_tactic(c.group, c.l_out(), c.c_out, c.gsi() * c.k);
    wide_1d_route(
        tactic,
        Wide1dGeometry {
            spatial: if c.two_d { 2 } else { 1 },
            batch: c.batch,
            group: c.group,
            stride: c.stride,
            c_in: c.c_in,
            c_out: c.c_out,
            taps: c.k,
            dil: c.dil,
            pad: c.pads.0,
            l_in: c.l_in,
            l_out: c.l_out(),
        },
        limits,
    )
}

const BLOCKED_128: Tactic = Tactic::Blocked {
    output_tile: 128,
    k_step: BLOCKED_K_STEP,
    micro_tile: BLOCKED_MICRO_TILE_SIZE,
};

/// Inputs; with `poison`, non-finite values at the borders: `W[0, 0, last]` NaN,
/// `W[1, 0, 0]` +Inf, `X[·, 3, 0]` +Inf, `X[·, 5, L-1]` NaN.
fn inputs(c: &Case, poison: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut x = pseudo(c.batch * c.c_in * c.l_in, 7);
    // outputs of order 1 whatever K, so the tolerance means the same everywhere
    let scale = 1.0 / ((c.gsi() * c.k) as f32).sqrt();
    let mut w: Vec<f32> = pseudo(c.c_out * c.gsi() * c.k, 11)
        .into_iter()
        .map(|v| v * scale)
        .collect();
    let b = pseudo(c.c_out, 13);
    if poison {
        w[c.k - 1] = f32::NAN;
        w[c.gsi() * c.k] = f32::INFINITY;
        for n in 0..c.batch {
            x[(n * c.c_in + 3) * c.l_in] = f32::INFINITY;
            x[(n * c.c_in + 5) * c.l_in + c.l_in - 1] = f32::NAN;
        }
    }
    (x, w, b)
}

/// Runs the node through the interpreter; also returns whether the wide
/// pipeline was built.
fn run(c: &Case, x: &[f32], w: &[f32], b: &[f32]) -> (Vec<f32>, bool) {
    let tail = |v: usize| {
        if c.two_d {
            vec![v as i64, 1]
        } else {
            vec![v as i64]
        }
    };
    let mut attrs = HashMap::new();
    attrs.insert("kernel_shape".to_string(), AttrValue::Ints(tail(c.k)));
    attrs.insert("strides".to_string(), AttrValue::Ints(tail(c.stride)));
    attrs.insert("dilations".to_string(), AttrValue::Ints(tail(c.dil)));
    attrs.insert("group".to_string(), AttrValue::Int(c.group as i64));
    let pads = if c.two_d {
        vec![c.pads.0 as i64, 0, c.pads.1 as i64, 0]
    } else {
        vec![c.pads.0 as i64, c.pads.1 as i64]
    };
    attrs.insert("pads".to_string(), AttrValue::Ints(pads));

    let init = |shape: Vec<i64>, v: &[f32]| InitializerIr {
        dtype: FLOAT,
        shape,
        data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    let mut initializers = HashMap::new();
    let mut w_shape = vec![c.c_out as i64, c.gsi() as i64];
    w_shape.extend(tail(c.k));
    initializers.insert("W".to_string(), init(w_shape, w));
    let mut node_inputs = vec!["X".to_string(), "W".to_string()];
    if c.bias {
        initializers.insert("B".to_string(), init(vec![c.c_out as i64], b));
        node_inputs.push("B".to_string());
    }
    let ir = GraphIr {
        nodes: vec![NodeIr {
            domain: String::new(),
            op: "Conv".to_string(),
            opset: 11,
            name: "conv".to_string(),
            inputs: node_inputs,
            outputs: vec!["Y".to_string()],
            attrs,
        }],
        initializers,
        inputs: vec!["X".to_string()],
        outputs: vec!["Y".to_string()],
        ..Default::default()
    };

    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    let mut x_shape = vec![c.batch as i64, c.c_in as i64];
    x_shape.extend(tail(c.l_in));
    env.set("X", Tensor::Host(HostTensor::from_f32(x_shape, x)));
    execute(&ir, &mut env).expect("Conv execution");
    let got = env
        .host("Y")
        .expect("output on host")
        .clone()
        .to_f32()
        .expect("output f32");
    env.finish();
    let wide = cache
        .pipeline_keys()
        .iter()
        .any(|key| key.operation() == "Conv_wide1d");
    (got, wide)
}

/// The blocked 128 kernel the route replaces, dispatched directly.
fn shipped(c: &Case, x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let (l_out, kdepth) = (c.l_out(), c.c_in * c.k);
    let tactic = tuned_tactic(1, l_out, c.c_out, kdepth);
    assert_eq!(tactic, BLOCKED_128, "«{}»: shipped tactic", c.name);
    let ctx = VkContext::new().expect("Vulkan context");
    let source = conv::generated_source(tactic).expect("blocked 128 source");
    let pipe = ctx
        .create_pipeline(
            &compile_wgsl(&source).expect("WGSL"),
            conv::BINDINGS,
            conv::PUSH_BYTES,
        )
        .expect("pipeline");
    let upload = |v: &[f32]| {
        let buf = ctx
            .create_storage_buffer((4 * v.len().max(1)) as u64)
            .expect("buffer");
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        ctx.stream_upload(&buf, &bytes).expect("upload");
        buf
    };
    let (xb, wb, bb) = (upload(x), upload(w), upload(b));
    let total = c.c_out * l_out;
    let ob = ctx
        .create_storage_buffer((4 * total) as u64)
        .expect("buffer");
    let fields = [
        total,
        c.c_in,
        c.c_out,
        1,
        c.l_in,
        1,
        l_out,
        1,
        c.k,
        1,
        1,
        1,
        c.pads.0,
        0,
        c.dil,
        1,
        c.c_in,
        usize::from(c.bias),
        1,
    ];
    let push: Vec<u8> = fields
        .iter()
        .flat_map(|&v| (v as u32).to_le_bytes())
        .collect();
    let grid = tactic.dispatch_grid(total as u32, l_out as u32, c.c_out as u32, 1);
    ctx.stream_dispatch(&pipe, &[&xb, &wb, &bb, &ob], &push, grid)
        .expect("dispatch");
    ctx.stream_download(&ob, 4 * total)
        .expect("download")
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}

/// Indices worth checking on an axis of `n`: both borders, both sides of every
/// `tile` boundary, and every `stride`-th.
fn sample(n: usize, border: usize, tile: usize, stride: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..border.min(n))
        .chain(n.saturating_sub(border)..n)
        .collect();
    for t in (tile..n).step_by(tile) {
        v.extend([t - 1, t]);
    }
    v.extend((0..n).step_by(stride));
    v.sort_unstable();
    v.dedup();
    v
}

/// ONNX `Conv` in f64, with the padding semantics of the kernel that runs. The
/// GEMM family (blocked, wide) stages a tap outside `X` as a zero and multiplies
/// it, so a non-finite weight turns it into a NaN, as an im2col + GEMM CPU
/// implementation does; the direct kernel (grouped convolutions) skips the tap,
/// so the same weight contributes nothing there.
fn reference(c: &Case, x: &[f32], w: &[f32], b: &[f32], n: usize, m: usize, o: usize) -> f64 {
    let gso = c.c_out / c.group;
    let g = m / gso;
    let skip = matches!(
        tuned_tactic(c.group, c.l_out(), c.c_out, c.gsi() * c.k),
        Tactic::Direct { .. }
    );
    let mut acc = 0.0f64;
    for cg in 0..c.gsi() {
        let ic = g * c.gsi() + cg;
        for t in 0..c.k {
            let pos = (o * c.stride + t * c.dil) as i64 - c.pads.0 as i64;
            let xv = if pos >= 0 && (pos as usize) < c.l_in {
                x[(n * c.c_in + ic) * c.l_in + pos as usize] as f64
            } else if skip {
                continue;
            } else {
                0.0
            };
            acc += w[(m * c.gsi() + cg) * c.k + t] as f64 * xv;
        }
    }
    if c.bias {
        acc += b[m] as f64;
    }
    acc
}

fn close(got: f64, want: f64) -> bool {
    if want.is_nan() {
        got.is_nan()
    } else if want.is_infinite() {
        got == want
    } else {
        (got - want).abs() <= 1e-5 + 1e-4 * want.abs()
    }
}

fn check(c: &Case, poison: bool) {
    let (x, w, b) = inputs(c, poison);
    assert_eq!(route(c), c.wide, "«{}»: predicate", c.name);
    let (got, wide) = run(c, &x, &w, &b);
    assert_eq!(wide, c.wide, "«{}»: pipeline that ran", c.name);
    let l_out = c.l_out();
    assert_eq!(got.len(), c.batch * c.c_out * l_out, "«{}»: length", c.name);

    let mut worst = 0.0f64;
    for n in 0..c.batch {
        for m in sample(c.c_out, 8, 64, 13) {
            for o in sample(l_out, 40, 128, 97) {
                let want = reference(c, &x, &w, &b, n, m, o);
                let g = got[(n * c.c_out + m) * l_out + o] as f64;
                assert!(
                    close(g, want),
                    "«{}» poison={poison} [{n}, {m}, {o}]: {g} != {want}",
                    c.name
                );
                if want.is_finite() {
                    worst = worst.max((g - want).abs());
                }
            }
        }
    }

    // (bit-identical, NaN on both sides with different bits)
    let mut identical = None;
    if c.wide {
        let base = shipped(c, &x, &w, &b);
        for (i, (g, s)) in got.iter().zip(&base).enumerate() {
            assert!(
                close(*g as f64, *s as f64),
                "«{}» poison={poison} element {i}: wide {g} != blocked 128 {s}",
                c.name
            );
        }
        let pairs = || got.iter().zip(&base);
        let same = pairs().filter(|(g, s)| g.to_bits() == s.to_bits()).count();
        let nan_class = pairs()
            .filter(|(g, s)| g.to_bits() != s.to_bits() && g.is_nan() && s.is_nan())
            .count();
        identical = Some((same, nan_class));
    }
    eprintln!(
        "{:<34} C {:>3}→{:>3} L {:>5}→{:>5} k {:>2} d {} wide {:<5} poison {:<5} max|Δ ref| {worst:.1e} vs blocked 128: {}",
        c.name,
        c.c_in,
        c.c_out,
        c.l_in,
        l_out,
        c.k,
        c.dil,
        c.wide,
        poison,
        identical.map_or("-".to_string(), |(s, n)| format!(
            "bit-identical {s}/{}, NaN-class equivalent {n}",
            got.len()
        )),
    );
}

#[test]
fn conv_matches_the_reference_on_and_off_the_wide_route() {
    for c in cases() {
        check(&c, false);
    }
}

/// Non-finite weights and inputs at the borders land where the replaced kernel
/// and an im2col reference put them, on and off the route.
#[test]
fn non_finite_values_propagate_as_in_the_blocked_kernel() {
    for c in cases() {
        check(&c, true);
    }
}
