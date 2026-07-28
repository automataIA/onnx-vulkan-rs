//! 2D bilinear `GridSample`, the sampling of deformable attention.
//!
//! rfdetr uses it with `mode = bilinear`, `padding_mode = zeros`,
//! `align_corners = 0`; the tests cover that form plus `border` and
//! `align_corners = 1`, and verify that unsupported forms are not claimed.
//! The CPU reference is written here.

use onnx_vulkan_core::host_ops::HostTensor;
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, InitializerIr, KernelCache, NodeIr, Tensor, execute,
    is_implemented_node,
};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: "GridSample".to_string(),
        since_version: 16,
        name: "GridSample_0".to_string(),
        inputs: vec!["x".to_string(), "grid".to_string()],
        outputs: vec!["out".to_string()],
        attrs: attrs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect(),
    }
}

fn run(ir: &GraphIr, x: HostTensor, grid: HostTensor) -> (Vec<f32>, Vec<i64>) {
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    env.set("x", Tensor::Host(x));
    env.set("grid", Tensor::Host(grid));
    execute(ir, &mut env).expect("graph execution");
    let out = env.host("out").expect("output on host").clone();
    env.finish();
    (out.to_f32().expect("output f32"), out.shape)
}

fn graph(attrs: &[(&str, AttrValue)]) -> GraphIr {
    GraphIr {
        nodes: vec![node(attrs)],
        initializers: HashMap::<String, InitializerIr>::new(),
        inputs: vec!["x".to_string(), "grid".to_string()],
        outputs: vec!["out".to_string()],
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

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "different lengths");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5 + 1e-4 * w.abs(),
            "element {i}: {g} != {w}"
        );
    }
}

/// CPU reference: `x` [n, c, h, w], `grid` [n, ho, wo, 2] in (x, y) order.
#[allow(clippy::too_many_arguments)]
fn grid_sample_ref(
    x: &[f32],
    grid: &[f32],
    (n, c, h, w): (usize, usize, usize, usize),
    (ho, wo): (usize, usize),
    align: bool,
    border: bool,
) -> Vec<f32> {
    let unnorm = |v: f32, len: usize| -> f32 {
        if align {
            (v + 1.0) * (len as f32 - 1.0) * 0.5
        } else {
            ((v + 1.0) * len as f32 - 1.0) * 0.5
        }
    };
    let at = |bn: usize, ch: usize, iy: i64, ix: i64| -> f32 {
        let (mut yy, mut xx) = (iy, ix);
        if border {
            yy = yy.clamp(0, h as i64 - 1);
            xx = xx.clamp(0, w as i64 - 1);
        } else if yy < 0 || yy >= h as i64 || xx < 0 || xx >= w as i64 {
            return 0.0;
        }
        x[((bn * c + ch) * h + yy as usize) * w + xx as usize]
    };

    let mut out = vec![0.0f32; n * c * ho * wo];
    for bn in 0..n {
        for ch in 0..c {
            for oy in 0..ho {
                for ox in 0..wo {
                    let g = ((bn * ho + oy) * wo + ox) * 2;
                    let fx = unnorm(grid[g], w);
                    let fy = unnorm(grid[g + 1], h);
                    let (x0, y0) = (fx.floor(), fy.floor());
                    let (tx, ty) = (fx - x0, fy - y0);
                    let (x0, y0) = (x0 as i64, y0 as i64);
                    let top = at(bn, ch, y0, x0) * (1.0 - tx) + at(bn, ch, y0, x0 + 1) * tx;
                    let bot = at(bn, ch, y0 + 1, x0) * (1.0 - tx) + at(bn, ch, y0 + 1, x0 + 1) * tx;
                    out[((bn * c + ch) * ho + oy) * wo + ox] = top * (1.0 - ty) + bot * ty;
                }
            }
        }
    }
    out
}

/// The rfdetr case: bilinear, `zeros`, `align_corners = 0`. The grid goes
/// out of bounds in several places, so the `zeros` policy is actually exercised.
#[test]
fn bilinear_zeros_matches_cpu() {
    let (n, c, h, w, ho, wo) = (2usize, 3, 5, 4, 3, 6);
    let x = pseudo(n * c * h * w, 3);
    // pseudo() stays in [-1, 1); multiplying by 1.4 pushes part of it outside
    let grid: Vec<f32> = pseudo(n * ho * wo * 2, 9).iter().map(|v| v * 1.4).collect();
    let ir = graph(&[
        ("mode", AttrValue::String("bilinear".to_string())),
        ("padding_mode", AttrValue::String("zeros".to_string())),
        ("align_corners", AttrValue::Int(0)),
    ]);

    let (got, out_shape) = run(
        &ir,
        HostTensor::from_f32(vec![n as i64, c as i64, h as i64, w as i64], &x),
        HostTensor::from_f32(vec![n as i64, ho as i64, wo as i64, 2], &grid),
    );
    assert_eq!(out_shape, vec![n as i64, c as i64, ho as i64, wo as i64]);
    assert_close(
        &got,
        &grid_sample_ref(&x, &grid, (n, c, h, w), (ho, wo), false, false),
    );
}

/// `padding_mode = border` with `align_corners = 1`: the other implemented
/// combination, where points outside the boundary take the last valid value.
#[test]
fn bilinear_border_align_corners_matches_cpu() {
    let (n, c, h, w, ho, wo) = (1usize, 2, 4, 4, 4, 4);
    let x = pseudo(n * c * h * w, 17);
    let grid: Vec<f32> = pseudo(n * ho * wo * 2, 5).iter().map(|v| v * 1.5).collect();
    let ir = graph(&[
        ("mode", AttrValue::String("bilinear".to_string())),
        ("padding_mode", AttrValue::String("border".to_string())),
        ("align_corners", AttrValue::Int(1)),
    ]);

    let (got, _) = run(
        &ir,
        HostTensor::from_f32(vec![n as i64, c as i64, h as i64, w as i64], &x),
        HostTensor::from_f32(vec![n as i64, ho as i64, wo as i64, 2], &grid),
    );
    assert_close(
        &got,
        &grid_sample_ref(&x, &grid, (n, c, h, w), (ho, wo), true, true),
    );
}

/// Unsupported `mode` and `padding_mode` must not be claimed.
#[test]
fn unsupported_modes_are_not_claimed() {
    assert!(
        is_implemented_node(&node(&[(
            "mode",
            AttrValue::String("bilinear".to_string())
        )])),
        "bilinear is implemented"
    );
    assert!(
        is_implemented_node(&node(&[])),
        "bilinear is also the default"
    );
    assert!(
        !is_implemented_node(&node(&[("mode", AttrValue::String("cubic".to_string()))])),
        "cubic not implemented"
    );
    assert!(
        !is_implemented_node(&node(&[(
            "padding_mode",
            AttrValue::String("reflection".to_string())
        )])),
        "reflection not implemented"
    );
}
