//! Is the `Conv` implicit GEMM worth register blocking, and on which shapes?
//!
//! `Conv` with `group == 1` already runs as an implicit GEMM
//! (`shaders::conv::implicit_gemm`): `out[C_out, P] = W[C_out, K] × im2col(X)[K, P]`
//! with `K = C_in·kh·kw` and `P = H_out·W_out`, the im2col columns rebuilt from
//! their index instead of materialized. The algorithm is therefore not in
//! question here — no workspace, no extra read traffic on either side.
//!
//! What is in question is the kernel's shape. It stages a 16×16 tile and keeps
//! one output per thread, so the inner loop spends two shared reads per FMA.
//! That is character for character what `MatMul` and `Gemm` did before register
//! blocking took them to a 64×64 tile with a 4×4 micro-tile in registers —
//! eight reads per sixteen FMAs — and measured ~5× on tile-filling shapes.
//!
//! On the roofline this kernel sits at 3.1% of fp32 peak and 4.0% of bandwidth
//! (7.71 GFLOP and 176.7 MB of traffic in 8.684 ms on a 4070), so neither
//! resource is what binds it.
//!
//! This measures both kernels on every distinct geometry the three Conv-heavy
//! models in the suite actually run — ResNet-50 (20), yolov4 (27), yolov8n
//! (41) — and diffs them, so the routing predicate is written against
//! measurement rather than against the argument above. The shapes where the
//! blocked kernel loses are the point of the exercise, not a footnote: on
//! ResNet-50 it ranges from 0.50× to 3.53× and blanket routing is worth 1.04×,
//! which is how [`WG_FLOOR`] came to exist.
//!
//! Run: `cargo run --release -p onnx-vulkan-core --example conv_blocked`

use onnx_vulkan_core::shaders::conv;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;
use vk_compute::{ComputePipeline, GpuBuffer, VkContext, compile_wgsl};

/// One `Conv` geometry, with how many nodes of the model run it.
struct Shape {
    c_in: usize,
    c_out: usize,
    k: usize,
    h_in: usize,
    h_out: usize,
    stride: usize,
    pad: usize,
    count: usize,
}

/// Every distinct `Conv` geometry in `resnet50-v1-12-qdq`, extracted from the
/// graph rather than hand-written. All of them are square, so one `k` and one
/// spatial extent per side is enough.
#[rustfmt::skip]
const RESNET50: &[Shape] = &[
    Shape { c_in:  256, c_out:  256, k: 3, h_in:  14, h_out:  14, stride: 1, pad: 1, count: 6 },
    Shape { c_in:  256, c_out: 1024, k: 1, h_in:  14, h_out:  14, stride: 1, pad: 0, count: 6 },
    Shape { c_in: 1024, c_out:  256, k: 1, h_in:  14, h_out:  14, stride: 1, pad: 0, count: 5 },
    Shape { c_in:   64, c_out:  256, k: 1, h_in:  56, h_out:  56, stride: 1, pad: 0, count: 4 },
    Shape { c_in:  128, c_out:  128, k: 3, h_in:  28, h_out:  28, stride: 1, pad: 1, count: 4 },
    Shape { c_in:  128, c_out:  512, k: 1, h_in:  28, h_out:  28, stride: 1, pad: 0, count: 4 },
    Shape { c_in:   64, c_out:   64, k: 3, h_in:  56, h_out:  56, stride: 1, pad: 1, count: 3 },
    Shape { c_in:  512, c_out:  128, k: 1, h_in:  28, h_out:  28, stride: 1, pad: 0, count: 3 },
    Shape { c_in:  512, c_out:  512, k: 3, h_in:   7, h_out:   7, stride: 1, pad: 1, count: 3 },
    Shape { c_in:  512, c_out: 2048, k: 1, h_in:   7, h_out:   7, stride: 1, pad: 0, count: 3 },
    Shape { c_in:  256, c_out:   64, k: 1, h_in:  56, h_out:  56, stride: 1, pad: 0, count: 2 },
    Shape { c_in: 2048, c_out:  512, k: 1, h_in:   7, h_out:   7, stride: 1, pad: 0, count: 2 },
    Shape { c_in:    3, c_out:   64, k: 7, h_in: 224, h_out: 112, stride: 2, pad: 3, count: 1 },
    Shape { c_in:   64, c_out:   64, k: 1, h_in:  56, h_out:  56, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  256, c_out:  128, k: 1, h_in:  56, h_out:  28, stride: 2, pad: 0, count: 1 },
    Shape { c_in:  256, c_out:  512, k: 1, h_in:  56, h_out:  28, stride: 2, pad: 0, count: 1 },
    Shape { c_in:  512, c_out:  256, k: 1, h_in:  28, h_out:  14, stride: 2, pad: 0, count: 1 },
    Shape { c_in:  512, c_out: 1024, k: 1, h_in:  28, h_out:  14, stride: 2, pad: 0, count: 1 },
    Shape { c_in: 1024, c_out:  512, k: 1, h_in:  14, h_out:   7, stride: 2, pad: 0, count: 1 },
    Shape { c_in: 1024, c_out: 2048, k: 1, h_in:  14, h_out:   7, stride: 2, pad: 0, count: 1 },
];

/// `yolov4`, 110 `Conv` nodes in 27 geometries. Its spatial extents are an
/// order of magnitude larger than ResNet's, which is the whole reason it is
/// here: `P` is what the predicate turns on.
#[rustfmt::skip]
const YOLOV4: &[Shape] = &[
    Shape { c_in:  512, c_out:  512, k: 3, h_in:  13, h_out:  13, stride: 1, pad: 1, count: 4 },
    Shape { c_in:   64, c_out:   64, k: 1, h_in: 208, h_out: 208, stride: 1, pad: 0, count: 3 },
    Shape { c_in:   64, c_out:   64, k: 1, h_in: 104, h_out: 104, stride: 1, pad: 0, count: 3 },
    Shape { c_in:  128, c_out:  256, k: 3, h_in:  52, h_out:  52, stride: 1, pad: 1, count: 3 },
    Shape { c_in:  128, c_out:   64, k: 1, h_in: 104, h_out: 104, stride: 1, pad: 0, count: 2 },
    Shape { c_in:   64, c_out:   64, k: 3, h_in: 104, h_out: 104, stride: 1, pad: 1, count: 2 },
    Shape { c_in:    3, c_out:   32, k: 3, h_in: 416, h_out: 416, stride: 1, pad: 1, count: 1 },
    Shape { c_in:   32, c_out:   64, k: 3, h_in: 416, h_out: 208, stride: 2, pad: 1, count: 1 },
    Shape { c_in:   64, c_out:   32, k: 1, h_in: 208, h_out: 208, stride: 1, pad: 0, count: 1 },
    Shape { c_in:   32, c_out:   64, k: 3, h_in: 208, h_out: 208, stride: 1, pad: 1, count: 1 },
    Shape { c_in:  128, c_out:   64, k: 1, h_in: 208, h_out: 208, stride: 1, pad: 0, count: 1 },
    Shape { c_in:   64, c_out:  128, k: 3, h_in: 208, h_out: 104, stride: 2, pad: 1, count: 1 },
    Shape { c_in:  128, c_out:  128, k: 1, h_in: 104, h_out: 104, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  128, c_out:  256, k: 3, h_in: 104, h_out:  52, stride: 2, pad: 1, count: 1 },
    Shape { c_in:  256, c_out:  256, k: 1, h_in:  52, h_out:  52, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  256, c_out:  512, k: 3, h_in:  52, h_out:  26, stride: 2, pad: 1, count: 1 },
    Shape { c_in:  512, c_out:  512, k: 1, h_in:  26, h_out:  26, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  512, c_out: 1024, k: 3, h_in:  26, h_out:  13, stride: 2, pad: 1, count: 1 },
    Shape { c_in: 1024, c_out: 1024, k: 1, h_in:  13, h_out:  13, stride: 1, pad: 0, count: 1 },
    Shape { c_in: 2048, c_out:  512, k: 1, h_in:  13, h_out:  13, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  512, c_out:  256, k: 1, h_in:  13, h_out:  13, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  256, c_out:  128, k: 1, h_in:  26, h_out:  26, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  128, c_out:  256, k: 3, h_in:  52, h_out:  26, stride: 2, pad: 1, count: 1 },
    Shape { c_in:  256, c_out:  512, k: 3, h_in:  26, h_out:  13, stride: 2, pad: 1, count: 1 },
    Shape { c_in: 1024, c_out:  255, k: 1, h_in:  13, h_out:  13, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  512, c_out:  255, k: 1, h_in:  26, h_out:  26, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  256, c_out:  255, k: 1, h_in:  52, h_out:  52, stride: 1, pad: 0, count: 1 },
];

/// `yolov8n`, 64 `Conv` nodes in 41 geometries — the widest spread of the
/// three, from `P = 102400` down to `P = 16`, so it exercises both sides of
/// the predicate in one model.
#[rustfmt::skip]
const YOLOV8N: &[Shape] = &[
    Shape { c_in:   64, c_out:   64, k: 3, h_in:  40, h_out:  40, stride: 1, pad: 1, count: 9 },
    Shape { c_in:   32, c_out:   32, k: 3, h_in:  80, h_out:  80, stride: 1, pad: 1, count: 6 },
    Shape { c_in:  128, c_out:  128, k: 3, h_in:  20, h_out:  20, stride: 1, pad: 1, count: 4 },
    Shape { c_in:  384, c_out:  256, k: 1, h_in:  20, h_out:  20, stride: 1, pad: 0, count: 3 },
    Shape { c_in:  192, c_out:  128, k: 1, h_in:  40, h_out:  40, stride: 1, pad: 0, count: 3 },
    Shape { c_in:   16, c_out:   16, k: 3, h_in: 160, h_out: 160, stride: 1, pad: 1, count: 2 },
    Shape { c_in:   64, c_out:   64, k: 1, h_in:  80, h_out:  80, stride: 1, pad: 0, count: 2 },
    Shape { c_in:   64, c_out:   64, k: 3, h_in:  80, h_out:  80, stride: 1, pad: 1, count: 2 },
    Shape { c_in:    3, c_out:   16, k: 3, h_in: 640, h_out: 320, stride: 2, pad: 1, count: 1 },
    Shape { c_in:   16, c_out:   32, k: 3, h_in: 320, h_out: 160, stride: 2, pad: 1, count: 1 },
    Shape { c_in:   32, c_out:   32, k: 1, h_in: 160, h_out: 160, stride: 1, pad: 0, count: 1 },
    Shape { c_in:   48, c_out:   32, k: 1, h_in: 160, h_out: 160, stride: 1, pad: 0, count: 1 },
    Shape { c_in:   32, c_out:   64, k: 3, h_in: 160, h_out:  80, stride: 2, pad: 1, count: 1 },
    Shape { c_in:  128, c_out:   64, k: 1, h_in:  80, h_out:  80, stride: 1, pad: 0, count: 1 },
    Shape { c_in:   64, c_out:  128, k: 3, h_in:  80, h_out:  40, stride: 2, pad: 1, count: 1 },
    Shape { c_in:  128, c_out:  128, k: 1, h_in:  40, h_out:  40, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  256, c_out:  128, k: 1, h_in:  40, h_out:  40, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  128, c_out:  256, k: 3, h_in:  40, h_out:  20, stride: 2, pad: 1, count: 1 },
    Shape { c_in:  256, c_out:  256, k: 1, h_in:  20, h_out:  20, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  256, c_out:  128, k: 1, h_in:  20, h_out:  20, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  512, c_out:  256, k: 1, h_in:  20, h_out:  20, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  384, c_out:  128, k: 1, h_in:  40, h_out:  40, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  192, c_out:   64, k: 1, h_in:  80, h_out:  80, stride: 1, pad: 0, count: 1 },
    Shape { c_in:   96, c_out:   64, k: 1, h_in:  80, h_out:  80, stride: 1, pad: 0, count: 1 },
    Shape { c_in:   64, c_out:   64, k: 3, h_in:  80, h_out:  40, stride: 2, pad: 1, count: 1 },
    Shape { c_in:  128, c_out:  128, k: 3, h_in:  40, h_out:  20, stride: 2, pad: 1, count: 1 },
    Shape { c_in:   64, c_out:   80, k: 3, h_in:  80, h_out:  80, stride: 1, pad: 1, count: 1 },
    Shape { c_in:   80, c_out:   80, k: 3, h_in:  80, h_out:  80, stride: 1, pad: 1, count: 1 },
    Shape { c_in:   80, c_out:   80, k: 1, h_in:  80, h_out:  80, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  128, c_out:   64, k: 3, h_in:  40, h_out:  40, stride: 1, pad: 1, count: 1 },
    Shape { c_in:   64, c_out:   64, k: 1, h_in:  40, h_out:  40, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  128, c_out:   80, k: 3, h_in:  40, h_out:  40, stride: 1, pad: 1, count: 1 },
    Shape { c_in:   80, c_out:   80, k: 3, h_in:  40, h_out:  40, stride: 1, pad: 1, count: 1 },
    Shape { c_in:   80, c_out:   80, k: 1, h_in:  40, h_out:  40, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  256, c_out:   64, k: 3, h_in:  20, h_out:  20, stride: 1, pad: 1, count: 1 },
    Shape { c_in:   64, c_out:   64, k: 3, h_in:  20, h_out:  20, stride: 1, pad: 1, count: 1 },
    Shape { c_in:   64, c_out:   64, k: 1, h_in:  20, h_out:  20, stride: 1, pad: 0, count: 1 },
    Shape { c_in:  256, c_out:   80, k: 3, h_in:  20, h_out:  20, stride: 1, pad: 1, count: 1 },
    Shape { c_in:   80, c_out:   80, k: 3, h_in:  20, h_out:  20, stride: 1, pad: 1, count: 1 },
    Shape { c_in:   80, c_out:   80, k: 1, h_in:  20, h_out:  20, stride: 1, pad: 0, count: 1 },
    Shape { c_in:   16, c_out:    1, k: 1, h_in:   4, h_out:   4, stride: 1, pad: 0, count: 1 },
];

/// Measured `Conv` time in the gate (`runs/sync-fix-1`), so the per-shape
/// totals can be scaled to what the model would actually gain.
const MODELS: &[(&str, &[Shape], f64)] = &[
    ("resnet50-qdq", RESNET50, 8.684),
    ("yolov4", YOLOV4, 58.793),
    ("yolov8n", YOLOV8N, 8.764),
];

/// The implicit GEMM on a 64×64 output tile with a 4×4 micro-tile per thread.
///
/// Same bindings and push constants as `conv::implicit_gemm`, so it is a drop-in
/// for the same dispatch — only the grid divisor changes. `A` is `W`, read
/// straight; `B` is the im2col matrix, whose 16×64 staging tile rebuilds each
/// column from its `K` index exactly as the 16×16 kernel does.
const CONV_BLOCKED: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
struct Push {
    total: u32, c_in: u32, c_out: u32, group: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    phb: u32, pwb: u32, dh: u32, dw: u32, gsi: u32, has_bias: u32,
    split: u32,
}
var<immediate> pc: Push;

const TILE = 64u;
const KSTEP = 16u;
var<workgroup> w_tile: array<f32, 1024>;
var<workgroup> x_tile: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.y * 16u + lid.x;
    let row0 = wid.y * TILE;          // first output channel of the block
    let col0 = wid.x * TILE;          // first output pixel of the block
    let bn = wid.z;
    let pixels = pc.h_out * pc.w_out;
    let kdepth = pc.c_in * pc.kh * pc.kw;
    let ksize = pc.kh * pc.kw;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let ntiles = (kdepth + KSTEP - 1u) / KSTEP;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        let k0 = t * KSTEP;
        // --- stage W: 64 rows × 16 of K, 4 values per thread
        for (var s = 0u; s < 4u; s = s + 1u) {
            let l = tid + s * 256u;
            let gr = row0 + l / KSTEP;
            let gk = k0 + l % KSTEP;
            var v = 0.0;
            if (gr < pc.c_out && gk < kdepth) { v = w[gr * kdepth + gk]; }
            w_tile[l] = v;
        }
        // --- stage im2col: 16 of K × 64 pixels, rebuilt from the index
        for (var s = 0u; s < 4u; s = s + 1u) {
            let l = tid + s * 256u;
            let gk = k0 + l / TILE;
            let gc = col0 + l % TILE;
            var v = 0.0;
            if (gk < kdepth && gc < pixels) {
                let ic = gk / ksize;
                let rem = gk % ksize;
                let r = rem / pc.kw;
                let sx = rem % pc.kw;
                let oh = gc / pc.w_out;
                let ow = gc % pc.w_out;
                let ih = i32(oh) * i32(pc.sh) - i32(pc.phb) + i32(r) * i32(pc.dh);
                let iw = i32(ow) * i32(pc.sw) - i32(pc.pwb) + i32(sx) * i32(pc.dw);
                // out of bounds = zero: the conv's implicit padding
                if (ih >= 0 && ih < i32(pc.h_in) && iw >= 0 && iw < i32(pc.w_in)) {
                    v = x[((bn * pc.c_in + ic) * pc.h_in + u32(ih)) * pc.w_in + u32(iw)];
                }
            }
            x_tile[l] = v;
        }
        workgroupBarrier();
        // --- 4 scalars of W + 4 of im2col per 16 FMAs
        let arow = lid.y * 4u;
        let bcol = lid.x * 4u;
        for (var kk = 0u; kk < KSTEP; kk = kk + 1u) {
            let bo = kk * TILE + bcol;
            let bvec = vec4<f32>(x_tile[bo], x_tile[bo + 1u], x_tile[bo + 2u], x_tile[bo + 3u]);
            acc0 = fma(vec4<f32>(w_tile[(arow + 0u) * KSTEP + kk]), bvec, acc0);
            acc1 = fma(vec4<f32>(w_tile[(arow + 1u) * KSTEP + kk]), bvec, acc1);
            acc2 = fma(vec4<f32>(w_tile[(arow + 2u) * KSTEP + kk]), bvec, acc2);
            acc3 = fma(vec4<f32>(w_tile[(arow + 3u) * KSTEP + kk]), bvec, acc3);
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = row0 + lid.y * 4u + i;
        if (m >= pc.c_out) { continue; }
        var accv = acc0;
        if (i == 1u) { accv = acc1; }
        if (i == 2u) { accv = acc2; }
        if (i == 3u) { accv = acc3; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            let p = col0 + lid.x * 4u + j;
            if (p >= pixels) { continue; }
            var v = accv[j];
            if (pc.has_bias != 0u) { v = v + bias[m]; }
            out[(bn * pc.c_out + m) * pixels + p] = v;
        }
    }
}
"#;

fn pseudo(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        })
        .collect()
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn floats(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = VkContext::new()?;
    if std::env::args().any(|argument| argument == "--device-json") {
        let device = ctx.device_fingerprint();
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "schema_version": 1,
                "device": {
                    "name": device.name(), "vendor_id": device.vendor_id(),
                    "device_id": device.device_id(), "driver_version": device.driver_version(),
                    "api_version": device.api_version(),
                    "pipeline_cache_uuid": hex(device.pipeline_cache_uuid()),
                    "subgroup_size": device.subgroup_size(), "features": device.features(),
                },
                "implementation_digest": hex(conv::implementation_fingerprint().as_bytes()),
            }))?
        );
        return Ok(());
    }
    let tune_shapes = tune_shape_arguments()?;
    if !tune_shapes.is_empty() {
        return tune_model_shapes(&ctx, &tune_shapes);
    }
    if std::env::args().any(|arg| arg == "--list") {
        return report_candidate_space(&ctx);
    }
    if std::env::args().any(|arg| arg == "--search") {
        return search_candidates(&ctx);
    }
    let small = ctx.create_pipeline(
        &compile_wgsl(&conv::implicit_gemm())?,
        conv::BINDINGS,
        conv::PUSH_BYTES,
    )?;
    let blocked = ctx.create_pipeline(
        &compile_wgsl(CONV_BLOCKED)?,
        conv::BINDINGS,
        conv::PUSH_BYTES,
    )?;

    for (model, shapes, gate_ms) in MODELS {
        println!("\n===== {model} =====");
        println!(
            "{:>24} {:>7} {:>6} {:>5} {:>9} {:>7} {:>9} {:>7} {:>8} {:>9}",
            "geometry",
            "P",
            "K",
            "WGs",
            "16x16 ms",
            "TF/s",
            "block ms",
            "TF/s",
            "speedup",
            "max|rel|"
        );
        let (small_ms, blocked_ms, routed_ms, oracle_ms, rel) =
            sweep(&ctx, &small, &blocked, shapes)?;
        println!(
            "\n{model}: 16x16 {small_ms:.2} ms | all-blocked {blocked_ms:.2} ({:.2}x) | \
             predicate {routed_ms:.2} ({:.2}x) | oracle {oracle_ms:.2} ({:.2}x)",
            small_ms / blocked_ms,
            small_ms / routed_ms,
            small_ms / oracle_ms,
        );
        println!(
            "  gate Conv {gate_ms:.2} ms -> {:.2} ms with the predicate; worst relative diff {rel:.2e}",
            gate_ms * routed_ms / small_ms,
        );
    }
    Ok(())
}

fn report_candidate_space(ctx: &VkContext) -> Result<(), Box<dyn std::error::Error>> {
    let limits = conv::TacticLimits::from_vulkan(
        ctx.compute_limits(),
        ctx.compute_limits().max_storage_buffer_bytes,
    );
    let mut total_geometries = 0usize;
    let mut total_before = 0usize;
    let mut total_after = 0usize;
    let mut total_control_rejected = 0usize;
    let mut rejections = BTreeMap::<conv::TacticRejection, usize>::new();

    for (model, shapes, _) in MODELS {
        let mut model_before = 0usize;
        let mut model_after = 0usize;
        for shape in *shapes {
            let pixels = shape.h_out * shape.h_out;
            let kdepth = shape.c_in * shape.k * shape.k;
            let geometry = conv::TacticGeometry {
                batch: 1,
                group: 1,
                pixels: pixels as u32,
                c_out: shape.c_out as u32,
                kdepth: kdepth as u32,
                total: (pixels * shape.c_out) as u64,
            };
            let candidates = conv::candidate_space(geometry.group);
            model_before += candidates.len();
            for candidate in candidates {
                match candidate.viability(geometry, limits) {
                    Ok(()) => model_after += 1,
                    Err(reason) => *rejections.entry(reason).or_default() += 1,
                }
            }
            let control = conv::control_tactic(1, pixels, shape.c_out, kdepth);
            total_control_rejected += usize::from(!control.is_viable(geometry, limits));
        }
        println!(
            "{{\"kind\":\"conv_candidate_space_model\",\"model\":\"{model}\",\"geometries\":{},\"before\":{model_before},\"after\":{model_after}}}",
            shapes.len()
        );
        total_geometries += shapes.len();
        total_before += model_before;
        total_after += model_after;
    }

    let rejection_json = rejections
        .iter()
        .map(|(reason, count)| format!("\"{reason:?}\":{count}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{{\"kind\":\"conv_candidate_space_summary\",\"geometries\":{total_geometries},\"candidates_per_group1_geometry\":{},\"before\":{total_before},\"after\":{total_after},\"control_rejected\":{total_control_rejected},\"rejections\":{{{rejection_json}}}}}",
        conv::candidate_space(1).len()
    );
    Ok(())
}

const SEARCH_SAMPLES: u32 = 20;
const SEARCH_RANDOM_SEED: u64 = 0x6f6e_6e78_766b_7273;
const MAX_REL_ERROR: f32 = 1.0e-4;
type Err = Box<dyn std::error::Error>;

#[derive(Clone, Copy)]
struct CandidateMeasurement {
    ms: f64,
    max_rel: f32,
}

fn tune_shape_arguments() -> Result<Vec<Shape>, Err> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let mut shapes = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        if arguments[index] != "--tune-shape" {
            index += 1;
            continue;
        }
        let spec = arguments
            .get(index + 1)
            .ok_or("--tune-shape requires C_IN,C_OUT,K,H_IN,H_OUT,STRIDE,PAD")?;
        let values = spec
            .split(',')
            .map(|value| value.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() != 7 || values[..6].contains(&0) {
            return Err(
                "--tune-shape requires seven integers; all except PAD must be positive".into(),
            );
        }
        shapes.push(Shape {
            c_in: values[0],
            c_out: values[1],
            k: values[2],
            h_in: values[3],
            h_out: values[4],
            stride: values[5],
            pad: values[6],
            count: 1,
        });
        index += 2;
    }
    Ok(shapes)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn tune_model_shapes(ctx: &VkContext, shapes: &[Shape]) -> Result<(), Err> {
    let limits = conv::TacticLimits::from_vulkan(
        ctx.compute_limits(),
        ctx.compute_limits().max_storage_buffer_bytes,
    );
    let geometries = shapes
        .iter()
        .map(|shape| conv::TacticGeometry {
            batch: 1,
            group: 1,
            pixels: (shape.h_out * shape.h_out) as u32,
            c_out: shape.c_out as u32,
            kdepth: (shape.c_in * shape.k * shape.k) as u32,
            total: (shape.h_out * shape.h_out * shape.c_out) as u64,
        })
        .collect::<Vec<_>>();
    let mut pipeline_tactics = BTreeSet::new();
    for tactic in conv::focused_search_space() {
        if geometries
            .iter()
            .any(|geometry| tactic.is_viable(*geometry, limits))
        {
            pipeline_tactics.insert(normalized_pipeline_tactic(tactic));
        }
    }
    for geometry in &geometries {
        pipeline_tactics.insert(normalized_pipeline_tactic(conv::control_tactic(
            1,
            geometry.pixels as usize,
            geometry.c_out as usize,
            geometry.kdepth as usize,
        )));
    }
    let mut pipelines = BTreeMap::new();
    for tactic in pipeline_tactics {
        let source = conv::generated_source(tactic)
            .ok_or_else(|| format!("no generated source for {tactic:?}"))?;
        pipelines.insert(
            tactic,
            ctx.create_pipeline(&compile_wgsl(&source)?, conv::BINDINGS, conv::PUSH_BYTES)?,
        );
    }
    let reduce = ctx.create_pipeline(
        &compile_wgsl(conv::SPLIT_REDUCE)?,
        conv::SPLIT_REDUCE_BINDINGS,
        conv::PUSH_BYTES,
    )?;
    let device = ctx.device_fingerprint();
    for shape in shapes {
        let prepared = PreparedShape::new(ctx, shape)?;
        let geometry = prepared.geometry;
        let control = conv::control_tactic(
            1,
            geometry.pixels as usize,
            geometry.c_out as usize,
            geometry.kdepth as usize,
        );
        prepared.execute(ctx, tactic_pipeline(&pipelines, control)?, &reduce, control)?;
        let reference = prepared.output(ctx)?;
        let mut candidates = conv::focused_search_space()
            .into_iter()
            .filter(|tactic| tactic.is_viable(geometry, limits))
            .collect::<Vec<_>>();
        if !candidates.contains(&control) {
            candidates.push(control);
        }
        let mut measurements = BTreeMap::new();
        for tactic in &candidates {
            measurements.insert(
                *tactic,
                measure_candidate(ctx, &prepared, &pipelines, &reduce, *tactic, &reference)?,
            );
        }
        let (provisional_winner, _) = best_of(&candidates, &measurements)
            .ok_or("no Conv tactic passed the correctness gate")?;
        let (control_ms, winner_ms) = interleaved_pair(
            ctx,
            &prepared,
            &pipelines,
            &reduce,
            control,
            provisional_winner,
        )?;
        let winner = if winner_ms < control_ms {
            provisional_winner
        } else {
            control
        };
        let evidence = measurements
            .get(&winner)
            .and_then(|measurement| *measurement)
            .ok_or("selected Conv tactic has no correctness evidence")?;
        let samples = prepared.timestamp_samples(
            ctx,
            tactic_pipeline(&pipelines, winner)?,
            &reduce,
            winner,
        )?;
        let median = median_ns(samples.clone());
        let minimum = *samples
            .iter()
            .min()
            .ok_or("winner produced no timestamp sample")?;
        let maximum = *samples
            .iter()
            .max()
            .ok_or("winner produced no timestamp sample")?;
        let (tactic_id, parameters) = winner.persistent_parts();
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "schema_version": 1,
                "family": "conv-f32",
                "geometry": {
                    "c_in": shape.c_in, "c_out": shape.c_out, "kernel": shape.k,
                    "h_in": shape.h_in, "h_out": shape.h_out,
                    "stride": shape.stride, "pad": shape.pad,
                },
                "device": {
                    "name": device.name(), "vendor_id": device.vendor_id(),
                    "device_id": device.device_id(), "driver_version": device.driver_version(),
                    "api_version": device.api_version(),
                    "pipeline_cache_uuid": hex(device.pipeline_cache_uuid()),
                    "subgroup_size": device.subgroup_size(), "features": device.features(),
                },
                "implementation_digest": hex(conv::implementation_fingerprint().as_bytes()),
                "tactic": {
                    "family": tactic_id.family(), "id": tactic_id.variant(),
                    "parameters": parameters,
                },
                "measurement": {
                    "samples": samples.len(), "median_gpu_ns": median,
                    "min_gpu_ns": minimum, "max_gpu_ns": maximum,
                    "diagnostic_samples_gpu_ns": samples,
                },
                "correctness": {"kind": "max_rel", "passed": true, "max_rel": evidence.max_rel},
                "candidate_count": candidates.len(),
                "interleaved": {
                    "control_ms": control_ms, "provisional_winner_ms": winner_ms,
                    "selected_control": winner == control,
                },
            }))?
        );
        prepared.destroy(ctx);
    }
    ctx.destroy_pipeline(reduce);
    for (_, pipeline) in pipelines {
        ctx.destroy_pipeline(pipeline);
    }
    Ok(())
}

struct PreparedShape {
    x: GpuBuffer,
    w: GpuBuffer,
    b: GpuBuffer,
    out: GpuBuffer,
    partials: GpuBuffer,
    push: Vec<u8>,
    geometry: conv::TacticGeometry,
}

impl PreparedShape {
    fn new(ctx: &VkContext, shape: &Shape) -> Result<Self, Err> {
        let pixels = shape.h_out * shape.h_out;
        let kdepth = shape.c_in * shape.k * shape.k;
        let total = shape.c_out * pixels;
        let x = pseudo(shape.c_in * shape.h_in * shape.h_in, 3);
        let w = pseudo(shape.c_out * kdepth, 5);
        let b = pseudo(shape.c_out, 7);
        let x_buf = ctx.create_storage_buffer((4 * x.len()) as u64)?;
        let w_buf = ctx.create_storage_buffer((4 * w.len()) as u64)?;
        let b_buf = ctx.create_storage_buffer((4 * b.len()) as u64)?;
        let out = ctx.create_storage_buffer((4 * total) as u64)?;
        let partials = ctx.create_storage_buffer((4 * total * 32) as u64)?;
        ctx.stream_upload(&x_buf, &bytes(&x))?;
        ctx.stream_upload(&w_buf, &bytes(&w))?;
        ctx.stream_upload(&b_buf, &bytes(&b))?;
        ctx.flush()?;

        let mut push = Vec::with_capacity(conv::PUSH_BYTES as usize);
        for value in [
            total as u32,
            shape.c_in as u32,
            shape.c_out as u32,
            1,
            shape.h_in as u32,
            shape.h_in as u32,
            shape.h_out as u32,
            shape.h_out as u32,
            shape.k as u32,
            shape.k as u32,
            shape.stride as u32,
            shape.stride as u32,
            shape.pad as u32,
            shape.pad as u32,
            1,
            1,
            shape.c_in as u32,
            1,
            1,
        ] {
            push.extend_from_slice(&value.to_le_bytes());
        }
        Ok(Self {
            x: x_buf,
            w: w_buf,
            b: b_buf,
            out,
            partials,
            push,
            geometry: conv::TacticGeometry {
                batch: 1,
                group: 1,
                pixels: pixels as u32,
                c_out: shape.c_out as u32,
                kdepth: kdepth as u32,
                total: total as u64,
            },
        })
    }

    fn push_for(&self, tactic: conv::Tactic) -> Vec<u8> {
        let mut push = self.push.clone();
        push[72..76].copy_from_slice(&tactic.split().to_le_bytes());
        push
    }

    fn execute(
        &self,
        ctx: &VkContext,
        pipeline: &ComputePipeline,
        reduce: &ComputePipeline,
        tactic: conv::Tactic,
    ) -> Result<(), Err> {
        let push = self.push_for(tactic);
        let grid = tactic.dispatch_grid(
            self.geometry.total as u32,
            self.geometry.pixels,
            self.geometry.c_out,
            self.geometry.batch,
        );
        if tactic.split() > 1 {
            ctx.stream_dispatch(
                pipeline,
                &[&self.x, &self.w, &self.b, &self.partials],
                &push,
                grid,
            )?;
            ctx.stream_dispatch(
                reduce,
                &[&self.partials, &self.b, &self.out],
                &push,
                [(self.geometry.total as u32).div_ceil(256), 1, 1],
            )?;
        } else {
            ctx.stream_dispatch(
                pipeline,
                &[&self.x, &self.w, &self.b, &self.out],
                &push,
                grid,
            )?;
        }
        ctx.flush()?;
        Ok(())
    }

    fn output(&self, ctx: &VkContext) -> Result<Vec<f32>, Err> {
        Ok(floats(&ctx.stream_download(
            &self.out,
            self.geometry.total as usize * 4,
        )?))
    }

    fn timestamp_samples(
        &self,
        ctx: &VkContext,
        pipeline: &ComputePipeline,
        reduce: &ComputePipeline,
        tactic: conv::Tactic,
    ) -> Result<Vec<u64>, Err> {
        let push = self.push_for(tactic);
        let grid = tactic.dispatch_grid(
            self.geometry.total as u32,
            self.geometry.pixels,
            self.geometry.c_out,
            self.geometry.batch,
        );
        let main_buffers = if tactic.split() > 1 {
            [&self.x, &self.w, &self.b, &self.partials]
        } else {
            [&self.x, &self.w, &self.b, &self.out]
        };
        let mut samples =
            ctx.measure_dispatch_gpu(pipeline, &main_buffers, &push, grid, SEARCH_SAMPLES, 1)?;
        if tactic.split() > 1 {
            let reduction = ctx.measure_dispatch_gpu(
                reduce,
                &[&self.partials, &self.b, &self.out],
                &push,
                [(self.geometry.total as u32).div_ceil(256), 1, 1],
                SEARCH_SAMPLES,
                1,
            )?;
            for (sample, reduction) in samples.iter_mut().zip(reduction) {
                *sample = sample.saturating_add(reduction);
            }
        }
        Ok(samples)
    }

    fn destroy(self, ctx: &VkContext) {
        for buffer in [self.x, self.w, self.b, self.out, self.partials] {
            ctx.destroy_buffer(buffer);
        }
    }
}

fn normalized_pipeline_tactic(tactic: conv::Tactic) -> conv::Tactic {
    match tactic {
        conv::Tactic::SplitK {
            output_tile,
            k_step,
            micro_tile,
            ..
        } => conv::Tactic::SplitK {
            output_tile,
            k_step,
            micro_tile,
            split: 2,
        },
        other => other,
    }
}

fn median_ns(mut samples: Vec<u64>) -> u64 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn compare(reference: &[f32], candidate: &[f32]) -> Option<f32> {
    let mut max_rel = 0.0f32;
    for (&expected, &actual) in reference.iter().zip(candidate) {
        if !expected.is_finite() || !actual.is_finite() {
            return None;
        }
        max_rel = max_rel.max((expected - actual).abs() / expected.abs().max(1.0));
    }
    (max_rel <= MAX_REL_ERROR).then_some(max_rel)
}

fn random_candidates(
    geometry: conv::TacticGeometry,
    limits: conv::TacticLimits,
) -> Vec<conv::Tactic> {
    let mut candidates = conv::reduced_exhaustive_space()
        .into_iter()
        .filter(|&tactic| tactic.is_viable(geometry, limits))
        .collect::<Vec<_>>();
    let mut state = SEARCH_RANDOM_SEED
        ^ u64::from(geometry.pixels)
        ^ (u64::from(geometry.c_out) << 21)
        ^ (u64::from(geometry.kdepth) << 42);
    for index in (1..candidates.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        candidates.swap(index, (state as usize) % (index + 1));
    }
    candidates.truncate(conv::focused_search_space().len());
    candidates
}

fn tactic_pipeline(
    pipelines: &BTreeMap<conv::Tactic, ComputePipeline>,
    tactic: conv::Tactic,
) -> Result<&ComputePipeline, Err> {
    pipelines
        .get(&normalized_pipeline_tactic(tactic))
        .ok_or_else(|| format!("missing pipeline for {tactic:?}").into())
}

fn measure_candidate(
    ctx: &VkContext,
    prepared: &PreparedShape,
    pipelines: &BTreeMap<conv::Tactic, ComputePipeline>,
    reduce: &ComputePipeline,
    tactic: conv::Tactic,
    reference: &[f32],
) -> Result<Option<CandidateMeasurement>, Err> {
    let pipeline = tactic_pipeline(pipelines, tactic)?;
    prepared.execute(ctx, pipeline, reduce, tactic)?;
    let output = prepared.output(ctx)?;
    let Some(max_rel) = compare(reference, &output) else {
        return Ok(None);
    };
    let samples = prepared.timestamp_samples(ctx, pipeline, reduce, tactic)?;
    Ok(Some(CandidateMeasurement {
        ms: median_ns(samples) as f64 / 1.0e6,
        max_rel,
    }))
}

fn best_of(
    candidates: &[conv::Tactic],
    measurements: &BTreeMap<conv::Tactic, Option<CandidateMeasurement>>,
) -> Option<(conv::Tactic, CandidateMeasurement)> {
    candidates
        .iter()
        .filter_map(|tactic| {
            measurements
                .get(tactic)
                .and_then(|m| m.map(|m| (*tactic, m)))
        })
        .min_by(|a, b| a.1.ms.total_cmp(&b.1.ms))
}

fn interleaved_pair(
    ctx: &VkContext,
    prepared: &PreparedShape,
    pipelines: &BTreeMap<conv::Tactic, ComputePipeline>,
    reduce: &ComputePipeline,
    control: conv::Tactic,
    winner: conv::Tactic,
) -> Result<(f64, f64), Err> {
    if control == winner {
        let samples = prepared.timestamp_samples(
            ctx,
            tactic_pipeline(pipelines, control)?,
            reduce,
            control,
        )?;
        let ms = median_ns(samples) as f64 / 1.0e6;
        return Ok((ms, ms));
    }
    let mut control_samples = Vec::new();
    let mut winner_samples = Vec::new();
    for round in 0..3 {
        let pair = if round % 2 == 0 {
            [
                (control, &mut control_samples),
                (winner, &mut winner_samples),
            ]
        } else {
            [
                (winner, &mut winner_samples),
                (control, &mut control_samples),
            ]
        };
        for (tactic, samples) in pair {
            samples.extend(prepared.timestamp_samples(
                ctx,
                tactic_pipeline(pipelines, tactic)?,
                reduce,
                tactic,
            )?);
        }
    }
    Ok((
        median_ns(control_samples) as f64 / 1.0e6,
        median_ns(winner_samples) as f64 / 1.0e6,
    ))
}

fn search_candidates(ctx: &VkContext) -> Result<(), Err> {
    let limits = conv::TacticLimits::from_vulkan(
        ctx.compute_limits(),
        ctx.compute_limits().max_storage_buffer_bytes,
    );
    let all_geometries = MODELS
        .iter()
        .flat_map(|(_, shapes, _)| *shapes)
        .map(|shape| {
            let pixels = shape.h_out * shape.h_out;
            conv::TacticGeometry {
                batch: 1,
                group: 1,
                pixels: pixels as u32,
                c_out: shape.c_out as u32,
                kdepth: (shape.c_in * shape.k * shape.k) as u32,
                total: (pixels * shape.c_out) as u64,
            }
        })
        .collect::<Vec<_>>();
    let mut pipeline_tactics = BTreeSet::new();
    for tactic in conv::reduced_exhaustive_space() {
        if all_geometries
            .iter()
            .any(|&geometry| tactic.is_viable(geometry, limits))
        {
            pipeline_tactics.insert(normalized_pipeline_tactic(tactic));
        }
    }
    for geometry in &all_geometries {
        pipeline_tactics.insert(normalized_pipeline_tactic(conv::control_tactic(
            1,
            geometry.pixels as usize,
            geometry.c_out as usize,
            geometry.kdepth as usize,
        )));
    }
    let mut pipelines = BTreeMap::new();
    for tactic in pipeline_tactics {
        let source = conv::generated_source(tactic)
            .ok_or_else(|| format!("no generated source for {tactic:?}"))?;
        pipelines.insert(
            tactic,
            ctx.create_pipeline(&compile_wgsl(&source)?, conv::BINDINGS, conv::PUSH_BYTES)?,
        );
    }
    let reduce = ctx.create_pipeline(
        &compile_wgsl(conv::SPLIT_REDUCE)?,
        conv::SPLIT_REDUCE_BINDINGS,
        conv::PUSH_BYTES,
    )?;

    for (model, shapes, _) in MODELS {
        let mut control_total = 0.0;
        let mut focus_total = 0.0;
        let mut random_total = 0.0;
        let mut proposed_total = 0.0;
        let mut focus_wins = 0usize;
        let mut random_wins = 0usize;
        let mut proposed_wins = 0usize;
        for shape in *shapes {
            let prepared = PreparedShape::new(ctx, shape)?;
            let geometry = prepared.geometry;
            let control = conv::control_tactic(
                1,
                geometry.pixels as usize,
                geometry.c_out as usize,
                geometry.kdepth as usize,
            );
            let control_pipeline = tactic_pipeline(&pipelines, control)?;
            prepared.execute(ctx, control_pipeline, &reduce, control)?;
            let reference = prepared.output(ctx)?;

            let mut focused = conv::focused_search_space()
                .into_iter()
                .filter(|&tactic| tactic.is_viable(geometry, limits))
                .collect::<Vec<_>>();
            if !focused.contains(&control) {
                focused.push(control);
            }
            let proposed = conv::tuned_tactic(
                1,
                geometry.pixels as usize,
                geometry.c_out as usize,
                geometry.kdepth as usize,
            );
            if !focused.contains(&proposed) {
                focused.push(proposed);
            }
            let mut random = random_candidates(geometry, limits);
            if !random.contains(&control) {
                random.push(control);
            }
            let union = focused
                .iter()
                .chain(&random)
                .copied()
                .collect::<BTreeSet<_>>();
            let mut measurements = BTreeMap::new();
            for tactic in union {
                measurements.insert(
                    tactic,
                    measure_candidate(ctx, &prepared, &pipelines, &reduce, tactic, &reference)?,
                );
            }
            let control_measurement = measurements
                .get(&control)
                .and_then(|measurement| *measurement)
                .ok_or_else(|| format!("control rejected for {geometry:?}"))?;
            let (focus_tactic, focus_measurement) =
                best_of(&focused, &measurements).ok_or("focused search found no valid tactic")?;
            let (random_tactic, random_measurement) =
                best_of(&random, &measurements).ok_or("random search found no valid tactic")?;
            let (control_interleaved, focus_interleaved) =
                interleaved_pair(ctx, &prepared, &pipelines, &reduce, control, focus_tactic)?;
            let (proposed_control_ms, proposed_ms) =
                interleaved_pair(ctx, &prepared, &pipelines, &reduce, control, proposed)?;
            focus_wins += usize::from(focus_interleaved < control_interleaved);
            random_wins += usize::from(random_measurement.ms < control_measurement.ms);
            proposed_wins += usize::from(proposed_ms < proposed_control_ms);
            let count = shape.count as f64;
            control_total += control_interleaved * count;
            focus_total += focus_interleaved * count;
            random_total += random_measurement.ms * count;
            proposed_total += proposed_ms * count;
            println!(
                "{{\"kind\":\"conv_search_geometry\",\"model\":\"{model}\",\"c_in\":{},\"c_out\":{},\"kernel\":{},\"pixels\":{},\"count\":{},\"focused_candidates\":{},\"random_candidates\":{},\"control\":\"{control:?}\",\"focused_winner\":\"{focus_tactic:?}\",\"random_winner\":\"{random_tactic:?}\",\"proposed\":\"{proposed:?}\",\"control_ms\":{control_interleaved:.9},\"focused_ms\":{focus_interleaved:.9},\"random_ms\":{:.9},\"proposed_control_ms\":{proposed_control_ms:.9},\"proposed_ms\":{proposed_ms:.9},\"focused_max_rel\":{:.9},\"random_max_rel\":{:.9}}}",
                shape.c_in,
                shape.c_out,
                shape.k,
                geometry.pixels,
                shape.count,
                focused.len(),
                random.len(),
                random_measurement.ms,
                focus_measurement.max_rel,
                random_measurement.max_rel,
            );
            prepared.destroy(ctx);
        }
        println!(
            "{{\"kind\":\"conv_search_model\",\"model\":\"{model}\",\"geometries\":{},\"focused_strict_wins\":{focus_wins},\"random_strict_wins\":{random_wins},\"proposed_strict_wins\":{proposed_wins},\"control_ms\":{control_total:.9},\"focused_ms\":{focus_total:.9},\"random_ms\":{random_total:.9},\"proposed_ms\":{proposed_total:.9},\"focused_speedup\":{:.9},\"random_speedup\":{:.9},\"proposed_speedup\":{:.9}}}",
            shapes.len(),
            control_total / focus_total,
            control_total / random_total,
            control_total / proposed_total,
        );
    }

    ctx.destroy_pipeline(reduce);
    for (_, pipeline) in pipelines {
        ctx.destroy_pipeline(pipeline);
    }
    Ok(())
}

/// `ceil(P/64) · ceil(C_out/64)` — how many workgroups the blocked kernel
/// launches. Below roughly half the GPU's SM count the bigger tile cannot fill
/// the machine and the 16×16 kernel wins; this is the quantity the routing
/// predicate reads.
fn workgroups(pixels: usize, c_out: usize) -> usize {
    pixels.div_ceil(64) * c_out.div_ceil(64)
}

/// Measured on a 4070 (46 SMs): every geometry at or above this went faster
/// blocked, every geometry below it went slower, with no overlap.
const WG_FLOOR: usize = 24;

type Sweep = (f64, f64, f64, f64, f32);

fn sweep(
    ctx: &VkContext,
    small: &vk_compute::ComputePipeline,
    blocked: &vk_compute::ComputePipeline,
    shapes: &[Shape],
) -> Result<Sweep, Box<dyn std::error::Error>> {
    let (mut tot_small, mut tot_blocked) = (0.0f64, 0.0f64);
    let (mut tot_routed, mut tot_oracle) = (0.0f64, 0.0f64);
    let mut worst_rel = 0.0f32;

    for s in shapes {
        let (c_in, c_out, count) = (s.c_in, s.c_out, s.count);
        let (kh, kw) = (s.k, s.k);
        let (h_in, w_in, h_out, w_out) = (s.h_in, s.h_in, s.h_out, s.h_out);
        let (stride, pad) = (s.stride, s.pad);
        let pixels = h_out * w_out;
        let kdepth = c_in * kh * kw;
        let x = pseudo(c_in * h_in * w_in, 3);
        let w = pseudo(c_out * kdepth, 5);
        let b = pseudo(c_out, 7);

        let x_buf = ctx.create_storage_buffer((4 * x.len()) as u64)?;
        let w_buf = ctx.create_storage_buffer((4 * w.len()) as u64)?;
        let b_buf = ctx.create_storage_buffer((4 * b.len()) as u64)?;
        let out_a = ctx.create_storage_buffer((4 * c_out * pixels) as u64)?;
        let out_b = ctx.create_storage_buffer((4 * c_out * pixels) as u64)?;
        ctx.stream_upload(&x_buf, &bytes(&x))?;
        ctx.stream_upload(&w_buf, &bytes(&w))?;
        ctx.stream_upload(&b_buf, &bytes(&b))?;
        ctx.flush()?;

        let mut push = Vec::with_capacity(conv::PUSH_BYTES as usize);
        for v in [
            (c_out * pixels) as u32,
            c_in as u32,
            c_out as u32,
            1,
            h_in as u32,
            w_in as u32,
            h_out as u32,
            w_out as u32,
            kh as u32,
            kw as u32,
            stride as u32,
            stride as u32,
            pad as u32,
            pad as u32,
            1,
            1,
            c_in as u32,
            1,
            1,
        ] {
            push.extend_from_slice(&v.to_le_bytes());
        }

        let run = |pipe: &vk_compute::ComputePipeline,
                   out: &vk_compute::GpuBuffer,
                   tile: u32,
                   reps: u32|
         -> Result<f64, Box<dyn std::error::Error>> {
            let t = Instant::now();
            for _ in 0..reps {
                ctx.stream_dispatch(
                    pipe,
                    &[&x_buf, &w_buf, &b_buf, out],
                    &push,
                    [
                        (pixels as u32).div_ceil(tile),
                        (c_out as u32).div_ceil(tile),
                        1,
                    ],
                )?;
            }
            ctx.flush()?;
            Ok(t.elapsed().as_secs_f64() / reps as f64)
        };

        run(small, &out_a, conv::TILE_SIZE, 2)?;
        run(blocked, &out_b, 64, 2)?;
        let ta = (0..3).try_fold(f64::MAX, |acc, _| {
            run(small, &out_a, conv::TILE_SIZE, 8).map(|s| acc.min(s))
        })?;
        let tb = (0..3).try_fold(f64::MAX, |acc, _| {
            run(blocked, &out_b, 64, 8).map(|s| acc.min(s))
        })?;

        let nbytes = 4 * c_out * pixels;
        let fa = floats(&ctx.stream_download(&out_a, nbytes)?);
        let fb = floats(&ctx.stream_download(&out_b, nbytes)?);
        let mut rel = 0.0f32;
        for (a, b) in fa.iter().zip(&fb) {
            rel = rel.max((a - b).abs() / a.abs().max(1e-3));
        }
        worst_rel = worst_rel.max(rel);

        let wg = workgroups(pixels, c_out);
        let flops = 2.0 * (c_out * kdepth * pixels) as f64;
        println!(
            "{:>24} {pixels:>7} {kdepth:>6} {wg:>5} {:>9.3} {:>7.2} {:>9.3} {:>7.2} {:>7.2}× {rel:>9.1e}",
            format!("{c_in}→{c_out} {kh}x{kw} @{h_out}x{w_out}"),
            ta * 1e3,
            flops / ta / 1e12,
            tb * 1e3,
            flops / tb / 1e12,
            ta / tb,
        );
        let n = count as f64;
        tot_small += ta * 1e3 * n;
        tot_blocked += tb * 1e3 * n;
        tot_routed += if wg >= WG_FLOOR { tb } else { ta } * 1e3 * n;
        tot_oracle += ta.min(tb) * 1e3 * n;
    }

    Ok((tot_small, tot_blocked, tot_routed, tot_oracle, worst_rel))
}
