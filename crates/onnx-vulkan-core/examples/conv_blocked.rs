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
use std::time::Instant;
use vk_compute::{VkContext, compile_wgsl};

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
