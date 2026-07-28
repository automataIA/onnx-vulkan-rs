//! Does split-K fix the 18 `Conv` nodes of resnet50-qdq that cannot fill a 4070?
//!
//! `docs/resnet50-gap.md` attributes the model's 0.70× to the size of its
//! output tensors: 18 of 53 `Conv` nodes do not fill the machine even at one
//! output per thread, and §3 measured that no tile change reaches them at batch
//! 1 — every enlargement buys arithmetic intensity by giving back occupancy, at
//! par. It named **split-K** the one transformation left, and left it untested.
//! This is that test.
//!
//! These shapes are the mirror image of the GEMV case already solved in
//! `matmul_fp32::gemv_split`: `K = C_in·KH·KW` is 512–4608 while the output
//! `C_out × P` is as small as 25,088, so the reduction axis holds the
//! parallelism the output axes lack. The kernel is [`conv::implicit_gemm`] with
//! its `K` loop sliced across `wid.z`, writing one partial image per slice,
//! plus a reduction pass that also applies the bias.
//!
//! Run: `cargo run --release -p onnx-vulkan-core --example conv_splitk`
//! (cross-build and run on the Windows host — lavapipe is meaningless here).

use onnx_vulkan_core::shaders::conv;
use std::time::Instant;
use vk_compute::{ComputePipeline, GpuBuffer, VkContext, compile_wgsl};

/// One `Conv` geometry and how many nodes of resnet50-qdq run it.
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

/// Same table as `examples/conv_occupancy`. Every geometry is listed, and the
/// run reports only those `conv::prefer_blocked` keeps on the 16×16 tile — the
/// starved population — since the blocked ones already have a grid.
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

const SPLITS: &[u32] = &[2, 4, 8, 16, 32];

/// [`conv::implicit_gemm`] with the `K` loop sliced across `wid.z`.
///
/// Each slice covers `ceil(ntiles / split)` of the 16-wide steps over
/// `K = C_in·KH·KW` and writes its own image of partials, so the grid is
/// `split` times wider for the same arithmetic. The bias is deliberately not
/// applied here — adding it in every slice would multiply it by `split`; it
/// belongs to the reduction, which is the only pass that sees a whole sum.
const SPLITK: &str = r#"
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

const TILE = 16u;
var<workgroup> w_tile: array<f32, 256>;
var<workgroup> x_tile: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let ty = lid.y;
    let tx = lid.x;
    let m = wid.y * TILE + ty;           // output channel = row
    let p = wid.x * TILE + tx;           // output pixel = column
    let pixels = pc.h_out * pc.w_out;
    let kdepth = pc.c_in * pc.kh * pc.kw;

    // this slice's range of 16-wide steps over K
    let ntiles = (kdepth + TILE - 1u) / TILE;
    let tper = (ntiles + pc.split - 1u) / pc.split;
    let tstart = wid.z * tper;
    var tend = tstart + tper;
    if (tend > ntiles) { tend = ntiles; }

    var acc = 0.0;
    for (var t = tstart; t < tend; t = t + 1u) {
        let kw_idx = t * TILE + tx;
        if (m < pc.c_out && kw_idx < kdepth) {
            w_tile[ty * TILE + tx] = w[m * kdepth + kw_idx];
        } else {
            w_tile[ty * TILE + tx] = 0.0;
        }
        let kx_idx = t * TILE + ty;
        var value = 0.0;
        if (p < pixels && kx_idx < kdepth) {
            let ksize = pc.kh * pc.kw;
            let ic = kx_idx / ksize;
            let rem = kx_idx % ksize;
            let r = rem / pc.kw;
            let s = rem % pc.kw;
            let oh = p / pc.w_out;
            let ow = p % pc.w_out;
            let ih = i32(oh) * i32(pc.sh) - i32(pc.phb) + i32(r) * i32(pc.dh);
            let iw = i32(ow) * i32(pc.sw) - i32(pc.pwb) + i32(s) * i32(pc.dw);
            // out of bounds = zero: this is the conv's implicit padding
            if (ih >= 0 && ih < i32(pc.h_in) && iw >= 0 && iw < i32(pc.w_in)) {
                value = x[(ic * pc.h_in + u32(ih)) * pc.w_in + u32(iw)];
            }
        }
        x_tile[ty * TILE + tx] = value;
        workgroupBarrier();
        for (var i = 0u; i < TILE; i = i + 1u) {
            acc = acc + w_tile[ty * TILE + i] * x_tile[i * TILE + tx];
        }
        workgroupBarrier();
    }
    if (m >= pc.c_out || p >= pixels) { return; }
    out[wid.z * pc.total + m * pixels + p] = acc;
}
"#;

/// [`conv::blocked`] with the same `K` slicing, and the point of this probe.
///
/// §3 of `docs/resnet50-gap.md` measured that a wider tile is worth ~1.9× on
/// these shapes **once the machine is full**, and unreachable at batch 1
/// because a 64×64 tile divides the grid by 16. Split-K is what pays that back:
/// 4 workgroups × 8 slices is 32 where the tile alone left 4. This is the one
/// combination the attribution predicted and did not test.
const BLOCKED_SPLITK: &str = r#"
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
    let row0 = wid.y * TILE;
    let col0 = wid.x * TILE;
    let pixels = pc.h_out * pc.w_out;
    let kdepth = pc.c_in * pc.kh * pc.kw;
    let ksize = pc.kh * pc.kw;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let ntiles = (kdepth + KSTEP - 1u) / KSTEP;
    let tper = (ntiles + pc.split - 1u) / pc.split;
    let tstart = wid.z * tper;
    var tend = tstart + tper;
    if (tend > ntiles) { tend = ntiles; }

    for (var t = tstart; t < tend; t = t + 1u) {
        let k0 = t * KSTEP;
        for (var s = 0u; s < 4u; s = s + 1u) {
            let l = tid + s * 256u;
            let gr = row0 + l / KSTEP;
            let gk = k0 + l % KSTEP;
            var v = 0.0;
            if (gr < pc.c_out && gk < kdepth) { v = w[gr * kdepth + gk]; }
            w_tile[l] = v;
        }
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
                if (ih >= 0 && ih < i32(pc.h_in) && iw >= 0 && iw < i32(pc.w_in)) {
                    v = x[(ic * pc.h_in + u32(ih)) * pc.w_in + u32(iw)];
                }
            }
            x_tile[l] = v;
        }
        workgroupBarrier();
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
            out[wid.z * pc.total + m * pixels + p] = accv[j];
        }
    }
}
"#;

/// Sums the `split` partial images and applies the bias.
const REDUCE: &str = r#"
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
struct Push {
    total: u32, c_in: u32, c_out: u32, group: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    phb: u32, pwb: u32, dh: u32, dw: u32, gsi: u32, has_bias: u32,
    split: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.total) { return; }
    var acc = 0.0;
    for (var s = 0u; s < pc.split; s = s + 1u) {
        acc = acc + partials[s * pc.total + o];
    }
    if (pc.has_bias != 0u) { acc = acc + bias[o / (pc.h_out * pc.w_out)]; }
    out[o] = acc;
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

type Err = Box<dyn std::error::Error>;

/// Largest split any run dispatches, and so how many partial images the
/// scratch buffer has to hold.
const SPLIT_MAX: usize = 32;

struct Geom {
    x: GpuBuffer,
    w: GpuBuffer,
    b: GpuBuffer,
    out: GpuBuffer,
    partials: GpuBuffer,
    push: Vec<u8>,
    pixels: usize,
    c_out: usize,
    total: usize,
}

impl Geom {
    fn new(ctx: &VkContext, s: &Shape) -> Result<Self, Err> {
        let (h_in, w_in) = (s.h_in, s.h_in);
        let (h_out, w_out) = (s.h_out, s.h_out);
        let pixels = h_out * w_out;
        let kdepth = s.c_in * s.k * s.k;
        let total = s.c_out * pixels;

        let xb = ctx.create_storage_buffer((4 * s.c_in * h_in * w_in) as u64)?;
        let wb = ctx.create_storage_buffer((4 * s.c_out * kdepth) as u64)?;
        let bb = ctx.create_storage_buffer((4 * s.c_out) as u64)?;
        let out = ctx.create_storage_buffer((4 * total) as u64)?;
        let partials = ctx.create_storage_buffer((4 * total * SPLIT_MAX) as u64)?;
        ctx.stream_upload(&xb, &bytes(&pseudo(s.c_in * h_in * w_in, 3)))?;
        ctx.stream_upload(&wb, &bytes(&pseudo(s.c_out * kdepth, 5)))?;
        ctx.stream_upload(&bb, &bytes(&pseudo(s.c_out, 7)))?;
        ctx.flush()?;

        let mut push = Vec::new();
        #[rustfmt::skip]
        let fields = [
            total as u32, s.c_in as u32, s.c_out as u32, 1,
            h_in as u32, w_in as u32, h_out as u32, w_out as u32,
            s.k as u32, s.k as u32, s.stride as u32, s.stride as u32,
            s.pad as u32, s.pad as u32, 1, 1, s.c_in as u32, 1,
            1, // split, overwritten per run
        ];
        for v in fields {
            push.extend_from_slice(&v.to_le_bytes());
        }
        Ok(Self {
            x: xb,
            w: wb,
            b: bb,
            out,
            partials,
            push,
            pixels,
            c_out: s.c_out,
            total,
        })
    }

    fn with_split(&self, split: u32) -> Vec<u8> {
        let mut push = self.push.clone();
        let at = push.len() - 4;
        push[at..].copy_from_slice(&split.to_le_bytes());
        push
    }

    /// Best of five timings of the production kernel, one dispatch at batch 1.
    fn time_baseline(
        &self,
        ctx: &VkContext,
        pipe: &ComputePipeline,
        tile: u32,
    ) -> Result<f64, Err> {
        let grid = [
            (self.pixels as u32).div_ceil(tile),
            (self.c_out as u32).div_ceil(tile),
            1,
        ];
        // Its pipeline declares `conv::PUSH_BYTES`, so it gets the push
        // constants without the probe's trailing `split` field.
        let push = &self.push[..conv::PUSH_BYTES as usize];
        self.best(ctx, |_| {
            ctx.stream_dispatch(pipe, &[&self.x, &self.w, &self.b, &self.out], push, grid)?;
            Ok(())
        })
    }

    /// Best of five timings of the split-K pair, reduction included.
    fn time_splitk(
        &self,
        ctx: &VkContext,
        conv: &ComputePipeline,
        reduce: &ComputePipeline,
        tile: u32,
        split: u32,
    ) -> Result<f64, Err> {
        let push = self.with_split(split);
        let grid = [
            (self.pixels as u32).div_ceil(tile),
            (self.c_out as u32).div_ceil(tile),
            split,
        ];
        self.best(ctx, |_| {
            ctx.stream_dispatch(
                conv,
                &[&self.x, &self.w, &self.b, &self.partials],
                &push,
                grid,
            )?;
            ctx.stream_dispatch(
                reduce,
                &[&self.partials, &self.b, &self.out],
                &push,
                [(self.total as u32).div_ceil(256), 1, 1],
            )?;
            Ok(())
        })
    }

    /// Times the flush alone, so CPU recording stays out of the number.
    fn best(
        &self,
        ctx: &VkContext,
        mut enqueue: impl FnMut(usize) -> Result<(), Err>,
    ) -> Result<f64, Err> {
        const REPS: usize = 8;
        let mut best = f64::MAX;
        for i in 0..6 {
            for r in 0..REPS {
                enqueue(r)?;
            }
            let t = Instant::now();
            ctx.flush()?;
            if i > 0 {
                best = best.min(t.elapsed().as_secs_f64() / REPS as f64);
            }
        }
        Ok(best)
    }

    fn read_out(&self, ctx: &VkContext) -> Result<Vec<f32>, Err> {
        Ok(floats(&ctx.stream_download(&self.out, 4 * self.total)?))
    }
}

fn main() -> Result<(), Err> {
    let ctx = VkContext::new()?;
    let small = ctx.create_pipeline(
        &compile_wgsl(&conv::implicit_gemm())?,
        conv::BINDINGS,
        conv::PUSH_BYTES,
    )?;
    let push_bytes = conv::PUSH_BYTES + 4;
    let blocked = ctx.create_pipeline(
        &compile_wgsl(&conv::blocked())?,
        conv::BINDINGS,
        conv::PUSH_BYTES,
    )?;
    let splitk = ctx.create_pipeline(&compile_wgsl(SPLITK)?, conv::BINDINGS, push_bytes)?;
    let blocked_splitk =
        ctx.create_pipeline(&compile_wgsl(BLOCKED_SPLITK)?, conv::BINDINGS, push_bytes)?;
    let reduce = ctx.create_pipeline(&compile_wgsl(REDUCE)?, 3, push_bytes)?;

    println!(
        "Split-K on the resnet50-qdq geometries the routing keeps on the 16×16 tile.\n\
         `base` is the production kernel at batch 1; each split column is the\n\
         split-K pair including its reduction pass. `*` marks the geometries\n\
         `prefer_blocked` already routes to the 64×64 tile.\n"
    );
    print!(
        "{:>22} {:>5} {:>6} {:>8}",
        "geometry", "K", "WGs", "base ms"
    );
    for s in SPLITS {
        print!(" {:>9}", format!("split{s}"));
    }
    println!(" {:>8} {:>9}", "best", "max|rel|");

    let mut base_total = 0.0;
    let mut best_total = 0.0;
    let mut fixed8_total = 0.0;

    for s in RESNET50 {
        let pixels = s.h_out * s.h_out;
        // Every geometry, not just the starved ones: a routing predicate has to
        // know where split-K stops paying, and that edge is in the population
        // `prefer_blocked` already sends to the 64×64 tile.
        let routed_blocked = conv::prefer_blocked(pixels, s.c_out);
        let g = Geom::new(&ctx, s)?;
        let kdepth = s.c_in * s.k * s.k;
        let (base_pipe, base_tile) = if routed_blocked {
            (&blocked, conv::BLOCKED_TILE_SIZE)
        } else {
            (&small, conv::TILE_SIZE)
        };
        let wgs = (pixels as u32).div_ceil(base_tile) * (s.c_out as u32).div_ceil(base_tile);
        let base = g.time_baseline(&ctx, base_pipe, base_tile)?;
        let want = g.read_out(&ctx)?;
        print!(
            "{:>22} {:>5} {:>6} {:>8.3}",
            format!(
                "{}->{} {}x{} @{}{}",
                s.c_in,
                s.c_out,
                s.k,
                s.k,
                s.h_out,
                if routed_blocked { "*" } else { "" }
            ),
            kdepth,
            wgs,
            base * 1e3
        );

        let mut best = base;
        let mut best_tile = conv::TILE_SIZE;
        let mut best_split = 1;
        let mut rel = 0.0f32;
        // two rows per geometry: the 16×16 kernel split, then the 64×64 one,
        // whose tile these shapes could not afford before the split paid for it
        for (label, pipe, tile) in [
            ("  16×16 split", &splitk, conv::TILE_SIZE),
            ("  64×64 split", &blocked_splitk, conv::BLOCKED_TILE_SIZE),
        ] {
            if label.contains("64") {
                print!("\n{label:>22} {:>5} {:>6} {:>8}", "", "", "");
            }
            for &split in SPLITS {
                let t = g.time_splitk(&ctx, pipe, &reduce, tile, split)?;
                let got = g.read_out(&ctx)?;
                rel = rel.max(
                    want.iter()
                        .zip(&got)
                        .fold(0.0f32, |m, (w, x)| m.max((w - x).abs() / w.abs().max(1e-3))),
                );
                print!(" {:>8.2}×", base / t);
                if t < best {
                    best = t;
                    best_tile = tile;
                    best_split = split;
                }
                if split == 8 && tile == conv::TILE_SIZE {
                    fixed8_total += t * s.count as f64;
                }
            }
        }
        println!(
            " {:>7.2}× {rel:>9.1e}  best: {}×{} split{}",
            base / best,
            best_tile,
            best_tile,
            best_split
        );

        base_total += base * s.count as f64;
        best_total += best * s.count as f64;
    }

    println!("\n===== weighted by the node counts resnet50-qdq runs =====");
    println!("{:>28} {:>9} {:>8}", "", "ms", "speedup");
    println!(
        "{:>28} {:>9.3} {:>8}",
        "current routing",
        base_total * 1e3,
        "1.00×"
    );
    println!(
        "{:>28} {:>9.3} {:>7.2}×",
        "16×16 split 8 everywhere",
        fixed8_total * 1e3,
        base_total / fixed8_total
    );
    println!(
        "{:>28} {:>9.3} {:>7.2}×",
        "best split per geometry",
        best_total * 1e3,
        base_total / best_total
    );
    Ok(())
}
