//! Why is resnet50-qdq still at 0.74× of the CPU EP? Attribution, not a fix.
//!
//! The model is at 1 convex block with 0.26 ms of sync, and the roofline on its
//! 53 `Conv` nodes (7.71 GFLOP, 176.7 MB) puts the floor at 0.35 ms against
//! 6.47 ms measured — 18× above it, on neither compute nor bandwidth. What is
//! left is the two things a roofline cannot see: **how much of the machine each
//! dispatch fills**, and **what the stream costs per dispatch** now that every
//! one of them is preceded by a global barrier.
//!
//! Three experiments, in the order the questions were asked:
//!
//! 1. [`launch_floor`] — a null kernel, one workgroup, thousands of times
//!    through `stream_dispatch`. Splits the cost into CPU recording and GPU
//!    execution and gives the per-dispatch floor the model pays 348 times.
//! 2. [`barrier_cost`] — every resnet50 `Conv` geometry run two ways with
//!    identical total work: `R` separate dispatches at batch 1 (what the model
//!    does, barrier between each) versus **one** dispatch at batch `R` (same
//!    arithmetic, same buffers, one grid, no barriers). The gap is what
//!    serialization plus launch costs on real work.
//! 3. [`occupancy_curve`] — the same batch knob swept, `t(z)/z` against
//!    `t(1)`. If a single dispatch's grid cannot fill the machine, this curve
//!    falls; where it flattens is the grid size at which the GPU is saturated.
//!
//! Run: `cargo run --release -p onnx-vulkan-core --example conv_occupancy`

use onnx_vulkan_core::shaders::conv;
use std::time::Instant;
use vk_compute::{ComputePipeline, GpuBuffer, VkContext, compile_wgsl};

/// One `Conv` geometry and how many nodes of resnet50-qdq run it. Same table as
/// `examples/conv_blocked`, restricted to the model under investigation.
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

/// Batch multiplier for the collapse experiment. Every `z` slice is an
/// independent image, so `R` dispatches at batch 1 and one dispatch at batch
/// `R` perform exactly the same arithmetic over exactly the same buffers.
const R: usize = 8;

/// A kernel that does nothing, to price the stream itself.
const NULL_KERNEL: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
struct Push { n: u32 }
var<immediate> pc: Push;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= pc.n) { out[gid.x] = 1.0; }
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

type Err = Box<dyn std::error::Error>;

fn main() -> Result<(), Err> {
    let ctx = VkContext::new()?;
    let small = ctx.create_pipeline(
        &compile_wgsl(&conv::implicit_gemm())?,
        conv::BINDINGS,
        conv::PUSH_BYTES,
    )?;
    let blocked = ctx.create_pipeline(
        &compile_wgsl(&conv::blocked())?,
        conv::BINDINGS,
        conv::PUSH_BYTES,
    )?;

    launch_floor(&ctx)?;
    barrier_cost(&ctx, &small, &blocked)?;
    occupancy_curve(&ctx, &small, &blocked)?;
    what_binds_the_16_tile(&ctx, &small)?;
    Ok(())
}

/// Which resource binds the 16×16 kernel on the geometries the predicate keeps
/// there — shared-memory bandwidth, or barriers and exposed latency?
///
/// Those six geometries are half of resnet50's `Conv` time and they are the ones
/// [`occupancy_curve`] shows *not* to be grid-starved: batching them 16× buys
/// only 1.5–2.1×, and they still sit at 0.8–0.95 TF/s with the machine full. So
/// the limit is inside the K loop. Two variants, each changing exactly one term:
///
/// - [`WIDE_TILE`] keeps the 16-pixel tile but computes **4 output channels per
///   thread**, so the inner loop spends 5 shared reads per 4 FMAs instead of 2
///   per 1 — LDS traffic per FMA drops 1.6×, everything else is unchanged.
/// - [`DEEP_STEP`] keeps one output per thread and the same 2 reads per FMA, but
///   walks `K` in steps of 32, halving the number of `workgroupBarrier`s and of
///   staging rounds.
///
/// If the win is on `WIDE_TILE` the kernel is shared-bandwidth bound; if it is
/// on `DEEP_STEP` it is paying for synchronization and staging latency. Both are
/// probes, deliberately not wired into the routing.
const WIDE_TILE: &str = r#"
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

const PT = 16u;   // output pixels per workgroup
const MT = 64u;   // output channels per workgroup
const KSTEP = 16u;
var<workgroup> w_tile: array<f32, 1024>;   // MT × KSTEP
var<workgroup> x_tile: array<f32, 256>;    // KSTEP × PT

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.y * 16u + lid.x;
    let col0 = wid.x * PT;
    let row0 = wid.y * MT;
    let bn = wid.z;
    let pixels = pc.h_out * pc.w_out;
    let kdepth = pc.c_in * pc.kh * pc.kw;
    let ksize = pc.kh * pc.kw;

    var acc = vec4<f32>(0.0);
    let ntiles = (kdepth + KSTEP - 1u) / KSTEP;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        let k0 = t * KSTEP;
        for (var s = 0u; s < 4u; s = s + 1u) {
            let l = tid + s * 256u;
            let gr = row0 + l / KSTEP;
            let gk = k0 + l % KSTEP;
            var v = 0.0;
            if (gr < pc.c_out && gk < kdepth) { v = w[gr * kdepth + gk]; }
            w_tile[l] = v;
        }
        {
            let gk = k0 + lid.y;
            let gc = col0 + lid.x;
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
                    v = x[((bn * pc.c_in + ic) * pc.h_in + u32(ih)) * pc.w_in + u32(iw)];
                }
            }
            x_tile[lid.y * PT + lid.x] = v;
        }
        workgroupBarrier();
        let arow = lid.y * 4u;
        for (var kk = 0u; kk < KSTEP; kk = kk + 1u) {
            let xv = x_tile[kk * PT + lid.x];
            acc = fma(
                vec4<f32>(
                    w_tile[(arow + 0u) * KSTEP + kk],
                    w_tile[(arow + 1u) * KSTEP + kk],
                    w_tile[(arow + 2u) * KSTEP + kk],
                    w_tile[(arow + 3u) * KSTEP + kk],
                ),
                vec4<f32>(xv), acc);
        }
        workgroupBarrier();
    }

    let p = col0 + lid.x;
    if (p >= pixels) { return; }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = row0 + lid.y * 4u + i;
        if (m >= pc.c_out) { continue; }
        var v = acc[i];
        if (pc.has_bias != 0u) { v = v + bias[m]; }
        out[(bn * pc.c_out + m) * pixels + p] = v;
    }
}
"#;

/// The 16×16 kernel with `KSTEP = 32`: same one output per thread and same two
/// shared reads per FMA, half the barriers.
const DEEP_STEP: &str = r#"
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
const KSTEP = 32u;
var<workgroup> w_tile: array<f32, 512>;   // TILE × KSTEP
var<workgroup> x_tile: array<f32, 512>;   // KSTEP × TILE

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m = wid.y * TILE + lid.y;
    let p = wid.x * TILE + lid.x;
    let bn = wid.z;
    let pixels = pc.h_out * pc.w_out;
    let kdepth = pc.c_in * pc.kh * pc.kw;
    let ksize = pc.kh * pc.kw;

    var acc = 0.0;
    let ntiles = (kdepth + KSTEP - 1u) / KSTEP;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        let k0 = t * KSTEP;
        for (var s = 0u; s < 2u; s = s + 1u) {
            let kk = lid.x + s * 16u;
            var v = 0.0;
            if (m < pc.c_out && k0 + kk < kdepth) { v = w[m * kdepth + k0 + kk]; }
            w_tile[lid.y * KSTEP + kk] = v;

            let gk = k0 + lid.y + s * 16u;
            var xv = 0.0;
            if (p < pixels && gk < kdepth) {
                let ic = gk / ksize;
                let rem = gk % ksize;
                let r = rem / pc.kw;
                let sx = rem % pc.kw;
                let oh = p / pc.w_out;
                let ow = p % pc.w_out;
                let ih = i32(oh) * i32(pc.sh) - i32(pc.phb) + i32(r) * i32(pc.dh);
                let iw = i32(ow) * i32(pc.sw) - i32(pc.pwb) + i32(sx) * i32(pc.dw);
                if (ih >= 0 && ih < i32(pc.h_in) && iw >= 0 && iw < i32(pc.w_in)) {
                    xv = x[((bn * pc.c_in + ic) * pc.h_in + u32(ih)) * pc.w_in + u32(iw)];
                }
            }
            x_tile[(lid.y + s * 16u) * TILE + lid.x] = xv;
        }
        workgroupBarrier();
        for (var i = 0u; i < KSTEP; i = i + 1u) {
            acc = acc + w_tile[lid.y * KSTEP + i] * x_tile[i * TILE + lid.x];
        }
        workgroupBarrier();
    }
    if (m >= pc.c_out || p >= pixels) { return; }
    if (pc.has_bias != 0u) { acc = acc + bias[m]; }
    out[(bn * pc.c_out + m) * pixels + p] = acc;
}
"#;

fn what_binds_the_16_tile(ctx: &VkContext, small: &ComputePipeline) -> Result<(), Err> {
    println!("\n===== 4. what binds the 16x16 kernel on the shapes it keeps =====");
    let wide = ctx.create_pipeline(&compile_wgsl(WIDE_TILE)?, conv::BINDINGS, conv::PUSH_BYTES)?;
    let deep = ctx.create_pipeline(&compile_wgsl(DEEP_STEP)?, conv::BINDINGS, conv::PUSH_BYTES)?;
    println!(
        "  z=1 is what the model runs; z=16 is the same kernels with the machine full,\n  \
         which separates a grid too small from a loop too slow.\n"
    );
    println!(
        "{:>22} {:>6} {:>4} {:>9} {:>9} {:>8} {:>9} {:>8} {:>9}",
        "geometry", "K", "z", "16x16 us", "wide us", "wide×", "deep us", "deep×", "TF/s best"
    );
    let (mut base, mut w_tot, mut d_tot) = (0.0f64, 0.0f64, 0.0f64);
    for s in RESNET50 {
        let g = Geom::new(ctx, s)?;
        if conv::prefer_blocked(g.pixels, s.c_out) {
            continue; // the blocked kernel already owns these
        }
        let kdepth = s.c_in * s.k * s.k;
        // wide tile: 16 pixels per workgroup in x, 64 channels in y
        let wide_grid = [
            (g.pixels as u32).div_ceil(16),
            (s.c_out as u32).div_ceil(64),
            1,
        ];
        let square_grid = [
            (g.pixels as u32).div_ceil(conv::TILE_SIZE),
            (s.c_out as u32).div_ceil(conv::TILE_SIZE),
            1,
        ];
        // a probe that computes something else measures nothing: check both
        // variants against the kernel in production before timing them
        let reference = g.run_and_read(ctx, small, square_grid)?;
        for (name, pipe, grid) in [("wide", &wide, wide_grid), ("deep", &deep, square_grid)] {
            let got = g.run_and_read(ctx, pipe, grid)?;
            let rel = reference
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs() / a.abs().max(1e-3))
                .fold(0.0f32, f32::max);
            assert!(
                rel < 1e-4,
                "{name} diverges on {kdepth}: max relative {rel:.2e}"
            );
        }
        for z in [1u32, 16] {
            let zf = z as f64;
            let t16 = g.time(ctx, small, conv::TILE_SIZE, z, 1)? / zf;
            let tw = g.time_grid(ctx, &wide, [wide_grid[0], wide_grid[1], z], 1)? / zf;
            let td = g.time(ctx, &deep, conv::TILE_SIZE, z, 1)? / zf;
            println!(
                "{:>22} {kdepth:>6} {z:>4} {:>9.1} {:>9.1} {:>7.2}× {:>9.1} {:>7.2}× {:>9.2}",
                if z == 1 {
                    format!("{}->{} {}x{} @{}", s.c_in, s.c_out, s.k, s.k, s.h_out)
                } else {
                    String::new()
                },
                t16 * 1e6,
                tw * 1e6,
                t16 / tw,
                td * 1e6,
                t16 / td,
                g.flops / t16.min(tw).min(td) / 1e12,
            );
            if z == 1 {
                base += t16 * s.count as f64;
                w_tot += tw * s.count as f64;
                d_tot += td * s.count as f64;
            }
        }
    }
    println!(
        "\n  the {:.3} ms these nodes cost today would be {:.3} ms wide ({:.2}×) \
         or {:.3} ms deep ({:.2}×)",
        base * 1e3,
        w_tot * 1e3,
        base / w_tot,
        d_tot * 1e3,
        base / d_tot,
    );
    Ok(())
}

/// What one `stream_dispatch` costs when the kernel costs nothing.
///
/// resnet50-qdq records 348 of them into a single command buffer, each preceded
/// by a global `COMPUTE → COMPUTE` barrier, so this is a floor the model pays
/// whatever its kernels do. CPU recording and GPU execution are separated
/// because only the second shows up in the profiler's Pareto — the first is
/// invisible to it and lands in the wall clock.
fn launch_floor(ctx: &VkContext) -> Result<(), Err> {
    println!("===== 1. per-dispatch floor (null kernel, 1 workgroup) =====");
    let pipe = ctx.create_pipeline(&compile_wgsl(NULL_KERNEL)?, 1, 4)?;
    let out = ctx.create_storage_buffer(256)?;
    let push = 0u32.to_le_bytes();

    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12}",
        "N", "record ms", "flush ms", "rec us/disp", "gpu us/disp"
    );
    for n in [1usize, 64, 256, 1024, 4096] {
        // warm-up: first submission pays pipeline/descriptor set-up
        for _ in 0..n.min(64) {
            ctx.stream_dispatch(&pipe, &[&out], &push, [1, 1, 1])?;
        }
        ctx.flush()?;

        let (mut rec, mut fl) = (f64::MAX, f64::MAX);
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..n {
                ctx.stream_dispatch(&pipe, &[&out], &push, [1, 1, 1])?;
            }
            let r = t.elapsed().as_secs_f64();
            let t = Instant::now();
            ctx.flush()?;
            fl = fl.min(t.elapsed().as_secs_f64());
            rec = rec.min(r);
        }
        println!(
            "{n:>8} {:>12.3} {:>12.3} {:>12.2} {:>12.2}",
            rec * 1e3,
            fl * 1e3,
            rec / n as f64 * 1e6,
            fl / n as f64 * 1e6,
        );
    }
    Ok(())
}

/// The same work as `R` barrier-separated dispatches, then as one dispatch.
///
/// Batch is the only thing that changes: the `z` grid dimension is the image
/// index, so `R` launches at `z = 1` and one launch at `z = R` read the same
/// weights, touch the same number of input and output bytes, and issue the same
/// FMAs. What differs is that the second has one grid `R` times wider and no
/// barrier inside it. The gap is serialization plus launch, priced on real
/// kernels rather than on a null one.
fn barrier_cost(
    ctx: &VkContext,
    small: &ComputePipeline,
    blocked: &ComputePipeline,
) -> Result<(), Err> {
    println!("\n===== 2. {R} dispatches at batch 1 vs 1 dispatch at batch {R} =====");
    println!(
        "{:>22} {:>5} {:>7} {:>10} {:>10} {:>8} {:>9} {:>9}",
        "geometry", "kern", "wg/disp", "split us", "fused us", "ratio", "TF/s spl", "TF/s fus"
    );
    let (mut tot_split, mut tot_fused) = (0.0f64, 0.0f64);
    for s in RESNET50 {
        let g = Geom::new(ctx, s)?;
        let use_blocked = conv::prefer_blocked(g.pixels, s.c_out);
        let (pipe, tile) = if use_blocked {
            (blocked, conv::BLOCKED_TILE_SIZE)
        } else {
            (small, conv::TILE_SIZE)
        };
        let split = g.time(ctx, pipe, tile, 1, R)?;
        let fused = g.time(ctx, pipe, tile, R as u32, 1)? / R as f64;
        let wg = (g.pixels as u32).div_ceil(tile) * (s.c_out as u32).div_ceil(tile);
        println!(
            "{:>22} {:>5} {:>7} {:>10.1} {:>10.1} {:>7.2}× {:>9.2} {:>9.2}",
            format!("{}->{} {}x{} @{}", s.c_in, s.c_out, s.k, s.k, s.h_out),
            if use_blocked { "64" } else { "16" },
            wg,
            split * 1e6,
            fused * 1e6,
            split / fused,
            g.flops / split / 1e12,
            g.flops / fused / 1e12,
        );
        tot_split += split * s.count as f64;
        tot_fused += fused * s.count as f64;
    }
    println!(
        "\n  53 Conv: split {:.3} ms | fused {:.3} ms ({:.2}× ) — {:.3} ms is grid width + barriers",
        tot_split * 1e3,
        tot_fused * 1e3,
        tot_split / tot_fused,
        (tot_split - tot_fused) * 1e3,
    );
    Ok(())
}

/// `t(z)/z` as the grid grows: where it stops falling, the GPU is full.
///
/// Only a spread of geometries, so the table stays readable: the two extremes
/// of workgroup count and two in between.
fn occupancy_curve(
    ctx: &VkContext,
    small: &ComputePipeline,
    blocked: &ComputePipeline,
) -> Result<(), Err> {
    println!("\n===== 3. occupancy curve: per-image us at batch z =====");
    print!("{:>22} {:>5} {:>7}", "geometry", "kern", "wg z=1");
    const ZS: &[u32] = &[1, 2, 4, 8, 16];
    for z in ZS {
        print!(" {:>9}", format!("z={z}"));
    }
    println!(" {:>8}", "z16/z1");
    for s in RESNET50 {
        let g = Geom::new(ctx, s)?;
        let use_blocked = conv::prefer_blocked(g.pixels, s.c_out);
        let (pipe, tile) = if use_blocked {
            (blocked, conv::BLOCKED_TILE_SIZE)
        } else {
            (small, conv::TILE_SIZE)
        };
        let wg = (g.pixels as u32).div_ceil(tile) * (s.c_out as u32).div_ceil(tile);
        print!(
            "{:>22} {:>5} {:>7}",
            format!("{}->{} {}x{} @{}", s.c_in, s.c_out, s.k, s.k, s.h_out),
            if use_blocked { "64" } else { "16" },
            wg,
        );
        let mut first = 0.0;
        let mut last = 0.0;
        for (i, z) in ZS.iter().enumerate() {
            let t = g.time(ctx, pipe, tile, *z, 1)? / *z as f64;
            if i == 0 {
                first = t;
            }
            last = t;
            print!(" {:>9.1}", t * 1e6);
        }
        println!(" {:>7.2}×", first / last);
    }
    Ok(())
}

/// Buffers and push constants for one geometry, sized for the largest batch the
/// sweep uses so every `z` reuses the same allocation.
struct Geom {
    x: GpuBuffer,
    w: GpuBuffer,
    b: GpuBuffer,
    out: GpuBuffer,
    push: Vec<u8>,
    pixels: usize,
    c_out: usize,
    flops: f64,
}

/// Largest batch any experiment here dispatches.
const ZMAX: usize = 16;

impl Geom {
    fn new(ctx: &VkContext, s: &Shape) -> Result<Self, Err> {
        let (h_in, w_in) = (s.h_in, s.h_in);
        let (h_out, w_out) = (s.h_out, s.h_out);
        let pixels = h_out * w_out;
        let kdepth = s.c_in * s.k * s.k;
        let x = pseudo(s.c_in * h_in * w_in * ZMAX, 3);
        let w = pseudo(s.c_out * kdepth, 5);
        let b = pseudo(s.c_out, 7);

        let xb = ctx.create_storage_buffer((4 * x.len()) as u64)?;
        let wb = ctx.create_storage_buffer((4 * w.len()) as u64)?;
        let bb = ctx.create_storage_buffer((4 * b.len()) as u64)?;
        let out = ctx.create_storage_buffer((4 * s.c_out * pixels * ZMAX) as u64)?;
        ctx.stream_upload(&xb, &bytes(&x))?;
        ctx.stream_upload(&wb, &bytes(&w))?;
        ctx.stream_upload(&bb, &bytes(&b))?;
        ctx.flush()?;

        let mut push = Vec::with_capacity(conv::PUSH_BYTES as usize);
        for v in [
            (s.c_out * pixels) as u32,
            s.c_in as u32,
            s.c_out as u32,
            1,
            h_in as u32,
            w_in as u32,
            h_out as u32,
            w_out as u32,
            s.k as u32,
            s.k as u32,
            s.stride as u32,
            s.stride as u32,
            s.pad as u32,
            s.pad as u32,
            1,
            1,
            s.c_in as u32,
            1,
            1,
        ] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        Ok(Self {
            x: xb,
            w: wb,
            b: bb,
            out,
            push,
            pixels,
            c_out: s.c_out,
            flops: 2.0 * (s.c_out * kdepth * pixels) as f64,
        })
    }

    /// Best of five timings of `reps` dispatches at batch `z`, returned per
    /// dispatch. Measures the flush alone, so CPU recording does not enter the
    /// number — that cost is what [`launch_floor`] reports separately.
    fn time(
        &self,
        ctx: &VkContext,
        pipe: &ComputePipeline,
        tile: u32,
        z: u32,
        reps: usize,
    ) -> Result<f64, Err> {
        let groups = [
            (self.pixels as u32).div_ceil(tile),
            (self.c_out as u32).div_ceil(tile),
            z,
        ];
        self.time_grid(ctx, pipe, groups, reps)
    }

    /// One dispatch at batch 1, then the whole output image back on the host.
    fn run_and_read(
        &self,
        ctx: &VkContext,
        pipe: &ComputePipeline,
        groups: [u32; 3],
    ) -> Result<Vec<f32>, Err> {
        ctx.stream_dispatch(
            pipe,
            &[&self.x, &self.w, &self.b, &self.out],
            &self.push,
            groups,
        )?;
        let raw = ctx.stream_download(&self.out, 4 * self.c_out * self.pixels)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// [`Self::time`] with the grid given explicitly, for the probe kernels
    /// whose workgroup covers a rectangle rather than a square.
    fn time_grid(
        &self,
        ctx: &VkContext,
        pipe: &ComputePipeline,
        groups: [u32; 3],
        reps: usize,
    ) -> Result<f64, Err> {
        let mut best = f64::MAX;
        for i in 0..6 {
            for _ in 0..reps {
                ctx.stream_dispatch(
                    pipe,
                    &[&self.x, &self.w, &self.b, &self.out],
                    &self.push,
                    groups,
                )?;
            }
            let t = Instant::now();
            ctx.flush()?;
            if i > 0 {
                best = best.min(t.elapsed().as_secs_f64() / reps as f64);
            }
        }
        Ok(best)
    }
}
