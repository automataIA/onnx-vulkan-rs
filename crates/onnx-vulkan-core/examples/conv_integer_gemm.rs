//! Does the implicit-GEMM ladder that fixed `Conv` also reach `ConvInteger`?
//!
//! `ConvInteger` is 14.806 of resnet50-int8's 16.007 ms of GPU compute — 92.5%
//! over **23 dispatches** (`runs/int8-win`) — and it is the *direct*
//! convolution, one thread per output element walking `K` serially. The f32
//! path left that shape behind three times over: implicit GEMM, then the 64×64
//! blocked tile, then split-K. None of it exists on the integer path.
//!
//! The comparison isolates the kernel exactly. `resnet50-qdq` and
//! `resnet50-int8` are the same weights and the same 53 convolutions; the QDQ
//! variant dequantizes to f32 and runs the tuned `Conv` in 3.53 ms of GPU, the
//! QOperator one stays in int32 and runs 16.0. The epilogue is not the
//! difference — `Requantize` is 1.9% of it.
//!
//! **The 23 are not all of `ConvInteger`.** `interp::conv_integer` already
//! routes 1×1/stride-1/pad-0 convolutions to the tiled `MatMulInteger` kernel,
//! and resnet50's 30 pointwise nodes cost 0.53 ms there — 28× less than the 23
//! left on the direct kernel. So the population measured here is precisely the
//! one the fast path refuses: the 3×3s, the 7×7 stem, and the strided 1×1s
//! (which fail the `stride == 1` test but are implicit GEMMs all the same).
//!
//! `mobilenetv2-int8` is included for its boundary, not its prize: 17 of its 18
//! direct nodes are depthwise, where `K = C_in/group · KH · KW` is **9** and
//! each output channel sees one input channel — there is no GEMM to make
//! implicit and nothing to split. They total 0.209 ms, which is the ceiling on
//! that model however good this kernel gets.
//!
//! Unlike the f32 case, split-K here is **exact**: `i32` addition is
//! associative, and `K · 255 · 255 ≤ 4608 · 65025 ≈ 3.0e8` never approaches
//! `i32::MAX`, so reassociating the sum cannot move a bit. The run asserts
//! equality, not a tolerance.
//!
//! Run: `cargo run --release -p onnx-vulkan-core --example conv_integer_gemm`
//! (on the Windows host — lavapipe is meaningless here).

use onnx_vulkan_core::shaders::{conv, conv_integer};
use std::time::Instant;
use vk_compute::{ComputePipeline, GpuBuffer, VkContext, compile_wgsl};

/// One `ConvInteger` geometry and how many nodes of its model run it.
struct Shape {
    c_in: usize,
    c_out: usize,
    k: usize,
    group: usize,
    h_in: usize,
    h_out: usize,
    stride: usize,
    pad: usize,
    count: usize,
}

impl Shape {
    /// `K` of the implicit GEMM: the reduction depth of one output channel.
    fn kdepth(&self) -> usize {
        self.c_in / self.group * self.k * self.k
    }
    fn pixels(&self) -> usize {
        self.h_out * self.h_out
    }
}

/// The 23 nodes of resnet50-int8 that reach the direct kernel — every
/// `QLinearConv` the pointwise fast path refuses. Geometries from
/// `QLinearConv` + hand-propagated shapes (ONNX shape inference stops at the
/// first `com.microsoft` node); they match the f32 table in `conv_splitk`
/// node for node, since it is the same network.
#[rustfmt::skip]
const RESNET50_INT8: &[Shape] = &[
    Shape { c_in:  256, c_out:  256, k: 3, group: 1, h_in:  14, h_out:  14, stride: 1, pad: 1, count: 6 },
    Shape { c_in:  128, c_out:  128, k: 3, group: 1, h_in:  28, h_out:  28, stride: 1, pad: 1, count: 4 },
    Shape { c_in:   64, c_out:   64, k: 3, group: 1, h_in:  56, h_out:  56, stride: 1, pad: 1, count: 3 },
    Shape { c_in:  512, c_out:  512, k: 3, group: 1, h_in:   7, h_out:   7, stride: 1, pad: 1, count: 3 },
    Shape { c_in:    3, c_out:   64, k: 7, group: 1, h_in: 224, h_out: 112, stride: 2, pad: 3, count: 1 },
    Shape { c_in:  256, c_out:  512, k: 1, group: 1, h_in:  56, h_out:  28, stride: 2, pad: 0, count: 1 },
    Shape { c_in:  256, c_out:  128, k: 1, group: 1, h_in:  56, h_out:  28, stride: 2, pad: 0, count: 1 },
    Shape { c_in:  512, c_out: 1024, k: 1, group: 1, h_in:  28, h_out:  14, stride: 2, pad: 0, count: 1 },
    Shape { c_in:  512, c_out:  256, k: 1, group: 1, h_in:  28, h_out:  14, stride: 2, pad: 0, count: 1 },
    Shape { c_in: 1024, c_out: 2048, k: 1, group: 1, h_in:  14, h_out:   7, stride: 2, pad: 0, count: 1 },
    Shape { c_in: 1024, c_out:  512, k: 1, group: 1, h_in:  14, h_out:   7, stride: 2, pad: 0, count: 1 },
];

/// The 18 nodes of mobilenetv2-int8 that reach the direct kernel: 17 depthwise
/// plus the `3→32` stem. Every other convolution in the model is a pointwise
/// 1×1 already routed to `MatMulInteger`.
#[rustfmt::skip]
const MOBILENETV2_INT8: &[Shape] = &[
    Shape { c_in: 384, c_out: 384, k: 3, group: 384, h_in:  14, h_out:  14, stride: 1, pad: 1, count: 4 },
    Shape { c_in: 960, c_out: 960, k: 3, group: 960, h_in:   7, h_out:   7, stride: 1, pad: 1, count: 3 },
    Shape { c_in: 192, c_out: 192, k: 3, group: 192, h_in:  28, h_out:  28, stride: 1, pad: 1, count: 2 },
    Shape { c_in: 576, c_out: 576, k: 3, group: 576, h_in:  14, h_out:  14, stride: 1, pad: 1, count: 2 },
    Shape { c_in:  32, c_out:  32, k: 3, group:  32, h_in: 112, h_out: 112, stride: 1, pad: 1, count: 1 },
    Shape { c_in:  96, c_out:  96, k: 3, group:  96, h_in: 112, h_out:  56, stride: 2, pad: 1, count: 1 },
    Shape { c_in: 144, c_out: 144, k: 3, group: 144, h_in:  56, h_out:  56, stride: 1, pad: 1, count: 1 },
    Shape { c_in: 144, c_out: 144, k: 3, group: 144, h_in:  56, h_out:  28, stride: 2, pad: 1, count: 1 },
    Shape { c_in: 192, c_out: 192, k: 3, group: 192, h_in:  28, h_out:  14, stride: 2, pad: 1, count: 1 },
    Shape { c_in: 576, c_out: 576, k: 3, group: 576, h_in:  14, h_out:   7, stride: 2, pad: 1, count: 1 },
    Shape { c_in:   3, c_out:  32, k: 3, group:   1, h_in: 224, h_out: 112, stride: 2, pad: 1, count: 1 },
];

const SPLITS: &[u32] = &[2, 4, 8, 16, 32];
/// Largest split any run dispatches, and so how many partial images the
/// scratch buffer has to hold.
const SPLIT_MAX: usize = 32;

/// The one variant that is **not** shipped, and the reason the table has a
/// `16×16 split` row: splitting the small tile instead of the wide one. It
/// exists to show the same thing `conv::split_k` documents for f32 — that the
/// split and the wide tile are worth little apart and much together — so the
/// prelude is duplicated here rather than exported from the shader module for
/// a kernel production has no use for.
const SPLIT16: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<u32>;   // u8 packed
@group(0) @binding(1) var<storage, read> w: array<u32>;   // u8 packed
@group(0) @binding(2) var<storage, read> azp: array<u32>;
@group(0) @binding(3) var<storage, read> wzp: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<i32>;
struct Push {
    total: u32, c_in: u32, c_out: u32, group: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    phb: u32, pwb: u32, dh: u32, dw: u32, gsi: u32,
    x_signed: u32, w_signed: u32, split: u32,
}
var<immediate> pc: Push;

fn as_signed(raw: u32, signed: u32) -> i32 {
    let v = i32(raw & 0xffu);
    if (signed != 0u && v > 127) { return v - 256; }
    return v;
}
fn xat(idx: u32) -> i32 { return as_signed(x[idx >> 2u] >> ((idx & 3u) * 8u), pc.x_signed); }
fn wat(idx: u32) -> i32 { return as_signed(w[idx >> 2u] >> ((idx & 3u) * 8u), pc.w_signed); }

const TILE = 16u;
var<workgroup> w_tile: array<i32, 256>;
var<workgroup> x_tile: array<i32, 256>;

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
    let a_zp = as_signed(azp[0], pc.x_signed);
    let w_zp = as_signed(wzp[0], pc.w_signed);

    var acc = 0i;
    let ntiles = (kdepth + TILE - 1u) / TILE;
    // batch is 1 in this bench, so wid.z carries the slice alone
    let tper = (ntiles + pc.split - 1u) / pc.split;
    let tstart = wid.z * tper;
    var tend = tstart + tper;
    if (tend > ntiles) { tend = ntiles; }
    for (var t = tstart; t < tend; t = t + 1u) {
        let kw_idx = t * TILE + tx;
        var wv = 0i;
        if (m < pc.c_out && kw_idx < kdepth) { wv = wat(m * kdepth + kw_idx) - w_zp; }
        w_tile[ty * TILE + tx] = wv;
        let kx_idx = t * TILE + ty;
        var value = 0i;
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
            // out of bounds = the quantized zero, i.e. a_zp, i.e. 0 once folded
            if (ih >= 0 && ih < i32(pc.h_in) && iw >= 0 && iw < i32(pc.w_in)) {
                value = xat((ic * pc.h_in + u32(ih)) * pc.w_in + u32(iw)) - a_zp;
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

/// Deterministic bytes, so a rerun compares against the same numbers.
fn pseudo(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

fn ints(raw: &[u8]) -> Vec<i32> {
    raw.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

type Err = Box<dyn std::error::Error>;

/// The dtypes and zero points resnet50-int8 and mobilenetv2-int8 actually
/// export: `x` is `uint8` with a zero point that is nonzero on some nodes, `w`
/// is `int8` with a zero point that is zero on all 105 of them
/// (`scripts/qlinear-census.py`). A nonzero `a_zp` is the worse case for the
/// staging fold, so it is the one measured.
const X_SIGNED: u32 = 0;
const W_SIGNED: u32 = 1;
const A_ZP: u8 = 114;
const W_ZP: u8 = 0;

struct Geom {
    x: GpuBuffer,
    w: GpuBuffer,
    azp: GpuBuffer,
    wzp: GpuBuffer,
    out: GpuBuffer,
    partials: GpuBuffer,
    push: Vec<u8>,
    pixels: usize,
    c_out: usize,
    total: usize,
    reps: usize,
}

impl Geom {
    fn new(ctx: &VkContext, s: &Shape, reps: usize) -> Result<Self, Err> {
        let (h_in, w_in) = (s.h_in, s.h_in);
        let (h_out, w_out) = (s.h_out, s.h_out);
        let pixels = s.pixels();
        let total = s.c_out * pixels;
        let x_bytes = s.c_in * h_in * w_in;
        let w_bytes = s.c_out * s.kdepth();

        let xb = ctx.create_storage_buffer(x_bytes.div_ceil(4) as u64 * 4)?;
        let wb = ctx.create_storage_buffer(w_bytes.div_ceil(4) as u64 * 4)?;
        let azp = ctx.create_storage_buffer(4)?;
        let wzp = ctx.create_storage_buffer(4)?;
        let out = ctx.create_storage_buffer((4 * total) as u64)?;
        let partials = ctx.create_storage_buffer((4 * total * SPLIT_MAX) as u64)?;
        ctx.stream_upload(&xb, &pseudo(x_bytes, 3))?;
        ctx.stream_upload(&wb, &pseudo(w_bytes, 5))?;
        ctx.stream_upload(&azp, &[A_ZP, 0, 0, 0])?;
        ctx.stream_upload(&wzp, &[W_ZP, 0, 0, 0])?;
        ctx.flush()?;

        let mut push = Vec::new();
        #[rustfmt::skip]
        let fields = [
            total as u32, s.c_in as u32, s.c_out as u32, s.group as u32,
            h_in as u32, w_in as u32, h_out as u32, w_out as u32,
            s.k as u32, s.k as u32, s.stride as u32, s.stride as u32,
            s.pad as u32, s.pad as u32, 1, 1, (s.c_in / s.group) as u32,
            X_SIGNED, W_SIGNED,
            1, // split, overwritten per run
        ];
        for v in fields {
            push.extend_from_slice(&v.to_le_bytes());
        }
        Ok(Self {
            x: xb,
            w: wb,
            azp,
            wzp,
            out,
            partials,
            push,
            pixels,
            c_out: s.c_out,
            total,
            reps,
        })
    }

    fn with_split(&self, split: u32) -> Vec<u8> {
        let mut push = self.push.clone();
        let at = push.len() - 4;
        push[at..].copy_from_slice(&split.to_le_bytes());
        push
    }

    /// The production kernel: one thread per output element, `K` walked
    /// serially. This is what the 23 dispatches cost today.
    fn time_direct(&self, ctx: &VkContext, pipe: &ComputePipeline) -> Result<f64, Err> {
        let push = &self.push[..conv_integer::PUSH_BYTES as usize];
        self.best(ctx, |_| {
            ctx.stream_dispatch(
                pipe,
                &[&self.x, &self.w, &self.azp, &self.wzp, &self.out],
                push,
                [(self.total as u32).div_ceil(256), 1, 1],
            )?;
            Ok(())
        })
    }

    /// One implicit-GEMM pass at the given output tile.
    fn time_gemm(&self, ctx: &VkContext, pipe: &ComputePipeline, tile: u32) -> Result<f64, Err> {
        let grid = [
            (self.pixels as u32).div_ceil(tile),
            (self.c_out as u32).div_ceil(tile),
            1,
        ];
        self.best(ctx, |_| {
            ctx.stream_dispatch(
                pipe,
                &[&self.x, &self.w, &self.azp, &self.wzp, &self.out],
                &self.push,
                grid,
            )?;
            Ok(())
        })
    }

    /// The split-K pair, reduction included.
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
                &[&self.x, &self.w, &self.azp, &self.wzp, &self.partials],
                &push,
                grid,
            )?;
            ctx.stream_dispatch(
                reduce,
                &[&self.partials, &self.out],
                &push,
                [(self.total as u32).div_ceil(256), 1, 1],
            )?;
            Ok(())
        })
    }

    /// Times the flush alone, so CPU recording stays out of the number.
    ///
    /// At `reps == 1` (`--check`) the timing is meaningless and ignored: only
    /// the output the last dispatch leaves in `out` is being asked for.
    fn best(
        &self,
        ctx: &VkContext,
        mut enqueue: impl FnMut(usize) -> Result<(), Err>,
    ) -> Result<f64, Err> {
        let iters = if self.reps == 1 { 2 } else { 6 };
        let mut best = f64::MAX;
        for i in 0..iters {
            for r in 0..self.reps {
                enqueue(r)?;
            }
            let t = Instant::now();
            ctx.flush()?;
            if i > 0 {
                best = best.min(t.elapsed().as_secs_f64() / self.reps as f64);
            }
        }
        Ok(best)
    }

    fn read_out(&self, ctx: &VkContext) -> Result<Vec<i32>, Err> {
        Ok(ints(&ctx.stream_download(&self.out, 4 * self.total)?))
    }
}

/// Kernels, in the order the table prints them.
struct Kernels {
    direct: ComputePipeline,
    gemm16: ComputePipeline,
    gemm64: ComputePipeline,
    split16: ComputePipeline,
    split64: ComputePipeline,
    reduce: ComputePipeline,
}

/// Runs one model's population, printing per-geometry speedups over the direct
/// kernel, the totals weighted by node count, and how many outputs disagreed.
fn sweep(
    ctx: &VkContext,
    k: &Kernels,
    name: &str,
    shapes: &[Shape],
    reps: usize,
) -> Result<(), Err> {
    println!("\n===== {name} =====");
    print!(
        "{:>24} {:>5} {:>6} {:>9} {:>9} {:>9}",
        "geometry", "K", "P", "direct ms", "gemm16", "gemm64"
    );
    for s in SPLITS {
        print!(" {:>9}", format!("split{s}"));
    }
    println!(" {:>8}", "best");

    let mut direct_total = 0.0;
    let mut routed_total = 0.0;
    let mut best_total = 0.0;
    let mut mismatches = 0usize;

    for s in shapes {
        let kdepth = s.kdepth();
        let g = Geom::new(ctx, s, reps)?;
        let direct = g.time_direct(ctx, &k.direct)?;
        let want = g.read_out(ctx)?;
        print!(
            "{:>24} {:>5} {:>6} {:>9.3}",
            format!(
                "{}->{} {}x{} @{}{}{}",
                s.c_in,
                s.c_out,
                s.k,
                s.k,
                s.h_out,
                if s.stride > 1 { "/2" } else { "" },
                if s.group > 1 { " dw" } else { "" }
            ),
            kdepth,
            s.pixels(),
            direct * 1e3
        );

        // Grouped convolutions are not one GEMM: each output channel sees only
        // its own slice of the input, so rows cannot share a staged column.
        // They stay on the direct kernel and the table says so.
        if s.group > 1 {
            println!("  depthwise — no GEMM to make implicit");
            direct_total += direct * s.count as f64;
            routed_total += direct * s.count as f64;
            best_total += direct * s.count as f64;
            continue;
        }

        let mut best = direct;
        let mut best_label = String::from("direct");

        for (pipe, tile, label) in [
            (&k.gemm16, TILE_SMALL, "gemm16"),
            (&k.gemm64, TILE_BLOCKED, "gemm64"),
        ] {
            let t = g.time_gemm(ctx, pipe, tile)?;
            mismatches += disagree(&want, &g.read_out(ctx)?);
            print!(" {:>8.2}×", direct / t);
            if t < best {
                best = t;
                best_label = label.to_string();
            }
        }

        // two rows of splits: the 16×16 kernel, then the 64×64 one, whose tile
        // these outputs cannot afford until the split pays the grid back
        for (pipe, tile, label) in [
            (&k.split16, TILE_SMALL, "16×16"),
            (&k.split64, TILE_BLOCKED, "64×64"),
        ] {
            if tile == TILE_BLOCKED {
                print!(
                    "\n{:>24} {:>5} {:>6} {:>9} {:>9} {:>9}",
                    "  64×64 split", "", "", "", "", ""
                );
            }
            for &split in SPLITS {
                let t = g.time_splitk(ctx, pipe, &k.reduce, tile, split)?;
                mismatches += disagree(&want, &g.read_out(ctx)?);
                print!(" {:>8.2}×", direct / t);
                if t < best {
                    best = t;
                    best_label = format!("{label} split{split}");
                }
            }
        }
        // What `conv`'s existing predicates would pick, unchanged. `best` is an
        // oracle nothing can ship; this is the number a routing decision is
        // allowed to claim, and whether it lands near `best` is the whole
        // question of whether the f32 calibration transfers to `i32`.
        let (routed, routed_label) = match conv::split_k(s.pixels(), s.c_out, kdepth) {
            Some(split) => (
                g.time_splitk(ctx, &k.split64, &k.reduce, TILE_BLOCKED, split)?,
                format!("64×64 split{split}"),
            ),
            None if conv::prefer_blocked(s.pixels(), s.c_out) => (
                g.time_gemm(ctx, &k.gemm64, TILE_BLOCKED)?,
                "gemm64".to_string(),
            ),
            None => (
                g.time_gemm(ctx, &k.gemm16, TILE_SMALL)?,
                "gemm16".to_string(),
            ),
        };
        mismatches += disagree(&want, &g.read_out(ctx)?);
        println!(
            " {:>7.2}×  best: {best_label:<16} routed: {:>6.2}× {routed_label}",
            direct / best,
            direct / routed
        );

        direct_total += direct * s.count as f64;
        routed_total += routed * s.count as f64;
        best_total += best * s.count as f64;
    }

    println!(
        "\n{:>34} {:>9.3} ms",
        "direct, weighted by node count",
        direct_total * 1e3
    );
    println!(
        "{:>34} {:>9.3} ms  {:.2}×",
        "conv::split_k / prefer_blocked",
        routed_total * 1e3,
        direct_total / routed_total
    );
    println!(
        "{:>34} {:>9.3} ms  {:.2}×",
        "best per geometry (oracle)",
        best_total * 1e3,
        direct_total / best_total
    );
    println!(
        "{:>34} {}",
        "outputs disagreeing with direct",
        if mismatches == 0 {
            "0 — bit-exact".to_string()
        } else {
            format!("{mismatches}  ** BUG **")
        }
    );
    Ok(())
}

/// Output tiles, taken from the kernels themselves so the grid this bench
/// dispatches is the grid production dispatches.
const TILE_SMALL: u32 = conv_integer::TILE_SIZE;
const TILE_BLOCKED: u32 = conv_integer::BLOCKED_TILE_SIZE;

/// How many outputs a candidate got wrong. `i32` accumulation is exact and
/// order-independent, so the only tolerable answer is zero.
fn disagree(want: &[i32], got: &[i32]) -> usize {
    want.iter().zip(got).filter(|(a, b)| a != b).count()
}

fn main() -> Result<(), Err> {
    let ctx = VkContext::new()?;
    // Every kernel but `split16` is the one that ships, compiled from the same
    // source `interp::conv_integer` dispatches — a tier-1 bench that measured a
    // copy could drift from production without either side noticing.
    let pb = conv_integer::PUSH_BYTES;
    let b = conv_integer::BINDINGS;
    let k = Kernels {
        direct: ctx.create_pipeline(&compile_wgsl(&conv_integer::direct())?, b, pb)?,
        gemm16: ctx.create_pipeline(&compile_wgsl(&conv_integer::implicit_gemm())?, b, pb)?,
        gemm64: ctx.create_pipeline(&compile_wgsl(&conv_integer::blocked())?, b, pb)?,
        split16: ctx.create_pipeline(&compile_wgsl(SPLIT16)?, b, pb)?,
        split64: ctx.create_pipeline(&compile_wgsl(&conv_integer::blocked_splitk())?, b, pb)?,
        reduce: ctx.create_pipeline(
            &compile_wgsl(conv_integer::SPLIT_REDUCE)?,
            conv_integer::SPLIT_REDUCE_BINDINGS,
            pb,
        )?,
    };

    // `--check` dispatches each candidate once and only reads its output. The
    // timings it prints are noise, but the mismatch count is not: integer
    // accumulation is exact on any device, so correctness can be gated here
    // even on lavapipe, where the milliseconds mean nothing.
    let reps = if std::env::args().any(|a| a == "--check") {
        1
    } else {
        8
    };

    println!(
        "ConvInteger on the geometries the pointwise fast path refuses — the 23\n\
         dispatches that are 92.5% of resnet50-int8's GPU time, and the 18 of\n\
         mobilenetv2-int8. `direct` is the production kernel; every other column\n\
         is a speedup over it. `dw` marks depthwise, where there is no GEMM.{}",
        if reps == 1 {
            "\n--check: correctness only, the milliseconds are meaningless."
        } else {
            ""
        }
    );

    sweep(&ctx, &k, "resnet50-int8  (23 nodes)", RESNET50_INT8, reps)?;
    sweep(
        &ctx,
        &k,
        "mobilenetv2-int8  (18 nodes)",
        MOBILENETV2_INT8,
        reps,
    )?;
    Ok(())
}
