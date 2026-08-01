//! Is the cooperative-matrix path reachable for the convolutions the pointwise
//! fast path refuses — and if it is reachable, is it worth reaching?
//!
//! `interp::conv_integer` sends 1×1/stride-1/pad-0 convolutions to
//! `MatMulInteger`, where `MMI_matmul_coop_k32` runs them on the tensor cores.
//! The other 23 nodes of resnet50-int8 run the implicit-GEMM ladder
//! (`ConvInteger_split`), which rebuilds each im2col column from its index and
//! never materializes one. Tensor cores cannot do that: `coopMatLoad` takes a
//! base offset and a row stride, so the operand has to *exist* in memory. The
//! only way onto that path is to write the im2col matrix out and multiply it.
//!
//! Two facts set the stakes before any kernel runs, both from the model itself:
//!
//! - resnet50-int8 is **3.006 GOP over the 30 pointwise nodes in 0.534 ms**
//!   (5.6 TOP/s, cooperative) and **4.706 GOP over the 23 others in 1.540 ms**
//!   (3.1 TOP/s, implicit GEMM). If a cooperative conv reached the pointwise
//!   rate exactly and im2col were free, the family would go 1.540 → 0.836 ms:
//!   **0.70 ms of a 2.97 ms GPU compute**, and less than that of a 4.1 ms wall.
//! - Both rates are ~2% of the card's int8 tensor peak, so neither path is
//!   compute-bound at batch 1 and the ceiling above is not obviously real.
//!
//! im2col is not free: it materializes `P·K` bytes per node, ~13 MB over the
//! eligible geometries, which is a write the implicit kernel does not do.
//!
//! The mapping needs no transpose and no pack, which is the one thing in favour
//! of trying. Take `M = C_out`, `N = P`, and the cooperative kernel's own
//! layout — A is `[M, K]` row major, B is `[N, K]` (column major for the `[K,N]`
//! operand), out is `[M, N]` row major:
//!
//! - A is the ONNX weight `[C_out, C_in, KH, KW]` flattened. Already correct.
//! - B is the im2col matrix `[P, K]`, one row per output pixel. What we write.
//! - out is `[C_out, P]`, which *is* NCHW. No transpose after the multiply.
//!
//! Eligibility is `coop_applies` unchanged: `M ≥ 16`, `N ≥ 16`, `K` a multiple
//! of the cooperative K (32 here), and no operand left signed in memory. The
//! weight is `int8` in this model, so it is flipped once — a constant, a
//! session cost, done on the host here. The 7×7 stem has `K = 147` and is
//! refused; it could be padded (`a = za`, `b = zb` contributes exactly 0), but
//! it is 0.066 of the 1.540 ms and would not pay for the code.
//!
//! Run: `cargo run --release -p onnx-vulkan-core --example conv_integer_coop`
//! on a device that reports a `u8×u8` cooperative combination. lavapipe reports
//! none and the bench says so instead of measuring nothing.

use onnx_vulkan_core::shaders::{conv, conv_integer, matmul_integer};
use std::time::Instant;
use vk_compute::{ComputePipeline, GpuBuffer, VkContext, compile_wgsl};

type Err = Box<dyn std::error::Error>;

/// One `ConvInteger` geometry and how many nodes of the model run it.
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

impl Shape {
    fn kdepth(&self) -> usize {
        self.c_in * self.k * self.k
    }
    fn pixels(&self) -> usize {
        self.h_out * self.h_out
    }
}

/// The 23 nodes of resnet50-int8 the pointwise fast path refuses, identical to
/// the table in `conv_integer_gemm` minus the `group` column: every one of them
/// is `group == 1`, so every one of them is a GEMM. Depthwise convolutions are
/// absent by construction — `mobilenetv2-int8` has 17 of them and none is a
/// matrix multiply at all, cooperative or otherwise.
#[rustfmt::skip]
const RESNET50_INT8: &[Shape] = &[
    Shape { c_in:  256, c_out:  256, k: 3, h_in:  14, h_out:  14, stride: 1, pad: 1, count: 6 },
    Shape { c_in:  128, c_out:  128, k: 3, h_in:  28, h_out:  28, stride: 1, pad: 1, count: 4 },
    Shape { c_in:   64, c_out:   64, k: 3, h_in:  56, h_out:  56, stride: 1, pad: 1, count: 3 },
    Shape { c_in:  512, c_out:  512, k: 3, h_in:   7, h_out:   7, stride: 1, pad: 1, count: 3 },
    Shape { c_in:    3, c_out:   64, k: 7, h_in: 224, h_out: 112, stride: 2, pad: 3, count: 1 },
    Shape { c_in:  256, c_out:  512, k: 1, h_in:  56, h_out:  28, stride: 2, pad: 0, count: 1 },
    Shape { c_in:  256, c_out:  128, k: 1, h_in:  56, h_out:  28, stride: 2, pad: 0, count: 1 },
    Shape { c_in:  512, c_out: 1024, k: 1, h_in:  28, h_out:  14, stride: 2, pad: 0, count: 1 },
    Shape { c_in:  512, c_out:  256, k: 1, h_in:  28, h_out:  14, stride: 2, pad: 0, count: 1 },
    Shape { c_in: 1024, c_out: 2048, k: 1, h_in:  14, h_out:   7, stride: 2, pad: 0, count: 1 },
    Shape { c_in: 1024, c_out:  512, k: 1, h_in:  14, h_out:   7, stride: 2, pad: 0, count: 1 },
];

/// What the model exports (`scripts/qlinear-census.py`): `x` is `uint8` with a
/// nonzero zero point, `w` is `int8` with a zero point of 0 on all 105 nodes.
const X_SIGNED: u32 = 0;
const W_SIGNED: u32 = 1;
const A_ZP: u8 = 114;
const W_ZP: u8 = 0;

/// The materialization the implicit-GEMM kernel exists to avoid: one `[P, K]`
/// row per output pixel, in the layout the cooperative kernel's B operand
/// wants. Written here rather than in `shaders/` because it ships only if this
/// bench says the multiply that follows it is worth the write.
///
/// Out-of-bounds taps are filled with the activation zero point, which is what
/// `ConvInteger` pads with, so the padded product `(a_zp - a_zp)(w - w_zp)`
/// vanishes without the multiply knowing anything about the geometry. Dilation
/// is not handled: it is 1 on all 23 nodes and this is a probe, not a kernel.
const IM2COL: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<u32>;    // u8 packed [C_in, H, W]
@group(0) @binding(1) var<storage, read> azp: array<u32>;
@group(0) @binding(2) var<storage, read_write> col: array<u32>;  // u8 packed [P, K]

struct Push {
    words: u32, kdepth: u32, khkw: u32,
    h_in: u32, w_in: u32, w_out: u32,
    kw: u32, sh: u32, sw: u32, phb: u32, pwb: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.words) { return; }
    let k4 = pc.kdepth / 4u;
    let p = i / k4;
    let k0 = (i % k4) * 4u;
    let oh = p / pc.w_out;
    let ow = p % pc.w_out;
    let a_zp = azp[0] & 0xffu;
    var word = 0u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let k = k0 + j;
        let c = k / pc.khkw;
        let r = (k % pc.khkw) / pc.kw;
        let s = k % pc.kw;
        let ih = i32(oh * pc.sh + r) - i32(pc.phb);
        let iw = i32(ow * pc.sw + s) - i32(pc.pwb);
        var b = a_zp;
        if (ih >= 0 && iw >= 0 && ih < i32(pc.h_in) && iw < i32(pc.w_in)) {
            let idx = (c * pc.h_in + u32(ih)) * pc.w_in + u32(iw);
            b = (x[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xffu;
        }
        word = word | (b << (j * 8u));
    }
    col[i] = word;
}
"#;

const IM2COL_BINDINGS: u32 = 3;
const IM2COL_PUSH_BYTES: u32 = 44;
/// Largest split `conv::split_k` returns, and so how many partial images the
/// scratch buffer of the reference path has to hold.
const SPLIT_MAX: usize = 32;
const TILE_SMALL: u32 = conv_integer::TILE_SIZE;
const TILE_BLOCKED: u32 = conv_integer::BLOCKED_TILE_SIZE;

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

/// Integer accumulation is exact and order-independent on both paths, so the
/// only tolerable disagreement is none.
fn disagree(want: &[i32], got: &[i32]) -> usize {
    want.iter().zip(got).filter(|(a, b)| a != b).count()
}

struct Geom {
    x: GpuBuffer,
    w: GpuBuffer,
    /// The weight with every sign bit flipped: the cooperative combination is
    /// `u8 × u8` and a cooperative matrix has no bitwise operations, so a
    /// signed operand has to be brought to unsigned by an earlier pass. It is a
    /// constant, so production pays this once per session (`FLIP_BYTES`); the
    /// bench does it on the host and leaves it out of the timing for the same
    /// reason.
    w_flip: GpuBuffer,
    azp: GpuBuffer,
    wzp: GpuBuffer,
    col: GpuBuffer,
    out: GpuBuffer,
    partials: GpuBuffer,
    push: Vec<u8>,
    im2col_push: Vec<u8>,
    coop_push: Vec<u8>,
    tiled_push: Vec<u8>,
    pixels: usize,
    c_out: usize,
    kdepth: usize,
    total: usize,
    reps: usize,
}

impl Geom {
    fn new(ctx: &VkContext, s: &Shape, reps: usize) -> Result<Self, Err> {
        let (h_in, w_in) = (s.h_in, s.h_in);
        let (h_out, w_out) = (s.h_out, s.h_out);
        let pixels = s.pixels();
        let kdepth = s.kdepth();
        let total = s.c_out * pixels;
        let x_bytes = s.c_in * h_in * w_in;
        let w_bytes = s.c_out * kdepth;

        let raw_w = pseudo(w_bytes, 5);
        let flipped: Vec<u8> = raw_w.iter().map(|b| b ^ 0x80).collect();

        let xb = ctx.create_storage_buffer(x_bytes.div_ceil(4) as u64 * 4)?;
        let wb = ctx.create_storage_buffer(w_bytes.div_ceil(4) as u64 * 4)?;
        let wf = ctx.create_storage_buffer(w_bytes.div_ceil(4) as u64 * 4)?;
        let azp = ctx.create_storage_buffer(4)?;
        let wzp = ctx.create_storage_buffer(4)?;
        let col = ctx.create_storage_buffer((pixels * kdepth).div_ceil(4) as u64 * 4)?;
        let out = ctx.create_storage_buffer((4 * total) as u64)?;
        let partials = ctx.create_storage_buffer((4 * total * SPLIT_MAX) as u64)?;
        ctx.stream_upload(&xb, &pseudo(x_bytes, 3))?;
        ctx.stream_upload(&wb, &raw_w)?;
        ctx.stream_upload(&wf, &flipped)?;
        ctx.stream_upload(&azp, &[A_ZP, 0, 0, 0])?;
        ctx.stream_upload(&wzp, &[W_ZP, 0, 0, 0])?;
        ctx.flush()?;

        let mut push = Vec::new();
        #[rustfmt::skip]
        let fields = [
            total as u32, s.c_in as u32, s.c_out as u32, 1,
            h_in as u32, w_in as u32, h_out as u32, w_out as u32,
            s.k as u32, s.k as u32, s.stride as u32, s.stride as u32,
            s.pad as u32, s.pad as u32, 1, 1, s.c_in as u32,
            X_SIGNED, W_SIGNED,
            1, // split, overwritten per run
        ];
        for v in fields {
            push.extend_from_slice(&v.to_le_bytes());
        }

        let mut im2col_push = Vec::new();
        #[rustfmt::skip]
        let fields = [
            (pixels * kdepth / 4) as u32, kdepth as u32, (s.k * s.k) as u32,
            h_in as u32, w_in as u32, w_out as u32,
            s.k as u32, s.stride as u32, s.stride as u32, s.pad as u32, s.pad as u32,
        ];
        for v in fields {
            im2col_push.extend_from_slice(&v.to_le_bytes());
        }

        // A is the weight, and it reaches the kernel flipped, so its zero point
        // needs the same shift; B is the activation, unsigned already.
        let mut coop_push = Vec::new();
        for v in [
            s.c_out as u32,
            kdepth as u32,
            pixels as u32,
            matmul_integer::SIGN_FLIP_BYTE,
            0,
        ] {
            coop_push.extend_from_slice(&v.to_le_bytes());
        }

        // Same problem for the WGSL kernel, which takes K in words and carries
        // one extra field: A does not need flipping *here* because it reaches
        // the kernel already flipped.
        let mut tiled_push = Vec::new();
        for v in [
            s.c_out as u32,
            (kdepth / 4) as u32,
            pixels as u32,
            0,
            matmul_integer::SIGN_FLIP_BYTE,
            0,
        ] {
            tiled_push.extend_from_slice(&v.to_le_bytes());
        }

        Ok(Self {
            x: xb,
            w: wb,
            w_flip: wf,
            azp,
            wzp,
            col,
            out,
            partials,
            push,
            im2col_push,
            coop_push,
            tiled_push,
            pixels,
            c_out: s.c_out,
            kdepth,
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

    fn enqueue_gemm(&self, ctx: &VkContext, pipe: &ComputePipeline, tile: u32) -> Result<(), Err> {
        ctx.stream_dispatch(
            pipe,
            &[&self.x, &self.w, &self.azp, &self.wzp, &self.out],
            &self.push,
            [
                (self.pixels as u32).div_ceil(tile),
                (self.c_out as u32).div_ceil(tile),
                1,
            ],
        )?;
        Ok(())
    }

    fn enqueue_splitk(
        &self,
        ctx: &VkContext,
        conv: &ComputePipeline,
        reduce: &ComputePipeline,
        split: u32,
    ) -> Result<(), Err> {
        let push = self.with_split(split);
        ctx.stream_dispatch(
            conv,
            &[&self.x, &self.w, &self.azp, &self.wzp, &self.partials],
            &push,
            [
                (self.pixels as u32).div_ceil(TILE_BLOCKED),
                (self.c_out as u32).div_ceil(TILE_BLOCKED),
                split,
            ],
        )?;
        ctx.stream_dispatch(
            reduce,
            &[&self.partials, &self.out],
            &push,
            [(self.total as u32).div_ceil(256), 1, 1],
        )?;
        Ok(())
    }

    fn enqueue_im2col(&self, ctx: &VkContext, pipe: &ComputePipeline) -> Result<(), Err> {
        let words = (self.pixels * self.kdepth / 4) as u32;
        ctx.stream_dispatch(
            pipe,
            &[&self.x, &self.azp, &self.col],
            &self.im2col_push,
            [words.div_ceil(256), 1, 1],
        )?;
        Ok(())
    }

    /// A is the flipped weight `[C_out, K]`, B the im2col `[P, K]`, out
    /// `[C_out, P]` — which is already the NCHW the conv has to produce.
    fn enqueue_coop(&self, ctx: &VkContext, pipe: &ComputePipeline) -> Result<(), Err> {
        let tile = matmul_integer::COOP_TILE;
        ctx.stream_dispatch(
            pipe,
            &[&self.w_flip, &self.col, &self.wzp, &self.azp, &self.out],
            &self.coop_push,
            [
                (self.pixels as u32).div_ceil(tile),
                (self.c_out as u32).div_ceil(tile),
                1,
            ],
        )?;
        Ok(())
    }

    /// The same multiply on the portable WGSL kernel. Same five buffers in the
    /// same layout — `PACK_B` produces exactly the `[N, K]` the im2col pass
    /// writes — so this column separates "the tensor cores win" from "stating
    /// the convolution as an explicit GEMM wins", which are not the same claim.
    fn enqueue_tiled(&self, ctx: &VkContext, pipe: &ComputePipeline) -> Result<(), Err> {
        ctx.stream_dispatch(
            pipe,
            &[&self.w_flip, &self.col, &self.wzp, &self.azp, &self.out],
            &self.tiled_push,
            [
                (self.pixels as u32).div_ceil(matmul_integer::TILE_SIZE),
                (self.c_out as u32).div_ceil(matmul_integer::TILE_SIZE),
                1,
            ],
        )?;
        Ok(())
    }

    fn read_out(&self, ctx: &VkContext) -> Result<Vec<i32>, Err> {
        Ok(ints(&ctx.stream_download(&self.out, 4 * self.total)?))
    }

    /// Times the flush alone, so CPU recording stays out of the number.
    fn best(
        &self,
        ctx: &VkContext,
        mut enqueue: impl FnMut() -> Result<(), Err>,
    ) -> Result<f64, Err> {
        let iters = if self.reps == 1 { 2 } else { 6 };
        let mut best = f64::MAX;
        for i in 0..iters {
            for _ in 0..self.reps {
                enqueue()?;
            }
            let t = Instant::now();
            ctx.flush()?;
            if i > 0 {
                best = best.min(t.elapsed().as_secs_f64() / self.reps as f64);
            }
        }
        Ok(best)
    }
}

struct Kernels {
    gemm16: ComputePipeline,
    gemm64: ComputePipeline,
    split64: ComputePipeline,
    reduce: ComputePipeline,
    im2col: ComputePipeline,
    tiled: ComputePipeline,
    /// Absent on any device that advertises no `u8×u8` combination — lavapipe,
    /// where the arithmetic can still be checked but the milliseconds cannot.
    coop: Option<ComputePipeline>,
}

fn main() -> Result<(), Err> {
    let ctx = VkContext::new()?;
    let variant = matmul_integer::coop_variant(&ctx.coop_u8, ctx.subgroup_size);

    let pb = conv_integer::PUSH_BYTES;
    let b = conv_integer::BINDINGS;
    let k = Kernels {
        gemm16: ctx.create_pipeline(&compile_wgsl(&conv_integer::implicit_gemm())?, b, pb)?,
        gemm64: ctx.create_pipeline(&compile_wgsl(&conv_integer::blocked())?, b, pb)?,
        split64: ctx.create_pipeline(&compile_wgsl(&conv_integer::blocked_splitk())?, b, pb)?,
        reduce: ctx.create_pipeline(
            &compile_wgsl(conv_integer::SPLIT_REDUCE)?,
            conv_integer::SPLIT_REDUCE_BINDINGS,
            pb,
        )?,
        im2col: ctx.create_pipeline(&compile_wgsl(IM2COL)?, IM2COL_BINDINGS, IM2COL_PUSH_BYTES)?,
        tiled: ctx.create_pipeline(
            &compile_wgsl(&matmul_integer::matmul(ctx.has_integer_dot_product))?,
            matmul_integer::MATMUL_BINDINGS,
            matmul_integer::MATMUL_PUSH_BYTES,
        )?,
        coop: variant
            .map(|v| {
                ctx.create_pipeline(
                    &v.spirv(),
                    matmul_integer::COOP_BINDINGS,
                    matmul_integer::COOP_PUSH_BYTES,
                )
            })
            .transpose()?,
    };

    let reps = if std::env::args().any(|a| a == "--check") {
        1
    } else {
        8
    };

    println!(
        "resnet50-int8's 23 non-pointwise ConvInteger nodes: what ships today\n\
         (`routed` = conv::split_k / prefer_blocked on the implicit-GEMM ladder)\n\
         against materializing im2col and multiplying it — on the portable WGSL\n\
         kernel (`wgsl`) and on the tensor cores (`coop`).\n\
         cooperative matrix: {}",
        match variant {
            Some(v) => format!("{} (K tile {}, subgroup {})", v.key, v.k_tile, v.workgroup),
            None => format!(
                "none advertised (subgroup {}) — the coop columns are blank and \
                 only the bit-exactness of the im2col GEMM is being checked",
                ctx.subgroup_size
            ),
        }
    );
    println!(
        "\n{:>22} {:>5} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "geometry", "K", "P", "routed ms", "im2col", "wgsl", "coop", "im+coop", "speedup"
    );

    let mut routed_total = 0.0;
    let mut coop_total = 0.0;
    let mut tiled_total = 0.0;
    let mut ineligible = 0.0;
    let mut mismatches = 0usize;
    let mut col_mb = 0.0;

    for s in RESNET50_INT8 {
        let kdepth = s.kdepth();
        let g = Geom::new(&ctx, s, reps)?;
        let label = format!(
            "{}->{} {}x{} @{}{}",
            s.c_in,
            s.c_out,
            s.k,
            s.k,
            s.h_out,
            if s.stride > 1 { "/2" } else { "" }
        );

        // What production dispatches for this geometry, unchanged.
        let routed = match conv::split_k(s.pixels(), s.c_out, kdepth) {
            Some(split) => g.best(&ctx, || {
                g.enqueue_splitk(&ctx, &k.split64, &k.reduce, split)
            })?,
            None if conv::prefer_blocked(s.pixels(), s.c_out) => {
                g.best(&ctx, || g.enqueue_gemm(&ctx, &k.gemm64, TILE_BLOCKED))?
            }
            None => g.best(&ctx, || g.enqueue_gemm(&ctx, &k.gemm16, TILE_SMALL))?,
        };
        let want = g.read_out(&ctx)?;
        print!(
            "{label:>22} {kdepth:>5} {:>6} {:>9.3}",
            s.pixels(),
            routed * 1e3
        );
        routed_total += routed * s.count as f64;

        // `coop_applies` unchanged, so eligibility here is the eligibility
        // `mmi_dispatch` would compute. Only K disqualifies anything: every
        // geometry has C_out ≥ 64 and P ≥ 49. With no variant to ask, the same
        // rule is applied by hand — and it has to be, because the WGSL kernel
        // needs `K % 4 == 0` and the 7×7 stem's K = 147 is not even that.
        let eligible = match variant {
            Some(v) => matmul_integer::coop_applies(v, s.c_out, kdepth, s.pixels(), false),
            None => kdepth.is_multiple_of(32) && s.c_out >= 16 && s.pixels() >= 16,
        };
        if !eligible {
            println!("  K not a multiple of 32 — coop refuses, stays routed");
            coop_total += routed * s.count as f64;
            tiled_total += routed * s.count as f64;
            ineligible += routed * s.count as f64;
            continue;
        }

        let im2col = g.best(&ctx, || g.enqueue_im2col(&ctx, &k.im2col))?;
        let tiled = g.best(&ctx, || {
            g.enqueue_im2col(&ctx, &k.im2col)?;
            g.enqueue_tiled(&ctx, &k.tiled)
        })?;
        mismatches += disagree(&want, &g.read_out(&ctx)?);
        col_mb += (s.pixels() * kdepth * s.count) as f64 / 1e6;
        tiled_total += tiled * s.count as f64;

        let Some(coop_pipe) = k.coop.as_ref() else {
            println!(
                " {:>9.3} {:>9.3} {:>9} {:>9} {:>7.2}×",
                im2col * 1e3,
                tiled * 1e3,
                "—",
                "—",
                routed / tiled
            );
            coop_total += tiled * s.count as f64;
            continue;
        };
        let coop = g.best(&ctx, || g.enqueue_coop(&ctx, coop_pipe))?;
        let both = g.best(&ctx, || {
            g.enqueue_im2col(&ctx, &k.im2col)?;
            g.enqueue_coop(&ctx, coop_pipe)
        })?;
        mismatches += disagree(&want, &g.read_out(&ctx)?);
        println!(
            " {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>7.2}×",
            im2col * 1e3,
            tiled * 1e3,
            coop * 1e3,
            both * 1e3,
            routed / both
        );
        coop_total += both * s.count as f64;
    }

    println!(
        "\n{:>38} {:>9.3} ms",
        "routed (ships today), by node count",
        routed_total * 1e3
    );
    println!(
        "{:>38} {:>9.3} ms  {:.2}×",
        "im2col + WGSL tiled matmul",
        tiled_total * 1e3,
        routed_total / tiled_total
    );
    println!(
        "{:>38} {:>9.3} ms  {:.2}×",
        "im2col + cooperative matrix",
        coop_total * 1e3,
        routed_total / coop_total
    );
    println!(
        "{:>38} {:>9.3} ms  (coop refuses these; counted as routed on both sides)",
        "of which unchanged",
        ineligible * 1e3
    );
    println!("{:>38} {col_mb:>9.1} MB", "im2col written per inference");
    println!(
        "{:>38} {}",
        "outputs disagreeing with routed",
        if mismatches == 0 {
            "0 — bit-exact".to_string()
        } else {
            format!("{mismatches}  ** BUG **")
        }
    );
    Ok(())
}
