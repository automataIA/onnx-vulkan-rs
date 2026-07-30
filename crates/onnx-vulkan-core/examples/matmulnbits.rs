//! What shape should `MatMulNBits` be at `M = 1`, i.e. in the decode step?
//!
//! Every weight of an int4 LLM goes through this op, and at `sequence_length = 1`
//! it is a matrix-vector product — 253 of them per token on qwen2.5-VL's decoder.
//! So this is the kernel that decides tokens/s, and the regime has to be
//! classified before anything is tuned.
//!
//! **It is bandwidth-bound, and not marginally.** gemma3's most frequent
//! geometry (`K = 1152, N = 6912`) reads 3.98 MB of packed weight to do
//! 15.9 MFLOP: **4.0 FLOP/byte**, against the 4070's ridge of ~57.8. The
//! arithmetic is free; the question is only whether the weight arrives at
//! 504 GB/s. Read the GB/s column.
//!
//! **The dequantization is the thing that can break that**, which is why the
//! TFLOP/s column is printed next to it: unpacking eight nibbles from a `u32` is
//! eight shift+mask+fma, ALU work proportional to bytes read rather than to
//! useful flops. A kernel can be starved of bandwidth *because* it is busy
//! unpacking, and the two columns are how that shows up.
//!
//! The search space is **not** the one `example gemv` explored, and the reason
//! is the layout. There, `B` was `[K, N]` and a coalesced read meant consecutive
//! threads on consecutive **columns**; the lever was split-K, because the grid
//! was `ceil(N/16)` workgroups and starved. Here:
//!
//! - `B` is `[N, n_blocks, blob]`, i.e. **K-contiguous within a column**, so a
//!   coalesced read means consecutive threads on consecutive **words of one
//!   column**. The mapping is transposed with respect to `gemv`, and getting it
//!   backwards produces a kernel that looks more parallel and reads 576 bytes
//!   apart per lane.
//! - the grid is *already* `N` workgroups — 1152 to 262144, never starved — so
//!   split-K would fabricate parallelism nobody needs and add a reduction pass.
//!
//! What is plausibly wrong with the current kernel is the opposite of roberta's
//! problem: at `K = 1152` a column is 144 `u32`, spread over `WG = 64` threads,
//! so each thread loads 2.25 words and then pays a **6-deep tree reduction with
//! a barrier per level**. The two candidate levers are therefore how many
//! threads share a column (reduction depth, and whether a column stays inside
//! one subgroup) and how wide each load is (`u32` vs `vec4<u32>`, 4 vs 16 bytes
//! per lane per transaction).
//!
//! Geometries are every distinct `(K, N)` the two q4 decoders in the matrix run,
//! with the node counts they run them at, from `scripts/matmulnbits-census.py` —
//! never synthetic. That includes the two `lm_head` outliers (`N = 151936` and
//! `262144`, 150+ MB in a single node), reported separately: they are one node
//! per token but ~30% of the packed weight a step reads, and a rule tuned on the
//! 1152-wide projections has no reason to hold there.
//!
//! Run: `cargo run --release -p onnx-vulkan-core --example matmulnbits`

use onnx_vulkan_core::shaders::matmul_nbits as base;
use std::time::Instant;
use vk_compute::{VkContext, compile_wgsl};

/// `(K, N, nodes per token, label)`. Both models at 4 bits, `block_size = 32`.
const SHAPES: &[(usize, usize, usize, &str)] = &[
    // gemma3-1b
    (1152, 256, 52, "gemma3 qkv"),
    (1152, 6912, 52, "gemma3 ffn-in"),
    (1152, 1024, 26, "gemma3 attn-out"),
    (1024, 1152, 26, "gemma3 proj"),
    (6912, 1152, 26, "gemma3 ffn-out"),
    // qwen2.5-VL decoder
    (2048, 2048, 72, "qwen qkv"),
    (2048, 256, 72, "qwen kv"),
    (2048, 11008, 72, "qwen ffn-in"),
    (11008, 2048, 36, "qwen ffn-out"),
];

/// The unembedding, benched apart: one node per token, but 150+ MB of it.
const LM_HEADS: &[(usize, usize, usize, &str)] = &[
    (1152, 262144, 1, "gemma3 lm_head"),
    (2048, 151936, 1, "qwen lm_head"),
];

const BLOCK_SIZE: usize = 32;
/// `u32` words per block: `block_size · bits / 8 / 4`.
const BLOB_WORDS: usize = BLOCK_SIZE * 4 / 8 / 4;

/// `(lanes per column, words per load)`. 256 threads either way, so
/// `256 / lanes` columns share a workgroup and each column is reduced across
/// its own `lanes`.
///
/// `lanes = 32` is the interesting one on NVIDIA: a column then lives inside a
/// single subgroup, so its reduction never leaves the warp.
///
/// `vec = 4` reads a whole `vec4<u32>` per lane and — in the first measurement —
/// **lost by 2–3×**. Kept in the table as the control, because the reason it
/// loses is the point: the four words of a quad each re-derive `block`, the
/// zero-point offset and the scale, so the wider load buys bandwidth and pays
/// four times the address arithmetic for it. `vec = 0` is what that finding
/// turns into: same 16-byte load, the per-block work done once.
const VARIANTS: &[(u32, u32)] = &[
    (256, 1),
    (128, 1),
    (64, 1),
    (32, 1),
    (16, 1),
    (8, 1),
    (64, 4),
    (32, 4),
    (16, 4),
    (8, 4),
    (64, 0),
    (32, 0),
    (16, 0),
    (8, 0),
    (64, 2),
    (32, 2),
    (16, 2),
    (8, 2),
];

/// Tree-reduce the `lanes` partials of a column; lane 0 writes. Consecutive
/// `tid` are consecutive lanes of the same column, so the stride is 1.
fn block_reduction(lanes: u32) -> String {
    if lanes == 1 {
        return "    if (col < pc.n) { out[row * pc.n + col] = acc; }".to_string();
    }
    r#"    red[tid] = acc;
    workgroupBarrier();
    for (var s = LANES / 2u; s > 0u; s = s / 2u) {
        if (lane < s) { red[tid] = red[tid] + red[tid + s]; }
        workgroupBarrier();
    }
    if (lane == 0u && col < pc.n) { out[row * pc.n + col] = red[tid]; }"#
        .to_string()
}

/// The candidate kernel, parameterized by reduction width and load width.
///
/// `col = group · COLS + tid / LANES` and `lane = tid % LANES`: the lane index
/// varies fastest, so the `LANES` threads of a column read `LANES` consecutive
/// words of that column. This is the mapping the `[N, blocks, blob]` layout
/// wants, and it is transposed with respect to `example gemv`'s.
///
/// `vec` widens each lane's load to `vec4<u32>` = 16 bytes, the widest single
/// transaction, at the price of 4× the unpacking per lane and `4·LANES`-word
/// granularity. `words` is a multiple of 4 for every export in the matrix
/// (`blob_words = 4`), so no tail handling is needed.
fn candidate(lanes: u32, vec: u32) -> String {
    let cols = 256 / lanes;
    let quant_ty = if vec == 4 { "vec4<u32>" } else { "u32" };
    // `vec = 0`: one whole block per lane iteration. A block is `blob_words = 4`
    // words = one `vec4<u32>` = 16 bytes = 32 weights, and it is the unit the
    // format is built around — one scale and one zero point cover it. So the
    // block index comes from the loop instead of from a division, `z` and
    // `scale` are loaded once instead of four times, and the 32 nibbles need no
    // bound check because `K` is a multiple of `block_size` in every export the
    // census found. What is left inside the loop is the arithmetic that is
    // actually required.
    if vec == 0 || vec == 2 {
        let reduction = block_reduction(lanes);
        // `vec = 2`: the same block form with **four independent accumulators**,
        // one per word of the quad, summed at the end. `vec = 0` chains all 32
        // fma of a block through a single `part`, and that chain is the reason
        // it loses: with one block per lane there is a 32-long dependency and a
        // single load to hide it behind, while the word form gets four
        // independent chains of 8 for free by resetting `part` per word.
        let body = if vec == 2 {
            r#"            var p0 = 0.0; var p1 = 0.0; var p2 = 0.0; var p3 = 0.0;
            for (var i = 0u; i < 8u; i = i + 1u) {
                let sh = 4u * i;
                p0 = fma(f32((quad.x >> sh) & 15u) - z, a[k0 + i], p0);
                p1 = fma(f32((quad.y >> sh) & 15u) - z, a[k0 + 8u + i], p1);
                p2 = fma(f32((quad.z >> sh) & 15u) - z, a[k0 + 16u + i], p2);
                p3 = fma(f32((quad.w >> sh) & 15u) - z, a[k0 + 24u + i], p3);
            }
            let part = (p0 + p1) + (p2 + p3);"#
        } else {
            r#"            var part = 0.0;
            for (var w = 0u; w < 4u; w = w + 1u) {
                let word = quad[w];
                let kw = k0 + w * 8u;
                for (var i = 0u; i < 8u; i = i + 1u) {
                    part = fma(f32((word >> (4u * i)) & 15u) - z, a[kw + i], part);
                }
            }"#
        };
        return format!(
            r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> quant: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> zero_points: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

struct Push {{
    k: u32, n: u32, n_blocks: u32, blob_words: u32,
    block_size: u32, zp_row_bytes: u32, gx: u32, pad: u32,
}}
var<immediate> pc: Push;

const LANES = {lanes}u;
const COLS = {cols}u;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {{
    let group = wid.y * pc.gx + wid.x;
    let col = group * COLS + tid / LANES;
    let lane = tid % LANES;
    let row = wid.z;
    var acc = 0.0;
    if (col < pc.n) {{
        let q_base = col * pc.n_blocks;
        let a_base = row * pc.k;
        for (var block = lane; block < pc.n_blocks; block = block + LANES) {{
            let quad = quant[q_base + block];
            let zoff = col * pc.zp_row_bytes + block / 2u;
            let z = f32((zero_points[zoff / 4u] >> (8u * (zoff % 4u) + 4u * (block % 2u))) & 15u);
            let k0 = a_base + block * pc.block_size;
{body}
            acc = fma(part, scales[q_base + block], acc);
        }}
    }}
{reduction}
}}
"#
        );
    }
    // the per-word body, shared by both load widths
    let word_body = |word: &str, w: &str| {
        format!(
            r#"
            {{
                let ww = {w};
                let block = ww / {BLOB_WORDS}u;
                let k0 = block * pc.block_size + (ww % {BLOB_WORDS}u) * 8u;
                let word = {word};
                let zoff = col * pc.zp_row_bytes + block / 2u;
                let z = f32((zero_points[zoff / 4u] >> (8u * (zoff % 4u) + 4u * (block % 2u))) & 15u);
                var part = 0.0;
                for (var i = 0u; i < 8u; i = i + 1u) {{
                    let k = k0 + i;
                    if (k >= pc.k) {{ break; }}
                    part = fma(f32((word >> (4u * i)) & 15u) - z, a[a_base + k], part);
                }}
                acc = fma(part, scales[col * pc.n_blocks + block], acc);
            }}"#
        )
    };
    let loop_body = if vec == 4 {
        format!(
            r#"
        for (var v = lane; v < words / 4u; v = v + LANES) {{
            let quad = quant[v];
            {}
            {}
            {}
            {}
        }}"#,
            word_body("quad.x", "v * 4u"),
            word_body("quad.y", "v * 4u + 1u"),
            word_body("quad.z", "v * 4u + 2u"),
            word_body("quad.w", "v * 4u + 3u"),
        )
    } else {
        format!(
            r#"
        for (var w = lane; w < words; w = w + LANES) {{
            {}
        }}"#,
            word_body("quant[q_base + w]", "w"),
        )
    };
    // with vec4 loads the base is in quads, not words
    let base_expr = if vec == 4 {
        "let q_base = col * words / 4u;"
    } else {
        "let q_base = col * words;"
    };
    let quant_index = if vec == 4 { "q_base + " } else { "" };
    let reduction = block_reduction(lanes);
    format!(
        r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> quant: array<{quant_ty}>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> zero_points: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

struct Push {{
    k: u32, n: u32, n_blocks: u32, blob_words: u32,
    block_size: u32, zp_row_bytes: u32, gx: u32, pad: u32,
}}
var<immediate> pc: Push;

const LANES = {lanes}u;
const COLS = {cols}u;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {{
    let group = wid.y * pc.gx + wid.x;
    let col = group * COLS + tid / LANES;
    let lane = tid % LANES;
    let row = wid.z;
    var acc = 0.0;
    if (col < pc.n) {{
        let words = pc.n_blocks * pc.blob_words;
        {base_expr}
        let a_base = row * pc.k;
        {loop_body}
    }}
{reduction}
}}
"#
    )
    .replace("quant[v]", &format!("quant[{quant_index}v]"))
}

/// Label of a load form: `vec0` is the per-block one, and reads as `block`.
fn kind(vec: u32) -> String {
    match vec {
        0 => "block".to_string(),
        2 => "block4".to_string(),
        v => format!("vec{v}"),
    }
}

fn pseudo_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        })
        .collect()
}

fn pseudo_u32(n: usize, seed: u64) -> Vec<u32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 32) as u32
        })
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn floats(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Grid `x` extent: `N` reaches 262144, past the guaranteed 65535 per axis.
fn fold(groups: u32) -> ([u32; 3], u32) {
    let gx = groups.clamp(1, 32768);
    ([gx, groups.div_ceil(gx), 1], gx)
}

fn push(k: usize, n: usize, n_blocks: usize, zp_row_bytes: usize, gx: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(base::PUSH_BYTES as usize);
    for v in [
        k as u32,
        n as u32,
        n_blocks as u32,
        BLOB_WORDS as u32,
        BLOCK_SIZE as u32,
        zp_row_bytes as u32,
        gx,
        0,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

type Err = Box<dyn std::error::Error>;

/// Benches one geometry; returns `(baseline ms, per-variant ms)`.
fn geometry(ctx: &VkContext, k: usize, n: usize, label: &str) -> Result<(f64, Vec<f64>), Err> {
    let n_blocks = k.div_ceil(BLOCK_SIZE);
    let words = n_blocks * BLOB_WORDS;
    let zp_row_bytes = n_blocks.div_ceil(2);
    let zp_words = (n * zp_row_bytes).div_ceil(4);

    let a = pseudo_f32(k, 3);
    let quant = pseudo_u32(n * words, 7);
    let scales = pseudo_f32(n * n_blocks, 11);
    let zero_points = pseudo_u32(zp_words, 13);

    let a_buf = ctx.create_storage_buffer((4 * k) as u64)?;
    let q_buf = ctx.create_storage_buffer((4 * n * words) as u64)?;
    let s_buf = ctx.create_storage_buffer((4 * n * n_blocks) as u64)?;
    let z_buf = ctx.create_storage_buffer((4 * zp_words) as u64)?;
    let out_ref = ctx.create_storage_buffer((4 * n) as u64)?;
    let out_got = ctx.create_storage_buffer((4 * n) as u64)?;
    ctx.stream_upload(&a_buf, &f32_bytes(&a))?;
    ctx.stream_upload(&q_buf, &u32_bytes(&quant))?;
    ctx.stream_upload(&s_buf, &f32_bytes(&scales))?;
    ctx.stream_upload(&z_buf, &u32_bytes(&zero_points))?;
    ctx.flush()?;

    // what the kernel must move: packed weight, scales, zero points. `a` is K
    // floats re-read from cache, and the output is N floats — both negligible
    // next to the weight, which is the point of the format.
    let traffic = (4 * n * words + 4 * n * n_blocks + n * zp_row_bytes) as f64;
    let flops = (2 * n * k) as f64;

    let bench = |run: &dyn Fn(u32) -> Result<f64, Err>| -> Result<f64, Err> {
        run(2)?;
        (0..3).try_fold(f64::MAX, |acc, _| run(10).map(|s| acc.min(s)))
    };

    // baseline: the kernel in `shaders::matmul_nbits`, one workgroup of 64 per
    // output element
    let pipe = ctx.create_pipeline(
        &compile_wgsl(base::MATMUL_NBITS)?,
        base::BINDINGS,
        base::PUSH_BYTES,
    )?;
    let (grid, gx) = fold(n as u32);
    let pc = push(k, n, n_blocks, zp_row_bytes, gx);
    let tb = bench(&|reps| {
        let t = Instant::now();
        for _ in 0..reps {
            ctx.stream_dispatch(&pipe, &[&a_buf, &q_buf, &s_buf, &z_buf, &out_ref], &pc, grid)?;
        }
        ctx.flush()?;
        Ok(t.elapsed().as_secs_f64() / reps as f64)
    })?;
    let expect = floats(&ctx.stream_download(&out_ref, 4 * n)?);

    println!(
        "\n{:>20} {:>7} {:>8.3} {:>7.0} {:>7.2} {:>8} {:>9}",
        format!("{label} k{k}xn{n}"),
        grid[0] * grid[1],
        tb * 1e3,
        traffic / tb / 1e9,
        flops / tb / 1e12,
        "1.00×",
        "(WG64)"
    );

    let mut times = Vec::with_capacity(VARIANTS.len());
    for &(lanes, vec) in VARIANTS {
        // a lane count above the words of a column leaves lanes idle and the
        // vec4 form needs whole quads per lane
        let unit = match vec {
            // the block form walks blocks, the quad form quads, the word form words
            0 | 2 | 4 => words / 4,
            _ => words,
        };
        if lanes as usize > unit {
            times.push(f64::NAN);
            continue;
        }
        let cols = 256 / lanes;
        let pipe = ctx.create_pipeline(
            &compile_wgsl(&candidate(lanes, vec))?,
            base::BINDINGS,
            base::PUSH_BYTES,
        )?;
        let (grid, gx) = fold((n as u32).div_ceil(cols));
        let pc = push(k, n, n_blocks, zp_row_bytes, gx);
        let t = bench(&|reps| {
            let t = Instant::now();
            for _ in 0..reps {
                ctx.stream_dispatch(&pipe, &[&a_buf, &q_buf, &s_buf, &z_buf, &out_got], &pc, grid)?;
            }
            ctx.flush()?;
            Ok(t.elapsed().as_secs_f64() / reps as f64)
        })?;
        let got = floats(&ctx.stream_download(&out_got, 4 * n)?);
        let rel = expect
            .iter()
            .zip(&got)
            .fold(0.0f32, |m, (e, g)| m.max((e - g).abs() / e.abs().max(1e-3)));

        println!(
            "{:>20} {:>7} {:>8.3} {:>7.0} {:>7.2} {:>7.2}× {rel:>9.1e}",
            format!("lanes{lanes} {}", kind(vec)),
            grid[0] * grid[1],
            t * 1e3,
            traffic / t / 1e9,
            flops / t / 1e12,
            tb / t,
        );
        times.push(t);
    }
    // the buffers are big (150 MB for an lm_head): release before the next
    // geometry rather than at the end of main
    for buf in [a_buf, q_buf, s_buf, z_buf, out_ref, out_got] {
        ctx.destroy_buffer(buf);
    }
    Ok((tb, times))
}

fn totals(name: &str, rows: &[(f64, Vec<f64>, usize)]) {
    let base: f64 = rows.iter().map(|(tb, _, c)| tb * 1e3 * *c as f64).sum();
    println!("\n===== {name} =====");
    println!("{:>20} {:>9} {:>8}", "variant", "ms", "speedup");
    println!("{:>20} {:>9.3} {:>8}", "WG64 (current)", base, "1.00×");
    for (i, &(lanes, vec)) in VARIANTS.iter().enumerate() {
        let mut sum = 0.0;
        let mut complete = true;
        for (_, times, count) in rows {
            match times.get(i) {
                Some(t) if t.is_finite() => sum += t * 1e3 * *count as f64,
                _ => complete = false,
            }
        }
        if complete {
            println!(
                "{:>20} {:>9.3} {:>7.2}×",
                format!("lanes{lanes} {}", kind(vec)),
                sum,
                base / sum
            );
        }
    }
}

fn main() -> Result<(), Err> {
    let ctx = VkContext::new()?;
    println!(
        "{:>20} {:>7} {:>8} {:>7} {:>7} {:>8} {:>9}",
        "shape / variant", "WGs", "ms", "GB/s", "TFLOP/s", "speedup", "max|rel|"
    );

    let mut rows = Vec::new();
    for &(k, n, count, label) in SHAPES {
        let (tb, times) = geometry(&ctx, k, n, label)?;
        rows.push((tb, times, count));
    }
    totals("weighted by the projections a decode step runs", &rows);

    let mut heads = Vec::new();
    for &(k, n, count, label) in LM_HEADS {
        let (tb, times) = geometry(&ctx, k, n, label)?;
        heads.push((tb, times, count));
    }
    totals("lm_head only (one node per token, 150+ MB)", &heads);
    Ok(())
}
