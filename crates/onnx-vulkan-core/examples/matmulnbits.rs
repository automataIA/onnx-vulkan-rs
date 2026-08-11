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
use onnx_vulkan_core::shaders::matmul_nbits::{
    CandidateConfig as Cfg, CandidateLoad, CandidateReduction,
};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
use std::time::Instant;
use vk_compute::{ComputePipeline, GpuBuffer, VkContext, compile_wgsl};

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
const WARMUP_DISPATCHES: u32 = 2;
const MEASURED_SAMPLES: u32 = 20;
const DISPATCHES_PER_SAMPLE: u32 = 10;
const HISTORICAL_WALL_BATCHES: u32 = 3;
const HISTORICAL_WALL_DISPATCHES_PER_BATCH: u32 = 10;
const EQUIVALENCE_PERF_TOLERANCE: f64 = 0.10;
const MAX_REL_ERROR: f32 = 1e-4;
const REL_ERROR_FLOOR: f32 = 1.0;

/// Reduce the `lanes` partials of a column; lane 0 writes. Consecutive `tid` are
/// consecutive lanes of the same column, so the stride is 1.
///
/// `sub` swaps the shared-memory tree for a **subgroup butterfly**. The lanes of
/// a column are `lanes` contiguous `tid`, aligned to `lanes`, and `lanes` is a
/// power of two no larger than the subgroup — so a column never straddles a
/// subgroup and `subgroupShuffleDown(acc, s)` for `s < lanes` always reads
/// inside it. Lane 0 accumulates the total; the other lanes end up holding
/// partial garbage, which is why only lane 0 writes (as in the tree form).
///
/// Two things this removes: `log2(lanes)` `workgroupBarrier`s, and the `red[]`
/// array — so the workgroup allocates no shared memory at all. Two things it
/// does **not** remove: the loads, which is what a memory-bound kernel is
/// actually waiting on.
///
/// The shifts are unrolled at generation time rather than looped, so the shuffle
/// delta is a literal and the call sits in unambiguously uniform control flow.
///
/// `enable subgroups;` is deliberately absent: naga 30 rejects that directive
/// (wgpu#5555) while accepting the builtins themselves, because `compile_wgsl`
/// validates under `Capabilities::all()`. See `vk-compute example sg`.
fn reduction(lanes: u32, sub: bool) -> String {
    if lanes == 1 {
        return "    if (col < pc.n) { out[row * pc.n + col] = acc; }".to_string();
    }
    if sub {
        let mut out = String::new();
        let mut s = lanes / 2;
        while s > 0 {
            out.push_str(&format!(
                "    acc = acc + subgroupShuffleDown(acc, {s}u);\n"
            ));
            s /= 2;
        }
        out.push_str("    if (lane == 0u && col < pc.n) { out[row * pc.n + col] = acc; }");
        return out;
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

/// The candidate kernel, parameterized by reduction width, load width and
/// workgroup size.
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
///
/// `wg` is the workgroup size, and it is the axis the original default
/// candidates never had: it was
/// hardcoded at 256 for every measurement in `cronologia.md` 2026-07-30. It
/// decides how many columns share a workgroup (`cols = wg / lanes`), so at a
/// fixed `lanes` it trades workgroup count against occupancy per workgroup.
/// Only `--sweep` and `--trial` reach it; the default table stays at 256.
///
/// `sub` picks the reduction: see [`reduction`].
fn candidate(
    lanes: u32,
    load: CandidateLoad,
    wg: u32,
    reduction_kind: CandidateReduction,
) -> String {
    let vec = load.code();
    let sub = reduction_kind.uses_subgroup();
    let cols = wg / lanes;
    // the subgroup form keeps its partials in registers, so the workgroup
    // allocates no shared memory at all
    let red = if sub || lanes == 1 {
        String::new()
    } else {
        format!("var<workgroup> red: array<f32, {wg}>;")
    };
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
        let reduction = reduction(lanes, sub);
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
{red}

@compute @workgroup_size({wg})
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
    let reduction = reduction(lanes, sub);
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
{red}

@compute @workgroup_size({wg})
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

/// One validated configuration summarized from raw Vulkan GPU timestamps.
#[derive(Clone, Copy)]
struct Measure {
    wgs: u32,
    samples: u32,
    ms: f64,
    min_ms: f64,
    max_ms: f64,
    gb_s: f64,
    tflops: f64,
    max_rel: f32,
}

struct RunResult {
    correctness_passed: bool,
    max_rel: f32,
    rejection: Option<&'static str>,
    comparison: onnx_vulkan_core::comparison::ComparisonReport,
    measurement: Option<Measure>,
    samples_ns: Vec<u64>,
}

/// Diagnostic reproduction of the pre-autotune harness. The historical sweep
/// selected the minimum of three wall-clock batches, each containing ten
/// dispatches followed by one flush. This is intentionally not a tuner score:
/// it includes host recording, submission, fence wake-up, and scheduler noise.
#[derive(Clone)]
struct WallMeasure {
    batches: u32,
    dispatches_per_batch: u32,
    selected_ms: f64,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    samples_ns: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShippedFamily {
    Row,
    Wide,
}

#[derive(Clone, Copy, Debug)]
struct RouteAssessment {
    selected_ms: f64,
    best_ms: f64,
    slowdown: f64,
    within_tolerance: bool,
}

impl ShippedFamily {
    const fn label(self) -> &'static str {
        match self {
            Self::Row => "row",
            Self::Wide => "wide",
        }
    }
}

const fn row_equivalent_config() -> Cfg {
    Cfg::new(
        base::WG,
        CandidateLoad::Word,
        base::WG,
        CandidateReduction::WorkgroupTree,
    )
}

const fn wide_equivalent_config() -> Cfg {
    Cfg::new(
        base::WIDE_LANES,
        CandidateLoad::BlockFourAccumulators,
        256,
        CandidateReduction::WorkgroupTree,
    )
}

struct Correctness {
    passed: bool,
    max_rel: f32,
    rejection: Option<&'static str>,
    report: onnx_vulkan_core::comparison::ComparisonReport,
}

/// The baseline row: `shaders::matmul_nbits` as it ships, one workgroup of 64
/// per output element. It is also the correctness oracle — `max_rel` is
/// measured against its output, not against a second reference.
type Base = Measure;

struct WorkloadBuffers {
    a: GpuBuffer,
    quant: GpuBuffer,
    scales: GpuBuffer,
    zero_points: GpuBuffer,
    reference: GpuBuffer,
    scratch: GpuBuffer,
}

struct PreparedCandidate {
    config: Cfg,
    pipeline: ComputePipeline,
    grid: [u32; 3],
    push_constants: Vec<u8>,
}

/// One resident geometry shared by the legacy sweep and the JSONL runner.
///
/// Construction is workload preparation: inputs and weights are uploaded once,
/// and the shipped kernel produces the reference output once. `prepare` then
/// compiles only the requested tactic, `run` measures it against the resident
/// buffers, and `result` exposes the most recent measurement.
struct PreparedGeometry<'ctx> {
    ctx: &'ctx VkContext,
    k: usize,
    n: usize,
    words: usize,
    zp_row_bytes: usize,
    traffic: f64,
    flops: f64,
    buffers: Option<WorkloadBuffers>,
    baseline_pipeline: Option<ComputePipeline>,
    baseline: Base,
    expected: Vec<f32>,
    candidate: Option<PreparedCandidate>,
    last_result: Option<RunResult>,
}

fn warm_up_pipeline(
    ctx: &VkContext,
    pipeline: &ComputePipeline,
    buffers: &[&GpuBuffer],
    push_constants: &[u8],
    grid: [u32; 3],
) -> Result<(), Err> {
    for _ in 0..WARMUP_DISPATCHES {
        ctx.stream_dispatch(pipeline, buffers, push_constants, grid)?;
    }
    ctx.flush()?;
    Ok(())
}

fn summarize_samples(
    samples_ns: Vec<u64>,
    wgs: u32,
    traffic: f64,
    flops: f64,
    max_rel: f32,
) -> Result<(Measure, Vec<u64>), Err> {
    if samples_ns.len() < MEASURED_SAMPLES as usize {
        return Err(format!(
            "expected at least {MEASURED_SAMPLES} GPU samples, got {}",
            samples_ns.len()
        )
        .into());
    }
    let mut ordered = samples_ns.clone();
    ordered.sort_unstable();
    let middle = ordered.len() / 2;
    let median_ns = if ordered.len().is_multiple_of(2) {
        ordered[middle - 1] + (ordered[middle] - ordered[middle - 1]) / 2
    } else {
        ordered[middle]
    };
    let min_ns = ordered[0];
    let max_ns = ordered[ordered.len() - 1];
    let seconds = median_ns as f64 / 1e9;
    let measurement = Measure {
        wgs,
        samples: samples_ns.len() as u32,
        ms: median_ns as f64 / 1e6,
        min_ms: min_ns as f64 / 1e6,
        max_ms: max_ns as f64 / 1e6,
        gb_s: traffic / seconds / 1e9,
        tflops: flops / seconds / 1e12,
        max_rel,
    };
    Ok((measurement, samples_ns))
}

fn summarize_wall_samples(samples_ns: Vec<u64>) -> Result<WallMeasure, Err> {
    if samples_ns.len() != HISTORICAL_WALL_BATCHES as usize {
        return Err(format!(
            "expected {HISTORICAL_WALL_BATCHES} historical wall batches, got {}",
            samples_ns.len()
        )
        .into());
    }
    if samples_ns.contains(&0) {
        return Err("historical wall measurement produced a zero-duration sample".into());
    }
    let mut ordered = samples_ns.clone();
    ordered.sort_unstable();
    let median_ns = ordered[ordered.len() / 2];
    Ok(WallMeasure {
        batches: samples_ns.len() as u32,
        dispatches_per_batch: HISTORICAL_WALL_DISPATCHES_PER_BATCH,
        selected_ms: ordered[0] as f64 / 1e6,
        median_ms: median_ns as f64 / 1e6,
        min_ms: ordered[0] as f64 / 1e6,
        max_ms: ordered[ordered.len() - 1] as f64 / 1e6,
        samples_ns,
    })
}

fn measure_dispatch_wall(
    ctx: &VkContext,
    pipeline: &ComputePipeline,
    buffers: &[&GpuBuffer],
    push_constants: &[u8],
    grid: [u32; 3],
) -> Result<WallMeasure, Err> {
    warm_up_pipeline(ctx, pipeline, buffers, push_constants, grid)?;
    let mut samples_ns = Vec::with_capacity(HISTORICAL_WALL_BATCHES as usize);
    for _ in 0..HISTORICAL_WALL_BATCHES {
        let started = Instant::now();
        for _ in 0..HISTORICAL_WALL_DISPATCHES_PER_BATCH {
            ctx.stream_dispatch(pipeline, buffers, push_constants, grid)?;
        }
        ctx.flush()?;
        let batch_ns: u64 = started
            .elapsed()
            .as_nanos()
            .try_into()
            .map_err(|_| "historical wall duration exceeds u64 nanoseconds")?;
        samples_ns.push(batch_ns / u64::from(HISTORICAL_WALL_DISPATCHES_PER_BATCH));
    }
    summarize_wall_samples(samples_ns)
}

fn compare_outputs(expected: &[f32], got: &[f32]) -> Correctness {
    let report = onnx_vulkan_core::comparison::compare_f32(
        expected,
        got,
        onnx_vulkan_core::comparison::FloatTolerance::new(0.0, f64::from(MAX_REL_ERROR))
            .with_relative_floor(f64::from(REL_ERROR_FLOOR)),
    );
    let passed = report.passed();
    let max_rel = report.max_relative.min(f64::from(f32::MAX)) as f32;
    let rejection = if expected.len() != got.len() {
        Some("output length mismatch")
    } else if report.nan_mismatches > 0 {
        Some("NaN classification changed")
    } else if report.infinity_mismatches > 0 {
        Some("infinity classification or sign changed")
    } else if !passed {
        Some("relative error exceeds 1e-4")
    } else {
        None
    };
    Correctness {
        passed,
        max_rel,
        rejection,
        report,
    }
}

impl<'ctx> PreparedGeometry<'ctx> {
    fn new(ctx: &'ctx VkContext, k: usize, n: usize) -> Result<Self, Err> {
        let n_blocks = k.div_ceil(BLOCK_SIZE);
        let words = n_blocks * BLOB_WORDS;
        let zp_row_bytes = n_blocks.div_ceil(2);
        let zp_words = (n * zp_row_bytes).div_ceil(4);

        let a = pseudo_f32(k, 3);
        let quant = pseudo_u32(n * words, 7);
        let scales = pseudo_f32(n * n_blocks, 11);
        let zero_points = pseudo_u32(zp_words, 13);

        let buffers = WorkloadBuffers {
            a: ctx.create_storage_buffer((4 * k) as u64)?,
            quant: ctx.create_storage_buffer((4 * n * words) as u64)?,
            scales: ctx.create_storage_buffer((4 * n * n_blocks) as u64)?,
            zero_points: ctx.create_storage_buffer((4 * zp_words) as u64)?,
            reference: ctx.create_storage_buffer((4 * n) as u64)?,
            scratch: ctx.create_storage_buffer((4 * n) as u64)?,
        };
        ctx.stream_upload(&buffers.a, &f32_bytes(&a))?;
        ctx.stream_upload(&buffers.quant, &u32_bytes(&quant))?;
        ctx.stream_upload(&buffers.scales, &f32_bytes(&scales))?;
        ctx.stream_upload(&buffers.zero_points, &u32_bytes(&zero_points))?;
        ctx.flush()?;

        // what the kernel must move: packed weight, scales, zero points. `a` is K
        // floats re-read from cache, and the output is N floats — both negligible
        // next to the weight, which is the point of the format.
        let traffic = (4 * n * words + 4 * n * n_blocks + n * zp_row_bytes) as f64;
        let flops = (2 * n * k) as f64;

        // baseline: the kernel in `shaders::matmul_nbits`, one workgroup of 64 per
        // output element
        let baseline_pipeline = ctx.create_pipeline(
            &compile_wgsl(base::MATMUL_NBITS)?,
            base::BINDINGS,
            base::PUSH_BYTES,
        )?;
        let (grid, gx) = fold(n as u32);
        let push_constants = push(k, n, n_blocks, zp_row_bytes, gx);
        let baseline_buffers = [
            &buffers.a,
            &buffers.quant,
            &buffers.scales,
            &buffers.zero_points,
            &buffers.reference,
        ];
        warm_up_pipeline(
            ctx,
            &baseline_pipeline,
            &baseline_buffers,
            &push_constants,
            grid,
        )?;
        let baseline_samples = ctx.measure_dispatch_gpu(
            &baseline_pipeline,
            &baseline_buffers,
            &push_constants,
            grid,
            MEASURED_SAMPLES,
            DISPATCHES_PER_SAMPLE,
        )?;
        let expected = floats(&ctx.stream_download(&buffers.reference, 4 * n)?);
        let (baseline, _) =
            summarize_samples(baseline_samples, grid[0] * grid[1], traffic, flops, 0.0)?;

        Ok(Self {
            ctx,
            k,
            n,
            words,
            zp_row_bytes,
            traffic,
            flops,
            buffers: Some(buffers),
            baseline_pipeline: Some(baseline_pipeline),
            baseline,
            expected,
            candidate: None,
            last_result: None,
        })
    }

    fn baseline(&self) -> Base {
        self.baseline
    }

    fn resident_bytes(&self) -> u64 {
        self.buffers.as_ref().map_or(0, |buffers| {
            buffers.a.size
                + buffers.quant.size
                + buffers.scales.size
                + buffers.zero_points.size
                + buffers.reference.size
                + buffers.scratch.size
        })
    }

    fn execute_pipeline(
        &self,
        pipeline: &ComputePipeline,
        grid: [u32; 3],
        push_constants: &[u8],
    ) -> Result<RunResult, Err> {
        let buffers = self
            .buffers
            .as_ref()
            .ok_or("resident workload resources are unavailable")?;
        let dispatch_buffers = [
            &buffers.a,
            &buffers.quant,
            &buffers.scales,
            &buffers.zero_points,
            &buffers.scratch,
        ];
        warm_up_pipeline(self.ctx, pipeline, &dispatch_buffers, push_constants, grid)?;
        let got = floats(&self.ctx.stream_download(&buffers.scratch, 4 * self.n)?);
        let correctness = compare_outputs(&self.expected, &got);
        if !correctness.passed {
            return Ok(RunResult {
                correctness_passed: false,
                max_rel: correctness.max_rel,
                rejection: correctness.rejection,
                comparison: correctness.report,
                measurement: None,
                samples_ns: Vec::new(),
            });
        }

        let samples_ns = self.ctx.measure_dispatch_gpu(
            pipeline,
            &dispatch_buffers,
            push_constants,
            grid,
            MEASURED_SAMPLES,
            DISPATCHES_PER_SAMPLE,
        )?;
        let (measurement, samples_ns) = summarize_samples(
            samples_ns,
            grid[0] * grid[1],
            self.traffic,
            self.flops,
            correctness.max_rel,
        )?;
        Ok(RunResult {
            correctness_passed: true,
            max_rel: correctness.max_rel,
            rejection: None,
            comparison: correctness.report,
            measurement: Some(measurement),
            samples_ns,
        })
    }

    fn measure_pipeline_wall(
        &self,
        pipeline: &ComputePipeline,
        grid: [u32; 3],
        push_constants: &[u8],
    ) -> Result<WallMeasure, Err> {
        let buffers = self
            .buffers
            .as_ref()
            .ok_or("resident workload resources are unavailable")?;
        measure_dispatch_wall(
            self.ctx,
            pipeline,
            &[
                &buffers.a,
                &buffers.quant,
                &buffers.scales,
                &buffers.zero_points,
                &buffers.scratch,
            ],
            push_constants,
            grid,
        )
    }

    fn measure_shipped_row_wall(&self) -> Result<WallMeasure, Err> {
        let pipeline = self
            .baseline_pipeline
            .as_ref()
            .ok_or("shipped row pipeline is unavailable")?;
        let (grid, gx) = fold(self.n as u32);
        let push_constants = push(
            self.k,
            self.n,
            self.k.div_ceil(BLOCK_SIZE),
            self.zp_row_bytes,
            gx,
        );
        self.measure_pipeline_wall(pipeline, grid, &push_constants)
    }

    /// Measure the exact wide-column source used by production, independently
    /// of the generated candidate that is expected to represent the same family.
    fn measure_shipped_wide(&self) -> Result<(RunResult, WallMeasure), Err> {
        let lanes = base::WIDE_LANES;
        let pipeline = self.ctx.create_pipeline(
            &compile_wgsl(&base::wide_source(lanes))?,
            base::BINDINGS,
            base::PUSH_BYTES,
        )?;
        let groups = (self.n as u32).div_ceil(256 / lanes);
        let (grid, gx) = fold(groups);
        let push_constants = push(
            self.k,
            self.n,
            self.k.div_ceil(BLOCK_SIZE),
            self.zp_row_bytes,
            gx,
        );
        let result = (|| {
            let timestamp = self.execute_pipeline(&pipeline, grid, &push_constants)?;
            let wall = self.measure_pipeline_wall(&pipeline, grid, &push_constants)?;
            Ok((timestamp, wall))
        })();
        self.ctx.destroy_pipeline(pipeline);
        result
    }

    fn measure_prepared_wall(&self) -> Result<WallMeasure, Err> {
        let prepared = self
            .candidate
            .as_ref()
            .ok_or("wall measurement requires a viable prepared candidate")?;
        self.measure_pipeline_wall(&prepared.pipeline, prepared.grid, &prepared.push_constants)
    }

    fn prepare(&mut self, config: Cfg) -> Result<bool, Err> {
        self.prepare_with_fault(config, false)
    }

    fn prepare_with_fault(&mut self, config: Cfg, inject_add_one: bool) -> Result<bool, Err> {
        if let Some(previous) = self.candidate.take() {
            self.ctx.destroy_pipeline(previous.pipeline);
        }
        self.last_result = None;
        if !config.is_viable(self.words, self.ctx.subgroup_size) {
            return Ok(false);
        }

        let cols = config.workgroup_size / config.lanes;
        let mut source = candidate(
            config.lanes,
            config.load,
            config.workgroup_size,
            config.reduction,
        );
        if inject_add_one {
            let original = source.clone();
            source = source
                .replace(" = acc; }", " = acc + 1.0; }")
                .replace(" = red[tid]; }", " = red[tid] + 1.0; }");
            if source == original {
                return Err("diagnostic fault injection found no output assignment".into());
            }
        }
        let pipeline =
            self.ctx
                .create_pipeline(&compile_wgsl(&source)?, base::BINDINGS, base::PUSH_BYTES)?;
        let (grid, gx) = fold((self.n as u32).div_ceil(cols));
        let n_blocks = self.k.div_ceil(BLOCK_SIZE);
        self.candidate = Some(PreparedCandidate {
            config,
            pipeline,
            grid,
            push_constants: push(self.k, self.n, n_blocks, self.zp_row_bytes, gx),
        });
        Ok(true)
    }

    fn run(&mut self) -> Result<(), Err> {
        let prepared = self
            .candidate
            .as_ref()
            .ok_or("run requires a viable prepared candidate")?;
        self.last_result = Some(self.execute_pipeline(
            &prepared.pipeline,
            prepared.grid,
            &prepared.push_constants,
        )?);
        Ok(())
    }

    fn result(&self) -> Option<(Cfg, &RunResult)> {
        Some((self.candidate.as_ref()?.config, self.last_result.as_ref()?))
    }
}

impl Drop for PreparedGeometry<'_> {
    fn drop(&mut self) {
        if let Some(candidate) = self.candidate.take() {
            self.ctx.destroy_pipeline(candidate.pipeline);
        }
        if let Some(pipeline) = self.baseline_pipeline.take() {
            self.ctx.destroy_pipeline(pipeline);
        }
        if let Some(buffers) = self.buffers.take() {
            for buffer in [
                buffers.a,
                buffers.quant,
                buffers.scales,
                buffers.zero_points,
                buffers.reference,
                buffers.scratch,
            ] {
                self.ctx.destroy_buffer(buffer);
            }
        }
    }
}

/// Benches one geometry over `configs`; returns the baseline and one entry per
/// config, `None` where viability refused it. The lifecycle is the same one the
/// persistent JSONL runner exposes, so both paths share preparation semantics.
fn geometry(
    ctx: &VkContext,
    k: usize,
    n: usize,
    configs: &[Cfg],
) -> Result<(Base, Vec<Option<Measure>>), Err> {
    let mut prepared = PreparedGeometry::new(ctx, k, n)?;
    let baseline = prepared.baseline();
    let rows = measure_configs(&mut prepared, configs)?;
    Ok((baseline, rows))
}

fn measure_configs(
    prepared: &mut PreparedGeometry<'_>,
    configs: &[Cfg],
) -> Result<Vec<Option<Measure>>, Err> {
    let mut rows = Vec::with_capacity(configs.len());
    for &config in configs {
        if !prepared.prepare(config)? {
            rows.push(None);
            continue;
        }
        prepared.run()?;
        let row = prepared.result().and_then(|(_, result)| result.measurement);
        if row.is_none()
            && let Some((_, result)) = prepared.result()
        {
            eprintln!(
                "candidate rejected before timing: {} (max_rel={:.3e})",
                result.rejection.unwrap_or("unknown correctness failure"),
                result.max_rel,
            );
        }
        rows.push(row);
    }
    Ok(rows)
}

/// The legacy table layout: the baseline row then one row per config. Timing is
/// now the median of 20 Vulkan GPU timestamp samples rather than batch wall.
fn print_geometry(
    label: &str,
    k: usize,
    n: usize,
    configs: &[Cfg],
    base: &Base,
    rows: &[Option<Measure>],
) {
    println!(
        "\n{:>20} {:>7} {:>8.3} {:>7.0} {:>7.2} {:>8} {:>9}",
        format!("{label} k{k}xn{n}"),
        base.wgs,
        base.ms,
        base.gb_s,
        base.tflops,
        "1.00×",
        "(WG64)"
    );
    for (&cfg, m) in configs.iter().zip(rows) {
        let Some(m) = m else { continue };
        println!(
            "{:>20} {:>7} {:>8.3} {:>7.0} {:>7.2} {:>7.2}× {:>9.1e}",
            format!("lanes{} {}", cfg.lanes, cfg.load.label()),
            m.wgs,
            m.ms,
            m.gb_s,
            m.tflops,
            base.ms / m.ms,
            m.max_rel,
        );
    }
}

fn totals(name: &str, configs: &[Cfg], rows: &[(Base, Vec<Option<Measure>>, usize)]) {
    let base: f64 = rows.iter().map(|(b, _, c)| b.ms * *c as f64).sum();
    println!("\n===== {name} =====");
    println!("{:>20} {:>9} {:>8}", "variant", "ms", "speedup");
    println!("{:>20} {:>9.3} {:>8}", "WG64 (current)", base, "1.00×");
    for (i, &cfg) in configs.iter().enumerate() {
        let mut sum = 0.0;
        let mut complete = true;
        for (_, ms, count) in rows {
            match ms.get(i) {
                Some(Some(m)) => sum += m.ms * *count as f64,
                _ => complete = false,
            }
        }
        if complete {
            println!(
                "{:>20} {:>9.3} {:>7.2}×",
                format!("lanes{} {}", cfg.lanes, cfg.load.label()),
                sum,
                base / sum
            );
        }
    }
}

/// The 18 shipped variants at the workgroup size they were measured at.
fn default_configs() -> Vec<Cfg> {
    base::DEFAULT_CANDIDATES.to_vec()
}

/// The widened space `--sweep` and `--trial` share: `lanes × vec × wg × sub`.
///
/// `wg` never appeared in the default candidates — every number in `cronologia.md`
/// 2026-07-30 was taken at 256 — and it is the axis that makes the space large
/// enough for a sampler to be worth comparing against enumeration. `sub` is the
/// subgroup reduction, carried as its own axis on purpose: the question is not
/// whether the best subgroup config beats the shipped kernel, it is whether the
/// subgroup form beats the tree **at the same `lanes` and `wg`**, and that is
/// only answerable if both are measured at every point.
fn wide_space() -> Vec<Cfg> {
    base::candidate_space()
}

fn ranked_candidates(configs: &[Cfg], rows: &[Option<Measure>]) -> Vec<(Cfg, Measure)> {
    let mut ranked = configs
        .iter()
        .zip(rows)
        .filter_map(|(config, measurement)| measurement.map(|m| (*config, m)))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| a.1.ms.total_cmp(&b.1.ms));
    ranked
}

/// `--geom K,N`, defaulting to gemma3's most frequent projection.
fn geom() -> Result<(usize, usize), Err> {
    let Some(spec) = arg_value("--geom") else {
        return Ok((1152, 6912));
    };
    let (k, n) = spec.split_once(',').ok_or("--geom wants K,N")?;
    Ok((k.trim().parse()?, n.trim().parse()?))
}

fn arg_value(name: &str) -> Option<String> {
    let mut it = std::env::args().skip_while(|a| a != name);
    it.next()?;
    it.next()
}

fn candidates_json() -> Value {
    let candidates = base::candidate_space()
        .into_iter()
        .map(|config| {
            json!({
                "lanes": config.lanes,
                "vec": config.load.code(),
                "wg": config.workgroup_size,
                "sub": config.reduction.code(),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "schema_version": 1,
        "family": "matmul_nbits",
        "axes": {
            "lanes": base::CANDIDATE_LANES,
            "vec": base::CANDIDATE_LOADS.iter().map(|load| load.code()).collect::<Vec<_>>(),
            "wg": base::CANDIDATE_WORKGROUP_SIZES,
            "sub": base::CANDIDATE_REDUCTIONS
                .iter()
                .map(|reduction| reduction.code())
                .collect::<Vec<_>>(),
        },
        "candidate_count": base::candidate_space().len(),
        "candidates": candidates,
    })
}

/// Search-space metadata as one JSON object, without initializing Vulkan.
fn list_candidates() {
    let numbers = |values: &[u32]| {
        values
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    };
    let loads = base::CANDIDATE_LOADS
        .iter()
        .map(|load| load.code().to_string())
        .collect::<Vec<_>>()
        .join(",");
    let reductions = base::CANDIDATE_REDUCTIONS
        .iter()
        .map(|reduction| reduction.code().to_string())
        .collect::<Vec<_>>()
        .join(",");
    println!(
        r#"{{"schema_version":1,"family":"matmul_nbits","axes":{{"lanes":[{}],"vec":[{loads}],"wg":[{}],"sub":[{reductions}]}},"candidate_count":{}}}"#,
        numbers(base::CANDIDATE_LANES),
        numbers(base::CANDIDATE_WORKGROUP_SIZES),
        base::candidate_space().len(),
    );
}

fn request_u32(request: &Value, field: &str) -> Result<u32, Err> {
    let value = request
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{field} must be an unsigned integer"))?;
    Ok(value
        .try_into()
        .map_err(|_| format!("{field} exceeds u32"))?)
}

fn request_config(request: &Value) -> Result<Cfg, Err> {
    let lanes = request_u32(request, "lanes")?;
    let vec = request_u32(request, "vec")?;
    let load = CandidateLoad::from_code(vec).ok_or("vec must be one of 0,1,2,4")?;
    let workgroup_size = request_u32(request, "wg")?;
    let reduction = match request_u32(request, "sub")? {
        0 => CandidateReduction::WorkgroupTree,
        1 => CandidateReduction::SubgroupShuffle,
        _ => return Err("sub must be 0 or 1".into()),
    };
    Ok(Cfg::new(lanes, load, workgroup_size, reduction))
}

fn result_json(prepared: &PreparedGeometry<'_>) -> Result<Value, Err> {
    let (config, result) = prepared.result().ok_or("result requires a completed run")?;
    Ok(result_payload(
        config,
        prepared.k,
        prepared.n,
        prepared.baseline.ms,
        result,
    ))
}

fn result_payload(config: Cfg, k: usize, n: usize, baseline_ms: f64, result: &RunResult) -> Value {
    let mut response = json!({
        "schema_version": 1,
        "op": "result",
        "ok": true,
        "viable": true,
        "eligible": result.correctness_passed,
        "lanes": config.lanes,
        "vec": config.load.code(),
        "wg": config.workgroup_size,
        "sub": config.reduction.code(),
        "k": k,
        "n": n,
        "correctness": {
            "kind": "max_rel",
            "passed": result.correctness_passed,
            "threshold": MAX_REL_ERROR,
            "denominator_floor": REL_ERROR_FLOOR,
            "max_rel": result.max_rel,
            "rejection": result.rejection,
            "dtype": result.comparison.dtype,
            "reference_len": result.comparison.reference_len,
            "candidate_len": result.comparison.candidate_len,
            "mismatches": result.comparison.mismatches,
            "max_abs": result.comparison.max_abs,
            "matching_nan": result.comparison.matching_nan,
            "nan_mismatches": result.comparison.nan_mismatches,
            "matching_infinity": result.comparison.matching_infinity,
            "infinity_mismatches": result.comparison.infinity_mismatches,
        },
    });
    if let Some(measurement) = result.measurement
        && let Some(object) = response.as_object_mut()
    {
        object.insert("wgs".into(), json!(measurement.wgs));
        object.insert("samples".into(), json!(measurement.samples));
        object.insert("samples_ns".into(), json!(result.samples_ns));
        object.insert("ms".into(), json!(measurement.ms));
        object.insert("min_ms".into(), json!(measurement.min_ms));
        object.insert("max_ms".into(), json!(measurement.max_ms));
        object.insert("gb_s".into(), json!(measurement.gb_s));
        object.insert("tflops".into(), json!(measurement.tflops));
        object.insert("max_rel".into(), json!(measurement.max_rel));
        object.insert("speedup".into(), json!(baseline_ms / measurement.ms));
    }
    if !result.correctness_passed
        && let Some(object) = response.as_object_mut()
    {
        object.insert(
            "diagnostic".into(),
            rejection_diagnostic(config, k, n, result, "candidate"),
        );
    }
    response
}

fn rejection_diagnostic(
    config: Cfg,
    k: usize,
    n: usize,
    result: &RunResult,
    tactic_id: &str,
) -> Value {
    json!({
        "outcome": "numerical_failure",
        "stage": "correctness",
        "code": match result.rejection {
            Some("output length mismatch") => "length_mismatch",
            Some("NaN classification changed") => "nan_classification",
            Some("infinity classification or sign changed") => "infinity_classification",
            _ => "tolerance_exceeded",
        },
        "detail": result.rejection,
        "workload": {
            "domain": "com.microsoft",
            "op": "MatMulNBits",
            "inputs": [{"dtype":"float32","dims":[1,k]}],
            "outputs": [{"dtype":"float32","dims":[1,n]}],
            "attributes": {"bits":4,"block_size":BLOCK_SIZE},
        },
        "tactic": {
            "family":"matmul_nbits",
            "id": tactic_id,
            "parameters": {
                "lanes":config.lanes,
                "vec":config.load.code(),
                "wg":config.workgroup_size,
                "sub":config.reduction.code(),
            },
        },
        "comparison": {
            "dtype": result.comparison.dtype,
            "reference_len": result.comparison.reference_len,
            "candidate_len": result.comparison.candidate_len,
            "mismatches": result.comparison.mismatches,
            "max_abs": result.comparison.max_abs,
            "max_relative": result.comparison.max_relative,
            "nan_mismatches": result.comparison.nan_mismatches,
            "infinity_mismatches": result.comparison.infinity_mismatches,
        },
        "timed": false,
    })
}

fn handle_request(
    prepared: &mut PreparedGeometry<'_>,
    request: &Value,
) -> Result<(Value, bool), Err> {
    let operation = request
        .get("op")
        .and_then(Value::as_str)
        .ok_or("op must be a string")?;
    match operation {
        "list" => {
            let mut response = candidates_json();
            response["k"] = json!(prepared.k);
            response["n"] = json!(prepared.n);
            response["resident_bytes"] = json!(prepared.resident_bytes());
            Ok((response, false))
        }
        "prepare" => {
            let config = request_config(request)?;
            let viable = prepared.prepare(config)?;
            Ok((
                json!({
                    "schema_version": 1,
                    "op": "prepare",
                    "ok": true,
                    "viable": viable,
                    "lanes": config.lanes,
                    "vec": config.load.code(),
                    "wg": config.workgroup_size,
                    "sub": config.reduction.code(),
                    "k": prepared.k,
                    "n": prepared.n,
                }),
                false,
            ))
        }
        "run" => {
            prepared.run()?;
            let (_, result) = prepared
                .result()
                .ok_or("run completed without producing a result")?;
            Ok((
                json!({
                    "schema_version": 1,
                    "op": "run",
                    "ok": true,
                    "correctness_passed": result.correctness_passed,
                    "measured": result.measurement.is_some(),
                }),
                false,
            ))
        }
        "result" => Ok((result_json(prepared)?, false)),
        "compare" => {
            let config = request_config(request)?;
            let inject_add_one = match request.get("inject_fault").and_then(Value::as_str) {
                None => false,
                Some("add_one")
                    if std::env::var("ONNX_VULKAN_ALLOW_DIAGNOSTIC_FAULT").as_deref()
                        == Ok("1") =>
                {
                    true
                }
                Some("add_one") => {
                    return Err("inject_fault requires ONNX_VULKAN_ALLOW_DIAGNOSTIC_FAULT=1".into());
                }
                Some(other) => return Err(format!("unknown diagnostic fault {other:?}").into()),
            };
            if !prepared.prepare_with_fault(config, inject_add_one)? {
                return Ok((
                    json!({
                        "schema_version": 1,
                        "op": "compare",
                        "ok": true,
                        "viable": false,
                        "eligible": false,
                        "lanes": config.lanes,
                        "vec": config.load.code(),
                        "wg": config.workgroup_size,
                        "sub": config.reduction.code(),
                        "k": prepared.k,
                        "n": prepared.n,
                        "rejection": "static viability",
                        "diagnostic": {
                            "outcome":"static_viability_failure",
                            "stage":"prepare",
                            "code":"static_viability",
                            "detail":"candidate geometry or capability constraints are not satisfied",
                            "workload": {"domain":"com.microsoft","op":"MatMulNBits","k":prepared.k,"n":prepared.n},
                            "tactic": {"family":"matmul_nbits","id":"candidate","parameters":{"lanes":config.lanes,"vec":config.load.code(),"wg":config.workgroup_size,"sub":config.reduction.code()}},
                            "timed":false,
                        },
                    }),
                    false,
                ));
            }
            prepared.run()?;
            let mut response = result_json(prepared)?;
            response["op"] = json!("compare");
            if inject_add_one {
                let (_, result) = prepared
                    .result()
                    .ok_or("fault-injected comparison produced no result")?;
                if result.correctness_passed {
                    return Err("fault-injected tactic unexpectedly passed correctness".into());
                }
                response["diagnostic"] = rejection_diagnostic(
                    config,
                    prepared.k,
                    prepared.n,
                    result,
                    "diagnostic-add-one",
                );
            }
            Ok((response, false))
        }
        "shutdown" => Ok((
            json!({"schema_version": 1, "op": "shutdown", "ok": true}),
            true,
        )),
        _ => Err(format!("unknown operation {operation:?}").into()),
    }
}

/// Persistent JSON Lines protocol. Each input object receives exactly one
/// stdout object; human-readable diagnostics stay on stderr.
fn serve() -> Result<(), Err> {
    let (k, n) = geom()?;
    let ctx = VkContext::new()?;
    let mut prepared = PreparedGeometry::new(&ctx, k, n)?;
    eprintln!("matmul_nbits runner ready for k{k}xn{n}");

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line?;
        let (response, shutdown) = match serde_json::from_str::<Value>(&line) {
            Ok(request) => match handle_request(&mut prepared, &request) {
                Ok(response) => response,
                Err(error) => {
                    eprintln!("runner request failed: {error}");
                    (
                        json!({"schema_version": 1, "ok": false, "error": error.to_string()}),
                        false,
                    )
                }
            },
            Err(error) => {
                eprintln!("runner received invalid JSON: {error}");
                (
                    json!({"schema_version": 1, "ok": false, "error": error.to_string()}),
                    false,
                )
            }
        };
        writeln!(output, "{response}")?;
        output.flush()?;
        if shutdown {
            break;
        }
    }
    Ok(())
}

/// One config, one geometry, one JSON object on stdout. This is what an
/// external tuner calls, and it pays a process start, a Vulkan init and a full
/// weight upload for a single sample — which is the cost `--sweep` amortizes
/// and `docs/autotuning.md` compares against.
/// `--trial lanes,vec,wg[,sub]`; `sub` defaults to 0 (the tree reduction).
fn trial(spec: &str) -> Result<(), Err> {
    let mut it = spec.split(',').map(|f| f.trim().parse::<u32>());
    let want = "--trial wants lanes,vec,wg[,sub]";
    let lanes = it.next().ok_or(want)??;
    let vec = it.next().ok_or(want)??;
    let load = CandidateLoad::from_code(vec).ok_or("--trial vec must be one of 0,1,2,4")?;
    let wg = it.next().ok_or(want)??;
    let sub = it.next().transpose()?.unwrap_or(0) != 0;
    let reduction = if sub {
        CandidateReduction::SubgroupShuffle
    } else {
        CandidateReduction::WorkgroupTree
    };
    let config = Cfg::new(lanes, load, wg, reduction);
    let (k, n) = geom()?;

    let ctx = VkContext::new()?;
    let (base, rows) = geometry(&ctx, k, n, &[config])?;
    let sub = u32::from(sub);
    match rows[0] {
        Some(m) => println!(
            r#"{{"viable":true,"lanes":{lanes},"vec":{vec},"wg":{wg},"sub":{sub},"k":{k},"n":{n},"wgs":{},"ms":{:.6},"gb_s":{:.3},"tflops":{:.4},"max_rel":{:.3e},"speedup":{:.4}}}"#,
            m.wgs,
            m.ms,
            m.gb_s,
            m.tflops,
            m.max_rel,
            base.ms / m.ms,
        ),
        None => println!(
            r#"{{"viable":false,"lanes":{lanes},"vec":{vec},"wg":{wg},"sub":{sub},"k":{k},"n":{n}}}"#
        ),
    }
    Ok(())
}

/// The whole widened space, in one process, on one geometry. The control arm.
fn sweep() -> Result<(), Err> {
    let (k, n) = geom()?;
    let configs = wide_space();
    let ctx = VkContext::new()?;

    let t = Instant::now();
    let (base, rows) = geometry(&ctx, k, n, &configs)?;
    let wall = t.elapsed().as_secs_f64();

    let best = ranked_candidates(&configs, &rows);

    println!(
        "sweep k{k}xn{n}: {} configs, {} measured, {wall:.1} s ({:.3} s/config)",
        configs.len(),
        best.len(),
        wall / best.len().max(1) as f64,
    );
    println!(
        "{:>28} {:>7} {:>8} {:>7} {:>7} {:>8} {:>9}",
        "lanes/form/wg/red", "WGs", "ms", "GB/s", "TFLOP/s", "speedup", "max|rel|"
    );
    println!(
        "{:>28} {:>7} {:>8.3} {:>7.0} {:>7.2} {:>8} {:>9}",
        "WG64 (current)", base.wgs, base.ms, base.gb_s, base.tflops, "1.00×", ""
    );
    for (cfg, m) in best.iter().take(15) {
        println!(
            "{:>28} {:>7} {:>8.3} {:>7.0} {:>7.2} {:>7.2}× {:>9.1e}",
            format!(
                "lanes{} {} wg{} {}",
                cfg.lanes,
                cfg.load.label(),
                cfg.workgroup_size,
                cfg.reduction.label(),
            ),
            m.wgs,
            m.ms,
            m.gb_s,
            m.tflops,
            base.ms / m.ms,
            m.max_rel,
        );
    }

    // The comparison the `sub` axis exists for: the subgroup butterfly against
    // the shared-memory tree at the *same* lanes, form and workgroup. A ranking
    // of best-configs cannot answer this; a pairing can.
    let find = |want: Cfg| {
        configs
            .iter()
            .zip(&rows)
            .find(|(c, _)| **c == want)
            .and_then(|(_, m)| *m)
    };
    let mut pairs: Vec<(Cfg, f64, f64)> = Vec::new();
    for &cfg in &configs {
        if cfg.reduction != CandidateReduction::SubgroupShuffle {
            continue;
        }
        let tree_config = Cfg {
            reduction: CandidateReduction::WorkgroupTree,
            ..cfg
        };
        if let (Some(tree), Some(subgroup)) = (find(tree_config), find(cfg)) {
            pairs.push((cfg, tree.ms, subgroup.ms));
        }
    }
    pairs.sort_by(|a, b| (b.1 / b.2).total_cmp(&(a.1 / a.2)));
    if pairs.is_empty() {
        println!("\nno pair had both reductions viable — nothing to compare");
        return Ok(());
    }
    let wins = pairs.iter().filter(|(_, t, s)| s < t).count();
    let geo: f64 = pairs.iter().map(|(_, t, s)| (t / s).ln()).sum::<f64>() / pairs.len() as f64;
    println!(
        "\nsubgroup vs tree, same lanes/form/wg: {} pairs, subgroup faster in {wins}, \
         geometric mean {:.3}×",
        pairs.len(),
        geo.exp(),
    );
    println!(
        "{:>28} {:>9} {:>9} {:>8}",
        "pair", "tree ms", "sub ms", "sub/tree"
    );
    for (cfg, tree, s) in pairs.iter().take(5) {
        println!(
            "{:>28} {tree:>9.4} {s:>9.4} {:>7.2}×",
            format!(
                "lanes{} {} wg{}",
                cfg.lanes,
                cfg.load.label(),
                cfg.workgroup_size
            ),
            tree / s
        );
    }
    for (cfg, tree, s) in pairs.iter().rev().take(3) {
        println!(
            "{:>28} {tree:>9.4} {s:>9.4} {:>7.2}×",
            format!(
                "lanes{} {} wg{}",
                cfg.lanes,
                cfg.load.label(),
                cfg.workgroup_size
            ),
            tree / s
        );
    }
    Ok(())
}

fn config_payload(config: Cfg) -> Value {
    json!({
        "lanes": config.lanes,
        "vec": config.load.code(),
        "load": config.load.label(),
        "wg": config.workgroup_size,
        "sub": config.reduction.code(),
        "reduction": config.reduction.label(),
    })
}

fn measurement_payload(measurement: Measure) -> Value {
    json!({
        "samples": measurement.samples,
        "ms": measurement.ms,
        "min_ms": measurement.min_ms,
        "max_ms": measurement.max_ms,
        "gb_s": measurement.gb_s,
        "tflops": measurement.tflops,
        "max_rel": measurement.max_rel,
    })
}

fn wall_measurement_payload(measurement: &WallMeasure) -> Value {
    json!({
        "selection": "minimum",
        "batches": measurement.batches,
        "dispatches_per_batch": measurement.dispatches_per_batch,
        "selected_ms": measurement.selected_ms,
        "median_ms": measurement.median_ms,
        "min_ms": measurement.min_ms,
        "max_ms": measurement.max_ms,
        "samples_ns": measurement.samples_ns,
    })
}

fn selected_family(n: usize) -> ShippedFamily {
    match base::route(1, n) {
        base::Route::Row => ShippedFamily::Row,
        base::Route::Wide { .. } => ShippedFamily::Wide,
    }
}

fn faster_family(row_ms: f64, wide_ms: f64) -> ShippedFamily {
    if wide_ms < row_ms {
        ShippedFamily::Wide
    } else {
        ShippedFamily::Row
    }
}

fn assess_route(selected: ShippedFamily, row_ms: f64, wide_ms: f64) -> RouteAssessment {
    let selected_ms = match selected {
        ShippedFamily::Row => row_ms,
        ShippedFamily::Wide => wide_ms,
    };
    let best_ms = row_ms.min(wide_ms);
    let slowdown = selected_ms / best_ms;
    RouteAssessment {
        selected_ms,
        best_ms,
        slowdown,
        within_tolerance: selected_ms <= best_ms * (1.0 + EQUIVALENCE_PERF_TOLERANCE),
    }
}

fn equivalence_geometry(
    ctx: &VkContext,
    k: usize,
    n: usize,
    label: &str,
    configs: &[Cfg],
    full_space: bool,
) -> Result<(Value, bool, bool), Err> {
    eprintln!("equivalence: measuring {label} k{k}xn{n}");
    let mut prepared = PreparedGeometry::new(ctx, k, n)?;
    let row = prepared.baseline();
    let shipped_row_wall = prepared.measure_shipped_row_wall()?;
    let (shipped_wide_result, shipped_wide_wall) = prepared.measure_shipped_wide()?;
    let shipped_wide = shipped_wide_result.measurement.ok_or_else(|| {
        format!(
            "shipped wide kernel failed correctness for {label} k{k}xn{n}: {} (max_rel={:.3e})",
            shipped_wide_result
                .rejection
                .unwrap_or("unknown correctness failure"),
            shipped_wide_result.max_rel,
        )
    })?;
    let rows = measure_configs(&mut prepared, configs)?;
    let ranked = ranked_candidates(configs, &rows);
    let (best_config, best_measurement) = ranked
        .first()
        .copied()
        .ok_or("no viable, correct candidate in equivalence sweep")?;

    let find = |wanted: Cfg| {
        configs
            .iter()
            .zip(&rows)
            .find(|(config, _)| **config == wanted)
            .and_then(|(_, measurement)| *measurement)
    };
    let row_config = row_equivalent_config();
    let wide_config = wide_equivalent_config();
    let generated_row = find(row_config).ok_or("row-equivalent candidate was not measurable")?;
    let generated_wide = find(wide_config).ok_or("wide-equivalent candidate was not measurable")?;
    if !prepared.prepare(row_config)? {
        return Err("row-equivalent candidate became non-viable during wall control".into());
    }
    let generated_row_wall = prepared.measure_prepared_wall()?;
    if !prepared.prepare(wide_config)? {
        return Err("wide-equivalent candidate became non-viable during wall control".into());
    }
    let generated_wide_wall = prepared.measure_prepared_wall()?;
    let rank_of = |wanted: Cfg| {
        ranked
            .iter()
            .position(|(config, _)| *config == wanted)
            .map(|index| index + 1)
    };

    let selected = selected_family(n);
    let timestamp_winner = faster_family(row.ms, shipped_wide.ms);
    let generated_timestamp_winner = faster_family(generated_row.ms, generated_wide.ms);
    let wall_winner = faster_family(shipped_row_wall.selected_ms, shipped_wide_wall.selected_ms);
    let generated_wall_winner = faster_family(
        generated_row_wall.selected_ms,
        generated_wide_wall.selected_ms,
    );
    let timestamp_rankings_agree = timestamp_winner == generated_timestamp_winner;
    let wall_rankings_agree = wall_winner == generated_wall_winner;
    let timestamp_vs_wall_agree = timestamp_winner == wall_winner;
    let route_matches_timestamp_winner = selected == timestamp_winner;
    let route_matches_historical_wall_winner = selected == wall_winner;
    let route_assessment = assess_route(selected, row.ms, shipped_wide.ms);
    let passed = timestamp_rankings_agree && route_assessment.within_tolerance;

    Ok((
        json!({
            "schema_version": 1,
            "op": "equivalence",
            "ok": true,
            "label": label,
            "k": k,
            "n": n,
            "production_route": selected.label(),
            "exact_shipped": {
                "gpu_timestamp": {
                    "row": measurement_payload(row),
                    "wide": measurement_payload(shipped_wide),
                    "winner": timestamp_winner.label(),
                },
                "historical_batched_wall": {
                    "row": wall_measurement_payload(&shipped_row_wall),
                    "wide": wall_measurement_payload(&shipped_wide_wall),
                    "winner": wall_winner.label(),
                },
            },
            "generated_equivalents": {
                "row": {
                    "config": config_payload(row_config),
                    "rank": rank_of(row_config),
                    "gpu_timestamp": measurement_payload(generated_row),
                    "historical_batched_wall": wall_measurement_payload(&generated_row_wall),
                },
                "wide": {
                    "config": config_payload(wide_config),
                    "rank": rank_of(wide_config),
                    "gpu_timestamp": measurement_payload(generated_wide),
                    "historical_batched_wall": wall_measurement_payload(&generated_wide_wall),
                },
                "gpu_timestamp_winner": generated_timestamp_winner.label(),
                "historical_batched_wall_winner": generated_wall_winner.label(),
            },
            "full_space": {
                "candidate_count": configs.len(),
                "measured_count": ranked.len(),
                "best_config": config_payload(best_config),
                "best_measurement": measurement_payload(best_measurement),
                "exhaustive": full_space,
            },
            "timestamp_rankings_agree": timestamp_rankings_agree,
            "historical_wall_rankings_agree": wall_rankings_agree,
            "timestamp_vs_wall_agree": timestamp_vs_wall_agree,
            "route_matches_timestamp_winner": route_matches_timestamp_winner,
            "timestamp_route_assessment": {
                "selected_ms": route_assessment.selected_ms,
                "best_ms": route_assessment.best_ms,
                "slowdown": route_assessment.slowdown,
                "performance_tolerance": EQUIVALENCE_PERF_TOLERANCE,
                "within_tolerance": route_assessment.within_tolerance,
            },
            "route_matches_historical_wall_winner": route_matches_historical_wall_winner,
            "passed": passed,
        }),
        passed,
        route_matches_timestamp_winner,
    ))
}

/// Audit the production route over every decoder geometry from the census.
/// Each geometry keeps one context/weight allocation while the exact shipped
/// row and wide kernels and their generated equivalents are measured. The
/// historical three-by-ten batched-wall control is reported beside, but never
/// substituted for, the tuner timestamp score. Passing `--full-space`
/// additionally ranks all candidates; it is deliberately opt-in because the
/// two 150+ MB heads make that diagnostic expensive.
fn equivalence() -> Result<(), Err> {
    let ctx = VkContext::new()?;
    let full_space = std::env::args().any(|arg| arg == "--full-space");
    let configs = if full_space {
        wide_space()
    } else {
        vec![row_equivalent_config(), wide_equivalent_config()]
    };
    let mut passed = 0usize;
    let mut strict_route_matches = 0usize;
    let total = SHAPES.len() + LM_HEADS.len();

    for &(k, n, _, label) in SHAPES.iter().chain(LM_HEADS) {
        let (payload, geometry_passed, strict_route_match) =
            equivalence_geometry(&ctx, k, n, label, &configs, full_space)?;
        passed += usize::from(geometry_passed);
        strict_route_matches += usize::from(strict_route_match);
        println!("{payload}");
    }
    println!(
        "{}",
        json!({
            "schema_version": 1,
            "op": "equivalence_summary",
            "ok": passed == total,
            "passed_geometries": passed,
            "strict_route_matches": strict_route_matches,
            "total_geometries": total,
            "performance_tolerance": EQUIVALENCE_PERF_TOLERANCE,
            "full_space": full_space,
        })
    );
    if passed != total {
        return Err(format!("routing equivalence passed {passed}/{total} geometries").into());
    }
    Ok(())
}

fn main() -> Result<(), Err> {
    if std::env::args().any(|arg| arg == "--list") {
        list_candidates();
        return Ok(());
    }
    if std::env::args().any(|arg| arg == "--serve") {
        return serve();
    }
    if let Some(spec) = arg_value("--trial") {
        return trial(&spec);
    }
    if std::env::args().any(|a| a == "--sweep") {
        return sweep();
    }
    if std::env::args().any(|a| a == "--equivalence") {
        return equivalence();
    }

    let ctx = VkContext::new()?;
    let configs = default_configs();
    println!(
        "{:>20} {:>7} {:>8} {:>7} {:>7} {:>8} {:>9}",
        "shape / variant", "WGs", "ms", "GB/s", "TFLOP/s", "speedup", "max|rel|"
    );

    let mut rows = Vec::new();
    for &(k, n, count, label) in SHAPES {
        let (base, ms) = geometry(&ctx, k, n, &configs)?;
        print_geometry(label, k, n, &configs, &base, &ms);
        rows.push((base, ms, count));
    }
    totals(
        "weighted by the projections a decode step runs",
        &configs,
        &rows,
    );

    let mut heads = Vec::new();
    for &(k, n, count, label) in LM_HEADS {
        let (base, ms) = geometry(&ctx, k, n, &configs)?;
        print_geometry(label, k, n, &configs, &base, &ms);
        heads.push((base, ms, count));
    }
    totals(
        "lm_head only (one node per token, 150+ MB)",
        &configs,
        &heads,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_metadata_matches_rust_candidate_space() {
        let metadata = candidates_json();
        assert_eq!(metadata["schema_version"], 1);
        assert_eq!(metadata["family"], "matmul_nbits");
        assert_eq!(metadata["candidate_count"], base::candidate_space().len());
        assert_eq!(metadata["axes"]["lanes"], json!(base::CANDIDATE_LANES));
        assert_eq!(metadata["axes"]["vec"], json!([0, 1, 2, 4]));
        assert_eq!(metadata["axes"]["sub"], json!([0, 1]));
        assert_eq!(
            metadata["candidates"]
                .as_array()
                .expect("candidate array")
                .len(),
            base::candidate_space().len()
        );
    }

    #[test]
    fn prepare_request_parses_into_typed_config() {
        let request = json!({"lanes": 4, "vec": 2, "wg": 256, "sub": 1});
        let config = request_config(&request).expect("valid request must parse");
        assert_eq!(
            config,
            Cfg::new(
                4,
                CandidateLoad::BlockFourAccumulators,
                256,
                CandidateReduction::SubgroupShuffle,
            )
        );
    }

    #[test]
    fn prepare_request_rejects_unknown_or_malformed_axes() {
        assert!(request_config(&json!({"lanes": 4, "vec": 3, "wg": 256, "sub": 0})).is_err());
        assert!(request_config(&json!({"lanes": 4, "vec": 1, "wg": 256, "sub": 2})).is_err());
        assert!(request_config(&json!({"lanes": -1, "vec": 1, "wg": 256, "sub": 0})).is_err());
        assert!(request_config(&json!({"lanes": 4, "vec": 1, "wg": 256})).is_err());
    }

    #[test]
    fn timestamp_summary_uses_median_and_preserves_raw_samples() {
        let samples = (1..=20).rev().map(|value| value * 100).collect::<Vec<_>>();
        let (measurement, raw) = summarize_samples(samples.clone(), 7, 1.0, 2.0, 3e-5)
            .expect("20 non-zero samples must summarize");
        assert_eq!(measurement.samples, 20);
        assert_eq!(measurement.wgs, 7);
        assert_eq!(measurement.ms, 0.00105);
        assert_eq!(measurement.min_ms, 0.0001);
        assert_eq!(measurement.max_ms, 0.002);
        assert_eq!(measurement.max_rel, 3e-5);
        assert_eq!(raw, samples);
        assert!(summarize_samples(vec![1; 19], 1, 1.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn historical_wall_summary_preserves_minimum_selection_contract() {
        let measurement = summarize_wall_samples(vec![300, 100, 200])
            .expect("three non-zero batches must summarize");
        assert_eq!(measurement.batches, 3);
        assert_eq!(measurement.dispatches_per_batch, 10);
        assert_eq!(measurement.selected_ms, 0.0001);
        assert_eq!(measurement.median_ms, 0.0002);
        assert_eq!(measurement.min_ms, 0.0001);
        assert_eq!(measurement.max_ms, 0.0003);
        assert_eq!(measurement.samples_ns, vec![300, 100, 200]);
        assert!(summarize_wall_samples(vec![1, 2]).is_err());
        assert!(summarize_wall_samples(vec![1, 0, 2]).is_err());
    }

    #[test]
    fn route_assessment_keeps_strict_winner_separate_from_tolerance() {
        let strict_mismatch = assess_route(ShippedFamily::Row, 10.4, 10.0);
        assert_eq!(faster_family(10.4, 10.0), ShippedFamily::Wide);
        assert_eq!(strict_mismatch.selected_ms, 10.4);
        assert_eq!(strict_mismatch.best_ms, 10.0);
        assert_eq!(strict_mismatch.slowdown, 1.04);
        assert!(strict_mismatch.within_tolerance);

        let inclusive_boundary = assess_route(ShippedFamily::Row, 11.0, 10.0);
        assert!(inclusive_boundary.within_tolerance);

        let material_regression = assess_route(ShippedFamily::Row, 11.1, 10.0);
        assert!(!material_regression.within_tolerance);

        let strict_match = assess_route(ShippedFamily::Wide, 12.0, 10.0);
        assert_eq!(strict_match.slowdown, 1.0);
        assert!(strict_match.within_tolerance);
    }

    #[test]
    fn correctness_gate_handles_tolerance_and_non_finite_classes() {
        let accepted = compare_outputs(
            &[1.0, 0.0, f32::INFINITY, f32::NAN],
            &[1.00005, 0.00005, f32::INFINITY, f32::NAN],
        );
        assert!(accepted.passed);
        assert!(accepted.max_rel <= MAX_REL_ERROR);

        let relative = compare_outputs(&[1.0], &[1.001]);
        assert!(!relative.passed);
        assert_eq!(relative.rejection, Some("relative error exceeds 1e-4"));

        let nan = compare_outputs(&[0.0], &[f32::NAN]);
        assert!(!nan.passed);
        assert_eq!(nan.rejection, Some("NaN classification changed"));

        let infinity = compare_outputs(&[f32::INFINITY], &[f32::NEG_INFINITY]);
        assert!(!infinity.passed);
        assert_eq!(
            infinity.rejection,
            Some("infinity classification or sign changed")
        );
    }

    #[test]
    fn rejected_result_has_correctness_evidence_but_no_speed() {
        let result = RunResult {
            correctness_passed: false,
            max_rel: 0.25,
            rejection: Some("relative error exceeds 1e-4"),
            comparison: onnx_vulkan_core::comparison::compare_f32(
                &[1.0],
                &[1.25],
                onnx_vulkan_core::comparison::FloatTolerance::new(0.0, 1e-4)
                    .with_relative_floor(1.0),
            ),
            measurement: None,
            samples_ns: Vec::new(),
        };
        let payload = result_payload(
            Cfg::new(
                4,
                CandidateLoad::Word,
                256,
                CandidateReduction::WorkgroupTree,
            ),
            1152,
            6912,
            0.1,
            &result,
        );
        assert_eq!(payload["eligible"], false);
        assert_eq!(payload["correctness"]["passed"], false);
        assert!(payload.get("samples_ns").is_none());
        assert!(payload.get("ms").is_none());
        assert!(payload.get("speedup").is_none());
        assert_eq!(payload["diagnostic"]["outcome"], "numerical_failure");
        assert_eq!(payload["diagnostic"]["stage"], "correctness");
        assert_eq!(payload["diagnostic"]["workload"]["op"], "MatMulNBits");
        assert_eq!(
            payload["diagnostic"]["workload"]["inputs"][0]["dims"],
            json!([1, 1152])
        );
        assert_eq!(payload["diagnostic"]["timed"], false);
    }
}
