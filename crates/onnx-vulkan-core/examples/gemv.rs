//! Is `MatMul` at `M = 1` worth a dedicated kernel, and with how much split-K?
//!
//! At sequence length 1 every `MatMul` in a transformer is a matrix-vector
//! product, and `shaders::matmul_fp32::MATMUL_SMALL` handles it badly for two
//! independent reasons. It stages a 16×16 tile, so with one row of output 15 of
//! every 16 threads compute a result that is thrown away; and the grid is
//! `ceil(N/16)` workgroups, which for `N = 768` is 48 — barely one per SM on a
//! 4070, far too few to hide memory latency behind.
//!
//! A thinner tile fixes the first and worsens the second. Only splitting `K`
//! fixes both: it fabricates workgroups where the output shape has none, at the
//! cost of a second pass to sum the partials.
//!
//! The regime is also not the one the blocked kernel was tuned for. These
//! shapes are **bandwidth-bound, not compute-bound**: roberta at `seq_len = 1`
//! runs 170 MFLOP against 340 MB of weight reads, so at the 4070's ~504 GB/s
//! the floor is ~0.67 ms and arithmetic intensity is irrelevant. Read the GB/s
//! column, not the TFLOP/s one.
//!
//! The three shapes below are every `MatMul` geometry `roberta-base-11` runs at
//! `seq_len = 1` that is not trivially small, with the node counts the model
//! executes, taken from the graph rather than assumed from the architecture.
//!
//! Run: `cargo run --release -p onnx-vulkan-core --example gemv`

use onnx_vulkan_core::shaders::matmul_fp32 as mm;
use std::time::Instant;
use vk_compute::{ComputePipeline, GpuBuffer, VkContext, compile_wgsl};

/// (K, N, how many nodes of this shape the model runs).
const SHAPES: &[(usize, usize, usize)] = &[
    (768, 768, 48),  // Q, K, V and the attention output projection
    (768, 3072, 12), // FFN in
    (3072, 768, 12), // FFN out
];

/// Columns per workgroup × K-splits. 256 threads either way, so `256 / COLS`
/// lanes cooperate on each column and are reduced in shared memory.
const VARIANTS: &[(u32, u32)] = &[
    (256, 1),
    (64, 1),
    (64, 4),
    (64, 8),
    (32, 1),
    (32, 2),
    (32, 4),
    (32, 8),
    (32, 16),
    (32, 32),
    (16, 32),
];

/// One workgroup per `COLS` columns per K-split; each column is summed by
/// `256 / COLS` lanes striding over its slice of K.
///
/// `col = col0 + tid % COLS` on purpose: consecutive threads take consecutive
/// columns, so a lane's read of `b[k * n + col]` is coalesced across the row.
/// The `a` vector is broadcast — every column wants the same `a[k]` — and is
/// left to the cache rather than staged, since it is 3 KB at most here.
fn gemv_source(cols: u32) -> String {
    let lanes = 256 / cols;
    format!(
        r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Push {{ k: u32, n: u32, split: u32, pad: u32 }}
var<immediate> pc: Push;

const COLS = {cols}u;
const LANES = {lanes}u;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {{
    let col = wid.x * COLS + tid % COLS;
    let lane = tid / COLS;

    // this workgroup's slice of K
    let kper = (pc.k + pc.split - 1u) / pc.split;
    let kstart = wid.y * kper;
    var kend = kstart + kper;
    if (kend > pc.k) {{ kend = pc.k; }}

    var acc = 0.0;
    if (col < pc.n) {{
        for (var k = kstart + lane; k < kend; k = k + LANES) {{
            acc = fma(a[k], b[k * pc.n + col], acc);
        }}
    }}
    red[tid] = acc;
    workgroupBarrier();

    // tree-reduce the LANES partials of each column; lane 0 holds the result
    for (var s = LANES / 2u; s > 0u; s = s / 2u) {{
        if (lane < s) {{ red[tid] = red[tid] + red[tid + s * COLS]; }}
        workgroupBarrier();
    }}
    if (lane == 0u && col < pc.n) {{
        // one partial per split; a second pass sums them when split > 1
        out[wid.y * pc.n + col] = red[tid];
    }}
}}
"#
    )
}

/// Sums the `split` partial vectors. Only dispatched when `split > 1`.
const REDUCE: &str = r#"
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Push { k: u32, n: u32, split: u32, pad: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= pc.n) { return; }
    var acc = 0.0;
    for (var s = 0u; s < pc.split; s = s + 1u) {
        acc = acc + partials[s * pc.n + col];
    }
    out[col] = acc;
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

/// Push constants for `MATMUL_SMALL`: m, k, n, rank then six vec4 of strides.
fn matmul_push(k: usize, n: usize) -> Vec<u8> {
    let mut push = Vec::with_capacity(mm::PUSH_BYTES as usize);
    for v in [1u32, k as u32, n as u32, 2] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    // rank 2: no batch dimensions, so every stride is zero
    push.resize(mm::PUSH_BYTES as usize, 0);
    push
}

fn gemv_push(k: usize, n: usize, split: u32) -> Vec<u8> {
    let mut push = Vec::new();
    for v in [k as u32, n as u32, split, 0] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    push
}

type Err = Box<dyn std::error::Error>;

fn main() -> Result<(), Err> {
    let ctx = VkContext::new()?;
    let baseline = ctx.create_pipeline(
        &compile_wgsl(mm::MATMUL_SMALL)?,
        mm::BINDINGS,
        mm::PUSH_BYTES,
    )?;
    let reduce = ctx.create_pipeline(&compile_wgsl(REDUCE)?, 2, 16)?;
    let variants: Vec<(u32, u32, ComputePipeline)> = VARIANTS
        .iter()
        .map(|&(cols, split)| {
            // the split is a grid and push-constant parameter only, so two
            // variants sharing `cols` compile the same source
            let p = ctx.create_pipeline(&compile_wgsl(&gemv_source(cols))?, 3, 16)?;
            Ok::<_, Err>((cols, split, p))
        })
        .collect::<Result<_, _>>()?;

    println!(
        "{:>14} {:>6} {:>9} {:>8} {:>8} {:>9}",
        "shape", "WGs", "ms", "GB/s", "speedup", "max|rel|"
    );

    let mut totals = vec![0.0f64; variants.len()];
    let mut total_base = 0.0f64;

    for &(k, n, count) in SHAPES {
        let a = pseudo(k, 3);
        let b = pseudo(k * n, 5);
        let a_buf = ctx.create_storage_buffer((4 * k) as u64)?;
        let b_buf = ctx.create_storage_buffer((4 * k * n) as u64)?;
        let out_ref = ctx.create_storage_buffer((4 * n) as u64)?;
        let out_gemv = ctx.create_storage_buffer((4 * n) as u64)?;
        let partials = ctx.create_storage_buffer((4 * n * 32) as u64)?;
        ctx.stream_upload(&a_buf, &bytes(&a))?;
        ctx.stream_upload(&b_buf, &bytes(&b))?;
        ctx.flush()?;

        let traffic = (4 * k * n) as f64;
        let mpush = matmul_push(k, n);

        let run_base = |reps: u32| -> Result<f64, Err> {
            let t = Instant::now();
            for _ in 0..reps {
                ctx.stream_dispatch(
                    &baseline,
                    &[&a_buf, &b_buf, &out_ref],
                    &mpush,
                    [(n as u32).div_ceil(16), 1, 1],
                )?;
            }
            ctx.flush()?;
            Ok(t.elapsed().as_secs_f64() / reps as f64)
        };
        run_base(2)?;
        let tb = (0..3).try_fold(f64::MAX, |acc, _| run_base(10).map(|s| acc.min(s)))?;
        let expect = floats(&ctx.stream_download(&out_ref, 4 * n)?);

        println!(
            "\n{:>14} {:>6} {:>9.3} {:>8.0} {:>8} {:>9}",
            format!("k{k}xn{n} x{count}"),
            (n as u32).div_ceil(16),
            tb * 1e3,
            traffic / tb / 1e9,
            "1.00× (MatMul16)",
            ""
        );
        total_base += tb * 1e3 * count as f64;

        for (i, (cols, split, pipe)) in variants.iter().enumerate() {
            let (cols, split) = (*cols, *split);
            // A split wider than K leaves workgroups with nothing to do.
            if split as usize > k {
                continue;
            }
            let gpush = gemv_push(k, n, split);
            let grid = [(n as u32).div_ceil(cols), split, 1];
            let target: &GpuBuffer = if split > 1 { &partials } else { &out_gemv };

            let run = |reps: u32| -> Result<f64, Err> {
                let t = Instant::now();
                for _ in 0..reps {
                    ctx.stream_dispatch(pipe, &[&a_buf, &b_buf, target], &gpush, grid)?;
                    if split > 1 {
                        ctx.stream_dispatch(
                            &reduce,
                            &[&partials, &out_gemv],
                            &gpush,
                            [(n as u32).div_ceil(256), 1, 1],
                        )?;
                    }
                }
                ctx.flush()?;
                Ok(t.elapsed().as_secs_f64() / reps as f64)
            };
            run(2)?;
            let t = (0..3).try_fold(f64::MAX, |acc, _| run(10).map(|s| acc.min(s)))?;

            let got = floats(&ctx.stream_download(&out_gemv, 4 * n)?);
            let rel = expect
                .iter()
                .zip(&got)
                .fold(0.0f32, |m, (e, g)| m.max((e - g).abs() / e.abs().max(1e-3)));

            println!(
                "{:>14} {:>6} {:>9.3} {:>8.0} {:>7.2}× {rel:>9.1e}",
                format!("cols{cols} split{split}"),
                grid[0] * grid[1],
                t * 1e3,
                traffic / t / 1e9,
                tb / t,
            );
            totals[i] += t * 1e3 * count as f64;
        }
    }

    println!("\n===== weighted by the node counts roberta actually runs =====");
    println!("{:>20} {:>9} {:>8}", "variant", "ms", "speedup");
    println!("{:>20} {:>9.3} {:>8}", "MatMul16", total_base, "1.00×");
    for (i, (cols, split)) in VARIANTS.iter().enumerate() {
        if totals[i] > 0.0 {
            println!(
                "{:>20} {:>9.3} {:>7.2}×",
                format!("cols{cols} split{split}"),
                totals[i],
                total_base / totals[i]
            );
        }
    }
    Ok(())
}
