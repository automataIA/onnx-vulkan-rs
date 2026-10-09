//! FlashAttention in f32 (WGSL) against f16 cooperative matrices, on the
//! attention geometries SAM 3's vision encoder and text detector run.
//!
//! Prints milliseconds per dispatch, TFLOP/s (`4·bh·m·n·d`, both products),
//! GB/s over the operands read once and the output written once, and the
//! largest difference of the f16 output from the f32 one.
//!
//! Run: `cargo run --release -p onnx-vulkan-core --example flash_attention`

use onnx_vulkan_core::shaders::flash_attention as fa;
use std::time::Instant;
use vk_compute::{VkContext, compile_wgsl};

/// (model, name, bh, queries m, keys n, head size, layers of this geometry
/// per run). Vision encoder: 16 heads of 64, 28 window layers over 9 windows
/// of 24×24 tokens and 4 global layers over 72×72 tokens. Detector: 8 heads
/// of 32, 6 encoder self-attentions over 72×72 tokens, 6 encoder
/// cross-attentions from the image tokens to 32 text tokens, 6 decoder
/// self-attentions over 201 queries and 6 decoder cross-attentions from them
/// to the image tokens.
const SHAPES: &[(&str, &str, usize, usize, usize, usize, usize)] = &[
    ("vision", "window", 9 * 16, 576, 576, 64, 28),
    ("vision", "global", 16, 5184, 5184, 64, 4),
    ("detector", "encoder", 8, 5184, 5184, 32, 6),
    ("detector", "to text", 8, 5184, 32, 32, 6),
    ("detector", "decoder", 8, 201, 201, 32, 6),
    ("detector", "to image", 8, 201, 5184, 32, 6),
];
const REPS: u32 = 20;

fn pseudo(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((state >> 33) as f32 / (1u64 << 30) as f32) - 1.0) * scale
        })
        .collect()
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = VkContext::new()?;
    if !(ctx.coop_f16 && ctx.subgroup_size == fa::COOP_SUBGROUP) {
        return Err("no f16 cooperative matrix on subgroups of 32".into());
    }
    let limits = ctx.compute_limits();
    println!("f16 kernel: 64-key tiles, 64 query rows");
    let mut totals: Vec<(&str, f64, f64)> = Vec::new();

    for &(model, name, bh, m, n, d, layers) in SHAPES {
        let du = d as u32;
        let bc = fa::key_tile(du, du, limits.max_shared_memory_bytes).ok_or("no key tile fits")?;
        let f32_pipe = ctx.create_pipeline(
            &compile_wgsl(&fa::source(du, du, bc, false))?,
            fa::BINDINGS,
            fa::PUSH_BYTES,
        )?;
        let f16_pipe =
            ctx.create_pipeline(&fa::coop_spirv(du, false), fa::BINDINGS, fa::PUSH_BYTES)?;
        // the workgroup-scope kernel, when the device has it
        let wg_pipe = if ctx.coop_f16_wg {
            Some(ctx.create_pipeline(&fa::wg_spirv(du, false), fa::BINDINGS, fa::PUSH_BYTES)?)
        } else {
            None
        };
        // queries carry the 1/sqrt(d) scale
        let q = pseudo(bh * m * d, 3, 3.0 / (d as f32).sqrt());
        let kt = pseudo(bh * d * n, 5, 3.0);
        let v = pseudo(bh * n * d, 7, 2.0);
        let upload = |data: &[f32]| -> Result<_, Box<dyn std::error::Error>> {
            let buf = ctx.create_storage_buffer((4 * data.len()) as u64)?;
            ctx.upload(&buf, &bytes(data))?;
            Ok(buf)
        };
        let (qb, kb, vb) = (upload(&q)?, upload(&kt)?, upload(&v)?);
        let out32 = ctx.create_storage_buffer((4 * bh * m * d) as u64)?;
        let out16 = ctx.create_storage_buffer((4 * bh * m * d) as u64)?;
        let out_wg = ctx.create_storage_buffer((4 * bh * m * d) as u64)?;
        let mut push = Vec::new();
        // no bias: nan_to_zero off, then heads 1 and zero strides
        for value in [m as u32, n as u32, 1, 1, 0, 1, 0, 0, 0] {
            push.extend_from_slice(&value.to_le_bytes());
        }
        let time = |pipe, out, br: u32| -> Result<f64, Box<dyn std::error::Error>> {
            let dispatch = || {
                ctx.stream_dispatch(
                    pipe,
                    &[&qb, &kb, &vb, out],
                    &push,
                    [(m as u32).div_ceil(br), bh as u32, 1],
                )
            };
            dispatch()?;
            ctx.flush()?;
            let t = Instant::now();
            for _ in 0..REPS {
                dispatch()?;
            }
            ctx.flush()?;
            Ok(t.elapsed().as_secs_f64() * 1e3 / f64::from(REPS))
        };
        let ms32 = time(&f32_pipe, &out32, fa::BR)?;
        let ms16 = time(&f16_pipe, &out16, fa::COOP_BR)?;
        let (o32, o16) = (
            floats(&ctx.download(&out32)?),
            floats(&ctx.download(&out16)?),
        );
        let worst = o32
            .iter()
            .zip(&o16)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if let Some(pipe) = &wg_pipe {
            let ms_wg = time(pipe, &out_wg, fa::COOP_BR)?;
            let o_wg = floats(&ctx.download(&out_wg)?);
            let worst_wg = o32
                .iter()
                .zip(&o_wg)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!(
                "{name:8} workgroup f16: {ms_wg:8.3} ms ({:.2}x the f16 kernel), max |Δ| {worst_wg:.3e}",
                ms16 / ms_wg,
            );
        }
        let flop = 4.0 * (bh * m * n * d) as f64;
        let gb = 4.0 * (bh * (2 * m + 2 * n) * d) as f64 / 1e9;
        for (kernel, ms) in [("f32", ms32), ("f16", ms16)] {
            println!(
                "{name:8} bh={bh:3} m={m:4} n={n:4} d={d} {kernel} (bc {}): {ms:8.3} ms  {:6.2} TFLOP/s  {:6.1} GB/s",
                if kernel == "f32" { bc } else { fa::COOP_BR },
                flop / ms / 1e9,
                gb / ms * 1e3,
            );
        }
        println!(
            "{name:8} f16 vs f32: {:.2}x, max |Δ| {worst:.3e} (max |v| 2), ×{layers} layers: {:.1} → {:.1} ms",
            ms32 / ms16,
            ms32 * layers as f64,
            ms16 * layers as f64,
        );
        match totals.iter_mut().find(|t| t.0 == model) {
            Some(t) => (t.1, t.2) = (t.1 + ms32 * layers as f64, t.2 + ms16 * layers as f64),
            None => totals.push((model, ms32 * layers as f64, ms16 * layers as f64)),
        }
    }
    for (model, f32_ms, f16_ms) in totals {
        println!("{model} layers: f32 {f32_ms:.1} ms, f16 {f16_ms:.1} ms");
    }
    Ok(())
}
