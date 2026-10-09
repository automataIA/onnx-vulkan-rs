//! The fused interleaved rotary embedding on the geometries SAM 3's vision
//! encoder runs: Q (`[B, S, H, D]` in, `[B, H, S, D]` out, scaled) and K (the
//! same, written transposed as `[B, H, D, S]` for `Q · Kᵀ`), over the window
//! and the global layers.
//!
//! Prints milliseconds per dispatch and GB/s over `x` read once, the output
//! written once and both tables read once.
//!
//! Run: `cargo run --release -p onnx-vulkan-core --example interleaved_rope`

use onnx_vulkan_core::shaders::elementwise as ew;
use std::time::Instant;
use vk_compute::VkContext;

/// (name, B, S, H, D, layers of this geometry per run). 16 heads of 64; 28
/// window layers over 9 windows of 24×24 tokens, 4 global layers over 72×72.
const SHAPES: &[(&str, usize, usize, usize, usize, usize)] = &[
    ("window", 9, 576, 16, 64, 28),
    ("global", 1, 5184, 16, 64, 4),
];
const REPS: u32 = 50;

fn pseudo(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..n)
        .flat_map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((((state >> 33) as f32) / (1u64 << 30) as f32) - 1.0).to_le_bytes()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = VkContext::new()?;
    let pipe = ctx.create_pipeline(
        &ew::interleaved_rope_spirv(),
        ew::INTERLEAVED_ROPE_BINDINGS,
        ew::INTERLEAVED_ROPE_PUSH_BYTES,
    )?;
    let mut total = 0.0;
    for &(name, b, s, h, d, layers) in SHAPES {
        let n = b * s * h * d;
        let upload = |count: usize, seed: u64| -> Result<_, Box<dyn std::error::Error>> {
            let buf = ctx.create_storage_buffer((4 * count) as u64)?;
            ctx.upload(&buf, &pseudo(count, seed))?;
            Ok(buf)
        };
        let (x, cos, sin) = (upload(n, 3)?, upload(s * d, 5)?, upload(s * d, 7)?);
        let out = ctx.create_storage_buffer((4 * n) as u64)?;
        for (role, transpose) in [("Q", false), ("K", true)] {
            let mut flags = ew::INTERLEAVED_ROPE_BSHD | ew::INTERLEAVED_ROPE_SCALE;
            if transpose {
                flags |= ew::INTERLEAVED_ROPE_TRANSPOSE;
            }
            let mut push = Vec::new();
            for value in [n as u32, h as u32, s as u32, d as u32, flags] {
                push.extend_from_slice(&value.to_le_bytes());
            }
            push.extend_from_slice(&0.125f32.to_le_bytes());
            let groups =
                ew::interleaved_rope_groups(b * h, s, d, transpose).ok_or("too many workgroups")?;
            let dispatch = || ctx.stream_dispatch(&pipe, &[&x, &cos, &sin, &out], &push, groups);
            dispatch()?;
            ctx.flush()?;
            let t = Instant::now();
            for _ in 0..REPS {
                dispatch()?;
            }
            ctx.flush()?;
            let ms = t.elapsed().as_secs_f64() * 1e3 / f64::from(REPS);
            let gb = 4.0 * (2 * n + 2 * s * d) as f64 / 1e9;
            println!(
                "{name:6} {role} B={b} S={s} H={h} D={d}: {ms:7.3} ms  {:6.1} GB/s  ×{layers} = {:.2} ms",
                gb / ms * 1e3,
                ms * layers as f64
            );
            total += ms * layers as f64;
        }
    }
    println!("vision encoder, all layers: {total:.2} ms");
    Ok(())
}
