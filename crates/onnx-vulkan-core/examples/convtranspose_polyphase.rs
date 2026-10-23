//! Is the polyphase GEMM faster than the gather kernel on Piper's upsamplers,
//! and does it give the same bits?
//!
//! Piper (VITS / HiFi-GAN) runs four 1D `ConvTranspose` nodes whose footprints
//! overlap (kernel = 2·stride) and whose pads crop both borders, so the phase
//! route never applies and all four took the gather kernel. This times both
//! kernels on those exact shapes, compares them bit for bit, and measures each
//! against an f64 CPU sum.
//!
//! Run: `cargo run --release -p onnx-vulkan-core --example convtranspose_polyphase`

use onnx_vulkan_core::shaders::conv_transpose;
use onnx_vulkan_core::shaders::gemm;
use std::time::Instant;
use vk_compute::{VkContext, compile_wgsl};

/// (node, C_in, C_out, L_in, kernel, stride, pad) of Piper Serena high at the
/// 560-frame input of the profiled sentence.
const SHAPES: &[(&str, usize, usize, usize, usize, usize, usize)] = &[
    ("ups.0", 512, 256, 560, 16, 8, 4),
    ("ups.1", 256, 128, 4480, 16, 8, 4),
    ("ups.2", 128, 64, 35840, 4, 2, 1),
    ("ups.3", 64, 32, 71680, 4, 2, 1),
];

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

fn push(fields: &[u32]) -> Vec<u8> {
    fields.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The operator in f64, scatter form (the other formulation from both kernels).
#[allow(clippy::too_many_arguments)]
fn reference(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    c_in: usize,
    c_out: usize,
    l_in: usize,
    k: usize,
    s: usize,
    p: usize,
    l_out: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; c_out * l_out];
    for m in 0..c_out {
        out[m * l_out..(m + 1) * l_out].fill(b[m] as f64);
    }
    for ic in 0..c_in {
        for m in 0..c_out {
            let row = &mut out[m * l_out..(m + 1) * l_out];
            for r in 0..k {
                let wv = w[(ic * c_out + m) * k + r] as f64;
                for i in 0..l_in {
                    let o = (i * s + r) as i64 - p as i64;
                    if o >= 0 && (o as usize) < l_out {
                        row[o as usize] += wv * x[ic * l_in + i] as f64;
                    }
                }
            }
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = VkContext::new()?;
    let gather = ctx.create_pipeline(
        &compile_wgsl(&conv_transpose::direct())?,
        conv_transpose::BINDINGS,
        conv_transpose::PUSH_BYTES,
    )?;
    let poly = ctx.create_pipeline(
        &compile_wgsl(conv_transpose::POLYPHASE_1D)?,
        conv_transpose::POLYPHASE_BINDINGS,
        conv_transpose::POLYPHASE_PUSH_BYTES,
    )?;

    let (mut tot_a, mut tot_b) = (0.0f64, 0.0f64);
    for &(name, c_in, c_out, l_in, k, s, p) in SHAPES {
        let l_out = (l_in - 1) * s + k - 2 * p;
        let x = pseudo(c_in * l_in, 3);
        let w = pseudo(c_in * c_out * k, 5);
        let b = pseudo(c_out, 7);

        let x_buf = ctx.create_storage_buffer((4 * x.len()) as u64)?;
        let w_buf = ctx.create_storage_buffer((4 * w.len()) as u64)?;
        let b_buf = ctx.create_storage_buffer((4 * b.len()) as u64)?;
        let out_a = ctx.create_storage_buffer((4 * c_out * l_out) as u64)?;
        let out_b = ctx.create_storage_buffer((4 * c_out * l_out) as u64)?;
        ctx.stream_upload(&x_buf, &bytes(&x))?;
        ctx.stream_upload(&w_buf, &bytes(&w))?;
        ctx.stream_upload(&b_buf, &bytes(&b))?;
        ctx.flush()?;

        let total = (c_out * l_out) as u32;
        // 1D normalized to 2D with W = 1, as the interpreter does
        let pa = push(&[
            total,
            c_in as u32,
            c_out as u32,
            1,
            l_in as u32,
            1,
            l_out as u32,
            1,
            k as u32,
            1,
            s as u32,
            1,
            p as u32,
            0,
            1,
            1,
            c_out as u32,
            1,
        ]);
        let run_gather = |reps: u32| -> Result<f64, Box<dyn std::error::Error>> {
            let t = Instant::now();
            for _ in 0..reps {
                ctx.stream_dispatch(
                    &gather,
                    &[&x_buf, &w_buf, &b_buf, &out_a],
                    &pa,
                    [total.div_ceil(256), 1, 1],
                )?;
            }
            ctx.flush()?;
            Ok(t.elapsed().as_secs_f64() / reps as f64)
        };

        let nq = (l_out - 1 + p) / s + 1;
        let pb = push(&[
            c_in as u32,
            c_out as u32,
            l_in as u32,
            l_out as u32,
            k as u32,
            s as u32,
            p as u32,
            nq as u32,
            1,
        ]);
        let run_poly = |reps: u32| -> Result<f64, Box<dyn std::error::Error>> {
            let t = Instant::now();
            for _ in 0..reps {
                ctx.stream_dispatch(
                    &poly,
                    &[&w_buf, &x_buf, &b_buf, &out_b],
                    &pb,
                    [
                        (nq as u32).div_ceil(gemm::TILE_SIZE),
                        (c_out as u32).div_ceil(gemm::TILE_SIZE),
                        s as u32,
                    ],
                )?;
            }
            ctx.flush()?;
            Ok(t.elapsed().as_secs_f64() / reps as f64)
        };

        run_gather(2)?;
        run_poly(2)?;
        let ga = (0..5).try_fold(f64::MAX, |acc, _| run_gather(4).map(|t| acc.min(t)))?;
        let gb = (0..5).try_fold(f64::MAX, |acc, _| run_poly(4).map(|t| acc.min(t)))?;

        let read = |buf| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            Ok(ctx
                .stream_download(buf, 4 * c_out * l_out)?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        };
        let (fa, fb) = (read(&out_a)?, read(&out_b)?);
        let exact = reference(&x, &w, &b, c_in, c_out, l_in, k, s, p, l_out);
        let err = |got: &[f32]| {
            got.iter()
                .zip(&exact)
                .fold((0.0f64, 0.0f64), |(a, r), (&g, &e)| {
                    let d = (g as f64 - e).abs();
                    (a.max(d), r.max(d / e.abs().max(1e-6)))
                })
        };
        let bit_exact = fa
            .iter()
            .zip(&fb)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        let (ea, eb) = (err(&fa), err(&fb));

        let flops = 2.0 * (c_in * c_out * k * l_in) as f64;
        println!(
            "{name}: {c_in}→{c_out} L {l_in}→{l_out} k{k} s{s} p{p}\n  \
             gather     {:8.3} ms  {:6.3} TFLOP/s   vs f64 max|Δ| {:.3e} rel {:.3e}\n  \
             polyphase  {:8.3} ms  {:6.3} TFLOP/s   vs f64 max|Δ| {:.3e} rel {:.3e}   speedup {:.1}×\n  \
             gather vs polyphase: bit-exact {bit_exact}/{}",
            ga * 1e3,
            flops / ga / 1e12,
            ea.0,
            ea.1,
            gb * 1e3,
            flops / gb / 1e12,
            eb.0,
            eb.1,
            ga / gb,
            fa.len(),
        );
        tot_a += ga * 1e3;
        tot_b += gb * 1e3;
    }
    println!("\nPiper ConvTranspose total: gather {tot_a:.2} ms → polyphase {tot_b:.2} ms");
    Ok(())
}
