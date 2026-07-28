//! Is `ConvTranspose` a GEMM when stride ≥ kernel, and is it faster as one?
//!
//! When the stride is at least the kernel size, with no padding and no
//! dilation, the kernel footprints never overlap: every output pixel is fed by
//! exactly one kernel offset `(r, s)`, decided by `oh % sh` and `ow % sw`. The
//! operator then splits into `kh*kw` independent products
//!
//! ```text
//!   phase[r][s][m][p] = Σ_ic W[ic, m, r, s] · X[ic, p]      p over input pixels
//! ```
//!
//! each a GEMM of M = C_out, K = C_in, N = H_in·W_in, followed by an interleave
//! that writes `phase[r][s][m][ih, iw]` to `out[m, ih*sh + r, iw*sw + s]`.
//!
//! This measures both paths on the shapes sam3 ViT-H actually runs and diffs
//! them, so the routing logic is written against a verified equivalence rather
//! than against the derivation above.
//!
//! Run: `cargo run --release -p onnx-vulkan-core --example convtranspose_gemm`

use onnx_vulkan_core::shaders::conv_transpose;
use onnx_vulkan_core::shaders::gemm;
use std::time::Instant;
use vk_compute::{VkContext, compile_wgsl};

/// (C_in, C_out, H_in, W_in, how many nodes of this shape ViT-H runs)
const SHAPES: &[(usize, usize, usize, usize, usize)] =
    &[(1024, 512, 72, 72, 2), (512, 256, 144, 144, 1)];

/// Writes each phase's `[M, N]` result back to its strided home in the output.
const INTERLEAVE: &str = r#"
@group(0) @binding(0) var<storage, read> phase: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
struct Push { total: u32, m: u32, h_in: u32, w_in: u32, sh: u32, sw: u32, r: u32, s: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.total) { return; }
    let iw = i % pc.w_in;
    let t = i / pc.w_in;
    let ih = t % pc.h_in;
    let ch = t / pc.h_in;
    let h_out = pc.h_in * pc.sh;
    let w_out = pc.w_in * pc.sw;
    let oh = ih * pc.sh + pc.r;
    let ow = iw * pc.sw + pc.s;
    out[(ch * h_out + oh) * w_out + ow] = phase[ch * pc.h_in * pc.w_in + ih * pc.w_in + iw];
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = VkContext::new()?;
    let gather = ctx.create_pipeline(
        &compile_wgsl(&conv_transpose::direct())?,
        conv_transpose::BINDINGS,
        conv_transpose::PUSH_BYTES,
    )?;
    let gemm_pipe =
        ctx.create_pipeline(&compile_wgsl(gemm::GEMM)?, gemm::BINDINGS, gemm::PUSH_BYTES)?;
    let inter = ctx.create_pipeline(&compile_wgsl(INTERLEAVE)?, 2, 32)?;

    let mut tot_gather = 0.0f64;
    let mut tot_gemm = 0.0f64;

    for &(c_in, c_out, h_in, w_in, count) in SHAPES {
        let (kh, kw, sh, sw) = (2usize, 2usize, 2usize, 2usize);
        let (h_out, w_out) = (h_in * sh, w_in * sw);
        let n = h_in * w_in;
        let x = pseudo(c_in * n, 3);
        let w = pseudo(c_in * c_out * kh * kw, 5);

        let x_buf = ctx.create_storage_buffer((4 * x.len()) as u64)?;
        let w_buf = ctx.create_storage_buffer((4 * w.len()) as u64)?;
        let zero = ctx.create_storage_buffer(4)?;
        let out_a = ctx.create_storage_buffer((4 * c_out * h_out * w_out) as u64)?;
        let out_b = ctx.create_storage_buffer((4 * c_out * h_out * w_out) as u64)?;
        let phase = ctx.create_storage_buffer((4 * c_out * n) as u64)?;
        ctx.stream_upload(&x_buf, &bytes(&x))?;
        ctx.stream_upload(&w_buf, &bytes(&w))?;

        // --- path A: the current gather kernel, one thread per output element
        let total = (c_out * h_out * w_out) as u32;
        let mut pa = Vec::new();
        for v in [
            total,
            c_in as u32,
            c_out as u32,
            1,
            h_in as u32,
            w_in as u32,
            h_out as u32,
            w_out as u32,
            kh as u32,
            kw as u32,
            sh as u32,
            sw as u32,
            0,
            0,
            1,
            1,
            c_out as u32,
            0,
        ] {
            pa.extend_from_slice(&v.to_le_bytes());
        }
        let run_gather = |reps: u32| -> Result<f64, Box<dyn std::error::Error>> {
            let t = Instant::now();
            for _ in 0..reps {
                ctx.stream_dispatch(
                    &gather,
                    &[&x_buf, &w_buf, &zero, &out_a],
                    &pa,
                    [total.div_ceil(256), 1, 1],
                )?;
            }
            ctx.flush()?;
            Ok(t.elapsed().as_secs_f64() / reps as f64)
        };

        // --- path B: kh*kw GEMMs of [M=c_out, K=c_in, N=n] + interleave.
        //
        // A phase's weight slice `W[:, :, r, s]` has stride kh*kw between
        // consecutive output channels, so it is packed once into [K][M] and
        // fed to the GEMM as `trans_a`. In the compiler this packing belongs
        // with the other cached constant weights, so it is not timed here.
        let mut packed = Vec::with_capacity(kh * kw);
        for r in 0..kh {
            for s in 0..kw {
                let mut p = vec![0.0f32; c_in * c_out];
                for ic in 0..c_in {
                    for m in 0..c_out {
                        p[ic * c_out + m] = w[((ic * c_out + m) * kh + r) * kw + s];
                    }
                }
                let buf = ctx.create_storage_buffer((4 * p.len()) as u64)?;
                ctx.stream_upload(&buf, &bytes(&p))?;
                packed.push(buf);
            }
        }
        ctx.flush()?;

        let run_gemm = |reps: u32| -> Result<f64, Box<dyn std::error::Error>> {
            let t = Instant::now();
            for _ in 0..reps {
                for r in 0..kh {
                    for s in 0..kw {
                        let mut pg = Vec::new();
                        for v in [c_out as u32, c_in as u32, n as u32, 1] {
                            pg.extend_from_slice(&v.to_le_bytes()); // flags = transA
                        }
                        pg.extend_from_slice(&1.0f32.to_le_bytes());
                        pg.extend_from_slice(&0.0f32.to_le_bytes());
                        pg.extend_from_slice(&1u32.to_le_bytes());
                        pg.extend_from_slice(&1u32.to_le_bytes());
                        ctx.stream_dispatch(
                            &gemm_pipe,
                            &[&packed[r * kw + s], &x_buf, &zero, &phase],
                            &pg,
                            [
                                (n as u32).div_ceil(gemm::TILE_SIZE),
                                (c_out as u32).div_ceil(gemm::TILE_SIZE),
                                1,
                            ],
                        )?;
                        let count = (c_out * n) as u32;
                        let mut pi = Vec::new();
                        for v in [
                            count,
                            c_out as u32,
                            h_in as u32,
                            w_in as u32,
                            sh as u32,
                            sw as u32,
                            r as u32,
                            s as u32,
                        ] {
                            pi.extend_from_slice(&v.to_le_bytes());
                        }
                        ctx.stream_dispatch(
                            &inter,
                            &[&phase, &out_b],
                            &pi,
                            [count.div_ceil(256), 1, 1],
                        )?;
                    }
                }
            }
            ctx.flush()?;
            Ok(t.elapsed().as_secs_f64() / reps as f64)
        };

        run_gather(2)?;
        run_gemm(2)?;
        let ga = (0..3).try_fold(f64::MAX, |acc, _| run_gather(4).map(|s| acc.min(s)))?;
        let gb = (0..3).try_fold(f64::MAX, |acc, _| run_gemm(4).map(|s| acc.min(s)))?;

        // --- diff
        let nbytes = 4 * c_out * h_out * w_out;
        let ra = ctx.stream_download(&out_a, nbytes)?;
        let rb = ctx.stream_download(&out_b, nbytes)?;
        let fa: Vec<f32> = ra
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let fb: Vec<f32> = rb
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut exact = 0usize;
        let mut maxd = 0.0f32;
        let mut maxrel = 0.0f32;
        for (a, b) in fa.iter().zip(&fb) {
            if a.to_bits() == b.to_bits() {
                exact += 1;
            }
            let d = (a - b).abs();
            maxd = maxd.max(d);
            maxrel = maxrel.max(d / a.abs().max(1e-6));
        }

        // Every output pixel takes C_in MACs, and there are kh*kw times as many
        // output pixels as input ones — i.e. all four phases together.
        let flops = 2.0 * (c_in * c_out * h_out * w_out) as f64;
        println!(
            "{c_in}→{c_out} {h_in}×{w_in} (×{count})\n  \
             gather {:8.2} ms  {:5.3} TFLOP/s\n  \
             gemm   {:8.2} ms  {:5.3} TFLOP/s   speedup {:.2}×\n  \
             bit-esatti {exact}/{}  max|Δ| {maxd:.3e}  rel {maxrel:.3e}",
            ga * 1e3,
            flops / ga / 1e12,
            gb * 1e3,
            flops / gb / 1e12,
            ga / gb,
            fa.len(),
        );
        tot_gather += ga * 1e3 * count as f64;
        tot_gemm += gb * 1e3 * count as f64;
    }

    println!("\nViT-H ConvTranspose total: gather {tot_gather:.1} ms → gemm {tot_gemm:.1} ms");
    Ok(())
}
