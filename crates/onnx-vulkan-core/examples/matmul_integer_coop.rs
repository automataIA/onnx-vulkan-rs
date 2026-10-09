//! Cooperative-matrix `MatMulInteger`: every compiled block size, on the
//! geometries that decide which one runs.
//!
//! SAM 3's int8 vision encoder runs 192 `MatMulInteger` nodes, all with
//! `M = 5184` tokens (72×72, or 9 windows of 24×24): 128 with `K = N = 1024`,
//! 32 with `K = 1024, N = 4736` and 32 with `K = 4736, N = 1024`.
//! resnet50-int8's convolutions, stated as the GEMMs
//! `interp::conv_im2col_coop` runs (`M = C_out`, `N = P`), are small enough
//! that a large block leaves the device half empty; together they decide
//! `matmul_integer::coop_variant_for`.
//!
//! Each variant is checked bit for bit against the WGSL `dot4` kernel, which
//! uses no cooperative matrix, then timed with GPU timestamps.
//!
//! Run: `cargo run --release -p onnx-vulkan-core --example matmul_integer_coop`
//! on a device with a `u8×u8` cooperative combination. Extra SPIR-V builds of
//! `glsl/matmul_integer_coop.comp` or `glsl/matmul_integer_coop_smem.comp` can
//! be passed as `path.spv:BM/16:BN/16`, the workgroup block in 16×16 tiles.

use onnx_vulkan_core::shaders::matmul_integer::{
    self, COOP_BINDINGS, COOP_PUSH_BYTES, COOP_TILE, MATMUL_BINDINGS, MATMUL_PUSH_BYTES, TILE_SIZE,
};
use vk_compute::{ComputePipeline, VkContext, compile_wgsl};

type Err = Box<dyn std::error::Error>;

/// `(m, k, n, nodes)` of SAM 3's vision encoder.
const ENCODER: [(usize, usize, usize, usize); 3] = [
    (5184, 1024, 1024, 128),
    (5184, 1024, 4736, 32),
    (5184, 4736, 1024, 32),
];

/// `(m, k, n, nodes)` run once per prompt: SAM 3's int8 text encoder (32
/// tokens) and the text detector's cooperative-eligible products (201 decoder
/// queries, 32 text tokens, 5184 image tokens).
const PER_PROMPT: [(usize, usize, usize, usize); 13] = [
    (32, 1024, 1024, 96),
    (32, 1024, 4096, 24),
    (32, 4096, 1024, 24),
    (32, 1024, 256, 1),
    (201, 256, 256, 48),
    (200, 256, 256, 19),
    (201, 256, 2048, 6),
    (201, 2048, 256, 6),
    (200, 512, 256, 6),
    (32, 256, 256, 26),
    (5184, 256, 256, 50),
    (5184, 256, 2048, 6),
    (5184, 2048, 256, 6),
];

/// `(m, k, n)` of resnet50-int8's cooperative-eligible convolutions.
const RESNET50_CONV: [(usize, usize, usize); 10] = [
    (256, 2304, 196),
    (128, 1152, 784),
    (64, 576, 3136),
    (512, 4608, 49),
    (512, 256, 784),
    (128, 256, 784),
    (1024, 512, 196),
    (256, 512, 196),
    (2048, 1024, 49),
    (512, 1024, 49),
];

struct Variant {
    name: String,
    pipeline: ComputePipeline,
    /// Output rows and columns per workgroup.
    block: [u32; 2],
}

/// Deterministic bytes.
fn bytes(count: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u8
        })
        .collect()
}

fn spirv_words(raw: &[u8]) -> Vec<u32> {
    raw.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Checks and times every variant on one `[m, k] × [k, n]`; returns the
/// milliseconds per dispatch, in the order of `variants`.
fn run_shape(
    ctx: &VkContext,
    reference: &ComputePipeline,
    variants: &[Variant],
    (m, k, n): (usize, usize, usize),
) -> Result<Vec<f64>, Err> {
    let a = ctx.create_storage_buffer((m * k) as u64)?;
    let b = ctx.create_storage_buffer((n * k) as u64)?;
    let azp = ctx.create_storage_buffer(4)?;
    let bzp = ctx.create_storage_buffer(4)?;
    let out = ctx.create_storage_buffer((m * n * 4) as u64)?;
    ctx.upload(&a, &bytes(m * k, 1))?;
    ctx.upload(&b, &bytes(n * k, 2))?;
    ctx.upload(&azp, &[117, 0, 0, 0])?;
    ctx.upload(&bzp, &[128, 0, 0, 0])?;
    let buffers = [&a, &b, &azp, &bzp, &out];

    let mut push = Vec::new();
    for v in [m as u32, (k / 4) as u32, n as u32, 0, 0, 0] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    let grid = [
        (n as u32).div_ceil(TILE_SIZE),
        (m as u32).div_ceil(TILE_SIZE),
        1,
    ];
    ctx.dispatch(reference, &buffers, &push, grid)?;
    let expected = ctx.download(&out)?;

    let mut push = Vec::new();
    for v in [m as u32, k as u32, n as u32, 0, 0] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    let picked = matmul_integer::coop_variant_for(ctx, m, n).map_or("none", |v| v.key);
    let gop = 2.0 * (m * k * n) as f64 / 1e9;
    println!("M={m} K={k} N={n}  ({gop:.2} GOP; picked: {picked})");
    let mut times = Vec::with_capacity(variants.len());
    for variant in variants {
        let [bm, bn] = variant.block;
        let grid = [(n as u32).div_ceil(bn), (m as u32).div_ceil(bm), 1];
        ctx.upload(&out, &vec![0u8; m * n * 4])?;
        ctx.dispatch(&variant.pipeline, &buffers, &push, grid)?;
        let exact = ctx.download(&out)? == expected;
        let mut samples =
            ctx.measure_dispatch_gpu(&variant.pipeline, &buffers, &push, grid, 9, 5)?;
        samples.sort_unstable();
        let ms = samples[samples.len() / 2] as f64 / 1e6;
        println!(
            "  {:<32} {:>4} groups {ms:8.3} ms  {:6.1} TOPS  {}",
            variant.name,
            grid[0] * grid[1],
            gop / ms,
            if exact { "exact" } else { "MISMATCH" }
        );
        times.push(ms);
    }
    for buffer in [a, b, azp, bzp, out] {
        ctx.destroy_buffer(buffer);
    }
    Ok(times)
}

fn main() -> Result<(), Err> {
    let ctx = VkContext::new()?;
    let compiled = matmul_integer::coop_variants(&ctx);
    if compiled.is_empty() {
        println!("{}: no u8×u8 cooperative matrix variant", ctx.device_name);
        return Ok(());
    }
    println!("{}", ctx.device_name);

    let mut variants = Vec::new();
    for v in compiled {
        variants.push(Variant {
            name: v.key.to_owned(),
            pipeline: ctx.create_pipeline(&v.spirv(), COOP_BINDINGS, COOP_PUSH_BYTES)?,
            block: v.block,
        });
    }
    for arg in std::env::args().skip(1) {
        let parts: Vec<&str> = arg.split(':').collect();
        let [path, tm, tn] = parts[..] else {
            return Err(format!("expected path.spv:BM/16:BN/16, got {arg}").into());
        };
        variants.push(Variant {
            name: arg.clone(),
            pipeline: ctx.create_pipeline(
                &spirv_words(&std::fs::read(path)?),
                COOP_BINDINGS,
                COOP_PUSH_BYTES,
            )?,
            block: [
                tm.parse::<u32>()? * COOP_TILE,
                tn.parse::<u32>()? * COOP_TILE,
            ],
        });
    }
    let reference = ctx.create_pipeline(
        &compile_wgsl(&matmul_integer::matmul(ctx.has_integer_dot_product))?,
        MATMUL_BINDINGS,
        MATMUL_PUSH_BYTES,
    )?;

    println!("\nSAM 3 vision encoder");
    let mut totals = vec![0.0f64; variants.len()];
    for (m, k, n, nodes) in ENCODER {
        let times = run_shape(&ctx, &reference, &variants, (m, k, n))?;
        for (total, ms) in totals.iter_mut().zip(times) {
            *total += ms * nodes as f64;
        }
    }
    println!("encoder total (192 nodes):");
    for (variant, total) in variants.iter().zip(&totals) {
        println!("  {:<32} {total:8.1} ms", variant.name);
    }

    println!("\nSAM 3 text encoder and text detector, per prompt");
    let mut totals = vec![0.0f64; variants.len()];
    for (m, k, n, nodes) in PER_PROMPT {
        let times = run_shape(&ctx, &reference, &variants, (m, k, n))?;
        for (total, ms) in totals.iter_mut().zip(times) {
            *total += ms * nodes as f64;
        }
    }
    println!("per-prompt total (each shape on one variant):");
    for (variant, total) in variants.iter().zip(&totals) {
        println!("  {:<32} {total:8.1} ms", variant.name);
    }

    println!("\nresnet50-int8 convolutions as GEMMs");
    for shape in RESNET50_CONV {
        run_shape(&ctx, &reference, &variants, shape)?;
    }

    for variant in variants {
        ctx.destroy_pipeline(variant.pipeline);
    }
    ctx.destroy_pipeline(reference);
    Ok(())
}
