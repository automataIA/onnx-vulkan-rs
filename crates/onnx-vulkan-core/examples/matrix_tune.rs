//! Measure the bounded rank-2 fp32 MatMul/Gemm family used by model tuning.
//!
//! The process is deliberately a small JSON-lines runner.  The inventory
//! sends one geometry per line and receives one complete, correctness-checked
//! tuning record.  MatMul measures the shipped 64x64 and 16x16 kernels;
//! Gemm measures 64x64 and 16x16 kernels, including transpose and C-bias
//! indexing.  Batched MatMul, GEMV (`M = 1`), and other ranks belong to a
//! future family and are rejected explicitly here.

use anyhow::{Result, bail, ensure};
use onnx_vulkan_core::shaders::{gemm, matmul_fp32, matrix};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
use vk_compute::{ComputePipeline, GpuBuffer, VkContext, compile_wgsl};

const SCHEMA_VERSION: u64 = 1;
const SAMPLES: u32 = 20;
const MAX_REL_ERROR: f32 = 1.0e-4;
const MAX_BUFFER_BYTES: usize = 256 * 1024 * 1024;

fn pseudo(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        })
        .collect()
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn floats(raw: &[u8]) -> Result<Vec<f32>> {
    ensure!(
        raw.len().is_multiple_of(4),
        "GPU output is not fp32-aligned"
    );
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| {
            let mut value = [0u8; 4];
            value.copy_from_slice(chunk);
            f32::from_le_bytes(value)
        })
        .collect())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        result.push(DIGITS[(byte & 0xf) as usize] as char);
    }
    result
}

fn number(geometry: &Value, name: &str) -> Result<usize> {
    let value = geometry
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("geometry.{name} must be a non-negative integer"))?;
    usize::try_from(value).map_err(|_| anyhow::anyhow!("geometry.{name} is too large"))
}

fn number_or(geometry: &Value, name: &str, default: usize) -> Result<usize> {
    match geometry.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("geometry.{name} must be a non-negative integer"))
            .and_then(|value| {
                usize::try_from(value).map_err(|_| anyhow::anyhow!("geometry.{name} is too large"))
            }),
    }
}

fn bool_or(geometry: &Value, name: &str, default: bool) -> Result<bool> {
    match geometry.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| anyhow::anyhow!("geometry.{name} must be boolean")),
    }
}

fn median(samples: &[u64]) -> Result<u64> {
    ensure!(!samples.is_empty(), "GPU measurement returned no samples");
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    Ok(sorted[sorted.len() / 2])
}

fn measurement(samples: Vec<u64>) -> Result<Value> {
    let median_gpu_ns = median(&samples)?;
    let min_gpu_ns = *samples
        .iter()
        .min()
        .ok_or_else(|| anyhow::anyhow!("GPU measurement returned no samples"))?;
    let max_gpu_ns = *samples
        .iter()
        .max()
        .ok_or_else(|| anyhow::anyhow!("GPU measurement returned no samples"))?;
    ensure!(min_gpu_ns > 0, "GPU timestamp sample was zero");
    Ok(json!({
        "samples": samples.len(),
        "median_gpu_ns": median_gpu_ns,
        "min_gpu_ns": min_gpu_ns,
        "max_gpu_ns": max_gpu_ns,
        "diagnostic_samples_gpu_ns": samples,
    }))
}

fn compare(reference: &[f32], candidate: &[f32]) -> Result<f32> {
    ensure!(
        reference.len() == candidate.len(),
        "GPU output has the wrong length"
    );
    let mut max_rel = 0.0f32;
    for (&expected, &actual) in reference.iter().zip(candidate) {
        ensure!(
            expected.is_finite() && actual.is_finite(),
            "GPU output is non-finite"
        );
        let denominator = expected.abs().max(1.0);
        max_rel = max_rel.max((actual - expected).abs() / denominator);
    }
    Ok(max_rel)
}

fn push_u32(push: &mut Vec<u8>, value: u32) {
    push.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(push: &mut Vec<u8>, value: f32) {
    push.extend_from_slice(&value.to_le_bytes());
}

struct Case {
    a: GpuBuffer,
    b: GpuBuffer,
    c: GpuBuffer,
    out: GpuBuffer,
    push: Vec<u8>,
    m: usize,
    n: usize,
    matmul_bindings: bool,
    reference: Vec<f32>,
}

impl Case {
    fn buffer_refs(&self) -> Vec<&GpuBuffer> {
        if self.matmul_bindings {
            vec![&self.a, &self.b, &self.out]
        } else {
            vec![&self.a, &self.b, &self.c, &self.out]
        }
    }

    fn dispatch(&self, ctx: &VkContext, pipeline: &ComputePipeline, grid: [u32; 3]) -> Result<()> {
        let buffers = self.buffer_refs();
        ctx.dispatch(pipeline, &buffers, &self.push, grid)?;
        Ok(())
    }

    fn timestamps(
        &self,
        ctx: &VkContext,
        pipeline: &ComputePipeline,
        grid: [u32; 3],
        samples: u32,
    ) -> Result<Vec<u64>> {
        let buffers = self.buffer_refs();
        ctx.measure_dispatch_gpu(pipeline, &buffers, &self.push, grid, samples, 1)
    }

    fn output(&self, ctx: &VkContext) -> Result<Vec<f32>> {
        floats(&ctx.stream_download(&self.out, self.reference.len() * 4)?)
    }

    fn destroy(self, ctx: &VkContext) {
        for buffer in [self.a, self.b, self.c, self.out] {
            ctx.destroy_buffer(buffer);
        }
    }
}

fn dims(value: &Value) -> Result<(usize, usize, usize)> {
    let m = number(value, "m")?;
    let k = number(value, "k")?;
    let n = number(value, "n")?;
    ensure!(
        m > 0 && k > 0 && n > 0,
        "matrix dimensions must be positive"
    );
    Ok((m, k, n))
}

fn u32_dimension(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| anyhow::anyhow!("geometry.{name} exceeds u32"))
}

fn element_count(left: usize, right: usize, name: &str) -> Result<usize> {
    let count = left
        .checked_mul(right)
        .ok_or_else(|| anyhow::anyhow!("{name} element count overflows usize"))?;
    let bytes = count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("{name} byte count overflows usize"))?;
    ensure!(
        bytes <= MAX_BUFFER_BYTES,
        "{name} exceeds the matrix runner buffer limit"
    );
    Ok(count)
}

fn create_buffer(ctx: &VkContext, values: &[f32]) -> Result<GpuBuffer> {
    let buffer = ctx.create_storage_buffer(u64::try_from(values.len() * 4)?)?;
    ctx.stream_upload(&buffer, &bytes(values))?;
    Ok(buffer)
}

fn matmul_case(ctx: &VkContext, geometry: &Value) -> Result<Case> {
    let (m, k, n) = dims(geometry)?;
    ensure!(
        m > 1,
        "matrix-f32 MatMul tuning rejects M=1 until GEMV tuning exists"
    );
    ensure!(
        m <= u32::MAX as usize && k <= u32::MAX as usize && n <= u32::MAX as usize,
        "MatMul dimensions exceed Vulkan push-constant limits"
    );
    ensure!(
        !bool_or(geometry, "trans_a", false)?,
        "MatMul does not support trans_a"
    );
    ensure!(
        !bool_or(geometry, "trans_b", false)?,
        "MatMul does not support trans_b"
    );
    ensure!(
        number_or(geometry, "c_rows", 0)? == 0 && number_or(geometry, "c_cols", 0)? == 0,
        "MatMul does not accept C"
    );
    ensure!(
        u32::try_from(number_or(
            geometry,
            "alpha_bits",
            1.0f32.to_bits() as usize
        )?)? == 1.0f32.to_bits()
            && u32::try_from(number_or(geometry, "beta_bits", 1.0f32.to_bits() as usize)?)?
                == 1.0f32.to_bits(),
        "MatMul does not accept alpha/beta"
    );

    let a_len = element_count(m, k, "MatMul A")?;
    let b_len = element_count(k, n, "MatMul B")?;
    let output_len = element_count(m, n, "MatMul output")?;
    let a_values = pseudo(a_len, 0x004d_4154_4d55_4c41);
    let b_values = pseudo(b_len, 0x004d_4154_4d55_4c42);
    let mut reference = vec![0.0; output_len];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for depth in 0..k {
                sum = a_values[row * k + depth].mul_add(b_values[depth * n + col], sum);
            }
            reference[row * n + col] = sum;
        }
    }
    let a = create_buffer(ctx, &a_values)?;
    let b = create_buffer(ctx, &b_values)?;
    let c = create_buffer(ctx, &[0.0])?;
    let out = ctx.create_storage_buffer(u64::try_from(reference.len() * 4)?)?;
    let mut push = Vec::with_capacity(matmul_fp32::PUSH_BYTES as usize);
    for value in [
        u32_dimension(m, "m")?,
        u32_dimension(k, "k")?,
        u32_dimension(n, "n")?,
        0,
    ] {
        push_u32(&mut push, value);
    }
    push.resize(matmul_fp32::PUSH_BYTES as usize, 0);
    Ok(Case {
        a,
        b,
        c,
        out,
        push,
        m,
        n,
        matmul_bindings: true,
        reference,
    })
}

fn gemm_case(ctx: &VkContext, geometry: &Value) -> Result<Case> {
    let (m, k, n) = dims(geometry)?;
    ensure!(
        m <= u32::MAX as usize && k <= u32::MAX as usize && n <= u32::MAX as usize,
        "Gemm dimensions exceed Vulkan push-constant limits"
    );
    let trans_a = bool_or(geometry, "trans_a", false)?;
    let trans_b = bool_or(geometry, "trans_b", false)?;
    let alpha_bits = u32::try_from(number_or(
        geometry,
        "alpha_bits",
        1.0f32.to_bits() as usize,
    )?)?;
    let beta_bits = u32::try_from(number_or(geometry, "beta_bits", 1.0f32.to_bits() as usize)?)?;
    let alpha = f32::from_bits(alpha_bits);
    let beta = f32::from_bits(beta_bits);
    ensure!(
        alpha.is_finite() && beta.is_finite(),
        "Gemm alpha/beta must be finite"
    );
    let c_rows = number_or(geometry, "c_rows", 0)?;
    let c_cols = number_or(geometry, "c_cols", 0)?;
    ensure!(
        (c_rows == 0) == (c_cols == 0),
        "Gemm C dimensions must both be zero or positive"
    );
    if c_rows > 0 {
        ensure!(
            c_rows <= m && c_cols <= n,
            "Gemm C is not broadcastable to MxN"
        );
    }
    for (value, name) in [(c_rows, "c_rows"), (c_cols, "c_cols")] {
        ensure!(
            value <= u32::MAX as usize,
            "geometry.{name} exceeds Vulkan push-constant limits"
        );
    }

    let a_len = element_count(m, k, "Gemm A")?;
    let b_len = element_count(k, n, "Gemm B")?;
    let output_len = element_count(m, n, "Gemm output")?;
    let c_len = if c_rows == 0 {
        1
    } else {
        element_count(c_rows, c_cols, "Gemm C")?
    };
    let a_values = pseudo(a_len, 0x0047_454d_4d41);
    let b_values = pseudo(b_len, 0x0047_454d_4d42);
    let c_values = if c_rows == 0 {
        vec![0.0]
    } else {
        pseudo(c_len, 0x0047_454d_4d43)
    };
    let mut reference = vec![0.0; output_len];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for depth in 0..k {
                let a_index = if trans_a {
                    depth * m + row
                } else {
                    row * k + depth
                };
                let b_index = if trans_b {
                    col * k + depth
                } else {
                    depth * n + col
                };
                sum = a_values[a_index].mul_add(b_values[b_index], sum);
            }
            let bias = if c_rows == 0 {
                0.0
            } else {
                let c_row = if c_rows > 1 { row } else { 0 };
                let c_col = if c_cols > 1 { col } else { 0 };
                c_values[c_row * c_cols + c_col]
            };
            reference[row * n + col] = alpha.mul_add(sum, beta * bias);
        }
    }
    let a = create_buffer(ctx, &a_values)?;
    let b = create_buffer(ctx, &b_values)?;
    let c = create_buffer(ctx, &c_values)?;
    let out = ctx.create_storage_buffer(u64::try_from(reference.len() * 4)?)?;
    let mut push = Vec::with_capacity(gemm::PUSH_BYTES as usize);
    for value in [
        u32_dimension(m, "m")?,
        u32_dimension(k, "k")?,
        u32_dimension(n, "n")?,
    ] {
        push_u32(&mut push, value);
    }
    let flags = u32::from(trans_a) | (u32::from(trans_b) << 1) | u32::from(c_rows > 0) << 2;
    push_u32(&mut push, flags);
    push_f32(&mut push, alpha);
    push_f32(&mut push, beta);
    push_u32(&mut push, u32_dimension(c_rows, "c_rows")?);
    push_u32(&mut push, u32_dimension(c_cols, "c_cols")?);
    Ok(Case {
        a,
        b,
        c,
        out,
        push,
        m,
        n,
        matmul_bindings: false,
        reference,
    })
}

fn device_json(ctx: &VkContext) -> Result<Value> {
    let device = ctx.device_fingerprint();
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "device": {
            "name": device.name(), "vendor_id": device.vendor_id(),
            "device_id": device.device_id(), "driver_version": device.driver_version(),
            "api_version": device.api_version(),
            "pipeline_cache_uuid": hex(device.pipeline_cache_uuid()),
            "subgroup_size": device.subgroup_size(), "features": device.features(),
        },
        "implementation_digest": hex(matrix::implementation_fingerprint().as_bytes()),
    }))
}

type Candidate<'a> = (&'a str, &'a ComputePipeline, [u32; 3]);

fn tune_request(
    ctx: &VkContext,
    request: &Value,
    matmul64: &ComputePipeline,
    matmul16: &ComputePipeline,
    gemm64: &ComputePipeline,
    gemm16: &ComputePipeline,
) -> Result<Value> {
    let geometry = request;
    ensure!(
        geometry.get("family").and_then(Value::as_str) == Some("matrix-f32"),
        "request family must be matrix-f32"
    );
    let op = geometry
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("geometry.op must be MatMul or Gemm"))?;
    let (case, candidates): (Case, Vec<Candidate<'_>>) = match op {
        "MatMul" => {
            let case = matmul_case(ctx, geometry)?;
            let grid64 = [
                u32_dimension(case.n.div_ceil(matmul_fp32::TILE_SIZE as usize), "grid_x")?,
                u32_dimension(case.m.div_ceil(matmul_fp32::TILE_SIZE as usize), "grid_y")?,
                1,
            ];
            let grid16 = [
                u32_dimension(
                    case.n.div_ceil(matmul_fp32::SMALL_TILE_SIZE as usize),
                    "grid_x",
                )?,
                u32_dimension(
                    case.m.div_ceil(matmul_fp32::SMALL_TILE_SIZE as usize),
                    "grid_y",
                )?,
                1,
            ];
            (
                case,
                vec![
                    ("matmul64", matmul64, grid64),
                    ("matmul16", matmul16, grid16),
                ],
            )
        }
        "Gemm" => {
            let case = gemm_case(ctx, geometry)?;
            let grid = [
                u32_dimension(case.n.div_ceil(gemm::TILE_SIZE as usize), "grid_x")?,
                u32_dimension(case.m.div_ceil(gemm::TILE_SIZE as usize), "grid_y")?,
                1,
            ];
            {
                let grid16 = [
                    (case.n as u32).div_ceil(16),
                    (case.m as u32).div_ceil(16),
                    1,
                ];
                (
                    case,
                    vec![("gemm64", gemm64, grid), ("gemm16", gemm16, grid16)],
                )
            }
        }
        other => bail!("unsupported matrix operation `{other}`"),
    };
    ctx.flush()?;

    // Verify every candidate before timing it.  A malformed or unsupported
    // geometry therefore cannot become a persisted performance claim.
    let mut correctness = Vec::with_capacity(candidates.len());
    for (_, pipeline, grid) in &candidates {
        case.dispatch(ctx, pipeline, *grid)?;
        correctness.push(compare(&case.reference, &case.output(ctx)?)?);
    }
    let max_rel = correctness.iter().copied().fold(0.0f32, f32::max);
    ensure!(
        max_rel <= MAX_REL_ERROR,
        "matrix candidate failed correctness: max_rel={max_rel:e}"
    );

    let mut samples = vec![Vec::<u64>::new(); candidates.len()];
    if candidates.len() == 1 {
        samples[0] = case.timestamps(ctx, candidates[0].1, candidates[0].2, SAMPLES)?;
    } else {
        for index in 0..SAMPLES {
            for candidate_index in [
                (index as usize) % candidates.len(),
                ((index as usize) + 1) % candidates.len(),
            ] {
                samples[candidate_index].extend(case.timestamps(
                    ctx,
                    candidates[candidate_index].1,
                    candidates[candidate_index].2,
                    1,
                )?);
            }
        }
    }
    let medians = samples
        .iter()
        .map(|sample| median(sample))
        .collect::<Result<Vec<_>>>()?;
    let winner = medians
        .iter()
        .enumerate()
        .min_by_key(|(_, value)| **value)
        .map(|(index, _)| index)
        .ok_or_else(|| anyhow::anyhow!("no matrix candidates were measured"))?;
    let selected_samples = samples.swap_remove(winner);
    let tactic = candidates[winner].0;
    let result = json!({
        "schema_version": SCHEMA_VERSION,
        "family": "matrix-f32",
        "geometry": geometry,
        "device": device_json(ctx)?.get("device").cloned().unwrap_or(Value::Null),
        "implementation_digest": hex(matrix::implementation_fingerprint().as_bytes()),
        "tactic": {"family": "matrix-f32", "id": tactic, "parameters": {}},
        "measurement": measurement(selected_samples)?,
        "correctness": {"kind": "max_rel", "passed": true, "max_rel": max_rel},
        "candidate_count": candidates.len(),
        "interleaved": {"median_gpu_ns": medians, "selected_index": winner},
    });
    case.destroy(ctx);
    Ok(result)
}

fn main() -> Result<()> {
    let context = VkContext::new()?;
    if std::env::args()
        .skip(1)
        .any(|argument| argument == "--device-json")
    {
        println!("{}", serde_json::to_string(&device_json(&context)?)?);
        return Ok(());
    }
    let matmul64 = context.create_pipeline(
        &compile_wgsl(matmul_fp32::MATMUL)?,
        matmul_fp32::BINDINGS,
        matmul_fp32::PUSH_BYTES,
    )?;
    let matmul16 = context.create_pipeline(
        &compile_wgsl(matmul_fp32::MATMUL_SMALL)?,
        matmul_fp32::BINDINGS,
        matmul_fp32::PUSH_BYTES,
    )?;
    let gemm64 =
        context.create_pipeline(&compile_wgsl(gemm::GEMM)?, gemm::BINDINGS, gemm::PUSH_BYTES)?;
    let gemm16 = context.create_pipeline(
        &compile_wgsl(matrix::GEMM16)?,
        gemm::BINDINGS,
        gemm::PUSH_BYTES,
    )?;
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for (line_number, line) in stdin.lock().lines().enumerate() {
        let line = line
            .map_err(|error| anyhow::anyhow!("read request line {}: {error}", line_number + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let request: Value = serde_json::from_str(&line).map_err(|error| {
            anyhow::anyhow!("request line {} is not JSON: {error}", line_number + 1)
        })?;
        let response = tune_request(&context, &request, &matmul64, &matmul16, &gemm64, &gemm16)?;
        writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
        stdout.flush()?;
    }
    context.destroy_pipeline(matmul64);
    context.destroy_pipeline(matmul16);
    context.destroy_pipeline(gemm64);
    context.destroy_pipeline(gemm16);
    Ok(())
}
