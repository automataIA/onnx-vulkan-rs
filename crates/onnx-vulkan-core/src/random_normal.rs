//! GPU implementation of ONNX `RandomNormalLike`.
//!
//! The stream is deterministic for an explicit ONNX seed, graph node identity,
//! and invocation number. It uses a counter hash and Box–Muller transform; the
//! contract is a normal distribution, not bitwise compatibility with ORT's RNG.

use crate::host_ops::FLOAT;
use crate::interp::with_pipeline;
use crate::{DeviceTensor, ExecutionEnv, NodeIr, Tensor, device_storage_bytes};
use anyhow::{Context as _, Result, bail, ensure};
use vk_compute::ComputePipeline;
use vk_compute::compile_wgsl;

const BINDINGS: u32 = 1;
const PUSH_BYTES: u32 = 24;
const WG: u32 = 128;

const SHADER: &str = r#"
struct Params { count: u32, seed_lo: u32, seed_hi: u32, mean: f32, scale: f32, _pad: u32 };
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
var<immediate> p: Params;
fn hash32(x0: u32) -> u32 {
  var x = x0;
  x = x ^ (x >> 16u); x = x * 0x7feb352du;
  x = x ^ (x >> 15u); x = x * 0x846ca68bu;
  return x ^ (x >> 16u);
}
fn random_word(index: u32, lane: u32) -> u32 {
  let lo = p.seed_lo + index * 0x9e3779b9u + lane * 0x85ebca6bu;
  let hi = p.seed_hi ^ (index * 0xc2b2ae35u) ^ (lane * 0x27d4eb2fu);
  return hash32(lo ^ hash32(hi));
}
fn uniform_open(word: u32) -> f32 {
  // 23 mantissa bits and an offset produce strictly positive values below 1.
  return (f32(word >> 9u) + 1.0) / 8388609.0;
}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.count) { return; }
  let u1 = uniform_open(random_word(i, 0u));
  let u2 = uniform_open(random_word(i, 1u));
  let z = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
  output[i] = p.mean + p.scale * z;
}
"#;

pub(crate) fn execute(env: &mut ExecutionEnv<'_, '_>, node: &NodeIr) -> Result<()> {
    ensure!(node.inputs.len() == 1, "RandomNormalLike expects one input");
    ensure!(
        node.outputs.len() == 1,
        "RandomNormalLike expects one output"
    );

    let mean = attr_f32(node, "mean", 0.0)?;
    let scale = attr_f32(node, "scale", 1.0)?;
    ensure!(mean.is_finite(), "RandomNormalLike mean must be finite");
    ensure!(
        scale.is_finite() && scale >= 0.0,
        "RandomNormalLike scale must be finite and nonnegative"
    );
    let output_dtype = match node.attrs.get("dtype") {
        Some(value) => value
            .as_i64()
            .context("RandomNormalLike dtype must be an integer")?,
        None => i64::from(env.dtype_of(&node.inputs[0])?),
    };
    if output_dtype != i64::from(FLOAT) {
        bail!("RandomNormalLike supports float32 output only (requested dtype {output_dtype})");
    }

    // shape_of reads tensor metadata only, allowing arbitrary input types and
    // avoiding an upload/download when the input exists solely for its shape.
    let shape = env.shape_of(&node.inputs[0])?;
    let count = shape.iter().try_fold(1usize, |count, &dimension| {
        ensure!(
            dimension >= 0,
            "RandomNormalLike has a negative input dimension"
        );
        count
            .checked_mul(dimension as usize)
            .context("RandomNormalLike element count overflow")
    })?;
    let count_u32 = u32::try_from(count)
        .context("RandomNormalLike element count exceeds u32 dispatch range")?;
    let output_bytes = device_storage_bytes(FLOAT, count)?;
    let limits = env.context().compute_limits();
    ensure!(
        output_bytes <= limits.max_storage_buffer_bytes,
        "RandomNormalLike output exceeds the device storage-buffer limit"
    );
    let node_key = format!("{}:{}:{}", node.domain, node.name, node.outputs[0]);
    let seed = node
        .attrs
        .get("seed")
        .map(|value| {
            value
                .as_f32()
                .context("RandomNormalLike seed must be a float")
        })
        .transpose()?;
    if let Some(seed) = seed {
        ensure!(seed.is_finite(), "RandomNormalLike seed must be finite");
    }
    let stream_seed = env.cache().next_random_seed(&node_key, seed);
    let ctx = env.context();
    let output = ctx.create_storage_buffer(output_bytes)?;

    if count_u32 > 0 {
        let mut push = Vec::with_capacity(PUSH_BYTES as usize);
        push.extend_from_slice(&count_u32.to_le_bytes());
        push.extend_from_slice(&(stream_seed as u32).to_le_bytes());
        push.extend_from_slice(&((stream_seed >> 32) as u32).to_le_bytes());
        push.extend_from_slice(&mean.to_le_bytes());
        push.extend_from_slice(&scale.to_le_bytes());
        push.extend_from_slice(&0u32.to_le_bytes());
        let groups = count_u32.div_ceil(WG);
        ensure!(groups > 0, "RandomNormalLike dispatch grid is empty");
        ensure!(
            WG <= limits.max_workgroup_size[0],
            "RandomNormalLike requires a workgroup size of at least {WG}"
        );
        ensure!(
            groups <= limits.max_workgroup_count[0],
            "RandomNormalLike dispatch exceeds the device workgroup-count limit"
        );
        with_pipeline(
            env.cache(),
            "RandomNormalLike",
            || ctx.create_pipeline(&compile_wgsl(SHADER)?, BINDINGS, PUSH_BYTES),
            |pipeline: &ComputePipeline| {
                ctx.stream_dispatch(pipeline, &[&output], &push, [groups, 1, 1])
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DeviceTensor {
            dtype: FLOAT,
            shape,
            elem_count: count,
            buf: crate::DeviceBuffer::Owned(output),
        }),
    );
    Ok(())
}

fn attr_f32(node: &NodeIr, key: &str, default: f32) -> Result<f32> {
    node.attrs
        .get(key)
        .map(|value| {
            value
                .as_f32()
                .with_context(|| format!("RandomNormalLike {key} must be a float"))
        })
        .unwrap_or(Ok(default))
}
