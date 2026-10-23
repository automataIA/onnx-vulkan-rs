//! Value-dependent shape handling for ONNX `NonZero`.
//!
//! The only host readback is one `u32` count per 256 input elements. The
//! result length cannot be known before evaluating the values, so those block
//! counts are needed to allocate the exact output. Input values and output
//! coordinates stay on the GPU.

use crate::host_ops::{BOOL, FLOAT, INT8, INT32, INT64, UINT8};
use crate::interp::with_pipeline;
use crate::{
    DeviceBuffer, DeviceTensor, ExecutionEnv, NodeIr, PipelineKey, Tensor, device_storage_bytes,
};
use anyhow::{Context as _, Result, ensure};
use vk_compute::compile_wgsl;

const WORKGROUP: usize = 256;
const MAX_ELEMENTS: usize = u32::MAX as usize;

/// Types with a simple fixed-width representation in the shader below.
pub(crate) fn supported_dtype(dtype: i32) -> bool {
    matches!(dtype, BOOL | FLOAT | INT8 | INT32 | INT64 | UINT8)
}

pub(crate) fn execute(env: &mut ExecutionEnv<'_, '_>, node: &NodeIr) -> Result<()> {
    ensure!(
        node.inputs.len() == 1 && !node.inputs[0].is_empty(),
        "NonZero expects one input"
    );
    ensure!(
        node.outputs.len() == 1 && !node.outputs[0].is_empty(),
        "NonZero expects one output"
    );

    let input_name = &node.inputs[0];
    let dtype = env.dtype_of(input_name)?;
    ensure!(
        supported_dtype(dtype),
        "NonZero input dtype {dtype} is not supported"
    );
    let shape = env.shape_of(input_name)?;
    ensure!(
        shape.iter().all(|&dim| dim >= 0),
        "NonZero input has a negative dimension"
    );
    let expected = shape
        .iter()
        .try_fold(1usize, |count, &dim| {
            count.checked_mul(usize::try_from(dim).ok()?)
        })
        .context("NonZero input element count overflow")?;
    env.ensure_device_dtype(input_name)?;
    let input = env.device(input_name)?;
    ensure!(
        input.elem_count == expected,
        "NonZero input shape and storage disagree"
    );
    ensure!(
        expected <= MAX_ELEMENTS,
        "NonZero input exceeds the u32 indexing limit"
    );

    let ctx = env.context();
    let blocks = expected.div_ceil(WORKGROUP);
    ensure!(
        blocks <= ctx.compute_limits().max_workgroup_count[0] as usize,
        "NonZero input exceeds the Vulkan workgroup-count limit"
    );
    let mut nnz = 0usize;
    let mut offsets = Vec::<u32>::new();

    if blocks != 0 {
        let count_bytes = blocks
            .checked_mul(4)
            .context("NonZero block-count buffer overflow")?;
        ensure!(
            count_bytes as u64 <= ctx.compute_limits().max_storage_buffer_bytes,
            "NonZero block-count buffer exceeds the Vulkan storage-buffer limit"
        );
        let counts = ctx.create_storage_buffer(count_bytes.max(4) as u64)?;
        ctx.stream_fill(&counts, 0, count_bytes as u64, 0)?;
        let source = count_shader(dtype);
        let count_push = u32::try_from(expected)
            .context("NonZero input too large for shader")?
            .to_le_bytes();
        with_pipeline(
            env.cache(),
            PipelineKey::tactic("NonZero", format!("count:dtype={dtype}")),
            || ctx.create_pipeline(&compile_wgsl(&source)?, 2, 4),
            |pipeline| {
                ctx.stream_dispatch(
                    pipeline,
                    &[input.buffer(), &counts],
                    &count_push,
                    [blocks as u32, 1, 1],
                )
            },
        )?;

        // This synchronization is necessary: output allocation is data-dependent.
        let bytes = ctx.stream_download(&counts, count_bytes)?;
        offsets.reserve(blocks);
        for (block, chunk) in bytes.chunks_exact(4).enumerate() {
            let count = u32::from_le_bytes(chunk.try_into().expect("four-byte count"));
            let start = u32::try_from(nnz).context("NonZero output count exceeds u32")?;
            offsets.push(start);
            nnz = nnz
                .checked_add(count as usize)
                .context("NonZero output count overflow")?;
            ensure!(
                nnz <= expected,
                "NonZero block {block} reported an invalid count"
            );
        }
    }

    let rank = shape.len();
    let rank_u32 = u32::try_from(rank).context("NonZero rank exceeds the shader limit")?;
    let rank_i64 = i64::try_from(rank).context("NonZero rank exceeds the output shape limit")?;
    let output_elems = rank
        .checked_mul(nnz)
        .context("NonZero output element count overflow")?;
    let output_words = output_elems
        .checked_mul(2)
        .context("NonZero output byte count overflow")?;
    ensure!(
        output_words <= u32::MAX as usize,
        "NonZero output exceeds the u32 shader-indexing limit"
    );
    let output_bytes = output_words
        .checked_mul(4)
        .context("NonZero output byte count overflow")?;
    ensure!(
        output_bytes as u64 <= ctx.compute_limits().max_storage_buffer_bytes,
        "NonZero output exceeds the Vulkan storage-buffer limit"
    );
    let output = ctx.create_storage_buffer(device_storage_bytes(INT64, output_elems)?)?;

    if blocks != 0 && nnz != 0 && rank != 0 {
        let offset_bytes = offsets
            .len()
            .checked_mul(4)
            .context("NonZero offsets byte count overflow")?;
        let block_offsets = ctx.create_storage_buffer(offset_bytes.max(4) as u64)?;
        let offset_data: Vec<u8> = offsets
            .iter()
            .flat_map(|offset| offset.to_le_bytes())
            .collect();
        ctx.stream_upload(&block_offsets, &offset_data)?;

        // Metadata is [stride_0..stride_rank, dim_0..dim_rank], all u32. An
        // input dimension is bounded by the total element count already checked.
        let metadata_bytes = rank
            .checked_mul(8)
            .context("NonZero dimensions metadata size overflow")?;
        let mut metadata = Vec::with_capacity(metadata_bytes);
        let mut stride = 1usize;
        let mut strides = vec![0u32; rank];
        for axis in (0..rank).rev() {
            strides[axis] = u32::try_from(stride).context("NonZero stride exceeds u32")?;
            stride = stride
                .checked_mul(shape[axis] as usize)
                .context("NonZero stride overflow")?;
        }
        metadata.extend(strides.iter().flat_map(|x| x.to_le_bytes()));
        metadata.extend(shape.iter().flat_map(|&x| (x as u32).to_le_bytes()));
        let dimensions = ctx.create_storage_buffer(metadata.len().max(4) as u64)?;
        if !metadata.is_empty() {
            ctx.stream_upload(&dimensions, &metadata)?;
        }

        let source = write_shader(dtype);
        let mut push = Vec::with_capacity(12);
        push.extend_from_slice(&(expected as u32).to_le_bytes());
        push.extend_from_slice(&(nnz as u32).to_le_bytes());
        push.extend_from_slice(&rank_u32.to_le_bytes());
        with_pipeline(
            env.cache(),
            PipelineKey::tactic("NonZero", format!("write:dtype={dtype}")),
            || ctx.create_pipeline(&compile_wgsl(&source)?, 4, 12),
            |pipeline| {
                ctx.stream_dispatch(
                    pipeline,
                    &[input.buffer(), &block_offsets, &dimensions, &output],
                    &push,
                    [blocks as u32, 1, 1],
                )
            },
        )?;
    }

    env.set(
        &node.outputs[0],
        Tensor::Device(DeviceTensor {
            dtype: INT64,
            shape: vec![
                rank_i64,
                i64::try_from(nnz).context("NonZero count exceeds i64")?,
            ],
            elem_count: output_elems,
            buf: DeviceBuffer::Owned(output),
        }),
    );
    Ok(())
}

fn nonzero_expr(dtype: i32) -> &'static str {
    match dtype {
        FLOAT => "(x[i] & 0x7fffffffu) != 0u",
        INT64 => "(x[2u * i] != 0u || x[2u * i + 1u] != 0u)",
        BOOL | INT8 | UINT8 => "((x[i >> 2u] >> ((i & 3u) * 8u)) & 255u) != 0u",
        INT32 => "x[i] != 0u",
        _ => unreachable!("dtype is checked before shader generation"),
    }
}

fn count_shader(dtype: i32) -> String {
    format!(
        r#"
struct Params {{ len: u32 }};
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> counts: array<atomic<u32>>;
var<immediate> params: Params;
fn nonzero(i: u32) -> bool {{ return {}; }}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {{
    let i = gid.x;
    if (i < params.len && nonzero(i)) {{ atomicAdd(&counts[wg.x], 1u); }}
}}
"#,
        nonzero_expr(dtype)
    )
}

fn write_shader(dtype: i32) -> String {
    format!(
        r#"
struct Params {{ len: u32, nnz: u32, rank: u32 }};
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> dimensions: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
var<immediate> params: Params;
var<workgroup> scan_a: array<u32, 256>;
var<workgroup> scan_b: array<u32, 256>;
fn nonzero(i: u32) -> bool {{ return {}; }}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {{
    let lane = lid3.x;
    let i = wg.x * 256u + lane;
    var flag = 0u;
    if (i < params.len && nonzero(i)) {{ flag = 1u; }}
    scan_a[lane] = flag;
    workgroupBarrier();
    var step = 1u;
    loop {{
        if (step >= 256u) {{ break; }}
        var sum = scan_a[lane];
        if (lane >= step) {{ sum += scan_a[lane - step]; }}
        scan_b[lane] = sum;
        workgroupBarrier();
        scan_a[lane] = scan_b[lane];
        workgroupBarrier();
        step *= 2u;
    }}
    if (flag != 0u) {{
        let position = offsets[wg.x] + scan_a[lane] - 1u;
        for (var axis = 0u; axis < params.rank; axis += 1u) {{
            let stride = dimensions[axis];
            let dim = dimensions[params.rank + axis];
            let coordinate = (i / stride) % dim;
            let slot = (axis * params.nnz + position) * 2u;
            out[slot] = coordinate;
            out[slot + 1u] = 0u;
        }}
    }}
}}
"#,
        nonzero_expr(dtype)
    )
}
