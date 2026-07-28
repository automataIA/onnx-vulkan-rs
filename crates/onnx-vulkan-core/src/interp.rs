//! Interpreter for the fused subgraph: **device-resident** execution.
//!
//! Intermediate tensors live in VRAM (`GpuBuffer`): inputs at fused node
//! boundaries are uploaded once, each internal node enqueues dispatches into the
//! `vk-compute` stream (no submit), and only boundary outputs execute
//! download — so an entire fused block costs **1 upload / 1 submit / 1
//! download** instead of per-node sync.
//!
//! Incremental op coverage: [`is_implemented`] lists ops the interpreter
//! can execute; `ep.rs` fuses **only** those (rest stays on CPU EP), so
//! transcription is verifiable as coverage grows. As coverage grows,
//! fused blocks merge and boundaries drop toward ~1.

use crate::KernelCache;
use crate::host_ops::{self, BinOp, FLOAT, HostTensor, INT8, INT32, UINT8};
use crate::shaders::conv::{
    BINDINGS as CONV_F32_BINDINGS, BLOCKED_TILE_SIZE as CONV_BLOCKED_TILE_SIZE,
    PUSH_BYTES as CONV_F32_PUSH_BYTES, SPLIT_REDUCE as CONV_SPLIT_REDUCE,
    SPLIT_REDUCE_BINDINGS as CONV_SPLIT_REDUCE_BINDINGS, TILE_SIZE as CONV_TILE_SIZE,
    blocked as conv_blocked_source, blocked_splitk as conv_blocked_splitk_source,
    direct as conv_direct_source, implicit_gemm as conv_gemm_source,
    prefer_blocked as conv_prefer_blocked, split_k as conv_split_k,
};
use crate::shaders::conv_integer::{
    BINDINGS as CONV_INTEGER_BINDINGS, CONV_INTEGER, PUSH_BYTES as CONV_INTEGER_PUSH_BYTES,
};
use crate::shaders::conv_transpose::{
    BINDINGS as CONV_T_BINDINGS, FILL as CONV_T_FILL, FILL_BINDINGS as CONV_T_FILL_BINDINGS,
    FILL_PUSH_BYTES as CONV_T_FILL_PUSH_BYTES, INTERLEAVE as CONV_T_INTERLEAVE,
    INTERLEAVE_BINDINGS as CONV_T_INTERLEAVE_BINDINGS,
    INTERLEAVE_PUSH_BYTES as CONV_T_INTERLEAVE_PUSH_BYTES, PACK_BINDINGS as CONV_T_PACK_BINDINGS,
    PACK_PHASE as CONV_T_PACK_PHASE, PACK_PUSH_BYTES as CONV_T_PACK_PUSH_BYTES,
    PUSH_BYTES as CONV_T_PUSH_BYTES, PhaseGeom, direct as conv_t_source, phase_gemm_applies,
};
use crate::shaders::elementwise::{
    BINARY as BINARY_TEMPLATE, CAST_DEV, CAST_DEV_BINDINGS, CAST_DEV_PUSH_BYTES, CLIP,
    CLIP_BINDINGS, CLIP_PUSH_BYTES, MAX_RANK, POW_EXPR, UNARY as UNARY_TEMPLATE, UNARY_HELPERS_ERF,
    WHERE, WHERE_BINDINGS, WHERE_PUSH_BYTES,
};
use crate::shaders::gemm::{
    BINDINGS as GEMM_BINDINGS, GEMM, PUSH_BYTES as GEMM_PUSH_BYTES, TILE_SIZE as GEMM_TILE_SIZE,
};
use crate::shaders::grid_sample::{
    BINDINGS as GS_BINDINGS, GRID_SAMPLE, PAD_BORDER, PAD_ZEROS, PUSH_BYTES as GS_PUSH_BYTES,
};
use crate::shaders::matmul_fp32::{
    BINDINGS as MM_BINDINGS, GEMV as MM_GEMV, GEMV_BINDINGS as MM_GEMV_BINDINGS,
    GEMV_COLS as MM_GEMV_COLS, GEMV_PUSH_BYTES as MM_GEMV_PUSH_BYTES,
    GEMV_REDUCE as MM_GEMV_REDUCE, GEMV_REDUCE_BINDINGS as MM_GEMV_REDUCE_BINDINGS,
    MATMUL as MM_MATMUL, MATMUL_SMALL as MM_MATMUL_SMALL, PUSH_BYTES as MM_PUSH_BYTES,
    SMALL_TILE_SIZE as MM_SMALL_TILE_SIZE, TILE_SIZE as MM_TILE_SIZE, gemv_split as mm_gemv_split,
    prefer_blocked as mm_prefer_blocked,
};
use crate::shaders::matmul_integer::{
    COOP_BINDINGS as MMI_COOP_BINDINGS, COOP_PUSH_BYTES as MMI_COOP_PUSH_BYTES,
    FLIP_BINDINGS as MMI_FLIP_BINDINGS, FLIP_BYTES as MMI_FLIP_BYTES, FLIP_KEY as MMI_FLIP_KEY,
    FLIP_PUSH_BYTES as MMI_FLIP_PUSH_BYTES, MATMUL_BINDINGS as MMI_BINDINGS,
    MATMUL_PUSH_BYTES as MMI_PUSH_BYTES, PACK_B as MMI_PACK_B, PACK_BINDINGS as MMI_PACK_BINDINGS,
    PACK_PUSH_BYTES as MMI_PACK_PUSH_BYTES, PACKED_KEY as MMI_PACKED_KEY,
    SIGN_FLIP_BYTE as MMI_SIGN_FLIP_BYTE, SIGN_FLIP_WORD as MMI_SIGN_FLIP_WORD,
    TILE_SIZE as MMI_TILE_SIZE, VECTOR_KEY as MMI_VECTOR_KEY, coop_applies as mmi_coop_applies,
    coop_variant as mmi_coop_variant, matmul as mmi_matmul,
};
use crate::shaders::movement::{
    CONCAT, CONCAT_BINDINGS, CONCAT_PUSH_BYTES, GATHER, GATHER_BINDINGS, GATHER_PUSH_BYTES, PAD,
    PAD_BINDINGS, PAD_PUSH_BYTES, SLICE, SLICE_BINDINGS, SLICE_PUSH_BYTES,
    TRANSPOSE as WGSL_TRANSPOSE, TRANSPOSE_BINDINGS, TRANSPOSE_PUSH_BYTES,
};
use crate::shaders::normalization::{
    LAYERNORM, LAYERNORM_BINDINGS, LAYERNORM_PUSH_BYTES, SOFTMAX, SOFTMAX_BINDINGS,
    SOFTMAX_PUSH_BYTES,
};
use crate::shaders::pooling::{
    AVG_ACC as POOL_AVG_ACC, AVG_FIN as POOL_AVG_FIN, AVG_INIT as POOL_AVG_INIT,
    BINDINGS as POOL_BINDINGS, MAX_ACC as POOL_MAX_ACC, MAX_FIN as POOL_MAX_FIN,
    MAX_INIT as POOL_MAX_INIT, PUSH_BYTES as POOL_PUSH_BYTES, source as pool_source,
};
use crate::shaders::push_vec4s;
use crate::shaders::quantize_linear::{
    BINDINGS as QDQ_BINDINGS, DEQUANTIZE, DEQUANTIZE_I32, PUSH_BYTES as QDQ_PUSH_BYTES, QUANTIZE,
};
use crate::shaders::reduction::{
    BINDINGS as RED_BINDINGS, MAX_ACC as RED_MAX_ACC, MAX_FIN as RED_MAX_FIN,
    MAX_INIT as RED_MAX_INIT, MEAN_ACC as RED_MEAN_ACC, MEAN_FIN as RED_MEAN_FIN,
    MEAN_INIT as RED_MEAN_INIT, MIN_ACC as RED_MIN_ACC, MIN_FIN as RED_MIN_FIN,
    MIN_INIT as RED_MIN_INIT, PUSH_BYTES as RED_PUSH_BYTES, SUM_ACC as RED_SUM_ACC,
    SUM_FIN as RED_SUM_FIN, SUM_INIT as RED_SUM_INIT, source as reduce_source,
};
use crate::shaders::resize::{
    BINDINGS as RESIZE_BINDINGS, COORD_ALIGN_CORNERS, COORD_ASYMMETRIC, COORD_HALF_PIXEL,
    COORD_PYTORCH_HALF_PIXEL, MODE_CUBIC, MODE_LINEAR, MODE_NEAREST, NEAREST_CEIL, NEAREST_FLOOR,
    NEAREST_ROUND_PREFER_CEIL, NEAREST_ROUND_PREFER_FLOOR, PUSH_BYTES as RESIZE_PUSH_BYTES, RESIZE,
};
use crate::{
    AttrValue, DeviceBuffer as BufRef, DeviceTensor as DevTensor, ExecutionEnv, GraphIr, NodeIr,
    Tensor, broadcast, device_storage_bytes, elem_size,
};
use anyhow::{Context as _, Result, bail, ensure};
use std::collections::HashMap;
use vk_compute::{ComputePipeline, GpuBuffer, VkContext, compile_wgsl};

type Env<'context, 'values> = ExecutionEnv<'context, 'values>;

/// Ops executable by interpreter. Must align with branch execution in
/// `exec_node` implement them: `ep.rs` fuses exactly these ops.
pub fn is_implemented(op: &str) -> bool {
    matches!(
        op,
        "Sigmoid"
            | "Relu"
            | "Softmax"
            | "LayerNormalization"
            | "Cast"
            | "Mul"
            | "Add"
            | "Sub"
            | "Div"
            | "Shape"
            | "Reshape"
            | "Unsqueeze"
            | "Squeeze"
            | "Transpose"
            | "Concat"
            | "DynamicQuantizeLinear"
            | "MatMulInteger"
            | "MatMul"
            | "Gather"
            | "Slice"
            | "Where"
            | "Conv"
            | "ConvInteger"
            | "ConvTranspose"
            | "Mod"
            | "QuantizeLinear"
            | "DequantizeLinear"
            | "Pad"
            | "Split"
            | "Floor"
            | "Not"
            | "And"
            | "Equal"
            | "Less"
            | "Greater"
            | "Gelu"
            | "GatherElements"
            | "ScatterND"
            | "TopK"
            | "ConstantOfShape"
            | "Expand"
            | "Range"
            | "Tile"
            | "Constant"
            | "MaxPool"
            | "AveragePool"
            | "GlobalAveragePool"
            | "ReduceMean"
            | "ReduceSum"
            | "ReduceMax"
            | "ReduceMin"
            | "Flatten"
            | "LeakyRelu"
            | "CumSum"
            | "GridSample"
            | "Gemm"
            | "Resize"
            | "Clip"
            | "Erf"
            | "Neg"
            | "Exp"
            | "Log"
            | "Sqrt"
            | "Abs"
            | "Tanh"
            | "Sin"
            | "Cos"
            | "Reciprocal"
            | "Pow"
            | "Min"
            | "Max"
    )
}

/// Like [`is_implemented`], but also inspects node **attributes**.
///
/// The op name alone is not enough: `Resize` has modes the kernels do not
/// cover (`cubic`), pooling has `ceil_mode`, and claiming a node that cannot
/// then be run is a runtime error instead of a CPU fallback
/// (`op-plan.md` §4b). Whoever decides coverage must use this.
pub fn is_implemented_node(node: &NodeIr) -> bool {
    if !is_implemented(&node.op) {
        return false;
    }
    let string_attr = |name: &str, default: &'static str| {
        node.attrs
            .get(name)
            .and_then(AttrValue::as_str)
            .unwrap_or(default)
            .to_string()
    };
    let int_attr = |name: &str, default: i64| {
        node.attrs
            .get(name)
            .and_then(AttrValue::as_i64)
            .unwrap_or(default)
    };
    match node.op.as_str() {
        "Resize" => {
            // exclude_outside = 1 would require zeroing and renormalizing the
            // weights of out-of-border neighbors: not implemented
            int_attr("exclude_outside", 0) == 0
                && matches!(
                    string_attr("mode", "nearest").as_str(),
                    "nearest" | "linear" | "cubic"
                )
                && matches!(
                    string_attr("coordinate_transformation_mode", "half_pixel").as_str(),
                    "half_pixel" | "asymmetric" | "align_corners" | "pytorch_half_pixel"
                )
                && matches!(
                    string_attr("nearest_mode", "round_prefer_floor").as_str(),
                    "round_prefer_floor" | "round_prefer_ceil" | "floor" | "ceil"
                )
        }
        "MaxPool" | "AveragePool" => int_attr("ceil_mode", 0) == 0 && node.outputs.len() == 1,
        // only the form with `axes` as an attribute and a single axis: with axes
        // as input the value is not known when looking at the node
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" => node
            .attrs
            .get("axes")
            .and_then(AttrValue::as_ints)
            .is_some_and(|a| a.len() == 1),
        "GridSample" => {
            string_attr("mode", "bilinear") == "bilinear"
                && matches!(
                    string_attr("padding_mode", "zeros").as_str(),
                    "zeros" | "border"
                )
        }
        "Gelu" => matches!(string_attr("approximate", "none").as_str(), "none" | "tanh"),
        // `add`/`mul`/`min`/`max` accumulate instead of overwriting
        "ScatterND" => string_attr("reduction", "none") == "none",
        "Conv" | "ConvInteger" => matches!(
            string_attr("auto_pad", "NOTSET").as_str(),
            "NOTSET" | "VALID" | "SAME_UPPER" | "SAME_LOWER"
        ),
        // `output_shape` inverts the relationship — pads are derived from the
        // requested output — and `SAME_*` is meaningful only in those terms
        "ConvTranspose" => {
            matches!(
                string_attr("auto_pad", "NOTSET").as_str(),
                "NOTSET" | "VALID"
            ) && !node.attrs.contains_key("output_shape")
        }
        _ => true,
    }
}

/// Runs a closure with the pipeline (cached in the session) for an op key.
fn with_pipeline<R>(
    cache: &KernelCache<'_>,
    key: &'static str,
    build: impl FnOnce() -> Result<ComputePipeline>,
    run: impl FnOnce(&ComputePipeline) -> Result<R>,
) -> Result<R> {
    vk_compute::stats::set_op(key); // Pareto attribution by dispatch type
    let pipeline = cache.pipeline(key, build)?;
    // SAFETY: the cache never removes entries and keeps them in `Box`: the
    // address stays valid for the lifetime of the cache, which outlives execution.
    run(unsafe { &*pipeline })
}

/// Executes the graph nodes on the provided environment, with no host dependency.
///
/// Internal errors (anyhow, with context chain) are flattened into the core's
/// typed error: the public API does not expose `anyhow`.
pub fn execute(ir: &GraphIr, env: &mut Env<'_, '_>) -> crate::Result<()> {
    execute_nodes(ir, env).map_err(|e| crate::Error::Backend(format!("{e:#}")))
}

fn execute_nodes(ir: &GraphIr, env: &mut Env<'_, '_>) -> Result<()> {
    let dead = dead_after(ir);
    for (index, node) in ir.nodes.iter().enumerate() {
        exec_node(env, node)?;
        for name in &dead[index] {
            env.release(name);
        }
    }
    Ok(())
}

/// For each node, the values for which that node is the last reader.
///
/// This is the liveness analysis of the block: a value lives from the node
/// that produces it to its last consumer, not until the end of the graph.
/// Graph outputs never appear — the host reads them after execution — and
/// neither do initializers, which are resident in VRAM and shared across runs.
fn dead_after(ir: &GraphIr) -> Vec<Vec<&str>> {
    let mut last: HashMap<&str, usize> = HashMap::new();
    for (index, node) in ir.nodes.iter().enumerate() {
        for name in node.inputs.iter().filter(|n| !n.is_empty()) {
            last.insert(name.as_str(), index);
        }
    }
    for name in &ir.outputs {
        last.remove(name.as_str());
    }
    for name in ir.initializers.keys() {
        last.remove(name.as_str());
    }
    let mut dead = vec![Vec::new(); ir.nodes.len()];
    for (name, index) in last {
        dead[index].push(name);
    }
    dead
}

/// Executes a single node, inserting outputs into `env`.
fn exec_node(env: &mut Env, node: &NodeIr) -> Result<()> {
    vk_compute::stats::set_op("compile");
    if log::log_enabled!(log::Level::Debug) {
        let ins: Vec<String> = node
            .inputs
            .iter()
            .map(|n| {
                let dev = env.on_device(n);
                let sh = env.shape_of(n).unwrap_or_default();
                format!("{n}{}{sh:?}", if dev { "@dev" } else { "@host" })
            })
            .collect();
        log::debug!("exec {} in={ins:?}", node.op);
    }
    let r = exec_dispatch(env, node);
    if log::log_enabled!(log::Level::Debug) && r.is_ok() {
        for out in &node.outputs {
            if let Some(Tensor::Host(h)) = env.value(out)
                && h.elem_count() <= 12
            {
                log::debug!("  -> {out} host{:?} = {:?}", h.shape, h.to_i64().ok());
            }
        }
    }
    r
}

fn exec_dispatch(env: &mut Env, node: &NodeIr) -> Result<()> {
    match node.op.as_str() {
        "Sigmoid" => unary(env, node, "sigmoid", "1.0 / (1.0 + exp(-v))"),
        "Relu" => unary(env, node, "relu", "max(v, 0.0)"),
        "Softmax" => softmax(env, node),
        "LayerNormalization" => layernorm(env, node),
        "Cast" => cast(env, node),
        "Mul" => elementwise_binary(env, node, "a[off_a] * b[off_b]", BinOp::Mul),
        "Add" => elementwise_binary(env, node, "a[off_a] + b[off_b]", BinOp::Add),
        "Sub" => elementwise_binary(env, node, "a[off_a] - b[off_b]", BinOp::Sub),
        "Div" => elementwise_binary(env, node, "a[off_a] / b[off_b]", BinOp::Div),
        "Shape" => shape_op(env, node),
        "Reshape" => reshape(env, node),
        "Unsqueeze" => unsqueeze(env, node),
        "Squeeze" => squeeze(env, node),
        "Transpose" => transpose(env, node),
        "Concat" => concat_op(env, node),
        "DynamicQuantizeLinear" => dynamic_quantize(env, node),
        "MatMulInteger" => matmul_integer(env, node),
        "MatMul" => matmul_fp32(env, node),
        "Gather" => gather(env, node),
        "Slice" => slice(env, node),
        "Where" => where_op(env, node),
        "Conv" => conv_f32(env, node),
        "ConvInteger" => conv_integer(env, node),
        "ConvTranspose" => conv_transpose(env, node),
        "Mod" => host_mod(env, node),
        "QuantizeLinear" => quantize_linear(env, node),
        "DequantizeLinear" => dequantize_linear(env, node),
        "Pad" => pad(env, node),
        "Split" => split(env, node),
        "Floor" => host_unary(env, node, host_ops::floor),
        "Not" => host_unary(env, node, host_ops::not),
        "And" => host_cmp(env, node, host_ops::CmpOp::And),
        "Equal" => host_cmp(env, node, host_ops::CmpOp::Equal),
        "Less" => host_cmp(env, node, host_ops::CmpOp::Less),
        "Greater" => host_cmp(env, node, host_ops::CmpOp::Greater),
        "GatherElements" => gather_elements(env, node),
        "ScatterND" => scatter_nd(env, node),
        "TopK" => top_k(env, node),
        "ConstantOfShape" => constant_of_shape(env, node),
        "Expand" => expand(env, node),
        "Range" => range(env, node),
        "Tile" => tile(env, node),
        "Constant" => constant(env, node),
        "MaxPool" => pool(env, node, PoolKind::Max),
        "AveragePool" => pool(env, node, PoolKind::Average),
        "GlobalAveragePool" => pool(env, node, PoolKind::GlobalAverage),
        "GridSample" => grid_sample(env, node),
        "ReduceMean" => reduce(env, node, ReduceKind::Mean),
        "ReduceSum" => reduce(env, node, ReduceKind::Sum),
        "ReduceMax" => reduce(env, node, ReduceKind::Max),
        "ReduceMin" => reduce(env, node, ReduceKind::Min),
        "Flatten" => flatten(env, node),
        "LeakyRelu" => leaky_relu(env, node),
        "CumSum" => cumsum(env, node),
        "Gemm" => gemm(env, node),
        "Resize" => resize(env, node),
        "Clip" => clip(env, node),
        "Erf" => unary_with_helpers(env, node, "Erf", "erf_approx(v)", UNARY_HELPERS_ERF),
        "Gelu" => gelu(env, node),
        "Neg" => unary(env, node, "Neg", "-v"),
        "Exp" => unary(env, node, "Exp", "exp(v)"),
        "Log" => unary(env, node, "Log", "log(v)"),
        "Sqrt" => unary(env, node, "Sqrt", "sqrt(v)"),
        "Abs" => unary(env, node, "Abs", "abs(v)"),
        "Tanh" => unary(env, node, "Tanh", "tanh(v)"),
        "Sin" => unary(env, node, "Sin", "sin(v)"),
        "Cos" => unary(env, node, "Cos", "cos(v)"),
        "Reciprocal" => unary(env, node, "Reciprocal", "1.0 / v"),
        "Pow" => elementwise_binary(env, node, POW_EXPR, BinOp::Pow),
        "Min" => elementwise_binary(env, node, "min(a[off_a], b[off_b])", BinOp::Min),
        "Max" => elementwise_binary(env, node, "max(a[off_a], b[off_b])", BinOp::Max),
        other => bail!("compiling EP: op '{other}' not implemented in the interpreter"),
    }
}

/// `Cast`: if the input is a 4-byte device activation (i32/f32) and the
/// destination is i32/f32 → **on-GPU** cast (stays in VRAM, e.g. the i32 of
/// MatMulInteger dequantized to f32 without a round-trip). Otherwise host-side
/// (shape/control: int64/bool/u8).
fn cast(env: &mut Env, node: &NodeIr) -> Result<()> {
    let to = node
        .attrs
        .get("to")
        .and_then(|a| a.as_i64())
        .context("Cast: attributo 'to' assente")? as i32;
    let src = &node.inputs[0];
    let from = env.dtype_of(src)?;
    if env.on_device(src)
        && elem_size(from) == 4
        && matches!(to, FLOAT | INT32)
        && matches!(from, FLOAT | INT32)
    {
        return cast_device(env, node, from, to);
    }
    let x = env.host(src)?;
    let out = host_ops::cast(&x, to)?;
    env.set(&node.outputs[0], Tensor::Host(out));
    Ok(())
}

/// GPU cast between 4-byte types (i32↔f32): an elementwise kernel. from==to =
/// copy. Keeps activations in VRAM.
fn cast_device(env: &mut Env, node: &NodeIr, from: i32, to: i32) -> Result<()> {
    let ctx = env.context();
    let d = env.device(&node.inputs[0])?;
    let (shape, n) = (d.shape.clone(), d.elem_count);
    let out = ctx.create_storage_buffer(device_storage_bytes(to, n)?)?;
    if n > 0 {
        if from == to {
            ctx.stream_copy(d.buffer(), &out, (n * 4) as u64)?;
        } else {
            let mode: u32 = if from == INT32 { 0 } else { 1 }; // 0: i32→f32, 1: f32→i32
            let mut push = Vec::with_capacity(8);
            push.extend_from_slice(&(n as u32).to_le_bytes());
            push.extend_from_slice(&mode.to_le_bytes());
            with_pipeline(
                env.cache(),
                "Cast_dev",
                || {
                    ctx.create_pipeline(
                        &compile_wgsl(CAST_DEV)?,
                        CAST_DEV_BINDINGS,
                        CAST_DEV_PUSH_BYTES,
                    )
                },
                |pipe| {
                    ctx.stream_dispatch(
                        pipe,
                        &[d.buffer(), &out],
                        &push,
                        [(n as u32).div_ceil(256), 1, 1],
                    )
                },
            )?;
        }
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: to,
            shape,
            elem_count: n,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `Shape`: returns the input shape as a host int64 tensor (optional sub-range
/// `start`/`end`). Does not touch data: metadata only.
fn shape_op(env: &mut Env, node: &NodeIr) -> Result<()> {
    let shape = env.shape_of(&node.inputs[0])?;
    let rank = shape.len() as i64;
    let norm = |v: i64| -> i64 {
        let v = if v < 0 { v + rank } else { v };
        v.clamp(0, rank)
    };
    let start = norm(
        node.attrs
            .get("start")
            .and_then(|a| a.as_i64())
            .unwrap_or(0),
    );
    let end = norm(
        node.attrs
            .get("end")
            .and_then(|a| a.as_i64())
            .unwrap_or(rank),
    );
    let slice = if start < end {
        shape[start as usize..end as usize].to_vec()
    } else {
        Vec::new()
    };
    let out = HostTensor::from_i64(vec![slice.len() as i64], &slice);
    env.set(&node.outputs[0], Tensor::Host(out));
    Ok(())
}

/// `Reshape`: changes metadata only (row-major layout unchanged). The new
/// shape (input[1], host int64) may contain -1 (inferred dim) and 0 (copy the
/// corresponding input dim).
fn reshape(env: &mut Env, node: &NodeIr) -> Result<()> {
    let target = env.host(&node.inputs[1])?.to_i64()?;
    let in_shape = env.shape_of(&node.inputs[0])?;
    let new_shape = resolve_reshape(&in_shape, &target)?;
    meta_reshape_out(env, node, new_shape)
}

/// `CumSum`: prefix sum along an axis. The axis is an input, not an attribute.
fn cumsum(env: &mut Env, node: &NodeIr) -> Result<()> {
    let x = env.host(&node.inputs[0])?;
    let axis = env.host(&node.inputs[1])?.to_i64()?;
    ensure!(axis.len() == 1, "CumSum: axis must be a scalar");
    let flag = |name: &str| {
        node.attrs
            .get(name)
            .and_then(AttrValue::as_i64)
            .unwrap_or(0)
            != 0
    };
    let out = host_ops::cumsum(&x, axis[0], flag("exclusive"), flag("reverse"))?;
    env.set(&node.outputs[0], Tensor::Host(out));
    Ok(())
}

/// `LeakyRelu`: `x` if positive, `alpha·x` otherwise. `alpha` is an attribute,
/// so it ends up in the source and every distinct value is a distinct pipeline —
/// as for the other unary ops, which are parameterized by the expression.
fn leaky_relu(env: &mut Env, node: &NodeIr) -> Result<()> {
    let alpha = node
        .attrs
        .get("alpha")
        .and_then(AttrValue::as_f32)
        .unwrap_or(0.01);
    unary(
        env,
        node,
        "LeakyRelu",
        &format!("select({alpha:?} * v, v, v >= 0.0)"),
    )
}

/// `Flatten`: collapses the tensor to 2D around `axis`; it is a
/// `Reshape` with the shape computed instead of read from an input.
fn flatten(env: &mut Env, node: &NodeIr) -> Result<()> {
    let in_shape = env.shape_of(&node.inputs[0])?;
    let rank = in_shape.len() as i64;
    let axis = node
        .attrs
        .get("axis")
        .and_then(AttrValue::as_i64)
        .unwrap_or(1);
    let axis = if axis < 0 { axis + rank } else { axis };
    ensure!(
        (0..=rank).contains(&axis),
        "Flatten: axis {axis} out of range (rank {rank})"
    );
    let split = axis as usize;
    let outer: i64 = in_shape[..split].iter().product();
    let inner: i64 = in_shape[split..].iter().product();
    meta_reshape_out(env, node, vec![outer, inner])
}

/// `Unsqueeze`: inserts size-1 dimensions (axes = input[1] or attr).
fn unsqueeze(env: &mut Env, node: &NodeIr) -> Result<()> {
    let axes = axes_arg(env, node)?;
    let in_shape = env.shape_of(&node.inputs[0])?;
    let out_rank = (in_shape.len() + axes.len()) as i64;
    let mut norm: Vec<i64> = axes
        .iter()
        .map(|&a| if a < 0 { a + out_rank } else { a })
        .collect();
    norm.sort_unstable();
    let mut out = Vec::with_capacity(out_rank as usize);
    let mut it = in_shape.iter();
    for pos in 0..out_rank {
        if norm.binary_search(&pos).is_ok() {
            out.push(1);
        } else {
            out.push(
                *it.next()
                    .context("Unsqueeze: axes incoerenti con l'input")?,
            );
        }
    }
    meta_reshape_out(env, node, out)
}

/// `Squeeze`: removes size-1 dimensions (when axes are given = only those;
/// otherwise all unit dims).
fn squeeze(env: &mut Env, node: &NodeIr) -> Result<()> {
    let in_shape = env.shape_of(&node.inputs[0])?;
    let rank = in_shape.len() as i64;
    let axes: Option<Vec<i64>> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        Some(env.host(&node.inputs[1])?.to_i64()?)
    } else {
        node.attrs
            .get("axes")
            .and_then(|a| a.as_ints())
            .map(|s| s.to_vec())
    };
    let out: Vec<i64> = match axes {
        Some(axes) => {
            let norm: Vec<i64> = axes
                .iter()
                .map(|&a| if a < 0 { a + rank } else { a })
                .collect();
            in_shape
                .iter()
                .enumerate()
                .filter(|(i, _)| !norm.contains(&(*i as i64)))
                .map(|(_, &d)| d)
                .collect()
        }
        None => in_shape.iter().copied().filter(|&d| d != 1).collect(),
    };
    meta_reshape_out(env, node, out)
}

/// Unsqueeze axes: input[1] (opset≥13) or `axes` attribute (legacy).
fn axes_arg(env: &Env, node: &NodeIr) -> Result<Vec<i64>> {
    if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        Ok(env.host(&node.inputs[1])?.to_i64()?)
    } else {
        node.attrs
            .get("axes")
            .and_then(|a| a.as_ints())
            .map(|s| s.to_vec())
            .context("Unsqueeze: axes missing (neither input nor attribute)")
    }
}

/// Resolves a `Reshape` target shape (handles 0 = copy dim, -1 = inferred).
fn resolve_reshape(in_shape: &[i64], target: &[i64]) -> Result<Vec<i64>> {
    let total: i64 = in_shape.iter().product::<i64>().max(0);
    let mut out = Vec::with_capacity(target.len());
    let mut neg: Option<usize> = None;
    let mut known: i64 = 1;
    for (i, &d) in target.iter().enumerate() {
        let dim = if d == 0 {
            *in_shape
                .get(i)
                .context("Reshape: dim 0 senza dim input corrispondente")?
        } else {
            d
        };
        if dim == -1 {
            ensure!(
                neg.is_none(),
                "Reshape: more than one -1 in the target shape"
            );
            neg = Some(i);
            out.push(-1);
        } else {
            known *= dim;
            out.push(dim);
        }
    }
    if let Some(i) = neg {
        ensure!(
            known != 0 && total % known == 0,
            "Reshape: -1 not divisible ({total} / {known})"
        );
        out[i] = total / known;
    }
    Ok(out)
}

/// Pure-metadata op (Reshape/Unsqueeze/Squeeze): same layout, new shape.
/// Host → new `HostTensor` (data cloned); device → GPU copy of the buffer
/// with the new shape (stays in VRAM, no CPU round-trip). The copy avoids
/// aliasing of owned buffers across multiple consumers.
fn meta_reshape_out(env: &mut Env, node: &NodeIr, new_shape: Vec<i64>) -> Result<()> {
    let src = &node.inputs[0];
    if env.on_device(src) {
        let ctx = env.context();
        let d = env.device(src)?;
        let nbytes = d.elem_count * elem_size(d.dtype);
        let (dtype, elem_count) = (d.dtype, d.elem_count);
        let out = ctx.create_storage_buffer(nbytes.max(1) as u64)?;
        if nbytes > 0 {
            ctx.stream_copy(d.buffer(), &out, nbytes as u64)?;
        }
        env.set(
            &node.outputs[0],
            Tensor::Device(DevTensor {
                dtype,
                shape: new_shape,
                elem_count,
                buf: BufRef::Owned(out),
            }),
        );
    } else {
        let h = env.host(src)?;
        env.set(
            &node.outputs[0],
            Tensor::Host(HostTensor::new(h.dtype, new_shape, h.data)),
        );
    }
    Ok(())
}

/// `Transpose`: permutes axes (attr `perm`, default = reversal). Device
/// (f32 activation) via a reused GPU kernel; host (shape/control) on CPU.
fn transpose(env: &mut Env, node: &NodeIr) -> Result<()> {
    let src = &node.inputs[0];
    let in_shape = env.shape_of(src)?;
    let rank = in_shape.len();
    let perm: Vec<usize> = match node.attrs.get("perm").and_then(|a| a.as_ints()) {
        Some(p) => p.iter().map(|&x| x as usize).collect(),
        None => (0..rank).rev().collect(),
    };
    ensure!(
        perm.len() == rank,
        "Transpose: perm rank {} != {rank}",
        perm.len()
    );
    let out_shape: Vec<i64> = perm.iter().map(|&p| in_shape[p]).collect();
    // GPU kernel only for 4-byte dtypes (f32/i32). A quantized activation
    // (u8/i8) goes host-side: its consumer (e.g. MatMulInteger) is still on
    // the CPU EP anyway, so a host output is natural.
    if env.on_device(src) && elem_size(env.dtype_of(src)?) == 4 {
        transpose_device(env, node, &in_shape, &perm, out_shape)
    } else {
        let h = env.host(src)?;
        let out = host_transpose(&h, &perm, &out_shape)?;
        env.set(&node.outputs[0], Tensor::Host(out));
        Ok(())
    }
}

/// GPU permutation of an activation (reuses `WGSL_TRANSPOSE`; 4-byte dtype).
fn transpose_device(
    env: &mut Env,
    node: &NodeIr,
    in_shape: &[i64],
    perm: &[usize],
    out_shape: Vec<i64>,
) -> Result<()> {
    let ctx = env.context();
    let rank = in_shape.len();
    ensure!(rank <= 8, "Transpose: rank {rank} > 8 not supported");
    // row-major input strides, then reordered by perm (one per output dim)
    let mut in_strides = vec![0u32; rank];
    let mut acc = 1u32;
    for d in (0..rank).rev() {
        in_strides[d] = acc;
        acc *= in_shape[d].max(0) as u32;
    }
    let perm_strides: Vec<u32> = perm.iter().map(|&p| in_strides[p]).collect();
    let out_dims: Vec<u32> = out_shape.iter().map(|&d| d.max(0) as u32).collect();
    let n: usize = out_shape.iter().product::<i64>().max(0) as usize;

    let d = env.device(&node.inputs[0])?;
    ensure!(
        elem_size(d.dtype) == 4,
        "Transpose device: elem size {} != 4 (kernel u32)",
        elem_size(d.dtype)
    );
    let dtype = d.dtype;
    let out = ctx.create_storage_buffer((n.max(1) * 4) as u64)?;
    if n > 0 {
        let mut push = Vec::with_capacity(80);
        push.extend_from_slice(&(n as u32).to_le_bytes());
        push.extend_from_slice(&(rank as u32).to_le_bytes());
        push.extend_from_slice(&0u32.to_le_bytes());
        push.extend_from_slice(&0u32.to_le_bytes());
        for d in 0..8 {
            push.extend_from_slice(&out_dims.get(d).copied().unwrap_or(1).to_le_bytes());
        }
        for d in 0..8 {
            push.extend_from_slice(&perm_strides.get(d).copied().unwrap_or(0).to_le_bytes());
        }
        with_pipeline(
            env.cache(),
            "Transpose",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(WGSL_TRANSPOSE)?,
                    TRANSPOSE_BINDINGS,
                    TRANSPOSE_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[d.buffer(), &out],
                    &push,
                    [(n as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype,
            shape: out_shape,
            elem_count: n,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// Dtype-generic host-side permutation (per-element bytes).
fn host_transpose(h: &HostTensor, perm: &[usize], out_shape: &[i64]) -> Result<HostTensor> {
    let rank = h.shape.len();
    let es = elem_size(h.dtype);
    ensure!(
        es > 0,
        "Transpose host: dtype {} with unknown size",
        h.dtype
    );
    let mut in_strides = vec![0usize; rank];
    let mut acc = 1usize;
    for d in (0..rank).rev() {
        in_strides[d] = acc;
        acc *= h.shape[d].max(0) as usize;
    }
    let perm_strides: Vec<usize> = perm.iter().map(|&p| in_strides[p]).collect();
    let out_dims: Vec<usize> = out_shape.iter().map(|&d| d.max(0) as usize).collect();
    let n: usize = out_dims.iter().product();
    let mut data = vec![0u8; n * es];
    for o in 0..n {
        let mut src = 0usize;
        let mut acc2 = n;
        for d in 0..rank {
            let sh = out_dims[d].max(1);
            acc2 /= sh;
            src += ((o / acc2.max(1)) % sh) * perm_strides[d];
        }
        data[o * es..(o + 1) * es].copy_from_slice(&h.data[src * es..(src + 1) * es]);
    }
    Ok(HostTensor::new(h.dtype, out_shape.to_vec(), data))
}

/// `Concat` along `axis`. If the inputs are f32 activations → GPU kernel (each
/// input copied at the proper offset in the output, stays in VRAM); otherwise host.
fn concat_op(env: &mut Env, node: &NodeIr) -> Result<()> {
    let axis = node
        .attrs
        .get("axis")
        .and_then(|a| a.as_i64())
        .context("Concat: attributo 'axis' assente")?;
    let ins: Vec<&str> = node
        .inputs
        .iter()
        .map(|s| s.as_str())
        .filter(|s| !s.is_empty())
        .collect();
    ensure!(!ins.is_empty(), "Concat: no input");
    // ONNX: all inputs share the dtype. 4-byte f32 → GPU.
    let on_gpu = env.dtype_of(ins[0])? == FLOAT;
    if on_gpu {
        concat_device(env, node, &ins, axis)
    } else {
        let tensors: Vec<HostTensor> = ins
            .iter()
            .map(|name| env.host(name))
            .collect::<crate::Result<_>>()?;
        let out = host_ops::concat(&tensors, axis)?;
        env.set(&node.outputs[0], Tensor::Host(out));
        Ok(())
    }
}

/// GPU concat of f32 activations: one dispatch per input at the proper axis-offset.
fn concat_device(env: &mut Env, node: &NodeIr, ins: &[&str], axis: i64) -> Result<()> {
    let ctx = env.context();
    for name in ins {
        env.ensure_device(name)?;
    }
    let base = env.device(ins[0])?.shape.clone();
    let rank = base.len() as i64;
    let ax = if axis < 0 { axis + rank } else { axis };
    ensure!(
        (0..rank).contains(&ax),
        "Concat: axis {axis} out of range (rank {rank})"
    );
    let ax = ax as usize;
    let inner: usize = base[ax + 1..].iter().product::<i64>().max(0) as usize;

    let mut out_axis = 0usize;
    for &name in ins {
        out_axis += env.device(name)?.shape[ax].max(0) as usize;
    }
    let mut out_shape = base.clone();
    out_shape[ax] = out_axis as i64;
    let elem_count: usize = out_shape.iter().product::<i64>().max(0) as usize;
    let out = ctx.create_storage_buffer((elem_count.max(1) * 4) as u64)?;

    let mut off = 0u32; // position along the axis in the output
    for &name in ins {
        let d = env.device(name)?;
        let a = d.shape[ax].max(0) as usize;
        let n = d.elem_count;
        if n > 0 {
            let mut push = Vec::with_capacity(20);
            push.extend_from_slice(&(n as u32).to_le_bytes());
            push.extend_from_slice(&(inner as u32).to_le_bytes());
            push.extend_from_slice(&(a as u32).to_le_bytes());
            push.extend_from_slice(&(out_axis as u32).to_le_bytes());
            push.extend_from_slice(&off.to_le_bytes());
            with_pipeline(
                env.cache(),
                "Concat",
                || ctx.create_pipeline(&compile_wgsl(CONCAT)?, CONCAT_BINDINGS, CONCAT_PUSH_BYTES),
                |pipe| {
                    ctx.stream_dispatch(
                        pipe,
                        &[d.buffer(), &out],
                        &push,
                        [(n as u32).div_ceil(256), 1, 1],
                    )
                },
            )?;
        }
        off += a as u32;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: out_shape,
            elem_count,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `DynamicQuantizeLinear`: f32 x → (u8 y, scalar f32 y_scale, scalar u8 y_zp),
/// all on-device. Reuses the 3 core shaders (partial/finalize/quantize).
fn dynamic_quantize(env: &mut Env, node: &NodeIr) -> Result<()> {
    use crate::shaders::dynamic_quantize as dq;
    let ctx = env.context();
    env.ensure_device(&node.inputs[0])?;
    let x = env.device(&node.inputs[0])?;
    let (shape, n) = (x.shape.clone(), x.elem_count);
    ensure!(n > 0, "DynamicQuantizeLinear: empty input");

    let y = ctx.create_storage_buffer(device_storage_bytes(UINT8, n)?)?;
    let y_scale = ctx.create_storage_buffer(4)?;
    let y_zp = ctx.create_storage_buffer(4)?;
    let groups = (n as u32).div_ceil(256).min(1024);
    let partial = ctx.create_storage_buffer(u64::from(groups) * 8)?;

    with_pipeline(
        env.cache(),
        "DQL_partial",
        || ctx.create_pipeline(&compile_wgsl(dq::PARTIAL)?, 2, 4),
        |pipe| {
            ctx.stream_dispatch(
                pipe,
                &[x.buffer(), &partial],
                &(n as u32).to_le_bytes(),
                [groups, 1, 1],
            )
        },
    )?;
    with_pipeline(
        env.cache(),
        "DQL_finalize",
        || ctx.create_pipeline(&compile_wgsl(dq::FINALIZE)?, 3, 4),
        |pipe| {
            ctx.stream_dispatch(
                pipe,
                &[&partial, &y_scale, &y_zp],
                &groups.to_le_bytes(),
                [1, 1, 1],
            )
        },
    )?;
    let words = n.div_ceil(4) as u32;
    with_pipeline(
        env.cache(),
        "DQL_quantize",
        || ctx.create_pipeline(&compile_wgsl(dq::QUANTIZE)?, 4, 4),
        |pipe| {
            ctx.stream_dispatch(
                pipe,
                &[x.buffer(), &y_scale, &y_zp, &y],
                &(n as u32).to_le_bytes(),
                [words.div_ceil(256), 1, 1],
            )
        },
    )?;
    ctx.defer_destroy(partial);

    let outs = &node.outputs;
    env.set(
        &outs[0],
        Tensor::Device(DevTensor {
            dtype: UINT8,
            shape,
            elem_count: n,
            buf: BufRef::Owned(y),
        }),
    );
    if outs.len() > 1 && !outs[1].is_empty() {
        env.set(
            &outs[1],
            Tensor::Device(DevTensor {
                dtype: FLOAT,
                shape: vec![],
                elem_count: 1,
                buf: BufRef::Owned(y_scale),
            }),
        );
    } else {
        ctx.defer_destroy(y_scale);
    }
    if outs.len() > 2 && !outs[2].is_empty() {
        env.set(
            &outs[2],
            Tensor::Device(DevTensor {
                dtype: UINT8,
                shape: vec![],
                elem_count: 1,
                buf: BufRef::Owned(y_zp),
            }),
        );
    } else {
        ctx.defer_destroy(y_zp);
    }
    Ok(())
}

/// `MatMulInteger`: A u8 [.., M, K] × B u8 [K, N] → i32 [.., M, N], scalar
/// per-tensor zero-points. Reuses the tiled 16×16 kernel; B is transposed+packed
/// and cached by name (constant). All on-device.
fn matmul_integer(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    env.ensure_device_dtype(&node.inputs[0])?; // A (u8)
    env.ensure_device_dtype(&node.inputs[2])?; // a_zero_point (u8 scalare)
    env.ensure_device_dtype(&node.inputs[3])?; // b_zero_point (u8 scalare)

    let a = env.device(&node.inputs[0])?;
    let a_shape = a.shape.clone();
    ensure!(
        a_shape.len() >= 2,
        "MatMulInteger: A rank {} < 2",
        a_shape.len()
    );
    let k = *a_shape.last().unwrap() as usize;
    let m: usize = a_shape[..a_shape.len() - 1].iter().product::<i64>().max(0) as usize;
    ensure!(
        k.is_multiple_of(4),
        "MatMulInteger: K={k} non multiplo di 4"
    );

    // B: [K, N] u8 initializer → packed [N, K/4], cached by name.
    let b_name = node.inputs[1].clone();
    let b_shape = env.shape_of(&b_name)?;
    ensure!(
        b_shape.len() == 2,
        "MatMulInteger: B rank {} != 2",
        b_shape.len()
    );
    let (bk, bn) = (b_shape[0] as usize, b_shape[1] as usize);
    ensure!(bk == k, "MatMulInteger: K incompatibile (A {k}, B {bk})");
    let b_dtype = env.dtype_of(&b_name)?;
    let a_dtype = env.dtype_of(&node.inputs[0])?;
    // B is a constant weight and goes through packing anyway: the sign flip is
    // done there once, so the kernel always sees unsigned bytes.
    let packed_ptr = packed_b(env, &b_name, bk, bn, b_dtype == INT8)?;

    let a = env.device(&node.inputs[0])?;
    let a_zp = env.device(&node.inputs[2])?;
    let b_zp = env.device(&node.inputs[3])?;
    let mut out_shape: Vec<i64> = a_shape[..a_shape.len() - 1].to_vec();
    out_shape.push(bn as i64);
    let elem_count = m * bn;
    let out = ctx.create_storage_buffer(device_storage_bytes(INT32, elem_count)?)?;
    if elem_count > 0 {
        let packed: &GpuBuffer = unsafe { &*packed_ptr };
        let buffers = [a.buffer(), packed, a_zp.buffer(), b_zp.buffer(), &out];
        mmi_dispatch(
            env.cache(),
            ctx,
            &buffers,
            MmiProblem {
                m,
                k,
                n: bn,
                // A is the activation, dynamic: flipping it would cost a pass per
                // execution, so it stays with the WGSL kernel
                a_byte_flip: if a_dtype == INT8 {
                    MMI_SIGN_FLIP_WORD
                } else {
                    0
                },
                a_zp_xor: if a_dtype == INT8 {
                    MMI_SIGN_FLIP_BYTE
                } else {
                    0
                },
                b_zp_xor: if b_dtype == INT8 {
                    MMI_SIGN_FLIP_BYTE
                } else {
                    0
                },
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: INT32,
            shape: out_shape,
            elem_count,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// Shape and signature of an integer matmul: `A [M, K] × B [K, N]`, `k` in bytes.
///
/// The sign of the operands is already reduced to masks: `a_byte_flip` is
/// non-zero only if A reaches the kernel with signed bytes (then the WGSL
/// shader has to do it, and the cooperative path is excluded), while the
/// `zp_xor` apply to every operand that *was* signed, because the zero point
/// in the buffer stays the original one.
struct MmiProblem {
    m: usize,
    k: usize,
    n: usize,
    a_byte_flip: u32,
    a_zp_xor: u32,
    b_zp_xor: u32,
}

/// Dispatches the integer matmul on the best variant for this device.
///
/// Three pipelines for the same op, in preference order: cooperative matrix
/// (tensor core, precompiled SPIR-V), integer dot product unit (`OpUDot`),
/// portable vector path. The choice depends on what the driver exposes and on
/// the problem shape — the cooperative variant does not cover signed operands
/// or matrices smaller than a tile — and each variant has a distinct cache
/// key, otherwise pipelines collide.
///
/// `buffers` is `[A, B packed, a_zp, b_zp, out]`.
fn mmi_dispatch(
    cache: &KernelCache<'_>,
    ctx: &VkContext,
    buffers: &[&GpuBuffer; 5],
    p: MmiProblem,
) -> Result<()> {
    let MmiProblem {
        m,
        k,
        n,
        a_byte_flip,
        a_zp_xor,
        b_zp_xor,
    } = p;
    let grid = [
        (n as u32).div_ceil(MMI_TILE_SIZE),
        (m as u32).div_ceil(MMI_TILE_SIZE),
        1,
    ];

    if let Some(v) = mmi_coop_variant(&ctx.coop_u8, ctx.subgroup_size)
        && mmi_coop_applies(v, m, k, n, a_byte_flip != 0)
    {
        let mut push = Vec::with_capacity(MMI_COOP_PUSH_BYTES as usize);
        for value in [m as u32, k as u32, n as u32, a_zp_xor, b_zp_xor] {
            push.extend_from_slice(&value.to_le_bytes());
        }
        return with_pipeline(
            cache,
            v.key,
            || ctx.create_pipeline(&v.spirv(), MMI_COOP_BINDINGS, MMI_COOP_PUSH_BYTES),
            |pipe| ctx.stream_dispatch(pipe, buffers, &push, grid),
        );
    }

    let dot4 = ctx.has_integer_dot_product;
    let mut push = Vec::with_capacity(MMI_PUSH_BYTES as usize);
    for value in [
        m as u32,
        (k / 4) as u32,
        n as u32,
        a_byte_flip,
        a_zp_xor,
        b_zp_xor,
    ] {
        push.extend_from_slice(&value.to_le_bytes());
    }
    with_pipeline(
        cache,
        if dot4 { MMI_PACKED_KEY } else { MMI_VECTOR_KEY },
        || {
            ctx.create_pipeline(
                &compile_wgsl(&mmi_matmul(dot4))?,
                MMI_BINDINGS,
                MMI_PUSH_BYTES,
            )
        },
        |pipe| ctx.stream_dispatch(pipe, buffers, &push, grid),
    )
}

/// Copy of a constant u8 tensor with the sign bit flipped, cached by name.
/// `bytes` is the logical size of the tensor.
///
/// Needed when the signed operand is A, which unlike B does not go through a
/// pack step where the flip can be inserted. Being a weight, the cost is once
/// per session. See `FLIP_BYTES` for the identity.
fn flipped_const(env: &mut Env, name: &str, bytes: usize) -> Result<*const GpuBuffer> {
    let cache = env.cache();
    let key = (format!("{name}#sign-flip"), bytes, 0);
    if let Some(cached) = cache.packed_weight_cached(&key) {
        return Ok(cached);
    }
    let src = env.device(name)?.buffer() as *const GpuBuffer;
    let ctx = env.context();
    let words = bytes.div_ceil(4);
    cache.packed_weight(key, || {
        let dst = ctx.create_storage_buffer((words * 4).max(4) as u64)?;
        let push = (words as u32).to_le_bytes().to_vec();
        with_pipeline(
            cache,
            MMI_FLIP_KEY,
            || {
                ctx.create_pipeline(
                    &compile_wgsl(MMI_FLIP_BYTES)?,
                    MMI_FLIP_BINDINGS,
                    MMI_FLIP_PUSH_BYTES,
                )
            },
            // SAFETY: `src` points to the constant tensor's buffer, alive for
            // the whole execution; the cache does not move it.
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[unsafe { &*src }, &dst],
                    &push,
                    [(words as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
        Ok(dst)
    })
}

/// Returns the packed buffer of B (cached by name), packing it on first
/// request. The pointer stays valid (Box on heap, cache never cleared).
fn packed_b(
    env: &mut Env,
    b_name: &str,
    bk: usize,
    bn: usize,
    flip: bool,
) -> Result<*const GpuBuffer> {
    let cache = env.cache();
    let key = (b_name.to_string(), bk, bn);
    // lookup precedes the host read of the weight: when already packed, the CPU
    // copy is not touched
    if let Some(cached) = cache.packed_weight_cached(&key) {
        return Ok(cached);
    }
    let ctx = env.context();
    let hb = env.host(b_name)?;
    ensure!(
        hb.dtype == UINT8 || hb.dtype == INT8,
        "MatMulInteger: B dtype {} is not uint8/int8",
        hb.dtype
    );
    let k4 = bk / 4;
    cache.packed_weight(key, || {
        let raw = ctx.create_storage_buffer(device_storage_bytes(hb.dtype, bk * bn)?)?;
        ctx.stream_upload(&raw, &hb.data)?;
        let packed = ctx.create_storage_buffer((bn * k4 * 4).max(4) as u64)?;
        let mut push = Vec::with_capacity(MMI_PACK_PUSH_BYTES as usize);
        push.extend_from_slice(&(bk as u32).to_le_bytes());
        push.extend_from_slice(&(bn as u32).to_le_bytes());
        push.extend_from_slice(&if flip { MMI_SIGN_FLIP_WORD } else { 0 }.to_le_bytes());
        with_pipeline(
            cache,
            "MMI_pack",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(MMI_PACK_B)?,
                    MMI_PACK_BINDINGS,
                    MMI_PACK_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[&raw, &packed],
                    &push,
                    [
                        (bn as u32).div_ceil(MMI_TILE_SIZE),
                        (k4 as u32).div_ceil(MMI_TILE_SIZE),
                        1,
                    ],
                )
            },
        )?;
        ctx.defer_destroy(raw);
        Ok(packed)
    })
}

/// `Gather` along `axis`. If `data` is an f32 activation → GPU kernel (indices
/// normalized and loaded into an i32 buffer); otherwise host-side (shape/table
/// int). Output = data.shape[:axis] + indices.shape + data.shape[axis+1:].
fn gather(env: &mut Env, node: &NodeIr) -> Result<()> {
    let axis = node.attrs.get("axis").and_then(|a| a.as_i64()).unwrap_or(0);
    let data = &node.inputs[0];
    if env.dtype_of(data)? == FLOAT {
        gather_device(env, node, axis)
    } else {
        let d = env.host(data)?;
        let idx = env.host(&node.inputs[1])?;
        let out = host_ops::gather(&d, &idx, axis)?;
        env.set(&node.outputs[0], Tensor::Host(out));
        Ok(())
    }
}

/// GPU gather of an f32 activation along `axis`.
fn gather_device(env: &mut Env, node: &NodeIr, axis: i64) -> Result<()> {
    let ctx = env.context();
    // indices → host i64, normalized (+axis_dim if negative), then i32 in VRAM
    let data_shape = env.shape_of(&node.inputs[0])?;
    let rank = data_shape.len() as i64;
    let ax = if axis < 0 { axis + rank } else { axis };
    ensure!(
        (0..rank).contains(&ax),
        "Gather: axis {axis} out of range (rank {rank})"
    );
    let ax = ax as usize;
    let axis_dim = data_shape[ax];
    let inner: usize = data_shape[ax + 1..].iter().product::<i64>().max(1) as usize;
    let outer: usize = data_shape[..ax].iter().product::<i64>().max(1) as usize;

    let idx_host = env.host(&node.inputs[1])?;
    let idx_shape = idx_host.shape.clone();
    let idx_i32: Vec<i32> = idx_host
        .to_i64()?
        .into_iter()
        .map(|mut g| {
            if g < 0 {
                g += axis_dim;
            }
            g as i32
        })
        .collect();
    let idx_count = idx_i32.len().max(1);

    env.ensure_device(&node.inputs[0])?;
    let d = env.device(&node.inputs[0])?;

    let mut out_shape = Vec::with_capacity(rank as usize - 1 + idx_shape.len());
    out_shape.extend_from_slice(&data_shape[..ax]);
    out_shape.extend_from_slice(&idx_shape);
    out_shape.extend_from_slice(&data_shape[ax + 1..]);
    let n = outer * idx_count * inner;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, n)?)?;

    let idx_bytes: Vec<u8> = idx_i32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let idx_buf = ctx.create_storage_buffer(idx_bytes.len().max(4) as u64)?;
    ctx.stream_upload(&idx_buf, &idx_bytes)?;

    if n > 0 {
        let mut push = Vec::with_capacity(16);
        for v in [n as u32, inner as u32, idx_count as u32, axis_dim as u32] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        with_pipeline(
            env.cache(),
            "Gather",
            || ctx.create_pipeline(&compile_wgsl(GATHER)?, GATHER_BINDINGS, GATHER_PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[d.buffer(), &idx_buf, &out],
                    &push,
                    [(n as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    ctx.defer_destroy(idx_buf);
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: out_shape,
            elem_count: n,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// Host-side unary op (Floor/Not): small shape/control input.
fn host_unary(
    env: &mut Env,
    node: &NodeIr,
    f: impl Fn(&HostTensor) -> crate::Result<HostTensor>,
) -> Result<()> {
    let x = env.host(&node.inputs[0])?;
    env.set(&node.outputs[0], Tensor::Host(f(&x)?));
    Ok(())
}

/// Host-side `Mod`: the two remainders (`fmod`) and broadcasting live in
/// `host_ops::modulo`.
fn host_mod(env: &mut Env, node: &NodeIr) -> Result<()> {
    let a = env.host(&node.inputs[0])?;
    let b = env.host(&node.inputs[1])?;
    let fmod = node.attrs.get("fmod").and_then(AttrValue::as_i64) == Some(1);
    env.set(
        &node.outputs[0],
        Tensor::Host(host_ops::modulo(&a, &b, fmod)?),
    );
    Ok(())
}

/// Host-side comparison/logic → bool (And/Equal/Less) with broadcasting.
fn host_cmp(env: &mut Env, node: &NodeIr, op: host_ops::CmpOp) -> Result<()> {
    let a = env.host(&node.inputs[0])?;
    let b = env.host(&node.inputs[1])?;
    env.set(
        &node.outputs[0],
        Tensor::Host(host_ops::compare(&a, &b, op)?),
    );
    Ok(())
}

/// `GatherElements`: per-element selection along an axis. On host because the
/// transformer queue tensors are small (tens of KB) and the real cost was the
/// block split, not the computation.
fn gather_elements(env: &mut Env, node: &NodeIr) -> Result<()> {
    let data = env.host(&node.inputs[0])?;
    let indices = env.host(&node.inputs[1])?;
    let axis = node
        .attrs
        .get("axis")
        .and_then(AttrValue::as_i64)
        .unwrap_or(0);
    let out = host_ops::gather_elements(&data, &indices, axis)?;
    env.set(&node.outputs[0], Tensor::Host(out));
    Ok(())
}

/// `ScatterND` with `reduction = none`. On host: in rfdetr it operates on
/// int64 tensors of two elements.
fn scatter_nd(env: &mut Env, node: &NodeIr) -> Result<()> {
    let data = env.host(&node.inputs[0])?;
    let indices = env.host(&node.inputs[1])?;
    let updates = env.host(&node.inputs[2])?;
    let out = host_ops::scatter_nd(&data, &indices, &updates)?;
    env.set(&node.outputs[0], Tensor::Host(out));
    Ok(())
}

/// `TopK`: `k` comes as an input (opset ≥ 10), always from a constant in the
/// graphs of interest. On host — the reduced axis is on the order of thousands
/// of elements and requires a sort, where the GPU pays off little.
fn top_k(env: &mut Env, node: &NodeIr) -> Result<()> {
    let x = env.host(&node.inputs[0])?;
    let k = env.host(&node.inputs[1])?.to_i64()?;
    ensure!(k.len() == 1, "TopK: k must have a single element");
    let axis = node
        .attrs
        .get("axis")
        .and_then(AttrValue::as_i64)
        .unwrap_or(-1);
    let largest = node
        .attrs
        .get("largest")
        .and_then(AttrValue::as_i64)
        .unwrap_or(1)
        != 0;
    ensure!(k[0] >= 0, "TopK: k negativo ({})", k[0]);
    let (values, indices) = host_ops::top_k(&x, k[0] as usize, axis, largest)?;
    env.set(&node.outputs[0], Tensor::Host(values));
    if node.outputs.len() > 1 && !node.outputs[1].is_empty() {
        env.set(&node.outputs[1], Tensor::Host(indices));
    }
    Ok(())
}

/// `ConstantOfShape`: shape from input[0] (host int), filled with the TENSOR
/// `value` attribute (dtype + bytes of the scalar); default f32 0 if absent.
fn constant_of_shape(env: &mut Env, node: &NodeIr) -> Result<()> {
    let shape = env.host(&node.inputs[0])?.to_i64()?;
    let (dtype, value) = match node.attrs.get("value").and_then(|a| a.as_tensor()) {
        Some(t) => (t.dtype, t.data.clone()),
        None => (FLOAT, vec![0u8; 4]),
    };
    let out = host_ops::const_of_shape(shape, dtype, &value)?;
    env.set(&node.outputs[0], Tensor::Host(out));
    Ok(())
}

/// `Expand`: broadcasts input[0] toward the shape input[1] (host int).
fn expand(env: &mut Env, node: &NodeIr) -> Result<()> {
    let target = env.host(&node.inputs[1])?.to_i64()?;
    let x = env.host(&node.inputs[0])?;
    env.set(
        &node.outputs[0],
        Tensor::Host(host_ops::expand(&x, &target)?),
    );
    Ok(())
}

/// `Tile`: replicates input[0] by repeats (input[1], host int).
fn tile(env: &mut Env, node: &NodeIr) -> Result<()> {
    let repeats = env.host(&node.inputs[1])?.to_i64()?;
    let x = env.host(&node.inputs[0])?;
    env.set(
        &node.outputs[0],
        Tensor::Host(host_ops::tile(&x, &repeats)?),
    );
    Ok(())
}

/// `Range`: sequence [start, limit) with step delta (host int/float scalars).
fn range(env: &mut Env, node: &NodeIr) -> Result<()> {
    let start = env.host(&node.inputs[0])?;
    let limit = env.host(&node.inputs[1])?.to_f32()?[0] as f64;
    let delta = env.host(&node.inputs[2])?.to_f32()?[0] as f64;
    let s0 = start.to_f32()?[0] as f64;
    ensure!(delta != 0.0, "Range: delta 0");
    let count = (((limit - s0) / delta).ceil()).max(0.0) as usize;
    let out = if start.dtype != FLOAT {
        let v: Vec<i64> = (0..count).map(|i| (s0 + i as f64 * delta) as i64).collect();
        HostTensor::from_i64(vec![count as i64], &v)
    } else {
        let v: Vec<f32> = (0..count).map(|i| (s0 + i as f64 * delta) as f32).collect();
        HostTensor::from_f32(vec![count as i64], &v)
    };
    env.set(&node.outputs[0], Tensor::Host(out));
    Ok(())
}

/// `Pad` (mode=constant). `pads` (input[1], host int) = [begin.., end..];
/// optional constant_value (input[2], scalar). Device (f32) via kernel;
/// otherwise host. Only mode=constant (the only one used by the model).
fn pad(env: &mut Env, node: &NodeIr) -> Result<()> {
    if let Some(m) = node.attrs.get("mode").and_then(|a| match a {
        crate::AttrValue::String(s) => Some(s.as_str()),
        _ => None,
    }) {
        ensure!(
            m == "constant",
            "Pad: mode '{m}' not supported (only constant)"
        );
    }
    let data = &node.inputs[0];
    let rank = env.shape_of(data)?.len();
    let pads = env.host(&node.inputs[1])?.to_i64()?;
    ensure!(
        pads.len() == 2 * rank,
        "Pad: pads len {} != 2*rank {}",
        pads.len(),
        2 * rank
    );
    let begins = pads[..rank].to_vec();
    let ends = pads[rank..].to_vec();

    if env.dtype_of(data)? == FLOAT {
        pad_device(env, node, &begins, &ends)
    } else {
        let d = env.host(data)?;
        let cval = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
            env.host(&node.inputs[2])?.data
        } else {
            Vec::new()
        };
        let out = host_ops::pad(&d, &begins, &ends, &cval)?;
        env.set(&node.outputs[0], Tensor::Host(out));
        Ok(())
    }
}

/// GPU (constant) Pad of an f32 activation: per-dim parameters in an i32 buffer,
/// f32 constant value in the push.
fn pad_device(env: &mut Env, node: &NodeIr, begins: &[i64], ends: &[i64]) -> Result<()> {
    let ctx = env.context();
    let data = &node.inputs[0];
    let in_shape = env.shape_of(data)?;
    let rank = in_shape.len();
    let in_str = host_ops::row_major_strides(&in_shape);
    let out_shape: Vec<i64> = (0..rank)
        .map(|d| in_shape[d] + begins[d] + ends[d])
        .collect();
    let n: usize = out_shape.iter().product::<i64>().max(0) as usize;
    let cval: f32 = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        *env.host(&node.inputs[2])?.to_f32()?.first().unwrap_or(&0.0)
    } else {
        0.0
    };

    // params i32: [rank, odim(rank), begin(rank), idim(rank), istride(rank)]
    let mut params: Vec<i32> = Vec::with_capacity(1 + rank * 4);
    params.push(rank as i32);
    params.extend(out_shape.iter().map(|&d| d as i32));
    params.extend(begins.iter().map(|&b| b as i32));
    params.extend(in_shape.iter().map(|&d| d as i32));
    params.extend(in_str.iter().map(|&s| s as i32));
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    let params_buf = ctx.create_storage_buffer(params_bytes.len().max(4) as u64)?;
    ctx.stream_upload(&params_buf, &params_bytes)?;

    env.ensure_device(data)?;
    let d = env.device(data)?;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, n)?)?;
    if n > 0 {
        let mut push = Vec::with_capacity(12);
        push.extend_from_slice(&(n as u32).to_le_bytes());
        push.extend_from_slice(&(rank as u32).to_le_bytes());
        push.extend_from_slice(&cval.to_le_bytes());
        with_pipeline(
            env.cache(),
            "Pad",
            || ctx.create_pipeline(&compile_wgsl(PAD)?, PAD_BINDINGS, PAD_PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[d.buffer(), &params_buf, &out],
                    &push,
                    [(n as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    ctx.defer_destroy(params_buf);
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: out_shape,
            elem_count: n,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// Geometry of a 1D/2D convolution normalized to 2D (the 1D case uses a trivial
/// W dimension = 1). Shared by `Conv` and `ConvInteger`: the two ops have the
/// same attributes and shape arithmetic, they differ only in dtype and kernel.
struct ConvGeom {
    n: i64,
    c_in: i64,
    c_out: i64,
    group: i64,
    /// Canali di input per gruppo (`W.shape[1]` = `c_in / group`).
    gsi: i64,
    h_in: i64,
    w_in: i64,
    h_out: i64,
    w_out: i64,
    kh: i64,
    kw: i64,
    sh: i64,
    sw: i64,
    dh: i64,
    dw: i64,
    phb: i64,
    phe: i64,
    pwb: i64,
    pwe: i64,
    out_shape: Vec<i64>,
    total: usize,
}

impl ConvGeom {
    /// Push constants shared by the two conv kernels: the 17 common fields, in
    /// the declaration order of the WGSL struct.
    fn push_common(&self) -> Vec<u8> {
        let fields = [
            self.total as u32,
            self.c_in as u32,
            self.c_out as u32,
            self.group as u32,
            self.h_in as u32,
            self.w_in as u32,
            self.h_out as u32,
            self.w_out as u32,
            self.kh as u32,
            self.kw as u32,
            self.sh as u32,
            self.sw as u32,
            self.phb as u32,
            self.pwb as u32,
            self.dh as u32,
            self.dw as u32,
            self.gsi as u32,
        ];
        let mut push = Vec::with_capacity(fields.len() * 4);
        for v in fields {
            push.extend_from_slice(&v.to_le_bytes());
        }
        push
    }
}

/// Begin/end padding on a dimension for `auto_pad` = `SAME_UPPER` or
/// `SAME_LOWER`: the output has size `ceil(in / stride)` and the total padding
/// is split, with the excess at the end (`SAME_UPPER`) or at the beginning
/// (`SAME_LOWER`).
fn same_pads(upper: bool, in_dim: i64, k: i64, stride: i64, dil: i64) -> (i64, i64) {
    let out = (in_dim + stride - 1) / stride;
    let needed = ((out - 1) * stride + (k - 1) * dil + 1 - in_dim).max(0);
    let half = needed / 2;
    if upper {
        (half, needed - half)
    } else {
        (needed - half, half)
    }
}

/// Attributes and output shape of `Conv`/`ConvInteger`, including `auto_pad`.
fn conv_geometry(node: &NodeIr, x_shape: &[i64], w_shape: &[i64]) -> Result<ConvGeom> {
    let op = node.op.as_str();
    let spatial = x_shape.len().saturating_sub(2);
    ensure!(
        spatial == 1 || spatial == 2,
        "{op}: only 1D/2D (rank {})",
        x_shape.len()
    );

    let ints = |k: &str| {
        node.attrs
            .get(k)
            .and_then(|a| a.as_ints())
            .map(|s| s.to_vec())
    };
    let group = node
        .attrs
        .get("group")
        .and_then(|a| a.as_i64())
        .unwrap_or(1);
    let ks = ints("kernel_shape").unwrap_or_else(|| w_shape[2..].to_vec());
    let strides = ints("strides").unwrap_or_else(|| vec![1; spatial]);
    let dils = ints("dilations").unwrap_or_else(|| vec![1; spatial]);
    let mut pads = ints("pads").unwrap_or_else(|| vec![0; 2 * spatial]);
    ensure!(
        ks.len() == spatial && strides.len() == spatial && dils.len() == spatial,
        "{op}: kernel_shape/strides/dilations incoerenti con rank spaziale {spatial}"
    );
    ensure!(pads.len() == 2 * spatial, "{op}: pads of wrong length");

    let auto_pad = node
        .attrs
        .get("auto_pad")
        .and_then(AttrValue::as_str)
        .unwrap_or("NOTSET");
    match auto_pad {
        "NOTSET" => {}
        "VALID" => pads = vec![0; 2 * spatial],
        "SAME_UPPER" | "SAME_LOWER" => {
            let upper = auto_pad == "SAME_UPPER";
            for d in 0..spatial {
                let (begin, end) = same_pads(upper, x_shape[2 + d], ks[d], strides[d], dils[d]);
                pads[d] = begin;
                pads[spatial + d] = end;
            }
        }
        other => bail!("{op}: auto_pad '{other}' not supported"),
    }

    let (n, c_in) = (x_shape[0], x_shape[1]);
    let c_out = w_shape[0];
    let gsi = w_shape[1]; // c_in / group
    let (h_in, w_in, kh, kw, sh, sw, dh, dw) = if spatial == 1 {
        (x_shape[2], 1, ks[0], 1, strides[0], 1, dils[0], 1)
    } else {
        (
            x_shape[2], x_shape[3], ks[0], ks[1], strides[0], strides[1], dils[0], dils[1],
        )
    };
    let (phb, phe, pwb, pwe) = if spatial == 1 {
        (pads[0], pads[1], 0, 0)
    } else {
        (pads[0], pads[2], pads[1], pads[3])
    };
    let h_out = (h_in + phb + phe - (dh * (kh - 1) + 1)) / sh + 1;
    let w_out = (w_in + pwb + pwe - (dw * (kw - 1) + 1)) / sw + 1;
    ensure!(h_out > 0 && w_out > 0, "{op}: shape di output degenere");
    let out_shape = if spatial == 1 {
        vec![n, c_out, h_out]
    } else {
        vec![n, c_out, h_out, w_out]
    };
    let total = (n * c_out * h_out * w_out).max(0) as usize;

    Ok(ConvGeom {
        n,
        c_in,
        c_out,
        group,
        gsi,
        h_in,
        w_in,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        dh,
        dw,
        phb,
        phe,
        pwb,
        pwe,
        out_shape,
        total,
    })
}

/// `Conv` (opset ≥1) floating-point, 1D/2D, with group/depthwise, stride, pad,
/// dilation and optional bias. Direct convolution: one thread per output element.
fn conv_f32(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    let x_name = node.inputs[0].clone();
    let w_name = node.inputs[1].clone();
    let x_shape = env.shape_of(&x_name)?;
    let w_shape = env.shape_of(&w_name)?;
    let g = conv_geometry(node, &x_shape, &w_shape)?;

    env.ensure_device(&x_name)?;
    env.ensure_device(&w_name)?;
    // bias absent ⇒ binding on the shared zero buffer, never read (has_bias = 0)
    let bias_name = node.inputs.get(2).filter(|n| !n.is_empty()).cloned();
    let bias: *const GpuBuffer = match &bias_name {
        Some(n) => {
            env.ensure_device(n)?;
            env.device(n)?.buffer() as *const GpuBuffer
        }
        None => env.cache().zero_scalar()?,
    };

    let x = env.device(&x_name)?;
    let w = env.device(&w_name)?;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, g.total)?)?;
    if g.total > 0 {
        let mut push = g.push_common();
        push.extend_from_slice(&u32::from(bias_name.is_some()).to_le_bytes());
        // group == 1 is the only case that is really a single GEMM: grouped and
        // depthwise stay on the direct conv (see `shaders::conv`)
        let gemm = g.group == 1;
        let pixels = (g.h_out * g.w_out) as u32;
        let kdepth = (g.gsi * g.kh * g.kw) as usize;
        // Splitting K buys back the grid the 64×64 tile spends, and neither
        // transformation pays alone; see `conv::split_k`.
        let split = if gemm {
            conv_split_k(pixels as usize, g.c_out as usize, kdepth)
        } else {
            None
        };
        push.extend_from_slice(&split.unwrap_or(1).to_le_bytes());
        let bias: &GpuBuffer = unsafe { &*bias };
        // Among the shapes left, only those whose grid fills the machine gain
        // from the 64×64 tile; see `conv::prefer_blocked`.
        let blocked = gemm && conv_prefer_blocked(pixels as usize, g.c_out as usize);
        let (key, source, tile) = match (gemm, split.is_some(), blocked) {
            (true, true, _) => ("Conv_split", conv_blocked_splitk_source(), CONV_BLOCKED_TILE_SIZE),
            (true, false, true) => ("Conv", conv_blocked_source(), CONV_BLOCKED_TILE_SIZE),
            (true, false, false) => ("Conv16", conv_gemm_source(), CONV_TILE_SIZE),
            _ => ("Conv_grouped", conv_direct_source(), 0),
        };
        let groups = if gemm {
            [
                pixels.div_ceil(tile),
                (g.c_out as u32).div_ceil(tile),
                (g.n as u32).max(1) * split.unwrap_or(1),
            ]
        } else {
            [(g.total as u32).div_ceil(256), 1, 1]
        };
        // With a split the kernel writes one partial image per slice and the
        // reduction pass below folds them onto `out`, bias included.
        let partials = split
            .map(|s| ctx.create_storage_buffer(device_storage_bytes(FLOAT, s as usize * g.total)?))
            .transpose()?;
        with_pipeline(
            env.cache(),
            key,
            || {
                ctx.create_pipeline(
                    &compile_wgsl(&source)?,
                    CONV_F32_BINDINGS,
                    CONV_F32_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[
                        x.buffer(),
                        w.buffer(),
                        bias,
                        partials.as_ref().unwrap_or(&out),
                    ],
                    &push,
                    groups,
                )
            },
        )?;
        if let Some(partials) = partials {
            with_pipeline(
                env.cache(),
                "Conv_split_reduce",
                || {
                    ctx.create_pipeline(
                        &compile_wgsl(CONV_SPLIT_REDUCE)?,
                        CONV_SPLIT_REDUCE_BINDINGS,
                        CONV_F32_PUSH_BYTES,
                    )
                },
                |pipe| {
                    ctx.stream_dispatch(
                        pipe,
                        &[&partials, bias, &out],
                        &push,
                        [(g.total as u32).div_ceil(256), 1, 1],
                    )
                },
            )?;
            ctx.defer_destroy(partials);
        }
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: g.out_shape,
            elem_count: g.total,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// Sizes the phase-GEMM route works from, all already normalized to 2D.
struct PhaseDims {
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
}

/// Buffers of one phase-GEMM dispatch. `x` and `bias` are raw because they
/// point into `Env`, which the packing step needs mutably.
struct PhaseRun<'a> {
    x: *const GpuBuffer,
    bias: *const GpuBuffer,
    has_bias: bool,
    out: &'a GpuBuffer,
    d: PhaseDims,
}

/// Weight slice `W[:, :, r, s]` as `[K = C_in][M = C_out]`, cached per phase.
///
/// `W` is a graph constant here (the caller checks), so this runs once per
/// session; the key carries the phase index because the four slices of one
/// weight are four different buffers.
fn packed_phase(
    env: &mut Env,
    w_name: &str,
    d: &PhaseDims,
    r: usize,
    s: usize,
) -> Result<*const GpuBuffer> {
    let cache = env.cache();
    let key = (format!("{w_name}#ct{}", r * d.kw + s), d.c_in, d.c_out);
    if let Some(cached) = cache.packed_weight_cached(&key) {
        return Ok(cached);
    }
    let src = env.device(w_name)?.buffer() as *const GpuBuffer;
    let ctx = env.context();
    let total = d.c_in * d.c_out;
    cache.packed_weight(key, || {
        let dst = ctx.create_storage_buffer(device_storage_bytes(FLOAT, total)?)?;
        let mut push = Vec::with_capacity(CONV_T_PACK_PUSH_BYTES as usize);
        for v in [total as u32, (d.kh * d.kw) as u32, (r * d.kw + s) as u32] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        with_pipeline(
            cache,
            "ConvTranspose_pack",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(CONV_T_PACK_PHASE)?,
                    CONV_T_PACK_BINDINGS,
                    CONV_T_PACK_PUSH_BYTES,
                )
            },
            // SAFETY: `src` is the weight's device buffer, owned by the
            // execution environment for the whole run.
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[unsafe { &*src }, &dst],
                    &push,
                    [(total as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
        Ok(dst)
    })
}

/// `ConvTranspose` as `kH·kW` GEMMs plus a scatter each; see
/// [`crate::shaders::conv_transpose`] for why this is the same operator.
///
/// The bias rides along as the GEMM's `C`, broadcast over the columns, so the
/// phases need no epilogue of their own.
fn conv_transpose_phase_gemm(env: &mut Env, w_name: &str, run: &PhaseRun) -> Result<()> {
    let ctx = env.context();
    let cache = env.cache();
    let d = &run.d;
    let n = d.h_in * d.w_in;
    // SAFETY: both point into the environment's buffers, alive for the run.
    let (x, bias) = unsafe { (&*run.x, &*run.bias) };
    let phase = ctx.create_storage_buffer(device_storage_bytes(FLOAT, d.c_out * n)?)?;

    // A stride wider than the kernel leaves output pixels no phase reaches:
    // they hold the bias alone, so they must be written before the phases.
    if d.sh > d.kh || d.sw > d.kw {
        let plane = d.h_out * d.w_out;
        let mut push = Vec::with_capacity(CONV_T_FILL_PUSH_BYTES as usize);
        for v in [
            (d.c_out * plane) as u32,
            plane as u32,
            u32::from(run.has_bias),
        ] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        with_pipeline(
            cache,
            "ConvTranspose_fill",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(CONV_T_FILL)?,
                    CONV_T_FILL_BINDINGS,
                    CONV_T_FILL_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[bias, run.out],
                    &push,
                    [((d.c_out * plane) as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }

    for r in 0..d.kh {
        for s in 0..d.kw {
            let packed = packed_phase(env, w_name, d, r, s)?;
            // A' is the packed slice read transposed ([K][M]); C is the bias,
            // one value per row, broadcast over the N columns.
            let flags = 1u32 | if run.has_bias { 4 } else { 0 };
            let mut push = Vec::with_capacity(GEMM_PUSH_BYTES as usize);
            for v in [d.c_out as u32, d.c_in as u32, n as u32, flags] {
                push.extend_from_slice(&v.to_le_bytes());
            }
            push.extend_from_slice(&1.0f32.to_le_bytes()); // alpha
            push.extend_from_slice(&1.0f32.to_le_bytes()); // beta
            push.extend_from_slice(&(d.c_out as u32).to_le_bytes()); // c_rows
            push.extend_from_slice(&1u32.to_le_bytes()); // c_cols
            with_pipeline(
                cache,
                "ConvTranspose_gemm",
                || ctx.create_pipeline(&compile_wgsl(GEMM)?, GEMM_BINDINGS, GEMM_PUSH_BYTES),
                // SAFETY: cached in `packed_weight`, which never moves entries.
                |pipe| {
                    ctx.stream_dispatch(
                        pipe,
                        &[unsafe { &*packed }, x, bias, &phase],
                        &push,
                        [
                            (n as u32).div_ceil(GEMM_TILE_SIZE),
                            (d.c_out as u32).div_ceil(GEMM_TILE_SIZE),
                            1,
                        ],
                    )
                },
            )?;

            let total = (d.c_out * n) as u32;
            let mut push = Vec::with_capacity(CONV_T_INTERLEAVE_PUSH_BYTES as usize);
            for v in [
                total,
                d.h_in as u32,
                d.w_in as u32,
                d.h_out as u32,
                d.w_out as u32,
                d.sh as u32,
                d.sw as u32,
                r as u32,
                s as u32,
            ] {
                push.extend_from_slice(&v.to_le_bytes());
            }
            with_pipeline(
                cache,
                "ConvTranspose_interleave",
                || {
                    ctx.create_pipeline(
                        &compile_wgsl(CONV_T_INTERLEAVE)?,
                        CONV_T_INTERLEAVE_BINDINGS,
                        CONV_T_INTERLEAVE_PUSH_BYTES,
                    )
                },
                |pipe| {
                    ctx.stream_dispatch(
                        pipe,
                        &[&phase, run.out],
                        &push,
                        [total.div_ceil(256), 1, 1],
                    )
                },
            )?;
        }
    }
    ctx.defer_destroy(phase);
    Ok(())
}

/// `ConvTranspose` f32 1D/2D.
///
/// Geometry is not `conv_geometry` with different numbers: `W` is
/// `[C_in, C_out/group, kH, kW]` (input channel first, the opposite of `Conv`)
/// and the spatial size **grows**:
///
/// ```text
///   out = (in - 1)*stride - pad_begin - pad_end + dilation*(k - 1) + 1 + output_padding
/// ```
///
/// `output_shape`, when present, is the requested output and it is the pads
/// that get derived from it; that inversion is not implemented, so
/// `is_implemented_node` refuses those nodes together with `auto_pad = SAME_*`,
/// which is only meaningful in terms of `output_shape`.
fn conv_transpose(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    let x_name = node.inputs[0].clone();
    let w_name = node.inputs[1].clone();
    let x_shape = env.shape_of(&x_name)?;
    let w_shape = env.shape_of(&w_name)?;

    let spatial = x_shape.len().saturating_sub(2);
    ensure!(
        spatial == 1 || spatial == 2,
        "ConvTranspose: only 1D/2D (rank {})",
        x_shape.len()
    );
    let ints = |k: &str| {
        node.attrs
            .get(k)
            .and_then(|a| a.as_ints())
            .map(<[i64]>::to_vec)
    };
    let group = node
        .attrs
        .get("group")
        .and_then(AttrValue::as_i64)
        .unwrap_or(1);
    ensure!(group >= 1, "ConvTranspose: group {group} is invalid");
    let ks = ints("kernel_shape").unwrap_or_else(|| w_shape[2..].to_vec());
    let strides = ints("strides").unwrap_or_else(|| vec![1; spatial]);
    let dils = ints("dilations").unwrap_or_else(|| vec![1; spatial]);
    let pads = ints("pads").unwrap_or_else(|| vec![0; 2 * spatial]);
    let outpad = ints("output_padding").unwrap_or_else(|| vec![0; spatial]);
    ensure!(
        ks.len() == spatial
            && strides.len() == spatial
            && dils.len() == spatial
            && outpad.len() == spatial
            && pads.len() == 2 * spatial,
        "ConvTranspose: attributi incoerenti con rank spaziale {spatial}"
    );

    let (n, c_in) = (x_shape[0], x_shape[1]);
    let gso = w_shape[1]; // C_out / group
    let c_out = gso * group;
    let (h_in, w_in, kh, kw, sh, sw, dh, dw) = if spatial == 1 {
        (x_shape[2], 1, ks[0], 1, strides[0], 1, dils[0], 1)
    } else {
        (
            x_shape[2], x_shape[3], ks[0], ks[1], strides[0], strides[1], dils[0], dils[1],
        )
    };
    let (phb, phe, pwb, pwe, oph, opw) = if spatial == 1 {
        (pads[0], pads[1], 0, 0, outpad[0], 0)
    } else {
        (pads[0], pads[2], pads[1], pads[3], outpad[0], outpad[1])
    };
    let h_out = (h_in - 1) * sh - phb - phe + dh * (kh - 1) + 1 + oph;
    let w_out = (w_in - 1) * sw - pwb - pwe + dw * (kw - 1) + 1 + opw;
    ensure!(
        h_out > 0 && w_out > 0,
        "ConvTranspose: shape di output degenere"
    );
    let out_shape = if spatial == 1 {
        vec![n, c_out, h_out]
    } else {
        vec![n, c_out, h_out, w_out]
    };
    let total = (n * c_out * h_out * w_out).max(0) as usize;

    env.ensure_device(&x_name)?;
    env.ensure_device(&w_name)?;
    // bias absent ⇒ binding on the shared zero buffer, never read (has_bias = 0)
    let bias_name = node.inputs.get(2).filter(|n| !n.is_empty()).cloned();
    let bias: *const GpuBuffer = match &bias_name {
        Some(n) => {
            env.ensure_device(n)?;
            env.device(n)?.buffer() as *const GpuBuffer
        }
        None => env.cache().zero_scalar()?,
    };
    let x = env.device(&x_name)?;
    let w = env.device(&w_name)?;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, total)?)?;

    // The phase decomposition needs a constant `W` (its packed slices are cached
    // by name across runs) on top of the geometric conditions.
    let geom = PhaseGeom {
        batch: n,
        group,
        kernel: (kh, kw),
        stride: (sh, sw),
        dilation: (dh, dw),
        zero_pads: pads.iter().all(|&p| p == 0),
        zero_output_padding: outpad.iter().all(|&p| p == 0),
        macs: c_in
            .saturating_mul(c_out)
            .saturating_mul(h_out)
            .saturating_mul(w_out),
    };
    if total > 0 && phase_gemm_applies(&geom) && env.is_initializer(&w_name) {
        let run = PhaseRun {
            x: x.buffer() as *const GpuBuffer,
            bias,
            has_bias: bias_name.is_some(),
            out: &out,
            d: PhaseDims {
                c_in: c_in as usize,
                c_out: c_out as usize,
                h_in: h_in as usize,
                w_in: w_in as usize,
                h_out: h_out as usize,
                w_out: w_out as usize,
                kh: kh as usize,
                kw: kw as usize,
                sh: sh as usize,
                sw: sw as usize,
            },
        };
        conv_transpose_phase_gemm(env, &w_name, &run)?;
    } else if total > 0 {
        let fields = [
            total as u32,
            c_in as u32,
            c_out as u32,
            group as u32,
            h_in as u32,
            w_in as u32,
            h_out as u32,
            w_out as u32,
            kh as u32,
            kw as u32,
            sh as u32,
            sw as u32,
            phb as u32,
            pwb as u32,
            dh as u32,
            dw as u32,
            gso as u32,
            u32::from(bias_name.is_some()),
        ];
        let mut push = Vec::with_capacity(fields.len() * 4);
        for v in fields {
            push.extend_from_slice(&v.to_le_bytes());
        }
        let bias: &GpuBuffer = unsafe { &*bias };
        with_pipeline(
            env.cache(),
            "ConvTranspose",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(&conv_t_source())?,
                    CONV_T_BINDINGS,
                    CONV_T_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[x.buffer(), w.buffer(), bias, &out],
                    &push,
                    [(total as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: out_shape,
            elem_count: total,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `ConvInteger` (opset ≥10): quantized u8×u8→i32 conv, per-tensor zero-point.
/// Direct 1D/2D conv (1D normalized to 2D with W=1): handles group (depthwise),
/// stride, pad, dilation. Zero-points read from buffers (no readback).
fn conv_integer(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    let x_name = node.inputs[0].clone();
    let w_name = node.inputs[1].clone();
    let x_shape = env.shape_of(&x_name)?;
    let w_shape = env.shape_of(&w_name)?;
    let g = conv_geometry(node, &x_shape, &w_shape)?;
    let ConvGeom {
        n,
        c_in,
        c_out,
        group,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        dh,
        dw,
        phb,
        phe,
        pwb,
        pwe,
        total,
        ..
    } = g;
    let out_shape = g.out_shape.clone();

    env.ensure_device_dtype(&x_name)?;
    env.ensure_device_dtype(&w_name)?;
    // ONNX allows both uint8 and int8 for X and W, independently
    let x_signed = u32::from(env.dtype_of(&x_name)? == INT8);
    let w_signed = u32::from(env.dtype_of(&w_name)? == INT8);
    ensure_zp(env, node.inputs.get(2))?; // zero-point of the X activation
    ensure_zp(env, node.inputs.get(3))?; // zero-point of the W weights

    // Fast path: a 1×1 (pointwise) conv, group=1, stride/dil=1, no pad is exactly
    // a matmul W[C_out,C_in] × X[C_in, T] → [C_out, T]. Routed to the shared
    // tiled MatMulInteger kernel, much faster than naive direct conv. Dominates
    // the encoder cost (48 pointwise convs).
    let pointwise = group == 1
        && n == 1
        && kh == 1
        && kw == 1
        && sh == 1
        && sw == 1
        && dh == 1
        && dw == 1
        && phb == 0
        && phe == 0
        && pwb == 0
        && pwe == 0
        && (c_in as usize).is_multiple_of(4);
    if pointwise && total > 0 {
        return conv_pointwise_matmul(
            env,
            node,
            &x_name,
            &w_name,
            out_shape,
            c_out as usize,
            c_in as usize,
            (h_out * w_out) as usize,
            node.inputs.get(2).cloned(),
            node.inputs.get(3).cloned(),
            x_signed,
            w_signed,
        );
    }

    let x = env.device(&x_name)?;
    let w = env.device(&w_name)?;
    let out = ctx.create_storage_buffer(device_storage_bytes(INT32, total)?)?;
    if total > 0 {
        let mut push = g.push_common();
        push.extend_from_slice(&x_signed.to_le_bytes());
        push.extend_from_slice(&w_signed.to_le_bytes());
        let azp = zp_ref(env, node.inputs.get(2))?;
        let wzp = zp_ref(env, node.inputs.get(3))?;
        with_pipeline(
            env.cache(),
            "ConvInteger",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(CONV_INTEGER)?,
                    CONV_INTEGER_BINDINGS,
                    CONV_INTEGER_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[x.buffer(), w.buffer(), azp, wzp, &out],
                    &push,
                    [(total as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: INT32,
            shape: out_shape,
            elem_count: total,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// Layout of the quantization parameters: `inner` is the stride below the
/// quantized axis and `axis_len` the number of scales. Per-tensor is the
/// degenerate case `axis_len = 1`, so a single shader covers both forms.
fn qdq_layout(node: &NodeIr, x_shape: &[i64], scale_shape: &[i64]) -> Result<(u32, u32)> {
    let axis_len: i64 = scale_shape.iter().product::<i64>().max(1);
    if scale_shape.is_empty() || axis_len == 1 {
        return Ok((1, 1)); // per-tensor
    }
    ensure!(
        scale_shape.len() == 1,
        "{}: per-axis scale must be 1-D (shape {scale_shape:?})",
        node.op
    );
    let rank = x_shape.len() as i64;
    let mut axis = node.attrs.get("axis").and_then(|a| a.as_i64()).unwrap_or(1);
    if axis < 0 {
        axis += rank;
    }
    ensure!(
        (0..rank).contains(&axis),
        "{}: axis {axis} out of rank {rank}",
        node.op
    );
    ensure!(
        x_shape[axis as usize] == axis_len,
        "{}: scale of length {axis_len}, axis {axis} of dimension {}",
        node.op,
        x_shape[axis as usize]
    );
    let inner: i64 = x_shape[axis as usize + 1..].iter().product::<i64>().max(1);
    Ok((inner as u32, axis_len as u32))
}

/// Push constants shared by `QuantizeLinear` and `DequantizeLinear`.
fn qdq_push(n: usize, inner: u32, axis_len: u32, signed: bool, has_zp: bool) -> Vec<u8> {
    let fields = [
        n as u32,
        inner,
        axis_len,
        u32::from(signed),
        u32::from(has_zp),
    ];
    let mut push = Vec::with_capacity(fields.len() * 4);
    for v in fields {
        push.extend_from_slice(&v.to_le_bytes());
    }
    push
}

/// `DequantizeLinear` (opset ≥10): `y = (x - zero_point) * scale`, per-tensor or
/// per-axis. Quantized u8/i8 packed input, f32 output.
fn dequantize_linear(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    let x_name = node.inputs[0].clone();
    let scale_name = node.inputs[1].clone();
    let x_shape = env.shape_of(&x_name)?;
    let scale_shape = env.shape_of(&scale_name)?;
    let (inner, axis_len) = qdq_layout(node, &x_shape, &scale_shape)?;
    let dtype = env.dtype_of(&x_name)?;
    ensure!(
        dtype == UINT8 || dtype == INT8 || dtype == INT32,
        "DequantizeLinear: dtype {dtype} not supported (only uint8/int8/int32)"
    );
    // the bias of QDQ convolutions is int32 and not packed: a different
    // shader, because byte extraction here would give the wrong value
    let packed = dtype != INT32;
    let n: usize = x_shape.iter().product::<i64>().max(0) as usize;

    env.ensure_device_dtype(&x_name)?;
    env.ensure_device(&scale_name)?;
    let zp_name = node.inputs.get(2).filter(|n| !n.is_empty());
    let has_zp = zp_name.is_some();
    ensure_zp(env, zp_name)?;

    let zp = zp_ref(env, zp_name)?;
    let x = env.device(&x_name)?;
    let scale = env.device(&scale_name)?;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, n)?)?;
    if n > 0 {
        let push = qdq_push(n, inner, axis_len, dtype == INT8, has_zp);
        with_pipeline(
            env.cache(),
            if packed {
                "DequantizeLinear"
            } else {
                "DequantizeLinear_i32"
            },
            || {
                let src = if packed { DEQUANTIZE } else { DEQUANTIZE_I32 };
                ctx.create_pipeline(&compile_wgsl(src)?, QDQ_BINDINGS, QDQ_PUSH_BYTES)
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[x.buffer(), scale.buffer(), zp, &out],
                    &push,
                    [(n as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: x_shape,
            elem_count: n,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `QuantizeLinear` (opset ≥10): `y = saturate(round(x / scale) + zero_point)`,
/// per-tensor or per-axis. The output dtype follows the zero-point if present,
/// otherwise the `output_dtype` attribute (opset 21), otherwise uint8.
fn quantize_linear(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    let x_name = node.inputs[0].clone();
    let scale_name = node.inputs[1].clone();
    let x_shape = env.shape_of(&x_name)?;
    let scale_shape = env.shape_of(&scale_name)?;
    let (inner, axis_len) = qdq_layout(node, &x_shape, &scale_shape)?;
    let n: usize = x_shape.iter().product::<i64>().max(0) as usize;

    let zp_name = node.inputs.get(2).filter(|n| !n.is_empty()).cloned();
    let out_dtype = match &zp_name {
        Some(name) => env.dtype_of(name)?,
        None => node
            .attrs
            .get("output_dtype")
            .and_then(|a| a.as_i64())
            .map(|v| v as i32)
            .unwrap_or(UINT8),
    };
    ensure!(
        out_dtype == UINT8 || out_dtype == INT8,
        "QuantizeLinear: output dtype {out_dtype} not supported (only uint8/int8)"
    );

    env.ensure_device(&x_name)?;
    env.ensure_device(&scale_name)?;
    let has_zp = zp_name.is_some();
    ensure_zp(env, zp_name.as_ref())?;

    let zp = zp_ref(env, zp_name.as_ref())?;
    let x = env.device(&x_name)?;
    let scale = env.device(&scale_name)?;
    let out = ctx.create_storage_buffer(device_storage_bytes(out_dtype, n)?)?;
    if n > 0 {
        let push = qdq_push(n, inner, axis_len, out_dtype == INT8, has_zp);
        with_pipeline(
            env.cache(),
            "QuantizeLinear",
            || ctx.create_pipeline(&compile_wgsl(QUANTIZE)?, QDQ_BINDINGS, QDQ_PUSH_BYTES),
            |pipe| {
                // one thread per u32 word, i.e. every 4 elements
                ctx.stream_dispatch(
                    pipe,
                    &[x.buffer(), scale.buffer(), zp, &out],
                    &push,
                    [(n as u32).div_ceil(4).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: out_dtype,
            shape: x_shape,
            elem_count: n,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// 1×1 pointwise conv (group=1) as a tiled int8 matmul: `out[C_out, T]` =
/// `W[C_out, C_in] × X[C_in, T]`. Reuses the shared MatMulInteger shaders:
/// packing of X (dynamic operand), then A=W raw and B=X packed. Zero-points:
/// A=W→`w_zp`, B=X→`a_zp` (activation). `nmat` = spatial output elements
/// (h_out·w_out).
#[allow(clippy::too_many_arguments)]
fn conv_pointwise_matmul(
    env: &mut Env,
    node: &NodeIr,
    x_name: &str,
    w_name: &str,
    out_shape: Vec<i64>,
    c_out: usize,
    c_in: usize,
    nmat: usize,
    a_zp: Option<String>, // zero-point of X (activation) → B binding
    w_zp: Option<String>, // zero-point of W (weights)    → A binding
    x_signed: u32,
    w_signed: u32,
) -> Result<()> {
    let ctx = env.context();
    let k4 = c_in / 4;
    let total = c_out * nmat;

    // pack of X [C_in, nmat] (raw u8) → [nmat, C_in/4] (u32), like MMI's B
    let x = env.device(x_name)?;
    let packed = ctx.create_storage_buffer((nmat * k4 * 4).max(4) as u64)?;
    let mut pack_push = Vec::with_capacity(MMI_PACK_PUSH_BYTES as usize);
    pack_push.extend_from_slice(&(c_in as u32).to_le_bytes());
    pack_push.extend_from_slice(&(nmat as u32).to_le_bytes());
    pack_push.extend_from_slice(&if x_signed != 0 { MMI_SIGN_FLIP_WORD } else { 0 }.to_le_bytes());
    with_pipeline(
        env.cache(),
        "MMI_pack",
        || {
            ctx.create_pipeline(
                &compile_wgsl(MMI_PACK_B)?,
                MMI_PACK_BINDINGS,
                MMI_PACK_PUSH_BYTES,
            )
        },
        |pipe| {
            ctx.stream_dispatch(
                pipe,
                &[x.buffer(), &packed],
                &pack_push,
                [
                    (nmat as u32).div_ceil(MMI_TILE_SIZE),
                    (k4 as u32).div_ceil(MMI_TILE_SIZE),
                    1,
                ],
            )
        },
    )?;

    // tiled matmul: A=W [C_out, C_in], packed=X [nmat, C_in/4] → out [C_out, nmat]
    // W is constant: if signed it is made unsigned once, so the cooperative path —
    // which cannot flip — can also use it.
    let w_ptr = if w_signed != 0 {
        flipped_const(env, w_name, c_out * c_in)?
    } else {
        env.device(w_name)?.buffer() as *const GpuBuffer
    };
    let ctx = env.context();
    let w: &GpuBuffer = unsafe { &*w_ptr };
    let out = ctx.create_storage_buffer(device_storage_bytes(INT32, total)?)?;
    let azp = zp_ref(env, w_zp.as_ref())?;
    let bzp = zp_ref(env, a_zp.as_ref())?;
    // in the matmul A=W and B=X: the sign flags follow the same swap
    mmi_dispatch(
        env.cache(),
        ctx,
        &[w, &packed, azp, bzp, &out],
        MmiProblem {
            m: c_out,
            k: c_in,
            n: nmat,
            a_byte_flip: 0, // W already unsigned in memory
            a_zp_xor: if w_signed != 0 { MMI_SIGN_FLIP_BYTE } else { 0 },
            b_zp_xor: if x_signed != 0 { MMI_SIGN_FLIP_BYTE } else { 0 },
        },
    )?;
    ctx.defer_destroy(packed);
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: INT32,
            shape: out_shape,
            elem_count: total,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// Makes the scalar zero-point resident in VRAM, if the input is present.
fn ensure_zp(env: &mut Env, name: Option<&String>) -> Result<()> {
    if let Some(n) = name.filter(|n| !n.is_empty()) {
        env.ensure_device_dtype(n)?;
    }
    Ok(())
}

/// VRAM buffer of the scalar zero-point: the input's if present, otherwise the
/// shared zero buffer (missing zero-point ⇒ 0).
///
/// Must be called **after** every mutation of the environment: tensors live
/// inside a `HashMap`, so a later `insert` can move them and invalidate a
/// reference taken earlier. Returning a borrow instead of a pointer lets the
/// borrow checker enforce this order.
fn zp_ref<'env>(env: &'env Env, name: Option<&String>) -> Result<&'env GpuBuffer> {
    match name.filter(|n| !n.is_empty()) {
        Some(n) => Ok(env.device(n)?.buffer()),
        // SAFETY: the zero buffer lives in the cache, which never removes entries
        // and keeps them in `Box`: the address stays valid as long as the cache,
        // which outlives the environment.
        None => Ok(unsafe { &*env.cache().zero_scalar()? }),
    }
}

/// `Where` (cond ? X : Y) with 3-way broadcasting. If X is an f32 activation →
/// GPU kernel; otherwise host-side. cond is bool.
fn where_op(env: &mut Env, node: &NodeIr) -> Result<()> {
    let (cond, x, y) = (&node.inputs[0], &node.inputs[1], &node.inputs[2]);
    let (cs, xs, ys) = (env.shape_of(cond)?, env.shape_of(x)?, env.shape_of(y)?);
    let ab = broadcast(&cs, &xs)?;
    let abc = broadcast(&ab.out_shape, &ys)?;
    let out_shape = abc.out_shape;

    if env.dtype_of(x)? == FLOAT {
        where_device(env, node, &out_shape)
    } else {
        let (hc, hx, hy) = (env.host(cond)?, env.host(x)?, env.host(y)?);
        let out = host_ops::where_op(&hc, &hx, &hy, &out_shape)?;
        env.set(&node.outputs[0], Tensor::Host(out));
        Ok(())
    }
}

/// GPU Where: X,Y f32 activations, cond bool; per-input broadcast strides in
/// an i32 buffer (odim/cond/x/y), one output element per thread.
fn where_device(env: &mut Env, node: &NodeIr, out_shape: &[i64]) -> Result<()> {
    let ctx = env.context();
    let (cond, x, y) = (&node.inputs[0], &node.inputs[1], &node.inputs[2]);
    let rank = out_shape.len();
    let cstr = host_ops::bcast_strides(&env.shape_of(cond)?, out_shape);
    let xstr = host_ops::bcast_strides(&env.shape_of(x)?, out_shape);
    let ystr = host_ops::bcast_strides(&env.shape_of(y)?, out_shape);
    let n: usize = out_shape.iter().product::<i64>().max(0) as usize;

    // params i32: [rank, odim(rank), cstr(rank), xstr(rank), ystr(rank)]
    let mut params: Vec<i32> = Vec::with_capacity(1 + rank * 4);
    params.push(rank as i32);
    params.extend(out_shape.iter().map(|&d| d as i32));
    params.extend(cstr.iter().map(|&s| s as i32));
    params.extend(xstr.iter().map(|&s| s as i32));
    params.extend(ystr.iter().map(|&s| s as i32));
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    let params_buf = ctx.create_storage_buffer(params_bytes.len().max(4) as u64)?;
    ctx.stream_upload(&params_buf, &params_bytes)?;

    env.ensure_device_dtype(cond)?; // bool → byte-esatto in VRAM
    env.ensure_device(x)?;
    env.ensure_device(y)?;
    let c = env.device(cond)?;
    let xd = env.device(x)?;
    let yd = env.device(y)?;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, n)?)?;
    if n > 0 {
        with_pipeline(
            env.cache(),
            "Where",
            || ctx.create_pipeline(&compile_wgsl(WHERE)?, WHERE_BINDINGS, WHERE_PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[c.buffer(), xd.buffer(), yd.buffer(), &out, &params_buf],
                    &(n as u32).to_le_bytes(),
                    [(n as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    ctx.defer_destroy(params_buf);
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: out_shape.to_vec(),
            elem_count: n,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `Slice` (opset ≥10). If `data` is an f32 activation → GPU kernel (per-dim
/// parameters in an i32 buffer); otherwise host-side. Parameters (starts/ends/axes/
/// steps) are host-read int inputs.
fn slice(env: &mut Env, node: &NodeIr) -> Result<()> {
    let data = &node.inputs[0];
    let data_shape = env.shape_of(data)?;
    let starts = env.host(&node.inputs[1])?.to_i64()?;
    let ends = env.host(&node.inputs[2])?.to_i64()?;
    let axes = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
        Some(env.host(&node.inputs[3])?.to_i64()?)
    } else {
        None
    };
    let steps = if node.inputs.len() > 4 && !node.inputs[4].is_empty() {
        Some(env.host(&node.inputs[4])?.to_i64()?)
    } else {
        None
    };
    let (out_shape, st, sp) = host_ops::slice_params(
        &data_shape,
        &starts,
        &ends,
        axes.as_deref(),
        steps.as_deref(),
    )?;

    if env.dtype_of(data)? == FLOAT {
        let src = data.clone();
        let t = slice_device(env, &src, &data_shape, &out_shape, &st, &sp)?;
        env.set(&node.outputs[0], Tensor::Device(t));
    } else {
        let d = env.host(data)?;
        let out = host_ops::slice(&d, &out_shape, &st, &sp)?;
        env.set(&node.outputs[0], Tensor::Host(out));
    }
    Ok(())
}

/// GPU slice of an f32 activation (`src` device): per-dim parameters (odim,
/// istride, start, step) in an i32 buffer; each thread maps one output element
/// to the input. Returns the resulting `DevTensor` (also reused by Split).
fn slice_device(
    env: &mut Env,
    src: &str,
    data_shape: &[i64],
    out_shape: &[i64],
    st: &[i64],
    sp: &[i64],
) -> Result<DevTensor<'static>> {
    let ctx = env.context();
    let rank = data_shape.len();
    let in_strides = host_ops::row_major_strides(data_shape);
    let n: usize = out_shape.iter().product::<i64>().max(0) as usize;

    // params i32: [odim.., istride.., start.., step..] (rank ciascuno)
    let mut params: Vec<i32> = Vec::with_capacity(rank * 4);
    params.extend(out_shape.iter().map(|&d| d as i32));
    params.extend(in_strides.iter().map(|&s| s as i32));
    params.extend(st.iter().map(|&s| s as i32));
    params.extend(sp.iter().map(|&s| s as i32));
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    let params_buf = ctx.create_storage_buffer(params_bytes.len().max(4) as u64)?;
    ctx.stream_upload(&params_buf, &params_bytes)?;

    env.ensure_device(src)?;
    let d = env.device(src)?;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, n)?)?;
    if n > 0 {
        let mut push = Vec::with_capacity(8);
        push.extend_from_slice(&(n as u32).to_le_bytes());
        push.extend_from_slice(&(rank as u32).to_le_bytes());
        with_pipeline(
            env.cache(),
            "Slice",
            || ctx.create_pipeline(&compile_wgsl(SLICE)?, SLICE_BINDINGS, SLICE_PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[d.buffer(), &params_buf, &out],
                    &push,
                    [(n as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    ctx.defer_destroy(params_buf);
    Ok(DevTensor {
        dtype: FLOAT,
        shape: out_shape.to_vec(),
        elem_count: n,
        buf: BufRef::Owned(out),
    })
}

/// `Split` along `axis`: each output is a contiguous slice. Reuses Slice
/// (device f32) or the host equivalent. `split` (input[1]) gives the sizes;
/// absent ⇒ equal split across outputs.
fn split(env: &mut Env, node: &NodeIr) -> Result<()> {
    let data = node.inputs[0].clone();
    let data_shape = env.shape_of(&data)?;
    let rank = data_shape.len();
    let axis = node.attrs.get("axis").and_then(|a| a.as_i64()).unwrap_or(0);
    let ax = if axis < 0 { axis + rank as i64 } else { axis };
    ensure!(
        (0..rank as i64).contains(&ax),
        "Split: axis {axis} out of range"
    );
    let ax = ax as usize;
    let dim = data_shape[ax];
    let n_out = node.outputs.len();

    // `split` migrated from attribute to input in opset 13: older models
    // (yolov4 is opset 11) carry unequal sizes in the attribute, and reading
    // only the input would push them onto the equal-split branch
    let sizes: Vec<i64> = if let Some(attr) = node.attrs.get("split").and_then(AttrValue::as_ints) {
        attr.to_vec()
    } else if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        env.host(&node.inputs[1])?.to_i64()?
    } else {
        ensure!(n_out > 0, "Split: no output");
        let base = dim / n_out as i64;
        ensure!(
            base * n_out as i64 == dim,
            "Split: {dim} not divisible into {n_out}"
        );
        vec![base; n_out]
    };
    ensure!(
        sizes.len() == n_out,
        "Split: {} taglie != {n_out} output",
        sizes.len()
    );

    let is_float = env.dtype_of(&data)? == FLOAT;
    let mut start = 0i64;
    for (j, &size) in sizes.iter().enumerate() {
        let mut out_shape = data_shape.clone();
        out_shape[ax] = size;
        let mut st = vec![0i64; rank];
        st[ax] = start;
        let sp = vec![1i64; rank];
        if is_float {
            let t = slice_device(env, &data, &data_shape, &out_shape, &st, &sp)?;
            env.set(&node.outputs[j], Tensor::Device(t));
        } else {
            let d = env.host(&data)?;
            let out = host_ops::slice(&d, &out_shape, &st, &sp)?;
            env.set(&node.outputs[j], Tensor::Host(out));
        }
        start += size;
    }
    Ok(())
}

/// `MatMul` f32 (opset ≥13) with batch broadcasting. Reuses the tiled 16×16 kernel.
fn matmul_fp32(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    env.ensure_device(&node.inputs[0])?;
    env.ensure_device(&node.inputs[1])?;
    let a = env.device(&node.inputs[0])?;
    let b = env.device(&node.inputs[1])?;
    let (a_shape, b_shape) = (a.shape.clone(), b.shape.clone());
    ensure!(
        a_shape.len() >= 2 && b_shape.len() >= 2,
        "MatMul: rank < 2 (A {a_shape:?}, B {b_shape:?})"
    );
    let (m, ka) = (
        a_shape[a_shape.len() - 2] as usize,
        a_shape[a_shape.len() - 1] as usize,
    );
    let (kb, n) = (
        b_shape[b_shape.len() - 2] as usize,
        b_shape[b_shape.len() - 1] as usize,
    );
    ensure!(ka == kb, "MatMul: K incompatibile ({ka} vs {kb})");

    let bc = broadcast(&a_shape[..a_shape.len() - 2], &b_shape[..b_shape.len() - 2])?;
    ensure!(
        bc.out_shape.len() <= MAX_RANK,
        "MatMul: batch rank {} > {}",
        bc.out_shape.len(),
        MAX_RANK
    );
    let batch: usize = bc.out_shape.iter().product::<i64>().max(0) as usize;
    ensure!(batch <= 65535, "MatMul: batch {batch} too large");
    let mut out_shape = bc.out_shape.clone();
    out_shape.push(m as i64);
    out_shape.push(n as i64);
    let elem_count = batch.max(1) * m * n;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, elem_count)?)?;

    if m > 0
        && n > 0
        && let Some(split) = mm_gemv_split(m, ka, n, batch.max(1))
    {
        // Matrix-vector: neither tiled kernel fills the machine, so parallelism
        // is fabricated along K. See `matmul_fp32::gemv_split`.
        let mut push = Vec::with_capacity(MM_GEMV_PUSH_BYTES as usize);
        for v in [ka as u32, n as u32, split, 0] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        let partials = (split > 1)
            .then(|| ctx.create_storage_buffer(device_storage_bytes(FLOAT, split as usize * n)?))
            .transpose()?;
        with_pipeline(
            env.cache(),
            "GEMV",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(MM_GEMV)?,
                    MM_GEMV_BINDINGS,
                    MM_GEMV_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[a.buffer(), b.buffer(), partials.as_ref().unwrap_or(&out)],
                    &push,
                    [(n as u32).div_ceil(MM_GEMV_COLS), split, 1],
                )
            },
        )?;
        if let Some(partials) = partials {
            with_pipeline(
                env.cache(),
                "GEMV_reduce",
                || {
                    ctx.create_pipeline(
                        &compile_wgsl(MM_GEMV_REDUCE)?,
                        MM_GEMV_REDUCE_BINDINGS,
                        MM_GEMV_PUSH_BYTES,
                    )
                },
                |pipe| {
                    ctx.stream_dispatch(
                        pipe,
                        &[&partials, &out],
                        &push,
                        [(n as u32).div_ceil(256), 1, 1],
                    )
                },
            )?;
            ctx.defer_destroy(partials);
        }
    } else if m > 0 && n > 0 {
        let mut push = Vec::with_capacity(MM_PUSH_BYTES as usize);
        for v in [m as u32, ka as u32, n as u32, bc.out_shape.len() as u32] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        push_vec4s(&mut push, &bc.out_strides);
        push_vec4s(&mut push, &bc.a_strides);
        push_vec4s(&mut push, &bc.b_strides);
        // Thin matrices lose more to the blocked kernel's 64×64 padding than
        // they gain from its register blocking; see `prefer_blocked`.
        let blocked = mm_prefer_blocked(m, n);
        let (key, src, tile) = if blocked {
            ("MatMul", MM_MATMUL, MM_TILE_SIZE)
        } else {
            ("MatMul16", MM_MATMUL_SMALL, MM_SMALL_TILE_SIZE)
        };
        with_pipeline(
            env.cache(),
            key,
            || ctx.create_pipeline(&compile_wgsl(src)?, MM_BINDINGS, MM_PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[a.buffer(), b.buffer(), &out],
                    &push,
                    [
                        (n as u32).div_ceil(tile),
                        (m as u32).div_ceil(tile),
                        (batch as u32).max(1),
                    ],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: out_shape,
            elem_count,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `Mul`/`Add`: GPU if one operand is a device activation, otherwise host-side
/// (shape-math). ONNX broadcasting on both paths.
fn elementwise_binary(
    env: &mut Env,
    node: &NodeIr,
    gpu_op: &'static str,
    host_op: BinOp,
) -> Result<()> {
    let (a, b) = (&node.inputs[0], &node.inputs[1]);
    // Decision by **dtype**, not just residence: only f32 activations go on the
    // GPU. All shape-math (int64/int32) runs host-side — ORT parks even shape
    // scalars in our device memory, but they stay integers and reading them as
    // f32 would corrupt them.
    let both_f32 = env.dtype_of(a)? == FLOAT && env.dtype_of(b)? == FLOAT;
    // Even with f32: if both operands are already small on host, this is shape-math
    // (anchors, strides, scalars). Sending it to the GPU forces a later download
    // for a Reshape or a Slice — a submit+fence for a handful of bytes.
    let small_host = env.host_resident_small(a, Env::SMALL_HOST_BYTES)
        && env.host_resident_small(b, Env::SMALL_HOST_BYTES);
    if both_f32 && !small_host {
        env.ensure_device(a)?;
        env.ensure_device(b)?;
        gpu_binary(env, node, gpu_op)
    } else {
        let (ha, hb) = (env.host(a)?, env.host(b)?);
        let out = host_ops::binary(&ha, &hb, host_op)?;
        env.set(&node.outputs[0], Tensor::Host(out));
        Ok(())
    }
}

/// f32 binary on GPU with broadcasting (reuses the elementwise template).
fn gpu_binary(env: &mut Env, node: &NodeIr, gpu_op: &'static str) -> Result<()> {
    let ctx = env.context();
    let a = env.device(&node.inputs[0])?;
    let b = env.device(&node.inputs[1])?;
    let bc = broadcast(&a.shape, &b.shape)?;
    ensure!(
        bc.out_shape.len() <= MAX_RANK,
        "elementwise: rank {} > {}",
        bc.out_shape.len(),
        MAX_RANK
    );
    let elem_count: usize = bc.out_shape.iter().product::<i64>().max(0) as usize;
    let out = ctx.create_storage_buffer((elem_count.max(1) * 4) as u64)?;
    if elem_count > 0 {
        let mut push = Vec::with_capacity(112);
        push.extend_from_slice(&(elem_count as u32).to_le_bytes());
        push.extend_from_slice(&(bc.out_shape.len() as u32).to_le_bytes());
        push.extend_from_slice(&0u32.to_le_bytes());
        push.extend_from_slice(&0u32.to_le_bytes());
        push_vec4s(&mut push, &bc.out_strides);
        push_vec4s(&mut push, &bc.a_strides);
        push_vec4s(&mut push, &bc.b_strides);
        let src = BINARY_TEMPLATE.replace("OP", gpu_op);
        with_pipeline(
            env.cache(),
            gpu_op,
            || ctx.create_pipeline(&compile_wgsl(&src)?, 3, 112),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[a.buffer(), b.buffer(), &out],
                    &push,
                    [(elem_count as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: bc.out_shape,
            elem_count,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// Row/column split for ops that reduce along the last dimension.
fn row_cols(shape: &[i64], elem_count: usize, op: &str) -> Result<(usize, usize)> {
    let c = *shape.last().unwrap_or(&1) as usize;
    let rows = elem_count / c.max(1);
    ensure!(rows <= 65535, "{op}: troppe righe ({rows})");
    Ok((c, rows))
}

/// `axis` attribute (default -1) normalized; must be the last dimension.
/// `axis` attribute normalized to a non-negative index.
fn resolve_axis(node: &NodeIr, rank: i64, op: &str) -> Result<usize> {
    let raw = node
        .attrs
        .get("axis")
        .and_then(|a| a.as_i64())
        .unwrap_or(-1);
    let axis = if raw < 0 { raw + rank } else { raw };
    ensure!(
        (0..rank).contains(&axis),
        "{op}: axis {axis} out of rank {rank}"
    );
    Ok(axis as usize)
}

fn last_axis(node: &NodeIr, rank: i64, op: &str) -> Result<()> {
    let raw = node
        .attrs
        .get("axis")
        .and_then(|a| a.as_i64())
        .unwrap_or(-1);
    let axis = if raw < 0 { raw + rank } else { raw };
    ensure!(
        axis == rank - 1,
        "{op}: axis {axis} != ultima dim (rank {rank})"
    );
    Ok(())
}

/// `Softmax` f32 on the last dimension (one workgroup per row).
fn softmax(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    env.ensure_device(&node.inputs[0])?;
    let x = env.device(&node.inputs[0])?;
    let (shape, elem_count) = (x.shape.clone(), x.elem_count);
    let rank = shape.len() as i64;
    let axis = resolve_axis(node, rank, "Softmax")?;
    // `c` elements along the axis, spaced by `inner`; one row per (outer, inner)
    // pair. With the last axis `inner = 1` (contiguous rows).
    let c = shape[axis] as usize;
    let inner: usize = shape[axis + 1..].iter().product::<i64>().max(1) as usize;
    let rows = elem_count.checked_div(c).unwrap_or(0);
    // the grid is 2D: `rows` often exceeds the 65535 workgroup-per-axis limit
    let gx = rows.clamp(1, 32768) as u32;
    let gy = (rows as u32).div_ceil(gx);
    let out = ctx.create_storage_buffer((elem_count.max(1) * 4) as u64)?;
    if elem_count > 0 {
        let mut push = Vec::with_capacity(16);
        for v in [c as u32, inner as u32, rows as u32, gx] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        with_pipeline(
            env.cache(),
            "Softmax",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(SOFTMAX)?,
                    SOFTMAX_BINDINGS,
                    SOFTMAX_PUSH_BYTES,
                )
            },
            |pipe| ctx.stream_dispatch(pipe, &[x.buffer(), &out], &push, [gx, gy, 1]),
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape,
            elem_count,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `LayerNormalization` f32 on the last dimension; scale (+optional bias)
/// typically a per-channel initializer.
fn layernorm(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    let has_bias = node.inputs.len() > 2 && !node.inputs[2].is_empty();
    env.ensure_device(&node.inputs[0])?;
    env.ensure_device(&node.inputs[1])?;
    if has_bias {
        env.ensure_device(&node.inputs[2])?;
    }
    let epsilon = node
        .attrs
        .get("epsilon")
        .and_then(|a| a.as_f32())
        .unwrap_or(1e-5);

    let x = env.device(&node.inputs[0])?;
    let scale = env.device(&node.inputs[1])?;
    let bias = if has_bias {
        env.device(&node.inputs[2])?
    } else {
        scale
    };
    let (shape, elem_count) = (x.shape.clone(), x.elem_count);
    last_axis(node, shape.len() as i64, "LayerNormalization")?;
    let (c, rows) = row_cols(&shape, elem_count, "LayerNormalization")?;
    ensure!(
        scale.elem_count == c,
        "LayerNormalization: scale elem {} != {c}",
        scale.elem_count
    );
    let out = ctx.create_storage_buffer((elem_count.max(1) * 4) as u64)?;
    if elem_count > 0 {
        let mut push = Vec::with_capacity(LAYERNORM_PUSH_BYTES as usize);
        push.extend_from_slice(&(c as u32).to_le_bytes());
        push.extend_from_slice(&epsilon.to_le_bytes());
        push.extend_from_slice(&(has_bias as u32).to_le_bytes());
        with_pipeline(
            env.cache(),
            "LayerNormalization",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(LAYERNORM)?,
                    LAYERNORM_BINDINGS,
                    LAYERNORM_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[x.buffer(), scale.buffer(), bias.buffer(), &out],
                    &push,
                    [rows as u32, 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape,
            elem_count,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `Clip` (opset ≥6): bounds from attributes (opset 6) or scalar inputs
/// (opset ≥11); a missing bound does not constrain that side.
fn clip(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    // the bound is a scalar: reading it host-side does not touch the GPU when it
    // is an initializer, and it is the only way to put it in the push constants
    let limit = |env: &Env, index: usize, attr: &str, default: f32| -> Result<f32> {
        if let Some(value) = node.attrs.get(attr).and_then(AttrValue::as_f32) {
            return Ok(value);
        }
        match node.inputs.get(index) {
            Some(name) if !name.is_empty() => {
                let floats = env.host(name)?.to_f32()?;
                Ok(floats.first().copied().unwrap_or(default))
            }
            _ => Ok(default),
        }
    };
    let lo = limit(env, 1, "min", f32::NEG_INFINITY)?;
    let hi = limit(env, 2, "max", f32::INFINITY)?;

    env.ensure_device(&node.inputs[0])?;
    let x = env.device(&node.inputs[0])?;
    let (shape, elem_count) = (x.shape.clone(), x.elem_count);
    let out = ctx.create_storage_buffer((elem_count.max(1) * 4) as u64)?;
    if elem_count > 0 {
        let mut push = Vec::with_capacity(CLIP_PUSH_BYTES as usize);
        push.extend_from_slice(&(elem_count as u32).to_le_bytes());
        push.extend_from_slice(&lo.to_le_bytes());
        push.extend_from_slice(&hi.to_le_bytes());
        with_pipeline(
            env.cache(),
            "Clip",
            || ctx.create_pipeline(&compile_wgsl(CLIP)?, CLIP_BINDINGS, CLIP_PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[x.buffer(), &out],
                    &push,
                    [(elem_count as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape,
            elem_count,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// Spatial pooling variants, differing only in how they accumulate.
#[derive(Clone, Copy, PartialEq)]
enum PoolKind {
    Max,
    Average,
    GlobalAverage,
}

/// `MaxPool` / `AveragePool` / `GlobalAveragePool` 1D-2D.
///
/// The geometry is the same as conv (kernel, stride, pad, dilation,
/// `auto_pad`), with a per-channel window instead of a sum over channels;
/// the global case is the window covering the whole map.
fn pool(env: &mut Env, node: &NodeIr, kind: PoolKind) -> Result<()> {
    let ctx = env.context();
    let x_name = node.inputs[0].clone();
    let x_shape = env.shape_of(&x_name)?;
    ensure!(
        x_shape.len() == 3 || x_shape.len() == 4,
        "{}: only 1D/2D (rank {})",
        node.op,
        x_shape.len()
    );
    let c = x_shape[1];
    // geometry reused from conv: fake W [C, 1, kh, kw] — pooling does not sum
    // over channels, so the group is irrelevant
    let (mut node, spatial) = (node.clone(), x_shape.len() - 2);
    if kind == PoolKind::GlobalAverage {
        // full window, no pad: equivalent to a mean over the whole map
        node.attrs.insert(
            "kernel_shape".to_string(),
            AttrValue::Ints(x_shape[2..].to_vec()),
        );
        node.attrs
            .insert("strides".to_string(), AttrValue::Ints(vec![1; spatial]));
        node.attrs
            .insert("pads".to_string(), AttrValue::Ints(vec![0; 2 * spatial]));
    }
    let ks = node
        .attrs
        .get("kernel_shape")
        .and_then(|a| a.as_ints())
        .map(|s| s.to_vec())
        .with_context(|| format!("{}: attributo 'kernel_shape' assente", node.op))?;
    ensure!(
        node.attrs
            .get("ceil_mode")
            .and_then(AttrValue::as_i64)
            .unwrap_or(0)
            == 0,
        "{}: ceil_mode=1 not supported",
        node.op
    );
    let mut w_shape = vec![c, 1];
    w_shape.extend_from_slice(&ks);
    let g = conv_geometry(&node, &x_shape, &w_shape)?;
    let pad_count = node
        .attrs
        .get("count_include_pad")
        .and_then(AttrValue::as_i64)
        .unwrap_or(0);

    env.ensure_device(&x_name)?;
    let x = env.device(&x_name)?;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, g.total)?)?;
    if g.total > 0 {
        let mut push = Vec::with_capacity(POOL_PUSH_BYTES as usize);
        for v in [
            g.total as u32,
            g.c_out as u32,
            g.h_in as u32,
            g.w_in as u32,
            g.h_out as u32,
            g.w_out as u32,
            g.kh as u32,
            g.kw as u32,
            g.sh as u32,
            g.sw as u32,
            g.phb as u32,
            g.pwb as u32,
            g.dh as u32,
            g.dw as u32,
            pad_count as u32,
        ] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        let (key, init, acc, fin) = match kind {
            PoolKind::Max => ("MaxPool", POOL_MAX_INIT, POOL_MAX_ACC, POOL_MAX_FIN),
            PoolKind::Average | PoolKind::GlobalAverage => {
                ("AveragePool", POOL_AVG_INIT, POOL_AVG_ACC, POOL_AVG_FIN)
            }
        };
        with_pipeline(
            env.cache(),
            key,
            || {
                let src = pool_source(init, acc, fin);
                ctx.create_pipeline(&compile_wgsl(&src)?, POOL_BINDINGS, POOL_PUSH_BYTES)
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[x.buffer(), &out],
                    &push,
                    [(g.total as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: g.out_shape.clone(),
            elem_count: g.total,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// 2D bilinear `GridSample`: samples `X` [N, C, H, W] at the points of `grid`
/// [N, H_out, W_out, 2], coordinates normalized in [-1, 1] with (x, y) order.
///
/// This is the heart of deformable attention. `mode` other than `bilinear` and
/// `padding_mode = reflection` are rejected by `is_implemented_node`.
fn grid_sample(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    let (x_name, grid_name) = (node.inputs[0].clone(), node.inputs[1].clone());
    let x_shape = env.shape_of(&x_name)?;
    let grid_shape = env.shape_of(&grid_name)?;
    ensure!(
        x_shape.len() == 4 && grid_shape.len() == 4 && grid_shape[3] == 2,
        "GridSample: only 2D, X {x_shape:?} grid {grid_shape:?}"
    );
    let (n, c, h_in, w_in) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    let (h_out, w_out) = (grid_shape[1], grid_shape[2]);
    ensure!(
        grid_shape[0] == n,
        "GridSample: batch discordi, X {n} grid {}",
        grid_shape[0]
    );
    let align = node
        .attrs
        .get("align_corners")
        .and_then(AttrValue::as_i64)
        .unwrap_or(0);
    let padding = match node
        .attrs
        .get("padding_mode")
        .and_then(AttrValue::as_str)
        .unwrap_or("zeros")
    {
        "border" => PAD_BORDER,
        _ => PAD_ZEROS,
    };

    let out_shape = vec![n, c, h_out, w_out];
    let total = out_shape.iter().product::<i64>().max(0) as usize;

    env.ensure_device(&x_name)?;
    env.ensure_device(&grid_name)?;
    let x = env.device(&x_name)?;
    let grid = env.device(&grid_name)?;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, total)?)?;
    if total > 0 {
        let wg_x = (total as u32).div_ceil(256).clamp(1, 32768);
        let stride_y = wg_x * 256;
        let gy = (total as u32).div_ceil(stride_y);
        let mut push = Vec::with_capacity(GS_PUSH_BYTES as usize);
        for v in [
            total as u32,
            c as u32,
            h_in as u32,
            w_in as u32,
            h_out as u32,
            w_out as u32,
            align as u32,
            padding,
            stride_y,
        ] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        with_pipeline(
            env.cache(),
            "GridSample",
            || ctx.create_pipeline(&compile_wgsl(GRID_SAMPLE)?, GS_BINDINGS, GS_PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[x.buffer(), grid.buffer(), &out],
                    &push,
                    [wg_x, gy, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: out_shape,
            elem_count: total,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

#[derive(PartialEq, Clone, Copy)]
enum ReduceKind {
    Mean,
    Sum,
    Max,
    Min,
}

/// `ReduceMean` / `ReduceSum` / `ReduceMax` on **a single axis**.
///
/// Axes come from the `axes` attribute, where `fold_constant_params` already
/// promoted them if they were a constant input (Sum from opset 13, Mean/Max from 18).
///
/// If the input is **not** f32 the reduction goes host-side: the support check
/// looks at the node and not the dtype, so a `ReduceSum` on an int64 mask is
/// claimed like the others and must have a working path.
///
/// On-device the layout is that of `Softmax`: `c` elements along the axis
/// spaced by `inner`, one thread per output element.
fn reduce(env: &mut Env, node: &NodeIr, kind: ReduceKind) -> Result<()> {
    let x_name = node.inputs[0].clone();
    let x_shape = env.shape_of(&x_name)?;
    let rank = x_shape.len() as i64;

    let axes = node
        .attrs
        .get("axes")
        .and_then(|a| a.as_ints())
        .with_context(|| format!("{}: attributo 'axes' assente", node.op))?;
    ensure!(
        axes.len() == 1,
        "{}: only one axis (axes = {axes:?})",
        node.op
    );
    let axis = if axes[0] < 0 { axes[0] + rank } else { axes[0] };
    ensure!(
        (0..rank).contains(&axis),
        "{}: axis {axis} out of rank {rank}",
        node.op
    );
    let axis = axis as usize;
    let keepdims = node
        .attrs
        .get("keepdims")
        .and_then(AttrValue::as_i64)
        .unwrap_or(1)
        != 0;

    if env.dtype_of(&x_name)? != FLOAT {
        let op = match kind {
            ReduceKind::Mean => host_ops::RedOp::Mean,
            ReduceKind::Sum => host_ops::RedOp::Sum,
            ReduceKind::Max => host_ops::RedOp::Max,
            ReduceKind::Min => host_ops::RedOp::Min,
        };
        let x = env.host(&x_name)?;
        let out = host_ops::reduce(&x, axis, op, keepdims)?;
        env.set(&node.outputs[0], Tensor::Host(out));
        return Ok(());
    }

    let ctx = env.context();
    env.ensure_device(&x_name)?;
    let x = env.device(&x_name)?;
    let elem_count = x.elem_count;
    let c = x_shape[axis] as usize;
    let inner: usize = x_shape[axis + 1..].iter().product::<i64>().max(1) as usize;
    let rows = elem_count.checked_div(c).unwrap_or(0);

    let mut out_shape = x_shape;
    if keepdims {
        out_shape[axis] = 1;
    } else {
        out_shape.remove(axis);
    }

    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, rows)?)?;
    if rows > 0 {
        // griglia 2D: `rows` supera spesso il limite di 65535 workgroup per asse
        let wg_x = (rows as u32).div_ceil(256).clamp(1, 32768);
        let stride_y = wg_x * 256;
        let gy = (rows as u32).div_ceil(stride_y);
        let mut push = Vec::with_capacity(RED_PUSH_BYTES as usize);
        for v in [c as u32, inner as u32, rows as u32, stride_y] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        let (key, init, acc, fin) = match kind {
            ReduceKind::Mean => ("ReduceMean", RED_MEAN_INIT, RED_MEAN_ACC, RED_MEAN_FIN),
            ReduceKind::Sum => ("ReduceSum", RED_SUM_INIT, RED_SUM_ACC, RED_SUM_FIN),
            ReduceKind::Max => ("ReduceMax", RED_MAX_INIT, RED_MAX_ACC, RED_MAX_FIN),
            ReduceKind::Min => ("ReduceMin", RED_MIN_INIT, RED_MIN_ACC, RED_MIN_FIN),
        };
        with_pipeline(
            env.cache(),
            key,
            || {
                let src = reduce_source(init, acc, fin);
                ctx.create_pipeline(&compile_wgsl(&src)?, RED_BINDINGS, RED_PUSH_BYTES)
            },
            |pipe| ctx.stream_dispatch(pipe, &[x.buffer(), &out], &push, [wg_x, gy, 1]),
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: out_shape,
            elem_count: rows,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `Gemm` (opset ≥7): `Y = alpha · A' · B' + beta · C`, always 2D, with `C`
/// broadcastable.
fn gemm(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    let flag = |name: &str| {
        node.attrs
            .get(name)
            .and_then(AttrValue::as_i64)
            .unwrap_or(0)
            != 0
    };
    let (trans_a, trans_b) = (flag("transA"), flag("transB"));
    let alpha = node
        .attrs
        .get("alpha")
        .and_then(AttrValue::as_f32)
        .unwrap_or(1.0);
    let beta = node
        .attrs
        .get("beta")
        .and_then(AttrValue::as_f32)
        .unwrap_or(1.0);

    let a_shape = env.shape_of(&node.inputs[0])?;
    let b_shape = env.shape_of(&node.inputs[1])?;
    ensure!(
        a_shape.len() == 2 && b_shape.len() == 2,
        "Gemm: operands not 2D (A {a_shape:?}, B {b_shape:?})"
    );
    let (m, ka) = if trans_a {
        (a_shape[1], a_shape[0])
    } else {
        (a_shape[0], a_shape[1])
    };
    let (kb, n) = if trans_b {
        (b_shape[1], b_shape[0])
    } else {
        (b_shape[0], b_shape[1])
    };
    ensure!(ka == kb, "Gemm: K incompatibile ({ka} vs {kb})");

    let c_name = node.inputs.get(2).filter(|s| !s.is_empty()).cloned();
    let (c_rows, c_cols) = match &c_name {
        Some(name) => {
            let shape = env.shape_of(name)?;
            match shape.len() {
                0 => (1, 1),
                1 => (1, shape[0]),
                2 => (shape[0], shape[1]),
                other => bail!("Gemm: C with rank {other} not supported"),
            }
        }
        None => (0, 0),
    };

    env.ensure_device(&node.inputs[0])?;
    env.ensure_device(&node.inputs[1])?;
    if let Some(name) = &c_name {
        env.ensure_device(name)?;
    }
    let zero = env.cache().zero_scalar()?;
    let a = env.device(&node.inputs[0])?;
    let b = env.device(&node.inputs[1])?;
    // SAFETY: the cache never removes entries, the address stays valid (see `cache.rs`)
    let c_buf = match &c_name {
        Some(name) => env.device(name)?.buffer(),
        None => unsafe { &*zero },
    };

    let elem_count = (m * n).max(0) as usize;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, elem_count)?)?;
    if elem_count > 0 {
        let flags =
            u32::from(trans_a) | (u32::from(trans_b) << 1) | (u32::from(c_name.is_some()) << 2);
        let mut push = Vec::with_capacity(GEMM_PUSH_BYTES as usize);
        for v in [m as u32, ka as u32, n as u32, flags] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        push.extend_from_slice(&alpha.to_le_bytes());
        push.extend_from_slice(&beta.to_le_bytes());
        for v in [c_rows.max(1) as u32, c_cols.max(1) as u32] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        with_pipeline(
            env.cache(),
            "Gemm",
            || ctx.create_pipeline(&compile_wgsl(GEMM)?, GEMM_BINDINGS, GEMM_PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[a.buffer(), b.buffer(), c_buf, &out],
                    &push,
                    [
                        (n as u32).div_ceil(GEMM_TILE_SIZE),
                        (m as u32).div_ceil(GEMM_TILE_SIZE),
                        1,
                    ],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: vec![m, n],
            elem_count,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `Resize` (opset ≥11) on [N, C, H, W]: nearest or bilinear, with `scales`
/// or `sizes`; non-spatial dimensions must stay unchanged.
fn resize(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    let x_name = node.inputs[0].clone();
    let x_shape = env.shape_of(&x_name)?;
    ensure!(
        x_shape.len() == 4,
        "Resize: only [N,C,H,W] tensors (rank {})",
        x_shape.len()
    );
    let attr_str = |name: &str, default: &'static str| {
        node.attrs
            .get(name)
            .and_then(AttrValue::as_str)
            .unwrap_or(default)
            .to_string()
    };
    let mode = match attr_str("mode", "nearest").as_str() {
        "nearest" => MODE_NEAREST,
        "linear" => MODE_LINEAR,
        "cubic" => MODE_CUBIC,
        other => bail!("Resize: mode '{other}' not supported"),
    };
    // with exclude_outside = 1 the out-of-border neighbor weights must be zeroed
    // and renormalized: not implemented, default is 0
    ensure!(
        node.attrs
            .get("exclude_outside")
            .and_then(AttrValue::as_i64)
            .unwrap_or(0)
            == 0,
        "Resize: exclude_outside = 1 not supported"
    );
    let cubic_a = node
        .attrs
        .get("cubic_coeff_a")
        .and_then(AttrValue::as_f32)
        .unwrap_or(-0.75);
    let coord = match attr_str("coordinate_transformation_mode", "half_pixel").as_str() {
        "half_pixel" => COORD_HALF_PIXEL,
        "asymmetric" => COORD_ASYMMETRIC,
        "align_corners" => COORD_ALIGN_CORNERS,
        "pytorch_half_pixel" => COORD_PYTORCH_HALF_PIXEL,
        other => bail!("Resize: coordinate_transformation_mode '{other}' not supported"),
    };
    let nearest = match attr_str("nearest_mode", "round_prefer_floor").as_str() {
        "round_prefer_floor" => NEAREST_ROUND_PREFER_FLOOR,
        "round_prefer_ceil" => NEAREST_ROUND_PREFER_CEIL,
        "floor" => NEAREST_FLOOR,
        "ceil" => NEAREST_CEIL,
        other => bail!("Resize: nearest_mode '{other}' not supported"),
    };

    // optional inputs: roi (ignored without tf_crop_and_resize), scales, sizes.
    // They are shape-math: they live host-side, so no download cost.
    let optional = |env: &Env, index: usize| -> Result<Option<Vec<f32>>> {
        match node.inputs.get(index) {
            Some(name) if !name.is_empty() => Ok(Some(env.host(name)?.to_f32()?)),
            _ => Ok(None),
        }
    };
    let scales = optional(env, 2)?.filter(|v| !v.is_empty());
    let sizes = optional(env, 3)?.filter(|v| !v.is_empty());
    let (h_in, w_in) = (x_shape[2], x_shape[3]);
    let (h_out, w_out, scale_h, scale_w) = match (&scales, &sizes) {
        (Some(s), _) => {
            ensure!(s.len() == 4, "Resize: scales of length {}", s.len());
            ensure!(
                (s[0] - 1.0).abs() < 1e-6 && (s[1] - 1.0).abs() < 1e-6,
                "Resize: scala != 1 su batch/canali ({}, {})",
                s[0],
                s[1]
            );
            let h = (h_in as f32 * s[2]).floor() as i64;
            let w = (w_in as f32 * s[3]).floor() as i64;
            (h, w, s[2], s[3])
        }
        (None, Some(s)) => {
            ensure!(s.len() == 4, "Resize: sizes of length {}", s.len());
            let (h, w) = (s[2] as i64, s[3] as i64);
            ensure!(
                s[0] as i64 == x_shape[0] && s[1] as i64 == x_shape[1],
                "Resize: sizes change batch/channels"
            );
            (h, w, h as f32 / h_in as f32, w as f32 / w_in as f32)
        }
        (None, None) => bail!("Resize: neither scales nor sizes"),
    };
    ensure!(h_out > 0 && w_out > 0, "Resize: output degenere");

    env.ensure_device(&x_name)?;
    let x = env.device(&x_name)?;
    let out_shape = vec![x_shape[0], x_shape[1], h_out, w_out];
    let elem_count = (x_shape[0] * x_shape[1] * h_out * w_out).max(0) as usize;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, elem_count)?)?;
    if elem_count > 0 {
        let mut push = Vec::with_capacity(RESIZE_PUSH_BYTES as usize);
        for v in [
            elem_count as u32,
            (x_shape[0] * x_shape[1]) as u32,
            h_in as u32,
            w_in as u32,
            h_out as u32,
            w_out as u32,
            coord,
            mode,
            nearest,
        ] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        push.extend_from_slice(&cubic_a.to_le_bytes());
        push.extend_from_slice(&scale_h.to_le_bytes());
        push.extend_from_slice(&scale_w.to_le_bytes());
        with_pipeline(
            env.cache(),
            "Resize",
            || ctx.create_pipeline(&compile_wgsl(RESIZE)?, RESIZE_BINDINGS, RESIZE_PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[x.buffer(), &out],
                    &push,
                    [(elem_count as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: out_shape,
            elem_count,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `Constant`: the value lives in an attribute, stays host until needed.
fn constant(env: &mut Env, node: &NodeIr) -> Result<()> {
    let tensor = node
        .attrs
        .get("value")
        .and_then(AttrValue::as_tensor)
        .with_context(|| format!("Constant '{}': attributo 'value' assente", node.name))?;
    env.set(
        &node.outputs[0],
        Tensor::Host(HostTensor::new(
            tensor.dtype,
            tensor.shape.clone(),
            tensor.data.clone(),
        )),
    );
    Ok(())
}

/// Elementwise f32 unary op (out[i] = OP(x[i])).
fn unary(env: &mut Env, node: &NodeIr, key: &'static str, op_expr: &str) -> Result<()> {
    unary_with_helpers(env, node, key, op_expr, "")
}

/// `Gelu` (opset 20). The two `approximate` modes are different functions, not
/// two precisions of one: `tanh` is the formula the exporter chose and the one
/// the reference implements, so it is reproduced literally rather than being
/// served by the exact branch.
fn gelu(env: &mut Env, node: &NodeIr) -> Result<()> {
    let approximate = node
        .attrs
        .get("approximate")
        .and_then(AttrValue::as_str)
        .unwrap_or("none");
    match approximate {
        // 0.5 * v * (1 + erf(v / sqrt(2)))
        "none" => unary_with_helpers(
            env,
            node,
            "Gelu",
            "0.5 * v * (1.0 + erf_approx(v * 0.70710678118654752))",
            UNARY_HELPERS_ERF,
        ),
        // 0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715 * v^3)))
        "tanh" => unary(
            env,
            node,
            "GeluTanh",
            "0.5 * v * (1.0 + tanh(0.79788456080286536 * fma(0.044715, v * v * v, v)))",
        ),
        other => bail!("Gelu: approximate '{other}' not supported"),
    }
}

/// Like [`unary`], but prepends to the source the helper functions used by
/// the expression (WGSL lacks `erf`, for example).
fn unary_with_helpers(
    env: &mut Env,
    node: &NodeIr,
    key: &'static str,
    op_expr: &str,
    helpers: &str,
) -> Result<()> {
    let ctx = env.context(); // &'static, Copy: independent of the env borrow
    env.ensure_device(&node.inputs[0])?;
    let x = env.device(&node.inputs[0])?;
    let (shape, elem_count) = (x.shape.clone(), x.elem_count);
    let out = ctx.create_storage_buffer((elem_count.max(1) * 4) as u64)?;
    if elem_count > 0 {
        let src = format!("{helpers}{}", UNARY_TEMPLATE.replace("OP", op_expr));
        with_pipeline(
            env.cache(),
            key,
            || {
                let spirv = compile_wgsl(&src)?;
                ctx.create_pipeline(&spirv, 2, 4)
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[x.buffer(), &out],
                    &(elem_count as u32).to_le_bytes(),
                    [(elem_count as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape,
            elem_count,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}
