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

use crate::host_ops::{self, BOOL, BinOp, FLOAT, HostTensor, INT8, INT32, INT64, UINT8};
use crate::shaders::attention::{
    OUT as ATTN_OUT, OUT_BINDINGS as ATTN_OUT_BINDINGS, OUT_PUSH_BYTES as ATTN_OUT_PUSH_BYTES,
    PACK as ATTN_PACK, PACK_BINDINGS as ATTN_PACK_BINDINGS,
    PACK_PUSH_BYTES as ATTN_PACK_PUSH_BYTES, PAST as ATTN_PAST,
    PAST_BINDINGS as ATTN_PAST_BINDINGS, PAST_PUSH_BYTES as ATTN_PAST_PUSH_BYTES,
    ROTARY as ATTN_ROTARY, ROTARY_BINDINGS as ATTN_ROTARY_BINDINGS,
    ROTARY_PUSH_BYTES as ATTN_ROTARY_PUSH_BYTES, SCORES as ATTN_SCORES,
    SCORES_BINDINGS as ATTN_SCORES_BINDINGS, SCORES_PUSH_BYTES as ATTN_SCORES_PUSH_BYTES,
    WG as ATTN_WG,
};
use crate::shaders::conv::{
    BINDINGS as CONV_F32_BINDINGS, PUSH_BYTES as CONV_F32_PUSH_BYTES,
    SPLIT_REDUCE as CONV_SPLIT_REDUCE, SPLIT_REDUCE_BINDINGS as CONV_SPLIT_REDUCE_BINDINGS,
    Tactic as ConvTactic, TacticGeometry as ConvTacticGeometry, TacticLimits as ConvTacticLimits,
    generated_source as conv_generated_source, implementation_fingerprint as conv_fingerprint,
    prefer_blocked as conv_prefer_blocked, split_k as conv_split_k,
    tuned_tactic as conv_tuned_tactic,
};
use crate::shaders::conv_integer::{
    BINDINGS as CONV_INTEGER_BINDINGS, BLOCKED_TILE_SIZE as CONV_I_BLOCKED_TILE_SIZE,
    IM2COL as CONV_I_IM2COL, IM2COL_BINDINGS as CONV_I_IM2COL_BINDINGS,
    IM2COL_HWC as CONV_I_IM2COL_HWC, IM2COL_HWC_BINDINGS as CONV_I_IM2COL_HWC_BINDINGS,
    IM2COL_HWC_KEY as CONV_I_IM2COL_HWC_KEY, IM2COL_HWC_PUSH_BYTES as CONV_I_IM2COL_HWC_PUSH_BYTES,
    IM2COL_PUSH_BYTES as CONV_I_IM2COL_PUSH_BYTES, PUSH_BYTES as CONV_INTEGER_PUSH_BYTES,
    SPLIT_REDUCE as CONV_I_SPLIT_REDUCE, SPLIT_REDUCE_BINDINGS as CONV_I_SPLIT_REDUCE_BINDINGS,
    TILE_SIZE as CONV_I_TILE_SIZE, TO_HWC as CONV_I_TO_HWC,
    TO_HWC_BINDINGS as CONV_I_TO_HWC_BINDINGS, TO_HWC_KEY as CONV_I_TO_HWC_KEY,
    TO_HWC_PUSH_BYTES as CONV_I_TO_HWC_PUSH_BYTES, TO_HWC_TILE as CONV_I_TO_HWC_TILE,
    blocked as conv_i_blocked_source, blocked_splitk as conv_i_blocked_splitk_source,
    direct as conv_i_direct_source, implicit_gemm as conv_i_gemm_source,
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
    ADD_TRANSPOSE_PAD, ADD_TRANSPOSE_PAD_BINDINGS, ADD_TRANSPOSE_PAD_PUSH_BYTES,
    BINARY as BINARY_TEMPLATE, CAST_DEV, CAST_DEV_BINDINGS, CAST_DEV_PUSH_BYTES, CLIP,
    CLIP_BINDINGS, CLIP_PUSH_BYTES, DECOMPOSED_GELU, DECOMPOSED_GELU_BINDINGS, DECOMPOSED_GELU_FN,
    DECOMPOSED_GELU_PUSH_BYTES, INTERLEAVED_ROPE_BINDINGS, INTERLEAVED_ROPE_BSHD,
    INTERLEAVED_ROPE_PUSH_BYTES, INTERLEAVED_ROPE_SCALE, INTERLEAVED_ROPE_TRANSPOSE, IS_NAN,
    IS_NAN_BINDINGS, IS_NAN_PUSH_BYTES, MAX_RANK, MUL_ADD_RELU_BINDINGS, MUL_ADD_RELU_PUSH_BYTES,
    PACK_POSITIVE_BITS, PACK_POSITIVE_BITS_BINDINGS, PACK_POSITIVE_BITS_PUSH_BYTES, POW_EXPR,
    UNARY as UNARY_TEMPLATE, UNARY_HELPERS_ERF, WHERE, WHERE_BINDINGS, WHERE_PUSH_BYTES,
    interleaved_rope_groups, interleaved_rope_spirv, mul_add_relu_spirv,
};
use crate::shaders::gather_block_quantized::{
    BINDINGS as GBQ_BINDINGS, GATHER_BLOCK_QUANTIZED as GBQ_GATHER, PUSH_BYTES as GBQ_PUSH_BYTES,
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
    ANY_K as MMI_ANY_K, ANY_K_BINDINGS as MMI_ANY_K_BINDINGS,
    ANY_K_EPILOGUE_BINDINGS as MMI_ANY_K_EPILOGUE_BINDINGS,
    ANY_K_EPILOGUE_KEY as MMI_ANY_K_EPILOGUE_KEY,
    ANY_K_EPILOGUE_PUSH_BYTES as MMI_ANY_K_EPILOGUE_PUSH_BYTES, ANY_K_KEY as MMI_ANY_K_KEY,
    ANY_K_PUSH_BYTES as MMI_ANY_K_PUSH_BYTES, COOP_BINDINGS as MMI_COOP_BINDINGS,
    COOP_EPILOGUE_BINDINGS as MMI_COOP_EPILOGUE_BINDINGS,
    COOP_EPILOGUE_PUSH_BYTES as MMI_COOP_EPILOGUE_PUSH_BYTES,
    COOP_PUSH_BYTES as MMI_COOP_PUSH_BYTES, CoopVariant, FLIP_BINDINGS as MMI_FLIP_BINDINGS,
    FLIP_BYTES as MMI_FLIP_BYTES, FLIP_KEY as MMI_FLIP_KEY, FLIP_PUSH_BYTES as MMI_FLIP_PUSH_BYTES,
    MATMUL_BINDINGS as MMI_BINDINGS, MATMUL_PUSH_BYTES as MMI_PUSH_BYTES, PACK_B as MMI_PACK_B,
    PACK_BINDINGS as MMI_PACK_BINDINGS, PACK_PUSH_BYTES as MMI_PACK_PUSH_BYTES,
    PACKED_KEY as MMI_PACKED_KEY, SIGN_FLIP_BYTE as MMI_SIGN_FLIP_BYTE,
    SIGN_FLIP_WORD as MMI_SIGN_FLIP_WORD, SMALL_EPILOGUE_KEY as MMI_SMALL_EPILOGUE_KEY,
    TILE_SIZE as MMI_TILE_SIZE, VECTOR_KEY as MMI_VECTOR_KEY,
    any_k_epilogue_spirv as mmi_any_k_epilogue_spirv, coop_applies as mmi_coop_applies,
    coop_variant_for as mmi_coop_variant, matmul as mmi_matmul, small_applies as mmi_small_applies,
    small_epilogue_spirv as mmi_small_epilogue_spirv, small_grid as mmi_small_grid,
};
use crate::shaders::matmul_nbits::{
    BINDINGS as MMNB_BINDINGS, MATMUL_NBITS as MMNB_MATMUL, PUSH_BYTES as MMNB_PUSH_BYTES,
    Route as MmnbRoute, route as mmnb_route, wide_source as mmnb_wide_source,
};
use crate::shaders::movement::{
    CONCAT, CONCAT_BINDINGS, CONCAT_PUSH_BYTES, GATHER, GATHER_BINDINGS, GATHER_PUSH_BYTES, PAD,
    PAD_BINDINGS, PAD_PUSH_BYTES, SLICE, SLICE_BINDINGS, SLICE_PUSH_BYTES,
    TRANSPOSE as WGSL_TRANSPOSE, TRANSPOSE_BINDINGS, TRANSPOSE_BYTES, TRANSPOSE_PUSH_BYTES,
    TRANSPOSE_ROWS, TRANSPOSE_TILED,
};
use crate::shaders::normalization::{
    INSTANCENORM, INSTANCENORM_BINDINGS, INSTANCENORM_CHUNK, INSTANCENORM_PUSH_BYTES, LAYERNORM,
    LAYERNORM_BINDINGS, LAYERNORM_PUSH_BYTES, SOFTMAX, SOFTMAX_BINDINGS, SOFTMAX_PUSH_BYTES,
};
use crate::shaders::pooling::{
    AVG_ACC as POOL_AVG_ACC, AVG_FIN as POOL_AVG_FIN, AVG_INIT as POOL_AVG_INIT,
    BINDINGS as POOL_BINDINGS, MAX_ACC as POOL_MAX_ACC, MAX_FIN as POOL_MAX_FIN,
    MAX_INIT as POOL_MAX_INIT, PUSH_BYTES as POOL_PUSH_BYTES, source as pool_source,
};
use crate::shaders::push_vec4s;
use crate::shaders::qlinear::{
    MAXPOOL_Q, MAXPOOL_Q_BINDINGS, MAXPOOL_Q_PUSH_BYTES, QLINEAR_ADD, QLINEAR_ADD_BINDINGS,
    QLINEAR_ADD_PUSH_BYTES, REQUANTIZE, REQUANTIZE_BINDINGS, REQUANTIZE_PUSH_BYTES,
};
use crate::shaders::quantize_linear::{
    BINDINGS as QDQ_BINDINGS, DEQUANTIZE, DEQUANTIZE_I32, PUSH_BYTES as QDQ_PUSH_BYTES, QUANTIZE,
};
use crate::shaders::reduction::{
    ARGMAX_BINDINGS, ARGMAX_FINAL, ARGMAX_FINAL_PUSH_BYTES, ARGMAX_PARTIAL,
    ARGMAX_PARTIAL_PUSH_BYTES, BINDINGS as RED_BINDINGS, MAX_ACC as RED_MAX_ACC,
    MAX_FIN as RED_MAX_FIN, MAX_INIT as RED_MAX_INIT, MEAN_ACC as RED_MEAN_ACC,
    MEAN_FIN as RED_MEAN_FIN, MEAN_INIT as RED_MEAN_INIT, MIN_ACC as RED_MIN_ACC,
    MIN_FIN as RED_MIN_FIN, MIN_INIT as RED_MIN_INIT, PUSH_BYTES as RED_PUSH_BYTES,
    SUM_ACC as RED_SUM_ACC, SUM_FIN as RED_SUM_FIN, SUM_INIT as RED_SUM_INIT,
    source as reduce_source, splits as argmax_splits,
};
use crate::shaders::resize::{
    BINDINGS as RESIZE_BINDINGS, COORD_ALIGN_CORNERS, COORD_ASYMMETRIC, COORD_HALF_PIXEL,
    COORD_PYTORCH_HALF_PIXEL, MODE_CUBIC, MODE_LINEAR, MODE_NEAREST, NEAREST_CEIL, NEAREST_FLOOR,
    NEAREST_ROUND_PREFER_CEIL, NEAREST_ROUND_PREFER_FLOOR, PUSH_BYTES as RESIZE_PUSH_BYTES, RESIZE,
};
use crate::{
    AttrValue, Broadcast, DeviceBuffer as BufRef, DeviceTensor as DevTensor, ExecutionEnv, GraphIr,
    NodeIr, Tensor, broadcast, device_storage_bytes, elem_size,
};
use crate::{KernelCache, PipelineKey};
use anyhow::{Context as _, Result, bail, ensure};
use std::collections::{HashMap, HashSet};
use vk_compute::{BufferSlice, ComputePipeline, GpuBuffer, VkContext, compile_wgsl};

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
            | "SimplifiedLayerNormalization"
            | "SkipSimplifiedLayerNormalization"
            | "GroupQueryAttention"
            | "RotaryEmbedding"
            | "Cast"
            | "Mul"
            | "Add"
            | "Sub"
            | "Div"
            | "Shape"
            | "Reshape"
            | "Identity"
            | "Unsqueeze"
            | "Squeeze"
            | "Transpose"
            | "Concat"
            | "DynamicQuantizeLinear"
            | "MatMulInteger"
            | "MatMul"
            | "MatMulNBits"
            | "Gather"
            | "GatherND"
            | "GatherBlockQuantized"
            | "Slice"
            | "Where"
            | "Conv"
            | "ConvInteger"
            | "ConvTranspose"
            | "QLinearConv"
            | "QLinearMatMul"
            | "QLinearAdd"
            | "QLinearGlobalAveragePool"
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
            | "LessOrEqual"
            | "Greater"
            | "GreaterOrEqual"
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
            | "ArgMax"
            | "ArgMin"
            | "Flatten"
            | "LeakyRelu"
            | "Elu"
            | "IsNaN"
            | "OneHot"
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
            | "Sign"
            | "InstanceNormalization"
            | "Einsum"
            | "FlashAttention"
            | "MatMulIntegerScaleBias"
            | "ConvIntegerScaleBias"
            | "DecomposedGelu"
            | "DecomposedGeluQuantize"
            | "InterleavedRope"
            | "PackPositiveBits"
            | "AddTransposePad"
            | "MulAddRelu"
            | "AddLayerNormalization"
    )
}

/// Like [`is_implemented`], but also inspects node **attributes**.
///
/// The op name alone is not enough: `Resize` has modes the kernels do not
/// cover (`cubic`), pooling has `ceil_mode`, and claiming a node that cannot
/// then be run is a runtime error instead of a CPU fallback
/// (`op-plan.md` §4b). Whoever decides coverage must use this.
/// Highest `ai.onnx` opset any model in the suite is validated against.
///
/// A model above it is refused whole, and that is the honest answer rather than
/// a conservative one: the kernels have not been read against a newer spec, and
/// nothing here would notice a semantic change. Exporting at a supported opset
/// is a one-line change in every export tool, which is a better outcome for a
/// user than a plausible wrong number.
pub const MAX_OPSET: i32 = 21;

/// Lowest `ai.onnx` opset whose *form* each kernel reads.
///
/// Only ops whose contract migrated under them are listed; anything absent has
/// no lower bound. Every entry is a form the code actually reads, not a guess
/// about the spec:
///
/// - `Pad` takes `pads` from `inputs[1]`; before opset 11 it is an attribute
///   and that input does not exist.
/// - `Slice` takes `starts`/`ends` from inputs; attributes before opset 10.
/// - `TopK` takes `K` from `inputs[1]`; an attribute before opset 10.
/// - `Resize`'s `coordinate_transformation_mode`, which
///   [`is_implemented_node`] reads with a `half_pixel` default, does not exist
///   before opset 11 — where the operator is `asymmetric` by definition, so the
///   default would silently pick the wrong transform.
///
/// `Squeeze`, `Unsqueeze`, `Clip` and `Split` also migrated, and are absent on
/// purpose: their implementations read the input when present and fall back to
/// the attribute, so both forms run.
const MIN_OPSET: &[(&str, i32)] = &[("Pad", 11), ("Resize", 11), ("Slice", 10), ("TopK", 10)];

/// Whether the opset the node's domain was imported at is one the kernels were
/// written against.
///
/// Contrib domains (`com.microsoft`) are versioned on their own axis at
/// version 1 and are exempt: the `ai.onnx` bounds would be meaningless there,
/// and the ops themselves are constrained by arity and attributes in
/// [`is_implemented_node`].
///
/// An opset of 0 means the producer could not resolve one. That is treated as
/// unknown and allowed: refusing a node over a number we failed to read would
/// reject working models to guard against a hypothetical one.
fn opset_supported(node: &NodeIr) -> bool {
    if !node.domain.is_empty() && node.domain != "ai.onnx" {
        return true;
    }
    if node.opset == 0 {
        return true;
    }
    let min = MIN_OPSET
        .iter()
        .find(|(op, _)| *op == node.op)
        .map_or(1, |(_, v)| *v);
    node.opset >= min && node.opset <= MAX_OPSET
}

pub fn is_implemented_node(node: &NodeIr) -> bool {
    if !is_implemented(&node.op) || !opset_supported(node) {
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
        // the schema's optional second output is `inv_std_var`, which the kernel
        // does not produce: claiming a node that asks for it would silently
        // leave a value undefined
        "SimplifiedLayerNormalization" => {
            node.outputs.iter().filter(|o| !o.is_empty()).count() == 1
        }
        // the skip form, same kernel plus the residual. `mean` and
        // `inv_std_var` are refused for the reason above; the fourth output —
        // the residual sum, which 71 of qwen2.5-VL's 72 nodes consume — is
        // produced. `beta` and `bias` are not implemented: no export in the
        // suite carries them, and adding a term nobody exercises is a term
        // nobody checks.
        "SkipSimplifiedLayerNormalization" => {
            let out = |index: usize| node.outputs.get(index).is_some_and(|o| !o.is_empty());
            let present = |index: usize| node.inputs.get(index).is_some_and(|n| !n.is_empty());
            present(0)
                && present(1)
                && present(2)
                && !present(3)
                && !present(4)
                && out(0)
                && !out(1)
                && !out(2)
        }
        // half-split rotary only, like `GroupQueryAttention`'s fused one, and
        // no `scale`: the schema's custom scale is not applied by the kernel,
        // so claiming a node that asks for one would silently ignore it.
        "RotaryEmbedding" => {
            let float_attr = |name: &str, default: f32| {
                node.attrs
                    .get(name)
                    .and_then(AttrValue::as_f32)
                    .unwrap_or(default)
            };
            int_attr("interleaved", 0) == 0
                && int_attr("is_packed_batching", 0) == 0
                && float_attr("scale", 1.0) == 1.0
                && node.inputs.len() == 4
                && node.inputs.iter().all(|n| !n.is_empty())
                && node.outputs.len() == 1
        }
        // the kernel covers the shape gemma3 and qwen2.5-VL export and nothing
        // more: separate Q/K/V (not packed), half-split rotary, no attention
        // softcap, and the KV cache written out of place, which needs both
        // `present_*` outputs to exist
        "GroupQueryAttention" => {
            let float_attr = |name: &str, default: f32| {
                node.attrs
                    .get(name)
                    .and_then(AttrValue::as_f32)
                    .unwrap_or(default)
            };
            let present = |index: usize| node.inputs.get(index).is_some_and(|n| !n.is_empty());
            let rotary = int_attr("do_rotary", 0) != 0;
            float_attr("softcap", 0.0) == 0.0
                && int_attr("rotary_interleaved", 0) == 0
                // ONNX Runtime hands the EP `smooth_softmax = -1` on a node
                // whose file carries no such attribute, so a negative value is
                // its "unset" marker and only a positive one means enabled
                && int_attr("smooth_softmax", 0) <= 0
                && present(1)
                && present(2)
                && present(3)
                && present(4)
                && (!rotary || (present(7) && present(8)))
                && node.outputs.len() == 3
                && node.outputs.iter().all(|o| !o.is_empty())
        }
        // 4 bits only, and only with an explicit zero point: a missing one
        // defaults to 8, but the kernel binds it unconditionally and there is no
        // export in the suite that omits it. `g_idx` (per-`k` block indirection)
        // and the fused bias are not implemented. `block_size` must be a
        // multiple of 8 so a packed `u32` never straddles two blocks.
        "MatMulNBits" => {
            let present = |index: usize| node.inputs.get(index).is_some_and(|n| !n.is_empty());
            let block_size = int_attr("block_size", 0);
            int_attr("bits", 0) == 4
                && block_size > 0
                && block_size % 8 == 0
                && int_attr("K", 0) > 0
                && int_attr("N", 0) > 0
                && present(2)
                && present(3)
                && !present(4)
                && !present(5)
        }
        // 4 bits, same packing as `MatMulNBits`, and the only axis pair that
        // makes a gathered row contiguous: gather along 0, quantize along the
        // last axis of a 2-D table. `block_size` must be even so a block starts
        // on a byte boundary. The zero point is required for the same reason as
        // in `MatMulNBits` — the kernel binds it unconditionally.
        "GatherBlockQuantized" => {
            let present = |index: usize| node.inputs.get(index).is_some_and(|n| !n.is_empty());
            let block_size = int_attr("block_size", 0);
            int_attr("bits", 4) == 4
                && block_size > 0
                && block_size % 2 == 0
                && int_attr("gather_axis", 0) == 0
                && int_attr("quantize_axis", 1) == 1
                && present(2)
                && present(3)
        }
        // only the form with `axes` as an attribute and a single axis: with axes
        // as input the value is not known when looking at the node
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" => node
            .attrs
            .get("axes")
            .and_then(AttrValue::as_ints)
            .is_some_and(|a| a.len() == 1),
        // `select_last_index = 1` reverses the tie-break, and the kernel keeps
        // the first occurrence; a graph that asks for the last one is refused
        // rather than answered with the other index
        "ArgMax" | "ArgMin" => int_attr("select_last_index", 0) == 0,
        "GatherND" => int_attr("batch_dims", 0) == 0,
        "GridSample" => {
            string_attr("mode", "bilinear") == "bilinear"
                && matches!(
                    string_attr("padding_mode", "zeros").as_str(),
                    "zeros" | "border"
                )
        }
        "Gelu" => matches!(string_attr("approximate", "none").as_str(), "none" | "tanh"),
        // only the equations that are a batched MatMul once reshaped; see
        // `einsum_as_matmul`
        "Einsum" => {
            node.inputs.len() == 2
                && node
                    .attrs
                    .get("equation")
                    .and_then(AttrValue::as_str)
                    .and_then(einsum_as_matmul)
                    .is_some()
        }
        "InstanceNormalization" => {
            node.inputs.len() == 3
                && node.inputs.iter().all(|n| !n.is_empty())
                && node.outputs.len() == 1
        }
        // `add`/`mul`/`min`/`max` accumulate instead of overwriting
        "ScatterND" => string_attr("reduction", "none") == "none",
        // `ConvIntegerScaleBias` is produced by
        // `rewrite::fuse_conv_integer_epilogue` and by nothing else: the same
        // name in a custom domain is somebody else's operator
        "ConvIntegerScaleBias"
            if node.domain != crate::rewrite::REWRITE_DOMAIN
                || !matches!(node.inputs.len(), 6 | 7)
                || node.outputs.len() != 1 =>
        {
            false
        }
        "Conv" | "ConvInteger" | "ConvIntegerScaleBias" => matches!(
            string_attr("auto_pad", "NOTSET").as_str(),
            "NOTSET" | "VALID" | "SAME_UPPER" | "SAME_LOWER"
        ),
        // The QOperator (static int8) family. What is checked here is what the
        // node carries; the constraints that live in the **values** — which
        // quantization parameters are constants, which are per-channel, and
        // whether a per-channel zero point is zero — are in
        // [`unsupported_quantization`], because a `NodeIr` does not carry its
        // initializers. Both refuse at load time; neither is a runtime check.
        "QLinearConv" => {
            let present = |index: usize| node.inputs.get(index).is_some_and(|n| !n.is_empty());
            matches!(
                string_attr("auto_pad", "NOTSET").as_str(),
                "NOTSET" | "VALID" | "SAME_UPPER" | "SAME_LOWER"
            ) && (8..=9).contains(&node.inputs.len())
                && (0..8).all(present)
                && node.outputs.len() == 1
        }
        "QLinearMatMul" => {
            let present = |index: usize| node.inputs.get(index).is_some_and(|n| !n.is_empty());
            node.inputs.len() == 8 && (0..8).all(present) && node.outputs.len() == 1
        }
        // contrib op: `A`, `B` and the three (scale, zero-point) pairs. The
        // schema makes the zero points optional; every export measured carries
        // all of them, and defaulting a missing one to zero is an assumption
        // nothing would check.
        "QLinearAdd" => {
            let present = |index: usize| node.inputs.get(index).is_some_and(|n| !n.is_empty());
            node.inputs.len() == 8 && (0..8).all(present) && node.outputs.len() == 1
        }
        // contrib op. `channels_last = 1` is NHWC, a different reduction and a
        // different output layout: refused rather than transposed underneath.
        "QLinearGlobalAveragePool" => {
            let present = |index: usize| node.inputs.get(index).is_some_and(|n| !n.is_empty());
            int_attr("channels_last", 0) == 0
                && node.inputs.len() == 5
                && (0..5).all(present)
                && node.outputs.len() == 1
        }
        // `output_shape` inverts the relationship — pads are derived from the
        // requested output — and `SAME_*` is meaningful only in those terms
        "ConvTranspose" => {
            matches!(
                string_attr("auto_pad", "NOTSET").as_str(),
                "NOTSET" | "VALID"
            ) && !node.attrs.contains_key("output_shape")
        }
        // produced by `rewrite::fuse_attention` and by nothing else: the same
        // name in a custom domain is somebody else's operator
        "FlashAttention" => {
            node.domain == crate::rewrite::REWRITE_DOMAIN
                && matches!(node.inputs.len(), 3 | 4)
                && node.outputs.len() == 1
        }
        // produced by `rewrite::fuse_matmul_integer_epilogue`, same reasoning
        "MatMulIntegerScaleBias" => {
            node.domain == crate::rewrite::REWRITE_DOMAIN
                && matches!(node.inputs.len(), 6 | 7)
                && node.outputs.len() == 1
        }
        // produced by `rewrite::fuse_decomposed_gelu`, same reasoning
        "DecomposedGelu" => {
            node.domain == crate::rewrite::REWRITE_DOMAIN
                && node.inputs.len() == 4
                && node.outputs.len() == 1
        }
        // produced by `rewrite::fuse_gelu_quantize`, same reasoning; the
        // outputs are those of `DynamicQuantizeLinear`
        "DecomposedGeluQuantize" => {
            node.domain == crate::rewrite::REWRITE_DOMAIN
                && node.inputs.len() == 4
                && (1..=3).contains(&node.outputs.len())
        }
        // produced by `rewrite::fuse_interleaved_rope`, same reasoning; the
        // fourth input, the scale, is optional
        "InterleavedRope" => {
            node.domain == crate::rewrite::REWRITE_DOMAIN
                && (3..=4).contains(&node.inputs.len())
                && node.outputs.len() == 1
        }
        // produced by `rewrite::fuse_pack_positive_bits`, same reasoning
        "PackPositiveBits" => {
            node.domain == crate::rewrite::REWRITE_DOMAIN
                && node.inputs.len() == 2
                && node.outputs.len() == 1
        }
        // produced by `rewrite::fuse_mul_add_relu`, same reasoning
        "MulAddRelu" => {
            node.domain == crate::rewrite::REWRITE_DOMAIN
                && node.inputs.len() == 3
                && node.outputs.len() == 1
        }
        // produced by `rewrite::fuse_add_transpose_pad`, same reasoning; the
        // fifth input, the pad value, is optional
        "AddTransposePad" => {
            node.domain == crate::rewrite::REWRITE_DOMAIN
                && (4..=5).contains(&node.inputs.len())
                && node.outputs.len() == 1
        }
        // produced by `rewrite::fuse_add_layernorm`, same reasoning; the bias
        // and the escaping sum are optional
        "AddLayerNormalization" => {
            node.domain == crate::rewrite::REWRITE_DOMAIN
                && (3..=4).contains(&node.inputs.len())
                && (1..=2).contains(&node.outputs.len())
        }
        _ => true,
    }
}

/// Ops whose kernel reads input 0 as `f32` while the ONNX schema admits more.
///
/// **This list is derived from the kernels, not from the schema, and not from
/// a bug report.** The first draft was written from the incident below and
/// contained `MaxPool`, `AveragePool`, `GlobalAveragePool` and `CumSum` — all
/// four wrong, because that incident was fixed *in the kernel*: `pool()`
/// branches on `UINT8`/`INT8` and dispatches `MaxPool_q`, and `CumSum` is a
/// host op over any dtype. Adding them here cost resnet50-int8 and both roberta
/// entries a convex block each, which the suite caught. Before adding an op,
/// read its dispatch and confirm there is no integer branch.
///
/// The list is short on purpose. It is not a transcription of the schema — it
/// is the set of ops whose kernel reads the buffer as `f32` while the spec
/// admits more, which is the shape of the one bug this exists to stop: a
/// `MaxPool` on a `uint8` tensor was claimed by the float kernel and answered
/// with reinterpreted bytes — `max|Δ| = 8.086`, argmax 489 → 611, wrong and
/// silent, caught only because that model ships golden data.
///
/// An op absent from this table is unconstrained, and a value whose type the
/// producer could not resolve is allowed: this refuses what is known to be
/// wrong, never what is merely unknown.
const FLOAT_ONLY_INPUT0: &[&str] = &[
    "Softmax",
    "LayerNormalization",
    "SimplifiedLayerNormalization",
    "SkipSimplifiedLayerNormalization",
    "AddLayerNormalization",
    "GroupQueryAttention",
    "RotaryEmbedding",
    "Conv",
    "ConvTranspose",
    "Gemm",
    "GridSample",
    "Sign",
    "InstanceNormalization",
    "Einsum",
    "FlashAttention",
];

/// Why this node's operand types are outside what the kernels read, or `None`.
///
/// The third of the load-time refusals, and the one that closes the silent
/// class. [`is_implemented_node`] reads the node — op, opset, attributes,
/// arity. [`unsupported_quantization`] reads the operand *values*. This reads
/// the operand *types*, which live in neither: a `NodeIr` does not carry them,
/// and an intermediate value is in no initializer. `GraphIr::value_types`
/// carries them instead, filled by whichever frontend built the graph.
pub fn unsupported_dtype(
    node: &NodeIr,
    types: &std::collections::HashMap<String, i32>,
) -> Option<String> {
    if !FLOAT_ONLY_INPUT0.contains(&node.op.as_str()) {
        return None;
    }
    let name = node.inputs.first().filter(|n| !n.is_empty())?;
    let dtype = *types.get(name.as_str())?;
    (dtype != FLOAT).then(|| {
        format!(
            "{} on a non-float input (ONNX element type {dtype}); the kernel reads f32",
            node.op
        )
    })
}

/// Why a QOperator node cannot be run, when the reason is in its **values**.
///
/// `is_implemented_node` sees a `NodeIr`: op, attributes, input names. The
/// static-int8 family puts the rest of its contract in the operands — a scale
/// is per-tensor or per-channel depending on its *length*, a zero point is
/// symmetric or not depending on its *content* — and neither is knowable from
/// the node alone. This is the other half of the same refusal, and it runs
/// where the initializers are: `Executor::new`, before anything executes.
///
/// The constraints are the ones measured on the QOperator exports in the
/// matrix (`scripts/qlinear-census.py`), not the ones the schema allows:
///
/// - every quantization parameter is a **constant**. A scale computed at
///   runtime would have to be read back to build the epilogue.
/// - activation scales and zero points are **per-tensor scalars**.
/// - the weight scale may be **per-channel**, and then the weight zero point
///   must match it in length and be **identically zero** — symmetric weights.
///   All 105 `QLinearConv` nodes of resnet50-int8 and mobilenetv2-int8 are.
///   A non-zero per-channel zero point would add a `w_zp[c]·Σx` correction
///   along the reduction axis; no model in reach exercises it, so it is
///   refused loudly instead of written blind.
pub fn unsupported_quantization(
    node: &NodeIr,
    initializers: &HashMap<String, crate::InitializerIr>,
) -> Option<String> {
    let constant = |index: usize| {
        node.inputs
            .get(index)
            .filter(|name| !name.is_empty())
            .and_then(|name| initializers.get(name))
    };
    let elems = |init: &crate::InitializerIr| init.shape.iter().product::<i64>().max(1);
    let refuse = |slot: &str, why: &str| Some(format!("{}: {slot} {why}", node.op));

    // (scale, zero point) pairs that must be per-tensor scalars, by input index
    let scalar_pairs: &[(usize, usize, &str)] = match node.op.as_str() {
        "QLinearConv" => &[(1, 2, "x"), (6, 7, "y")],
        "QLinearMatMul" => &[(1, 2, "a"), (6, 7, "y")],
        "QLinearAdd" => &[(1, 2, "A"), (4, 5, "B"), (6, 7, "C")],
        "QLinearGlobalAveragePool" => &[(1, 2, "x"), (3, 4, "y")],
        _ => return None,
    };
    for &(scale, zero, slot) in scalar_pairs {
        let (Some(scale), Some(zero)) = (constant(scale), constant(zero)) else {
            return refuse(slot, "scale or zero point is not a constant");
        };
        if scale.dtype != FLOAT {
            return refuse(slot, "scale is not float32");
        }
        if elems(scale) != 1 || elems(zero) != 1 {
            return refuse(slot, "scale or zero point is not per-tensor");
        }
    }

    // the weight side, where per-channel is allowed and asymmetry is not
    let weights: Option<(usize, usize, &str)> = match node.op.as_str() {
        "QLinearConv" => Some((4, 5, "w")),
        "QLinearMatMul" => Some((4, 5, "b")),
        _ => None,
    };
    if let Some((scale, zero, slot)) = weights {
        let (Some(scale), Some(zero)) = (constant(scale), constant(zero)) else {
            return refuse(slot, "scale or zero point is not a constant");
        };
        if scale.dtype != FLOAT {
            return refuse(slot, "scale is not float32");
        }
        if elems(scale) != elems(zero) {
            return refuse(slot, "scale and zero point have different lengths");
        }
        if elems(zero) > 1 && zero.data.iter().any(|byte| *byte != 0) {
            return refuse(
                slot,
                "per-channel zero point is not zero (asymmetric weights)",
            );
        }
    }

    // `QLinearConv`'s bias lives in the accumulator's domain, already scaled by
    // `x_scale · w_scale`: an f32 bias would mean a different epilogue
    if node.op == "QLinearConv"
        && let Some(bias) = constant(8)
        && bias.dtype != INT32
    {
        return refuse("B", "bias is not int32");
    }
    None
}

/// Runs a closure with the pipeline (cached in the session) for an op key.
fn with_pipeline<R>(
    cache: &KernelCache<'_>,
    key: impl Into<PipelineKey>,
    build: impl FnOnce() -> Result<ComputePipeline>,
    run: impl FnOnce(&ComputePipeline) -> Result<R>,
) -> Result<R> {
    let key = key.into();
    vk_compute::stats::set_op(key.operation()); // Pareto attribution by dispatch type
    let pipeline = cache.pipeline(key, build)?;
    // SAFETY: the cache never removes entries and keeps them in `Box`: the
    // address stays valid for the lifetime of the cache, which outlives execution.
    run(unsafe { &*pipeline })
}

/// A device buffer holding `bytes`, shared across nodes and across steps when
/// the payload repeats.
///
/// Index and parameter payloads are identical from one call to the next far more
/// often than not — the same zero index in 74 nodes of qwen2.5-VL, the same
/// shape block in 8 — and they are constants, so one buffer serves them all.
/// Past the cache's size cap this falls back to a private buffer, which is what
/// every caller used to do.
fn param_buffer<'cache>(
    env: &Env<'cache, '_>,
    bytes: &[u8],
    scratch: &'cache mut Option<GpuBuffer>,
) -> Result<&'cache GpuBuffer> {
    if let Some(shared) = env.cache().constant(bytes)? {
        // the cache owns it for as long as the session lives
        return Ok(unsafe { &*shared });
    }
    let buffer = env
        .context()
        .create_storage_buffer(bytes.len().max(4) as u64)?;
    env.context().stream_upload(&buffer, bytes)?;
    Ok(scratch.insert(buffer))
}

/// Executes the graph nodes on the provided environment, with no host dependency.
///
/// Internal errors (anyhow, with context chain) are flattened into the core's
/// typed error: the public API does not expose `anyhow`.
pub fn execute(ir: &GraphIr, env: &mut Env<'_, '_>) -> crate::Result<()> {
    execute_nodes(ir, env).map_err(|e| crate::Error::Backend(format!("{e:#}")))
}

/// Runs the graph while a capture is open, reporting which commands each node
/// issued.
///
/// The ranges are what lets a plan tell a node it can replay from one it has to
/// run again: a node that issued nothing is host computation, and its result is
/// what the uploads of the next step will carry.
pub fn execute_traced(
    ir: &GraphIr,
    env: &mut Env<'_, '_>,
) -> crate::Result<Vec<std::ops::Range<usize>>> {
    let mut ranges = Vec::with_capacity(ir.nodes.len());
    execute_nodes_inner(ir, env, Some(&mut ranges))
        .map_err(|e| crate::Error::Backend(format!("{e:#}")))?;
    Ok(ranges)
}

/// Runs the host-only nodes of a plan, in graph order.
///
/// Everything else the step does is issued from the plan, so this is the only
/// interpretation a replayed token pays for: the mask, the positions and the
/// shapes, which change with every token and which nothing can record.
pub fn execute_host_nodes(
    ir: &GraphIr,
    env: &mut Env<'_, '_>,
    nodes: &[usize],
) -> crate::Result<()> {
    // no liveness here: every value keeps its buffer
    env.set_first_input_dies(false);
    for &index in nodes {
        exec_node(env, &ir.nodes[index]).map_err(|e| crate::Error::Backend(format!("{e:#}")))?;
    }
    Ok(())
}

fn execute_nodes(ir: &GraphIr, env: &mut Env<'_, '_>) -> Result<()> {
    execute_nodes_inner(ir, env, None)
}

fn execute_nodes_inner(
    ir: &GraphIr,
    env: &mut Env<'_, '_>,
    mut trace: Option<&mut Vec<std::ops::Range<usize>>>,
) -> Result<()> {
    let dead = dead_after(ir);
    let probe = std::env::var_os("ALLOC_PROBE").is_some();
    // Host time per op, against the dispatches each op issued: what says whether
    // the interpreter's cost is bookkeeping around a dispatch or host-side
    // computation the graph asks for, which a replayed plan would still have to
    // run. See `NODE_PROBE`.
    let timing = std::env::var_os("NODE_PROBE").is_some();
    let mut by_op: HashMap<&str, (u64, u64, u64)> = HashMap::new();
    let _span = vk_compute::stats::span("execute nodes");
    for (index, node) in ir.nodes.iter().enumerate() {
        let before = probe.then(vk_compute::stats::allocs);
        let clock = timing.then(std::time::Instant::now);
        let dispatched = timing.then(vk_compute::stats::dispatches);
        let issued = env.context().capture_len();
        // Not while capturing: a node that takes its input's buffer issues
        // nothing, and a plan would take it for host work to redo each step.
        env.set_first_input_dies(
            issued.is_none()
                && node
                    .inputs
                    .first()
                    .is_some_and(|input| dead[index].contains(&input.as_str())),
        );
        exec_node(env, node)?;
        if let Some(trace) = trace.as_mut() {
            let start = issued.unwrap_or(0);
            trace.push(start..env.context().capture_len().unwrap_or(start));
        }
        if let (Some(clock), Some(dispatched)) = (clock, dispatched) {
            let entry = by_op.entry(node.op.as_str()).or_default();
            entry.0 += clock.elapsed().as_nanos() as u64;
            entry.1 += 1;
            entry.2 += vk_compute::stats::dispatches() - dispatched;
        }
        if let Some(before) = before {
            let delta = vk_compute::stats::allocs() - before;
            if delta > 0 {
                let shape = env.shape_of(&node.outputs[0]).unwrap_or_default();
                eprintln!(
                    "ALLOC {delta} {} '{}' out {} {shape:?}",
                    node.op, node.name, node.outputs[0]
                );
            }
        }
        for name in &dead[index] {
            env.release(name);
        }
    }
    env.set_first_input_dies(false);
    if timing {
        let mut rows: Vec<_> = by_op.into_iter().collect();
        rows.sort_by_key(|(_, (ns, _, _))| std::cmp::Reverse(*ns));
        for (op, (ns, nodes, dispatches)) in rows.into_iter().take(12) {
            eprintln!(
                "NODE {op:<28} {:>7.3} ms  {nodes:>4} nodes  {dispatches:>4} dispatch",
                ns as f64 / 1e6
            );
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
    // Built by walking the nodes and their inputs in order, not by draining
    // `last`: the order a value is released in is the order its buffer goes
    // back to the pool, and the pool hands them out again in that same order.
    // Draining a `HashMap` would shuffle two values of the same size against
    // each other from one step to the next — measured, and it is exactly how a
    // replayed step ends up reading the wrong tensor.
    let mut dead = vec![Vec::new(); ir.nodes.len()];
    let mut placed: HashSet<&str> = HashSet::new();
    for (index, node) in ir.nodes.iter().enumerate() {
        for name in node.inputs.iter().filter(|n| !n.is_empty()) {
            if last.get(name.as_str()) == Some(&index) && placed.insert(name.as_str()) {
                dead[index].push(name.as_str());
            }
        }
    }
    dead
}

/// Executes a single node, inserting outputs into `env`.
fn exec_node(env: &mut Env, node: &NodeIr) -> Result<()> {
    vk_compute::stats::begin_node();
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
    if r.is_ok() {
        // after the dispatch: the output shapes only exist now, and so does the
        // pipeline key the work is charged to
        crate::work::record(node, &*env);
    }
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
        "LayerNormalization" => layernorm(env, node, Norm::Centered),
        "SimplifiedLayerNormalization" => layernorm(env, node, Norm::Rms),
        "SkipSimplifiedLayerNormalization" => layernorm(env, node, Norm::RmsSkip),
        "AddLayerNormalization" => add_layernorm(env, node),
        "GroupQueryAttention" => group_query_attention(env, node),
        "RotaryEmbedding" => rotary_embedding(env, node),
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
        "MatMulNBits" => matmul_nbits(env, node),
        "Gather" => gather(env, node),
        "GatherND" => gather_nd(env, node),
        "GatherBlockQuantized" => gather_block_quantized(env, node),
        "Slice" => slice(env, node),
        "Where" => where_op(env, node),
        "Conv" => conv_f32(env, node),
        "ConvInteger" => conv_integer(env, node),
        "QLinearConv" => qlinear_conv(env, node),
        "QLinearMatMul" => qlinear_matmul(env, node),
        "QLinearAdd" => qlinear_add(env, node),
        "QLinearGlobalAveragePool" => qlinear_global_average_pool(env, node),
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
        "LessOrEqual" => host_cmp(env, node, host_ops::CmpOp::LessOrEqual),
        "Greater" => host_cmp(env, node, host_ops::CmpOp::Greater),
        "GreaterOrEqual" => host_cmp(env, node, host_ops::CmpOp::GreaterOrEqual),
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
        "ArgMax" => argmax(env, node),
        "ArgMin" => argmin(env, node),
        "Flatten" => flatten(env, node),
        "Identity" => identity(env, node),
        "LeakyRelu" => leaky_relu(env, node),
        "Elu" => elu(env, node),
        "IsNaN" => is_nan(env, node),
        "OneHot" => one_hot(env, node),
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
        "Sign" => unary(env, node, "Sign", "sign(v)"),
        "InstanceNormalization" => instance_norm(env, node),
        "Einsum" => einsum(env, node),
        "FlashAttention" => flash_attention(env, node),
        "MatMulIntegerScaleBias" => matmul_integer_scale_bias(env, node),
        "ConvIntegerScaleBias" => conv_integer_scale_bias(env, node),
        "DecomposedGelu" => decomposed_gelu(env, node),
        "DecomposedGeluQuantize" => decomposed_gelu_quantize(env, node),
        "InterleavedRope" => interleaved_rope(env, node),
        "PackPositiveBits" => pack_positive_bits(env, node),
        "AddTransposePad" => add_transpose_pad(env, node),
        "MulAddRelu" => mul_add_relu(env, node),
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

/// ONNX `Elu`: positive values pass through, negative values use
/// `alpha * (exp(x) - 1)`. Qwen3-TTS exports the default alpha, but preserving
/// the attribute keeps the implementation schema-correct.
fn elu(env: &mut Env, node: &NodeIr) -> Result<()> {
    let alpha = node
        .attrs
        .get("alpha")
        .and_then(AttrValue::as_f32)
        .unwrap_or(1.0);
    unary(
        env,
        node,
        "Elu",
        &format!("select({alpha:?} * (exp(v) - 1.0), v, v >= 0.0)"),
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

/// `Identity`: a copy with the same shape. Exporters and `onnxscript`'s
/// optimizer leave it where a no-op `Slice` or `Reshape` used to be.
fn identity(env: &mut Env, node: &NodeIr) -> Result<()> {
    let shape = env.shape_of(&node.inputs[0])?;
    meta_reshape_out(env, node, shape)
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
/// Host → new `HostTensor` (data cloned); device → the input's own buffer
/// when this node is its last reader and the runtime owns it, otherwise a GPU
/// copy with the new shape (stays in VRAM, no CPU round-trip). Either way one
/// buffer has one owner: no aliasing across consumers.
fn meta_reshape_out(env: &mut Env, node: &NodeIr, new_shape: Vec<i64>) -> Result<()> {
    let src = &node.inputs[0];
    // A large f32 constant is copied from its VRAM-resident upload: on host it
    // would be cloned every run and carried down the chain to the first GPU op
    // (SAM 3's 5.3 MB positional encoding through Tile, Unsqueeze, Reshape and
    // a host Transpose).
    if env.is_initializer(src)
        && !env.host_resident_small(src, Env::SMALL_HOST_BYTES)
        && env.dtype_of(src)? == FLOAT
    {
        env.ensure_device(src)?;
    }
    if let Some(moved) = env.take_dying_device(src) {
        env.set(
            &node.outputs[0],
            Tensor::Device(DevTensor {
                shape: new_shape,
                ..moved
            }),
        );
    } else if env.on_device(src) {
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
    // GPU kernels for 4-byte dtypes (f32/i32) and 1-byte ones (a quantized
    // activation feeding a device MatMulInteger, as in SAM 3's decoder).
    if env.on_device(src) && matches!(elem_size(env.dtype_of(src)?), 1 | 4) {
        transpose_device(env, node, &in_shape, &perm, out_shape)
    } else {
        let h = env.host(src)?;
        let out = host_transpose(&h, &perm, &out_shape)?;
        env.set(&node.outputs[0], Tensor::Host(out));
        Ok(())
    }
}

/// The same permutation on fewer axes: axes of size 1 are dropped, and input
/// axes that follow each other in the output too are merged into one. The
/// bytes move to the same places. Returns the input dims and the permutation.
fn collapse_permutation(shape: &[i64], perm: &[usize]) -> (Vec<i64>, Vec<usize>) {
    let kept: Vec<usize> = (0..shape.len()).filter(|&axis| shape[axis] != 1).collect();
    // runs of input axes, in output order: (first, last)
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &p in perm.iter().filter(|&&p| shape[p] != 1) {
        let p = kept.iter().position(|&axis| axis == p).expect("kept axis");
        match runs.last_mut() {
            Some(run) if run.1 + 1 == p => run.1 = p,
            _ => runs.push((p, p)),
        }
    }
    let mut order: Vec<usize> = (0..runs.len()).collect();
    order.sort_by_key(|&run| runs[run].0);
    let dims = order
        .iter()
        .map(|&run| {
            kept[runs[run].0..=runs[run].1]
                .iter()
                .map(|&a| shape[a])
                .product()
        })
        .collect();
    let perm = (0..runs.len())
        .map(|run| order.iter().position(|&r| r == run).expect("run"))
        .collect();
    (dims, perm)
}

/// Push constants and workgroups of `TRANSPOSE_TILED` for 4-byte elements on
/// collapsed axes, or `None` when the permutation does not fit it: a kept
/// innermost axis of more than 4 elements, more than 4 batch axes, or a grid
/// beyond 65535 workgroups along an axis. A kept innermost axis becomes the
/// unit the tile moves whole.
fn tiled_transpose(in_shape: &[i64], perm: &[usize]) -> Option<(Vec<u8>, [u32; 3])> {
    let mut dims: Vec<u64> = in_shape.iter().map(|&d| d.max(0) as u64).collect();
    let mut perm = perm.to_vec();
    let mut unit = 1;
    if perm.last() == Some(&(dims.len() - 1)) {
        unit = dims.pop()?;
        perm.pop();
    }
    let rank = dims.len();
    if !(1..=4).contains(&unit) || rank < 2 {
        return None;
    }
    let (a, b) = (rank - 1, perm[rank - 1]);
    if a == b {
        return None;
    }
    // strides in words: input row-major, output by output position
    let mut in_strides = vec![unit; rank];
    for d in (0..rank - 1).rev() {
        in_strides[d] = in_strides[d + 1] * dims[d + 1];
    }
    let mut out_strides = vec![unit; rank];
    for k in (0..rank - 1).rev() {
        out_strides[perm[k]] = out_strides[perm[k + 1]] * dims[perm[k + 1]];
    }
    let batch: Vec<usize> = perm
        .iter()
        .rev()
        .copied()
        .filter(|&x| x != a && x != b)
        .collect();
    let groups = [
        dims[a].div_ceil(32),
        dims[b].div_ceil(32),
        batch.iter().map(|&x| dims[x]).product(),
    ];
    if batch.len() > 4 || groups.iter().any(|&g| g > 65535) {
        return None;
    }
    let pick = |values: &[u64], fill: u64| -> Vec<u64> {
        (0..4)
            .map(|i| batch.get(i).map_or(fill, |&x| values[x]))
            .collect()
    };
    let header = [
        dims[a],
        dims[b],
        in_strides[b],
        out_strides[a],
        unit,
        batch.len() as u64,
        0,
        0,
    ];
    let push = header
        .into_iter()
        .chain(pick(&dims, 1))
        .chain(pick(&in_strides, 0))
        .chain(pick(&out_strides, 0))
        .flat_map(|value| (value as u32).to_le_bytes())
        .collect();
    Some((push, groups.map(|g| g as u32)))
}

/// GPU permutation of an activation, on its collapsed axes
/// ([`collapse_permutation`]): a plain copy if nothing moves,
/// `TRANSPOSE_ROWS` if the innermost axis stays innermost on rows of whole
/// 16-byte words, `TRANSPOSE_TILED` for the other 4-byte permutations it fits
/// ([`tiled_transpose`]), otherwise `WGSL_TRANSPOSE` for 4-byte dtypes and
/// `TRANSPOSE_BYTES` for 1-byte ones.
fn transpose_device(
    env: &mut Env,
    node: &NodeIr,
    in_shape: &[i64],
    perm: &[usize],
    out_shape: Vec<i64>,
) -> Result<()> {
    let ctx = env.context();
    let (in_shape, perm) = collapse_permutation(in_shape, perm);
    let rank = in_shape.len();
    ensure!(rank <= 8, "Transpose: rank {rank} > 8 not supported");
    // row-major input strides, then reordered by perm (one per output dim)
    let mut in_strides = vec![0u32; rank];
    let mut acc = 1u32;
    for d in (0..rank).rev() {
        in_strides[d] = acc;
        acc *= in_shape[d].max(0) as u32;
    }
    let mut perm_strides: Vec<u32> = perm.iter().map(|&p| in_strides[p]).collect();
    let mut out_dims: Vec<u32> = perm.iter().map(|&p| in_shape[p].max(0) as u32).collect();
    let n: usize = out_shape.iter().product::<i64>().max(0) as usize;

    let d = env.device(&node.inputs[0])?;
    let size = elem_size(d.dtype);
    let bytes = size == 1;
    ensure!(
        bytes || size == 4,
        "Transpose device: elem size {size} not 1 or 4"
    );
    let dtype = d.dtype;
    let out = ctx.create_storage_buffer(device_storage_bytes(dtype, n)?)?;
    let row_bytes = in_shape
        .last()
        .map_or(0, |&inner| inner.max(0) as usize * size);
    let rows = rank > 1 && perm[rank - 1] == rank - 1 && row_bytes % 16 == 0;
    let tiled = if rank > 1 && !rows && !bytes {
        tiled_transpose(&in_shape, &perm)
    } else {
        None
    };
    let (key, source, threads, header) = if rows {
        // the outer strides are multiples of a row, so of a word too
        let words = (n * size / 16) as u32;
        out_dims.pop();
        perm_strides.pop();
        for stride in &mut perm_strides {
            *stride = *stride * size as u32 / 16;
        }
        let header = [words, rank as u32 - 1, (row_bytes / 16) as u32];
        ("Transpose_rows", TRANSPOSE_ROWS, words as usize, header)
    } else if bytes {
        let header = [n as u32, rank as u32, 0];
        ("Transpose_bytes", TRANSPOSE_BYTES, n.div_ceil(4), header)
    } else {
        ("Transpose", WGSL_TRANSPOSE, n, [n as u32, rank as u32, 0])
    };
    if n > 0 && rank <= 1 {
        // nothing moves, e.g. a permutation of axes of size 1
        ctx.stream_copy(d.buffer(), &out, (n * size) as u64)?;
    } else if let (true, Some((push, groups))) = (n > 0, &tiled) {
        with_pipeline(
            env.cache(),
            "Transpose_tiled",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(TRANSPOSE_TILED)?,
                    TRANSPOSE_BINDINGS,
                    TRANSPOSE_PUSH_BYTES,
                )
            },
            |pipe| ctx.stream_dispatch(pipe, &[d.buffer(), &out], push, *groups),
        )?;
    } else if n > 0 {
        let mut push = Vec::with_capacity(80);
        for value in header {
            push.extend_from_slice(&value.to_le_bytes());
        }
        push.extend_from_slice(&0u32.to_le_bytes());
        for d in 0..8 {
            push.extend_from_slice(&out_dims.get(d).copied().unwrap_or(1).to_le_bytes());
        }
        for d in 0..8 {
            push.extend_from_slice(&perm_strides.get(d).copied().unwrap_or(0).to_le_bytes());
        }
        with_pipeline(
            env.cache(),
            key,
            || {
                ctx.create_pipeline(
                    &compile_wgsl(source)?,
                    TRANSPOSE_BINDINGS,
                    TRANSPOSE_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[d.buffer(), &out],
                    &push,
                    [(threads as u32).div_ceil(256), 1, 1],
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

/// Dtype-generic host-side permutation (per-element bytes). Also the load-time
/// fold of a constant `Transpose` (`rewrite::evaluate`).
pub(crate) fn host_transpose(
    h: &HostTensor,
    perm: &[usize],
    out_shape: &[i64],
) -> Result<HostTensor> {
    let rank = h.shape.len();
    let es = elem_size(h.dtype);
    ensure!(
        es > 0,
        "Transpose host: dtype {} with unknown size",
        h.dtype
    );
    ensure!(
        h.data.len()
            == h.shape
                .iter()
                .map(|&d| d.max(0) as usize)
                .product::<usize>()
                * es,
        "Transpose host: {} bytes for shape {:?}",
        h.data.len(),
        h.shape
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
/// all on-device, see `quantize_on_device`.
fn dynamic_quantize(env: &mut Env, node: &NodeIr) -> Result<()> {
    env.ensure_device(&node.inputs[0])?;
    let x = env.device(&node.inputs[0])?;
    let (shape, n) = (x.shape.clone(), x.elem_count);
    if n == 0 {
        // The range of no values is [0, 0], as for all zeros: ORT answers
        // scale 1, zero point 0 (SAM 3's decoder quantizes "no boxes").
        let outs = &node.outputs;
        env.set(
            &outs[0],
            Tensor::Host(HostTensor::new(UINT8, shape, vec![])),
        );
        if outs.len() > 1 && !outs[1].is_empty() {
            env.set(&outs[1], Tensor::Host(HostTensor::from_f32(vec![], &[1.0])));
        }
        if outs.len() > 2 && !outs[2].is_empty() {
            env.set(
                &outs[2],
                Tensor::Host(HostTensor::new(UINT8, vec![], vec![0])),
            );
        }
        return Ok(());
    }

    quantize_on_device(env, &node.inputs[0], &node.outputs, None)
}

/// The device passes of `DynamicQuantizeLinear` over the device tensor `x`,
/// which holds at least one value: min/max into a zeroed range slot, then
/// quantization. When the kernel that wrote `x` already folded its range into
/// a slot (`ExecutionEnv::set_range`), the first pass is skipped. With `gelu`, the push constants of
/// `gelu_push`, the passes read `DecomposedGelu(x)` instead of `x`.
fn quantize_on_device(env: &mut Env, x: &str, outs: &[String], gelu: Option<&[u8]>) -> Result<()> {
    use crate::shaders::dynamic_quantize as dq;
    let computed = env.computed_range(x).filter(|_| gelu.is_none());
    let (range_index, range_offset) = match computed {
        Some(slot) => slot,
        None => env.range_slot()?,
    };
    let range = BufferSlice {
        buf: env.range_buffer(range_index),
        offset: range_offset,
    };
    let ctx = env.context();
    let x = env.device(x)?;
    let (shape, n) = (x.shape.clone(), x.elem_count);
    let y = ctx.create_storage_buffer(device_storage_bytes(UINT8, n)?)?;
    let y_scale = ctx.create_storage_buffer(4)?;
    let y_zp = ctx.create_storage_buffer(4)?;
    let groups = (n as u32).div_ceil(256).min(1024);
    let tail = gelu.unwrap_or_default();
    let (partial_key, quantize_key) = match gelu {
        Some(_) => ("DecomposedGeluQuantize_partial", "DecomposedGeluQuantize"),
        None => ("DQL_partial", "DQL_quantize"),
    };

    if computed.is_none() {
        with_pipeline(
            env.cache(),
            partial_key,
            || {
                let src = match gelu {
                    Some(_) => dq::gelu_partial(),
                    None => dq::PARTIAL.to_owned(),
                };
                ctx.create_pipeline(&compile_wgsl(&src)?, 2, 4 + tail.len() as u32)
            },
            |pipe| {
                ctx.stream_dispatch_slices(
                    pipe,
                    &[x.buffer().into(), range],
                    &[&(n as u32).to_le_bytes()[..], tail].concat(),
                    [groups, 1, 1],
                )
            },
        )?;
    }
    let words = n.div_ceil(4) as u32;
    with_pipeline(
        env.cache(),
        quantize_key,
        || {
            let src = match gelu {
                Some(_) => dq::gelu_quantize(),
                None => dq::QUANTIZE.to_owned(),
            };
            ctx.create_pipeline(&compile_wgsl(&src)?, 5, 8 + tail.len() as u32)
        },
        |pipe| {
            ctx.stream_dispatch_slices(
                pipe,
                &[
                    x.buffer().into(),
                    range,
                    (&y_scale).into(),
                    (&y_zp).into(),
                    (&y).into(),
                ],
                // the second word is the shaders' opaque `zero`
                &[&[n as u32, 0].map(u32::to_le_bytes).concat()[..], tail].concat(),
                [words.div_ceil(256), 1, 1],
            )
        },
    )?;
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
    // The kernel reads rows as 4-byte words. A K that does not fill them is a
    // coordinate projection (SAM 3's point and box encodings: K = 2): on the
    // device when A already is, else a few host rows computed on host.
    let k = *env.shape_of(&node.inputs[0])?.last().unwrap_or(&0);
    if k % 4 != 0 {
        if env.on_device(&node.inputs[0]) && matmul_integer_any_k(env, node, None)? {
            return Ok(());
        }
        let input = |i: usize| env.host(&node.inputs[i]);
        let out = host_ops::matmul_integer(&input(0)?, &input(1)?, &input(2)?, &input(3)?)?;
        env.set(&node.outputs[0], Tensor::Host(out));
        return Ok(());
    }
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

/// `MatMulInteger` with a K the packed kernels cannot read, on the device
/// (see `shaders::matmul_integer::ANY_K`). `false` when the node is outside
/// what the shader covers: zero points other than one per tensor, a K that
/// could overflow `i32`, or a grid past the dispatch limits.
///
/// With `epilogue` = `(scale_step, bias_step)` the node is a
/// `MatMulIntegerScaleBias` and the output is the f32 `float(acc) · scale +
/// bias` of its inputs 4 and 5 (see `glsl/matmul_integer_any_k.comp`).
fn matmul_integer_any_k(
    env: &mut Env,
    node: &NodeIr,
    epilogue: Option<(u32, u32)>,
) -> Result<bool> {
    let inputs = &node.inputs;
    if inputs.len() < 4 || inputs[2].is_empty() || inputs[3].is_empty() {
        return Ok(false);
    }
    let a_shape = env.shape_of(&inputs[0])?;
    let b_shape = env.shape_of(&inputs[1])?;
    let scalar = |shape: Vec<i64>| shape.iter().product::<i64>() == 1;
    if a_shape.len() < 2
        || b_shape.len() != 2
        || b_shape[0] != *a_shape.last().unwrap_or(&0)
        || !scalar(env.shape_of(&inputs[2])?)
        || !scalar(env.shape_of(&inputs[3])?)
    {
        return Ok(false);
    }
    let k = b_shape[0] as usize;
    let n = b_shape[1] as usize;
    let m = a_shape[..a_shape.len() - 1].iter().product::<i64>().max(0) as usize;
    // |(a - za)(b - zb)| <= 255², so K terms fit i32 up to K = 33025
    let max_rows = 65535 * 16;
    if k > 33025 || m > max_rows || n > max_rows || m * k.max(n) > u32::MAX as usize {
        return Ok(false);
    }
    for input in &inputs[..4] {
        env.ensure_device_dtype(input)?;
    }
    if epilogue.is_some() {
        env.ensure_device(&inputs[4])?;
        env.ensure_device(&inputs[5])?;
    }
    let signed = |name: &str| -> Result<u32> { Ok(u32::from(env.dtype_of(name)? == INT8)) };
    let mut push = Vec::with_capacity(MMI_ANY_K_EPILOGUE_PUSH_BYTES as usize);
    for value in [
        m as u32,
        k as u32,
        n as u32,
        signed(&inputs[0])?,
        signed(&inputs[1])?,
    ] {
        push.extend_from_slice(&value.to_le_bytes());
    }
    let (dtype, key, bindings) = match epilogue {
        Some((scale_step, bias_step)) => {
            push.extend_from_slice(&scale_step.to_le_bytes());
            push.extend_from_slice(&bias_step.to_le_bytes());
            push.extend_from_slice(&u32::from(fused_relu(node)).to_le_bytes());
            (FLOAT, MMI_ANY_K_EPILOGUE_KEY, MMI_ANY_K_EPILOGUE_BINDINGS)
        }
        None => (INT32, MMI_ANY_K_KEY, MMI_ANY_K_BINDINGS),
    };
    let ctx = env.context();
    let elem_count = m * n;
    let out = ctx.create_storage_buffer(device_storage_bytes(dtype, elem_count)?)?;
    if elem_count > 0 {
        let mut buffers = vec![
            env.device(&inputs[0])?.buffer(),
            env.device(&inputs[1])?.buffer(),
            env.device(&inputs[2])?.buffer(),
            env.device(&inputs[3])?.buffer(),
            &out,
        ];
        if epilogue.is_some() {
            buffers.push(env.device(&inputs[4])?.buffer());
            buffers.push(env.device(&inputs[5])?.buffer());
        }
        with_pipeline(
            env.cache(),
            key,
            || match epilogue {
                Some(_) => ctx.create_pipeline(
                    &mmi_any_k_epilogue_spirv(),
                    bindings,
                    MMI_ANY_K_EPILOGUE_PUSH_BYTES,
                ),
                None => {
                    ctx.create_pipeline(&compile_wgsl(MMI_ANY_K)?, bindings, MMI_ANY_K_PUSH_BYTES)
                }
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &buffers,
                    &push,
                    [(n as u32).div_ceil(16), (m as u32).div_ceil(16), 1],
                )
            },
        )?;
    }
    let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
    out_shape.push(n as i64);
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype,
            shape: out_shape,
            elem_count,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(true)
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

    if let Some(v) = mmi_coop_variant(ctx, m, n)
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
            |pipe| ctx.stream_dispatch(pipe, buffers, &push, v.grid(m, n)),
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

/// `MatMulInteger → Cast(FLOAT) → Mul(scale) → Add(bias)`, fused by
/// `rewrite::fuse_matmul_integer_epilogue`: the cooperative kernel writes
/// `float(acc) · scale + bias` in f32 and the i32 product is never stored.
///
/// The epilogue rounds exactly as the three kernels it replaces (see the
/// shader), so the output is bit-identical to the unfused graph. It exists only
/// in the cooperative kernel, and it reads scale and bias as one value or one
/// per column; anything else — no cooperative matrix, signed A, a scale that
/// broadcasts over rows — runs the four original steps, so the rewrite never
/// changes what a graph computes or whether it runs.
///
/// Products the cooperative tiles cannot cover, fewer than 16 rows or
/// columns, run the same epilogue on a kernel with one output per thread
/// (`glsl/matmul_integer_small.comp`).
///
/// With a seventh input the scale is the product of inputs 4 and 6, which the
/// kernel forms as the `Mul` it replaces did; the paths that cannot run that
/// `Mul` first (see `with_scale_product`).
fn matmul_integer_scale_bias(env: &mut Env, node: &NodeIr) -> Result<()> {
    let a_shape = env.shape_of(&node.inputs[0])?;
    let b_shape = env.shape_of(&node.inputs[1])?;
    let k = *a_shape.last().unwrap_or(&0) as usize;
    let m: usize = a_shape[..a_shape.len().saturating_sub(1)]
        .iter()
        .product::<i64>()
        .max(0) as usize;
    let n = *b_shape.last().unwrap_or(&0) as usize;
    let ctx = env.context();
    let variant = mmi_coop_variant(ctx, m, n).filter(|v| mmi_coop_applies(v, m, k, n, false));
    let small = variant.is_none() && mmi_small_applies(m, k, n);
    let (scale, bias) = (&node.inputs[4], &node.inputs[5]);
    let rank = a_shape.len();
    let fits = a_shape.len() >= 2
        && b_shape.len() == 2
        && b_shape[0] as usize == k
        && k.is_multiple_of(4)
        && env.dtype_of(&node.inputs[0])? == UINT8
        && env.dtype_of(scale)? == FLOAT
        && env.dtype_of(bias)? == FLOAT;
    let steps = (
        column_step(&env.shape_of(scale)?, n, rank),
        column_step(&env.shape_of(bias)?, n, rank),
    );
    let product = node.inputs.get(6);
    if let Some(p) = product {
        let folds = (variant.is_some() || small)
            && fits
            && steps.0.is_some()
            && steps.1.is_some()
            && env.dtype_of(p)? == FLOAT
            && column_step(&env.shape_of(p)?, n, rank).is_some();
        if !folds {
            return with_scale_product(env, node, matmul_integer_scale_bias);
        }
    }
    let product_step = match product {
        Some(p) => column_step(&env.shape_of(p)?, n, rank).unwrap_or(0),
        None => 0,
    };
    // A K the packed kernels cannot read goes where the unfused MatMulInteger
    // would go, the any-K kernel, with the epilogue folded in
    if !k.is_multiple_of(4)
        && env.on_device(&node.inputs[0])
        && env.dtype_of(scale)? == FLOAT
        && env.dtype_of(bias)? == FLOAT
        && let (Some(scale_step), Some(bias_step)) = steps
        && matmul_integer_any_k(env, node, Some((scale_step, bias_step)))?
    {
        return Ok(());
    }
    let (true, true, (Some(scale_step), Some(bias_step))) =
        (variant.is_some() || small, fits, steps)
    else {
        return integer_scale_bias_unfused(env, node, "MatMulInteger");
    };

    env.ensure_device_dtype(&node.inputs[0])?;
    env.ensure_device_dtype(&node.inputs[2])?;
    env.ensure_device_dtype(&node.inputs[3])?;
    env.ensure_device(scale)?;
    env.ensure_device(bias)?;
    if let Some(p) = product {
        env.ensure_device(p)?;
    }
    let b_signed = env.dtype_of(&node.inputs[1])? == INT8;
    let range_slot = if fused_range(node) {
        Some(env.range_slot()?)
    } else {
        None
    };
    let packed_ptr = packed_b(env, &node.inputs[1], k, n, b_signed)?;
    // SAFETY: the packed weight lives in the session cache, which never
    // removes an entry (see `packed_b`).
    let packed: &GpuBuffer = unsafe { &*packed_ptr };
    let (a, a_zp, b_zp) = (
        env.device(&node.inputs[0])?,
        env.device(&node.inputs[2])?,
        env.device(&node.inputs[3])?,
    );
    let (scale, bias) = (env.device(scale)?, env.device(bias)?);
    let product = product.map(|p| env.device(p)).transpose()?;
    let elem_count = m * n;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, elem_count)?)?;
    let mut push = Vec::with_capacity(MMI_COOP_EPILOGUE_PUSH_BYTES as usize);
    let b_zp_xor = if b_signed { MMI_SIGN_FLIP_BYTE } else { 0 };
    let flags = u32::from(fused_relu(node))
        | u32::from(range_slot.is_some()) << 1
        | u32::from(product.is_some()) << 2;
    for value in [
        m as u32,
        k as u32,
        n as u32,
        0,
        b_zp_xor,
        scale_step,
        bias_step,
        flags,
        product_step,
    ] {
        push.extend_from_slice(&value.to_le_bytes());
    }
    let (key, grid) = match variant {
        Some(v) => (v.epilogue_key, v.grid(m, n)),
        // `small` checked the grid
        None => (
            MMI_SMALL_EPILOGUE_KEY,
            mmi_small_grid(m, n).unwrap_or([1; 3]),
        ),
    };
    with_pipeline(
        env.cache(),
        key,
        || {
            ctx.create_pipeline(
                &variant.map_or_else(mmi_small_epilogue_spirv, |v| v.spirv_epilogue()),
                MMI_COOP_EPILOGUE_BINDINGS,
                MMI_COOP_EPILOGUE_PUSH_BYTES,
            )
        },
        |pipe| {
            // unread without the range flag: any buffer fills the binding
            let range = match range_slot {
                Some((index, offset)) => BufferSlice {
                    buf: env.range_buffer(index),
                    offset,
                },
                None => (&out).into(),
            };
            // likewise the product without its flag
            let product = product.map_or(scale.buffer(), |p| p.buffer());
            ctx.stream_dispatch_slices(
                pipe,
                &[
                    a.buffer().into(),
                    packed.into(),
                    a_zp.buffer().into(),
                    b_zp.buffer().into(),
                    (&out).into(),
                    scale.buffer().into(),
                    bias.buffer().into(),
                    range,
                    product.into(),
                ],
                &push,
                grid,
            )
        },
    )?;
    if let Some(slot) = range_slot {
        env.set_range(&node.outputs[0], slot);
    }
    let mut out_shape = a_shape[..rank - 1].to_vec();
    out_shape.push(n as i64);
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

/// A fused integer node whose scale is the product of inputs 4 and 6, run
/// where the kernel cannot form that product: the `Mul` the rewrite removed,
/// then `run` on the six-input node reading its result.
fn with_scale_product(
    env: &mut Env,
    node: &NodeIr,
    run: fn(&mut Env, &NodeIr) -> Result<()>,
) -> Result<()> {
    let scale = format!("{}/scale", node.outputs[0]);
    let i = &node.inputs;
    let product = NodeIr {
        attrs: HashMap::new(),
        ..lowered(node, "Mul", &[&i[4], &i[6]], &[&scale])
    };
    exec_dispatch(env, &product)?;
    let mut fused = node.clone();
    fused.inputs.truncate(6);
    fused.inputs[4] = scale.clone();
    run(env, &fused)?;
    env.release(&scale);
    Ok(())
}

/// The epilogue stride of an operand broadcast against an `[.., n]` output of
/// rank `rank`: 0 for one value, 1 for one value per column, `None` when it
/// broadcasts over rows too or would raise the output's rank.
fn column_step(shape: &[i64], n: usize, rank: usize) -> Option<u32> {
    let leading_ones = shape.iter().rev().skip(1).all(|&d| d == 1);
    if shape.len() > rank || !leading_ones {
        return None;
    }
    match shape.last() {
        None | Some(1) => Some(0),
        Some(&d) if d == n as i64 => Some(1),
        _ => None,
    }
}

/// Whether `rewrite::fuse_matmul_integer_epilogue` folded the `Relu` that
/// followed into this `MatMulIntegerScaleBias`.
fn fused_relu(node: &NodeIr) -> bool {
    node.attrs.get("relu").and_then(AttrValue::as_i64) == Some(1)
}

/// Whether a fused `MatMulInteger` should fold the range of its result for the
/// `DynamicQuantizeLinear` that reads it (`range = 1`, see
/// `rewrite::fuse_matmul_integer_epilogue`).
fn fused_range(node: &NodeIr) -> bool {
    node.attrs.get("range").and_then(AttrValue::as_i64) == Some(1)
}

/// The nodes `rewrite::fuse_matmul_integer_epilogue` or
/// `rewrite::fuse_conv_integer_epilogue` replaced, run as they were: `op`, the
/// integer node, keeps the fused node's attributes but `relu` and `range`, the
/// others get none; a fused `Relu` runs last. No range is recorded: the
/// quantization that follows makes its own pass.
fn integer_scale_bias_unfused(env: &mut Env, node: &NodeIr, op: &str) -> Result<()> {
    let acc = format!("{}/acc", node.outputs[0]);
    let dequantized = format!("{}/dequantized", node.outputs[0]);
    let scaled = format!("{}/scaled", node.outputs[0]);
    let biased = format!("{}/biased", node.outputs[0]);
    let relu = fused_relu(node);
    let i = &node.inputs;
    let mut integer = lowered(
        node,
        op,
        &[&i[0], &i[1], &i[2], &i[3]].map(String::as_str),
        &[&acc],
    );
    integer.attrs.remove("relu");
    integer.attrs.remove("range");
    exec_dispatch(env, &integer)?;
    let plain = |op: &str, inputs: &[&str], output: &str| NodeIr {
        attrs: HashMap::new(),
        ..lowered(node, op, inputs, &[output])
    };
    let mut cast = plain("Cast", &[&acc], &dequantized);
    cast.attrs
        .insert("to".to_owned(), AttrValue::Int(i64::from(FLOAT)));
    exec_dispatch(env, &cast)?;
    env.release(&acc);
    exec_dispatch(env, &plain("Mul", &[&dequantized, &i[4]], &scaled))?;
    env.release(&dequantized);
    let sum = if relu { &biased } else { &node.outputs[0] };
    exec_dispatch(env, &plain("Add", &[&scaled, &i[5]], sum))?;
    env.release(&scaled);
    if relu {
        exec_dispatch(env, &plain("Relu", &[&biased], &node.outputs[0]))?;
        env.release(&biased);
    }
    Ok(())
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
    let mut idx_own = None;
    let idx_buf = param_buffer(env, &idx_bytes, &mut idx_own)?;

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
                    &[d.buffer(), idx_buf, &out],
                    &push,
                    [(n as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    // only if this call owns one: a shared constant belongs to the cache
    if let Some(buffer) = idx_own {
        ctx.defer_destroy(buffer);
    }
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

/// `GatherBlockQuantized` (com.microsoft): `Gather` on a 4-bit block-quantized
/// table, dequantizing on the way out.
///
/// The table's own shape is the packed one — `[rows, cols/2]` u8 — so the
/// logical column count comes from doubling its last dimension, exactly as
/// `MatMulNBits` takes `K` from an attribute because the packed shape cannot
/// state it. Indices are read on the host (they are a shape-sized control
/// tensor) and re-uploaded as `i32`, like `gather`.
fn gather_block_quantized(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    let block_size = node
        .attrs
        .get("block_size")
        .and_then(AttrValue::as_i64)
        .unwrap_or(0) as usize;

    let data_shape = env.shape_of(&node.inputs[0])?;
    ensure!(
        data_shape.len() == 2,
        "GatherBlockQuantized: expected a 2-D table, got {data_shape:?}"
    );
    let rows = data_shape[0].max(0) as usize;
    let row_bytes = data_shape[1].max(0) as usize;
    let cols = row_bytes * 2;
    ensure!(
        cols > 0 && cols.is_multiple_of(block_size),
        "GatherBlockQuantized: {cols} columns is not a multiple of block_size {block_size}"
    );
    let n_blocks = cols / block_size;

    let idx_host = env.host(&node.inputs[1])?;
    let idx_shape = idx_host.shape.clone();
    let idx_i32: Vec<i32> = idx_host
        .to_i64()?
        .into_iter()
        .map(|mut g| {
            if g < 0 {
                g += rows as i64;
            }
            g as i32
        })
        .collect();
    ensure!(
        idx_i32.iter().all(|&g| g >= 0 && (g as usize) < rows),
        "GatherBlockQuantized: an index falls outside the {rows}-row table"
    );
    let idx_count = idx_i32.len();

    // the table stays packed in VRAM: dequantizing it here would cost 8× the
    // memory of the model's largest initializer to read one row per token
    for name in [&node.inputs[0], &node.inputs[2], &node.inputs[3]] {
        env.ensure_device_dtype(name)?;
    }
    let quant = env.device(&node.inputs[0])?;
    let scales = env.device(&node.inputs[2])?;
    let zero_points = env.device(&node.inputs[3])?;
    ensure!(
        quant.dtype == UINT8 && zero_points.dtype == UINT8 && scales.dtype == FLOAT,
        "GatherBlockQuantized: expected a u8 table and zero point with f32 scales, got {}/{}/{}",
        quant.dtype,
        zero_points.dtype,
        scales.dtype
    );
    let zp_row_bytes = n_blocks.div_ceil(2);
    ensure!(
        scales.elem_count >= rows * n_blocks && zero_points.elem_count >= rows * zp_row_bytes,
        "GatherBlockQuantized: scales or zero points are short for {rows}×{n_blocks} blocks"
    );

    let mut out_shape = idx_shape;
    out_shape.push(cols as i64);
    let count = idx_count * cols;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, count)?)?;

    let idx_bytes: Vec<u8> = idx_i32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut idx_buf_own = None;
    let idx_buf = param_buffer(env, &idx_bytes, &mut idx_buf_own)?;

    if count > 0 {
        let groups = (count as u32).div_ceil(256);
        let gx = groups.min(32768);
        let mut push = Vec::with_capacity(GBQ_PUSH_BYTES as usize);
        for v in [
            count as u32,
            cols as u32,
            row_bytes as u32,
            n_blocks as u32,
            block_size as u32,
            zp_row_bytes as u32,
            gx,
            0,
        ] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        with_pipeline(
            env.cache(),
            "GatherBlockQuantized",
            || ctx.create_pipeline(&compile_wgsl(GBQ_GATHER)?, GBQ_BINDINGS, GBQ_PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[
                        quant.buffer(),
                        idx_buf,
                        scales.buffer(),
                        zero_points.buffer(),
                        &out,
                    ],
                    &push,
                    [gx, groups.div_ceil(gx), 1],
                )
            },
        )?;
    }
    // only if this call owns one: a shared constant belongs to the cache
    if let Some(buffer) = idx_buf_own {
        ctx.defer_destroy(buffer);
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: out_shape,
            elem_count: count,
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

/// `IsNaN`: on the GPU when the input already lives there, so a guard after
/// an attention `Softmax` (`Where(IsNaN(p), 0, p)`, 13 nodes of SAM 3's
/// detector, up to [1, 8, 5184, 32]) no longer reads the probabilities back.
fn is_nan(env: &mut Env, node: &NodeIr) -> Result<()> {
    let x = &node.inputs[0];
    if !env.on_device(x) || env.dtype_of(x)? != FLOAT {
        return host_unary(env, node, host_ops::is_nan);
    }
    let ctx = env.context();
    let xd = env.device(x)?;
    let (shape, n) = (xd.shape.clone(), xd.elem_count);
    let words = n.div_ceil(4) as u32;
    let out = ctx.create_storage_buffer(device_storage_bytes(BOOL, n)?)?;
    if n > 0 {
        with_pipeline(
            env.cache(),
            "IsNaN",
            || ctx.create_pipeline(&compile_wgsl(IS_NAN)?, IS_NAN_BINDINGS, IS_NAN_PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[xd.buffer(), &out],
                    &(n as u32).to_le_bytes(),
                    [words.div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: BOOL,
            shape,
            elem_count: n,
            buf: BufRef::Owned(out),
        }),
    );
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

/// `OneHot`: on host, like the comparisons. In the SAM 3 mask decoder it
/// expands a handful of constant indices.
fn one_hot(env: &mut Env, node: &NodeIr) -> Result<()> {
    let indices = env.host(&node.inputs[0])?;
    let depth = env.host(&node.inputs[1])?;
    let values = env.host(&node.inputs[2])?;
    let axis = node
        .attrs
        .get("axis")
        .and_then(AttrValue::as_i64)
        .unwrap_or(-1);
    let out = host_ops::one_hot(&indices, &depth, &values, axis)?;
    env.set(&node.outputs[0], Tensor::Host(out));
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

/// `GatherND` emitted by the codec uses the standard `batch_dims = 0` form.
fn gather_nd(env: &mut Env, node: &NodeIr) -> Result<()> {
    let batch_dims = node
        .attrs
        .get("batch_dims")
        .and_then(AttrValue::as_i64)
        .unwrap_or(0);
    ensure!(
        batch_dims == 0,
        "GatherND: batch_dims={batch_dims} is not supported"
    );
    let data = env.host(&node.inputs[0])?;
    let indices = env.host(&node.inputs[1])?;
    let out = host_ops::gather_nd(&data, &indices)?;
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
    let src = &node.inputs[0];
    // An f32 activation, or a small vector broadcast to one (SAM 3 spreads a
    // [1, 256, 1, 1] embedding over 72×72), is copied on the GPU: on host it
    // costs a readback and a re-upload. Shape math stays on host.
    if env.dtype_of(src)? == FLOAT {
        let bc = broadcast(&env.shape_of(src)?, &target)?;
        let out_bytes = bc.out_shape.iter().product::<i64>().max(0) as usize * 4;
        if env.on_device(src) || out_bytes > Env::SMALL_HOST_BYTES {
            env.ensure_device(src)?;
            // `b` only lends the target shape: the expression reads `a` alone
            return broadcast_dispatch(env, &node.outputs[0], src, src, bc, "a[off_a]");
        }
    }
    let x = env.host(src)?;
    env.set(
        &node.outputs[0],
        Tensor::Host(host_ops::expand(&x, &target)?),
    );
    Ok(())
}

/// `Tile`: replicates input[0] by repeats (input[1], host int).
fn tile(env: &mut Env, node: &NodeIr) -> Result<()> {
    let repeats = env.host(&node.inputs[1])?.to_i64()?;
    if repeats.iter().all(|&r| r == 1) {
        // a copy, kept where it lives (SAM 3 tiles its image embeddings by the
        // number of prompts, 1): on host it would round-trip megabytes
        let shape = env.shape_of(&node.inputs[0])?;
        return meta_reshape_out(env, node, shape);
    }
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

    // no padding, as SAM 3's window partition when the grid divides: a copy
    if pads.iter().all(|&p| p == 0) && env.on_device(data) && env.dtype_of(data)? == FLOAT {
        let shape = env.shape_of(data)?;
        return meta_reshape_out(env, node, shape);
    }
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
    let mut params_buf_own = None;
    let params_buf = param_buffer(env, &params_bytes, &mut params_buf_own)?;

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
                    &[d.buffer(), params_buf, &out],
                    &push,
                    [(n as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    // only if this call owns one: a shared constant belongs to the cache
    if let Some(buffer) = params_buf_own {
        ctx.defer_destroy(buffer);
    }
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
        let pixels = (g.h_out * g.w_out) as u32;
        let kdepth = (g.gsi * g.kh * g.kw) as usize;
        let committed =
            conv_tuned_tactic(g.group as usize, pixels as usize, g.c_out as usize, kdepth);
        let tensor_signature = |name: &str| -> Result<crate::tuning::TensorSignature> {
            let dimensions = env
                .shape_of(name)?
                .into_iter()
                .map(|dimension| {
                    u64::try_from(dimension).map_err(|_| {
                        anyhow::anyhow!(
                            "Conv tuning signature has negative dimension {dimension} for '{name}'"
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let dtype = crate::ElementType::try_from(env.dtype_of(name)?)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            Ok(crate::tuning::TensorSignature { dtype, dimensions })
        };
        let inputs = node
            .inputs
            .iter()
            .filter(|name| !name.is_empty())
            .map(|name| tensor_signature(name))
            .collect::<Result<Vec<_>>>()?;
        let output_dimensions = g
            .out_shape
            .iter()
            .copied()
            .map(|dimension| {
                u64::try_from(dimension).map_err(|_| {
                    anyhow::anyhow!("Conv tuning output has negative dimension {dimension}")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let tuning_key = crate::tuning::TuningKey {
            device: env.context().device_fingerprint().into(),
            workload: crate::tuning::WorkloadSignature {
                domain: node.domain.clone(),
                op: node.op.clone(),
                inputs,
                outputs: vec![crate::tuning::TensorSignature {
                    dtype: crate::ElementType::Float32,
                    dimensions: output_dimensions,
                }],
                attributes_digest: crate::tuning::attributes_digest(node),
            },
            implementation: conv_fingerprint(),
        };
        let geometry = ConvTacticGeometry {
            batch: g.n as u32,
            group: g.group as u32,
            pixels,
            c_out: g.c_out as u32,
            kdepth: kdepth as u32,
            total: g.total as u64,
        };
        let limits = ConvTacticLimits::from_vulkan(
            env.context().compute_limits(),
            env.context().compute_limits().max_storage_buffer_bytes,
        );
        let selected = env
            .cache()
            .tuning()
            .resolve(&tuning_key, |record| {
                ConvTactic::from_persistent(record.tactic(), record.parameters()).is_some_and(
                    |candidate| {
                        candidate.is_viable(geometry, limits)
                            && conv_generated_source(candidate).is_some()
                    },
                )
            })
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let tactic = selected
            .as_ref()
            .and_then(|record| ConvTactic::from_persistent(record.tactic(), record.parameters()))
            .unwrap_or(committed);
        let split = tactic.split();
        push.extend_from_slice(&split.to_le_bytes());
        let bias: &GpuBuffer = unsafe { &*bias };
        // Every tactic runs the source generated from all of its parameters, the
        // one the tuner measured: a family-wide committed source with this
        // tactic's dispatch grid leaves outputs unwritten when the sizes differ.
        let operation = match tactic {
            ConvTactic::Direct { .. } => "Conv_grouped",
            ConvTactic::ImplicitGemm { .. } => "Conv16",
            ConvTactic::Blocked {
                output_tile: 32, ..
            } => "Conv32",
            ConvTactic::Blocked {
                output_tile: 128, ..
            } => "Conv128",
            ConvTactic::Blocked { .. } => "Conv",
            ConvTactic::SplitK { .. } => "Conv_split",
        };
        let source = conv_generated_source(tactic)
            .ok_or_else(|| anyhow::anyhow!("unsupported generated Conv tactic {tactic:?}"))?;
        let key = PipelineKey::tactic(operation, format!("{tactic:?}"));
        let groups = tactic.dispatch_grid(g.total as u32, pixels, g.c_out as u32, g.n as u32);
        // With a split the kernel writes one partial image per slice and the
        // reduction pass below folds them onto `out`, bias included.
        let partials = (split > 1)
            .then(|| {
                ctx.create_storage_buffer(device_storage_bytes(FLOAT, split as usize * g.total)?)
            })
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
    ensure!(conv_integer_with(env, node, false)?);
    Ok(())
}

/// `ConvInteger → Cast(FLOAT) → Mul(scale) → Add(bias)`, fused by
/// `rewrite::fuse_conv_integer_epilogue`: the cooperative kernel writes
/// `float(acc) · scale + bias` in f32 and the i32 convolution is never stored.
///
/// The same epilogue as `MatMulIntegerScaleBias`, bit-identical to the unfused
/// graph, with scale and bias one value or one per output channel — a row of
/// the `[C_out, P]` product. Anything the cooperative paths of `ConvInteger`
/// do not take, or another broadcast, runs the four original steps.
///
/// A seventh input is a second factor of the scale, as in
/// `matmul_integer_scale_bias`.
fn conv_integer_scale_bias(env: &mut Env, node: &NodeIr) -> Result<()> {
    if conv_integer_with(env, node, true)? {
        return Ok(());
    }
    if node.inputs.len() == 7 {
        return with_scale_product(env, node, conv_integer_scale_bias);
    }
    integer_scale_bias_unfused(env, node, "ConvInteger")
}

/// The epilogue strides of a `ConvIntegerScaleBias` (see
/// `glsl/matmul_integer_coop.comp`): scale, bias and the scale's second
/// factor, 0 when there is none. `None` when one of them is not f32 or
/// broadcasts other than per tensor or per channel. Makes them resident.
fn conv_epilogue(
    env: &mut Env,
    node: &NodeIr,
    c_out: i64,
    rank: usize,
) -> Result<Option<[u32; 3]>> {
    let operands = [
        Some(&node.inputs[4]),
        Some(&node.inputs[5]),
        node.inputs.get(6),
    ];
    let mut steps = [0; 3];
    for (step, name) in steps.iter_mut().zip(operands) {
        let Some(name) = name else { continue };
        if env.dtype_of(name)? != FLOAT {
            return Ok(None);
        }
        let Some(s) = channel_step(&env.shape_of(name)?, c_out, rank) else {
            return Ok(None);
        };
        *step = s;
    }
    for name in operands.into_iter().flatten() {
        env.ensure_device(name)?;
    }
    Ok(Some(steps))
}

/// The epilogue stride of an operand broadcast against an `[N, C_out, ..]`
/// output of rank `rank`: 0 for one value, 2 (one per row of the product) for
/// one per channel, `None` otherwise or when it would raise the output's rank.
fn channel_step(shape: &[i64], c_out: i64, rank: usize) -> Option<u32> {
    let ones = |dims: &[i64]| dims.iter().all(|&d| d == 1);
    if shape.len() > rank {
        return None;
    }
    if ones(shape) {
        return Some(0);
    }
    // the channel axis, counted from the right as broadcasting aligns
    let spatial = rank.checked_sub(2)?;
    let channel = shape.len().checked_sub(spatial + 1)?;
    (shape[channel] == c_out && ones(&shape[..channel]) && ones(&shape[channel + 1..])).then_some(2)
}

/// `ConvInteger`, or with `fused` the `ConvIntegerScaleBias` whose inputs 4
/// and 5 are the scale and bias of its epilogue. `false` only when `fused`
/// and neither cooperative path takes the node: nothing has run then.
fn conv_integer_with(env: &mut Env, node: &NodeIr, fused: bool) -> Result<bool> {
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
    // Second fast path: state the convolution as an explicit GEMM and put it on
    // the tensor cores. It costs a materialized im2col matrix — the write the
    // implicit kernels below exist to avoid — and pays for it only because the
    // cooperative kernel is that much faster: measured over resnet50-int8's own
    // geometries (`examples/conv_integer_coop`) im2col + cooperative matrix is
    // **1.47×** the implicit ladder, while im2col + the WGSL matmul is 0.79×.
    // So the path is taken only when the cooperative variant actually applies,
    // never as a fallback.
    let pixels = (h_out * w_out) as usize;
    let kdepth = (g.gsi * kh * kw) as usize;
    let coop = if group == 1 && n == 1 && total > 0 {
        mmi_coop_variant(ctx, c_out as usize, pixels)
            .filter(|v| mmi_coop_applies(v, c_out as usize, kdepth, pixels, false))
    } else {
        None
    };
    // the epilogue exists only in the cooperative kernel, which the pointwise
    // path reaches with the same A, B and K as the general one
    let epilogue = match (fused, coop) {
        (false, _) => None,
        (true, None) => return Ok(false),
        (true, Some(variant)) => match conv_epilogue(env, node, c_out, out_shape.len())? {
            Some(steps) => Some((variant, steps)),
            None => return Ok(false),
        },
    };

    env.ensure_device_dtype(&x_name)?;
    env.ensure_device_dtype(&w_name)?;
    // ONNX allows both uint8 and int8 for X and W, independently
    let x_signed = u32::from(env.dtype_of(&x_name)? == INT8);
    let w_signed = u32::from(env.dtype_of(&w_name)? == INT8);
    ensure_zp(env, node.inputs.get(2))?; // zero-point of the X activation
    ensure_zp(env, node.inputs.get(3))?; // zero-point of the W weights

    if pointwise && total > 0 {
        conv_pointwise_matmul(
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
            epilogue,
        )?;
        return Ok(true);
    }
    if let Some(variant) = coop {
        conv_im2col_coop(
            env,
            node,
            &x_name,
            &w_name,
            &g,
            out_shape,
            pixels,
            kdepth,
            x_signed,
            w_signed,
            variant,
            epilogue.map(|(_, steps)| steps),
        )?;
        return Ok(true);
    }

    let x = env.device(&x_name)?;
    let w = env.device(&w_name)?;
    let out = ctx.create_storage_buffer(device_storage_bytes(INT32, total)?)?;
    if total > 0 {
        let mut push = g.push_common();
        push.extend_from_slice(&x_signed.to_le_bytes());
        push.extend_from_slice(&w_signed.to_le_bytes());
        // Same ladder as the f32 `Conv`, and the same two predicates: measured
        // on this model's own geometries they land within 1.9% of the best
        // configuration per shape, so nothing here is calibrated twice.
        let gemm = group == 1;
        let pixels = (h_out * w_out) as u32;
        let kdepth = (g.gsi * kh * kw) as usize;
        let split = if gemm {
            conv_split_k(pixels as usize, c_out as usize, kdepth)
        } else {
            None
        };
        push.extend_from_slice(&split.unwrap_or(1).to_le_bytes());
        let blocked = gemm && conv_prefer_blocked(pixels as usize, c_out as usize);
        let (key, source, tile) = match (gemm, split.is_some(), blocked) {
            (true, true, _) => (
                "ConvInteger_split",
                conv_i_blocked_splitk_source(),
                CONV_I_BLOCKED_TILE_SIZE,
            ),
            (true, false, true) => (
                "ConvInteger",
                conv_i_blocked_source(),
                CONV_I_BLOCKED_TILE_SIZE,
            ),
            (true, false, false) => ("ConvInteger16", conv_i_gemm_source(), CONV_I_TILE_SIZE),
            _ => ("ConvInteger_grouped", conv_i_direct_source(), 0),
        };
        let groups = if gemm {
            [
                pixels.div_ceil(tile),
                (c_out as u32).div_ceil(tile),
                (n as u32).max(1) * split.unwrap_or(1),
            ]
        } else {
            [(total as u32).div_ceil(256), 1, 1]
        };
        // With a split the kernel writes one partial image per slice and the
        // reduction below folds them onto `out`.
        let partials = split
            .map(|s| ctx.create_storage_buffer(device_storage_bytes(INT32, s as usize * total)?))
            .transpose()?;
        let azp = zp_ref(env, node.inputs.get(2))?;
        let wzp = zp_ref(env, node.inputs.get(3))?;
        with_pipeline(
            env.cache(),
            key,
            || {
                ctx.create_pipeline(
                    &compile_wgsl(&source)?,
                    CONV_INTEGER_BINDINGS,
                    CONV_INTEGER_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[
                        x.buffer(),
                        w.buffer(),
                        azp,
                        wzp,
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
                "ConvInteger_split_reduce",
                || {
                    ctx.create_pipeline(
                        &compile_wgsl(CONV_I_SPLIT_REDUCE)?,
                        CONV_I_SPLIT_REDUCE_BINDINGS,
                        CONV_INTEGER_PUSH_BYTES,
                    )
                },
                |pipe| {
                    ctx.stream_dispatch(
                        pipe,
                        &[&partials, &out],
                        &push,
                        [(total as u32).div_ceil(256), 1, 1],
                    )
                },
            )?;
            ctx.defer_destroy(partials);
        }
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
    Ok(true)
}

/// A node that exists only for the duration of one lowering.
///
/// The QOperator ops are an integer kernel plus an epilogue, and the integer
/// kernel is one this interpreter already has. Rather than duplicate
/// `conv_integer` — pointwise fast path, groups, padding and all — the
/// quantized op builds the `ConvInteger` node it *is* and runs it into a
/// private value. Nothing of this reaches the graph: the names are derived
/// from the node's own, so two nodes cannot collide, and the value is released
/// as soon as the epilogue has read it.
fn lowered(node: &NodeIr, op: &str, inputs: &[&str], outputs: &[&str]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: node.opset,
        name: format!("{}__{op}", node.name),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: node.attrs.clone(),
    }
}

/// The host-side half of the epilogue: `ratio[c] = x_scale · w_scale[c] / y_scale`.
///
/// One f32 per output channel, computed once from constants the graph carries
/// and uploaded through the content-keyed parameter cache, so identical
/// channel counts across nodes do not each get their own buffer. Doing the
/// division here rather than in the shader is not an optimization of the inner
/// loop — the epilogue is outside it — but of the *constant*: `y_scale` is one
/// number and dividing by it per element would be work with no result.
fn requantize_ratio(env: &Env, x_scale: &str, w_scale: &str, y_scale: &str) -> Result<Vec<f32>> {
    let x = env.host(x_scale)?.to_f32()?;
    let w = env.host(w_scale)?.to_f32()?;
    let y = env.host(y_scale)?.to_f32()?;
    ensure!(
        x.len() == 1 && y.len() == 1,
        "QLinear: activation scales must be per-tensor (x {}, y {})",
        x.len(),
        y.len()
    );
    ensure!(y[0] != 0.0, "QLinear: y_scale is zero");
    Ok(w.iter().map(|w| x[0] * w / y[0]).collect())
}

/// A zero point as a number, from the constant the graph carries.
fn zp_value(env: &Env, name: &str) -> Result<i32> {
    let values = env.host(name)?.to_i64()?;
    ensure!(
        values.len() == 1,
        "QLinear: zero point of {name} is not per-tensor ({} values)",
        values.len()
    );
    Ok(values[0] as i32)
}

/// The epilogue: int32 accumulator → quantized output.
///
/// `inner` and `axis_len` place the channel: `H·W` and `C_out` for a
/// convolution, `1` and `N` for a matmul. `bias` is the `int32` bias of
/// `QLinearConv`, already in the accumulator's domain — it is added before the
/// scaling, not after, which is what makes it int32 in the first place.
#[allow(clippy::too_many_arguments)]
fn requantize(
    env: &mut Env,
    node: &NodeIr,
    acc_name: &str,
    bias: Option<&str>,
    ratio: &[f32],
    inner: usize,
    y_zp_name: &str,
    out_dtype: i32,
) -> Result<()> {
    let ctx = env.context();
    let shape = env.shape_of(acc_name)?;
    let n: usize = shape.iter().product::<i64>().max(0) as usize;
    let y_zp = zp_value(env, y_zp_name)?;
    let axis_len = ratio.len();
    ensure!(axis_len > 0, "{}: empty scale", node.op);

    if let Some(bias) = bias {
        env.ensure_device_dtype(bias)?;
    }
    let out = ctx.create_storage_buffer(device_storage_bytes(out_dtype, n)?)?;
    if n > 0 {
        let fields = [
            n as u32,
            inner.max(1) as u32,
            axis_len as u32,
            u32::from(out_dtype == INT8),
            u32::from(bias.is_some()),
            y_zp as u32,
        ];
        let mut push = Vec::with_capacity(REQUANTIZE_PUSH_BYTES as usize);
        for value in fields {
            push.extend_from_slice(&value.to_le_bytes());
        }
        let ratio_bytes: Vec<u8> = ratio.iter().flat_map(|r| r.to_le_bytes()).collect();
        let mut scratch = None;
        let ratio_buffer = param_buffer(env, &ratio_bytes, &mut scratch)?;
        let acc = env.device(acc_name)?;
        // no bias ⇒ the shader never reads the binding, but a descriptor set
        // still needs a buffer there: the accumulator itself serves
        let bias_buffer = match bias {
            Some(name) => env.device(name)?.buffer(),
            None => acc.buffer(),
        };
        with_pipeline(
            env.cache(),
            "Requantize",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(REQUANTIZE)?,
                    REQUANTIZE_BINDINGS,
                    REQUANTIZE_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[acc.buffer(), bias_buffer, ratio_buffer, &out],
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
            shape,
            elem_count: n,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `QLinearConv`: `ConvInteger` on the same operands, then the epilogue.
///
/// Splitting it this way is not a shortcut around a fused kernel — it is the
/// operator's own definition. The integer part is bit-for-bit the convolution
/// `ConvInteger` computes, and the epilogue is outside its loop over `K`, so
/// fusing the two would save one pass over the output and change no number.
fn qlinear_conv(env: &mut Env, node: &NodeIr) -> Result<()> {
    let acc = format!("{}__acc", node.name);
    let ratio = requantize_ratio(env, &node.inputs[1], &node.inputs[4], &node.inputs[6])?;
    let out_dtype = env.dtype_of(&node.inputs[7])?;
    let conv = lowered(
        node,
        "ConvInteger",
        &[
            &node.inputs[0],
            &node.inputs[3],
            &node.inputs[2],
            &node.inputs[5],
        ],
        &[&acc],
    );
    conv_integer(env, &conv)?;

    let shape = env.shape_of(&acc)?;
    let inner: usize = shape[2..].iter().product::<i64>().max(1) as usize;
    // the epilogue indexes `bias` with the same channel as `ratio`: a
    // per-tensor weight scale must be broadcast to `C_out`, or every channel
    // would read `bias[0]`
    let channels = shape.get(1).copied().unwrap_or(1).max(1) as usize;
    let ratio = if ratio.len() == 1 {
        vec![ratio[0]; channels]
    } else {
        ratio
    };
    let bias = node.inputs.get(8).filter(|n| !n.is_empty()).cloned();
    requantize(
        env,
        node,
        &acc,
        bias.as_deref(),
        &ratio,
        inner,
        &node.inputs[7],
        out_dtype,
    )?;
    env.release(&acc);
    Ok(())
}

/// `QLinearMatMul`: `MatMulInteger`, then the epilogue on the columns.
///
/// The channel is the column, so `inner = 1` and `axis_len = N` — and `N` is
/// where the two models in the matrix disagree: resnet50 passes one scale for
/// the whole tensor, mobilenetv2 one per column. Both are the same shader with
/// a different `axis_len`.
fn qlinear_matmul(env: &mut Env, node: &NodeIr) -> Result<()> {
    let acc = format!("{}__acc", node.name);
    let mut ratio = requantize_ratio(env, &node.inputs[1], &node.inputs[4], &node.inputs[6])?;
    let out_dtype = env.dtype_of(&node.inputs[7])?;
    let matmul = lowered(
        node,
        "MatMulInteger",
        &[
            &node.inputs[0],
            &node.inputs[3],
            &node.inputs[2],
            &node.inputs[5],
        ],
        &[&acc],
    );
    matmul_integer(env, &matmul)?;

    let shape = env.shape_of(&acc)?;
    let columns = *shape.last().unwrap_or(&1) as usize;
    ensure!(
        ratio.len() == 1 || ratio.len() == columns,
        "QLinearMatMul: {} scales for {columns} columns",
        ratio.len()
    );
    if ratio.len() == 1 {
        ratio = vec![ratio[0]; 1];
    }
    requantize(env, node, &acc, None, &ratio, 1, &node.inputs[7], out_dtype)?;
    env.release(&acc);
    Ok(())
}

/// `com.microsoft::QLinearAdd`: one pass, on the output's scale.
fn qlinear_add(env: &mut Env, node: &NodeIr) -> Result<()> {
    let (a_name, b_name) = (node.inputs[0].clone(), node.inputs[3].clone());
    let scale = |name: &str| -> Result<f32> {
        let values = env.host(name)?.to_f32()?;
        ensure!(values.len() == 1, "QLinearAdd: {name} is not per-tensor");
        Ok(values[0])
    };
    let c_scale = scale(&node.inputs[6])?;
    ensure!(c_scale != 0.0, "QLinearAdd: C_scale is zero");
    let mut ra = scale(&node.inputs[1])? / c_scale;
    let mut rb = scale(&node.inputs[4])? / c_scale;
    let (mut a_zp, mut b_zp) = (
        zp_value(env, &node.inputs[2])?,
        zp_value(env, &node.inputs[5])?,
    );
    let c_zp = zp_value(env, &node.inputs[7])?;
    let out_dtype = env.dtype_of(&node.inputs[7])?;

    env.ensure_device_dtype(&a_name)?;
    env.ensure_device_dtype(&b_name)?;
    let ctx = env.context();
    let mut a = env.device(&a_name)?;
    let mut b = env.device(&b_name)?;
    let mut bc = broadcast(&a.shape, &b.shape)?;
    ensure!(
        bc.out_shape.len() <= MAX_RANK,
        "QLinearAdd: rank {} > {MAX_RANK}",
        bc.out_shape.len()
    );
    // ORT hands MLAS the operands exchanged when A is constant along the
    // innermost broadcast span (or the output is one element), and the kernel
    // is not symmetric in rounding, so neither is this one.
    let inner = bc.out_shape.iter().rposition(|&d| d > 1);
    if inner.is_none_or(|d| bc.a_strides[d] == 0) {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut bc.a_strides, &mut bc.b_strides);
        std::mem::swap(&mut ra, &mut rb);
        std::mem::swap(&mut a_zp, &mut b_zp);
    }
    // MLAS folds the zero points into one addend, with this contraction.
    let fixed = c_zp as f32 - ra.mul_add(a_zp as f32, rb * b_zp as f32);
    let elem_count: usize = bc.out_shape.iter().product::<i64>().max(0) as usize;
    let out = ctx.create_storage_buffer(device_storage_bytes(out_dtype, elem_count)?)?;
    if elem_count > 0 {
        let mut push = Vec::with_capacity(QLINEAR_ADD_PUSH_BYTES as usize);
        push.extend_from_slice(&(elem_count as u32).to_le_bytes());
        push.extend_from_slice(&(bc.out_shape.len() as u32).to_le_bytes());
        push.extend_from_slice(&0u32.to_le_bytes());
        push.extend_from_slice(&0u32.to_le_bytes());
        push_vec4s(&mut push, &bc.out_strides);
        push_vec4s(&mut push, &bc.a_strides);
        push_vec4s(&mut push, &bc.b_strides);
        for value in [ra, rb, fixed] {
            push.extend_from_slice(&value.to_le_bytes());
        }
        push.extend_from_slice(&u32::from(out_dtype == INT8).to_le_bytes());
        with_pipeline(
            env.cache(),
            "QLinearAdd",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(QLINEAR_ADD)?,
                    QLINEAR_ADD_BINDINGS,
                    QLINEAR_ADD_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[a.buffer(), b.buffer(), &out],
                    &push,
                    [(elem_count as u32).div_ceil(4).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: out_dtype,
            shape: bc.out_shape,
            elem_count,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `com.microsoft::QLinearGlobalAveragePool`: dequantize, pool, quantize.
///
/// The one node of the family that is *not* given its own kernel, and
/// deliberately: both models have exactly one, on a tensor the convolutions
/// have already reduced to 7×7, and the three steps are three kernels this
/// interpreter already has. Writing a fused reduction here would add a shader
/// to save microseconds on two nodes in the whole suite — and a fused one
/// would have to write single bytes from separate workgroups, which is the
/// awkward part, not the reduction.
fn qlinear_global_average_pool(env: &mut Env, node: &NodeIr) -> Result<()> {
    let real = format!("{}__f32", node.name);
    let pooled = format!("{}__pooled", node.name);
    let dequantize = lowered(
        node,
        "DequantizeLinear",
        &[&node.inputs[0], &node.inputs[1], &node.inputs[2]],
        &[&real],
    );
    dequantize_linear(env, &dequantize)?;
    let pooled_node = lowered(node, "GlobalAveragePool", &[&real], &[&pooled]);
    pool(env, &pooled_node, PoolKind::GlobalAverage)?;
    env.release(&real);
    let quantize = lowered(
        node,
        "QuantizeLinear",
        &[&pooled, &node.inputs[3], &node.inputs[4]],
        &[&node.outputs[0]],
    );
    quantize_linear(env, &quantize)?;
    env.release(&pooled);
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
        // `zero` of QuantizeLinear's `div_rn`, opaque to the driver
        0,
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
    epilogue: Option<(&'static CoopVariant, [u32; 3])>,
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
    let zp_xor = [w_signed, x_signed].map(|s| if s != 0 { MMI_SIGN_FLIP_BYTE } else { 0 });
    let (out, dtype) = if let Some((variant, steps)) = epilogue {
        let out = conv_coop_epilogue(
            env,
            node,
            variant,
            w,
            &packed,
            [c_out, c_in, nmat],
            zp_xor,
            steps,
        )?;
        (out, FLOAT)
    } else {
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
                a_zp_xor: zp_xor[0],
                b_zp_xor: zp_xor[1],
            },
        )?;
        (out, INT32)
    };
    ctx.defer_destroy(packed);
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype,
            shape: out_shape,
            elem_count: total,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `ConvInteger` as an explicit GEMM on the cooperative matrix unit.
///
/// The kernels in `shaders::conv_integer` rebuild every im2col column from its
/// index and never materialize the matrix. `coopMatLoad` takes a base offset
/// and a row stride, so a cooperative matrix cannot be fed that way: the
/// operand has to exist. This path writes it and multiplies it, and it is
/// entered **only** when the cooperative variant applies, because the
/// materialization loses against the implicit ladder on any other kernel —
/// 1.47× with the tensor cores, 0.79× with the WGSL matmul, measured over this
/// model's own geometries in `examples/conv_integer_coop`.
///
/// The mapping needs no transpose and no pack. `M = C_out`, `N = P`, and:
///
/// - A is the ONNX weight `[C_out, C_in·KH·KW]`, already the `[M, K]` the
///   kernel wants — flipped once per session if it is `int8`, since the
///   hardware combination is `u8 × u8` and a cooperative matrix has no bitwise
///   operations.
/// - B is the im2col matrix `[P, K]`, which is the `[N, K]` layout `PACK_B`
///   produces for the pointwise path.
/// - out is `[C_out, P]`, which is NCHW at `n == 1`. That restriction is why
///   `n == 1` gates this path, exactly as it gates the pointwise one.
///
/// A signed activation costs nothing extra here: the im2col pass already
/// touches every byte, so it flips them on the way out.
#[allow(clippy::too_many_arguments)]
fn conv_im2col_coop(
    env: &mut Env,
    node: &NodeIr,
    x_name: &str,
    w_name: &str,
    g: &ConvGeom,
    out_shape: Vec<i64>,
    pixels: usize,
    kdepth: usize,
    x_signed: u32,
    w_signed: u32,
    variant: &'static CoopVariant,
    epilogue: Option<[u32; 3]>,
) -> Result<()> {
    let c_out = g.c_out as usize;
    let total = c_out * pixels;
    let words = pixels * kdepth / 4;

    // Channels and planes that start on a word go through HWC (see
    // `shaders::conv_integer::TO_HWC`), with K reordered to (kh, kw, c)
    let hwc = g.gsi % 4 == 0 && (g.h_in * g.w_in) % 4 == 0;
    let ctx = env.context();
    // Back to the pool when the product is recorded, like a tensor: destroyed,
    // the next run allocates it again — 8 ms for the 191 MB of SAM 3's
    // 3×3 convolution at 288², more than its GPU time.
    let col = ctx.create_storage_buffer((words * 4) as u64)?;
    if hwc {
        conv_im2col_hwc(env, node, x_name, g, &col, words, x_signed)?;
    } else {
        let x = env.device(x_name)?;
        let azp = zp_ref(env, node.inputs.get(2))?;
        let mut push = Vec::with_capacity(CONV_I_IM2COL_PUSH_BYTES as usize);
        for value in [
            words as u32,
            kdepth as u32,
            (g.kh * g.kw) as u32,
            g.h_in as u32,
            g.w_in as u32,
            g.w_out as u32,
            g.kw as u32,
            g.sh as u32,
            g.sw as u32,
            g.phb as u32,
            g.pwb as u32,
            g.dh as u32,
            g.dw as u32,
            if x_signed != 0 { MMI_SIGN_FLIP_WORD } else { 0 },
        ] {
            push.extend_from_slice(&value.to_le_bytes());
        }
        with_pipeline(
            env.cache(),
            "ConvInteger_im2col",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(CONV_I_IM2COL)?,
                    CONV_I_IM2COL_BINDINGS,
                    CONV_I_IM2COL_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[x.buffer(), azp, &col],
                    &push,
                    [(words as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }

    let w_ptr = if hwc {
        hwc_weight(env, w_name, g, w_signed != 0)?
    } else if w_signed != 0 {
        flipped_const(env, w_name, c_out * kdepth)?
    } else {
        env.device(w_name)?.buffer() as *const GpuBuffer
    };
    let ctx = env.context();
    // SAFETY: the flipped weight lives in the cache, which never removes
    // entries and keeps them boxed; an unflipped one is the tensor's own
    // buffer, alive for the whole execution.
    let w: &GpuBuffer = unsafe { &*w_ptr };
    let zp_xor = [w_signed, x_signed].map(|s| if s != 0 { MMI_SIGN_FLIP_BYTE } else { 0 });
    if let Some(steps) = epilogue {
        let out = conv_coop_epilogue(
            env,
            node,
            variant,
            w,
            &col,
            [c_out, kdepth, pixels],
            zp_xor,
            steps,
        )?;
        ctx.recycle_storage_buffer(col);
        env.set(
            &node.outputs[0],
            Tensor::Device(DevTensor {
                dtype: FLOAT,
                shape: out_shape,
                elem_count: total,
                buf: BufRef::Owned(out),
            }),
        );
        return Ok(());
    }
    let out = ctx.create_storage_buffer(device_storage_bytes(INT32, total)?)?;
    let azp = zp_ref(env, node.inputs.get(3))?; // A = W → weight zero point
    let bzp = zp_ref(env, node.inputs.get(2))?; // B = im2col → activation zero point

    // Dispatched here rather than through `mmi_dispatch` for one reason: the
    // Pareto attributes by pipeline key, and folding these into
    // `MMI_matmul_coop_k32` would merge the convolutions with the pointwise
    // matmuls in the one profile that decides what to optimize next.
    let mut push = Vec::with_capacity(MMI_COOP_PUSH_BYTES as usize);
    for value in [
        c_out as u32,
        kdepth as u32,
        pixels as u32,
        if w_signed != 0 { MMI_SIGN_FLIP_BYTE } else { 0 },
        if x_signed != 0 { MMI_SIGN_FLIP_BYTE } else { 0 },
    ] {
        push.extend_from_slice(&value.to_le_bytes());
    }
    with_pipeline(
        env.cache(),
        PipelineKey::tactic("ConvInteger_coop", variant.key),
        || ctx.create_pipeline(&variant.spirv(), MMI_COOP_BINDINGS, MMI_COOP_PUSH_BYTES),
        |pipe| {
            ctx.stream_dispatch(
                pipe,
                &[w, &col, azp, bzp, &out],
                &push,
                variant.grid(c_out, pixels),
            )
        },
    )?;
    ctx.recycle_storage_buffer(col);
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

/// The cooperative product `W [C_out, K] × B [P, K]ᵀ` of a
/// `ConvIntegerScaleBias` with the dequantizing epilogue: the f32
/// `float(acc) · scale + bias` of its inputs 4 and 5 (the scale times input
/// 6 when there is one), NCHW at `n == 1`, with `steps` from `conv_epilogue`. `zp_xor` is the zero-point shift of W and of
/// X, as in `conv_im2col_coop`.
#[allow(clippy::too_many_arguments)]
fn conv_coop_epilogue(
    env: &Env,
    node: &NodeIr,
    variant: &'static CoopVariant,
    w: &GpuBuffer,
    b: &GpuBuffer,
    [m, k, n]: [usize; 3],
    zp_xor: [u32; 2],
    steps: [u32; 3],
) -> Result<GpuBuffer> {
    let ctx = env.context();
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, m * n)?)?;
    let azp = zp_ref(env, node.inputs.get(3))?; // A = W → weight zero point
    let bzp = zp_ref(env, node.inputs.get(2))?; // B = X → activation zero point
    let (scale, bias) = (env.device(&node.inputs[4])?, env.device(&node.inputs[5])?);
    // unread without its flag: any buffer fills the binding
    let product = match node.inputs.get(6) {
        Some(name) => env.device(name)?.buffer(),
        None => scale.buffer(),
    };
    let flags = u32::from(node.inputs.len() == 7) << 2;
    let mut push = Vec::with_capacity(MMI_COOP_EPILOGUE_PUSH_BYTES as usize);
    for value in [
        m as u32, k as u32, n as u32, zp_xor[0], zp_xor[1], steps[0], steps[1], flags, steps[2],
    ] {
        push.extend_from_slice(&value.to_le_bytes());
    }
    with_pipeline(
        env.cache(),
        PipelineKey::tactic("ConvInteger_coop_epilogue", variant.epilogue_key),
        || {
            ctx.create_pipeline(
                &variant.spirv_epilogue(),
                MMI_COOP_EPILOGUE_BINDINGS,
                MMI_COOP_EPILOGUE_PUSH_BYTES,
            )
        },
        |pipe| {
            ctx.stream_dispatch(
                pipe,
                // the range binding is unread: a convolution never folds it
                &[
                    w,
                    b,
                    azp,
                    bzp,
                    &out,
                    scale.buffer(),
                    bias.buffer(),
                    &out,
                    product,
                ],
                &push,
                variant.grid(m, n),
            )
        },
    )?;
    Ok(out)
}

/// The im2col matrix of `conv_im2col_coop` with K ordered `(kh, kw, c)`:
/// X to HWC, then whole words of it to `col` (see
/// `shaders::conv_integer::IM2COL_HWC`). Needs `C_in` and `H·W` multiples of
/// four.
fn conv_im2col_hwc(
    env: &mut Env,
    node: &NodeIr,
    x_name: &str,
    g: &ConvGeom,
    col: &GpuBuffer,
    words: usize,
    x_signed: u32,
) -> Result<()> {
    let ctx = env.context();
    let channels = g.gsi as u32;
    let plane = (g.h_in * g.w_in) as u32;
    let xt = ctx.create_storage_buffer(u64::from(channels) * u64::from(plane))?;
    let x = env.device(x_name)?;
    let mut push = Vec::with_capacity(CONV_I_TO_HWC_PUSH_BYTES as usize);
    for value in [channels, plane] {
        push.extend_from_slice(&value.to_le_bytes());
    }
    with_pipeline(
        env.cache(),
        CONV_I_TO_HWC_KEY,
        || {
            ctx.create_pipeline(
                &compile_wgsl(CONV_I_TO_HWC)?,
                CONV_I_TO_HWC_BINDINGS,
                CONV_I_TO_HWC_PUSH_BYTES,
            )
        },
        |pipe| {
            ctx.stream_dispatch(
                pipe,
                &[x.buffer(), &xt],
                &push,
                [
                    plane.div_ceil(CONV_I_TO_HWC_TILE),
                    channels.div_ceil(CONV_I_TO_HWC_TILE),
                    1,
                ],
            )
        },
    )?;
    let azp = zp_ref(env, node.inputs.get(2))?;
    let channel_words = channels / 4;
    let mut push = Vec::with_capacity(CONV_I_IM2COL_HWC_PUSH_BYTES as usize);
    for value in [
        words as u32,
        channel_words * (g.kh * g.kw) as u32,
        channel_words,
        g.h_in as u32,
        g.w_in as u32,
        g.w_out as u32,
        g.kw as u32,
        g.sh as u32,
        g.sw as u32,
        g.phb as u32,
        g.pwb as u32,
        g.dh as u32,
        g.dw as u32,
        if x_signed != 0 { MMI_SIGN_FLIP_WORD } else { 0 },
    ] {
        push.extend_from_slice(&value.to_le_bytes());
    }
    let max_groups = ctx.compute_limits().max_workgroup_count[0];
    with_pipeline(
        env.cache(),
        CONV_I_IM2COL_HWC_KEY,
        || {
            ctx.create_pipeline(
                &compile_wgsl(CONV_I_IM2COL_HWC)?,
                CONV_I_IM2COL_HWC_BINDINGS,
                CONV_I_IM2COL_HWC_PUSH_BYTES,
            )
        },
        |pipe| {
            ctx.stream_dispatch(
                pipe,
                &[&xt, azp, col],
                &push,
                [(words as u32).div_ceil(256).min(max_groups), 1, 1],
            )
        },
    )?;
    ctx.defer_destroy(xt);
    Ok(())
}

/// The constant weight `[C_out, C_in, KH, KW]` permuted to
/// `[C_out, KH, KW, C_in]` for `conv_im2col_hwc`, sign-flipped when it is
/// `int8`: done once on the host, cached for the session.
fn hwc_weight(env: &mut Env, name: &str, g: &ConvGeom, flip: bool) -> Result<*const GpuBuffer> {
    let (c_out, c_in, taps) = (g.c_out as usize, g.gsi as usize, (g.kh * g.kw) as usize);
    let bytes = c_out * c_in * taps;
    let cache = env.cache();
    let key = (format!("{name}#hwc"), bytes, usize::from(flip));
    if let Some(cached) = cache.packed_weight_cached(&key) {
        return Ok(cached);
    }
    let w = env.host(name)?;
    ensure!(
        w.data.len() == bytes,
        "ConvInteger: weight {name} has {} bytes, expected {bytes}",
        w.data.len()
    );
    let xor = if flip { 0x80 } else { 0 };
    let mut permuted = vec![0u8; bytes.next_multiple_of(4).max(4)];
    for o in 0..c_out {
        for c in 0..c_in {
            for t in 0..taps {
                permuted[(o * taps + t) * c_in + c] = w.data[(o * c_in + c) * taps + t] ^ xor;
            }
        }
    }
    let ctx = env.context();
    cache.packed_weight(key, || {
        let dst = ctx.create_storage_buffer(permuted.len() as u64)?;
        ctx.stream_upload(&dst, &permuted)?;
        Ok(dst)
    })
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
    let mut params_buf_own = None;
    let params_buf = param_buffer(env, &params_bytes, &mut params_buf_own)?;

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
                    &[c.buffer(), xd.buffer(), yd.buffer(), &out, params_buf],
                    &(n as u32).to_le_bytes(),
                    [(n as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    // only if this call owns one: a shared constant belongs to the cache
    if let Some(buffer) = params_buf_own {
        ctx.defer_destroy(buffer);
    }
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

    // the whole input, as SAM 3's window unpartition when the grid divides: a copy
    let whole = out_shape == data_shape && st.iter().all(|&s| s == 0) && sp.iter().all(|&s| s == 1);
    if whole && env.on_device(data) && env.dtype_of(data)? == FLOAT {
        return meta_reshape_out(env, node, data_shape);
    }
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
    let mut params_buf_own = None;
    let params_buf = param_buffer(env, &params_bytes, &mut params_buf_own)?;

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
                    &[d.buffer(), params_buf, &out],
                    &push,
                    [(n as u32).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    // only if this call owns one: a shared constant belongs to the cache
    if let Some(buffer) = params_buf_own {
        ctx.defer_destroy(buffer);
    }
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

/// Resolve only geometries covered by the matrix family runner.
fn matrix_tactic(
    env: &Env,
    node: &NodeIr,
    output: &[i64],
    allowed: &[&str],
) -> Result<Option<String>> {
    let inputs = node
        .inputs
        .iter()
        .filter(|name| !name.is_empty())
        .map(|name| {
            Ok(crate::tuning::TensorSignature {
                dtype: crate::ElementType::try_from(env.dtype_of(name)?)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?,
                dimensions: env
                    .shape_of(name)?
                    .into_iter()
                    .map(u64::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let key = crate::tuning::TuningKey {
        device: env.context().device_fingerprint().into(),
        workload: crate::tuning::WorkloadSignature {
            domain: node.domain.clone(),
            op: node.op.clone(),
            inputs,
            outputs: vec![crate::tuning::TensorSignature {
                dtype: crate::ElementType::Float32,
                dimensions: output
                    .iter()
                    .copied()
                    .map(u64::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            }],
            attributes_digest: crate::tuning::attributes_digest(node),
        },
        implementation: crate::shaders::matrix::implementation_fingerprint(),
    };
    Ok(env
        .cache()
        .tuning()
        .resolve(&key, |record| {
            record.tactic().family() == "matrix-f32"
                && record.parameters().is_empty()
                && allowed.contains(&record.tactic().variant())
        })
        .map_err(|error| anyhow::anyhow!(error.to_string()))?
        .map(|record| record.tactic().variant().to_owned()))
}

/// `MatMul` f32 (opset ≥13) with batch broadcasting. Reuses the tiled 16×16 kernel.
fn matmul_fp32(env: &mut Env, node: &NodeIr) -> Result<()> {
    env.ensure_device(&node.inputs[0])?;
    env.ensure_device(&node.inputs[1])?;
    let a_shape = env.device(&node.inputs[0])?.shape.clone();
    let b_shape = env.device(&node.inputs[1])?.shape.clone();
    let (out, out_shape, elem_count) = matmul_fp32_shaped(env, node, &a_shape, &b_shape)?;
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

/// The MatMul of `node`'s two device inputs read as `a_shape` and `b_shape`,
/// which may differ from their stored shapes by a reshape: `Einsum` runs its
/// matmul forms here without copying operands into reshaped buffers. Returns the
/// output buffer, its shape and its element count.
fn matmul_fp32_shaped(
    env: &mut Env,
    node: &NodeIr,
    a_shape: &[i64],
    b_shape: &[i64],
) -> Result<(GpuBuffer, Vec<i64>, usize)> {
    let ctx = env.context();
    let a = env.device(&node.inputs[0])?;
    let b = env.device(&node.inputs[1])?;
    let numel = |shape: &[i64]| shape.iter().product::<i64>().max(0) as usize;
    ensure!(
        numel(a_shape) == a.elem_count && numel(b_shape) == b.elem_count,
        "MatMul: shapes A {a_shape:?}, B {b_shape:?} do not match {} and {} elements",
        a.elem_count,
        b.elem_count
    );
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
        let selected = if a_shape.len() == 2 && b_shape.len() == 2 && m > 1 {
            matrix_tactic(env, node, &out_shape, &["matmul16", "matmul64"])?
        } else {
            None
        };
        let blocked = selected
            .as_deref()
            .map(|id| id == "matmul64")
            .unwrap_or_else(|| mm_prefer_blocked(m, n));
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
    Ok((out, out_shape, elem_count))
}

/// How many labels of an `Einsum` equation fall in each MatMul role: batch (in
/// both inputs and the output), `m` (first input and output), `k` (both
/// inputs, summed) and `n` (second input and output).
#[derive(Debug, PartialEq, Eq)]
struct EinsumMatmul {
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
}

/// The equations that are a batched MatMul once the operands are reshaped:
/// `A = batch·m·k`, `B = batch·k·n`, `out = batch·m·n`, each group in the same
/// label order everywhere (SAM 3's mask head, `bqc,bchw->bqhw`). Anything that
/// would need a transpose, a diagonal, an ellipsis, an implicit output or a
/// label summed on one side only is `None`.
fn einsum_as_matmul(equation: &str) -> Option<EinsumMatmul> {
    let equation: String = equation.chars().filter(|c| !c.is_whitespace()).collect();
    let (inputs, out) = equation.split_once("->")?;
    let (a, b) = inputs.split_once(',')?;
    let labels = |term: &str| -> Option<Vec<char>> {
        let chars: Vec<char> = term.chars().collect();
        let distinct: HashSet<char> = chars.iter().copied().collect();
        (chars.iter().all(char::is_ascii_alphabetic) && distinct.len() == chars.len())
            .then_some(chars)
    };
    let (a, b, out) = (labels(a)?, labels(b)?, labels(out)?);
    let pick = |from: &[char], in_b: bool, in_out: bool, other: &[char]| -> Vec<char> {
        from.iter()
            .copied()
            .filter(|c| other.contains(c) == in_b && out.contains(c) == in_out)
            .collect()
    };
    let batch = pick(&a, true, true, &b);
    let m = pick(&a, false, true, &b);
    let k = pick(&a, true, false, &b);
    let n = pick(&b, false, true, &a);
    let joined = |parts: [&[char]; 3]| parts.concat();
    (a == joined([&batch, &m, &k])
        && b == joined([&batch, &k, &n])
        && out == joined([&batch, &m, &n]))
    .then_some(EinsumMatmul {
        batch: batch.len(),
        m: m.len(),
        k: k.len(),
        n: n.len(),
    })
}

/// `Einsum` in its MatMul forms ([`einsum_as_matmul`]): the operands are read
/// as `[batch.., M, K]` and `[batch.., K, N]` by the MatMul kernel, and its
/// `[batch.., M, N]` output is already the einsum's layout.
fn einsum(env: &mut Env, node: &NodeIr) -> Result<()> {
    let equation = node
        .attrs
        .get("equation")
        .and_then(AttrValue::as_str)
        .context("Einsum: attribute 'equation' missing")?;
    let plan = einsum_as_matmul(equation)
        .with_context(|| format!("Einsum: '{equation}' is not a MatMul form"))?;
    env.ensure_device(&node.inputs[0])?;
    env.ensure_device(&node.inputs[1])?;
    let a = env.device(&node.inputs[0])?.shape.clone();
    let b = env.device(&node.inputs[1])?.shape.clone();
    let EinsumMatmul { batch, m, k, n } = plan;
    ensure!(
        a.len() == batch + m + k && b.len() == batch + k + n,
        "Einsum '{equation}': ranks {} and {} disagree with the labels",
        a.len(),
        b.len()
    );
    ensure!(
        a[..batch] == b[..batch] && a[batch + m..] == b[batch..batch + k],
        "Einsum '{equation}': shared labels differ in size ({a:?}, {b:?})"
    );
    let product = |dims: &[i64]| dims.iter().product::<i64>();
    let (rows, inner, cols) = (
        product(&a[batch..batch + m]),
        product(&a[batch + m..]),
        product(&b[batch + k..]),
    );
    let a_shape = [&a[..batch], &[rows, inner]].concat();
    let b_shape = [&b[..batch], &[inner, cols]].concat();
    let (out, _, elem_count) = matmul_fp32_shaped(env, node, &a_shape, &b_shape)?;
    let shape = [&a[..batch + m], &b[batch + k..]].concat();
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

/// `InstanceNormalization` f32: each `(n, c)` slice of `[N, C, D..]` normalized
/// over its spatial elements, then scaled and shifted per channel. One
/// workgroup per slice; see `INSTANCENORM`.
fn instance_norm(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    for name in &node.inputs[..3] {
        env.ensure_device(name)?;
    }
    let epsilon = node
        .attrs
        .get("epsilon")
        .and_then(AttrValue::as_f32)
        .unwrap_or(1e-5);
    let x = env.device(&node.inputs[0])?;
    let scale = env.device(&node.inputs[1])?;
    let bias = env.device(&node.inputs[2])?;
    let (shape, elem_count) = (x.shape.clone(), x.elem_count);
    ensure!(
        shape.len() >= 2,
        "InstanceNormalization: rank {} < 2",
        shape.len()
    );
    let channels = shape[1].max(0) as usize;
    ensure!(
        scale.elem_count == channels && bias.elem_count == channels,
        "InstanceNormalization: scale {} / bias {} elements for {channels} channels",
        scale.elem_count,
        bias.elem_count
    );
    let rows = shape[0].max(0) as usize * channels;
    ensure!(
        rows <= 65535,
        "InstanceNormalization: {rows} slices is too many"
    );
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, elem_count)?)?;
    if elem_count > 0 {
        let len = (elem_count / rows) as u32;
        let chunks = len.div_ceil(INSTANCENORM_CHUNK);
        ensure!(
            chunks <= 65535,
            "InstanceNormalization: {len} elements per slice is too many"
        );
        // sums of each chunk, then sums of squared deviations
        let part =
            ctx.create_storage_buffer(device_storage_bytes(FLOAT, 2 * rows * chunks as usize)?)?;
        for mode in 0u32..3 {
            let mut push = Vec::with_capacity(INSTANCENORM_PUSH_BYTES as usize);
            push.extend_from_slice(&len.to_le_bytes());
            push.extend_from_slice(&(channels as u32).to_le_bytes());
            push.extend_from_slice(&epsilon.to_le_bytes());
            push.extend_from_slice(&chunks.to_le_bytes());
            push.extend_from_slice(&(rows as u32).to_le_bytes());
            push.extend_from_slice(&mode.to_le_bytes());
            with_pipeline(
                env.cache(),
                "InstanceNormalization",
                || {
                    ctx.create_pipeline(
                        &compile_wgsl(INSTANCENORM)?,
                        INSTANCENORM_BINDINGS,
                        INSTANCENORM_PUSH_BYTES,
                    )
                },
                |pipe| {
                    ctx.stream_dispatch(
                        pipe,
                        &[x.buffer(), scale.buffer(), bias.buffer(), &out, &part],
                        &push,
                        [chunks, rows as u32, 1],
                    )
                },
            )?;
        }
        ctx.defer_destroy(part);
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

/// `MatMulNBits` (com.microsoft): `A [.., M, K] × dequant(B) [K, N]`.
///
/// `K` and `N` come from the attributes rather than from `B`'s shape, which
/// states neither: `B` is `[N, n_blocks, blob]` and its `K` is only implied by
/// `n_blocks · block_size`, rounded up. The check that they agree with `A` is
/// the one that catches a mis-parsed graph.
///
/// The batch dimensions of `A` are folded into the row index — `B` has none, so
/// there is nothing to broadcast against.
fn matmul_nbits(env: &mut Env, node: &NodeIr) -> Result<()> {
    let ctx = env.context();
    let int_attr = |name: &str| {
        node.attrs
            .get(name)
            .and_then(AttrValue::as_i64)
            .unwrap_or(0)
    };
    let k = int_attr("K") as usize;
    let n = int_attr("N") as usize;
    let block_size = int_attr("block_size") as usize;
    let n_blocks = k.div_ceil(block_size);
    let blob_words = block_size / 8;

    env.ensure_device(&node.inputs[0])?;
    // the packed weight and its zero points stay u8 in VRAM: the kernel unpacks
    // them, and widening them here would undo the point of the format
    for name in &node.inputs[1..4] {
        env.ensure_device_dtype(name)?;
    }
    let a = env.device(&node.inputs[0])?;
    let quant = env.device(&node.inputs[1])?;
    let scales = env.device(&node.inputs[2])?;
    let zero_points = env.device(&node.inputs[3])?;
    let a_shape = a.shape.clone();
    ensure!(!a_shape.is_empty(), "MatMulNBits: A is a scalar");
    ensure!(
        *a_shape.last().unwrap() as usize == k,
        "MatMulNBits: A's last dimension {:?} disagrees with K={k}",
        a_shape.last()
    );
    ensure!(
        quant.dtype == UINT8 && zero_points.dtype == UINT8 && scales.dtype == FLOAT,
        "MatMulNBits: expected u8 weight and zero point with f32 scales, got {}/{}/{}",
        quant.dtype,
        zero_points.dtype,
        scales.dtype
    );
    ensure!(
        quant.elem_count >= n * n_blocks * blob_words * 4
            && scales.elem_count >= n * n_blocks
            && zero_points.elem_count >= n * n_blocks.div_ceil(2),
        "MatMulNBits: packed weight is short for K={k} N={n} block_size={block_size}"
    );

    let rows: usize = a_shape[..a_shape.len() - 1].iter().product::<i64>().max(0) as usize;
    // one workgroup per output element, and `rows` is a grid dimension
    ensure!(rows <= 65535, "MatMulNBits: {rows} rows of A is too many");
    let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
    out_shape.push(n as i64);
    let elem_count = rows * n;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, elem_count)?)?;

    // The decode step is where this op lives: at `rows == 1` every MatMulNBits of
    // a decoder is a matrix-vector product, 253 of them per token on
    // qwen2.5-VL's. Which kernel is fastest there depends on `N` and not on `K`,
    // measured over all 11 geometries of the two q4 decoders
    // (`example matmulnbits`, RTX 4070):
    //
    // | N | kernel | measured |
    // |---|---|---|
    // | < 4096 | `MATMUL_NBITS` | the wide-column form is 0.91× here |
    // | ≥ 4096 | `wide_source(4)` | 1.32× .. 1.43×, and 413 GB/s of the card's 504 |
    //
    // Below 4096 columns the node runs at the dispatch floor — 0.010 ms, 60 GB/s
    // of a 0.6 MB weight — so there is nothing to win and a wider kernel loses.
    // Above it there used to be a second fork at `N = 65536` onto a word-loading
    // kernel at 16 lanes; the full sweep of all 11 census geometries retired it,
    // because the block form at 4 lanes beats it at every width above the floor
    // (`docs/autotuning.md` §2).
    // `rows > 1` (prefill) keeps the original kernel: it re-reads the weight per
    // row, which is the honest cost of not having a tiled form yet, and no
    // measurement of prefill exists to justify choosing differently.
    let variant = mmnb_route(rows, n);
    if rows > 0 && n > 0 {
        // one workgroup per column in the shipped kernel, one per `256 / lanes`
        // columns in the decode variants
        let groups = match variant {
            MmnbRoute::Wide { lanes } => (n as u32).div_ceil(256 / lanes),
            MmnbRoute::Row => n as u32,
        };
        let gx = groups.clamp(1, 32768);
        let mut push = Vec::with_capacity(MMNB_PUSH_BYTES as usize);
        for v in [
            k as u32,
            n as u32,
            n_blocks as u32,
            blob_words as u32,
            block_size as u32,
            n_blocks.div_ceil(2) as u32,
            gx,
            0,
        ] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        // separate Pareto keys: the two forms cost differently and the split is
        // the thing to watch when a decoder's profile is read
        let (operation, source) = match variant {
            MmnbRoute::Wide { lanes } => ("MatMulNBits_wide", mmnb_wide_source(lanes)),
            MmnbRoute::Row => ("MatMulNBits", MMNB_MATMUL.to_string()),
        };
        let key = PipelineKey::tactic(operation, format!("{variant:?}"));
        with_pipeline(
            env.cache(),
            key,
            || ctx.create_pipeline(&compile_wgsl(&source)?, MMNB_BINDINGS, MMNB_PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[
                        a.buffer(),
                        quant.buffer(),
                        scales.buffer(),
                        zero_points.buffer(),
                        &out,
                    ],
                    &push,
                    [gx, groups.div_ceil(gx), rows as u32],
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
    let (a, b) = (&node.inputs[0], &node.inputs[1]);
    let bc = broadcast(&env.device(a)?.shape, &env.device(b)?.shape)?;
    broadcast_dispatch(env, &node.outputs[0], a, b, bc, gpu_op)
}

/// Runs the elementwise template over `bc` on two device f32 tensors.
fn broadcast_dispatch(
    env: &mut Env,
    output: &str,
    a: &str,
    b: &str,
    bc: Broadcast,
    gpu_op: &'static str,
) -> Result<()> {
    let ctx = env.context();
    let (a, b) = (env.device(a)?, env.device(b)?);
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
        output,
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
        let mut push = Vec::with_capacity(SOFTMAX_PUSH_BYTES as usize);
        // rows are packed: the pitch is the row itself
        for v in [c as u32, inner as u32, rows as u32, c as u32, gx] {
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

/// `MatMul(a, b) → Softmax → MatMul(p, v)`, fused by
/// `rewrite::fuse_attention`, in one dispatch that never stores the scores.
///
/// The kernel covers what the rewrite cannot check without shapes: f32
/// operands, identical batch dimensions (no broadcast) and head sizes the
/// shader keeps in registers. Anything else runs the three original steps, so
/// the rewrite never turns a graph that ran into one that is refused.
///
/// With `v_bshd`, `v` is `[B, n, H, DV]` and stands for its
/// `Transpose(·, [0, 2, 1, 3])`; with `out_bshd`, the output is written as
/// `[B, m, H, DV]`, the same transpose of the attention's result.
///
/// A fourth input is a bias added to the scores; the kernel takes it when it
/// is f32 and broadcasts to the scores without growing them
/// ([`attention_bias_strides`]). `nan_to_zero` zeroes the rows whose
/// probabilities would be NaN.
fn flash_attention(env: &mut Env, node: &NodeIr) -> Result<()> {
    use crate::shaders::flash_attention as fa;
    for name in &node.inputs {
        env.ensure_device(name)?;
    }
    let (v_bshd, out_bshd) = attention_layout(node);
    let ctx = env.context();
    let (a, b, v) = (
        env.device(&node.inputs[0])?,
        env.device(&node.inputs[1])?,
        env.device(&node.inputs[2])?,
    );
    let bias = node
        .inputs
        .get(3)
        .map(|name| env.device(name))
        .transpose()?;
    let nan_to_zero = node.attrs.get("nan_to_zero").and_then(AttrValue::as_i64) == Some(1);
    let limits = ctx.compute_limits();
    let swap_heads = |shape: &[i64]| match *shape {
        [batch, x, heads, y] => Some(vec![batch, heads, x, y]),
        _ => None,
    };
    let v_shape = if v_bshd {
        swap_heads(&v.shape)
    } else {
        Some(v.shape.clone())
    };
    let Some((bh, m, n, d, dv, bc)) = v_shape
        .and_then(|v_shape| flash_attention_dims(&a.shape, &b.shape, &v_shape))
        .filter(|_| !out_bshd || a.shape.len() == 4)
        .filter(|_| [a.dtype, b.dtype, v.dtype].iter().all(|&t| t == FLOAT))
        .filter(|&(bh, m, n, d, dv)| {
            let fits_u32 = |elems: usize| u32::try_from(elems).is_ok();
            m.div_ceil(fa::BR as usize) <= limits.max_workgroup_count[0] as usize
                && bh <= limits.max_workgroup_count[1] as usize
                && fits_u32(bh * m * d as usize)
                && fits_u32(bh * n * d.max(dv) as usize)
                && fits_u32(bh * m * dv as usize)
        })
        .and_then(|(bh, m, n, d, dv)| {
            let bc = fa::key_tile(d, dv, limits.max_shared_memory_bytes)?;
            Some((bh, m, n, d, dv, bc))
        })
    else {
        return attention_unfused(env, node);
    };
    // `[heads, outer, head, row]`, all zero without a bias
    let bias_push = match bias {
        None => Some([1, 0, 0, 0]),
        Some(bias) if bias.dtype == FLOAT => attention_bias_strides(&a.shape, n, &bias.shape),
        Some(_) => None,
    };
    let Some(bias_push) = bias_push else {
        return attention_unfused(env, node);
    };
    let mut out_shape = a.shape.clone();
    *out_shape.last_mut().expect("rank checked") = i64::from(dv);
    // heads of the [B, ·, H, DV] sides; 1 leaves a side [bh, ·, DV]
    let v_heads = if v_bshd { v.shape[2] as u32 } else { 1 };
    let out_heads = if out_bshd {
        out_shape = swap_heads(&out_shape).expect("rank 4 checked");
        out_shape[2] as u32
    } else {
        1
    };
    let elem_count = bh * m * dv as usize;
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, elem_count)?)?;
    let mut push = Vec::with_capacity(fa::PUSH_BYTES as usize);
    for value in [
        m as u32,
        n as u32,
        v_heads,
        out_heads,
        u32::from(nan_to_zero),
    ]
    .into_iter()
    .chain(bias_push)
    {
        push.extend_from_slice(&value.to_le_bytes());
    }
    let mut buffers = vec![a.buffer(), b.buffer(), v.buffer(), &out];
    buffers.extend(bias.map(|bias| bias.buffer()));
    let (bindings, suffix) = if bias.is_some() {
        (fa::BIAS_BINDINGS, "_bias")
    } else {
        (fa::BINDINGS, "")
    };
    if attention_fp16()
        && coop_attention_fits(ctx, m, d, dv)
        && ctx.coop_f16_wg
        && n <= fa::WG_MAX_KEYS
    {
        with_pipeline(
            env.cache(),
            PipelineKey::tactic("FlashAttention_f16_wg", format!("d{d}_dv{dv}{suffix}")),
            || ctx.create_pipeline(&fa::wg_spirv(d, bias.is_some()), bindings, fa::PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &buffers,
                    &push,
                    [(m as u32).div_ceil(fa::COOP_BR), bh as u32, 1],
                )
            },
        )?;
    } else if attention_fp16() && coop_attention_fits(ctx, m, d, dv) {
        with_pipeline(
            env.cache(),
            PipelineKey::tactic("FlashAttention_f16", format!("d{d}_dv{dv}{suffix}")),
            || ctx.create_pipeline(&fa::coop_spirv(d, bias.is_some()), bindings, fa::PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &buffers,
                    &push,
                    [(m as u32).div_ceil(fa::COOP_BR), bh as u32, 1],
                )
            },
        )?;
    } else {
        with_pipeline(
            env.cache(),
            PipelineKey::tactic("FlashAttention", format!("d{d}_dv{dv}_bc{bc}{suffix}")),
            || {
                ctx.create_pipeline(
                    &compile_wgsl(&fa::source(d, dv, bc, bias.is_some()))?,
                    bindings,
                    fa::PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &buffers,
                    &push,
                    [(m as u32).div_ceil(fa::BR), bh as u32, 1],
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

/// Environment switch for the f16 FlashAttention kernel: `1` turns it on.
pub const ATTENTION_FP16_ENV: &str = "ONNX_VULKAN_ATTENTION_FP16";

/// Whether `ONNX_VULKAN_ATTENTION_FP16=1`: fused attention with heads of 32
/// or 64 multiplies in f16 on cooperative matrices (f32 accumulation, softmax
/// and normalization) instead of in f32. Off by default because the results
/// change beyond summation order: Q, K, V and the probabilities are rounded to
/// f16. An application turns it on only behind its own quality gate. Read once
/// per process, so the EP and standalone hosts of one run agree.
pub fn attention_fp16() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var(ATTENTION_FP16_ENV).is_ok_and(|v| v == "1");
        if on {
            log::warn!(
                "{ATTENTION_FP16_ENV}=1: FlashAttention with heads of 32 or 64 multiplies in f16 \
                 (f32 accumulation); results differ from the f32 kernel"
            );
        }
        on
    })
}

/// Whether the f16 cooperative-matrix kernel runs this attention on `ctx`:
/// the device advertises the `f16 × f16 → f32` configuration on subgroups of
/// 32, and the heads are the kernel's.
fn coop_attention_fits(ctx: &VkContext, m: usize, d: u32, dv: u32) -> bool {
    use crate::shaders::flash_attention as fa;
    let limits = ctx.compute_limits();
    ctx.coop_f16
        && ctx.subgroup_size == fa::COOP_SUBGROUP
        && fa::COOP_HEADS.contains(&d)
        && dv == d
        && fa::COOP_INVOCATIONS <= limits.max_workgroup_invocations
        && fa::COOP_INVOCATIONS <= limits.max_workgroup_size[0]
        && fa::coop_shared_bytes(d) <= limits.max_shared_memory_bytes
        && m.div_ceil(fa::COOP_BR as usize) <= limits.max_workgroup_count[0] as usize
}

/// `(batch, m, n, d, dv)` when `a [.., m, d]`, `b [.., d, n]` and
/// `v [.., n, dv]` share their batch dimensions exactly and fit the shader.
fn flash_attention_dims(
    a: &[i64],
    b: &[i64],
    v: &[i64],
) -> Option<(usize, usize, usize, u32, u32)> {
    use crate::shaders::flash_attention::{DV_MULTIPLE, MAX_D, MAX_DV};
    let rank = a.len();
    if rank < 2 || b.len() != rank || v.len() != rank {
        return None;
    }
    let batch = &a[..rank - 2];
    if b[..rank - 2] != *batch || v[..rank - 2] != *batch || batch.iter().any(|&x| x < 1) {
        return None;
    }
    let (m, d, n, dv) = (a[rank - 2], a[rank - 1], b[rank - 1], v[rank - 1]);
    let fits = m >= 1
        && n >= 1
        && b[rank - 2] == d
        && v[rank - 2] == n
        && (1..=i64::from(MAX_D)).contains(&d)
        && (i64::from(DV_MULTIPLE)..=i64::from(MAX_DV)).contains(&dv)
        && dv % i64::from(DV_MULTIPLE) == 0;
    fits.then(|| {
        let bh = batch.iter().product::<i64>() as usize;
        (bh, m as usize, n as usize, d as u32, dv as u32)
    })
}

/// The kernel's bias push constants `[heads, outer, head, row]` for scores
/// `[.., H, m, n]` (queries `a` are `[.., H, m, d]`), when `bias` is
/// contiguous and broadcasts to the scores without growing them, and its
/// batch dimensions before `H` are all broadcast or all present: the kernel
/// splits a slice index only into `(outer, head)`.
fn attention_bias_strides(a: &[i64], n: usize, bias: &[i64]) -> Option<[u32; 4]> {
    let rank = a.len();
    if bias.len() > rank || rank < 2 {
        return None;
    }
    let mut padded = vec![1; rank - bias.len()];
    padded.extend_from_slice(bias);
    let (m, n) = (a[rank - 2], i64::try_from(n).ok()?);
    let (bm, bn) = (padded[rank - 2], padded[rank - 1]);
    let heads = if rank >= 3 { a[rank - 3] } else { 1 };
    let bh = if rank >= 3 { padded[rank - 3] } else { 1 };
    let (outer, b_outer) = (
        &a[..rank.saturating_sub(3)],
        &padded[..rank.saturating_sub(3)],
    );
    let outer_ones = b_outer.iter().all(|&x| x == 1);
    let fits = bn == n
        && (bm == 1 || bm == m)
        && (bh == 1 || bh == heads)
        && (outer_ones || b_outer == outer)
        && u32::try_from(padded.iter().product::<i64>()).is_ok();
    if !fits {
        return None;
    }
    let row = if bm == 1 { 0 } else { n };
    let head = if bh == 1 { 0 } else { bm * n };
    let outer = if outer_ones { 0 } else { bh * bm * n };
    let [heads, outer, head, row] = [heads, outer, head, row].map(|x| u32::try_from(x).ok());
    Some([heads?, outer?, head?, row?])
}

/// The nodes `rewrite::fuse_attention` replaced, run as they were.
/// `(v_bshd, out_bshd)`: which head transposes `rewrite::fuse_attention`
/// folded into the node.
fn attention_layout(node: &NodeIr) -> (bool, bool) {
    let flag = |key| node.attrs.get(key).and_then(AttrValue::as_i64) == Some(1);
    (flag("v_bshd"), flag("out_bshd"))
}

fn attention_unfused(env: &mut Env, node: &NodeIr) -> Result<()> {
    let scores = format!("{}/attention_scores", node.outputs[0]);
    let probs = format!("{}/attention_probs", node.outputs[0]);
    let (v_bshd, out_bshd) = attention_layout(node);
    let v_heads = format!("{}/attention_v", node.outputs[0]);
    let product = format!("{}/attention_product", node.outputs[0]);
    let heads_swap = |input: &str, output: &str| {
        let mut step = lowered(node, "Transpose", &[input], &[output]);
        step.attrs = HashMap::from([("perm".to_owned(), AttrValue::Ints(vec![0, 2, 1, 3]))]);
        step
    };
    let v = if v_bshd {
        exec_dispatch(env, &heads_swap(&node.inputs[2], &v_heads))?;
        v_heads.as_str()
    } else {
        node.inputs[2].as_str()
    };
    let y = if out_bshd {
        product.as_str()
    } else {
        node.outputs[0].as_str()
    };
    let step = |op: &str, inputs: &[&str], output: &str, attrs| NodeIr {
        domain: String::new(),
        op: op.to_owned(),
        opset: node.opset,
        name: node.name.clone(),
        inputs: inputs
            .iter()
            .filter(|n| !n.is_empty())
            .map(|&n| n.to_owned())
            .collect(),
        outputs: vec![output.to_owned()],
        attrs,
    };
    let last_axis = HashMap::from([("axis".to_owned(), AttrValue::Int(-1))]);
    exec_dispatch(
        env,
        &step(
            "MatMul",
            &[&node.inputs[0], &node.inputs[1]],
            &scores,
            HashMap::new(),
        ),
    )?;
    if let Some(bias) = node.inputs.get(3) {
        let biased = format!("{}/attention_biased", node.outputs[0]);
        exec_dispatch(env, &step("Add", &[&scores, bias], &biased, HashMap::new()))?;
        env.release(&scores);
        exec_dispatch(env, &step("Softmax", &[&biased], &probs, last_axis))?;
        env.release(&biased);
    } else {
        exec_dispatch(env, &step("Softmax", &[&scores], &probs, last_axis))?;
        env.release(&scores);
    }
    let cleaned = format!("{}/attention_cleaned", node.outputs[0]);
    let probs = if node.attrs.get("nan_to_zero").and_then(AttrValue::as_i64) == Some(1) {
        let (is_nan, zero) = (
            format!("{}/attention_is_nan", node.outputs[0]),
            format!("{}/attention_zero", node.outputs[0]),
        );
        exec_dispatch(env, &step("IsNaN", &[&probs], &is_nan, HashMap::new()))?;
        env.set(&zero, Tensor::Host(HostTensor::from_f32(vec![], &[0.0])));
        exec_dispatch(
            env,
            &step("Where", &[&is_nan, &zero, &probs], &cleaned, HashMap::new()),
        )?;
        for name in [&is_nan, &zero, &probs] {
            env.release(name);
        }
        cleaned.as_str()
    } else {
        probs.as_str()
    };
    exec_dispatch(env, &step("MatMul", &[probs, v], y, HashMap::new()))?;
    env.release(probs);
    if v_bshd {
        env.release(&v_heads);
    }
    if out_bshd {
        exec_dispatch(env, &heads_swap(&product, &node.outputs[0]))?;
        env.release(&product);
    }
    Ok(())
}

/// Which of the three row normalizations `layernorm` is running.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Norm {
    /// `LayerNormalization`: mean subtracted, optional bias.
    Centered,
    /// `SimplifiedLayerNormalization`: RMS, scale only.
    Rms,
    /// `SkipSimplifiedLayerNormalization`: RMS over `x + skip`, and the sum
    /// itself as the optional fourth output.
    RmsSkip,
    /// `AddLayerNormalization`: `Centered` over `x + skip`, and the sum as the
    /// optional second output.
    AddCentered,
}

/// `LayerNormalization` f32 on the last dimension; scale (+optional bias)
/// typically a per-channel initializer.
///
/// The simplified forms are the same reduction with no mean subtracted
/// (`Norm::Rms`), optionally over `x` plus a residual (`Norm::RmsSkip`); the
/// three share one pipeline and differ by push constants.
fn layernorm(env: &mut Env, node: &NodeIr, form: Norm) -> Result<()> {
    let ctx = env.context();
    let simplified = matches!(form, Norm::Rms | Norm::RmsSkip);
    let op = match form {
        Norm::Centered => "LayerNormalization",
        Norm::Rms => "SimplifiedLayerNormalization",
        Norm::RmsSkip => "SkipSimplifiedLayerNormalization",
        Norm::AddCentered => "AddLayerNormalization",
    };
    let has_skip = matches!(form, Norm::RmsSkip | Norm::AddCentered);
    // the skip forms' second input is the residual, not a scale: their scale
    // is input 2. The RMS one has no bias in the shape this kernel claims
    let scale_index = if has_skip { 2 } else { 1 };
    let bias_index = scale_index + 1;
    let has_bias =
        !simplified && node.inputs.len() > bias_index && !node.inputs[bias_index].is_empty();
    // where the residual sum goes when the graph asks for it
    let sum_index = if form == Norm::RmsSkip { 3 } else { 1 };
    env.ensure_device(&node.inputs[0])?;
    env.ensure_device(&node.inputs[scale_index])?;
    if has_bias {
        env.ensure_device(&node.inputs[bias_index])?;
    }
    if has_skip {
        env.ensure_device(&node.inputs[1])?;
    }
    let epsilon = node
        .attrs
        .get("epsilon")
        .and_then(|a| a.as_f32())
        .unwrap_or(1e-5);

    let x = env.device(&node.inputs[0])?;
    let scale = env.device(&node.inputs[scale_index])?;
    let bias = if has_bias {
        env.device(&node.inputs[bias_index])?
    } else {
        scale
    };
    let skip = if has_skip {
        let skip = env.device(&node.inputs[1])?;
        ensure!(
            skip.elem_count == x.elem_count,
            "{op}: skip elem {} != input {}",
            skip.elem_count,
            x.elem_count
        );
        skip
    } else {
        x
    };
    let (shape, elem_count) = (x.shape.clone(), x.elem_count);
    last_axis(node, shape.len() as i64, op)?;
    let (c, rows) = row_cols(&shape, elem_count, op)?;
    ensure!(
        scale.elem_count == c,
        "{op}: scale elem {} != {c}",
        scale.elem_count
    );
    let out = ctx.create_storage_buffer((elem_count.max(1) * 4) as u64)?;
    // the residual sum, only when the graph asks for it: `x + skip` is what the
    // next layer adds onto, and recomputing it there would be a second pass
    let has_sum = has_skip && node.outputs.get(sum_index).is_some_and(|o| !o.is_empty());
    let sum = if has_sum {
        Some(ctx.create_storage_buffer((elem_count.max(1) * 4) as u64)?)
    } else {
        None
    };
    if elem_count > 0 {
        let mut push = Vec::with_capacity(LAYERNORM_PUSH_BYTES as usize);
        push.extend_from_slice(&(c as u32).to_le_bytes());
        push.extend_from_slice(&epsilon.to_le_bytes());
        push.extend_from_slice(&(has_bias as u32).to_le_bytes());
        push.extend_from_slice(&(simplified as u32).to_le_bytes());
        push.extend_from_slice(&(has_skip as u32).to_le_bytes());
        push.extend_from_slice(&(has_sum as u32).to_le_bytes());
        // separate Pareto keys for the same pipeline: the forms cost
        // differently and an LLM runs only one of them
        with_pipeline(
            env.cache(),
            op,
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
                    &[
                        x.buffer(),
                        scale.buffer(),
                        bias.buffer(),
                        &out,
                        skip.buffer(),
                        sum.as_ref().unwrap_or(&out),
                    ],
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
            shape: shape.clone(),
            elem_count,
            buf: BufRef::Owned(out),
        }),
    );
    if let Some(sum) = sum {
        env.set(
            &node.outputs[sum_index],
            Tensor::Device(DevTensor {
                dtype: FLOAT,
                shape,
                elem_count,
                buf: BufRef::Owned(sum),
            }),
        );
    }
    Ok(())
}

/// The node `rewrite::fuse_add_layernorm` writes: inputs `a`, `b`, scale and
/// optional bias; outputs the normalization and, when it escapes, `a + b`.
///
/// The kernel reads the residual element for element, so it claims only two
/// `f32` operands of one shape. A broadcast or shape-math `Add` runs as the
/// graph wrote it.
fn add_layernorm(env: &mut Env, node: &NodeIr) -> Result<()> {
    let (a, b) = (&node.inputs[0], &node.inputs[1]);
    // the rule of `elementwise_binary`: small host operands are shape math
    let small_host = env.host_resident_small(a, Env::SMALL_HOST_BYTES)
        && env.host_resident_small(b, Env::SMALL_HOST_BYTES);
    if env.dtype_of(a)? == FLOAT
        && env.dtype_of(b)? == FLOAT
        && env.shape_of(a)? == env.shape_of(b)?
        && !small_host
    {
        return layernorm(env, node, Norm::AddCentered);
    }
    let out = node.outputs[0].as_str();
    let escapes = node.outputs.get(1).filter(|o| !o.is_empty());
    let temp = format!("{out}/sum");
    let sum = escapes.map_or(temp.as_str(), String::as_str);
    let mut add = lowered(node, "Add", &[a, b], &[sum]);
    add.attrs.clear();
    exec_dispatch(env, &add)?;
    let mut norm = vec![sum];
    norm.extend(node.inputs[2..].iter().map(String::as_str));
    exec_dispatch(env, &lowered(node, "LayerNormalization", &norm, &[out]))?;
    if escapes.is_none() {
        env.release(&temp);
    }
    Ok(())
}

/// Workgroups for a flat `count` of threads, folded into 2D because `count`
/// over the scores tensor easily exceeds 65535 workgroups on one axis.
/// Returns the grid and its `x` extent, which the shader needs to unfold it.
fn attn_grid(count: usize) -> ([u32; 3], u32) {
    let groups = count.div_ceil(ATTN_WG as usize) as u32;
    let gx = groups.clamp(1, 32768);
    ([gx, groups.div_ceil(gx), 1], gx)
}

/// `RotaryEmbedding` (com.microsoft), 72 nodes of qwen2.5-VL's decoder.
///
/// The rotation `GroupQueryAttention` fuses, as its own node: the positions
/// come from `position_ids` rather than from the cache length, and the layout is
/// untouched. Two shapes are admitted, `[b, n, s, H]` and `[b, s, n·H]`, because
/// the schema admits both and only the rank tells them apart — which is also why
/// `num_heads` is read as the attribute for the 3-D form and checked against the
/// shape for the 4-D one.
///
/// `position_ids` is taken to the host and re-uploaded as i32: WGSL has no
/// 64-bit scalar, and the tensor is already on the host in every graph that
/// builds it with `Range` — the same route `Gather` takes for its indices.
fn rotary_embedding(env: &mut Env, node: &NodeIr) -> Result<()> {
    let int_attr = |name: &str, default: i64| {
        node.attrs
            .get(name)
            .and_then(AttrValue::as_i64)
            .unwrap_or(default)
    };
    // i64 → i32, and the position indexes a cache row so a negative one is not
    // a wrap-around but a broken graph
    let host_positions = env.host(&node.inputs[1])?.to_i64()?;
    let positions: Vec<i32> = host_positions
        .iter()
        .map(|p| {
            ensure!(*p >= 0, "RotaryEmbedding: negative position {p}");
            Ok(*p as i32)
        })
        .collect::<Result<_>>()?;
    // Every layer rotates against the **same** positions: qwen2.5-VL has 72 of
    // these nodes and they all read one value. Uploading it per node was 72 of
    // the 90 transfers a decode step made, each with its own buffer, and the
    // sizes are identical so they cost 72 allocations as well. The converted
    // tensor is memoized under a derived name, which the value table already
    // serves for anything not in the graph.
    let pos_key = format!("{}#i32", node.inputs[1]);
    if !positions.is_empty() && !env.on_device(&pos_key) {
        let pos_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_le_bytes()).collect();
        let context = env.context();
        let buffer = context.create_storage_buffer(pos_bytes.len() as u64)?;
        context.stream_upload(&buffer, &pos_bytes)?;
        env.set(
            &pos_key,
            Tensor::Device(DevTensor {
                dtype: INT32,
                shape: vec![positions.len() as i64],
                elem_count: positions.len(),
                buf: BufRef::Owned(buffer),
            }),
        );
    }

    let ctx = env.context();
    for index in [0, 2, 3] {
        env.ensure_device(&node.inputs[index])?;
    }
    let x = env.device(&node.inputs[0])?;
    let cos = env.device(&node.inputs[2])?;
    let sin = env.device(&node.inputs[3])?;
    ensure!(
        cos.shape.len() == 2 && cos.shape == sin.shape,
        "RotaryEmbedding: cos/sin caches must be [positions, rotary_dim/2], got {:?}/{:?}",
        cos.shape,
        sin.shape
    );
    let cache_half = cos.shape[1].max(0) as usize;

    let bnsh = x.shape.len() == 4;
    ensure!(
        bnsh || x.shape.len() == 3,
        "RotaryEmbedding: input must be [b, n, s, H] or [b, s, n·H], got {:?}",
        x.shape
    );
    let attr_heads = int_attr("num_heads", 0).max(0) as usize;
    let (b, n, s, head_size) = if bnsh {
        let n = x.shape[1].max(0) as usize;
        ensure!(
            attr_heads == 0 || attr_heads == n,
            "RotaryEmbedding: num_heads {attr_heads} != {n} heads in {:?}",
            x.shape
        );
        (
            x.shape[0].max(0) as usize,
            n,
            x.shape[2].max(0) as usize,
            x.shape[3].max(0) as usize,
        )
    } else {
        ensure!(
            attr_heads > 0,
            "RotaryEmbedding: a [b, s, n·H] input needs num_heads"
        );
        let hidden = x.shape[2].max(0) as usize;
        ensure!(
            hidden.is_multiple_of(attr_heads),
            "RotaryEmbedding: hidden {hidden} is not a multiple of {attr_heads} heads"
        );
        (
            x.shape[0].max(0) as usize,
            attr_heads,
            x.shape[1].max(0) as usize,
            hidden / attr_heads,
        )
    };
    // a zero `rotary_embedding_dim` means the whole head, per the schema
    let rot_dim = match int_attr("rotary_embedding_dim", 0).max(0) as usize {
        0 => head_size,
        dim => dim,
    };
    ensure!(
        rot_dim <= head_size && rot_dim % 2 == 0 && rot_dim / 2 <= cache_half,
        "RotaryEmbedding: rotary dim {rot_dim} does not fit head {head_size} / cache {cache_half}"
    );
    ensure!(
        positions.len() == b * s,
        "RotaryEmbedding: {} position ids for {b}×{s} tokens",
        positions.len()
    );

    let (shape, elem_count) = (x.shape.clone(), x.elem_count);
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, elem_count)?)?;
    if elem_count > 0 {
        let pos_buf = env.device(&pos_key)?.buffer();
        let (grid, gx) = attn_grid(elem_count);
        let mut push = Vec::with_capacity(ATTN_ROTARY_PUSH_BYTES as usize);
        for v in [
            elem_count as u32,
            head_size as u32,
            s as u32,
            n as u32,
            (rot_dim / 2) as u32,
            cache_half as u32,
            u32::from(bnsh),
            gx,
        ] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        with_pipeline(
            env.cache(),
            "RotaryEmbedding",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(ATTN_ROTARY)?,
                    ATTN_ROTARY_BINDINGS,
                    ATTN_ROTARY_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[x.buffer(), pos_buf, cos.buffer(), sin.buffer(), &out],
                    &push,
                    grid,
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

/// `GroupQueryAttention` (com.microsoft), 26 nodes of gemma3-1b.
///
/// Stateless by default: `present_*` is a fresh tensor holding a copy of
/// `past_*` plus the new step, so the node is a pure function of its inputs and
/// can be diffed node by node against the CPU EP.
///
/// A caller running a generation loop can instead **bind** `present_key` and
/// `present_value` to buffers it owns (`Executor::run_with_outputs`) and pass
/// the same buffers back as `past_key`/`past_value`. Then the physical time
/// extent is the buffer's capacity rather than `total`, the new token is
/// written straight to its slot, and the cache copy — `GQA_past`, the whole
/// cache per node per token — is skipped because source and destination are
/// the same memory. The arithmetic is unchanged: both paths run the same
/// dispatches with the same push constants except for `stride`.
///
/// `seqlens_k` and `total_sequence_length` are **not read**: the lengths follow
/// from the shapes of `past_key` and `key`, which is equivalent as long as
/// every sequence in the batch has the same length. That holds trivially at
/// `b = 1` and cannot be checked without a readback otherwise, so `b > 1` is
/// refused rather than guessed.
/// Elements an attention bias must cover to stay the same buffer across a
/// generation: the bound cache's physical time extent, one row per query.
///
/// `None` when the cache is not bound — a stateless run rebuilds it at the
/// step's own length, where the bias already has the right size.
fn bias_capacity(env: &Env<'_, '_>, node: &NodeIr) -> Result<Option<usize>> {
    let Some((_, cap)) = env.bound_output(&node.outputs[1]) else {
        return Ok(None);
    };
    let past = env.device(&node.inputs[3])?;
    let query = env.device(&node.inputs[0])?;
    if past.shape.len() != 4 || query.shape.len() != 3 {
        return Ok(None);
    }
    let b = past.shape[0].max(0) as usize;
    let row = b * past.shape[1].max(0) as usize * past.shape[3].max(0) as usize;
    let s = query.shape[1].max(0) as usize;
    if row == 0 {
        return Ok(None);
    }
    Ok(Some(b * s * (cap / row)))
}

fn group_query_attention<'values>(env: &mut Env<'_, 'values>, node: &NodeIr) -> Result<()> {
    let int_attr = |name: &str, default: i64| {
        node.attrs
            .get(name)
            .and_then(AttrValue::as_i64)
            .unwrap_or(default)
    };
    let nh = int_attr("num_heads", 0).max(0) as usize;
    let kvh = int_attr("kv_num_heads", 0).max(0) as usize;
    ensure!(
        nh > 0 && kvh > 0 && nh.is_multiple_of(kvh),
        "GroupQueryAttention: num_heads {nh} is not a multiple of kv_num_heads {kvh}"
    );
    let do_rotary = int_attr("do_rotary", 0) != 0;
    let window = int_attr("local_window_size", -1);
    let has_bias = node.inputs.len() > 10 && !node.inputs[10].is_empty();

    let ctx = env.context();
    for index in 0..5 {
        env.ensure_device(&node.inputs[index])?;
    }
    if do_rotary {
        env.ensure_device(&node.inputs[7])?;
        env.ensure_device(&node.inputs[8])?;
    }
    // The bias is `[b, 1, s, total]` and gains one element per token, so a
    // buffer sized to it is a fresh allocation — and a fresh `VkBuffer` — at
    // every decode step: the last value in a decoder whose identity moves.
    // Against a bound cache it goes into a buffer laid out for the cache's whole
    // physical extent instead. The rows stay where the graph put them, `total`
    // apart, so nothing about the addressing changes; the padding is all at the
    // end, past the last key, and `GQA_scores` masks every key at or past this
    // step's length before it ever reads the bias.
    if has_bias {
        match bias_capacity(env, node)? {
            Some(capacity) => {
                env.ensure_device_padded(&node.inputs[10], capacity)?;
            }
            None => env.ensure_device(&node.inputs[10])?,
        }
    }

    let q = env.device(&node.inputs[0])?;
    let k = env.device(&node.inputs[1])?;
    let v = env.device(&node.inputs[2])?;
    let past_k = env.device(&node.inputs[3])?;
    let past_v = env.device(&node.inputs[4])?;
    ensure!(
        q.shape.len() == 3 && k.shape.len() == 3 && v.shape.len() == 3,
        "GroupQueryAttention: query/key/value must be [batch, seq, hidden]"
    );
    ensure!(
        past_k.shape.len() == 4 && past_v.shape == past_k.shape,
        "GroupQueryAttention: past_key/past_value must be [batch, kv_heads, past, head_size]"
    );
    let b = q.shape[0].max(0) as usize;
    ensure!(
        b == 1,
        "GroupQueryAttention: batch {b} > 1 would need the per-sequence lengths in seqlens_k"
    );
    let s = q.shape[1].max(0) as usize;
    let head_size = k.shape[2].max(0) as usize / kvh;
    ensure!(
        head_size > 0 && q.shape[2].max(0) as usize == nh * head_size && k.shape == v.shape,
        "GroupQueryAttention: hidden sizes {:?}/{:?} disagree with {nh}/{kvh} heads",
        q.shape,
        k.shape
    );
    let past = past_k.shape[2].max(0) as usize;
    ensure!(
        past_k.shape[1].max(0) as usize == kvh && past_k.shape[3].max(0) as usize == head_size,
        "GroupQueryAttention: past cache {:?} disagrees with {kvh} heads of {head_size}",
        past_k.shape
    );
    let total = past + s;
    let scale = match node.attrs.get("scale").and_then(AttrValue::as_f32) {
        Some(scale) if scale != 0.0 => scale,
        _ => 1.0 / (head_size as f32).sqrt(),
    };

    let (cos, sin) = if do_rotary {
        (env.device(&node.inputs[7])?, env.device(&node.inputs[8])?)
    } else {
        (q, q)
    };
    let rot_half = if do_rotary {
        let half = (*cos.shape.last().unwrap_or(&0)).max(0) as usize;
        ensure!(
            half * 2 == head_size && cos.shape == sin.shape,
            "GroupQueryAttention: rotary width {} != head size {head_size}",
            half * 2
        );
        half
    } else {
        0
    };
    let bias = if has_bias {
        env.device(&node.inputs[10])?
    } else {
        q
    };

    let alloc = |elems: usize| ctx.create_storage_buffer((elems.max(1) * 4) as u64);

    // Physical time extent of `present_key`/`present_value`, as opposed to the
    // `total` the graph fills. They coincide while the cache is rebuilt on
    // every call; a bound cache is laid out for `max_seq_len` instead, because
    // the row stride of `[b, kvh, T, H]` is `T` and a growing `T` would move
    // every token already written.
    let row = b * kvh * head_size;
    let bound = match (
        env.bound_output(&node.outputs[1]),
        env.bound_output(&node.outputs[2]),
    ) {
        (Some(key), Some(value)) => Some((key, value)),
        (None, None) => None,
        _ => {
            bail!("GroupQueryAttention: bind present_key and present_value together or not at all")
        }
    };
    let (present_k, present_v, kv_stride, padded_rows) = match bound {
        Some(((k_buf, k_cap), (v_buf, v_cap))) => {
            ensure!(
                row > 0 && k_cap == v_cap && k_cap.is_multiple_of(row),
                "GroupQueryAttention: a bound cache of {k_cap}/{v_cap} elements is not a whole \
                 number of {row}-element time steps"
            );
            let stride = k_cap / row;
            ensure!(
                stride >= total,
                "GroupQueryAttention: the bound cache holds {stride} tokens, the graph asks for {total}"
            );
            // Every kernel below addresses the cache through `kv_stride`, so a
            // padded cache is as addressable at `kvh = 8` as at `kvh = 1` — and
            // it has to be, because grouped attention with `1 < kvh < nh` is
            // what Llama 3, Qwen and most of the field export. What multiple
            // rows do change is the **shape** of the present outputs below:
            // `[b, kvh, total, head_size]` is the logical cache, but the written
            // tokens of each row sit `stride` apart, so it stops describing the
            // buffer's leading `total · row` elements. Reading it back on the
            // host is what that breaks — the next step consumes it in VRAM,
            // where the stride is what the kernels use — so the padded rows are
            // reported to the caller rather than refused here.
            let padded_rows = stride != total && b * kvh > 1;
            (
                BufRef::Borrowed(k_buf),
                BufRef::Borrowed(v_buf),
                stride,
                padded_rows,
            )
        }
        None => (
            BufRef::Owned(alloc(row * total)?),
            BufRef::Owned(alloc(row * total)?),
            total,
            false,
        ),
    };
    let (present_k_buf, present_v_buf) = (present_k.get(), present_v.get());

    // Key extent the score and probability scratch are laid out for, and the one
    // the two dispatches over them cover. Against a resident cache this is the
    // cache's physical stride rather than the `total` this step reached, which
    // makes both the allocation sizes and the grids **independent of the step**:
    // that is the precondition for reusing a recorded dispatch across tokens,
    // where `total` grows by one each time and would otherwise change the grid
    // of every attention node. The padded keys are not a special case for the
    // kernels — `GQA_scores` already writes `-3.4e38` at every `t > past + sq`,
    // so softmax gives them exactly zero — and `GQA_out` still sums only `total`
    // of them, so the padding is never read.
    let keys = if kv_stride > total { kv_stride } else { total };
    let q_rot = alloc(b * nh * s * head_size)?;
    let scores = alloc(b * nh * s * keys)?;
    let probs = alloc(b * nh * s * keys)?;
    let out = alloc(b * s * nh * head_size)?;

    // 1) past cache into the head of `present` — unless it is already there.
    //    A caller that bound the cache and passed it back as `past_*` is
    //    handing us the same memory for source and destination, so the copy is
    //    a no-op that would race with itself.
    let aliased =
        |past: &DevTensor<'_>, present: &GpuBuffer| past.buffer().buffer == present.buffer;
    let in_place = aliased(past_k, present_k_buf);
    ensure!(
        in_place == aliased(past_v, present_v_buf),
        "GroupQueryAttention: past_key and past_value must both alias the present cache or neither"
    );
    let past_count = b * kvh * past * head_size;
    if past_count > 0 && !in_place {
        let (grid, gx) = attn_grid(past_count);
        let mut push = Vec::with_capacity(ATTN_PAST_PUSH_BYTES as usize);
        for value in [
            past_count as u32,
            past as u32,
            kv_stride as u32,
            head_size as u32,
            gx,
        ] {
            push.extend_from_slice(&value.to_le_bytes());
        }
        for (src, dst) in [(past_k, present_k_buf), (past_v, present_v_buf)] {
            with_pipeline(
                env.cache(),
                "GQA_past",
                || {
                    ctx.create_pipeline(
                        &compile_wgsl(ATTN_PAST)?,
                        ATTN_PAST_BINDINGS,
                        ATTN_PAST_PUSH_BYTES,
                    )
                },
                |pipe| ctx.stream_dispatch(pipe, &[src.buffer(), dst], &push, grid),
            )?;
        }
    }

    // 2) rotary and head-major layout: query into its own buffer, key and value
    //    appended to the cache at time offset `past`
    let pack = |cache: &KernelCache<'_>,
                x: &GpuBuffer,
                dst: &GpuBuffer,
                n: usize,
                dst_len: usize,
                dst_off: usize,
                rotary: bool|
     -> Result<()> {
        let count = b * s * n * head_size;
        if count == 0 {
            return Ok(());
        }
        let (grid, gx) = attn_grid(count);
        let mut push = Vec::with_capacity(ATTN_PACK_PUSH_BYTES as usize);
        for value in [
            count as u32,
            s as u32,
            n as u32,
            head_size as u32,
            dst_len as u32,
            dst_off as u32,
            past as u32,
            rot_half as u32,
            u32::from(rotary),
            gx,
        ] {
            push.extend_from_slice(&value.to_le_bytes());
        }
        with_pipeline(
            cache,
            "GQA_pack",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(ATTN_PACK)?,
                    ATTN_PACK_BINDINGS,
                    ATTN_PACK_PUSH_BYTES,
                )
            },
            |pipe| ctx.stream_dispatch(pipe, &[x, cos.buffer(), sin.buffer(), dst], &push, grid),
        )
    };
    pack(env.cache(), q.buffer(), &q_rot, nh, s, 0, do_rotary)?;
    pack(
        env.cache(),
        k.buffer(),
        present_k_buf,
        kvh,
        kv_stride,
        past,
        do_rotary,
    )?;
    pack(
        env.cache(),
        v.buffer(),
        present_v_buf,
        kvh,
        kv_stride,
        past,
        false,
    )?;

    let score_count = b * nh * s * keys;
    if score_count > 0 {
        // 3) masked scores
        let (grid, gx) = attn_grid(score_count);
        let mut push = Vec::with_capacity(ATTN_SCORES_PUSH_BYTES as usize);
        for value in [
            score_count as u32,
            nh as u32,
            kvh as u32,
            head_size as u32,
            s as u32,
            total as u32,
            past as u32,
        ] {
            push.extend_from_slice(&value.to_le_bytes());
        }
        push.extend_from_slice(&scale.to_le_bytes());
        push.extend_from_slice(&(window.clamp(-1, i32::MAX as i64) as i32).to_le_bytes());
        push.extend_from_slice(&u32::from(has_bias).to_le_bytes());
        // physical time extent of the K/V buffers; see the note on
        // `shaders::attention`. Equal to `total` while the cache is rebuilt
        // per run, and `max_seq_len` once it is resident.
        push.extend_from_slice(&(kv_stride as u32).to_le_bytes());
        // row extent of the score buffer, which is the padded one
        push.extend_from_slice(&(keys as u32).to_le_bytes());
        push.extend_from_slice(&gx.to_le_bytes());
        with_pipeline(
            env.cache(),
            "GQA_scores",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(ATTN_SCORES)?,
                    ATTN_SCORES_BINDINGS,
                    ATTN_SCORES_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[&q_rot, present_k_buf, bias.buffer(), &scores],
                    &push,
                    grid,
                )
            },
        )?;

        // 4) softmax over the key axis — the row kernel, with its own Pareto key
        let rows = b * nh * s;
        let sgx = rows.clamp(1, 32768) as u32;
        let sgy = (rows as u32).div_ceil(sgx);
        let mut push = Vec::with_capacity(SOFTMAX_PUSH_BYTES as usize);
        // `total` keys to normalize, `keys` apart: the padding of a resident
        // cache is not touched, so a row whose real keys are all masked keeps its
        // probability inside the row
        for value in [total as u32, 1, rows as u32, keys as u32, sgx] {
            push.extend_from_slice(&value.to_le_bytes());
        }
        with_pipeline(
            env.cache(),
            "GQA_softmax",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(SOFTMAX)?,
                    SOFTMAX_BINDINGS,
                    SOFTMAX_PUSH_BYTES,
                )
            },
            |pipe| ctx.stream_dispatch(pipe, &[&scores, &probs], &push, [sgx, sgy, 1]),
        )?;

        // 5) probabilities against the value cache, back to [batch, seq, hidden]
        let out_count = b * s * nh * head_size;
        let (grid, gx) = attn_grid(out_count);
        let mut push = Vec::with_capacity(ATTN_OUT_PUSH_BYTES as usize);
        for value in [
            out_count as u32,
            nh as u32,
            kvh as u32,
            head_size as u32,
            s as u32,
            total as u32,
            kv_stride as u32,
            keys as u32,
            gx,
        ] {
            push.extend_from_slice(&value.to_le_bytes());
        }
        with_pipeline(
            env.cache(),
            "GQA_out",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(ATTN_OUT)?,
                    ATTN_OUT_BINDINGS,
                    ATTN_OUT_PUSH_BYTES,
                )
            },
            |pipe| ctx.stream_dispatch(pipe, &[&probs, present_v_buf, &out], &push, grid),
        )?;
    }

    let device = |shape: Vec<i64>, buf: GpuBuffer| {
        let elem_count = shape.iter().map(|d| (*d).max(0) as usize).product();
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape,
            elem_count,
            buf: BufRef::Owned(buf),
        })
    };
    // the three scratch tensors go back to the pool. Dropping a `GpuBuffer`
    // instead does not free it: `destroy_buffer` is what releases the
    // allocation and what records the free, so a dropped one leaks silently and
    // the VRAM counter never notices. `scores` and `probs` are
    // `num_heads · total` floats per layer, which is why the leak grew with the
    // sequence — measured at +0.19 MB per token on gemma3-1b, worsening as the
    // cache filled.
    for scratch in [q_rot, scores, probs] {
        ctx.recycle_storage_buffer(scratch);
    }
    let (b, s, nh, kvh, total, head_size) = (
        b as i64,
        s as i64,
        nh as i64,
        kvh as i64,
        total as i64,
        head_size as i64,
    );
    env.set(&node.outputs[0], device(vec![b, s, nh * head_size], out));
    // the cache keeps whichever ownership it came with: `Owned` when this call
    // allocated it, `Borrowed` when the caller did — and `Borrowed` is exactly
    // what `release`/`release_owned` step over, so a bound cache survives the run
    for (name, buf) in [(&node.outputs[1], present_k), (&node.outputs[2], present_v)] {
        let shape = vec![b, kvh, total, head_size];
        env.set(
            name,
            Tensor::Device(DevTensor {
                dtype: FLOAT,
                elem_count: shape.iter().map(|d| (*d).max(0) as usize).product(),
                shape,
                buf,
            }),
        );
        // grouped attention against a cache longer than the sequence: the shape
        // is the logical cache and the buffer holds it `kv_stride` apart, so a
        // download of the leading elements would mix row 0's tokens with the
        // padding behind them. The kernels are unaffected — they use the stride
        if padded_rows {
            env.mark_row_padded(name, kv_stride);
        }
    }
    Ok(())
}

/// `Clip` (opset ≥6): bounds from attributes (opset 6) or scalar inputs
/// (opset ≥11); a missing bound does not constrain that side.
fn clip(env: &mut Env, node: &NodeIr) -> Result<()> {
    if env.dtype_of(&node.inputs[0])? != FLOAT {
        // integer Clip (SAM 3 clips its point labels): a few values, on host
        let bound = |index: usize| -> Result<Option<i64>> {
            match node.inputs.get(index) {
                Some(name) if !name.is_empty() => Ok(env.host(name)?.to_i64()?.first().copied()),
                _ => Ok(None),
            }
        };
        let out = host_ops::clip_int(&env.host(&node.inputs[0])?, bound(1)?, bound(2)?)?;
        env.set(&node.outputs[0], Tensor::Host(out));
        return Ok(());
    }
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

    // A QOperator graph pools without leaving the quantized domain: the input
    // is packed u8/i8 and the f32 kernel would read the bytes as floats.
    let x_dtype = env.dtype_of(&x_name)?;
    if x_dtype == UINT8 || x_dtype == INT8 {
        return quantized_max_pool(env, &node, &g, x_dtype, kind);
    }

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

/// `MaxPool` on packed u8/i8, staying in the quantized domain.
///
/// Quantization is monotone, so the maximum of the codes is the code of the
/// maximum: no scale, no zero point, and the result is bit-identical to
/// dequantizing, pooling and quantizing back with the same parameters.
/// Averaging is a different matter — it would need the zero point — and the
/// QOperator family gives it its own operator, so it is refused here.
fn quantized_max_pool(
    env: &mut Env,
    node: &NodeIr,
    g: &ConvGeom,
    dtype: i32,
    kind: PoolKind,
) -> Result<()> {
    ensure!(
        kind == PoolKind::Max,
        "{}: a quantized input is implemented for MaxPool only",
        node.op
    );
    let ctx = env.context();
    let x_name = node.inputs[0].clone();
    env.ensure_device_dtype(&x_name)?;
    let x = env.device(&x_name)?;
    let out = ctx.create_storage_buffer(device_storage_bytes(dtype, g.total)?)?;
    if g.total > 0 {
        let mut push = Vec::with_capacity(MAXPOOL_Q_PUSH_BYTES as usize);
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
            u32::from(dtype == INT8),
        ] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        with_pipeline(
            env.cache(),
            "MaxPool_q",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(MAXPOOL_Q)?,
                    MAXPOOL_Q_BINDINGS,
                    MAXPOOL_Q_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[x.buffer(), &out],
                    &push,
                    [(g.total as u32).div_ceil(4).div_ceil(256), 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype,
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

/// `ArgMax` on one axis, on the device, as an int64 output.
///
/// Returned as a tensor and not as a number because it is the ONNX op: a graph
/// that ends in `ArgMax` gets the same kernel a generation loop asks for through
/// [`argmax_of`].
fn argmax(env: &mut Env, node: &NodeIr) -> Result<()> {
    arg_extreme(env, node, false)
}

fn argmin(env: &mut Env, node: &NodeIr) -> Result<()> {
    arg_extreme(env, node, true)
}

fn arg_extreme(env: &mut Env, node: &NodeIr, minimum: bool) -> Result<()> {
    let x_name = node.inputs[0].clone();
    let x_shape = env.shape_of(&x_name)?;
    let rank = x_shape.len() as i64;
    let raw = node
        .attrs
        .get("axis")
        .and_then(AttrValue::as_i64)
        .unwrap_or(0);
    let axis = if raw < 0 { raw + rank } else { raw };
    ensure!(
        (0..rank).contains(&axis),
        "{}: axis {axis} out of rank {rank}",
        if minimum { "ArgMin" } else { "ArgMax" }
    );
    let keepdims = node
        .attrs
        .get("keepdims")
        .and_then(AttrValue::as_i64)
        .unwrap_or(1)
        != 0;
    let out = arg_extreme_axis(env, &x_name, axis as usize, keepdims, minimum)?;
    env.set(&node.outputs[0], out);
    Ok(())
}

/// The index of the largest element along the **last** axis of a value the
/// current run produced, computed where the value already is.
///
/// This is what a decode step needs and the graph does not have: an exported
/// decoder emits logits, so sampling them means either downloading a row of
/// 151936 floats to keep one index, or enqueueing this into the same command
/// buffer the step is still building. Only the indices cross the bus.
pub fn argmax_of(env: &mut Env<'_, '_>, name: &str) -> crate::Result<Vec<i64>> {
    argmax_last_axis(env, name).map_err(|e| crate::Error::Backend(format!("{e:#}")))
}

fn argmax_last_axis(env: &mut Env<'_, '_>, name: &str) -> Result<Vec<i64>> {
    let shape = env.shape_of(name)?;
    ensure!(!shape.is_empty(), "{name} is a scalar: it has no axis");
    let out = arg_extreme_axis(env, name, shape.len() - 1, false, false)?;
    let key = format!("{name}#argmax");
    env.set(&key, out);
    Ok(env.host(&key)?.to_i64()?)
}

/// Shared body: two dispatches, one scratch pair, an int64 result.
///
/// The input must be f32 on the device. A non-f32 axis (an int64 mask, say) goes
/// host-side, like the value reductions: the support check looks at the node and
/// cannot see the dtype, so every dtype needs a path that works.
fn arg_extreme_axis(
    env: &mut Env,
    x_name: &str,
    axis: usize,
    keepdims: bool,
    minimum: bool,
) -> Result<Tensor<'static>> {
    let x_shape = env.shape_of(x_name)?;
    let c = x_shape[axis].max(0) as usize;
    let inner: usize = x_shape[axis + 1..].iter().product::<i64>().max(1) as usize;
    let mut out_shape = x_shape.clone();
    if keepdims {
        out_shape[axis] = 1;
    } else {
        out_shape.remove(axis);
    }
    let rows: usize = out_shape.iter().map(|d| (*d).max(0) as usize).product();

    if env.dtype_of(x_name)? != FLOAT {
        let x = env.host(x_name)?.to_f32()?;
        let mut idx = Vec::with_capacity(rows);
        for r in 0..rows {
            let j = r % inner;
            let base = (r / inner) * c * inner + j;
            let mut best = if minimum {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
            let mut at = 0i64;
            for k in 0..c {
                let candidate = x[base + k * inner];
                if (minimum && candidate < best) || (!minimum && candidate > best) {
                    best = candidate;
                    at = k as i64;
                }
            }
            idx.push(at);
        }
        return Ok(Tensor::Host(HostTensor::from_i64(out_shape, &idx)));
    }

    let ctx = env.context();
    env.ensure_device(x_name)?;
    let x = env.device(x_name)?;
    let out = ctx.create_storage_buffer(device_storage_bytes(INT64, rows)?)?;
    if rows > 0 && c > 0 {
        let splits = argmax_splits(rows, c);
        let scratch = |elems: usize| ctx.create_storage_buffer(device_storage_bytes(FLOAT, elems)?);
        let best = scratch(rows * splits)?;
        let best_idx = scratch(rows * splits)?;

        let grid = |wgs: usize| {
            // 2D: `rows · splits` passes the 65535 workgroups-per-axis limit as
            // soon as the batch is not tiny
            let gx = (wgs as u32).clamp(1, 32768);
            ([gx, (wgs as u32).div_ceil(gx), 1], gx)
        };
        let ([gx, gy, _], stride) = grid(rows * splits);
        let mut push = Vec::with_capacity(ARGMAX_PARTIAL_PUSH_BYTES as usize);
        for v in [c as u32, inner as u32, rows as u32, splits as u32, stride] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        let partial_source = if minimum {
            ARGMAX_PARTIAL
                .replace("-3.4028235e38", "3.4028235e38")
                .replace("v > bv", "v < bv")
                .replace("sv[o] > sv[lid.x]", "sv[o] < sv[lid.x]")
        } else {
            ARGMAX_PARTIAL.to_string()
        };
        let final_source = if minimum {
            ARGMAX_FINAL
                .replace("-3.4028235e38", "3.4028235e38")
                .replace("v > bv", "v < bv")
                .replace("sv[o] > sv[lid.x]", "sv[o] < sv[lid.x]")
        } else {
            ARGMAX_FINAL.to_string()
        };
        let partial_key = if minimum {
            "ArgMin_partial"
        } else {
            "ArgMax_partial"
        };
        let final_key = if minimum {
            "ArgMin_final"
        } else {
            "ArgMax_final"
        };
        with_pipeline(
            env.cache(),
            partial_key,
            || {
                ctx.create_pipeline(
                    &compile_wgsl(&partial_source)?,
                    ARGMAX_BINDINGS,
                    ARGMAX_PARTIAL_PUSH_BYTES,
                )
            },
            |pipe| ctx.stream_dispatch(pipe, &[x.buffer(), &best, &best_idx], &push, [gx, gy, 1]),
        )?;

        let ([fx, fy, _], fstride) = grid(rows);
        let mut push = Vec::with_capacity(ARGMAX_FINAL_PUSH_BYTES as usize);
        for v in [rows as u32, splits as u32, fstride] {
            push.extend_from_slice(&v.to_le_bytes());
        }
        with_pipeline(
            env.cache(),
            final_key,
            || {
                ctx.create_pipeline(
                    &compile_wgsl(&final_source)?,
                    ARGMAX_BINDINGS,
                    ARGMAX_FINAL_PUSH_BYTES,
                )
            },
            |pipe| ctx.stream_dispatch(pipe, &[&best, &best_idx, &out], &push, [fx, fy, 1]),
        )?;
        // back to the pool, not dropped: `destroy_buffer` is what frees an
        // allocation, and a decode loop calls this once per token
        ctx.recycle_storage_buffer(best);
        ctx.recycle_storage_buffer(best_idx);
    }
    Ok(Tensor::Device(DevTensor {
        dtype: INT64,
        shape: out_shape,
        elem_count: rows,
        buf: BufRef::Owned(out),
    }))
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
        let selected = matrix_tactic(env, node, &[m, n], &["gemm64", "gemm16"])?;
        let (label, source, tile) = if selected.as_deref() == Some("gemm16") {
            ("Gemm16", crate::shaders::matrix::GEMM16, 16)
        } else {
            ("Gemm", GEMM, GEMM_TILE_SIZE)
        };
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
            label,
            || ctx.create_pipeline(&compile_wgsl(source)?, GEMM_BINDINGS, GEMM_PUSH_BYTES),
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[a.buffer(), b.buffer(), c_buf, &out],
                    &push,
                    [(n as u32).div_ceil(tile), (m as u32).div_ceil(tile), 1],
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

/// The five nodes of an exact GELU that `rewrite::fuse_decomposed_gelu`
/// replaced, `Div(x, c1) → Erf → Add(c2) → Mul(x) → Mul(c3)` or, with
/// `factor_first`, `… → Mul(c3) → Mul(x)`, in one pass with the same rounding
/// (see `DECOMPOSED_GELU`).
///
/// The `Div` alone may run on host: `elementwise_binary` keeps two small host
/// operands there. That case, and a non-f32 operand, replays the five steps
/// rather than move the division to the GPU.
fn decomposed_gelu(env: &mut Env, node: &NodeIr) -> Result<()> {
    let Some(gelu) = gelu_push(env, node)? else {
        return decomposed_gelu_unfused(env, node);
    };

    let ctx = env.context();
    let x = &node.inputs[0];
    env.ensure_device(x)?;
    let x = env.device(x)?;
    let (shape, elem_count) = (x.shape.clone(), x.elem_count);
    let out = ctx.create_storage_buffer((elem_count.max(1) * 4) as u64)?;
    if elem_count > 0 {
        let mut push = (elem_count as u32).to_le_bytes().to_vec();
        push.extend_from_slice(&gelu);
        with_pipeline(
            env.cache(),
            "DecomposedGelu",
            || {
                let src = format!("{UNARY_HELPERS_ERF}{DECOMPOSED_GELU_FN}{DECOMPOSED_GELU}");
                ctx.create_pipeline(
                    &compile_wgsl(&src)?,
                    DECOMPOSED_GELU_BINDINGS,
                    DECOMPOSED_GELU_PUSH_BYTES,
                )
            },
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

/// Whether the fused GELU scales `erf + c2` by `c3` before gating by `x`.
fn gelu_factor_first(node: &NodeIr) -> bool {
    node.attrs.get("factor_first").and_then(AttrValue::as_i64) == Some(1)
}

/// The push constants `DECOMPOSED_GELU_FN` reads after the shader's own —
/// `divisor`, `offset`, `factor`, `factor_first` — or `None` when the GELU of
/// `node` (inputs `x, c1, c2, c3`) must replay its five steps: a non-f32
/// operand, a constant that is not one value, or an `x` and `c1` small and on
/// host, where `elementwise_binary` keeps the `Div`.
fn gelu_push(env: &Env, node: &NodeIr) -> Result<Option<Vec<u8>>> {
    let [x, c1, c2, c3] = [0, 1, 2, 3].map(|i| node.inputs[i].as_str());
    for name in [x, c1, c2, c3] {
        if env.dtype_of(name)? != FLOAT {
            return Ok(None);
        }
    }
    if env.host_resident_small(x, Env::SMALL_HOST_BYTES)
        && env.host_resident_small(c1, Env::SMALL_HOST_BYTES)
    {
        return Ok(None);
    }
    let mut push = Vec::with_capacity(16);
    for name in [c1, c2, c3] {
        match env.host(name)?.to_f32()?.as_slice() {
            [only] => push.extend_from_slice(&only.to_le_bytes()),
            _ => return Ok(None),
        }
    }
    push.extend_from_slice(&u32::from(gelu_factor_first(node)).to_le_bytes());
    Ok(Some(push))
}

/// `DynamicQuantizeLinear(DecomposedGelu(x))`, which
/// `rewrite::fuse_gelu_quantize` made one node: the min/max and the quantize
/// passes each compute the GELU from `x` again, with the same function as
/// `DecomposedGelu`, so the f32 GELU is never stored and every byte matches
/// the two nodes. What `DecomposedGelu` would replay, and an empty `x`, run as
/// the two nodes.
fn decomposed_gelu_quantize(env: &mut Env, node: &NodeIr) -> Result<()> {
    let gelu = gelu_push(env, node)?;
    let x = &node.inputs[0];
    if let Some(gelu) = gelu
        && env.shape_of(x)?.iter().product::<i64>() > 0
    {
        env.ensure_device(x)?;
        return quantize_on_device(env, x, &node.outputs, Some(&gelu));
    }
    let activation = format!("{}/gelu", node.outputs[0]);
    let mut gelu = lowered(
        node,
        "DecomposedGelu",
        &node.inputs.iter().map(String::as_str).collect::<Vec<_>>(),
        &[&activation],
    );
    gelu.domain = crate::rewrite::REWRITE_DOMAIN.to_owned();
    decomposed_gelu(env, &gelu)?;
    let outputs: Vec<&str> = node.outputs.iter().map(String::as_str).collect();
    let quantize = lowered(node, "DynamicQuantizeLinear", &[&activation], &outputs);
    dynamic_quantize(env, &quantize)?;
    env.release(&activation);
    Ok(())
}

/// The five nodes `rewrite::fuse_decomposed_gelu` replaced, run as they were.
fn decomposed_gelu_unfused(env: &mut Env, node: &NodeIr) -> Result<()> {
    let out = &node.outputs[0];
    let [scaled, erf, shifted, product] =
        ["scaled", "erf", "shifted", "product"].map(|step| format!("{out}/{step}"));
    let i = &node.inputs;
    // the operand the first product multiplies `erf + c2` by, then the last's
    let (first, last) = if gelu_factor_first(node) {
        (i[3].as_str(), i[0].as_str())
    } else {
        (i[0].as_str(), i[3].as_str())
    };
    let steps = [
        ("Div", vec![i[0].as_str(), i[1].as_str()], scaled.as_str()),
        ("Erf", vec![scaled.as_str()], erf.as_str()),
        ("Add", vec![erf.as_str(), i[2].as_str()], shifted.as_str()),
        ("Mul", vec![first, shifted.as_str()], product.as_str()),
        ("Mul", vec![product.as_str(), last], out.as_str()),
    ];
    let mut previous: Option<&str> = None;
    for (op, inputs, output) in steps {
        exec_dispatch(env, &lowered(node, op, &inputs, &[output]))?;
        if let Some(done) = previous {
            env.release(done);
        }
        previous = Some(output);
    }
    Ok(())
}

/// The interleaved rotary embedding `rewrite::fuse_interleaved_rope`
/// replaced, `x · cos + rot(x) · sin`, with the Transpose before it, the
/// Transpose after it and the scale when the rewrite folded them, in one pass
/// with the same roundings (see `glsl/interleaved_rope.comp`).
///
/// What the rewrite could not see without shapes is checked here: f32
/// operands, `cos` and `sin` broadcasting as `[S, D]`, an even `D`. Anything
/// else, and an input small enough that the original `Mul` would have run on
/// host, replays the original steps instead.
fn interleaved_rope(env: &mut Env, node: &NodeIr) -> Result<()> {
    let [x, cos, sin] = [0, 1, 2].map(|i| node.inputs[i].as_str());
    let scale = node.inputs.get(3).map(String::as_str);
    let flag = |name: &str| node.attrs.get(name).and_then(AttrValue::as_i64) == Some(1);
    let (bshd, transpose) = (flag("bshd"), flag("transpose_output"));
    let shape = env.shape_of(x)?;
    let [b, s, h, d] = match (bshd, shape.as_slice()) {
        (true, &[b, s, h, d]) | (false, &[b, h, s, d]) => [b, s, h, d],
        _ => bail!(
            "InterleavedRope: input of rank {} instead of 4",
            shape.len()
        ),
    };
    // the dynamo form's reshapes have constant shapes: they assumed this one
    if let Some(assumed) = node.attrs.get("shape").and_then(AttrValue::as_ints)
        && assumed != [b, h, s, d]
    {
        bail!(
            "InterleavedRope: input [B, H, S, D] = {:?}, the graph's reshapes assume {assumed:?}",
            [b, h, s, d]
        );
    }

    let mut all_f32 = true;
    for name in [x, cos, sin].into_iter().chain(scale) {
        all_f32 &= env.dtype_of(name)? == FLOAT;
    }
    let mut tables = true;
    for name in [cos, sin] {
        let table = env.shape_of(name)?;
        let (lead, last) = table.split_at(table.len().saturating_sub(2));
        tables &= table.len() <= 4 && last == [s, d] && lead.iter().all(|&dim| dim == 1);
    }
    let n = b * h * s * d;
    let factor = match scale {
        Some(name) => match env.host(name)?.to_f32()?.as_slice() {
            [only] => Some(*only),
            _ => None,
        },
        None => Some(1.0),
    };
    let small_host = env.host_resident_small(x, Env::SMALL_HOST_BYTES);
    let limits = env.context().compute_limits().max_workgroup_count;
    let groups = interleaved_rope_groups((b * h) as usize, s as usize, d as usize, transpose)
        .filter(|groups| groups.iter().zip(limits).all(|(&g, max)| g <= max));
    let (Some(factor), true, true, false, 0, Ok(n32), Some(groups)) = (
        factor,
        all_f32,
        tables,
        small_host,
        d % 2,
        u32::try_from(n),
        groups,
    ) else {
        return interleaved_rope_unfused(env, node, [b, h, s, d], bshd, transpose);
    };

    for name in [x, cos, sin] {
        env.ensure_device(name)?;
    }
    let ctx = env.context();
    let (xt, ct, st) = (env.device(x)?, env.device(cos)?, env.device(sin)?);
    let out = ctx.create_storage_buffer((n.max(1) * 4) as u64)?;
    if n > 0 {
        let mut flags = 0;
        for (on, bit) in [
            (bshd, INTERLEAVED_ROPE_BSHD),
            (transpose, INTERLEAVED_ROPE_TRANSPOSE),
            (scale.is_some(), INTERLEAVED_ROPE_SCALE),
        ] {
            if on {
                flags |= bit;
            }
        }
        let mut push = Vec::with_capacity(INTERLEAVED_ROPE_PUSH_BYTES as usize);
        for value in [n32, h as u32, s as u32, d as u32, flags] {
            push.extend_from_slice(&value.to_le_bytes());
        }
        push.extend_from_slice(&factor.to_le_bytes());
        with_pipeline(
            env.cache(),
            "InterleavedRope",
            || {
                ctx.create_pipeline(
                    &interleaved_rope_spirv(),
                    INTERLEAVED_ROPE_BINDINGS,
                    INTERLEAVED_ROPE_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[xt.buffer(), ct.buffer(), st.buffer(), &out],
                    &push,
                    groups,
                )
            },
        )?;
    }
    let shape = if transpose {
        vec![b, h, d, s]
    } else {
        vec![b, h, s, d]
    };
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape,
            elem_count: n as usize,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `Mul(x, scale) → Add(·, bias) → Relu`, the node `rewrite::fuse_mul_add_relu`
/// made, in one pass with the rounding of the three kernels
/// (`glsl/mul_add_relu.comp`). Needs f32 operands, an `x` of rank ≥ 2 on the
/// device and `scale`, `bias` one value or one per channel (axis 1); anything
/// else replays the three steps.
fn mul_add_relu(env: &mut Env, node: &NodeIr) -> Result<()> {
    let [x, scale, bias] = [0, 1, 2].map(|i| node.inputs[i].as_str());
    let shape = env.shape_of(x)?;
    let rank = shape.len();
    let mut all_f32 = true;
    for name in [x, scale, bias] {
        all_f32 &= env.dtype_of(name)? == FLOAT;
    }
    let channels = shape.get(1).copied().unwrap_or(0);
    let step = |name: &str| -> Result<Option<u32>> {
        Ok(channel_step(&env.shape_of(name)?, channels, rank).map(|step| step.min(1)))
    };
    let (scale_step, bias_step) = (step(scale)?, step(bias)?);
    let n: i64 = shape.iter().product();
    let plane: i64 = shape.iter().skip(2).product();
    let small_host = env.host_resident_small(x, Env::SMALL_HOST_BYTES);
    // below 2^31 the shader's grid-stride index cannot wrap
    let n32 = u32::try_from(n).ok().filter(|&n| n <= i32::MAX as u32);
    let (true, Some(scale_step), Some(bias_step), false, true, Some(n32), Ok(plane32), Ok(c32)) = (
        all_f32,
        scale_step,
        bias_step,
        small_host,
        rank >= 2,
        n32,
        u32::try_from(plane),
        u32::try_from(channels),
    ) else {
        return mul_add_relu_unfused(env, node);
    };

    for name in [x, scale, bias] {
        env.ensure_device(name)?;
    }
    let ctx = env.context();
    let (xt, st, bt) = (env.device(x)?, env.device(scale)?, env.device(bias)?);
    let out = ctx.create_storage_buffer((n.max(1) * 4) as u64)?;
    if n > 0 {
        let mut push = Vec::with_capacity(MUL_ADD_RELU_PUSH_BYTES as usize);
        for value in [n32, plane32, c32, scale_step, bias_step] {
            push.extend_from_slice(&value.to_le_bytes());
        }
        let groups = n32
            .div_ceil(256)
            .min(ctx.compute_limits().max_workgroup_count[0]);
        with_pipeline(
            env.cache(),
            "MulAddRelu",
            || {
                ctx.create_pipeline(
                    &mul_add_relu_spirv(),
                    MUL_ADD_RELU_BINDINGS,
                    MUL_ADD_RELU_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[xt.buffer(), st.buffer(), bt.buffer(), &out],
                    &push,
                    [groups, 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape,
            elem_count: n as usize,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// The three nodes `rewrite::fuse_mul_add_relu` replaced, run as they were.
fn mul_add_relu_unfused(env: &mut Env, node: &NodeIr) -> Result<()> {
    let scaled = format!("{}/scaled", node.outputs[0]);
    let shifted = format!("{}/shifted", node.outputs[0]);
    let i = &node.inputs;
    exec_dispatch(env, &lowered(node, "Mul", &[&i[0], &i[1]], &[&scaled]))?;
    exec_dispatch(env, &lowered(node, "Add", &[&scaled, &i[2]], &[&shifted]))?;
    env.release(&scaled);
    exec_dispatch(
        env,
        &lowered(node, "Relu", &[&shifted], &[&node.outputs[0]]),
    )?;
    env.release(&shifted);
    Ok(())
}

/// `numpy.packbits(x > 0, axis = -1)`, the node `rewrite::fuse_pack_positive_bits`
/// made of `Reshape → Greater → Cast → Mul → ReduceSum → Cast`. Input 1 is the
/// reshape target `[..., W/8, 8]`; the output is that shape without its last
/// axis, as u8. On the GPU when `x` is there, else on host: both pack the
/// flattened `x` eight at a time, which is what the reshape groups.
fn pack_positive_bits(env: &mut Env, node: &NodeIr) -> Result<()> {
    let x = node.inputs[0].as_str();
    ensure!(
        env.dtype_of(x)? == FLOAT,
        "PackPositiveBits: input of type {} instead of f32",
        env.dtype_of(x)?
    );
    let target = env.host(&node.inputs[1])?.to_i64()?;
    let grouped = resolve_reshape(&env.shape_of(x)?, &target)?;
    let Some((&8, shape)) = grouped.split_last() else {
        bail!("PackPositiveBits: last axis of {grouped:?} is not 8");
    };
    let shape = shape.to_vec();
    let bytes = shape.iter().product::<i64>() as usize;

    let on_device = env.on_device(x);
    let Ok(bytes32) = u32::try_from(bytes) else {
        bail!("PackPositiveBits: {bytes} output bytes exceed u32 indexing");
    };
    if !on_device {
        let values = env.host(x)?.to_f32()?;
        let data = values
            .chunks_exact(8)
            .map(|group| {
                group
                    .iter()
                    .fold(0u8, |byte, &v| (byte << 1) | u8::from(v > 0.0))
            })
            .collect();
        env.set(
            &node.outputs[0],
            Tensor::Host(HostTensor::new(UINT8, shape, data)),
        );
        return Ok(());
    }

    let ctx = env.context();
    let xt = env.device(x)?;
    let out = ctx.create_storage_buffer(device_storage_bytes(UINT8, bytes)?)?;
    if bytes > 0 {
        let words = bytes32.div_ceil(4);
        let max_groups = ctx.compute_limits().max_workgroup_count[0];
        let groups = words.div_ceil(256).min(max_groups);
        with_pipeline(
            env.cache(),
            "PackPositiveBits",
            || {
                ctx.create_pipeline(
                    &compile_wgsl(PACK_POSITIVE_BITS)?,
                    PACK_POSITIVE_BITS_BINDINGS,
                    PACK_POSITIVE_BITS_PUSH_BYTES,
                )
            },
            |pipe| {
                ctx.stream_dispatch(
                    pipe,
                    &[xt.buffer(), &out],
                    &bytes32.to_le_bytes(),
                    [groups, 1, 1],
                )
            },
        )?;
    }
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: UINT8,
            shape,
            elem_count: bytes,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// `Add(a, b) → Reshape → Transpose → Pad`, the node
/// `rewrite::fuse_add_transpose_pad` writes: inputs `a`, `b`, the reshape
/// target, the pads and the optional pad value, the permutation as attribute.
/// One `ADD_TRANSPOSE_PAD` pass when both operands are f32 for the GPU, the
/// reshape only merges axes, no pad is negative or falls on a merged axis, and
/// at most four axes remain once unit axes are dropped and neighbours merged;
/// otherwise the four original steps.
fn add_transpose_pad(env: &mut Env, node: &NodeIr) -> Result<()> {
    match add_transpose_pad_plan(env, node)? {
        Some(plan) => add_transpose_pad_fused(env, node, plan),
        None => add_transpose_pad_unfused(env, node),
    }
}

/// One axis of the padded output as `ADD_TRANSPOSE_PAD` walks it.
#[derive(Clone, Copy)]
struct PaddedAxis {
    extent: u32,
    begin: u32,
    interior: u32,
    sa: u32,
    sb: u32,
}

struct AddTransposePadPlan {
    /// outer first, at most four
    axes: Vec<PaddedAxis>,
    shape: Vec<i64>,
    n: u32,
    value: f32,
}

fn add_transpose_pad_plan(env: &mut Env, node: &NodeIr) -> Result<Option<AddTransposePadPlan>> {
    let (a, b) = (&node.inputs[0], &node.inputs[1]);
    // the rule of `elementwise_binary`: small host operands are shape math
    let small_host = env.host_resident_small(a, Env::SMALL_HOST_BYTES)
        && env.host_resident_small(b, Env::SMALL_HOST_BYTES);
    if env.dtype_of(a)? != FLOAT || env.dtype_of(b)? != FLOAT || small_host {
        return Ok(None);
    }
    let bc = broadcast(&env.shape_of(a)?, &env.shape_of(b)?)?;
    let sum = &bc.out_shape;
    let reshaped = resolve_reshape(sum, &env.host(&node.inputs[2])?.to_i64()?)?;
    let rank = reshaped.len();
    let perm: Vec<usize> = match node.attrs.get("perm").and_then(AttrValue::as_ints) {
        Some(perm) => perm.iter().map(|&p| p as usize).collect(),
        None => (0..rank).rev().collect(),
    };
    let mut sorted = perm.clone();
    sorted.sort_unstable();
    let pads = env.host(&node.inputs[3])?.to_i64()?;
    if !sorted.iter().copied().eq(0..rank)
        || pads.len() != 2 * rank
        || pads.iter().any(|&p| p < 0)
        || sum.contains(&0)
    {
        return Ok(None);
    }

    // the axes of the sum each reshaped axis merges, unit axes left out
    let mut groups: Vec<Vec<usize>> = Vec::with_capacity(rank);
    let mut next = 0;
    for &dim in &reshaped {
        let (mut group, mut size) = (Vec::new(), 1);
        while size < dim && next < sum.len() {
            if sum[next] != 1 {
                group.push(next);
            }
            size *= sum[next];
            next += 1;
        }
        if size != dim {
            return Ok(None);
        }
        groups.push(group);
    }
    if sum[next..].iter().any(|&d| d != 1) {
        return Ok(None);
    }

    let shape: Vec<i64> = (0..rank)
        .map(|k| reshaped[perm[k]] + pads[k] + pads[rank + k])
        .collect();
    let Ok(n) = u32::try_from(shape.iter().product::<i64>()) else {
        return Ok(None);
    };
    let mut axes: Vec<PaddedAxis> = Vec::new();
    for (k, &p) in perm.iter().enumerate() {
        let (begin, extent) = (pads[k] as u32, shape[k] as u32);
        match groups[p].as_slice() {
            [] => axes.push(PaddedAxis {
                extent,
                begin,
                interior: 1,
                sa: 0,
                sb: 0,
            }),
            [d] => axes.push(PaddedAxis {
                extent,
                begin,
                interior: sum[*d] as u32,
                sa: bc.a_strides[*d],
                sb: bc.b_strides[*d],
            }),
            merged if extent as i64 == reshaped[p] => axes.extend(merged.iter().map(|&d| {
                let size = sum[d] as u32;
                PaddedAxis {
                    extent: size,
                    begin: 0,
                    interior: size,
                    sa: bc.a_strides[d],
                    sb: bc.b_strides[d],
                }
            })),
            _ => return Ok(None),
        }
    }
    // neighbours both operands walk contiguously become one axis
    let mut merged: Vec<PaddedAxis> = Vec::new();
    for axis in axes.into_iter().filter(|axis| axis.extent > 1) {
        let plain = |axis: &PaddedAxis| axis.begin == 0 && axis.extent == axis.interior;
        match merged.last_mut() {
            Some(outer)
                if plain(outer)
                    && plain(&axis)
                    && axis.sa.checked_mul(axis.extent) == Some(outer.sa)
                    && axis.sb.checked_mul(axis.extent) == Some(outer.sb) =>
            {
                outer.extent *= axis.extent;
                outer.interior = outer.extent;
                outer.sa = axis.sa;
                outer.sb = axis.sb;
            }
            _ => merged.push(axis),
        }
    }
    if merged.len() > 4 {
        return Ok(None);
    }
    let value = match node.inputs.get(4).filter(|v| !v.is_empty()) {
        Some(v) => *env.host(v)?.to_f32()?.first().unwrap_or(&0.0),
        None => 0.0,
    };
    Ok(Some(AddTransposePadPlan {
        axes: merged,
        shape,
        n,
        value,
    }))
}

fn add_transpose_pad_fused(env: &mut Env, node: &NodeIr, plan: AddTransposePadPlan) -> Result<()> {
    let (a, b) = (&node.inputs[0], &node.inputs[1]);
    env.ensure_device(a)?;
    env.ensure_device(b)?;
    let ctx = env.context();
    let (ta, tb) = (env.device(a)?, env.device(b)?);
    let out = ctx.create_storage_buffer(device_storage_bytes(FLOAT, plan.n as usize)?)?;
    let unit = PaddedAxis {
        extent: 1,
        begin: 0,
        interior: 1,
        sa: 0,
        sb: 0,
    };
    let mut axes = vec![unit; 4 - plan.axes.len()];
    axes.extend(plan.axes);
    let mut push = Vec::with_capacity(ADD_TRANSPOSE_PAD_PUSH_BYTES as usize);
    push.extend_from_slice(&plan.n.to_le_bytes());
    push.extend_from_slice(&plan.value.to_le_bytes());
    push.extend_from_slice(&[0; 8]);
    let fields: [fn(&PaddedAxis) -> u32; 5] = [
        |axis| axis.extent,
        |axis| axis.begin,
        |axis| axis.interior,
        |axis| axis.sa,
        |axis| axis.sb,
    ];
    for field in fields {
        for axis in &axes {
            push.extend_from_slice(&field(axis).to_le_bytes());
        }
    }
    let max_groups = ctx.compute_limits().max_workgroup_count[0];
    let groups = plan.n.div_ceil(256).min(max_groups);
    with_pipeline(
        env.cache(),
        "AddTransposePad",
        || {
            ctx.create_pipeline(
                &compile_wgsl(ADD_TRANSPOSE_PAD)?,
                ADD_TRANSPOSE_PAD_BINDINGS,
                ADD_TRANSPOSE_PAD_PUSH_BYTES,
            )
        },
        |pipe| {
            ctx.stream_dispatch(
                pipe,
                &[ta.buffer(), tb.buffer(), &out],
                &push,
                [groups, 1, 1],
            )
        },
    )?;
    env.set(
        &node.outputs[0],
        Tensor::Device(DevTensor {
            dtype: FLOAT,
            shape: plan.shape,
            elem_count: plan.n as usize,
            buf: BufRef::Owned(out),
        }),
    );
    Ok(())
}

/// The steps `rewrite::fuse_add_transpose_pad` replaced, run as the graph
/// wrote them.
fn add_transpose_pad_unfused(env: &mut Env, node: &NodeIr) -> Result<()> {
    let out = node.outputs[0].as_str();
    let sum = format!("{out}/sum");
    let reshaped = format!("{out}/reshaped");
    let transposed = format!("{out}/transposed");
    let i: Vec<&str> = node.inputs.iter().map(String::as_str).collect();
    exec_dispatch(env, &lowered(node, "Add", &[i[0], i[1]], &[&sum]))?;
    exec_dispatch(env, &lowered(node, "Reshape", &[&sum, i[2]], &[&reshaped]))?;
    env.release(&sum);
    exec_dispatch(
        env,
        &lowered(node, "Transpose", &[&reshaped], &[&transposed]),
    )?;
    env.release(&reshaped);
    let mut pad = vec![transposed.as_str()];
    pad.extend(&i[3..]);
    exec_dispatch(env, &lowered(node, "Pad", &pad, &[out]))?;
    env.release(&transposed);
    Ok(())
}

/// The steps `rewrite::fuse_interleaved_rope` replaced, run as the graph wrote
/// them. Squeeze and Unsqueeze are left out: they only rename a shape. `dims`
/// is `[B, H, S, D]`.
fn interleaved_rope_unfused(
    env: &mut Env,
    node: &NodeIr,
    dims: [i64; 4],
    bshd: bool,
    transpose: bool,
) -> Result<()> {
    let [b, h, s, _] = dims;
    let out = node.outputs[0].clone();
    let name = |step: &str| format!("{out}/{step}");
    let mut temps: Vec<String> = Vec::new();
    for (step, values) in [
        ("pairs_shape", vec![b, h, s, -1, 2]),
        ("halves", vec![1, 1]),
        ("rot_shape", vec![b, h, s, -1]),
    ] {
        let tensor = HostTensor::from_i64(vec![values.len() as i64], &values);
        env.set(&name(step), Tensor::Host(tensor));
        temps.push(name(step));
    }
    let run = |env: &mut Env,
               op: &str,
               inputs: &[&str],
               outputs: &[&str],
               attrs: Vec<(&str, AttrValue)>|
     -> Result<()> {
        let mut step = lowered(node, op, inputs, outputs);
        step.attrs = attrs
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect();
        exec_dispatch(env, &step)
    };
    let i = &node.inputs;
    let t = if bshd {
        let t = name("t");
        let perm = AttrValue::Ints(vec![0, 2, 1, 3]);
        run(env, "Transpose", &[&i[0]], &[&t], vec![("perm", perm)])?;
        temps.push(t.clone());
        t
    } else {
        i[0].clone()
    };
    let last_axis = || vec![("axis", AttrValue::Int(-1))];
    let [
        direct,
        pairs,
        even,
        odd,
        neg,
        swapped,
        rot,
        rotated,
        sum,
        transposed,
    ] = [
        "direct",
        "pairs",
        "even",
        "odd",
        "neg",
        "swapped",
        "rot",
        "rotated",
        "sum",
        "transposed",
    ]
    .map(name);
    run(env, "Mul", &[&t, &i[1]], &[&direct], vec![])?;
    run(
        env,
        "Reshape",
        &[&t, &name("pairs_shape")],
        &[&pairs],
        vec![],
    )?;
    run(
        env,
        "Split",
        &[&pairs, &name("halves")],
        &[&even, &odd],
        last_axis(),
    )?;
    run(env, "Neg", &[&odd], &[&neg], vec![])?;
    run(env, "Concat", &[&neg, &even], &[&swapped], last_axis())?;
    run(
        env,
        "Reshape",
        &[&swapped, &name("rot_shape")],
        &[&rot],
        vec![],
    )?;
    run(env, "Mul", &[&rot, &i[2]], &[&rotated], vec![])?;
    temps.extend([
        direct.clone(),
        pairs,
        even,
        odd,
        neg,
        swapped,
        rot,
        rotated.clone(),
    ]);

    // each remaining step writes the output when it is the last one
    let scale = i.get(3);
    let sum = if transpose || scale.is_some() {
        sum
    } else {
        out.clone()
    };
    run(env, "Add", &[&direct, &rotated], &[&sum], vec![])?;
    let mut value = sum;
    if transpose {
        let target = if scale.is_some() {
            transposed
        } else {
            out.clone()
        };
        let perm = AttrValue::Ints(vec![0, 1, 3, 2]);
        run(
            env,
            "Transpose",
            &[&value],
            &[&target],
            vec![("perm", perm)],
        )?;
        temps.push(std::mem::replace(&mut value, target));
    }
    if let Some(scale) = scale {
        run(env, "Mul", &[&value, scale], &[&out], vec![])?;
        temps.push(value);
    }
    for temp in &temps {
        env.release(temp);
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::collapse_permutation;

    #[test]
    fn window_partition_collapses_to_rows() {
        // [B, a, u, b, v, C] → [B, a, b, u, v, C]: B is dropped, (a) stays,
        // u and b swap, (v, C) merge into rows
        assert_eq!(
            collapse_permutation(&[1, 3, 24, 3, 24, 1024], &[0, 1, 3, 2, 4, 5]),
            (vec![3, 24, 3, 24 * 1024], vec![0, 2, 1, 3])
        );
    }

    #[test]
    fn adjacent_axes_merge_and_unit_axes_drop() {
        assert_eq!(
            collapse_permutation(&[2, 3, 4, 5], &[2, 3, 0, 1]),
            (vec![6, 20], vec![1, 0])
        );
        assert_eq!(
            collapse_permutation(&[1, 4, 1, 6], &[2, 3, 0, 1]),
            (vec![4, 6], vec![1, 0])
        );
        assert_eq!(
            collapse_permutation(&[2, 3, 4], &[2, 1, 0]),
            (vec![2, 3, 4], vec![2, 1, 0])
        );
    }

    #[test]
    fn nothing_moves_collapses_to_one_axis_or_none() {
        assert_eq!(collapse_permutation(&[3, 4], &[0, 1]), (vec![12], vec![0]));
        assert_eq!(
            collapse_permutation(&[1, 5, 1], &[2, 1, 0]),
            (vec![5], vec![0])
        );
        assert_eq!(collapse_permutation(&[1, 1], &[1, 0]), (vec![], vec![]));
    }
}
