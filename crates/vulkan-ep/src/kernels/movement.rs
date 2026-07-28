//! Movement/shape ops: Reshape, Squeeze, Unsqueeze (identity-copy on data),
//! Transpose (permutation). Covering them keeps chains on the GPU and avoids
//! CPU↔GPU round-trips at boundaries — the main bottleneck (see profiler).
//!
//! The data tensor (input 0) is on device (`BufferEntry`); any shape/axes
//! input (input 1) is on CPU (registered with `OrtMemTypeCPUInput`).

use crate::kernels::{base_kernel_impl, device_in_sized, device_out};
use crate::ort_util::{apis, attr_ints, kernel_input, to_status};
use crate::vk;
use anyhow::{Result, ensure};
use onnx_vulkan_core::shaders::movement::{
    TRANSPOSE as WGSL_TRANSPOSE, TRANSPOSE_BINDINGS, TRANSPOSE_PUSH_BYTES,
};
use ort_ep_sys as sys;
use vk_compute::{ComputePipeline, compile_wgsl};

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Reshape,
    Squeeze,
    Unsqueeze,
    Transpose,
}

#[repr(C)]
struct MovementKernel {
    base: sys::OrtKernelImpl,
    kind: Kind,
    /// perm (Transpose) or axes (Squeeze/Unsqueeze) from attribute, if present.
    attr_axes: Option<Vec<i64>>,
    transpose_pipeline: Option<ComputePipeline>,
}

fn make_create(
    kind: Kind,
    info: *const sys::OrtKernelInfo,
    kernel_out: *mut *mut sys::OrtKernelImpl,
) -> *mut sys::OrtStatus {
    let result = (|| -> Result<()> {
        // perm/axes may be an attribute (old opsets) or an input (newer ones)
        let attr_name = match kind {
            Kind::Transpose => Some(c"perm"),
            Kind::Squeeze | Kind::Unsqueeze => Some(c"axes"),
            Kind::Reshape => None,
        };
        let attr_axes = attr_name.and_then(|n| unsafe { attr_ints(info, n) });
        let transpose_pipeline = if kind == Kind::Transpose {
            Some(vk::context()?.create_pipeline(
                &compile_wgsl(WGSL_TRANSPOSE)?,
                TRANSPOSE_BINDINGS,
                TRANSPOSE_PUSH_BYTES,
            )?)
        } else {
            None
        };
        let kernel = Box::new(MovementKernel {
            base: base_kernel_impl(compute, release),
            kind,
            attr_axes,
            transpose_pipeline,
        });
        unsafe { *kernel_out = Box::into_raw(kernel).cast::<sys::OrtKernelImpl>() };
        Ok(())
    })();
    to_status(result)
}

macro_rules! create_fn {
    ($name:ident, $kind:expr) => {
        /// # Safety
        /// Called by ORT with valid pointers.
        pub unsafe extern "C" fn $name(
            _state: *mut std::ffi::c_void,
            info: *const sys::OrtKernelInfo,
            kernel_out: *mut *mut sys::OrtKernelImpl,
        ) -> *mut sys::OrtStatus {
            make_create($kind, info, kernel_out)
        }
    };
}

create_fn!(create_reshape, Kind::Reshape);
create_fn!(create_squeeze, Kind::Squeeze);
create_fn!(create_unsqueeze, Kind::Unsqueeze);
create_fn!(create_transpose, Kind::Transpose);

/// Reads a CPU int64 input (shape/axes) as Vec<i64>.
unsafe fn cpu_i64_input(ctx: *const sys::OrtKernelContext, index: usize) -> Result<Vec<i64>> {
    let view = unsafe { kernel_input(ctx, index)? };
    let n = view.elem_count;
    let ptr = view.data.cast::<i64>();
    Ok((0..n).map(|i| unsafe { *ptr.add(i) }).collect())
}

/// Number of inputs of the node.
unsafe fn num_inputs(ctx: *const sys::OrtKernelContext) -> usize {
    let api = apis().ort;
    let mut n = 0usize;
    let _ = unsafe { (api.KernelContext_GetInputCount.expect("GetInputCount"))(ctx, &mut n) };
    n
}

unsafe extern "C" fn compute(
    this_ptr: *mut sys::OrtKernelImpl,
    ctx_ptr: *mut sys::OrtKernelContext,
) -> sys::OrtStatusPtr {
    let kernel = unsafe { &*this_ptr.cast::<MovementKernel>() };
    vk_compute::stats::set_op(match kernel.kind {
        Kind::Reshape => "Reshape",
        Kind::Squeeze => "Squeeze",
        Kind::Unsqueeze => "Unsqueeze",
        Kind::Transpose => "Transpose",
    });
    to_status(unsafe { compute_impl(kernel, ctx_ptr) })
}

/// Output shape according to the ONNX semantics of the op.
unsafe fn output_shape(
    kernel: &MovementKernel,
    ctx_ptr: *mut sys::OrtKernelContext,
    in_shape: &[i64],
    elem_count: usize,
) -> Result<Vec<i64>> {
    Ok(match kernel.kind {
        Kind::Reshape => {
            let target = unsafe { cpu_i64_input(ctx_ptr, 1)? };
            let mut out = vec![0i64; target.len()];
            let mut infer = None;
            let mut known = 1i64;
            for (i, &d) in target.iter().enumerate() {
                if d == -1 {
                    infer = Some(i);
                } else if d == 0 {
                    out[i] = in_shape[i]; // copied from the input dim
                    known *= out[i];
                } else {
                    out[i] = d;
                    known *= d;
                }
            }
            if let Some(i) = infer {
                out[i] = elem_count as i64 / known.max(1);
            }
            out
        }
        Kind::Squeeze => {
            let axes = unsafe { axes_from_attr_or_input(kernel, ctx_ptr, in_shape.len())? };
            match axes {
                Some(ax) => in_shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !ax.contains(&(*i as i64)))
                    .map(|(_, &d)| d)
                    .collect(),
                None => in_shape.iter().copied().filter(|&d| d != 1).collect(),
            }
        }
        Kind::Unsqueeze => {
            let axes = unsafe { axes_from_attr_or_input(kernel, ctx_ptr, in_shape.len())? }
                .unwrap_or_default();
            let out_rank = in_shape.len() + axes.len();
            let norm: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (a + out_rank as i64) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            let mut out = Vec::with_capacity(out_rank);
            let mut it = in_shape.iter();
            for i in 0..out_rank {
                if norm.contains(&i) {
                    out.push(1);
                } else {
                    out.push(*it.next().unwrap());
                }
            }
            out
        }
        Kind::Transpose => {
            let perm = kernel
                .attr_axes
                .clone()
                .unwrap_or_else(|| (0..in_shape.len() as i64).rev().collect());
            perm.iter().map(|&p| in_shape[p as usize]).collect()
        }
    })
}

unsafe fn axes_from_attr_or_input(
    kernel: &MovementKernel,
    ctx_ptr: *mut sys::OrtKernelContext,
    _rank: usize,
) -> Result<Option<Vec<i64>>> {
    if let Some(a) = &kernel.attr_axes {
        return Ok(Some(a.clone()));
    }
    if unsafe { num_inputs(ctx_ptr) } > 1 {
        return Ok(Some(unsafe { cpu_i64_input(ctx_ptr, 1)? }));
    }
    Ok(None)
}

unsafe fn compute_impl(kernel: &MovementKernel, ctx_ptr: *mut sys::OrtKernelContext) -> Result<()> {
    let ctx = vk::context()?;
    let (src, in_shape, elem_count, elem_size) = unsafe { device_in_sized(ctx_ptr, 0)? };
    let out_shape = unsafe { output_shape(kernel, ctx_ptr, &in_shape, elem_count)? };

    match kernel.kind {
        Kind::Transpose => {
            let perm: Vec<usize> = kernel
                .attr_axes
                .clone()
                .unwrap_or_else(|| (0..in_shape.len() as i64).rev().collect())
                .iter()
                .map(|&p| p as usize)
                .collect();
            unsafe {
                transpose(
                    ctx, kernel, &src, elem_size, &in_shape, &perm, &out_shape, ctx_ptr,
                )
            }
        }
        _ => {
            // identity on data: GPU→GPU copy (same dtype and count)
            let out = unsafe { device_out(ctx_ptr, 0, &out_shape)? };
            let bytes = (elem_count * elem_size) as u64;
            if bytes > 0 {
                ctx.stream_copy_range(src.buffer(), src.offset, out.buffer(), out.offset, bytes)?;
            }
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn transpose(
    ctx: &vk_compute::VkContext,
    kernel: &MovementKernel,
    src: &crate::device_mem::DeviceRegion<'_>,
    elem_size: usize,
    in_shape: &[i64],
    perm: &[usize],
    out_shape: &[i64],
    ctx_ptr: *mut sys::OrtKernelContext,
) -> Result<()> {
    let rank = in_shape.len();
    ensure!(rank <= 8, "Transpose: rank {rank} > 8 not supported");
    // input strides, row-major
    let mut in_strides = vec![0u32; rank];
    let mut acc = 1u32;
    for d in (0..rank).rev() {
        in_strides[d] = acc;
        acc *= in_shape[d] as u32;
    }
    // for each output dim d: input stride of dim perm[d]
    let perm_strides: Vec<u32> = perm.iter().map(|&p| in_strides[p]).collect();
    let out_dims: Vec<u32> = out_shape.iter().map(|&d| d as u32).collect();
    let n: usize = out_shape.iter().product::<i64>() as usize;

    // dtype: the permutation works on words; check elem size == 4
    ensure!(
        elem_size == 4,
        "Transpose: elem size {elem_size} != 4 not supported (u32 kernel)"
    );
    let out = unsafe { device_out(ctx_ptr, 0, out_shape)? };
    if n == 0 {
        return Ok(());
    }

    let pipeline = kernel.transpose_pipeline.as_ref().unwrap();
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
    ctx.stream_dispatch_slices(
        pipeline,
        &[src.slice(), out.slice()],
        &push,
        [(n as u32).div_ceil(256), 1, 1],
    )?;
    Ok(())
}

unsafe extern "C" fn release(this_ptr: *mut sys::OrtKernelImpl) {
    if this_ptr.is_null() {
        return;
    }
    let kernel = unsafe { Box::from_raw(this_ptr.cast::<MovementKernel>()) };
    if let (Ok(ctx), Some(p)) = (vk::context(), kernel.transpose_pipeline) {
        let _ = ctx.flush();
        ctx.destroy_pipeline(p);
    }
}
