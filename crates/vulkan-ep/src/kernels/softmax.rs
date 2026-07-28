//! `Softmax` kernel (opset ≥13) f32, any axis.
//!
//! One workgroup per row: max → sum of exp → normalization,
//! with reductions in shared memory (numerically stable softmax).

use crate::kernels::{base_kernel_impl, device_in, device_out};
use crate::ort_util::{attr_i64, to_status};
use crate::vk;
use anyhow::{Result, ensure};
use onnx_vulkan_core::shaders::normalization::{SOFTMAX, SOFTMAX_BINDINGS, SOFTMAX_PUSH_BYTES};
use ort_ep_sys as sys;
use vk_compute::{ComputePipeline, compile_wgsl};

#[repr(C)]
struct SoftmaxKernel {
    base: sys::OrtKernelImpl,
    pipeline: ComputePipeline,
    axis: i64,
}

/// # Safety
/// Called by ORT with valid pointers.
pub unsafe extern "C" fn create_kernel(
    _state: *mut std::ffi::c_void,
    info: *const sys::OrtKernelInfo,
    kernel_out: *mut *mut sys::OrtKernelImpl,
) -> *mut sys::OrtStatus {
    let result = (|| -> Result<()> {
        let ctx = vk::context()?;
        let kernel = Box::new(SoftmaxKernel {
            base: base_kernel_impl(compute, release),
            pipeline: ctx.create_pipeline(
                &compile_wgsl(SOFTMAX)?,
                SOFTMAX_BINDINGS,
                SOFTMAX_PUSH_BYTES,
            )?,
            axis: unsafe { attr_i64(info, c"axis", -1) },
        });
        unsafe { *kernel_out = Box::into_raw(kernel).cast::<sys::OrtKernelImpl>() };
        Ok(())
    })();
    to_status(result)
}

unsafe extern "C" fn compute(
    this_ptr: *mut sys::OrtKernelImpl,
    ctx_ptr: *mut sys::OrtKernelContext,
) -> sys::OrtStatusPtr {
    vk_compute::stats::set_op("Softmax");
    let kernel = unsafe { &*this_ptr.cast::<SoftmaxKernel>() };
    to_status(unsafe { compute_impl(kernel, ctx_ptr) })
}

unsafe fn compute_impl(kernel: &SoftmaxKernel, ctx_ptr: *mut sys::OrtKernelContext) -> Result<()> {
    let ctx = vk::context()?;
    let (x, x_shape, n) = unsafe { device_in(ctx_ptr, 0)? };
    let rank = x_shape.len() as i64;
    let axis = if kernel.axis < 0 {
        kernel.axis + rank
    } else {
        kernel.axis
    };
    ensure!(
        (0..rank).contains(&axis),
        "Softmax: axis {axis} out of rank {rank}"
    );
    // `c` elements along the axis, spaced by `inner`; one row per pair
    // (outer, inner). With the last axis `inner = 1` (contiguous rows).
    let axis = axis as usize;
    let c = x_shape[axis] as usize;
    let inner: usize = x_shape[axis + 1..].iter().product::<i64>().max(1) as usize;
    let rows = n.checked_div(c.max(1)).unwrap_or(0);
    // 2D grid: `rows` often exceeds the 65535 workgroups-per-axis limit
    let gx = rows.clamp(1, 32768) as u32;
    let gy = (rows as u32).div_ceil(gx);

    let out = unsafe { device_out(ctx_ptr, 0, &x_shape)? };
    if n == 0 {
        return Ok(());
    }
    let mut push = Vec::with_capacity(16);
    for v in [c as u32, inner as u32, rows as u32, gx] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    ctx.stream_dispatch_slices(
        &kernel.pipeline,
        &[x.slice(), out.slice()],
        &push,
        [gx, gy, 1],
    )?;
    Ok(())
}

unsafe extern "C" fn release(this_ptr: *mut sys::OrtKernelImpl) {
    if this_ptr.is_null() {
        return;
    }
    let kernel = unsafe { Box::from_raw(this_ptr.cast::<SoftmaxKernel>()) };
    if let Ok(ctx) = vk::context() {
        let _ = ctx.flush();
        ctx.destroy_pipeline(kernel.pipeline);
    }
}
