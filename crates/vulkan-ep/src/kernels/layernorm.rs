//! `LayerNormalization` kernel (opset ≥17) f32, axis = last dimension.
//!
//! One workgroup (256 threads) per row: sum/sum-of-squares reduction in
//! shared memory → mean and variance → normalization with scale and bias.

use crate::kernels::{base_kernel_impl, device_in, device_out};
use crate::ort_util::{attr_f32, attr_i64, to_status};
use crate::vk;
use anyhow::{Result, ensure};
use onnx_vulkan_core::shaders::normalization::{
    LAYERNORM, LAYERNORM_BINDINGS, LAYERNORM_PUSH_BYTES,
};
use ort_ep_sys as sys;
use vk_compute::{ComputePipeline, compile_wgsl};

#[repr(C)]
struct LayerNormKernel {
    base: sys::OrtKernelImpl,
    pipeline: ComputePipeline,
    epsilon: f32,
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
        let kernel = Box::new(LayerNormKernel {
            base: base_kernel_impl(compute, release),
            pipeline: ctx.create_pipeline(
                &compile_wgsl(LAYERNORM)?,
                LAYERNORM_BINDINGS,
                LAYERNORM_PUSH_BYTES,
            )?,
            epsilon: unsafe { attr_f32(info, c"epsilon", 1e-5) },
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
    vk_compute::stats::set_op("LayerNormalization");
    let kernel = unsafe { &*this_ptr.cast::<LayerNormKernel>() };
    to_status(unsafe { compute_impl(kernel, ctx_ptr) })
}

unsafe fn compute_impl(
    kernel: &LayerNormKernel,
    ctx_ptr: *mut sys::OrtKernelContext,
) -> Result<()> {
    let ctx = vk::context()?;
    let (x, x_shape, n) = unsafe { device_in(ctx_ptr, 0)? };
    let rank = x_shape.len() as i64;
    let axis = if kernel.axis < 0 {
        kernel.axis + rank
    } else {
        kernel.axis
    };
    ensure!(
        axis == rank - 1,
        "LayerNormalization: axis {axis} != last dimension (rank {rank})"
    );
    let c = *x_shape.last().unwrap() as usize;
    let rows = n / c.max(1);
    ensure!(rows <= 65535, "LayerNormalization: too many rows ({rows})");

    let (scale, scale_shape, _) = unsafe { device_in(ctx_ptr, 1)? };
    ensure!(
        scale_shape == vec![c as i64],
        "LayerNormalization: scale shape {scale_shape:?} != [{c}]"
    );
    // optional bias: if absent, rebind scale (has_bias=0 ignores it)
    let bias_in = unsafe { device_in(ctx_ptr, 2) };
    let (has_bias, bias_slice) = match &bias_in {
        Ok((b, _, _)) => (1u32, b.slice()),
        Err(_) => (0u32, scale.slice()),
    };

    let out = unsafe { device_out(ctx_ptr, 0, &x_shape)? };
    let mut push = Vec::with_capacity(LAYERNORM_PUSH_BYTES as usize);
    push.extend_from_slice(&(c as u32).to_le_bytes());
    push.extend_from_slice(&kernel.epsilon.to_le_bytes());
    push.extend_from_slice(&has_bias.to_le_bytes());
    ctx.stream_dispatch_slices(
        &kernel.pipeline,
        &[x.slice(), scale.slice(), bias_slice, out.slice()],
        &push,
        [rows as u32, 1, 1],
    )?;
    Ok(())
}

unsafe extern "C" fn release(this_ptr: *mut sys::OrtKernelImpl) {
    if this_ptr.is_null() {
        return;
    }
    let kernel = unsafe { Box::from_raw(this_ptr.cast::<LayerNormKernel>()) };
    if let Ok(ctx) = vk::context() {
        let _ = ctx.flush();
        ctx.destroy_pipeline(kernel.pipeline);
    }
}
