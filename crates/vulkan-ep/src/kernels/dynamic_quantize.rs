//! Vulkan kernel for `DynamicQuantizeLinear` (ONNX opset ≥11), device memory.
//!
//! x: f32 [*] device → y: u8 [*], y_scale: f32 [], y_zero_point: u8 [] device.
//! Three passes enqueued on the stream (no readback): 1) partial min/max per
//! workgroup, 2) final reduction + scale/zp written to the output tensors,
//! 3) elementwise quantization. The range always includes 0 (ONNX spec);
//! WGSL round() is round-half-to-even, matching the ONNX reference.

use crate::kernels::{base_kernel_impl, device_in, device_out};
use crate::ort_util::to_status;
use crate::vk;
use anyhow::Result;
use onnx_vulkan_core::shaders::dynamic_quantize::{FINALIZE, PARTIAL, QUANTIZE};
use ort_ep_sys as sys;
use vk_compute::{ComputePipeline, compile_wgsl};

#[repr(C)]
struct DynamicQuantizeKernel {
    base: sys::OrtKernelImpl,
    partial: ComputePipeline,
    finalize: ComputePipeline,
    quantize: ComputePipeline,
}

/// Entry point registered in the kernel registry.
///
/// # Safety
/// Called by ORT with valid pointers.
pub unsafe extern "C" fn create_kernel(
    _state: *mut std::ffi::c_void,
    _info: *const sys::OrtKernelInfo,
    kernel_out: *mut *mut sys::OrtKernelImpl,
) -> *mut sys::OrtStatus {
    let result = (|| -> Result<()> {
        let ctx = vk::context()?;
        let kernel = Box::new(DynamicQuantizeKernel {
            base: base_kernel_impl(compute, release),
            partial: ctx.create_pipeline(&compile_wgsl(PARTIAL)?, 2, 4)?,
            finalize: ctx.create_pipeline(&compile_wgsl(FINALIZE)?, 3, 4)?,
            quantize: ctx.create_pipeline(&compile_wgsl(QUANTIZE)?, 4, 4)?,
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
    vk_compute::stats::set_op("DynamicQuantizeLinear");
    let kernel = unsafe { &*this_ptr.cast::<DynamicQuantizeKernel>() };
    to_status(unsafe { compute_impl(kernel, ctx_ptr) })
}

unsafe fn compute_impl(
    kernel: &DynamicQuantizeKernel,
    ctx_ptr: *mut sys::OrtKernelContext,
) -> Result<()> {
    let ctx = vk::context()?;
    let (x, x_shape, n) = unsafe { device_in(ctx_ptr, 0)? };
    anyhow::ensure!(n > 0, "DynamicQuantizeLinear: empty input");

    let y = unsafe { device_out(ctx_ptr, 0, &x_shape)? };
    let y_scale = unsafe { device_out(ctx_ptr, 1, &[])? };
    let y_zp = unsafe { device_out(ctx_ptr, 2, &[])? };

    // scratch for partial min/max (destroyed after flush)
    let groups = ((n as u32).div_ceil(256)).min(1024);
    let partial = ctx.create_storage_buffer(u64::from(groups) * 8)?;

    ctx.stream_dispatch_slices(
        &kernel.partial,
        &[x.slice(), (&partial).into()],
        &(n as u32).to_le_bytes(),
        [groups, 1, 1],
    )?;
    ctx.stream_dispatch_slices(
        &kernel.finalize,
        &[(&partial).into(), y_scale.slice(), y_zp.slice()],
        &groups.to_le_bytes(),
        [1, 1, 1],
    )?;
    let words = n.div_ceil(4);
    ctx.stream_dispatch_slices(
        &kernel.quantize,
        &[x.slice(), y_scale.slice(), y_zp.slice(), y.slice()],
        &(n as u32).to_le_bytes(),
        [(words as u32).div_ceil(256), 1, 1],
    )?;

    ctx.defer_destroy(partial);
    Ok(())
}

unsafe extern "C" fn release(this_ptr: *mut sys::OrtKernelImpl) {
    if this_ptr.is_null() {
        return;
    }
    let kernel = unsafe { Box::from_raw(this_ptr.cast::<DynamicQuantizeKernel>()) };
    if let Ok(ctx) = vk::context() {
        let _ = ctx.flush();
        ctx.destroy_pipeline(kernel.partial);
        ctx.destroy_pipeline(kernel.finalize);
        ctx.destroy_pipeline(kernel.quantize);
    }
}
