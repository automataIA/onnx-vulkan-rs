//! `MatMul` kernel f32 (opset ≥13) with ONNX batch broadcasting.
//!
//! A: [ba..., M, K], B: [bb..., K, N] → out: [broadcast(ba,bb)..., M, N].
//! gid.z indexes the batch; batch strides (in matrix units, 0 on broadcast
//! dimensions) are passed in push constants.

use crate::kernels::{base_kernel_impl, device_in, device_out};
use crate::ort_util::to_status;
use crate::vk;
use anyhow::{Result, ensure};
use onnx_vulkan_core::broadcast;
use onnx_vulkan_core::shaders::elementwise::MAX_RANK;
use onnx_vulkan_core::shaders::matmul_fp32::{BINDINGS, MATMUL, PUSH_BYTES, TILE_SIZE};
use onnx_vulkan_core::shaders::push_vec4s;
use ort_ep_sys as sys;
use vk_compute::{ComputePipeline, compile_wgsl};

#[repr(C)]
struct MatMulKernel {
    base: sys::OrtKernelImpl,
    pipeline: ComputePipeline,
}

/// # Safety
/// Called by ORT with valid pointers.
pub unsafe extern "C" fn create_kernel(
    _state: *mut std::ffi::c_void,
    _info: *const sys::OrtKernelInfo,
    kernel_out: *mut *mut sys::OrtKernelImpl,
) -> *mut sys::OrtStatus {
    let result = (|| -> Result<()> {
        let ctx = vk::context()?;
        let kernel = Box::new(MatMulKernel {
            base: base_kernel_impl(compute, release),
            pipeline: ctx.create_pipeline(&compile_wgsl(MATMUL)?, BINDINGS, PUSH_BYTES)?,
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
    vk_compute::stats::set_op("MatMul");
    let kernel = unsafe { &*this_ptr.cast::<MatMulKernel>() };
    to_status(unsafe { compute_impl(kernel, ctx_ptr) })
}

unsafe fn compute_impl(kernel: &MatMulKernel, ctx_ptr: *mut sys::OrtKernelContext) -> Result<()> {
    let ctx = vk::context()?;
    let (a, a_shape, _) = unsafe { device_in(ctx_ptr, 0)? };
    let (b, b_shape, _) = unsafe { device_in(ctx_ptr, 1)? };
    ensure!(
        a_shape.len() >= 2 && b_shape.len() >= 2,
        "MatMul: rank < 2 not supported (A {a_shape:?}, B {b_shape:?})"
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
    let batch_out = bc.out_shape;
    let batch: usize = batch_out.iter().product::<i64>() as usize;
    ensure!(batch <= 65535, "MatMul: batch {batch} too large");

    let mut out_shape = batch_out.clone();
    out_shape.push(m as i64);
    out_shape.push(n as i64);
    let out = unsafe { device_out(ctx_ptr, 0, &out_shape)? };

    let mut push = Vec::with_capacity(PUSH_BYTES as usize);
    for v in [m as u32, ka as u32, n as u32, batch_out.len() as u32] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    push_vec4s(&mut push, &bc.out_strides);
    push_vec4s(&mut push, &bc.a_strides);
    push_vec4s(&mut push, &bc.b_strides);
    ctx.stream_dispatch_slices(
        &kernel.pipeline,
        &[a.slice(), b.slice(), out.slice()],
        &push,
        [
            (n as u32).div_ceil(TILE_SIZE),
            (m as u32).div_ceil(TILE_SIZE),
            (batch as u32).max(1),
        ],
    )?;
    Ok(())
}

unsafe extern "C" fn release(this_ptr: *mut sys::OrtKernelImpl) {
    if this_ptr.is_null() {
        return;
    }
    let kernel = unsafe { Box::from_raw(this_ptr.cast::<MatMulKernel>()) };
    if let Ok(ctx) = vk::context() {
        let _ = ctx.flush();
        ctx.destroy_pipeline(kernel.pipeline);
    }
}
