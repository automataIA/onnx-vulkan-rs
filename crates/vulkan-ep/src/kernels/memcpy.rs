//! `MemcpyFromHost` / `MemcpyToHost` kernels.
//!
//! ORT inserts these automatically at the boundaries between VulkanEP nodes
//! and CPU nodes (MemcpyTransformer): they are the channel through which
//! tensors enter and leave VRAM. FromHost enqueues an upload into the stream
//! (no submit); ToHost forces a flush — the only synchronization point of the
//! pipeline.

use crate::data_transfer::value_byte_size;
use crate::device_mem::region_from_ptr;
use crate::kernels::base_kernel_impl;
use crate::ort_util::{apis, check, to_status};
use crate::vk;
use anyhow::Result;
use ort_ep_sys as sys;

#[derive(Clone, Copy, PartialEq)]
enum Direction {
    FromHost,
    ToHost,
}

#[repr(C)]
struct MemcpyKernel {
    base: sys::OrtKernelImpl,
    direction: Direction,
}

/// # Safety
/// Called by ORT with valid pointers.
pub unsafe extern "C" fn create_from_host(
    _state: *mut std::ffi::c_void,
    _info: *const sys::OrtKernelInfo,
    kernel_out: *mut *mut sys::OrtKernelImpl,
) -> *mut sys::OrtStatus {
    create(Direction::FromHost, kernel_out)
}

/// # Safety
/// Called by ORT with valid pointers.
pub unsafe extern "C" fn create_to_host(
    _state: *mut std::ffi::c_void,
    _info: *const sys::OrtKernelInfo,
    kernel_out: *mut *mut sys::OrtKernelImpl,
) -> *mut sys::OrtStatus {
    create(Direction::ToHost, kernel_out)
}

fn create(direction: Direction, kernel_out: *mut *mut sys::OrtKernelImpl) -> *mut sys::OrtStatus {
    let kernel = Box::new(MemcpyKernel {
        base: base_kernel_impl(compute, release),
        direction,
    });
    unsafe { *kernel_out = Box::into_raw(kernel).cast::<sys::OrtKernelImpl>() };
    std::ptr::null_mut()
}

unsafe extern "C" fn compute(
    this_ptr: *mut sys::OrtKernelImpl,
    ctx_ptr: *mut sys::OrtKernelContext,
) -> sys::OrtStatusPtr {
    let kernel = unsafe { &*this_ptr.cast::<MemcpyKernel>() };
    vk_compute::stats::set_op(match kernel.direction {
        Direction::FromHost => "MemcpyFromHost",
        Direction::ToHost => "MemcpyToHost",
    });
    to_status(unsafe { compute_impl(kernel, ctx_ptr) })
}

unsafe fn compute_impl(kernel: &MemcpyKernel, ctx_ptr: *mut sys::OrtKernelContext) -> Result<()> {
    let api = apis().ort;
    let ctx = vk::context()?;
    unsafe {
        // input value + shape
        let mut in_value: *const sys::OrtValue = std::ptr::null();
        check(
            (api.KernelContext_GetInput.expect("KernelContext_GetInput"))(
                ctx_ptr,
                0,
                &mut in_value,
            ),
            "KernelContext_GetInput",
        )?;
        let view = crate::ort_util::value_view(in_value)?;
        let bytes = value_byte_size(in_value)?;

        let mut out_value: *mut sys::OrtValue = std::ptr::null_mut();
        check(
            (api.KernelContext_GetOutput
                .expect("KernelContext_GetOutput"))(
                ctx_ptr,
                0,
                view.shape.as_ptr(),
                view.shape.len(),
                &mut out_value,
            ),
            "KernelContext_GetOutput",
        )?;
        let mut out_data: *mut std::ffi::c_void = std::ptr::null_mut();
        check(
            (api.GetTensorMutableData.expect("GetTensorMutableData"))(out_value, &mut out_data),
            "GetTensorMutableData",
        )?;

        if bytes == 0 {
            return Ok(());
        }
        match kernel.direction {
            Direction::FromHost => {
                // input: CPU data; output: device region
                let dst = region_from_ptr(out_data.cast_const())?;
                let src = std::slice::from_raw_parts(view.data, bytes);
                ctx.stream_upload_at(dst.buffer(), dst.offset, src)?;
            }
            Direction::ToHost => {
                // input: device region; output: CPU data → flush
                let src = region_from_ptr(view.data.cast())?;
                let data = ctx.stream_download_at(src.buffer(), src.offset, bytes)?;
                std::ptr::copy_nonoverlapping(data.as_ptr(), out_data.cast::<u8>(), bytes);
            }
        }
        Ok(())
    }
}

unsafe extern "C" fn release(this_ptr: *mut sys::OrtKernelImpl) {
    if !this_ptr.is_null() {
        drop(unsafe { Box::from_raw(this_ptr.cast::<MemcpyKernel>()) });
    }
}
