//! Vulkan kernels of the plugin.
//!
//! Device-memory model: kernel inputs/outputs are `BufferEntry`s
//! (tensors resident in VRAM, allocated by our `OrtAllocator`). Kernels
//! enqueue dispatches into the `vk-compute` stream; no submit or readback
//! happens here — synchronization occurs only in the GPU→CPU DataTransfer.

pub mod dynamic_quantize;
pub mod elementwise;
pub mod layernorm;
pub mod matmul_fp32;
pub mod matmul_integer;
pub mod memcpy;
pub mod movement;
pub mod softmax;

use crate::device_mem::{DeviceRegion, region_from_ptr};
use crate::ort_util::{kernel_input, kernel_output};
use anyhow::Result;
use ort_ep_sys as sys;

/// Common base: `OrtKernelImpl` must be the first field (`#[repr(C)]`).
pub fn base_kernel_impl(
    compute: unsafe extern "C" fn(
        *mut sys::OrtKernelImpl,
        *mut sys::OrtKernelContext,
    ) -> sys::OrtStatusPtr,
    release: unsafe extern "C" fn(*mut sys::OrtKernelImpl),
) -> sys::OrtKernelImpl {
    let mut base: sys::OrtKernelImpl = unsafe { std::mem::zeroed() };
    base.ort_version_supported = sys::ORT_API_VERSION;
    base.Compute = Some(compute);
    base.Release = Some(release);
    base
}

/// Input tensor resident on device: (region, shape, elem_count).
///
/// The region may be a **slice** of an allocation: with the memory pattern
/// enabled, ORT allocates a single block and assigns tensors `handle + offset`.
/// The dispatch binds the binding starting from that offset (`DeviceRegion::slice`).
///
/// # Safety
/// `ctx` valid, `index` within the node inputs, tensor allocated by
/// our allocator (verified via magic).
pub unsafe fn device_in(
    ctx: *const sys::OrtKernelContext,
    index: usize,
) -> Result<(DeviceRegion<'static>, Vec<i64>, usize)> {
    let (region, shape, elem_count, _) = unsafe { device_in_sized(ctx, index)? };
    Ok((region, shape, elem_count))
}

/// Like [`device_in`], but also reports bytes per element: useful for callers
/// that need to compute the tensor byte length, which with the memory pattern
/// does **not** match the allocation size.
///
/// # Safety
/// Like [`device_in`].
pub unsafe fn device_in_sized(
    ctx: *const sys::OrtKernelContext,
    index: usize,
) -> Result<(DeviceRegion<'static>, Vec<i64>, usize, usize)> {
    let view = unsafe { kernel_input(ctx, index)? };
    let region = unsafe { region_from_ptr(view.data.cast())? };
    Ok((region, view.shape, view.elem_count, view.elem_size))
}

/// Output tensor resident on device.
///
/// # Safety
/// Like [`device_in`].
pub unsafe fn device_out(
    ctx: *mut sys::OrtKernelContext,
    index: usize,
    shape: &[i64],
) -> Result<DeviceRegion<'static>> {
    let ptr = unsafe { kernel_output(ctx, index, shape)? };
    unsafe { region_from_ptr(ptr.cast()) }
}
