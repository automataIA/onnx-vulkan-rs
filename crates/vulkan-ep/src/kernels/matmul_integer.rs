//! Vulkan kernel for `MatMulInteger` (ONNX opset ≥10), device memory.
//!
//! Supported case (the one produced by ORT dynamic quantization):
//!   A: u8 [..., M, K] device (K multiple of 4)   a_zp: u8 scalar device
//!   B: u8 [K, N] initializer device              b_zp: u8 scalar device
//!   out: i32 [..., M, N] device
//! B is transposed+packed [N, K/4] on GPU at the first Compute and cached.
//! A is read directly as u32 words (zero copy). Zero points are read by the
//! shader (no CPU readback).

use crate::kernels::{base_kernel_impl, device_in, device_out};
use crate::ort_util::to_status;
use crate::vk;
use anyhow::{Result, ensure};
use onnx_vulkan_core::shaders::matmul_integer::{
    COOP_BINDINGS, COOP_PUSH_BYTES, CoopVariant, MATMUL_BINDINGS, MATMUL_PUSH_BYTES, PACK_B,
    PACK_BINDINGS, PACK_PUSH_BYTES, TILE_SIZE, coop_applies, coop_variant, matmul,
};
use ort_ep_sys as sys;
use vk_compute::{ComputePipeline, GpuBuffer, compile_wgsl};

struct PackedB {
    buf: GpuBuffer,
    k: usize,
    n: usize,
}

#[repr(C)]
struct MatMulIntegerKernel {
    base: sys::OrtKernelImpl,
    pack_pipeline: ComputePipeline,
    matmul_pipeline: ComputePipeline,
    /// Cooperative matrix variant, if the device exposes it. It does not cover
    /// every shape, so `matmul_pipeline` remains the reference.
    coop: Option<(&'static CoopVariant, ComputePipeline)>,
    packed_b: Option<PackedB>,
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
        let kernel = Box::new(MatMulIntegerKernel {
            base: base_kernel_impl(compute, release),
            pack_pipeline: ctx.create_pipeline(
                &compile_wgsl(PACK_B)?,
                PACK_BINDINGS,
                PACK_PUSH_BYTES,
            )?,
            matmul_pipeline: ctx.create_pipeline(
                &compile_wgsl(&matmul(ctx.has_integer_dot_product))?,
                MATMUL_BINDINGS,
                MATMUL_PUSH_BYTES,
            )?,
            coop: match coop_variant(&ctx.coop_u8, ctx.subgroup_size) {
                Some(v) => Some((
                    v,
                    ctx.create_pipeline(&v.spirv(), COOP_BINDINGS, COOP_PUSH_BYTES)?,
                )),
                None => None,
            },
            packed_b: None,
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
    vk_compute::stats::set_op("MatMulInteger");
    let kernel = unsafe { &mut *this_ptr.cast::<MatMulIntegerKernel>() };
    to_status(unsafe { compute_impl(kernel, ctx_ptr) })
}

unsafe fn compute_impl(
    kernel: &mut MatMulIntegerKernel,
    ctx_ptr: *mut sys::OrtKernelContext,
) -> Result<()> {
    let ctx = vk::context()?;

    let (a, a_shape, _) = unsafe { device_in(ctx_ptr, 0)? };
    ensure!(
        a_shape.len() >= 2,
        "MatMulInteger: A with rank {} not supported",
        a_shape.len()
    );
    let k = *a_shape.last().unwrap() as usize;
    let m: usize = a_shape[..a_shape.len() - 1].iter().product::<i64>() as usize;
    ensure!(
        k.is_multiple_of(4),
        "MatMulInteger: K={k} not a multiple of 4"
    );

    let (a_zp, _, a_zp_count) = unsafe { device_in(ctx_ptr, 2)? };
    let (b_zp, _, b_zp_count) = unsafe { device_in(ctx_ptr, 3)? };
    ensure!(
        a_zp_count == 1 && b_zp_count == 1,
        "MatMulInteger: per-channel zero point not supported"
    );

    // pack B on first execution (B is a constant initializer)
    if kernel.packed_b.is_none() {
        let (b, b_shape, _) = unsafe { device_in(ctx_ptr, 1)? };
        ensure!(
            b_shape.len() == 2,
            "MatMulInteger: B with rank {} not supported",
            b_shape.len()
        );
        let (bk, bn) = (b_shape[0] as usize, b_shape[1] as usize);
        ensure!(bk == k, "MatMulInteger: K incompatibile (A: {k}, B: {bk})");
        let k4 = bk / 4;
        let packed = ctx.create_storage_buffer((bn * k4 * 4).max(4) as u64)?;
        // the registry constrains this kernel to `uint8`: no sign flip
        let mut push = Vec::with_capacity(PACK_PUSH_BYTES as usize);
        push.extend_from_slice(&(bk as u32).to_le_bytes());
        push.extend_from_slice(&(bn as u32).to_le_bytes());
        push.extend_from_slice(&0u32.to_le_bytes());
        ctx.stream_dispatch_slices(
            &kernel.pack_pipeline,
            &[b.slice(), (&packed).into()],
            &push,
            [
                (bn as u32).div_ceil(TILE_SIZE),
                (k4 as u32).div_ceil(TILE_SIZE),
                1,
            ],
        )?;
        kernel.packed_b = Some(PackedB {
            buf: packed,
            k: bk,
            n: bn,
        });
        log::debug!("MatMulInteger: B [{bk}, {bn}] trasposta+packed su GPU");
    }
    let packed = kernel.packed_b.as_ref().unwrap();
    ensure!(packed.k == k, "MatMulInteger: K changed between runs");
    let n = packed.n;
    let k4 = k / 4;

    let mut out_shape: Vec<i64> = a_shape[..a_shape.len() - 1].to_vec();
    out_shape.push(n as i64);
    let out = unsafe { device_out(ctx_ptr, 0, &out_shape)? };

    let buffers = [
        a.slice(),
        (&packed.buf).into(),
        a_zp.slice(),
        b_zp.slice(),
        out.slice(),
    ];
    let grid = [
        (n as u32).div_ceil(TILE_SIZE),
        (m as u32).div_ceil(TILE_SIZE),
        1,
    ];

    // the registry constrains this kernel to `uint8` (see `registry.rs`), so
    // no operand is signed: the int8 path lives in the core interpreter
    if let Some((v, pipeline)) = &kernel.coop
        && coop_applies(v, m, k, n, false)
    {
        let mut push = Vec::with_capacity(COOP_PUSH_BYTES as usize);
        for value in [m as u32, k as u32, n as u32, 0, 0] {
            push.extend_from_slice(&value.to_le_bytes());
        }
        return ctx.stream_dispatch_slices(pipeline, &buffers, &push, grid);
    }

    let mut push = Vec::with_capacity(MATMUL_PUSH_BYTES as usize);
    for v in [m as u32, k4 as u32, n as u32, 0, 0, 0] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    ctx.stream_dispatch_slices(&kernel.matmul_pipeline, &buffers, &push, grid)?;
    Ok(())
}

unsafe extern "C" fn release(this_ptr: *mut sys::OrtKernelImpl) {
    if this_ptr.is_null() {
        return;
    }
    let kernel = unsafe { Box::from_raw(this_ptr.cast::<MatMulIntegerKernel>()) };
    if let Ok(ctx) = vk::context() {
        let _ = ctx.flush(); // any pending commands referencing the pipelines
        if let Some(packed) = kernel.packed_b {
            ctx.defer_destroy(packed.buf);
        }
        ctx.destroy_pipeline(kernel.pack_pipeline);
        ctx.destroy_pipeline(kernel.matmul_pipeline);
        if let Some((_, pipeline)) = kernel.coop {
            ctx.destroy_pipeline(pipeline);
        }
    }
}
