//! f32 elementwise kernels: binary with ONNX broadcasting (Mul, Add),
//! unary (Sigmoid, Relu), and Cast i32→f32.
//!
//! Broadcasting: for each output dimension the shader is given
//! out_stride (to decompose the linear index) and the strides of A and B
//! (0 on broadcast dimensions). Max rank 8 (2×vec4 in push constants).

use crate::kernels::{base_kernel_impl, device_in, device_out};
use crate::ort_util::to_status;
use crate::vk;
use anyhow::{Result, ensure};
use onnx_vulkan_core::broadcast;
use onnx_vulkan_core::shaders::elementwise::{
    BINARY as BINARY_TEMPLATE, CAST_I32_F32, MAX_RANK, UNARY as UNARY_TEMPLATE,
};
use onnx_vulkan_core::shaders::push_vec4s;
use ort_ep_sys as sys;
use vk_compute::{ComputePipeline, compile_wgsl};

#[derive(Clone, Copy)]
enum OpKind {
    Binary(&'static str),
    Unary(&'static str),
    CastI32F32,
}

#[repr(C)]
struct ElementwiseKernel {
    base: sys::OrtKernelImpl,
    kind: OpKind,
    pipeline: ComputePipeline,
}

fn make_create(kind: OpKind, kernel_out: *mut *mut sys::OrtKernelImpl) -> *mut sys::OrtStatus {
    let result = (|| -> Result<()> {
        let ctx = vk::context()?;
        let (src, bindings, push) = match kind {
            OpKind::Binary(op) => (BINARY_TEMPLATE.replace("OP", op), 3, 112),
            OpKind::Unary(op) => (UNARY_TEMPLATE.replace("OP", op), 2, 4),
            OpKind::CastI32F32 => (CAST_I32_F32.to_string(), 2, 4),
        };
        let kernel = Box::new(ElementwiseKernel {
            base: base_kernel_impl(compute, release),
            kind,
            pipeline: ctx.create_pipeline(&compile_wgsl(&src)?, bindings, push)?,
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
            _info: *const sys::OrtKernelInfo,
            kernel_out: *mut *mut sys::OrtKernelImpl,
        ) -> *mut sys::OrtStatus {
            make_create($kind, kernel_out)
        }
    };
}

create_fn!(create_mul, OpKind::Binary("a[off_a] * b[off_b]"));
create_fn!(create_add, OpKind::Binary("a[off_a] + b[off_b]"));
create_fn!(create_sigmoid, OpKind::Unary("1.0 / (1.0 + exp(-v))"));
create_fn!(create_relu, OpKind::Unary("max(v, 0.0)"));
create_fn!(create_cast, OpKind::CastI32F32);

unsafe extern "C" fn compute(
    this_ptr: *mut sys::OrtKernelImpl,
    ctx_ptr: *mut sys::OrtKernelContext,
) -> sys::OrtStatusPtr {
    vk_compute::stats::set_op("elementwise");
    let kernel = unsafe { &*this_ptr.cast::<ElementwiseKernel>() };
    to_status(unsafe { compute_impl(kernel, ctx_ptr) })
}

unsafe fn compute_impl(
    kernel: &ElementwiseKernel,
    ctx_ptr: *mut sys::OrtKernelContext,
) -> Result<()> {
    let ctx = vk::context()?;
    match kernel.kind {
        OpKind::Binary(_) => {
            let (a, a_shape, _) = unsafe { device_in(ctx_ptr, 0)? };
            let (b, b_shape, _) = unsafe { device_in(ctx_ptr, 1)? };
            let bc = broadcast(&a_shape, &b_shape)?;
            ensure!(
                bc.out_shape.len() <= MAX_RANK,
                "broadcast: rank {} > {MAX_RANK}",
                bc.out_shape.len()
            );
            let n: usize = bc.out_shape.iter().product::<i64>() as usize;
            let out = unsafe { device_out(ctx_ptr, 0, &bc.out_shape)? };
            if n == 0 {
                return Ok(());
            }
            let mut push = Vec::with_capacity(112);
            push.extend_from_slice(&(n as u32).to_le_bytes());
            push.extend_from_slice(&(bc.out_shape.len() as u32).to_le_bytes());
            push.extend_from_slice(&0u32.to_le_bytes());
            push.extend_from_slice(&0u32.to_le_bytes());
            push_vec4s(&mut push, &bc.out_strides);
            push_vec4s(&mut push, &bc.a_strides);
            push_vec4s(&mut push, &bc.b_strides);
            ctx.stream_dispatch_slices(
                &kernel.pipeline,
                &[a.slice(), b.slice(), out.slice()],
                &push,
                [(n as u32).div_ceil(256), 1, 1],
            )?;
        }
        OpKind::Unary(_) | OpKind::CastI32F32 => {
            let (x, x_shape, n) = unsafe { device_in(ctx_ptr, 0)? };
            let out = unsafe { device_out(ctx_ptr, 0, &x_shape)? };
            if n == 0 {
                return Ok(());
            }
            ctx.stream_dispatch_slices(
                &kernel.pipeline,
                &[x.slice(), out.slice()],
                &(n as u32).to_le_bytes(),
                [(n as u32).div_ceil(256), 1, 1],
            )?;
        }
    }
    Ok(())
}

unsafe extern "C" fn release(this_ptr: *mut sys::OrtKernelImpl) {
    if this_ptr.is_null() {
        return;
    }
    let kernel = unsafe { Box::from_raw(this_ptr.cast::<ElementwiseKernel>()) };
    if let Ok(ctx) = vk::context() {
        let _ = ctx.flush();
        ctx.destroy_pipeline(kernel.pipeline);
    }
}
