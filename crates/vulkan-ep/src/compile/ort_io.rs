//! I/O adapter between `OrtKernelContext` and standalone interpreter.

use crate::device_mem::region_from_ptr;
use crate::ort_util::{apis, check};
use crate::vk;
use anyhow::{Result, bail, ensure};
use onnx_vulkan_core::host_ops::{self, HostTensor};
use onnx_vulkan_core::{
    DeviceBuffer, DeviceTensor, Executor, Outputs, Tensor, device_storage_bytes, element_count,
    storage_len,
};
use ort_ep_sys as sys;

/// Executes graph translating values only at ORT boundary.
///
/// The graph runs through the core's public `Executor`: this adapter only
/// translates values at the boundary, and shares the execution path with any
/// other host.
///
/// # Safety
/// `ctx` valid for call duration and names follow ORT binding order.
pub unsafe fn run(
    executor: &Executor<'static>,
    boundary_inputs: &[String],
    boundary_outputs: &[String],
    ctx: *mut sys::OrtKernelContext,
) -> Result<()> {
    let vkctx = vk::context()?;

    let mut inputs = Vec::with_capacity(boundary_inputs.len());
    for (index, name) in boundary_inputs.iter().enumerate() {
        if !name.is_empty() {
            inputs.push((name.as_str(), unsafe { read_input(vkctx, ctx, index)? }));
        }
    }

    let outputs = executor.run(inputs)?;

    for (index, name) in boundary_outputs.iter().enumerate() {
        unsafe { write_output(vkctx, ctx, index, &outputs, name)? };
    }

    // copies to ORT buffers are enqueued with the dispatches: the submit
    // includes them, and remains the only synchronization point
    vkctx.flush()?;
    outputs.finish();
    Ok(())
}

/// # Safety
/// `ctx` is valid and `index` identifies an existing input.
unsafe fn read_input<'a>(
    vkctx: &vk_compute::VkContext,
    ctx: *const sys::OrtKernelContext,
    index: usize,
) -> Result<Tensor<'a>> {
    let api = apis().ort;
    let mut value: *const sys::OrtValue = std::ptr::null();
    check(
        unsafe {
            (api.KernelContext_GetInput.expect("KernelContext_GetInput"))(ctx, index, &mut value)
        },
        "KernelContext_GetInput",
    )?;
    ensure!(!value.is_null(), "input {index} is null");
    let (dtype, shape, data) = unsafe { value_meta(value)? };
    let elem_count = element_count(&shape)?;

    if let Ok(region) = unsafe { region_from_ptr(data.cast()) } {
        if region.is_whole() {
            return Ok(Tensor::Device(DeviceTensor {
                dtype,
                shape,
                elem_count,
                buf: DeviceBuffer::Borrowed(&region.entry.buf),
            }));
        }
        // the tensor is a slice of a larger allocation (ORT memory pattern):
        // kernels bind the whole buffer, so the slice must be copied into a
        // buffer of its own.
        let byte_len = storage_len(dtype, elem_count)
            .ok_or_else(|| anyhow::anyhow!("dtype {dtype} has no fixed storage size"))?;
        let owned = vkctx.create_storage_buffer(device_storage_bytes(dtype, elem_count)?)?;
        vkctx.stream_copy_range(&region.entry.buf, region.offset, &owned, 0, byte_len as u64)?;
        return Ok(Tensor::Device(DeviceTensor {
            dtype,
            shape,
            elem_count,
            buf: DeviceBuffer::Owned(owned),
        }));
    }

    let byte_len = host_ops::expected_bytes(dtype, &shape);
    let bytes = if byte_len == 0 || data.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(data, byte_len) }.to_vec()
    };
    Ok(Tensor::Host(HostTensor::new(dtype, shape, bytes)))
}

/// # Safety
/// `ctx` is valid and `index` identifies an existing output.
unsafe fn write_output(
    vkctx: &vk_compute::VkContext,
    ctx: *mut sys::OrtKernelContext,
    index: usize,
    outputs: &Outputs<'_>,
    name: &str,
) -> Result<()> {
    let api = apis().ort;
    let shape = outputs.shape_of(name)?;
    let mut value: *mut sys::OrtValue = std::ptr::null_mut();
    check(
        unsafe {
            (api.KernelContext_GetOutput
                .expect("KernelContext_GetOutput"))(
                ctx,
                index,
                shape.as_ptr(),
                shape.len(),
                &mut value,
            )
        },
        "KernelContext_GetOutput",
    )?;
    let mut data: *mut std::ffi::c_void = std::ptr::null_mut();
    check(
        unsafe { (api.GetTensorMutableData.expect("GetTensorMutableData"))(value, &mut data) },
        "GetTensorMutableData",
    )?;
    let dst_device = unsafe { region_from_ptr(data.cast_const()) }.ok();

    match outputs.value(name) {
        Some(Tensor::Device(tensor)) => {
            let byte_len = storage_len(tensor.dtype, tensor.elem_count)
                .ok_or_else(|| anyhow::anyhow!("dtype {} has no fixed storage size", tensor.dtype))?;
            if byte_len == 0 {
                return Ok(());
            }
            if let Some(dst) = dst_device {
                vkctx.stream_copy_range(
                    tensor.buffer(),
                    0,
                    &dst.entry.buf,
                    dst.offset,
                    byte_len as u64,
                )?;
            } else {
                ensure!(!data.is_null(), "output {index} is null");
                let bytes = vkctx.stream_download(tensor.buffer(), byte_len)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.cast::<u8>(), bytes.len())
                };
            }
        }
        Some(Tensor::Host(tensor)) => {
            if tensor.data.is_empty() {
                return Ok(());
            }
            if let Some(dst) = dst_device {
                vkctx.stream_upload_at(dst.buffer(), dst.offset, &tensor.data)?;
            } else {
                ensure!(!data.is_null(), "output {index} is null");
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tensor.data.as_ptr(),
                        data.cast::<u8>(),
                        tensor.data.len(),
                    )
                };
            }
        }
        None => bail!("output '{name}' not produced by the subgraph"),
    }
    Ok(())
}

/// Metadata and data pointer for `OrtValue`.
///
/// # Safety
/// `value` must be a valid ORT tensor.
unsafe fn value_meta(value: *const sys::OrtValue) -> Result<(i32, Vec<i64>, *const u8)> {
    let api = apis().ort;
    unsafe {
        let mut info: *mut sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        check(
            (api.GetTensorTypeAndShape.expect("GetTensorTypeAndShape"))(value, &mut info),
            "GetTensorTypeAndShape",
        )?;
        let mut dtype = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        check(
            (api.GetTensorElementType.expect("GetTensorElementType"))(info, &mut dtype),
            "GetTensorElementType",
        )?;
        let mut dimension_count = 0usize;
        check(
            (api.GetDimensionsCount.expect("GetDimensionsCount"))(info, &mut dimension_count),
            "GetDimensionsCount",
        )?;
        let mut shape = vec![0i64; dimension_count];
        check(
            (api.GetDimensions.expect("GetDimensions"))(info, shape.as_mut_ptr(), dimension_count),
            "GetDimensions",
        )?;
        (api.ReleaseTensorTypeAndShapeInfo
            .expect("ReleaseTensorTypeAndShapeInfo"))(info);

        let mut data: *mut std::ffi::c_void = std::ptr::null_mut();
        check(
            (api.GetTensorMutableData.expect("GetTensorMutableData"))(value.cast_mut(), &mut data),
            "GetTensorMutableData",
        )?;
        Ok((dtype as i32, shape, data.cast::<u8>()))
    }
}
