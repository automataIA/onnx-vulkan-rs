//! `OrtDataTransferImpl`: CPU↔GPU and GPU↔GPU copies for VulkanEP tensors.
//!
//! CPU→GPU and GPU→GPU copies are enqueued on the stream (no submit);
//! GPU→CPU forces stream flush (only sync point).

use crate::device_mem::region_from_ptr;
use crate::ort_util::{apis, check, to_status};
use crate::vk;
use anyhow::{Result, bail};
use ort_ep_sys as sys;
use std::ffi::c_void;

#[repr(C)]
pub struct VulkanDataTransfer {
    base: sys::OrtDataTransferImpl,
}

impl VulkanDataTransfer {
    pub fn new_boxed() -> *mut sys::OrtDataTransferImpl {
        let mut base: sys::OrtDataTransferImpl = unsafe { std::mem::zeroed() };
        base.ort_version_supported = sys::ORT_API_VERSION;
        base.Release = Some(release_impl);
        base.CanCopy = Some(can_copy_impl);
        base.CopyTensors = Some(copy_tensors_impl);
        Box::into_raw(Box::new(Self { base })).cast::<sys::OrtDataTransferImpl>()
    }
}

unsafe extern "C" fn release_impl(this_ptr: *mut sys::OrtDataTransferImpl) {
    if !this_ptr.is_null() {
        drop(unsafe { Box::from_raw(this_ptr.cast::<VulkanDataTransfer>()) });
    }
}

fn device_type(d: *const sys::OrtMemoryDevice) -> sys::OrtMemoryInfoDeviceType {
    let ep_api = apis().ep;
    unsafe {
        (ep_api
            .MemoryDevice_GetDeviceType
            .expect("MemoryDevice_GetDeviceType"))(d)
    }
}

unsafe extern "C" fn can_copy_impl(
    _this: *const sys::OrtDataTransferImpl,
    src: *const sys::OrtMemoryDevice,
    dst: *const sys::OrtMemoryDevice,
) -> bool {
    use sys::OrtMemoryInfoDeviceType as D;
    let (s, d) = (device_type(src), device_type(dst));
    matches!(
        (s, d),
        (
            D::OrtMemoryInfoDeviceType_GPU,
            D::OrtMemoryInfoDeviceType_CPU
        ) | (
            D::OrtMemoryInfoDeviceType_CPU,
            D::OrtMemoryInfoDeviceType_GPU
        ) | (
            D::OrtMemoryInfoDeviceType_GPU,
            D::OrtMemoryInfoDeviceType_GPU
        )
    )
}

/// Byte size of OrtValue tensor.
///
/// # Safety
/// `value` must be a valid OrtValue tensor.
pub unsafe fn value_byte_size(value: *const sys::OrtValue) -> Result<usize> {
    unsafe { value_data_size(value).map(|(_, s)| s) }
}

/// (data ptr, byte size) of OrtValue tensor.
unsafe fn value_data_size(value: *const sys::OrtValue) -> Result<(*mut c_void, usize)> {
    let api = apis().ort;
    unsafe {
        let mut info: *mut sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        check(
            (api.GetTensorTypeAndShape.expect("GetTensorTypeAndShape"))(value, &mut info),
            "GetTensorTypeAndShape",
        )?;
        let mut count = 0usize;
        let mut elem_type = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        let r = check(
            (api.GetTensorShapeElementCount
                .expect("GetTensorShapeElementCount"))(info, &mut count),
            "GetTensorShapeElementCount",
        )
        .and_then(|()| {
            check(
                (api.GetTensorElementType.expect("GetTensorElementType"))(info, &mut elem_type),
                "GetTensorElementType",
            )
        });
        (api.ReleaseTensorTypeAndShapeInfo
            .expect("ReleaseTensorTypeAndShapeInfo"))(info);
        r?;

        let mut data: *mut c_void = std::ptr::null_mut();
        check(
            (api.GetTensorMutableData.expect("GetTensorMutableData"))(value.cast_mut(), &mut data),
            "GetTensorMutableData",
        )?;
        Ok((data, count * elem_byte_size(elem_type)?))
    }
}

pub fn elem_byte_size(t: sys::ONNXTensorElementDataType) -> Result<usize> {
    use sys::ONNXTensorElementDataType as T;
    Ok(match t {
        T::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
        | T::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        | T::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => 1,
        T::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
        | T::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
        | T::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
        | T::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => 2,
        T::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        | T::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        | T::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => 4,
        T::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
        | T::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        | T::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => 8,
        other => bail!("unsupported tensor type: {other:?}"),
    })
}

unsafe extern "C" fn copy_tensors_impl(
    _this: *mut sys::OrtDataTransferImpl,
    src_tensors: *mut *const sys::OrtValue,
    dst_tensors: *mut *mut sys::OrtValue,
    _streams: *mut *mut sys::OrtSyncStream,
    num_tensors: usize,
) -> sys::OrtStatusPtr {
    to_status(unsafe { copy_tensors(src_tensors, dst_tensors, num_tensors) })
}

unsafe fn copy_tensors(
    src_tensors: *mut *const sys::OrtValue,
    dst_tensors: *mut *mut sys::OrtValue,
    num_tensors: usize,
) -> Result<()> {
    use sys::OrtMemoryInfoDeviceType as D;
    let ep_api = apis().ep;
    let ctx = vk::context()?;
    let srcs = unsafe { std::slice::from_raw_parts(src_tensors, num_tensors) };
    let dsts = unsafe { std::slice::from_raw_parts(dst_tensors, num_tensors) };

    for (&src, &dst) in srcs.iter().zip(dsts) {
        let src_dev =
            unsafe { (ep_api.Value_GetMemoryDevice.expect("Value_GetMemoryDevice"))(src) };
        let dst_dev = unsafe {
            (ep_api.Value_GetMemoryDevice.expect("Value_GetMemoryDevice"))(dst.cast_const())
        };
        let (src_ptr, src_len) = unsafe { value_data_size(src)? };
        let (dst_ptr, dst_len) = unsafe { value_data_size(dst.cast_const())? };
        let bytes = src_len.min(dst_len);
        if bytes == 0 {
            continue;
        }
        match (device_type(src_dev), device_type(dst_dev)) {
            (D::OrtMemoryInfoDeviceType_CPU, D::OrtMemoryInfoDeviceType_GPU) => {
                let dst = unsafe { region_from_ptr(dst_ptr)? };
                let data = unsafe { std::slice::from_raw_parts(src_ptr.cast::<u8>(), bytes) };
                ctx.stream_upload_at(dst.buffer(), dst.offset, data)?;
            }
            (D::OrtMemoryInfoDeviceType_GPU, D::OrtMemoryInfoDeviceType_CPU) => {
                let src = unsafe { region_from_ptr(src_ptr)? };
                let data = ctx.stream_download_at(src.buffer(), src.offset, bytes)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst_ptr.cast::<u8>(), bytes)
                };
            }
            (D::OrtMemoryInfoDeviceType_GPU, D::OrtMemoryInfoDeviceType_GPU) => {
                let s = unsafe { region_from_ptr(src_ptr)? };
                let d = unsafe { region_from_ptr(dst_ptr)? };
                ctx.stream_copy_range(s.buffer(), s.offset, d.buffer(), d.offset, bytes as u64)?;
            }
            other => bail!("unsupported copy direction: {other:?}"),
        }
    }
    Ok(())
}
