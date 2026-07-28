//! Helpers for using the ORT C API inside the plugin.

use ort_ep_sys as sys;
use std::ffi::CString;

/// API context shared by the plugin (set in `CreateEpFactories`).
pub struct Apis {
    pub ort: &'static sys::OrtApi,
    pub ep: &'static sys::OrtEpApi,
}

static APIS: std::sync::OnceLock<Apis> = std::sync::OnceLock::new();

pub fn set_apis(ort: &'static sys::OrtApi, ep: &'static sys::OrtEpApi) {
    let _ = APIS.set(Apis { ort, ep });
}

pub fn apis() -> &'static Apis {
    APIS.get().expect("APIs ORT non inizializzate")
}

/// Creates an error `OrtStatus` from the given message.
pub fn error_status(msg: &str) -> sys::OrtStatusPtr {
    let api = apis().ort;
    let create = api.CreateStatus.expect("CreateStatus");
    let c = CString::new(msg).unwrap_or_else(|_| CString::new("VulkanEP error").unwrap());
    unsafe { create(sys::OrtErrorCode::ORT_EP_FAIL, c.as_ptr()) }
}

/// Converts a `Result` into an `OrtStatusPtr` (null = success).
pub fn to_status(result: anyhow::Result<()>) -> sys::OrtStatusPtr {
    match result {
        Ok(()) => std::ptr::null_mut(),
        Err(e) => {
            log::error!("VulkanEP: {e:#}");
            error_status(&format!("{e:#}"))
        }
    }
}

/// Propagates a non-null ORT status as an anyhow error (and releases it).
pub fn check(status: sys::OrtStatusPtr, what: &str) -> anyhow::Result<()> {
    if status.is_null() {
        return Ok(());
    }
    let api = apis().ort;
    let msg = unsafe {
        std::ffi::CStr::from_ptr((api.GetErrorMessage.expect("GetErrorMessage"))(status))
    }
    .to_string_lossy()
    .into_owned();
    unsafe { (api.ReleaseStatus.expect("ReleaseStatus"))(status) };
    anyhow::bail!("{what}: {msg}")
}

/// Data and shape of a kernel-context input tensor (CPU-resident).
pub struct TensorView {
    pub data: *const u8,
    pub elem_count: usize,
    pub shape: Vec<i64>,
    /// Bytes per element: with the memory pattern the tensor is a slice of
    /// the allocation, so its size cannot be deduced from the buffer.
    pub elem_size: usize,
}

/// # Safety
/// `ctx` valid, `index` within the node's input count.
pub unsafe fn kernel_input(
    ctx: *const sys::OrtKernelContext,
    index: usize,
) -> anyhow::Result<TensorView> {
    let api = apis().ort;
    unsafe {
        let mut value: *const sys::OrtValue = std::ptr::null();
        check(
            (api.KernelContext_GetInput.expect("KernelContext_GetInput"))(ctx, index, &mut value),
            "KernelContext_GetInput",
        )?;
        anyhow::ensure!(!value.is_null(), "input {index} is null");
        value_view(value)
    }
}

/// Data+shape view of a CPU-resident `OrtValue` tensor.
///
/// # Safety
/// `value` must be a valid OrtValue tensor.
pub unsafe fn value_view(value: *const sys::OrtValue) -> anyhow::Result<TensorView> {
    let api = apis().ort;
    unsafe {
        let mut info: *mut sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        check(
            (api.GetTensorTypeAndShape.expect("GetTensorTypeAndShape"))(value, &mut info),
            "GetTensorTypeAndShape",
        )?;
        let mut num_dims = 0usize;
        check(
            (api.GetDimensionsCount.expect("GetDimensionsCount"))(info, &mut num_dims),
            "GetDimensionsCount",
        )?;
        let mut shape = vec![0i64; num_dims];
        check(
            (api.GetDimensions.expect("GetDimensions"))(info, shape.as_mut_ptr(), num_dims),
            "GetDimensions",
        )?;
        let mut elem_count = 0usize;
        check(
            (api.GetTensorShapeElementCount
                .expect("GetTensorShapeElementCount"))(info, &mut elem_count),
            "GetTensorShapeElementCount",
        )?;
        let mut elem_type = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        check(
            (api.GetTensorElementType.expect("GetTensorElementType"))(info, &mut elem_type),
            "GetTensorElementType",
        )?;
        (api.ReleaseTensorTypeAndShapeInfo
            .expect("ReleaseTensorTypeAndShapeInfo"))(info);
        let elem_size = onnx_vulkan_core::elem_size(elem_type as i32);

        let mut data: *mut std::ffi::c_void = std::ptr::null_mut();
        check(
            (api.GetTensorMutableData.expect("GetTensorMutableData"))(value.cast_mut(), &mut data),
            "GetTensorMutableData",
        )?;
        Ok(TensorView {
            data: data.cast_const().cast::<u8>(),
            elem_count,
            shape,
            elem_size,
        })
    }
}

/// Float attribute of the node, with default if absent.
///
/// # Safety
/// `info` must be a valid OrtKernelInfo.
pub unsafe fn attr_f32(
    info: *const sys::OrtKernelInfo,
    name: &std::ffi::CStr,
    default: f32,
) -> f32 {
    let api = apis().ort;
    let mut out = 0f32;
    let status = unsafe {
        (api.KernelInfoGetAttribute_float
            .expect("GetAttribute_float"))(info, name.as_ptr(), &mut out)
    };
    if status.is_null() {
        out
    } else {
        unsafe { (api.ReleaseStatus.expect("ReleaseStatus"))(status) };
        default
    }
}

/// Int64 attribute of the node, with default if absent.
///
/// # Safety
/// `info` must be a valid OrtKernelInfo.
pub unsafe fn attr_i64(
    info: *const sys::OrtKernelInfo,
    name: &std::ffi::CStr,
    default: i64,
) -> i64 {
    let api = apis().ort;
    let mut out = 0i64;
    let status = unsafe {
        (api.KernelInfoGetAttribute_int64
            .expect("GetAttribute_int64"))(info, name.as_ptr(), &mut out)
    };
    if status.is_null() {
        out
    } else {
        unsafe { (api.ReleaseStatus.expect("ReleaseStatus"))(status) };
        default
    }
}

/// Int64-list attribute of the node (e.g. `perm`, `axes`), if present.
///
/// # Safety
/// `info` must be a valid OrtKernelInfo.
pub unsafe fn attr_ints(
    info: *const sys::OrtKernelInfo,
    name: &std::ffi::CStr,
) -> Option<Vec<i64>> {
    let api = apis().ort;
    let get = api.KernelInfoGetAttributeArray_int64?;
    // first call: size
    let mut len = 0usize;
    let status = unsafe { get(info, name.as_ptr(), std::ptr::null_mut(), &mut len) };
    // ORT returns a "shape insufficient" error but populates len; we ignore it
    if !status.is_null() {
        unsafe { (api.ReleaseStatus.expect("ReleaseStatus"))(status) };
    }
    if len == 0 {
        return None;
    }
    let mut out = vec![0i64; len];
    let status = unsafe { get(info, name.as_ptr(), out.as_mut_ptr(), &mut len) };
    if status.is_null() {
        Some(out)
    } else {
        unsafe { (api.ReleaseStatus.expect("ReleaseStatus"))(status) };
        None
    }
}

/// Output tensor with given shape: returns the mutable data pointer.
///
/// # Safety
/// `ctx` valid, `index` within the node's output count.
pub unsafe fn kernel_output(
    ctx: *mut sys::OrtKernelContext,
    index: usize,
    shape: &[i64],
) -> anyhow::Result<*mut u8> {
    let api = apis().ort;
    unsafe {
        let mut value: *mut sys::OrtValue = std::ptr::null_mut();
        check(
            (api.KernelContext_GetOutput
                .expect("KernelContext_GetOutput"))(
                ctx,
                index,
                shape.as_ptr(),
                shape.len(),
                &mut value,
            ),
            "KernelContext_GetOutput",
        )?;
        let mut data: *mut std::ffi::c_void = std::ptr::null_mut();
        check(
            (api.GetTensorMutableData.expect("GetTensorMutableData"))(value, &mut data),
            "GetTensorMutableData",
        )?;
        Ok(data.cast::<u8>())
    }
}
