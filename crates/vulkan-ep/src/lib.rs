//! Vulkan Execution Provider plugin for ONNX Runtime (cdylib).
//!
//! Exports `CreateEpFactories` / `ReleaseEpFactory` (Plugin EP API, ORT ≥1.23).
//! M3: skeleton — the factory enumerates Vulkan devices, the EP claims no
//! nodes (empty `GetCapability`) → the whole graph stays on the CPU EP.

mod compile;
mod data_transfer;
mod device_mem;
mod ep;
mod factory;
mod kernels;
mod ort_util;
mod registry;
mod vk;

use ort_ep_sys as sys;
use std::ffi::c_char;

use factory::VulkanEpFactory;

/// # Safety
/// Called by ONNX Runtime with valid pointers per the Plugin EP API
/// contract (`onnxruntime_ep_c_api.h`).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn CreateEpFactories(
    _registration_name: *const c_char,
    ort_api_base: *const sys::OrtApiBase,
    _default_logger: *const sys::OrtLogger,
    factories: *mut *mut sys::OrtEpFactory,
    max_factories: usize,
    num_factories: *mut usize,
) -> *mut sys::OrtStatus {
    let _ = env_logger::try_init();

    let api_base = unsafe { &*ort_api_base };
    let get_api = api_base.GetApi.expect("OrtApiBase::GetApi is null");
    let ort_api = unsafe { get_api(sys::ORT_API_VERSION) };
    if ort_api.is_null() {
        // runtime older than the headers we are compiled against
        return std::ptr::null_mut();
    }
    let ort_api = unsafe { &*ort_api };
    let get_ep_api = ort_api.GetEpApi.expect("OrtApi::GetEpApi is null");
    let ep_api = unsafe { &*get_ep_api() };
    ort_util::set_apis(ort_api, ep_api);

    if max_factories < 1 {
        let create_status = ort_api.CreateStatus.expect("CreateStatus is null");
        return unsafe {
            create_status(
                sys::OrtErrorCode::ORT_INVALID_ARGUMENT,
                c"room for at least one factory is required".as_ptr(),
            )
        };
    }

    let factory = match VulkanEpFactory::new(ort_api, ep_api) {
        Ok(f) => Box::new(f),
        Err(e) => {
            log::error!("VulkanEP: creazione factory fallita: {e:#}");
            return ort_util::error_status(&format!("creazione factory fallita: {e:#}")).cast();
        }
    };
    unsafe {
        *factories = Box::into_raw(factory).cast::<sys::OrtEpFactory>();
        *num_factories = 1;
    }
    log::info!(
        "VulkanEP factory creata (ORT_API_VERSION={})",
        sys::ORT_API_VERSION
    );
    std::ptr::null_mut()
}

/// # Safety
/// `factory` must be a pointer returned by [`CreateEpFactories`].
#[unsafe(no_mangle)]
pub unsafe extern "system" fn ReleaseEpFactory(
    factory: *mut sys::OrtEpFactory,
) -> *mut sys::OrtStatus {
    if !factory.is_null() {
        drop(unsafe { Box::from_raw(factory.cast::<VulkanEpFactory>()) });
    }
    std::ptr::null_mut()
}
