//! `OrtEpFactory` for the Vulkan EP.
//!
//! Layout: `base` (C struct with function pointers) as the first field, so the
//! `*mut OrtEpFactory` pointer ORT hands us coincides with the pointer to
//! `VulkanEpFactory` (same scheme as the C++ sample `example_plugin_ep_kernel_registry`).

use crate::ep::VulkanEp;
use ort_ep_sys as sys;
use std::ffi::{CStr, c_char};
use std::ptr;

const EP_NAME: &CStr = c"VulkanEP";
const VENDOR: &CStr = c"onnx-vulkan-rs";
const VENDOR_ID: u32 = 0;
const VERSION: &CStr = c"0.1.0";

#[repr(C)]
pub struct VulkanEpFactory {
    base: sys::OrtEpFactory,
    pub ort_api: &'static sys::OrtApi,
    pub ep_api: &'static sys::OrtEpApi,
    /// Kernel registry shared across EP instances (created lazily).
    registry: std::sync::OnceLock<RegistryPtr>,
    /// `OrtMemoryInfo` of the device memory (VRAM) exposed to ORT.
    device_memory_info: *mut sys::OrtMemoryInfo,
}

/// Send/Sync wrapper for the registry pointer (owned by the factory).
struct RegistryPtr(*mut sys::OrtKernelRegistry);
unsafe impl Send for RegistryPtr {}
unsafe impl Sync for RegistryPtr {}

impl VulkanEpFactory {
    pub fn new(
        ort_api: &'static sys::OrtApi,
        ep_api: &'static sys::OrtEpApi,
    ) -> anyhow::Result<Self> {
        // zeroed: all Option<fn> become None; the needed ones are set afterwards
        let mut base: sys::OrtEpFactory = unsafe { std::mem::zeroed() };
        base.ort_version_supported = sys::ORT_API_VERSION;
        base.GetName = Some(get_name);
        base.GetVendor = Some(get_vendor);
        base.GetVendorId = Some(get_vendor_id);
        base.GetVersion = Some(get_version);
        base.GetSupportedDevices = Some(get_supported_devices);
        base.CreateEp = Some(create_ep);
        base.ReleaseEp = Some(release_ep);
        base.CreateAllocator = Some(create_allocator);
        base.ReleaseAllocator = Some(release_allocator);
        base.CreateDataTransfer = Some(create_data_transfer);
        base.IsStreamAware = Some(is_stream_aware);
        base.CreateSyncStreamForDevice = Some(create_sync_stream_for_device);

        // device memory (VRAM): ORT will allocate tensors of claimed nodes here
        let mut device_memory_info: *mut sys::OrtMemoryInfo = ptr::null_mut();
        crate::ort_util::check(
            unsafe {
                (ort_api.CreateMemoryInfo_V2.expect("CreateMemoryInfo_V2"))(
                    c"VulkanEP GPU".as_ptr(),
                    sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
                    /*vendor*/ 0,
                    /*device_id*/ 0,
                    sys::OrtDeviceMemoryType::OrtDeviceMemoryType_DEFAULT,
                    /*alignment*/ 0,
                    sys::OrtAllocatorType::OrtDeviceAllocator,
                    &mut device_memory_info,
                )
            },
            "CreateMemoryInfo_V2",
        )?;

        Ok(Self {
            base,
            ort_api,
            ep_api,
            registry: std::sync::OnceLock::new(),
            device_memory_info,
        })
    }

    unsafe fn from_ptr<'a>(p: *mut sys::OrtEpFactory) -> &'a mut Self {
        unsafe { &mut *p.cast::<Self>() }
    }

    /// Plugin kernel registry, created on first request.
    pub fn kernel_registry(&self) -> anyhow::Result<*const sys::OrtKernelRegistry> {
        if let Some(reg) = self.registry.get() {
            return Ok(reg.0.cast_const());
        }
        let reg = crate::registry::create_registry()?;
        let stored = self.registry.get_or_init(|| RegistryPtr(reg));
        if !std::ptr::eq(stored.0, reg) {
            // lost race: another thread already created the registry
            unsafe {
                (self
                    .ep_api
                    .ReleaseKernelRegistry
                    .expect("ReleaseKernelRegistry"))(reg);
            }
        }
        Ok(stored.0.cast_const())
    }
}

impl Drop for VulkanEpFactory {
    fn drop(&mut self) {
        if let Some(reg) = self.registry.take() {
            unsafe {
                (self
                    .ep_api
                    .ReleaseKernelRegistry
                    .expect("ReleaseKernelRegistry"))(reg.0);
            }
        }
        if !self.device_memory_info.is_null() {
            unsafe {
                (self.ort_api.ReleaseMemoryInfo.expect("ReleaseMemoryInfo"))(
                    self.device_memory_info,
                )
            };
        }
    }
}

unsafe extern "C" fn get_name(_this: *const sys::OrtEpFactory) -> *const c_char {
    EP_NAME.as_ptr()
}

unsafe extern "C" fn get_vendor(_this: *const sys::OrtEpFactory) -> *const c_char {
    VENDOR.as_ptr()
}

unsafe extern "C" fn get_vendor_id(_this: *const sys::OrtEpFactory) -> u32 {
    VENDOR_ID
}

unsafe extern "C" fn get_version(_this: *const sys::OrtEpFactory) -> *const c_char {
    VERSION.as_ptr()
}

/// Claims `OrtHardwareDevice`s of GPU type if Vulkan is available.
/// If ORT exposes no GPU (e.g. WSL2) but Vulkan exists (lavapipe), as a
/// development fallback it claims the CPU device.
unsafe extern "C" fn get_supported_devices(
    this_ptr: *mut sys::OrtEpFactory,
    devices: *const *const sys::OrtHardwareDevice,
    num_devices: usize,
    ep_devices: *mut *mut sys::OrtEpDevice,
    max_ep_devices: usize,
    num_ep_devices: *mut usize,
) -> sys::OrtStatusPtr {
    let factory = unsafe { VulkanEpFactory::from_ptr(this_ptr) };
    unsafe { *num_ep_devices = 0 };

    let vk_devices = match vk_compute::VkContext::enumerate_devices() {
        Ok(d) if !d.is_empty() => d,
        Ok(_) | Err(_) => {
            log::warn!("VulkanEP: no Vulkan device available, EP not offered");
            return ptr::null_mut();
        }
    };
    log::info!(
        "VulkanEP: Vulkan devices found: {:?}",
        vk_devices
            .iter()
            .map(|(n, _, _)| n.as_str())
            .collect::<Vec<_>>()
    );

    let hw_type_of = factory
        .ort_api
        .HardwareDevice_Type
        .expect("HardwareDevice_Type");
    let create_ep_device = factory.ep_api.CreateEpDevice.expect("CreateEpDevice");
    let hw_devices = unsafe { std::slice::from_raw_parts(devices, num_devices) };

    let mut count = 0usize;
    for pass in [
        sys::OrtHardwareDeviceType::OrtHardwareDeviceType_GPU,
        sys::OrtHardwareDeviceType::OrtHardwareDeviceType_CPU,
    ] {
        for &hw in hw_devices {
            if count >= max_ep_devices {
                break;
            }
            if unsafe { hw_type_of(hw) } != pass {
                continue;
            }
            let mut ep_device: *mut sys::OrtEpDevice = ptr::null_mut();
            let status =
                unsafe { create_ep_device(this_ptr, hw, ptr::null(), ptr::null(), &mut ep_device) };
            if !status.is_null() {
                return status;
            }
            // register device memory: enables allocator + data transfer
            let add_info = factory
                .ep_api
                .EpDevice_AddAllocatorInfo
                .expect("AddAllocatorInfo");
            let status = unsafe { add_info(ep_device, factory.device_memory_info) };
            if !status.is_null() {
                return status;
            }
            unsafe {
                *ep_devices.add(count) = ep_device;
            }
            count += 1;
            if pass == sys::OrtHardwareDeviceType::OrtHardwareDeviceType_CPU {
                break; // fallback dev: a single CPU device
            }
        }
        if count > 0 {
            break; // GPUs found: no CPU fallback
        }
    }

    unsafe { *num_ep_devices = count };
    log::info!("VulkanEP: offered {count} EP devices");
    ptr::null_mut()
}

unsafe extern "C" fn create_ep(
    this_ptr: *mut sys::OrtEpFactory,
    _devices: *const *const sys::OrtHardwareDevice,
    _ep_metadata_pairs: *const *const sys::OrtKeyValuePairs,
    num_devices: usize,
    _session_options: *const sys::OrtSessionOptions,
    _logger: *const sys::OrtLogger,
    ep: *mut *mut sys::OrtEp,
) -> sys::OrtStatusPtr {
    let factory = unsafe { VulkanEpFactory::from_ptr(this_ptr) };
    unsafe { *ep = ptr::null_mut() };

    if num_devices != 1 {
        let create_status = factory.ort_api.CreateStatus.expect("CreateStatus");
        return unsafe {
            create_status(
                sys::OrtErrorCode::ORT_INVALID_ARGUMENT,
                c"VulkanEP supports selecting a single device".as_ptr(),
            )
        };
    }

    let instance = Box::new(VulkanEp::new(EP_NAME, factory as *mut VulkanEpFactory));
    unsafe { *ep = Box::into_raw(instance).cast::<sys::OrtEp>() };
    log::info!("VulkanEP: EP instance created");
    ptr::null_mut()
}

unsafe extern "C" fn release_ep(_this: *mut sys::OrtEpFactory, ep: *mut sys::OrtEp) {
    if !ep.is_null() {
        drop(unsafe { Box::from_raw(ep.cast::<VulkanEp>()) });
    }
}

unsafe extern "C" fn create_allocator(
    this_ptr: *mut sys::OrtEpFactory,
    memory_info: *const sys::OrtMemoryInfo,
    _allocator_options: *const sys::OrtKeyValuePairs,
    allocator: *mut *mut sys::OrtAllocator,
) -> sys::OrtStatusPtr {
    let _factory = unsafe { VulkanEpFactory::from_ptr(this_ptr) };
    log::info!("VulkanEP: CreateAllocator chiamata");
    let boxed = Box::new(crate::device_mem::VulkanOrtAllocator::new(memory_info));
    unsafe { *allocator = Box::into_raw(boxed).cast::<sys::OrtAllocator>() };
    ptr::null_mut()
}

unsafe extern "C" fn release_allocator(
    _this: *mut sys::OrtEpFactory,
    allocator: *mut sys::OrtAllocator,
) {
    if !allocator.is_null() {
        drop(unsafe { Box::from_raw(allocator.cast::<crate::device_mem::VulkanOrtAllocator>()) });
    }
}

unsafe extern "C" fn create_data_transfer(
    _this: *mut sys::OrtEpFactory,
    data_transfer: *mut *mut sys::OrtDataTransferImpl,
) -> sys::OrtStatusPtr {
    log::info!("VulkanEP: CreateDataTransfer chiamata");
    unsafe { *data_transfer = crate::data_transfer::VulkanDataTransfer::new_boxed() };
    ptr::null_mut()
}

unsafe extern "C" fn is_stream_aware(_this: *const sys::OrtEpFactory) -> bool {
    false
}

unsafe extern "C" fn create_sync_stream_for_device(
    _this: *mut sys::OrtEpFactory,
    _memory_device: *const sys::OrtMemoryDevice,
    _stream_options: *const sys::OrtKeyValuePairs,
    stream: *mut *mut sys::OrtSyncStreamImpl,
) -> sys::OrtStatusPtr {
    unsafe { *stream = ptr::null_mut() };
    ptr::null_mut()
}
