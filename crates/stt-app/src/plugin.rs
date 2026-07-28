//! Registration of the Vulkan EP plugin via the raw Plugin EP API (`ort::sys`).
//!
//! The `ort` crate does not yet expose these APIs safely, but `ort::api()`
//! and `AsPointer` give access to the raw `OrtApi`/`OrtEnv`/`OrtSessionOptions`.

#[cfg(not(windows))]
use anyhow::Context;
use anyhow::{Result, bail};
use ort::AsPointer;
use ort::session::builder::SessionBuilder;
use ort::sys;
use std::ffi::{CStr, CString};
use std::path::Path;

pub const EP_NAME: &str = "VulkanEP";

fn check_status(status: sys::OrtStatusPtr, what: &str) -> Result<()> {
    if status.0.is_null() {
        return Ok(());
    }
    let api = ort::api();
    let msg = unsafe { CStr::from_ptr((api.GetErrorMessage)(status.0)) }
        .to_string_lossy()
        .into_owned();
    unsafe { (api.ReleaseStatus)(status.0) };
    bail!("{what}: {msg}");
}

fn env_ptr() -> Result<*mut sys::OrtEnv> {
    let env = ort::environment::get_environment()?;
    Ok(env.ptr().cast_mut())
}

/// Registers the plugin library in the current ORT environment.
/// `ortchar` is `c_char` on Linux and `u16` (wide string) on Windows.
pub fn register(plugin_path: &Path) -> Result<()> {
    let api = ort::api();
    let name = CString::new(EP_NAME)?;

    #[cfg(windows)]
    let status = {
        use std::os::windows::ffi::OsStrExt;
        let mut path: Vec<u16> = plugin_path.as_os_str().encode_wide().collect();
        path.push(0);
        unsafe { (api.RegisterExecutionProviderLibrary)(env_ptr()?, name.as_ptr(), path.as_ptr()) }
    };
    #[cfg(not(windows))]
    let status = {
        let path = CString::new(plugin_path.to_str().context("plugin path is not UTF-8")?)?;
        unsafe { (api.RegisterExecutionProviderLibrary)(env_ptr()?, name.as_ptr(), path.as_ptr()) }
    };
    check_status(status, "RegisterExecutionProviderLibrary")
}

/// Unregisters the library. Call only after destroying all
/// sessions that use the EP.
pub fn unregister() -> Result<()> {
    let api = ort::api();
    let name = CString::new(EP_NAME)?;
    let status = unsafe { (api.UnregisterExecutionProviderLibrary)(env_ptr()?, name.as_ptr()) };
    check_status(status, "UnregisterExecutionProviderLibrary")
}

/// Appends VulkanEP device EPs to the session (selection via
/// `SessionOptionsAppendExecutionProvider_V2`). Returns device count.
pub fn append_to_session(builder: &mut SessionBuilder) -> Result<usize> {
    let api = ort::api();
    let env = env_ptr()?;

    let mut devices: *const *const sys::OrtEpDevice = std::ptr::null();
    let mut num_devices: usize = 0;
    let status = unsafe { (api.GetEpDevices)(env, &mut devices, &mut num_devices) };
    check_status(status, "GetEpDevices")?;

    let all = unsafe { std::slice::from_raw_parts(devices, num_devices) };
    let ours: Vec<*const sys::OrtEpDevice> = all
        .iter()
        .copied()
        .filter(|&d| {
            let name = unsafe { CStr::from_ptr((api.EpDevice_EpName)(d)) };
            name.to_string_lossy() == EP_NAME
        })
        .collect();

    if ours.is_empty() {
        bail!("no EP device {EP_NAME} available (GetEpDevices: {num_devices} total)");
    }
    // The EP handles one device per session: the first one is used
    let selected = &ours[..1];

    let status = unsafe {
        (api.SessionOptionsAppendExecutionProvider_V2)(
            builder.ptr_mut(),
            env,
            selected.as_ptr(),
            selected.len(),
            std::ptr::null(),
            std::ptr::null(),
            0,
        )
    };
    check_status(status, "SessionOptionsAppendExecutionProvider_V2")?;
    Ok(selected.len())
}
