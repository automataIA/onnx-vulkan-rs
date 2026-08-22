//! Registration of a plugin execution provider via the raw Plugin EP API
//! (`ort::sys`).
//!
//! The `ort` crate does not yet expose these APIs safely, but `ort::api()`
//! and `AsPointer` give access to the raw `OrtApi`/`OrtEnv`/`OrtSessionOptions`.
//!
//! The sequence is the same for every plugin EP — ours and third-party ones
//! alike — so only the registration name and the library path vary. Both
//! `stt-app` and `model-runner` used to carry a byte-identical copy of this
//! file hardcoded to `VulkanEP`; the parameterised form is what lets
//! `model-runner` also drive Microsoft's standalone WebGPU EP as a reference
//! backend.

#[cfg(not(windows))]
use anyhow::Context;
use anyhow::{Result, bail};
use ort::AsPointer;
use ort::session::builder::SessionBuilder;
use ort::sys;
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};

/// Registration name of our own plugin.
pub const VULKAN_EP_NAME: &str = "VulkanEP";

/// EP name Microsoft's standalone WebGPU plugin reports from
/// `EpDevice_EpName`. Not a name we choose: it must match what the library
/// announces, or the device filter in [`PluginEp::append_to_session`] finds
/// nothing.
pub const WEBGPU_EP_NAME: &str = "WebGpuExecutionProvider";

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

/// A plugin EP identified by the name it registers under and the dylib that
/// provides it.
pub struct PluginEp {
    pub name: String,
    pub library: PathBuf,
}

impl PluginEp {
    pub fn new(name: impl Into<String>, library: impl Into<PathBuf>) -> Self {
        Self {
            name: name.into(),
            library: library.into(),
        }
    }

    /// Our own Vulkan plugin.
    pub fn vulkan(library: impl Into<PathBuf>) -> Self {
        Self::new(VULKAN_EP_NAME, library)
    }

    /// Microsoft's standalone WebGPU plugin (Dawn -> Vulkan on Linux).
    pub fn webgpu(library: impl Into<PathBuf>) -> Self {
        Self::new(WEBGPU_EP_NAME, library)
    }

    /// Registers the plugin library in the current ORT environment.
    /// `ortchar` is `c_char` on Linux and `u16` (wide string) on Windows.
    pub fn register(&self) -> Result<()> {
        let api = ort::api();
        let name = CString::new(self.name.as_str())?;
        let path: &Path = &self.library;

        #[cfg(windows)]
        let status = {
            use std::os::windows::ffi::OsStrExt;
            let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
            wide.push(0);
            unsafe {
                (api.RegisterExecutionProviderLibrary)(env_ptr()?, name.as_ptr(), wide.as_ptr())
            }
        };
        #[cfg(not(windows))]
        let status = {
            let path = CString::new(path.to_str().context("plugin path is not UTF-8")?)?;
            unsafe {
                (api.RegisterExecutionProviderLibrary)(env_ptr()?, name.as_ptr(), path.as_ptr())
            }
        };
        check_status(status, "RegisterExecutionProviderLibrary")
    }

    /// Unregisters the library. Call only after destroying all
    /// sessions that use the EP.
    pub fn unregister(&self) -> Result<()> {
        let api = ort::api();
        let name = CString::new(self.name.as_str())?;
        let status = unsafe { (api.UnregisterExecutionProviderLibrary)(env_ptr()?, name.as_ptr()) };
        check_status(status, "UnregisterExecutionProviderLibrary")
    }

    /// Appends this EP's device EPs to the session (selection via
    /// `SessionOptionsAppendExecutionProvider_V2`). Returns device count.
    pub fn append_to_session(&self, builder: &mut SessionBuilder) -> Result<usize> {
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
                name.to_string_lossy() == self.name
            })
            .collect();

        if ours.is_empty() {
            let want = &self.name;
            bail!("no EP device {want} available (GetEpDevices: {num_devices} total)");
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
}
