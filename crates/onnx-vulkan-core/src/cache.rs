//! GPU resources reused across executions, **owned by the caller**.
//!
//! Compiled pipelines, packed weights, and utility buffers live here instead
//! rather than in global or thread-local variables: lifecycle follows the session
//! that owns the cache, and destruction frees VRAM.
//!
//! The cache is bound to the `VkContext` it was created with — defining the
//! device, so keys do not repeat it. Entries are never
//! removed: returned addresses (`Box` on heap) remain valid while the cache lives.
//! the cache.

use anyhow::Result;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use vk_compute::{ComputePipeline, GpuBuffer, VkContext};

/// Identity of a packed weight: value name plus dimensions of the
/// produced layout. Name alone is insufficient to distinguish two weights with
/// the same label, and shape alters buffer contents.
type PackedKey = (String, usize, usize);

/// Identity of an initializer loaded into VRAM: name, dtype, and byte count.
/// The dtype and length are part of the key because two graphs can reuse the
/// same name for different constants.
type UploadKey = (String, i32, usize);

pub struct KernelCache<'context> {
    context: &'context VkContext,
    pipelines: Mutex<HashMap<&'static str, Box<ComputePipeline>>>,
    packed: Mutex<HashMap<PackedKey, Box<GpuBuffer>>>,
    uploads: Mutex<HashMap<UploadKey, Box<GpuBuffer>>>,
    zero_scalar: Mutex<Option<Box<GpuBuffer>>>,
    pipeline_builds: AtomicUsize,
    packed_builds: AtomicUsize,
    upload_builds: AtomicUsize,
}

impl<'context> KernelCache<'context> {
    pub fn new(context: &'context VkContext) -> Self {
        Self {
            context,
            pipelines: Mutex::new(HashMap::new()),
            packed: Mutex::new(HashMap::new()),
            uploads: Mutex::new(HashMap::new()),
            zero_scalar: Mutex::new(None),
            pipeline_builds: AtomicUsize::new(0),
            packed_builds: AtomicUsize::new(0),
            upload_builds: AtomicUsize::new(0),
        }
    }

    pub fn context(&self) -> &'context VkContext {
        self.context
    }

    /// How many pipelines have been compiled and how many weights packed since
    /// the cache was created: on a warm session these numbers stop growing.
    pub fn builds(&self) -> (usize, usize) {
        (
            self.pipeline_builds.load(Ordering::Relaxed),
            self.packed_builds.load(Ordering::Relaxed),
        )
    }

    /// Pipeline for `key` (one per shader variant), compiled on first request.
    /// The lock is not held during dispatch.
    ///
    /// The pointer stays valid as long as the cache lives.
    pub(crate) fn pipeline(
        &self,
        key: &'static str,
        build: impl FnOnce() -> Result<ComputePipeline>,
    ) -> Result<*const ComputePipeline> {
        {
            let map = self.pipelines.lock().expect("poisoned pipeline cache");
            if let Some(existing) = map.get(key) {
                return Ok(&**existing as *const ComputePipeline);
            }
        }
        // compilation outside the lock: this is the expensive part
        self.pipeline_builds.fetch_add(1, Ordering::Relaxed);
        let built = Box::new(build()?);
        let mut map = self.pipelines.lock().expect("poisoned pipeline cache");
        match map.entry(key) {
            Entry::Occupied(entry) => {
                // another thread won the race: our copy must be destroyed
                self.context.destroy_pipeline(*built);
                Ok(&**entry.into_mut() as *const ComputePipeline)
            }
            Entry::Vacant(entry) => Ok(&**entry.insert(built) as *const ComputePipeline),
        }
    }

    /// Already-packed weight, if present: saves the caller from preparing the
    /// input (host read of the weight) when the cache is warm.
    ///
    /// The pointer stays valid as long as the cache lives.
    pub(crate) fn packed_weight_cached(&self, key: &PackedKey) -> Option<*const GpuBuffer> {
        let map = self.packed.lock().expect("poisoned packed-weights cache");
        map.get(key).map(|buffer| &**buffer as *const GpuBuffer)
    }

    /// Packed weight for `key`, produced on first request.
    ///
    /// The pointer stays valid as long as the cache lives.
    pub(crate) fn packed_weight(
        &self,
        key: PackedKey,
        build: impl FnOnce() -> Result<GpuBuffer>,
    ) -> Result<*const GpuBuffer> {
        if let Some(existing) = self.packed_weight_cached(&key) {
            return Ok(existing);
        }
        self.packed_builds.fetch_add(1, Ordering::Relaxed);
        let built = Box::new(build()?);
        let mut map = self.packed.lock().expect("poisoned packed-weights cache");
        match map.entry(key) {
            Entry::Occupied(entry) => {
                self.context.destroy_buffer(*built);
                Ok(&**entry.into_mut() as *const GpuBuffer)
            }
            Entry::Vacant(entry) => Ok(&**entry.insert(built) as *const GpuBuffer),
        }
    }

    /// Initializer resident in VRAM, uploaded on first request.
    ///
    /// Weights do not change across runs: reloading them on every execution
    /// costs PCIe bandwidth, a staging buffer, and a CPU copy per tensor.
    /// Keeping them here is what makes the model truly resident on the device.
    ///
    /// The pointer stays valid as long as the cache lives.
    pub fn initializer(
        &self,
        key: UploadKey,
        build: impl FnOnce() -> Result<GpuBuffer>,
    ) -> Result<*const GpuBuffer> {
        {
            let map = self.uploads.lock().expect("poisoned upload cache");
            if let Some(existing) = map.get(&key) {
                return Ok(&**existing as *const GpuBuffer);
            }
        }
        self.upload_builds.fetch_add(1, Ordering::Relaxed);
        let built = Box::new(build()?);
        let mut map = self.uploads.lock().expect("poisoned upload cache");
        match map.entry(key) {
            Entry::Occupied(entry) => {
                self.context.destroy_buffer(*built);
                Ok(&**entry.into_mut() as *const GpuBuffer)
            }
            Entry::Vacant(entry) => Ok(&**entry.insert(built) as *const GpuBuffer),
        }
    }

    /// How many initializers have been uploaded to VRAM since the cache was
    /// created: on a warm session this stops growing after the first run.
    pub fn uploads(&self) -> usize {
        self.upload_builds.load(Ordering::Relaxed)
    }

    /// Shared zero scalar buffer (4 bytes): missing zero-points read as 0.
    ///
    /// The pointer stays valid as long as the cache lives.
    pub(crate) fn zero_scalar(&self) -> Result<*const GpuBuffer> {
        let mut slot = self.zero_scalar.lock().expect("poisoned zero cache");
        if slot.is_none() {
            let buffer = self.context.create_storage_buffer(4)?;
            self.context.stream_upload(&buffer, &[0u8; 4])?;
            *slot = Some(Box::new(buffer));
        }
        Ok(&**slot.as_ref().expect("zero just inserted") as *const GpuBuffer)
    }
}

impl Drop for KernelCache<'_> {
    fn drop(&mut self) {
        // in-flight work may still reference these resources
        let _ = self.context.flush();
        for (_, pipeline) in self.pipelines.get_mut().expect("pipeline cache").drain() {
            self.context.destroy_pipeline(*pipeline);
        }
        for (_, buffer) in self.packed.get_mut().expect("packed-weights cache").drain() {
            self.context.destroy_buffer(*buffer);
        }
        for (_, buffer) in self.uploads.get_mut().expect("upload cache").drain() {
            self.context.destroy_buffer(*buffer);
        }
        if let Some(buffer) = self.zero_scalar.get_mut().expect("zero cache").take() {
            self.context.destroy_buffer(*buffer);
        }
    }
}
