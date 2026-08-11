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
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use vk_compute::{ComputePipeline, GpuBuffer, VkContext};

/// Stable identity of one compiled pipeline within a session.
///
/// `operation` remains the human-readable profiler label. `tactic` is present
/// for generated or artifact-selected implementations whose shader can differ
/// while serving the same operation. The map owns the typed key and its tactic
/// ID; only the closed-set profiler label remains static, so persistent tactic
/// IDs never have to be leaked into `'static` storage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    operation: &'static str,
    tactic: Option<String>,
}

impl PipelineKey {
    pub fn committed(operation: &'static str) -> Self {
        Self {
            operation,
            tactic: None,
        }
    }

    pub fn tactic(operation: &'static str, tactic: impl Into<String>) -> Self {
        Self {
            operation,
            tactic: Some(tactic.into()),
        }
    }

    pub fn operation(&self) -> &'static str {
        self.operation
    }

    pub fn tactic_id(&self) -> Option<&str> {
        self.tactic.as_deref()
    }
}

impl From<&'static str> for PipelineKey {
    fn from(operation: &'static str) -> Self {
        Self::committed(operation)
    }
}

/// Identity of a packed weight: value name plus dimensions of the
/// produced layout. Name alone is insufficient to distinguish two weights with
/// the same label, and shape alters buffer contents.
type PackedKey = (String, usize, usize);

/// Read-only inventory of one session-owned transformed weight.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PackedWeightInfo {
    pub name: String,
    pub rows: usize,
    pub columns: usize,
    pub bytes: u64,
}

/// Identity of an initializer loaded into VRAM: name, dtype, and byte count.
/// The dtype and length are part of the key because two graphs can reuse the
/// same name for different constants.
type UploadKey = (String, i32, usize);

pub struct KernelCache<'context> {
    context: &'context VkContext,
    tuning: Arc<crate::tuning::TacticResolver>,
    pipelines: Mutex<HashMap<PipelineKey, Box<ComputePipeline>>>,
    packed: Mutex<HashMap<PackedKey, Box<GpuBuffer>>>,
    uploads: Mutex<HashMap<UploadKey, Box<GpuBuffer>>>,
    zero_scalar: Mutex<Option<Box<GpuBuffer>>>,
    /// Small parameter payloads, keyed by their bytes — see `constant`.
    constants: Mutex<HashMap<Vec<u8>, Box<GpuBuffer>>>,
    pipeline_builds: AtomicUsize,
    packed_builds: AtomicUsize,
    upload_builds: AtomicUsize,
}

impl<'context> KernelCache<'context> {
    pub fn new(context: &'context VkContext) -> Self {
        Self::with_tuning(context, Arc::new(crate::tuning::TacticResolver::off()))
    }

    pub fn with_tuning(
        context: &'context VkContext,
        tuning: Arc<crate::tuning::TacticResolver>,
    ) -> Self {
        Self {
            context,
            tuning,
            pipelines: Mutex::new(HashMap::new()),
            packed: Mutex::new(HashMap::new()),
            uploads: Mutex::new(HashMap::new()),
            zero_scalar: Mutex::new(None),
            constants: Mutex::new(HashMap::new()),
            pipeline_builds: AtomicUsize::new(0),
            packed_builds: AtomicUsize::new(0),
            upload_builds: AtomicUsize::new(0),
        }
    }

    pub fn context(&self) -> &'context VkContext {
        self.context
    }

    pub fn tuning(&self) -> &crate::tuning::TacticResolver {
        &self.tuning
    }

    /// How many pipelines have been compiled and how many weights packed since
    /// the cache was created: on a warm session these numbers stop growing.
    pub fn builds(&self) -> (usize, usize) {
        (
            self.pipeline_builds.load(Ordering::Relaxed),
            self.packed_builds.load(Ordering::Relaxed),
        )
    }

    /// Compiled identities for diagnostics and structural tests.
    pub fn pipeline_keys(&self) -> Vec<PipelineKey> {
        self.pipelines
            .lock()
            .expect("poisoned pipeline cache")
            .keys()
            .cloned()
            .collect()
    }

    /// Canonical metadata for transformed weights already materialized by a
    /// real run. The underlying bytes and device handles never leave the
    /// session cache.
    pub fn packed_weights(&self) -> Vec<PackedWeightInfo> {
        let mut weights = self
            .packed
            .lock()
            .expect("poisoned packed-weights cache")
            .iter()
            .map(|((name, rows, columns), buffer)| PackedWeightInfo {
                name: name.clone(),
                rows: *rows,
                columns: *columns,
                bytes: buffer.size,
            })
            .collect::<Vec<_>>();
        weights.sort_unstable();
        weights
    }

    /// Pipeline for `key` (one per shader variant), compiled on first request.
    /// The lock is not held during dispatch.
    ///
    /// The pointer stays valid as long as the cache lives.
    pub(crate) fn pipeline(
        &self,
        key: PipelineKey,
        build: impl FnOnce() -> Result<ComputePipeline>,
    ) -> Result<*const ComputePipeline> {
        {
            let map = self.pipelines.lock().expect("poisoned pipeline cache");
            if let Some(existing) = map.get(&key) {
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

    /// Largest payload the constant cache holds, and how many it keeps.
    ///
    /// The entries are index and shape parameters — tens of bytes each — so the
    /// caps exist to bound a pathological graph, not to ration anything real.
    const CONST_MAX_BYTES: usize = 256;
    const CONST_MAX_ENTRIES: usize = 4096;

    /// A device buffer holding exactly `bytes`, uploaded once and shared by
    /// every later request for the same content.
    ///
    /// Kernels pass indices and shape parameters that do not fit in push
    /// constants through a small buffer, and they build it per call: qwen2.5-VL
    /// uploaded the same four zero bytes **74 times per decode step**, plus
    /// eight copies of one 64-byte parameter block. Keyed by content rather than
    /// by node, because that is what makes them the same buffer — and the cache
    /// belongs to the session, so a generation loop pays for them once and not
    /// once per token.
    ///
    /// Returns `None` for payloads past the cap, which the caller uploads
    /// itself.
    pub(crate) fn constant(&self, bytes: &[u8]) -> Result<Option<*const GpuBuffer>> {
        if bytes.is_empty() || bytes.len() > Self::CONST_MAX_BYTES {
            return Ok(None);
        }
        let mut map = self.constants.lock().expect("poisoned constant cache");
        if let Some(buffer) = map.get(bytes) {
            return Ok(Some(&**buffer as *const GpuBuffer));
        }
        if map.len() >= Self::CONST_MAX_ENTRIES {
            return Ok(None);
        }
        let buffer = self.context.create_storage_buffer(bytes.len() as u64)?;
        self.context.stream_upload(&buffer, bytes)?;
        let entry = map.entry(bytes.to_vec()).or_insert(Box::new(buffer));
        Ok(Some(&**entry as *const GpuBuffer))
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
        for (_, buffer) in self.constants.get_mut().expect("constant cache").drain() {
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

#[cfg(test)]
mod tests {
    use super::PipelineKey;

    #[test]
    fn pipeline_key_separates_tactics_for_one_operation() {
        let tile32 = PipelineKey::tactic("Conv", "blocked:tile32");
        let tile128 = PipelineKey::tactic("Conv", "blocked:tile128");
        assert_ne!(tile32, tile128);
        assert_eq!(tile32.operation(), "Conv");
        assert_eq!(tile32.tactic_id(), Some("blocked:tile32"));
        assert_eq!(PipelineKey::committed("Conv").tactic_id(), None);
    }
}
