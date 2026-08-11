//! Domain types for exact, hardware-aware kernel tactic selection.
//!
//! This module deliberately contains no serialization or Vulkan API types. The
//! runtime and future tuning CLI share these keys, while `vk-compute` remains
//! responsible for extracting a [`DeviceFingerprint`] from the selected device.

use crate::{AttrValue, ElementType, NodeIr};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fmt;

/// Exact Vulkan environment in which a tactic was measured.
///
/// `features` is sorted and deduplicated by [`Self::new`], so independently
/// discovered capability sets produce the same key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceFingerprint {
    name: String,
    vendor_id: u32,
    device_id: u32,
    driver_version: u32,
    api_version: u32,
    pipeline_cache_uuid: [u8; 16],
    subgroup_size: u32,
    features: Vec<String>,
}

impl DeviceFingerprint {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        vendor_id: u32,
        device_id: u32,
        driver_version: u32,
        api_version: u32,
        pipeline_cache_uuid: [u8; 16],
        subgroup_size: u32,
        mut features: Vec<String>,
    ) -> Self {
        features.sort_unstable();
        features.dedup();
        Self {
            name,
            vendor_id,
            device_id,
            driver_version,
            api_version,
            pipeline_cache_uuid,
            subgroup_size,
            features,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn vendor_id(&self) -> u32 {
        self.vendor_id
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn driver_version(&self) -> u32 {
        self.driver_version
    }

    pub fn api_version(&self) -> u32 {
        self.api_version
    }

    pub fn pipeline_cache_uuid(&self) -> &[u8; 16] {
        &self.pipeline_cache_uuid
    }

    pub fn subgroup_size(&self) -> u32 {
        self.subgroup_size
    }

    pub fn features(&self) -> &[String] {
        &self.features
    }
}

impl From<&vk_compute::DeviceFingerprint> for DeviceFingerprint {
    fn from(fingerprint: &vk_compute::DeviceFingerprint) -> Self {
        Self::new(
            fingerprint.name().to_owned(),
            fingerprint.vendor_id(),
            fingerprint.device_id(),
            fingerprint.driver_version(),
            fingerprint.api_version(),
            *fingerprint.pipeline_cache_uuid(),
            fingerprint.subgroup_size(),
            fingerprint.features().to_vec(),
        )
    }
}

/// Concrete tensor geometry used during tuning.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorSignature {
    pub dtype: ElementType,
    pub dimensions: Vec<u64>,
}

/// Performance-relevant identity of one concrete operator invocation.
///
/// `attributes_digest` is produced at the graph boundary from a canonical
/// representation of the attributes that affect dispatch or shader behavior.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WorkloadSignature {
    pub domain: String,
    pub op: String,
    pub inputs: Vec<TensorSignature>,
    pub outputs: Vec<TensorSignature>,
    pub attributes_digest: [u8; 32],
}

/// Stable SHA-256 digest of every implementation detail that can change tactic
/// behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImplementationFingerprint([u8; 32]);

impl ImplementationFingerprint {
    /// Computes a versioned, cross-language fingerprint from canonical inputs.
    pub fn compute(
        inputs: &ImplementationFingerprintInputs<'_>,
    ) -> Result<Self, ImplementationFingerprintError> {
        inputs.validate()?;

        let mut sources = inputs.shader_sources.iter().collect::<Vec<_>>();
        sources.sort_unstable_by_key(|source| source.name);
        if sources.windows(2).any(|pair| pair[0].name == pair[1].name) {
            return Err(ImplementationFingerprintError::DuplicateShaderName);
        }

        let mut hasher = Sha256::new();
        hash_bytes(&mut hasher, b"onnx-vulkan-rs:implementation-fingerprint");
        hasher.update(IMPLEMENTATION_FINGERPRINT_ENCODING_VERSION.to_le_bytes());
        hasher.update(inputs.schema_version.to_le_bytes());
        hash_bytes(&mut hasher, inputs.generator_logic);
        hash_bytes(&mut hasher, inputs.dispatch_logic);
        hash_bytes(&mut hasher, inputs.compiler_settings);
        hash_bytes(&mut hasher, inputs.runtime_abi);

        hash_len(&mut hasher, sources.len());
        for source in sources {
            hash_bytes(&mut hasher, source.name.as_bytes());
            hash_bytes(&mut hasher, source.contents);
        }

        hash_len(&mut hasher, inputs.specialization_constants.len());
        for (name, value) in inputs.specialization_constants {
            hash_bytes(&mut hasher, name.as_bytes());
            hasher.update(value.to_le_bytes());
        }

        Ok(Self(hasher.finalize().into()))
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Version of the byte encoding hashed by [`ImplementationFingerprint`].
///
/// Increment this when field order, integer encoding, or domain separation
/// changes. `ImplementationFingerprintInputs::schema_version` is separate and
/// belongs to the kernel/runtime implementation contract.
pub const IMPLEMENTATION_FINGERPRINT_ENCODING_VERSION: u32 = 1;

/// One named shader input. `contents` may be generated WGSL, GLSL, or committed
/// SPIR-V bytes; the name identifies its role in a multi-stage tactic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamedShaderSource<'a> {
    pub name: &'a str,
    pub contents: &'a [u8],
}

/// Complete set of inputs whose change invalidates a measured tactic.
///
/// The SHA-256 preimage is domain-separated and consists of:
///
/// 1. encoding version and implementation schema version as little-endian
///    `u32` values;
/// 2. generator logic, dispatch logic, compiler settings, and runtime ABI as
///    `u64`-length-prefixed byte strings in that order;
/// 3. shader sources sorted by name, each name and contents length-prefixed;
/// 4. specialization constants in `BTreeMap` order, with a length-prefixed name
///    and a little-endian `i64` value.
///
/// Compiler settings should include the compiler version and relevant options.
/// Runtime ABI should include descriptor bindings, push-constant layout, and
/// any temporary-buffer contract. Revision text may be source code or a stable
/// reviewed version token, but must not be empty.
#[derive(Clone, Copy, Debug)]
pub struct ImplementationFingerprintInputs<'a> {
    pub schema_version: u32,
    pub shader_sources: &'a [NamedShaderSource<'a>],
    pub specialization_constants: &'a BTreeMap<String, i64>,
    pub generator_logic: &'a [u8],
    pub dispatch_logic: &'a [u8],
    pub compiler_settings: &'a [u8],
    pub runtime_abi: &'a [u8],
}

impl ImplementationFingerprintInputs<'_> {
    fn validate(&self) -> Result<(), ImplementationFingerprintError> {
        if self.schema_version == 0 {
            return Err(ImplementationFingerprintError::ZeroSchemaVersion);
        }
        if self.shader_sources.is_empty() {
            return Err(ImplementationFingerprintError::NoShaderSources);
        }
        if self
            .shader_sources
            .iter()
            .any(|source| source.name.is_empty())
        {
            return Err(ImplementationFingerprintError::EmptyShaderName);
        }
        if self
            .shader_sources
            .iter()
            .any(|source| source.contents.is_empty())
        {
            return Err(ImplementationFingerprintError::EmptyShaderContents);
        }
        if self.specialization_constants.keys().any(String::is_empty) {
            return Err(ImplementationFingerprintError::EmptySpecializationName);
        }
        for (name, value) in [
            ("generator_logic", self.generator_logic),
            ("dispatch_logic", self.dispatch_logic),
            ("compiler_settings", self.compiler_settings),
            ("runtime_abi", self.runtime_abi),
        ] {
            if value.is_empty() {
                return Err(ImplementationFingerprintError::EmptyRequiredInput(name));
            }
        }
        Ok(())
    }
}

/// Invalid input to a persistent implementation fingerprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImplementationFingerprintError {
    ZeroSchemaVersion,
    NoShaderSources,
    EmptyShaderName,
    EmptyShaderContents,
    DuplicateShaderName,
    EmptySpecializationName,
    EmptyRequiredInput(&'static str),
}

impl fmt::Display for ImplementationFingerprintError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSchemaVersion => formatter.write_str("implementation schema version is zero"),
            Self::NoShaderSources => formatter.write_str("implementation has no shader sources"),
            Self::EmptyShaderName => formatter.write_str("shader source name is empty"),
            Self::EmptyShaderContents => formatter.write_str("shader source contents are empty"),
            Self::DuplicateShaderName => formatter.write_str("shader source names are not unique"),
            Self::EmptySpecializationName => {
                formatter.write_str("specialization constant name is empty")
            }
            Self::EmptyRequiredInput(name) => write!(formatter, "required input {name} is empty"),
        }
    }
}

impl Error for ImplementationFingerprintError {}

fn hash_len(hasher: &mut Sha256, len: usize) {
    hasher.update((len as u64).to_le_bytes());
}

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hash_len(hasher, bytes.len());
    hasher.update(bytes);
}

/// Canonical SHA-256 of every ONNX attribute on a node.
///
/// Attribute maps in `GraphIr` are hash maps, so names are sorted explicitly.
/// Tensor attributes include their type, concrete shape and content digest.
pub fn attributes_digest(node: &NodeIr) -> [u8; 32] {
    if node.op == "Conv" && (node.domain.is_empty() || node.domain == "ai.onnx") {
        return conv_attributes_digest(node);
    }
    canonical_attributes_digest(&node.attrs)
}

fn conv_attributes_digest(node: &NodeIr) -> [u8; 32] {
    let rank = node
        .attrs
        .get("kernel_shape")
        .and_then(AttrValue::as_ints)
        .map(|values| values.len())
        .or_else(|| {
            node.attrs
                .get("dilations")
                .and_then(AttrValue::as_ints)
                .map(|values| values.len())
        })
        .or_else(|| {
            node.attrs
                .get("strides")
                .and_then(AttrValue::as_ints)
                .map(|values| values.len())
        })
        .unwrap_or(2);
    let value = |name: &str, default: AttrValue| node.attrs.get(name).cloned().unwrap_or(default);
    let normalized = HashMap::from([
        (
            "auto_pad".into(),
            value("auto_pad", AttrValue::String("NOTSET".into())),
        ),
        ("group".into(), value("group", AttrValue::Int(1))),
        (
            "dilations".into(),
            value("dilations", AttrValue::Ints(vec![1; rank])),
        ),
        (
            "pads".into(),
            value("pads", AttrValue::Ints(vec![0; rank * 2])),
        ),
        (
            "strides".into(),
            value("strides", AttrValue::Ints(vec![1; rank])),
        ),
    ]);
    canonical_attributes_digest(&normalized)
}

fn canonical_attributes_digest(attributes: &HashMap<String, AttrValue>) -> [u8; 32] {
    let mut names = attributes.keys().collect::<Vec<_>>();
    names.sort_unstable();
    let mut hasher = Sha256::new();
    hash_bytes(&mut hasher, b"onnx-vulkan-rs:attributes:v1");
    hash_len(&mut hasher, names.len());
    for name in names {
        hash_bytes(&mut hasher, name.as_bytes());
        match &attributes[name] {
            AttrValue::Int(value) => {
                hasher.update([0]);
                hasher.update(value.to_le_bytes());
            }
            AttrValue::Ints(values) => {
                hasher.update([1]);
                hash_len(&mut hasher, values.len());
                for value in values {
                    hasher.update(value.to_le_bytes());
                }
            }
            AttrValue::Float(value) => {
                hasher.update([2]);
                hasher.update(value.to_bits().to_le_bytes());
            }
            AttrValue::Floats(values) => {
                hasher.update([3]);
                hash_len(&mut hasher, values.len());
                for value in values {
                    hasher.update(value.to_bits().to_le_bytes());
                }
            }
            AttrValue::String(value) => {
                hasher.update([4]);
                hash_bytes(&mut hasher, value.as_bytes());
            }
            AttrValue::Tensor(value) => {
                hasher.update([5]);
                hasher.update(value.dtype.to_le_bytes());
                hash_len(&mut hasher, value.shape.len());
                for dimension in &value.shape {
                    hasher.update(dimension.to_le_bytes());
                }
                hash_bytes(&mut hasher, &Sha256::digest(&value.data));
            }
        }
    }
    hasher.finalize().into()
}

/// Exact cache key. No nearest-shape or same-vendor fallback is implied.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TuningKey {
    pub device: DeviceFingerprint,
    pub workload: WorkloadSignature,
    pub implementation: ImplementationFingerprint,
}

/// Stable identifier understood by the kernel family that owns the tactic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TacticId {
    family: String,
    variant: String,
}

impl TacticId {
    pub fn new(family: String, variant: String) -> Self {
        Self { family, variant }
    }

    pub fn family(&self) -> &str {
        &self.family
    }

    pub fn variant(&self) -> &str {
        &self.variant
    }
}

/// Canonically ordered tactic parameters for inspection and persistence.
pub type TacticParameters = BTreeMap<String, i64>;

/// Summary of GPU timestamp samples from one measurement session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Measurement {
    samples: u32,
    median_gpu_ns: u64,
    min_gpu_ns: u64,
    max_gpu_ns: u64,
}

impl Measurement {
    /// Builds a measurement only when its summary can describe real samples.
    pub fn new(samples: u32, median_gpu_ns: u64, min_gpu_ns: u64, max_gpu_ns: u64) -> Option<Self> {
        if samples == 0
            || min_gpu_ns == 0
            || min_gpu_ns > median_gpu_ns
            || median_gpu_ns > max_gpu_ns
        {
            return None;
        }
        Some(Self {
            samples,
            median_gpu_ns,
            min_gpu_ns,
            max_gpu_ns,
        })
    }

    pub fn samples(&self) -> u32 {
        self.samples
    }

    pub fn median_gpu_ns(&self) -> u64 {
        self.median_gpu_ns
    }

    pub fn min_gpu_ns(&self) -> u64 {
        self.min_gpu_ns
    }

    pub fn max_gpu_ns(&self) -> u64 {
        self.max_gpu_ns
    }
}

/// A correctness-validated tactic eligible for runtime selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuningRecord {
    tactic: TacticId,
    parameters: TacticParameters,
    measurement: Measurement,
}

impl TuningRecord {
    /// Creates a record after the caller has validated the tactic's output.
    pub fn new(tactic: TacticId, parameters: TacticParameters, measurement: Measurement) -> Self {
        Self {
            tactic,
            parameters,
            measurement,
        }
    }

    pub fn tactic(&self) -> &TacticId {
        &self.tactic
    }

    pub fn parameters(&self) -> &TacticParameters {
        &self.parameters
    }

    pub fn measurement(&self) -> Measurement {
        self.measurement
    }
}

/// Result of offering a validated record to [`TuningTable`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InsertOutcome {
    Inserted,
    ReplacedSlower,
    KeptExisting,
}

/// Exact best-tactic records for one or more devices and workloads.
///
/// Incorrect candidates never enter this type: correctness validation belongs
/// at the candidate execution boundary, before a `TuningRecord` is constructed.
#[derive(Clone, Default)]
pub struct TuningTable {
    records: HashMap<TuningKey, TuningRecord>,
}

/// Runtime policy for persistent tactic data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TuningMode {
    /// Ignore every tuning record and preserve committed routing.
    #[default]
    Off,
    /// Select exact, known records and otherwise preserve committed routing.
    CacheOnly,
    /// Refuse preparation if any required exact signature is absent or unknown.
    RequireCache,
}

/// Why one concrete signature did or did not use a persistent tactic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SelectionKind {
    Off,
    Hit,
    Miss,
    UnknownTactic,
}

/// One unique-signature decision, suitable for one-shot session diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectionEvent {
    pub key: TuningKey,
    pub kind: SelectionKind,
}

/// Error produced by strict cache resolution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequireCacheError {
    missing: Vec<SelectionEvent>,
}

impl RequireCacheError {
    pub fn events(&self) -> &[SelectionEvent] {
        &self.missing
    }
}

impl fmt::Display for RequireCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "require-cache could not resolve {} concrete signature(s): ",
            self.missing.len()
        )?;
        for (index, event) in self.missing.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            write!(
                formatter,
                "{}:{} ({:?})",
                event.key.workload.domain, event.key.workload.op, event.kind
            )?;
        }
        Ok(())
    }
}

impl Error for RequireCacheError {}

/// Exact runtime resolver shared by every host.
///
/// The resolver has no filesystem or Vulkan side effects. Hosts load and
/// validate an artifact once, convert it into a [`TuningTable`], then pass the
/// same resolver type to the shared executor. A caller supplies the set of
/// tactic IDs understood by the owning kernel family; an unknown ID is a miss,
/// never executable external input.
pub struct TacticResolver {
    mode: TuningMode,
    table: TuningTable,
    events: std::sync::Mutex<HashMap<TuningKey, SelectionKind>>,
}

impl TacticResolver {
    pub fn new(mode: TuningMode, table: TuningTable) -> Self {
        Self {
            mode,
            table,
            events: std::sync::Mutex::new(HashMap::new()),
        }
    }

    pub fn off() -> Self {
        Self::new(TuningMode::Off, TuningTable::new())
    }

    pub fn mode(&self) -> TuningMode {
        self.mode
    }

    /// Resolves one key, recording only its first decision in the summary.
    pub fn resolve(
        &self,
        key: &TuningKey,
        is_known: impl FnOnce(&TuningRecord) -> bool,
    ) -> std::result::Result<Option<TuningRecord>, RequireCacheError> {
        let (record, kind) = match self.mode {
            TuningMode::Off => (None, SelectionKind::Off),
            TuningMode::CacheOnly | TuningMode::RequireCache => match self.table.get(key) {
                Some(record) if is_known(record) => (Some(record.clone()), SelectionKind::Hit),
                Some(_) => (None, SelectionKind::UnknownTactic),
                None => (None, SelectionKind::Miss),
            },
        };
        self.events
            .lock()
            .expect("poisoned tuning selection summary")
            .entry(key.clone())
            .or_insert(kind);
        if self.mode == TuningMode::RequireCache && record.is_none() {
            return Err(RequireCacheError {
                missing: vec![SelectionEvent {
                    key: key.clone(),
                    kind,
                }],
            });
        }
        Ok(record)
    }

    /// Resolves a model's complete known signature set and reports every
    /// strict-mode failure together instead of stopping at the first node.
    pub fn resolve_all<'a>(
        &self,
        keys: impl IntoIterator<Item = &'a TuningKey>,
        is_known: impl Fn(&TuningRecord) -> bool,
    ) -> std::result::Result<Vec<Option<TuningRecord>>, RequireCacheError> {
        let mut resolved = Vec::new();
        let mut missing = Vec::new();
        for key in keys {
            match self.resolve(key, &is_known) {
                Ok(record) => resolved.push(record),
                Err(error) => {
                    missing.extend_from_slice(error.events());
                    resolved.push(None);
                }
            }
        }
        if missing.is_empty() {
            Ok(resolved)
        } else {
            Err(RequireCacheError { missing })
        }
    }

    pub fn summary(&self) -> Vec<SelectionEvent> {
        let mut events = self
            .events
            .lock()
            .expect("poisoned tuning selection summary")
            .iter()
            .map(|(key, kind)| SelectionEvent {
                key: key.clone(),
                kind: *kind,
            })
            .collect::<Vec<_>>();
        events.sort_unstable_by(|left, right| {
            selection_sort_key(left).cmp(&selection_sort_key(right))
        });
        events
    }
}

impl Drop for TacticResolver {
    fn drop(&mut self) {
        let events = self
            .events
            .get_mut()
            .expect("poisoned tuning selection summary");
        if events.is_empty() || self.mode == TuningMode::Off {
            return;
        }
        let mut counts = [0_usize; 4];
        for kind in events.values() {
            counts[match kind {
                SelectionKind::Off => 0,
                SelectionKind::Hit => 1,
                SelectionKind::Miss => 2,
                SelectionKind::UnknownTactic => 3,
            }] += 1;
        }
        log::info!(
            "tuning selection summary: mode={:?}, signatures={}, hits={}, misses={}, unknown_tactics={}",
            self.mode,
            events.len(),
            counts[1],
            counts[2],
            counts[3]
        );
        if let Some(key) = events.keys().next() {
            log::info!(
                "tuning device: name={:?}, vendor_id={}, device_id={}, driver_version={}, api_version={}, pipeline_cache_uuid={}, subgroup_size={}, features={:?}",
                key.device.name(),
                key.device.vendor_id(),
                key.device.device_id(),
                key.device.driver_version(),
                key.device.api_version(),
                encode_hex(key.device.pipeline_cache_uuid()),
                key.device.subgroup_size(),
                key.device.features(),
            );
        }
        let mut ordered = events.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by(|(left, _), (right, _)| {
            (
                &left.workload.domain,
                &left.workload.op,
                left.workload
                    .inputs
                    .first()
                    .map(|tensor| tensor.dimensions.as_slice()),
                left.workload.attributes_digest,
            )
                .cmp(&(
                    &right.workload.domain,
                    &right.workload.op,
                    right
                        .workload
                        .inputs
                        .first()
                        .map(|tensor| tensor.dimensions.as_slice()),
                    right.workload.attributes_digest,
                ))
        });
        for (key, kind) in ordered {
            log::info!(
                "tuning selection: kind={kind:?}, domain={:?}, op={:?}, inputs={}, outputs={}, attributes_digest={}, implementation_digest={}",
                key.workload.domain,
                key.workload.op,
                format_tensors(&key.workload.inputs),
                format_tensors(&key.workload.outputs),
                encode_hex(&key.workload.attributes_digest),
                encode_hex(key.implementation.as_bytes()),
            );
        }
    }
}

fn format_tensors(tensors: &[TensorSignature]) -> String {
    tensors
        .iter()
        .map(|tensor| format!("{}:{:?}", tensor.dtype as u32, tensor.dimensions))
        .collect::<Vec<_>>()
        .join(";")
}

fn encode_hex(bytes: &[u8]) -> String {
    use fmt::Write as _;
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing hexadecimal into a String cannot fail");
    }
    encoded
}

fn selection_sort_key(event: &SelectionEvent) -> (&str, &str, &[u64], SelectionKind) {
    let dimensions = event
        .key
        .workload
        .inputs
        .first()
        .map_or(&[][..], |tensor| tensor.dimensions.as_slice());
    (
        &event.key.workload.domain,
        &event.key.workload.op,
        dimensions,
        event.kind,
    )
}

impl TuningTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn get(&self, key: &TuningKey) -> Option<&TuningRecord> {
        self.records.get(key)
    }

    /// Keeps the lower-median record for an exact key.
    ///
    /// Ties keep the existing record so insertion order cannot silently change
    /// a stable selection between equally timed tactics.
    pub fn insert_best(&mut self, key: TuningKey, record: TuningRecord) -> InsertOutcome {
        match self.records.get_mut(&key) {
            None => {
                self.records.insert(key, record);
                InsertOutcome::Inserted
            }
            Some(existing)
                if record.measurement.median_gpu_ns() < existing.measurement.median_gpu_ns() =>
            {
                *existing = record;
                InsertOutcome::ReplacedSlower
            }
            Some(_) => InsertOutcome::KeptExisting,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(driver_version: u32) -> DeviceFingerprint {
        DeviceFingerprint::new(
            "test gpu".into(),
            0x10de,
            0x2786,
            driver_version,
            1,
            [7; 16],
            32,
            vec!["integer_dot".into(), "coop_u8:k16:signed".into()],
        )
    }

    fn key(shape: &[u64], driver_version: u32, implementation: u8) -> TuningKey {
        TuningKey {
            device: device(driver_version),
            workload: WorkloadSignature {
                domain: "com.microsoft".into(),
                op: "MatMulNBits".into(),
                inputs: vec![TensorSignature {
                    dtype: ElementType::Float32,
                    dimensions: shape.to_vec(),
                }],
                outputs: Vec::new(),
                attributes_digest: [3; 32],
            },
            implementation: ImplementationFingerprint::from_bytes([implementation; 32]),
        }
    }

    fn record(variant: &str, median_gpu_ns: u64) -> TuningRecord {
        TuningRecord::new(
            TacticId::new("wide".into(), variant.into()),
            BTreeMap::from([("lanes".into(), 4)]),
            Measurement::new(20, median_gpu_ns, median_gpu_ns - 1, median_gpu_ns + 1)
                .expect("test measurement is ordered and non-zero"),
        )
    }

    fn implementation_inputs<'a>(
        sources: &'a [NamedShaderSource<'a>],
        specializations: &'a BTreeMap<String, i64>,
    ) -> ImplementationFingerprintInputs<'a> {
        ImplementationFingerprintInputs {
            schema_version: 3,
            shader_sources: sources,
            specialization_constants: specializations,
            generator_logic: b"generator-v2",
            dispatch_logic: b"grid-v1",
            compiler_settings: b"naga=30;spv=1.3;entry=main",
            runtime_abi: b"bindings=4;push=24",
        }
    }

    #[test]
    fn device_features_are_canonical() {
        let a = DeviceFingerprint::new(
            "gpu".into(),
            1,
            2,
            3,
            4,
            [5; 16],
            32,
            vec!["z".into(), "a".into(), "z".into()],
        );
        assert_eq!(a.features(), ["a", "z"]);
    }

    #[test]
    fn attribute_digest_is_independent_of_hash_map_order_and_value_sensitive() {
        let mut left = NodeIr {
            domain: String::new(),
            op: "Conv".into(),
            opset: 13,
            name: "left".into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            attrs: HashMap::from([
                ("pads".into(), AttrValue::Ints(vec![1, 1, 1, 1])),
                ("group".into(), AttrValue::Int(1)),
            ]),
        };
        let right = NodeIr {
            name: "right".into(),
            attrs: HashMap::from([
                ("group".into(), AttrValue::Int(1)),
                ("pads".into(), AttrValue::Ints(vec![1, 1, 1, 1])),
            ]),
            ..left.clone()
        };
        assert_eq!(attributes_digest(&left), attributes_digest(&right));
        left.attrs.insert("group".into(), AttrValue::Int(2));
        assert_ne!(attributes_digest(&left), attributes_digest(&right));
    }

    #[test]
    fn conv_attribute_digest_normalizes_ort_materialized_defaults() {
        let compact = NodeIr {
            domain: String::new(),
            op: "Conv".into(),
            opset: 13,
            name: "standalone".into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            attrs: HashMap::from([
                ("kernel_shape".into(), AttrValue::Ints(vec![3, 3])),
                ("pads".into(), AttrValue::Ints(vec![1, 1, 1, 1])),
                ("strides".into(), AttrValue::Ints(vec![2, 2])),
            ]),
        };
        let mut materialized = compact.clone();
        materialized.name = "ep".into();
        materialized
            .attrs
            .insert("auto_pad".into(), AttrValue::String("NOTSET".into()));
        materialized
            .attrs
            .insert("dilations".into(), AttrValue::Ints(vec![1, 1]));
        materialized.attrs.insert("group".into(), AttrValue::Int(1));
        assert_eq!(
            attributes_digest(&compact),
            attributes_digest(&materialized)
        );
    }

    #[test]
    fn runtime_resolver_is_exact_and_deduplicates_summary() {
        let exact = key(&[1, 4096], 59584, 9);
        let mut table = TuningTable::new();
        table.insert_best(exact.clone(), record("block4", 10));
        let resolver = TacticResolver::new(TuningMode::CacheOnly, table);
        let known = |candidate: &TuningRecord| candidate.tactic().variant() == "block4";

        assert!(resolver.resolve(&exact, known).unwrap().is_some());
        assert!(resolver.resolve(&exact, known).unwrap().is_some());
        assert!(
            resolver
                .resolve(&key(&[1, 2048], 59584, 9), known)
                .unwrap()
                .is_none()
        );
        assert!(
            resolver
                .resolve(&key(&[1, 4096], 59585, 9), known)
                .unwrap()
                .is_none()
        );
        assert!(
            resolver
                .resolve(&key(&[1, 4096], 59584, 10), known)
                .unwrap()
                .is_none()
        );

        let summary = resolver.summary();
        assert_eq!(summary.len(), 4);
        assert_eq!(
            summary
                .iter()
                .filter(|event| event.kind == SelectionKind::Hit)
                .count(),
            1
        );
    }

    #[test]
    fn unknown_tactic_falls_back_or_fails_by_mode() {
        let exact = key(&[1, 4096], 59584, 9);
        let mut table = TuningTable::new();
        table.insert_best(exact.clone(), record("external-code", 10));

        let cache_only = TacticResolver::new(TuningMode::CacheOnly, table.clone());
        assert!(cache_only.resolve(&exact, |_| false).unwrap().is_none());
        assert_eq!(cache_only.summary()[0].kind, SelectionKind::UnknownTactic);

        let required = TacticResolver::new(TuningMode::RequireCache, table);
        let error = required
            .resolve(&exact, |_| false)
            .expect_err("unknown external tactic must not execute");
        assert_eq!(error.events()[0].kind, SelectionKind::UnknownTactic);
    }

    #[test]
    fn require_cache_reports_partial_model_coverage_together() {
        let hit = key(&[1, 4096], 59584, 9);
        let miss_shape = key(&[1, 2048], 59584, 9);
        let miss_driver = key(&[1, 4096], 59585, 9);
        let mut table = TuningTable::new();
        table.insert_best(hit.clone(), record("block4", 10));
        let resolver = TacticResolver::new(TuningMode::RequireCache, table);

        let error = resolver
            .resolve_all([&hit, &miss_shape, &miss_driver], |_| true)
            .expect_err("two signatures lack exact records");
        assert_eq!(error.events().len(), 2);
        assert!(
            error
                .events()
                .iter()
                .all(|event| event.kind == SelectionKind::Miss)
        );
    }

    #[test]
    fn implementation_fingerprint_matches_the_versioned_golden_vector() {
        let sources = [
            NamedShaderSource {
                name: "reduce",
                contents: b"@compute reduce",
            },
            NamedShaderSource {
                name: "main",
                contents: b"@compute main",
            },
        ];
        let specializations = BTreeMap::from([("signed".into(), -1), ("lanes".into(), 4)]);

        let fingerprint =
            ImplementationFingerprint::compute(&implementation_inputs(&sources, &specializations))
                .expect("fixture contains every required implementation input");

        assert_eq!(
            fingerprint.as_bytes(),
            &[
                0xc2, 0x4c, 0x3b, 0x6e, 0x91, 0x27, 0x44, 0x74, 0xc7, 0x9c, 0x08, 0xf9, 0x01, 0x84,
                0x5b, 0x3f, 0x4c, 0x6a, 0x01, 0x1e, 0xa7, 0xf3, 0x33, 0x62, 0xee, 0x06, 0x9a, 0x2b,
                0x52, 0x46, 0xee, 0x87,
            ]
        );
    }

    #[test]
    fn implementation_fingerprint_canonicalizes_source_order() {
        let a = [
            NamedShaderSource {
                name: "main",
                contents: b"main",
            },
            NamedShaderSource {
                name: "reduce",
                contents: b"reduce",
            },
        ];
        let b = [a[1], a[0]];
        let specializations = BTreeMap::from([("lanes".into(), 4)]);

        assert_eq!(
            ImplementationFingerprint::compute(&implementation_inputs(&a, &specializations)),
            ImplementationFingerprint::compute(&implementation_inputs(&b, &specializations)),
        );
    }

    #[test]
    fn implementation_fingerprint_changes_with_every_contract_category() {
        let sources = [NamedShaderSource {
            name: "main",
            contents: b"main-v1",
        }];
        let changed_sources = [NamedShaderSource {
            name: "main",
            contents: b"main-v2",
        }];
        let specializations = BTreeMap::from([("lanes".into(), 4)]);
        let changed_specializations = BTreeMap::from([("lanes".into(), 8)]);
        let base = implementation_inputs(&sources, &specializations);
        let fingerprint = ImplementationFingerprint::compute(&base)
            .expect("fixture contains every required implementation input");

        let variants = [
            ImplementationFingerprintInputs {
                schema_version: 4,
                ..base
            },
            ImplementationFingerprintInputs {
                shader_sources: &changed_sources,
                ..base
            },
            ImplementationFingerprintInputs {
                specialization_constants: &changed_specializations,
                ..base
            },
            ImplementationFingerprintInputs {
                generator_logic: b"generator-v3",
                ..base
            },
            ImplementationFingerprintInputs {
                dispatch_logic: b"grid-v2",
                ..base
            },
            ImplementationFingerprintInputs {
                compiler_settings: b"naga=31;spv=1.3;entry=main",
                ..base
            },
            ImplementationFingerprintInputs {
                runtime_abi: b"bindings=4;push=28",
                ..base
            },
        ];

        for variant in variants {
            assert_ne!(
                ImplementationFingerprint::compute(&variant)
                    .expect("variant contains every required implementation input"),
                fingerprint
            );
        }
    }

    #[test]
    fn implementation_fingerprint_rejects_ambiguous_or_partial_inputs() {
        let duplicate_sources = [
            NamedShaderSource {
                name: "main",
                contents: b"v1",
            },
            NamedShaderSource {
                name: "main",
                contents: b"v2",
            },
        ];
        let specializations = BTreeMap::new();
        assert_eq!(
            ImplementationFingerprint::compute(&implementation_inputs(
                &duplicate_sources,
                &specializations
            )),
            Err(ImplementationFingerprintError::DuplicateShaderName)
        );

        let sources = [NamedShaderSource {
            name: "main",
            contents: b"main",
        }];
        let mut partial = implementation_inputs(&sources, &specializations);
        partial.runtime_abi = b"";
        assert_eq!(
            ImplementationFingerprint::compute(&partial),
            Err(ImplementationFingerprintError::EmptyRequiredInput(
                "runtime_abi"
            ))
        );

        let empty_source = [NamedShaderSource {
            name: "main",
            contents: b"",
        }];
        assert_eq!(
            ImplementationFingerprint::compute(&implementation_inputs(
                &empty_source,
                &specializations
            )),
            Err(ImplementationFingerprintError::EmptyShaderContents)
        );

        let empty_specialization = BTreeMap::from([(String::new(), 1)]);
        assert_eq!(
            ImplementationFingerprint::compute(&implementation_inputs(
                &sources,
                &empty_specialization
            )),
            Err(ImplementationFingerprintError::EmptySpecializationName)
        );
    }

    #[test]
    fn measurement_rejects_invalid_summaries() {
        assert!(Measurement::new(0, 10, 9, 11).is_none());
        assert!(Measurement::new(20, 10, 0, 11).is_none());
        assert!(Measurement::new(20, 8, 9, 11).is_none());
        assert!(Measurement::new(20, 12, 9, 11).is_none());
    }

    #[test]
    fn exact_lookup_does_not_cross_shape_driver_or_implementation() {
        let mut table = TuningTable::new();
        let exact = key(&[1, 1152], 59584, 1);
        assert_eq!(
            table.insert_best(exact.clone(), record("block4", 100)),
            InsertOutcome::Inserted
        );

        assert!(table.get(&exact).is_some());
        assert!(table.get(&key(&[1, 4096], 59584, 1)).is_none());
        assert!(table.get(&key(&[1, 1152], 61074, 1)).is_none());
        assert!(table.get(&key(&[1, 1152], 59584, 2)).is_none());
    }

    #[test]
    fn only_a_strictly_faster_record_replaces_the_winner() {
        let mut table = TuningTable::new();
        let key = key(&[1, 1152], 59584, 1);

        assert_eq!(
            table.insert_best(key.clone(), record("control", 100)),
            InsertOutcome::Inserted
        );
        assert_eq!(
            table.insert_best(key.clone(), record("slower", 110)),
            InsertOutcome::KeptExisting
        );
        assert_eq!(
            table.get(&key).expect("record exists").tactic().variant(),
            "control"
        );
        assert_eq!(
            table.insert_best(key.clone(), record("faster", 90)),
            InsertOutcome::ReplacedSlower
        );
        assert_eq!(
            table.get(&key).expect("record exists").tactic().variant(),
            "faster"
        );
        assert_eq!(table.len(), 1);
    }
}
