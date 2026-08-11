use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, btree_map::Entry};
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub const SCHEMA_VERSION: u32 = 1;
pub const OUTPUT_SCHEMA_VERSION: u32 = 1;
pub const EXECUTION_PLAN_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct CreatedBy {
    #[serde(default)]
    pub onnx_vulkan_version: String,
    #[serde(default)]
    pub git_commit: String,
    #[serde(default)]
    pub dirty: bool,
    #[serde(default)]
    pub command_line: Vec<String>,
    #[serde(default)]
    pub created_unix_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Device {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u32,
    pub api_version: u32,
    pub pipeline_cache_uuid: String,
    pub subgroup_size: u32,
    #[serde(default)]
    pub features: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Profile {
    pub model_digest: String,
    #[serde(default)]
    pub symbol_values: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct MeasurementPolicy {
    pub warmup_iterations: u32,
    pub measured_iterations: u32,
    pub statistic: String,
    pub clock: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tensor {
    pub dtype: u32,
    pub dims: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Workload {
    #[serde(default)]
    pub domain: String,
    pub op: String,
    #[serde(default)]
    pub inputs: Vec<Tensor>,
    #[serde(default)]
    pub outputs: Vec<Tensor>,
    pub attributes_digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tactic {
    pub family: String,
    pub id: String,
    #[serde(default)]
    pub parameters: BTreeMap<String, i64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Measurement {
    pub samples: u32,
    pub median_gpu_ns: u64,
    pub min_gpu_ns: u64,
    pub max_gpu_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic_samples_gpu_ns: Option<Vec<u64>>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Correctness {
    pub kind: String,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Record {
    pub workload: Workload,
    pub implementation_digest: String,
    pub tactic: Tactic,
    pub measurement: Measurement,
    pub correctness: Correctness,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Diagnostic {
    pub workload: Workload,
    pub tactic_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tactic: Option<Tactic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implementation_digest: Option<String>,
    pub outcome: String,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct FailureDetail {
    pub outcome: String,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ReducedFixture {
    pub schema_version: u32,
    pub artifact_schema_version: u32,
    pub device: Device,
    pub profile: Profile,
    pub workload: Workload,
    pub tactic_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tactic: Option<Tactic>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub implementation_digest: Option<String>,
    pub failure: FailureDetail,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Artifact {
    pub schema_version: u32,
    #[serde(default)]
    pub created_by: CreatedBy,
    pub device: Device,
    pub profile: Profile,
    pub measurement_policy: MeasurementPolicy,
    #[serde(default)]
    pub records: Vec<Record>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<Diagnostic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<ModelCoverage>,
}

/// Scope completeness recorded by the model-level builder. Legacy/manual
/// artifacts omit it; when present, strict runtime mode can reject a partial
/// family inventory before any dispatch.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelCoverage {
    pub scope: String,
    pub inventory_workloads: u64,
    pub tuned_workloads: u64,
    pub untunable_nodes: u64,
}

impl ModelCoverage {
    pub fn strict_compatible(&self) -> bool {
        self.tuned_workloads == self.inventory_workloads && self.untunable_nodes == 0
    }
}

/// GPU-free inventory of exact workloads that the model-level tuner can feed
/// to a family runner. Unsupported nodes remain visible in `skipped`.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ModelInventory {
    pub schema_version: u32,
    pub model_digest: String,
    pub profile: Profile,
    pub workloads: Vec<InventoryWorkload>,
    pub skipped: Vec<InventorySkip>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct InventoryWorkload {
    pub node: String,
    pub workload: Workload,
    pub implementation_digest: String,
    pub runner: ConvRunnerGeometry,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConvRunnerGeometry {
    pub family: String,
    pub c_in: u64,
    pub c_out: u64,
    pub kernel: u64,
    pub h_in: u64,
    pub h_out: u64,
    pub stride: u64,
    pub pad: u64,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct InventorySkip {
    pub node: String,
    pub op: String,
    pub reason: String,
}

/// Device-specific reconstruction manifest for a model execution plan.
///
/// This deliberately contains no graph nodes, shader binaries, weight bytes,
/// Vulkan handles, or captured commands. Those remain owned by the ordinary
/// model loader and the runtime `StepPlan` capture path; this manifest only
/// proves that their identities and resource contracts still match.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExecutionPlanArtifact {
    pub schema_version: u32,
    #[serde(default)]
    pub created_by: CreatedBy,
    pub device: Device,
    pub graph_digest: String,
    pub implementation_digests: BTreeMap<String, String>,
    pub profiles: Vec<ConcreteProfile>,
    #[serde(default)]
    pub resolved_tactics: Vec<ResolvedTactic>,
    #[serde(default)]
    pub packed_weights: Vec<PackedWeightMetadata>,
    #[serde(default)]
    pub steps: Vec<StepPlanMetadata>,
}

/// Resource/profile metadata supplied by a build inventory. Identity and
/// resolved tactics are intentionally absent and derived from trusted inputs.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExecutionPlanDraft {
    pub profiles: Vec<ConcreteProfile>,
    #[serde(default)]
    pub packed_weights: Vec<PackedWeightMetadata>,
    #[serde(default)]
    pub steps: Vec<StepPlanMetadata>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionPlanIdentity {
    pub device: Device,
    pub graph_digest: String,
    pub implementation_digests: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConcreteProfile {
    pub id: String,
    pub inputs: BTreeMap<String, Tensor>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResolvedTactic {
    pub workload: Workload,
    pub implementation: String,
    pub tactic: Tactic,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct PackedWeightMetadata {
    pub name: String,
    pub dtype: u32,
    pub dims: Vec<u64>,
    pub layout: String,
    pub bytes: u64,
    pub source_digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct TemporaryBufferMetadata {
    pub name: String,
    pub bytes: u64,
    pub alignment: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct StepPlanMetadata {
    pub profile: String,
    pub dispatches: u64,
    pub host_nodes: u64,
    #[serde(default)]
    pub temporary_buffers: Vec<TemporaryBufferMetadata>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactError {
    Json(String),
    OldSchema(u32),
    NewerSchema(u32),
    Invalid(String),
    ConflictingKey(String),
    Incompatible(String),
    Io(String),
}

impl ArtifactError {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Json(_) | Self::OldSchema(_) | Self::NewerSchema(_) | Self::Invalid(_) => 3,
            Self::Io(_) => 4,
            Self::ConflictingKey(_) | Self::Incompatible(_) => 5,
        }
    }
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(message) => write!(formatter, "malformed artifact JSON: {message}"),
            Self::OldSchema(version) => write!(
                formatter,
                "artifact schema {version} is obsolete; migrate it to schema {SCHEMA_VERSION}"
            ),
            Self::NewerSchema(version) => write!(
                formatter,
                "artifact schema {version} is newer than supported schema {SCHEMA_VERSION}; update onnx-vulkan-tune"
            ),
            Self::Invalid(message) => write!(formatter, "invalid artifact: {message}"),
            Self::ConflictingKey(key) => {
                write!(formatter, "conflicting records for exact key {key}")
            }
            Self::Incompatible(message) => write!(formatter, "incompatible artifacts: {message}"),
            Self::Io(message) => write!(formatter, "artifact I/O failed: {message}"),
        }
    }
}

impl std::error::Error for ArtifactError {}

pub fn parse_artifact(bytes: &[u8]) -> Result<Artifact, ArtifactError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| ArtifactError::Json(error.to_string()))?;
    let version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| ArtifactError::Invalid("schema_version is missing or not a u32".into()))?;
    let version = u32::try_from(version)
        .map_err(|_| ArtifactError::Invalid("schema_version does not fit u32".into()))?;
    match version {
        SCHEMA_VERSION => {}
        0..SCHEMA_VERSION => return Err(ArtifactError::OldSchema(version)),
        _ => return Err(ArtifactError::NewerSchema(version)),
    }
    serde_json::from_value(value).map_err(|error| ArtifactError::Json(error.to_string()))
}

pub fn read_artifact(path: &Path) -> Result<Artifact, ArtifactError> {
    let bytes = fs::read(path)
        .map_err(|error| ArtifactError::Io(format!("read {}: {error}", path.display())))?;
    let artifact = parse_artifact(&bytes)?;
    validate_artifact(&artifact)?;
    Ok(artifact)
}

pub fn parse_execution_plan(bytes: &[u8]) -> Result<ExecutionPlanArtifact, ArtifactError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| ArtifactError::Json(error.to_string()))?;
    let version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| ArtifactError::Invalid("schema_version is missing or not a u32".into()))?;
    let version = u32::try_from(version)
        .map_err(|_| ArtifactError::Invalid("schema_version does not fit u32".into()))?;
    match version {
        EXECUTION_PLAN_SCHEMA_VERSION => {}
        0..EXECUTION_PLAN_SCHEMA_VERSION => return Err(ArtifactError::OldSchema(version)),
        _ => return Err(ArtifactError::NewerSchema(version)),
    }
    let plan: ExecutionPlanArtifact =
        serde_json::from_value(value).map_err(|error| ArtifactError::Json(error.to_string()))?;
    validate_execution_plan(&plan)?;
    Ok(plan)
}

pub fn read_execution_plan(path: &Path) -> Result<ExecutionPlanArtifact, ArtifactError> {
    let bytes = fs::read(path)
        .map_err(|error| ArtifactError::Io(format!("read {}: {error}", path.display())))?;
    parse_execution_plan(&bytes)
}

pub fn canonicalize_execution_plan(plan: &mut ExecutionPlanArtifact) -> Result<(), ArtifactError> {
    plan.device.features.sort_unstable();
    plan.device.features.dedup();
    plan.profiles.sort_unstable();
    plan.resolved_tactics.sort_unstable();
    plan.packed_weights.sort_unstable();
    for step in &mut plan.steps {
        step.temporary_buffers.sort_unstable();
    }
    plan.steps.sort_unstable();
    validate_execution_plan(plan)
}

pub fn canonical_execution_plan_json(
    plan: &ExecutionPlanArtifact,
) -> Result<Vec<u8>, ArtifactError> {
    let mut canonical = plan.clone();
    canonicalize_execution_plan(&mut canonical)?;
    let mut bytes = serde_json::to_vec_pretty(&canonical)
        .map_err(|error| ArtifactError::Json(error.to_string()))?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn write_execution_plan(
    path: &Path,
    plan: &ExecutionPlanArtifact,
) -> Result<(), ArtifactError> {
    atomic_write(path, &canonical_execution_plan_json(plan)?)
}

pub fn validate_execution_plan(plan: &ExecutionPlanArtifact) -> Result<(), ArtifactError> {
    match plan.schema_version {
        EXECUTION_PLAN_SCHEMA_VERSION => {}
        0..EXECUTION_PLAN_SCHEMA_VERSION => {
            return Err(ArtifactError::OldSchema(plan.schema_version));
        }
        version => return Err(ArtifactError::NewerSchema(version)),
    }
    validate_created_by(&plan.created_by)?;
    validate_device(&plan.device)?;
    validate_hex("graph_digest", &plan.graph_digest, 32)?;
    if plan.implementation_digests.is_empty() {
        return invalid("implementation_digests is empty");
    }
    for (name, digest) in &plan.implementation_digests {
        require_nonempty("implementation_digests key", name)?;
        validate_hex(&format!("implementation_digests.{name}"), digest, 32)?;
    }
    if plan.profiles.is_empty() {
        return invalid("profiles is empty");
    }
    let mut profiles = std::collections::BTreeSet::new();
    for (index, profile) in plan.profiles.iter().enumerate() {
        require_nonempty(&format!("profiles[{index}].id"), &profile.id)?;
        if profile.inputs.is_empty() {
            return invalid(format!("profiles[{index}].inputs is empty"));
        }
        if !profiles.insert(profile.id.as_str()) {
            return invalid(format!("duplicate profile id `{}`", profile.id));
        }
        for (name, tensor) in &profile.inputs {
            require_nonempty(&format!("profiles[{index}].inputs key"), name)?;
            if tensor.dtype == 0 {
                return invalid(format!("profiles[{index}].inputs.{name}.dtype is zero"));
            }
        }
    }
    let mut tactics = std::collections::BTreeSet::new();
    for (index, resolved) in plan.resolved_tactics.iter().enumerate() {
        validate_workload(
            &resolved.workload,
            &format!("resolved_tactics[{index}].workload"),
        )?;
        validate_tactic(
            &resolved.tactic,
            &format!("resolved_tactics[{index}].tactic"),
        )?;
        if !plan
            .implementation_digests
            .contains_key(&resolved.implementation)
        {
            return invalid(format!(
                "resolved_tactics[{index}] references unknown implementation `{}`",
                resolved.implementation
            ));
        }
        let key = serde_json::to_string(&resolved.workload)
            .map_err(|error| ArtifactError::Json(error.to_string()))?;
        if !tactics.insert(key) {
            return invalid(format!(
                "resolved_tactics[{index}] duplicates a workload signature"
            ));
        }
    }
    let mut weights = std::collections::BTreeSet::new();
    for (index, weight) in plan.packed_weights.iter().enumerate() {
        require_nonempty(&format!("packed_weights[{index}].name"), &weight.name)?;
        require_nonempty(&format!("packed_weights[{index}].layout"), &weight.layout)?;
        if weight.dtype == 0 || weight.bytes == 0 {
            return invalid(format!(
                "packed_weights[{index}] has zero dtype or byte length"
            ));
        }
        validate_hex(
            &format!("packed_weights[{index}].source_digest"),
            &weight.source_digest,
            32,
        )?;
        if !weights.insert((&weight.name, &weight.layout)) {
            return invalid(format!(
                "packed_weights[{index}] duplicates name/layout metadata"
            ));
        }
    }
    for (index, step) in plan.steps.iter().enumerate() {
        if !profiles.contains(step.profile.as_str()) {
            return invalid(format!(
                "steps[{index}] references unknown profile `{}`",
                step.profile
            ));
        }
        if step.dispatches == 0 {
            return invalid(format!("steps[{index}].dispatches is zero"));
        }
        let mut buffers = std::collections::BTreeSet::new();
        for (buffer_index, buffer) in step.temporary_buffers.iter().enumerate() {
            require_nonempty(
                &format!("steps[{index}].temporary_buffers[{buffer_index}].name"),
                &buffer.name,
            )?;
            if buffer.bytes == 0 || buffer.alignment == 0 || !buffer.alignment.is_power_of_two() {
                return invalid(format!(
                    "steps[{index}].temporary_buffers[{buffer_index}] has invalid size/alignment"
                ));
            }
            if !buffers.insert(buffer.name.as_str()) {
                return invalid(format!(
                    "steps[{index}] duplicates temporary buffer `{}`",
                    buffer.name
                ));
            }
        }
    }
    Ok(())
}

/// Refuses stale plans before any Vulkan resource is reconstructed.
pub fn validate_execution_plan_for(
    plan: &ExecutionPlanArtifact,
    device: &Device,
    graph_digest: &str,
    implementation_digests: &BTreeMap<String, String>,
) -> Result<(), ArtifactError> {
    validate_execution_plan(plan)?;
    if &plan.device != device {
        return Err(ArtifactError::Incompatible(
            "execution-plan device fingerprint changed".into(),
        ));
    }
    if plan.graph_digest != graph_digest {
        return Err(ArtifactError::Incompatible(
            "execution-plan graph digest changed".into(),
        ));
    }
    if &plan.implementation_digests != implementation_digests {
        return Err(ArtifactError::Incompatible(
            "execution-plan implementation digest set changed".into(),
        ));
    }
    Ok(())
}

/// Builds plan identity from a validated tuning artifact and the same
/// normalized graph the executor runs. The draft can describe resources, but
/// cannot override device, graph, implementation, or tactic identity.
pub fn build_execution_plan(
    tuning: &Artifact,
    model: &onnx_vulkan_frontend::Model,
    draft: ExecutionPlanDraft,
    created_by: CreatedBy,
) -> Result<ExecutionPlanArtifact, ArtifactError> {
    validate_profile_inventory(model, &draft.profiles)?;
    let identity = execution_plan_identity(tuning, &model.graph)?;
    let mut resolved_tactics = Vec::with_capacity(tuning.records.len());
    for record in &tuning.records {
        resolved_tactics.push(ResolvedTactic {
            workload: record.workload.clone(),
            implementation: record.tactic.family.clone(),
            tactic: record.tactic.clone(),
        });
    }
    let plan = ExecutionPlanArtifact {
        schema_version: EXECUTION_PLAN_SCHEMA_VERSION,
        created_by,
        device: identity.device,
        graph_digest: identity.graph_digest,
        implementation_digests: identity.implementation_digests,
        profiles: draft.profiles,
        resolved_tactics,
        packed_weights: draft.packed_weights,
        steps: draft.steps,
    };
    validate_execution_plan(&plan)?;
    Ok(plan)
}

fn validate_profile_inventory(
    model: &onnx_vulkan_frontend::Model,
    profiles: &[ConcreteProfile],
) -> Result<(), ArtifactError> {
    let expected = model
        .graph
        .inputs
        .iter()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    for (index, profile) in profiles.iter().enumerate() {
        let supplied = profile
            .inputs
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        if supplied != expected {
            return invalid(format!(
                "profiles[{index}] inputs do not exactly match graph inputs: expected {expected:?}, got {supplied:?}"
            ));
        }
        for (name, tensor) in &profile.inputs {
            let declared = model.input_type(name);
            if let Some(dtype) = declared.and_then(|declared| declared.dtype) {
                let expected_dtype = u32::try_from(dtype).map_err(|_| {
                    ArtifactError::Invalid(format!("model input `{name}` has negative dtype"))
                })?;
                if tensor.dtype != expected_dtype {
                    return invalid(format!(
                        "profiles[{index}].inputs.{name}.dtype is {}, graph requires {expected_dtype}",
                        tensor.dtype
                    ));
                }
            }
            if let Some(shape) = declared.and_then(|declared| declared.shape.as_ref()) {
                if shape.len() != tensor.dims.len() {
                    return invalid(format!(
                        "profiles[{index}].inputs.{name} has rank {}, model declares {}",
                        tensor.dims.len(),
                        shape.len()
                    ));
                }
            }
        }
        let mut symbols = BTreeMap::<&str, u64>::new();
        for (name, tensor) in &profile.inputs {
            let Some(shape) = model
                .input_type(name)
                .and_then(|declared| declared.shape.as_ref())
            else {
                continue;
            };
            for (axis, (declared, &actual)) in shape.iter().zip(&tensor.dims).enumerate() {
                match declared {
                    onnx_vulkan_frontend::Dim::Fixed(fixed) => {
                        let fixed = u64::try_from(*fixed).map_err(|_| {
                            ArtifactError::Invalid(format!(
                                "model input `{name}` axis {axis} has negative fixed dimension"
                            ))
                        })?;
                        if actual != fixed {
                            return invalid(format!(
                                "profiles[{index}].inputs.{name}.dims[{axis}] is {actual}, model requires {fixed}"
                            ));
                        }
                    }
                    onnx_vulkan_frontend::Dim::Symbol(symbol) => {
                        if let Some(previous) = symbols.insert(symbol, actual)
                            && previous != actual
                        {
                            return invalid(format!(
                                "profiles[{index}] resolves symbol `{symbol}` inconsistently as {previous} and {actual}"
                            ));
                        }
                    }
                    onnx_vulkan_frontend::Dim::Unknown => {}
                }
            }
        }
    }
    Ok(())
}

pub fn execution_plan_identity(
    tuning: &Artifact,
    graph: &onnx_vulkan_core::GraphIr,
) -> Result<ExecutionPlanIdentity, ArtifactError> {
    validate_artifact(tuning)?;
    let mut graph = graph.clone();
    onnx_vulkan_core::prepare_graph(&mut graph);
    let mut implementation_digests = BTreeMap::new();
    for record in &tuning.records {
        match implementation_digests.entry(record.tactic.family.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(record.implementation_digest.clone());
            }
            Entry::Occupied(entry) if entry.get() == &record.implementation_digest => {}
            Entry::Occupied(entry) => {
                return invalid(format!(
                    "tactic family `{}` has multiple implementation digests",
                    entry.key()
                ));
            }
        }
    }
    Ok(ExecutionPlanIdentity {
        device: tuning.device.clone(),
        graph_digest: encode_digest(&onnx_vulkan_core::graph_digest(&graph)),
        implementation_digests,
    })
}

pub fn encode_digest(bytes: &[u8]) -> String {
    use fmt::Write as _;

    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing hexadecimal into a String cannot fail");
    }
    encoded
}

/// Converts validated persistence data into the serialization-free runtime
/// domain table shared by standalone and EP hosts.
pub fn runtime_table(
    artifact: &Artifact,
) -> Result<onnx_vulkan_core::tuning::TuningTable, ArtifactError> {
    use onnx_vulkan_core::tuning::{
        DeviceFingerprint, ImplementationFingerprint, Measurement as RuntimeMeasurement, TacticId,
        TensorSignature, TuningKey, TuningRecord, TuningTable, WorkloadSignature,
    };

    validate_artifact(artifact)?;
    let device = DeviceFingerprint::new(
        artifact.device.name.clone(),
        artifact.device.vendor_id,
        artifact.device.device_id,
        artifact.device.driver_version,
        artifact.device.api_version,
        decode_hex_array::<16>(
            "device.pipeline_cache_uuid",
            &artifact.device.pipeline_cache_uuid,
        )?,
        artifact.device.subgroup_size,
        artifact.device.features.clone(),
    );
    let mut table = TuningTable::new();
    for record in &artifact.records {
        let tensors = |values: &[Tensor]| -> Result<Vec<TensorSignature>, ArtifactError> {
            values
                .iter()
                .map(|tensor| {
                    let dtype = i32::try_from(tensor.dtype)
                        .ok()
                        .and_then(|code| onnx_vulkan_core::ElementType::try_from(code).ok())
                        .filter(|dtype| *dtype != onnx_vulkan_core::ElementType::Undefined)
                        .ok_or_else(|| {
                            ArtifactError::Invalid(format!(
                                "runtime does not understand ONNX dtype {}",
                                tensor.dtype
                            ))
                        })?;
                    Ok(TensorSignature {
                        dtype,
                        dimensions: tensor.dims.clone(),
                    })
                })
                .collect()
        };
        let workload = WorkloadSignature {
            domain: record.workload.domain.clone(),
            op: record.workload.op.clone(),
            inputs: tensors(&record.workload.inputs)?,
            outputs: tensors(&record.workload.outputs)?,
            attributes_digest: decode_hex_array::<32>(
                "record.workload.attributes_digest",
                &record.workload.attributes_digest,
            )?,
        };
        let measurement = RuntimeMeasurement::new(
            record.measurement.samples,
            record.measurement.median_gpu_ns,
            record.measurement.min_gpu_ns,
            record.measurement.max_gpu_ns,
        )
        .ok_or_else(|| ArtifactError::Invalid("invalid runtime measurement summary".into()))?;
        table.insert_best(
            TuningKey {
                device: device.clone(),
                workload,
                implementation: ImplementationFingerprint::from_bytes(decode_hex_array::<32>(
                    "record.implementation_digest",
                    &record.implementation_digest,
                )?),
            },
            TuningRecord::new(
                TacticId::new(record.tactic.family.clone(), record.tactic.id.clone()),
                record.tactic.parameters.clone(),
                measurement,
            ),
        );
    }
    Ok(table)
}

/// Loads runtime policy and artifact once at a session/model-build boundary.
pub fn runtime_resolver_from_env()
-> Result<Arc<onnx_vulkan_core::tuning::TacticResolver>, ArtifactError> {
    let raw_mode = std::env::var("ONNX_VULKAN_TUNING_MODE").unwrap_or_else(|_| "off".into());
    let path = std::env::var_os("ONNX_VULKAN_TUNING_ARTIFACT").map(PathBuf::from);
    runtime_resolver(&raw_mode, path.as_deref())
}

fn runtime_resolver(
    raw_mode: &str,
    path: Option<&Path>,
) -> Result<Arc<onnx_vulkan_core::tuning::TacticResolver>, ArtifactError> {
    use onnx_vulkan_core::tuning::{TacticResolver, TuningMode, TuningTable};

    let mode = match raw_mode {
        "off" => TuningMode::Off,
        "cache-only" => TuningMode::CacheOnly,
        "require-cache" => TuningMode::RequireCache,
        _ => {
            return invalid(format!(
                "ONNX_VULKAN_TUNING_MODE must be off, cache-only, or require-cache; got `{raw_mode}`"
            ));
        }
    };
    if mode == TuningMode::Off {
        return Ok(Arc::new(TacticResolver::off()));
    }
    let table = match path {
        Some(path) => {
            let artifact = read_artifact(path)?;
            if mode == TuningMode::RequireCache
                && artifact
                    .coverage
                    .as_ref()
                    .is_some_and(|coverage| !coverage.strict_compatible())
            {
                return Err(ArtifactError::Incompatible(
                    "require-cache refuses an artifact with partial model coverage".into(),
                ));
            }
            runtime_table(&artifact)?
        }
        None if mode == TuningMode::CacheOnly => TuningTable::new(),
        None => {
            return invalid("ONNX_VULKAN_TUNING_ARTIFACT is required when mode is require-cache");
        }
    };
    Ok(Arc::new(TacticResolver::new(mode, table)))
}

fn decode_hex_array<const N: usize>(field: &str, value: &str) -> Result<[u8; N], ArtifactError> {
    validate_hex(field, value, N)?;
    let mut decoded = [0_u8; N];
    for (output, pair) in decoded.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        let digit = |byte: u8| match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            _ => unreachable!("validate_hex accepted only lowercase hexadecimal"),
        };
        *output = digit(pair[0]) << 4 | digit(pair[1]);
    }
    Ok(decoded)
}

pub fn canonicalize(artifact: &mut Artifact) -> Result<(), ArtifactError> {
    artifact.device.features.sort_unstable();
    artifact.device.features.dedup();
    artifact.records.sort_by_key(record_key);
    artifact.diagnostics.sort_by(|left, right| {
        (&left.workload, &left.tactic_id, &left.outcome, &left.detail).cmp(&(
            &right.workload,
            &right.tactic_id,
            &right.outcome,
            &right.detail,
        ))
    });

    let mut unique = Vec::with_capacity(artifact.records.len());
    for record in artifact.records.drain(..) {
        if let Some(previous) = unique.last() {
            if record_key(previous) == record_key(&record) {
                if previous == &record {
                    continue;
                }
                return Err(ArtifactError::ConflictingKey(record_key(&record)));
            }
        }
        unique.push(record);
    }
    artifact.records = unique;
    validate_artifact(artifact)
}

pub fn canonical_json(artifact: &Artifact) -> Result<Vec<u8>, ArtifactError> {
    let mut canonical = artifact.clone();
    canonicalize(&mut canonical)?;
    let mut bytes = serde_json::to_vec_pretty(&canonical)
        .map_err(|error| ArtifactError::Json(error.to_string()))?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn validate_artifact(artifact: &Artifact) -> Result<(), ArtifactError> {
    match artifact.schema_version {
        SCHEMA_VERSION => {}
        0..SCHEMA_VERSION => return Err(ArtifactError::OldSchema(artifact.schema_version)),
        version => return Err(ArtifactError::NewerSchema(version)),
    }
    validate_created_by(&artifact.created_by)?;
    validate_device(&artifact.device)?;
    validate_hex("profile.model_digest", &artifact.profile.model_digest, 32)?;
    if artifact.profile.symbol_values.keys().any(String::is_empty) {
        return invalid("profile.symbol_values contains an empty name");
    }
    if artifact.measurement_policy.warmup_iterations == 0 {
        return invalid("measurement_policy.warmup_iterations is zero");
    }
    if artifact.measurement_policy.measured_iterations < 20 {
        return invalid("measurement_policy.measured_iterations must be at least 20");
    }
    if artifact.measurement_policy.statistic != "median" {
        return invalid("measurement_policy.statistic must be `median`");
    }
    if artifact.measurement_policy.clock != "vulkan_timestamp" {
        return invalid("measurement_policy.clock must be `vulkan_timestamp`");
    }
    if artifact.records.is_empty() {
        return invalid("records is empty");
    }
    if let Some(coverage) = &artifact.coverage {
        require_nonempty("coverage.scope", &coverage.scope)?;
        if coverage.inventory_workloads == 0 {
            return invalid("coverage.inventory_workloads is zero");
        }
        if coverage.tuned_workloads != artifact.records.len() as u64 {
            return invalid("coverage.tuned_workloads does not match records");
        }
        if coverage.tuned_workloads > coverage.inventory_workloads {
            return invalid("coverage tunes more workloads than its inventory contains");
        }
    }

    let mut exact = BTreeMap::<String, &Record>::new();
    for (index, record) in artifact.records.iter().enumerate() {
        validate_record(
            record,
            index,
            artifact.measurement_policy.measured_iterations,
        )?;
        let key = record_key(record);
        match exact.entry(key.clone()) {
            Entry::Vacant(slot) => {
                slot.insert(record);
            }
            Entry::Occupied(slot) if slot.get() == &record => {}
            Entry::Occupied(_) => return Err(ArtifactError::ConflictingKey(key)),
        }
    }
    for (index, diagnostic) in artifact.diagnostics.iter().enumerate() {
        validate_workload(
            &diagnostic.workload,
            &format!("diagnostics[{index}].workload"),
        )?;
        require_nonempty(
            &format!("diagnostics[{index}].tactic_id"),
            &diagnostic.tactic_id,
        )?;
        require_nonempty(
            &format!("diagnostics[{index}].outcome"),
            &diagnostic.outcome,
        )?;
        require_nonempty(&format!("diagnostics[{index}].detail"), &diagnostic.detail)?;
        if let Some(digest) = &diagnostic.implementation_digest {
            validate_hex(
                &format!("diagnostics[{index}].implementation_digest"),
                digest,
                32,
            )?;
        }
        if let Some(tactic) = &diagnostic.tactic {
            validate_tactic(tactic, &format!("diagnostics[{index}].tactic"))?;
        }
    }
    Ok(())
}

fn validate_created_by(created_by: &CreatedBy) -> Result<(), ArtifactError> {
    require_nonempty(
        "created_by.onnx_vulkan_version",
        &created_by.onnx_vulkan_version,
    )?;
    require_nonempty("created_by.git_commit", &created_by.git_commit)?;
    if created_by.command_line.is_empty() {
        return invalid("created_by.command_line is empty");
    }
    if created_by.created_unix_seconds == 0 {
        return invalid("created_by.created_unix_seconds is zero");
    }
    Ok(())
}

fn validate_device(device: &Device) -> Result<(), ArtifactError> {
    require_nonempty("device.name", &device.name)?;
    validate_hex(
        "device.pipeline_cache_uuid",
        &device.pipeline_cache_uuid,
        16,
    )?;
    if device.subgroup_size == 0 {
        return invalid("device.subgroup_size is zero");
    }
    if device.features.iter().any(String::is_empty) {
        return invalid("device.features contains an empty identifier");
    }
    Ok(())
}

fn validate_record(
    record: &Record,
    index: usize,
    required_samples: u32,
) -> Result<(), ArtifactError> {
    let prefix = format!("records[{index}]");
    validate_workload(&record.workload, &format!("{prefix}.workload"))?;
    validate_hex(
        &format!("{prefix}.implementation_digest"),
        &record.implementation_digest,
        32,
    )?;
    validate_tactic(&record.tactic, &format!("{prefix}.tactic"))?;
    let measurement = &record.measurement;
    if measurement.samples < required_samples {
        return invalid(format!(
            "{prefix}.measurement.samples {} is below policy {}",
            measurement.samples, required_samples
        ));
    }
    if measurement.min_gpu_ns == 0
        || measurement.min_gpu_ns > measurement.median_gpu_ns
        || measurement.median_gpu_ns > measurement.max_gpu_ns
    {
        return invalid(format!("{prefix}.measurement summary is zero or unordered"));
    }
    if let Some(samples) = &measurement.diagnostic_samples_gpu_ns {
        if samples.is_empty() || samples.len() > measurement.samples as usize {
            return invalid(format!(
                "{prefix}.measurement.diagnostic_samples_gpu_ns has invalid length"
            ));
        }
        if samples.contains(&0) {
            return invalid(format!(
                "{prefix}.measurement.diagnostic_samples_gpu_ns contains zero"
            ));
        }
    }
    require_nonempty(
        &format!("{prefix}.correctness.kind"),
        &record.correctness.kind,
    )?;
    if !record.correctness.passed {
        return invalid(format!("{prefix}.correctness.passed is false"));
    }
    Ok(())
}

fn validate_workload(workload: &Workload, prefix: &str) -> Result<(), ArtifactError> {
    require_nonempty(&format!("{prefix}.op"), &workload.op)?;
    validate_hex(
        &format!("{prefix}.attributes_digest"),
        &workload.attributes_digest,
        32,
    )?;
    for (tensor_index, tensor) in workload.inputs.iter().chain(&workload.outputs).enumerate() {
        if tensor.dtype == 0 {
            return invalid(format!("{prefix}.tensor[{tensor_index}].dtype is zero"));
        }
    }
    Ok(())
}

fn validate_tactic(tactic: &Tactic, prefix: &str) -> Result<(), ArtifactError> {
    require_nonempty(&format!("{prefix}.family"), &tactic.family)?;
    require_nonempty(&format!("{prefix}.id"), &tactic.id)?;
    if tactic.parameters.keys().any(String::is_empty) {
        return invalid(format!("{prefix}.parameters contains an empty name"));
    }
    Ok(())
}

pub fn reduce_diagnostic(
    artifact: &Artifact,
    diagnostic_index: usize,
) -> Result<ReducedFixture, ArtifactError> {
    validate_artifact(artifact)?;
    let diagnostic = artifact.diagnostics.get(diagnostic_index).ok_or_else(|| {
        ArtifactError::Invalid(format!(
            "diagnostic index {diagnostic_index} is out of range for {} diagnostics",
            artifact.diagnostics.len()
        ))
    })?;
    Ok(ReducedFixture {
        schema_version: 1,
        artifact_schema_version: artifact.schema_version,
        device: artifact.device.clone(),
        profile: artifact.profile.clone(),
        workload: diagnostic.workload.clone(),
        tactic_id: diagnostic.tactic_id.clone(),
        tactic: diagnostic.tactic.clone(),
        implementation_digest: diagnostic.implementation_digest.clone(),
        failure: FailureDetail {
            outcome: diagnostic.outcome.clone(),
            detail: diagnostic.detail.clone(),
        },
    })
}

pub fn write_reduced_fixture(path: &Path, fixture: &ReducedFixture) -> Result<(), ArtifactError> {
    let mut bytes = serde_json::to_vec_pretty(fixture)
        .map_err(|error| ArtifactError::Json(error.to_string()))?;
    bytes.push(b'\n');
    atomic_write(path, &bytes)
}

fn validate_hex(field: &str, value: &str, bytes: usize) -> Result<(), ArtifactError> {
    if value.len() != bytes * 2 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return invalid(format!("{field} must be exactly {bytes} hexadecimal bytes"));
    }
    if value.bytes().any(|byte| byte.is_ascii_uppercase()) {
        return invalid(format!("{field} must use canonical lowercase hexadecimal"));
    }
    Ok(())
}

fn require_nonempty(field: &str, value: &str) -> Result<(), ArtifactError> {
    if value.is_empty() {
        invalid(format!("{field} is empty"))
    } else {
        Ok(())
    }
}

fn invalid<T>(message: impl Into<String>) -> Result<T, ArtifactError> {
    Err(ArtifactError::Invalid(message.into()))
}

pub fn record_key(record: &Record) -> String {
    let workload = serde_json::to_string(&record.workload)
        .unwrap_or_else(|_| "<unserializable-workload>".to_owned());
    format!("{}:{workload}", record.implementation_digest)
}

pub fn merge_artifacts(artifacts: &[Artifact]) -> Result<Artifact, ArtifactError> {
    let Some(first) = artifacts.first() else {
        return invalid("merge requires at least one artifact");
    };
    let mut merged = first.clone();
    canonicalize(&mut merged)?;
    let mut records = merged
        .records
        .drain(..)
        .map(|record| (record_key(&record), record))
        .collect::<BTreeMap<_, _>>();

    for artifact in &artifacts[1..] {
        validate_artifact(artifact)?;
        compatible(&merged, artifact)?;
        for record in &artifact.records {
            let key = record_key(record);
            match records.entry(key.clone()) {
                Entry::Vacant(slot) => {
                    slot.insert(record.clone());
                }
                Entry::Occupied(slot) if slot.get() == record => {}
                Entry::Occupied(_) => return Err(ArtifactError::ConflictingKey(key)),
            }
        }
        merged.diagnostics.extend(artifact.diagnostics.clone());
    }
    merged.records = records.into_values().collect();
    canonicalize(&mut merged)?;
    Ok(merged)
}

/// Validates that recorded tactics can be forced against a freshly generated
/// target inventory. The returned artifact retains the recorded tactics only
/// after device, model/profile, workload coverage, and implementation digests
/// match exactly.
pub fn prepare_replay(recorded: &Artifact, target: &Artifact) -> Result<Artifact, ArtifactError> {
    validate_artifact(recorded)?;
    validate_artifact(target)?;
    if recorded.device != target.device {
        return Err(ArtifactError::Incompatible(
            "replay device fingerprint is stale".into(),
        ));
    }
    if recorded.profile != target.profile {
        return Err(ArtifactError::Incompatible(
            "replay model/profile fingerprint is stale".into(),
        ));
    }
    let recorded_by_workload = comparison_records(recorded)?;
    let target_by_workload = comparison_records(target)?;
    let mut failures = Vec::new();
    for (workload, current) in &target_by_workload {
        match recorded_by_workload.get(workload) {
            None => failures.push(format!("missing workload {workload}")),
            Some(previous) if previous.implementation_digest != current.implementation_digest => {
                failures.push(format!(
                    "stale implementation digest for workload {workload}: recorded {}, current {}",
                    previous.implementation_digest, current.implementation_digest
                ));
            }
            Some(_) => {}
        }
    }
    if !failures.is_empty() {
        return Err(ArtifactError::Incompatible(format!(
            "replay compatibility failed: {}",
            failures.join("; ")
        )));
    }
    let mut replay = recorded.clone();
    replay.records.retain(|record| {
        let workload = serde_json::to_string(&record.workload)
            .unwrap_or_else(|_| "<unserializable-workload>".to_owned());
        target_by_workload.contains_key(&workload)
    });
    canonicalize(&mut replay)?;
    Ok(replay)
}

fn compatible(left: &Artifact, right: &Artifact) -> Result<(), ArtifactError> {
    if left.schema_version != right.schema_version {
        return Err(ArtifactError::Incompatible("schema versions differ".into()));
    }
    if left.device != right.device {
        return Err(ArtifactError::Incompatible(
            "device fingerprints differ".into(),
        ));
    }
    if left.profile != right.profile {
        return Err(ArtifactError::Incompatible("model profiles differ".into()));
    }
    if left.measurement_policy != right.measurement_policy {
        return Err(ArtifactError::Incompatible(
            "measurement policies differ".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct InspectOutput {
    pub schema_version: u32,
    pub output_schema_version: u32,
    pub device_name: String,
    pub model_digest: String,
    pub record_count: usize,
    pub diagnostic_count: usize,
    pub operators: BTreeMap<String, usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coverage: Option<ModelCoverage>,
}

pub fn inspect(artifact: &Artifact) -> Result<InspectOutput, ArtifactError> {
    validate_artifact(artifact)?;
    let mut operators = BTreeMap::new();
    for record in &artifact.records {
        *operators.entry(record.workload.op.clone()).or_insert(0) += 1;
    }
    Ok(InspectOutput {
        schema_version: artifact.schema_version,
        output_schema_version: OUTPUT_SCHEMA_VERSION,
        device_name: artifact.device.name.clone(),
        model_digest: artifact.profile.model_digest.clone(),
        record_count: artifact.records.len(),
        diagnostic_count: artifact.diagnostics.len(),
        operators,
        coverage: artifact.coverage.clone(),
    })
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ChangedRecord {
    pub workload_key: String,
    pub source_digest_changed: bool,
    pub tactic_changed: bool,
    pub timing_changed: bool,
    pub correctness_changed: bool,
    pub before: Record,
    pub after: Record,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct MetadataDiff {
    pub schema_changed: bool,
    pub device_changed: bool,
    pub profile_changed: bool,
    pub policy_changed: bool,
    pub device_before: Device,
    pub device_after: Device,
    pub profile_before: Profile,
    pub profile_after: Profile,
    pub policy_before: MeasurementPolicy,
    pub policy_after: MeasurementPolicy,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct DiffOutput {
    pub output_schema_version: u32,
    pub compatible: bool,
    pub metadata: MetadataDiff,
    pub added: Vec<Record>,
    pub removed: Vec<Record>,
    pub changed: Vec<ChangedRecord>,
}

#[derive(Debug, Deserialize)]
struct IntermediateManifest {
    schema_version: u32,
    intermediates: Vec<IntermediateNode>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct IntermediateNode {
    pub output_name: String,
    pub dtype: String,
    pub node_index: usize,
    pub node_name: String,
    pub domain: String,
    pub op: String,
    pub tactic_family: String,
}

#[derive(Debug, Deserialize)]
struct BackendComparisonReport {
    schema_version: u32,
    outputs: Vec<BackendOutputComparison>,
}

#[derive(Debug, Deserialize)]
struct BackendOutputComparison {
    name: String,
    passed: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct BisectOutput {
    pub output_schema_version: u32,
    pub compared_intermediates: usize,
    pub all_passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_divergence: Option<IntermediateNode>,
}

/// Finds the first divergent exposed graph value in original node order.
pub fn bisect_reports(
    manifest_bytes: &[u8],
    report_bytes: &[u8],
) -> Result<BisectOutput, ArtifactError> {
    let mut manifest: IntermediateManifest = serde_json::from_slice(manifest_bytes)
        .map_err(|error| ArtifactError::Json(format!("intermediate manifest: {error}")))?;
    let report: BackendComparisonReport = serde_json::from_slice(report_bytes)
        .map_err(|error| ArtifactError::Json(format!("backend comparison report: {error}")))?;
    if manifest.schema_version != 1 {
        return invalid(format!(
            "unsupported intermediate manifest schema {}",
            manifest.schema_version
        ));
    }
    if report.schema_version != 1 {
        return invalid(format!(
            "unsupported backend report schema {}",
            report.schema_version
        ));
    }
    manifest.intermediates.sort_by_key(|node| node.node_index);
    let outcomes = report
        .outputs
        .into_iter()
        .map(|output| (output.name, output.passed))
        .collect::<BTreeMap<_, _>>();
    let mut compared = 0;
    for node in manifest.intermediates {
        let passed = outcomes.get(&node.output_name).ok_or_else(|| {
            ArtifactError::Invalid(format!(
                "backend report has no exposed output {:?} for node {}",
                node.output_name, node.node_index
            ))
        })?;
        compared += 1;
        if !passed {
            return Ok(BisectOutput {
                output_schema_version: OUTPUT_SCHEMA_VERSION,
                compared_intermediates: compared,
                all_passed: false,
                first_divergence: Some(node),
            });
        }
    }
    Ok(BisectOutput {
        output_schema_version: OUTPUT_SCHEMA_VERSION,
        compared_intermediates: compared,
        all_passed: true,
        first_divergence: None,
    })
}

pub fn diff(left: &Artifact, right: &Artifact) -> Result<DiffOutput, ArtifactError> {
    validate_artifact(left)?;
    validate_artifact(right)?;
    let metadata = MetadataDiff {
        schema_changed: left.schema_version != right.schema_version,
        device_changed: left.device != right.device,
        profile_changed: left.profile != right.profile,
        policy_changed: left.measurement_policy != right.measurement_policy,
        device_before: left.device.clone(),
        device_after: right.device.clone(),
        profile_before: left.profile.clone(),
        profile_after: right.profile.clone(),
        policy_before: left.measurement_policy.clone(),
        policy_after: right.measurement_policy.clone(),
    };
    let left_records = comparison_records(left)?;
    let right_records = comparison_records(right)?;
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();
    for (key, record) in &right_records {
        match left_records.get(key) {
            None => added.push((*record).clone()),
            Some(before) if *before != *record => changed.push(ChangedRecord {
                workload_key: key.clone(),
                source_digest_changed: before.implementation_digest != record.implementation_digest,
                tactic_changed: before.tactic != record.tactic,
                timing_changed: before.measurement != record.measurement,
                correctness_changed: before.correctness != record.correctness,
                before: (*before).clone(),
                after: (*record).clone(),
            }),
            Some(_) => {}
        }
    }
    for (key, record) in left_records {
        if !right_records.contains_key(&key) {
            removed.push(record.clone());
        }
    }
    Ok(DiffOutput {
        output_schema_version: OUTPUT_SCHEMA_VERSION,
        compatible: !metadata.schema_changed
            && !metadata.device_changed
            && !metadata.profile_changed
            && !metadata.policy_changed,
        metadata,
        added,
        removed,
        changed,
    })
}

fn comparison_records(artifact: &Artifact) -> Result<BTreeMap<String, &Record>, ArtifactError> {
    let mut records = BTreeMap::new();
    for record in &artifact.records {
        let key = serde_json::to_string(&record.workload)
            .map_err(|error| ArtifactError::Json(error.to_string()))?;
        if records.insert(key.clone(), record).is_some() {
            return Err(ArtifactError::ConflictingKey(format!(
                "multiple implementation records share workload {key}"
            )));
        }
    }
    Ok(records)
}

static TEMP_SERIAL: AtomicU64 = AtomicU64::new(0);

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), ArtifactError> {
    atomic_write_impl(path, bytes, false)
}

fn atomic_write_impl(
    path: &Path,
    bytes: &[u8],
    fail_before_rename: bool,
) -> Result<(), ArtifactError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ArtifactError::Io(format!("invalid output path {}", path.display())))?;
    let mut temporary = None;
    for _ in 0..100 {
        let serial = TEMP_SERIAL.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{name}.tmp-{}-{serial}", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(ArtifactError::Io(format!(
                    "create temporary file beside {}: {error}",
                    path.display()
                )));
            }
        }
    }
    let Some((temporary_path, mut file)) = temporary else {
        return Err(ArtifactError::Io(format!(
            "could not reserve temporary file beside {}",
            path.display()
        )));
    };
    let result = (|| {
        file.write_all(bytes).map_err(|error| {
            ArtifactError::Io(format!("write {}: {error}", temporary_path.display()))
        })?;
        file.sync_all().map_err(|error| {
            ArtifactError::Io(format!("flush {}: {error}", temporary_path.display()))
        })?;
        drop(file);
        if fail_before_rename {
            return Err(ArtifactError::Io(
                "simulated interruption before rename".into(),
            ));
        }
        fs::rename(&temporary_path, path).map_err(|error| {
            ArtifactError::Io(format!(
                "replace {} with {}: {error}",
                path.display(),
                temporary_path.display()
            ))
        })?;
        if let Ok(directory) = OpenOptions::new().read(true).open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

pub fn write_artifact(path: &Path, artifact: &Artifact) -> Result<(), ArtifactError> {
    atomic_write(path, &canonical_json(artifact)?)
}

pub fn sha256_file(path: &Path) -> Result<String, ArtifactError> {
    let bytes = fs::read(path)
        .map_err(|error| ArtifactError::Io(format!("read model {}: {error}", path.display())))?;
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing hexadecimal into a String cannot fail");
    }
    Ok(encoded)
}

/// Builds the exact, GPU-free workload inventory for the first model-level
/// tuning slice. Only static/profile-resolved, group-1, square 2D fp32 Conv is
/// executable by the current generic family runner; every other node is
/// retained as an explicit skip instead of being silently ignored.
pub fn inventory_model(
    path: &Path,
    symbol_values: BTreeMap<String, u64>,
) -> Result<ModelInventory, ArtifactError> {
    use onnx_vulkan_core::AttrValue;
    use onnx_vulkan_frontend::shape::Dim;

    let mut model = onnx_vulkan_frontend::load(path).map_err(|error| {
        ArtifactError::Invalid(format!("load model {}: {error}", path.display()))
    })?;
    onnx_vulkan_core::prepare_graph(&mut model.graph);
    let model_digest = sha256_file(path)?;
    let mut workloads = Vec::new();
    let mut skipped = Vec::new();

    let resolve_tensor = |name: &str| -> Result<Tensor, String> {
        let value = model
            .types
            .get(name)
            .ok_or_else(|| format!("shape/type inference has no value `{name}`"))?;
        let dtype = u32::try_from(
            value
                .dtype
                .ok_or_else(|| format!("dtype of `{name}` is unknown"))?,
        )
        .map_err(|_| format!("dtype of `{name}` is negative"))?;
        let dimensions = value
            .shape
            .as_ref()
            .ok_or_else(|| format!("shape of `{name}` is unknown"))?
            .iter()
            .map(|dimension| match dimension {
                Dim::Fixed(value) => u64::try_from(*value)
                    .map_err(|_| format!("shape of `{name}` has negative dimension {value}")),
                Dim::Symbol(symbol) => symbol_values
                    .get(symbol)
                    .copied()
                    .ok_or_else(|| format!("shape of `{name}` requires --dim {symbol}=N")),
                Dim::Unknown => Err(format!(
                    "shape of `{name}` has an unnamed unknown dimension"
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if dimensions.contains(&0) {
            return Err(format!("shape of `{name}` contains a zero dimension"));
        }
        Ok(Tensor {
            dtype,
            dims: dimensions,
        })
    };

    for (index, node) in model.graph.nodes.iter().enumerate() {
        let node_name = if node.name.is_empty() {
            format!("{}#{index}", node.op)
        } else {
            node.name.clone()
        };
        let skip = |reason: String| InventorySkip {
            node: node_name.clone(),
            op: node.op.clone(),
            reason,
        };
        if node.op != "Conv" || !(node.domain.is_empty() || node.domain == "ai.onnx") {
            skipped.push(skip("no model-level family runner".into()));
            continue;
        }
        let group = node
            .attrs
            .get("group")
            .and_then(AttrValue::as_i64)
            .unwrap_or(1);
        if group != 1 {
            skipped.push(skip(format!(
                "Conv group={group}; v1 runner requires group=1"
            )));
            continue;
        }
        let inputs = match node
            .inputs
            .iter()
            .filter(|name| !name.is_empty())
            .map(|name| resolve_tensor(name))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(inputs) => inputs,
            Err(reason) => {
                skipped.push(skip(reason));
                continue;
            }
        };
        let outputs = match node
            .outputs
            .iter()
            .filter(|name| !name.is_empty())
            .map(|name| resolve_tensor(name))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(outputs) => outputs,
            Err(reason) => {
                skipped.push(skip(reason));
                continue;
            }
        };
        if inputs.len() < 2 || outputs.len() != 1 {
            skipped.push(skip("Conv needs data/weight and exactly one output".into()));
            continue;
        }
        if inputs
            .iter()
            .chain(&outputs)
            .any(|tensor| tensor.dtype != 1)
        {
            skipped.push(skip("v1 runner supports only fp32 Conv tensors".into()));
            continue;
        }
        let (x, weight, output) = (&inputs[0].dims, &inputs[1].dims, &outputs[0].dims);
        if x.len() != 4 || weight.len() != 4 || output.len() != 4 {
            skipped.push(skip("v1 runner requires rank-4 NCHW Conv".into()));
            continue;
        }
        if x[0] != 1 || output[0] != 1 {
            skipped.push(skip("v1 runner requires batch=1".into()));
            continue;
        }
        if x[2] != x[3] || output[2] != output[3] || weight[2] != weight[3] {
            skipped.push(skip(
                "v1 runner requires square spatial and kernel dimensions".into(),
            ));
            continue;
        }
        if weight[0] != output[1] || weight[1] != x[1] {
            skipped.push(skip("Conv channel dimensions are inconsistent".into()));
            continue;
        }
        let ints = |name: &str, default: Vec<i64>| {
            node.attrs
                .get(name)
                .and_then(AttrValue::as_ints)
                .map_or(default, <[i64]>::to_vec)
        };
        let strides = ints("strides", vec![1, 1]);
        let pads = ints("pads", vec![0, 0, 0, 0]);
        let dilations = ints("dilations", vec![1, 1]);
        let auto_pad = node
            .attrs
            .get("auto_pad")
            .and_then(AttrValue::as_str)
            .unwrap_or("NOTSET");
        if strides.len() != 2
            || strides[0] != strides[1]
            || pads.len() != 4
            || !pads.iter().all(|pad| *pad == pads[0])
            || dilations != [1, 1]
            || auto_pad != "NOTSET"
        {
            skipped.push(skip(
                "v1 runner requires equal strides, symmetric equal pads, dilation=1, auto_pad=NOTSET"
                    .into(),
            ));
            continue;
        }
        let positive = |name: &str, value: i64| {
            u64::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| format!("{name} must be positive"))
        };
        let stride = match positive("stride", strides[0]) {
            Ok(value) => value,
            Err(reason) => {
                skipped.push(skip(reason));
                continue;
            }
        };
        let pad = match u64::try_from(pads[0]) {
            Ok(value) => value,
            Err(_) => {
                skipped.push(skip("pad must be non-negative".into()));
                continue;
            }
        };
        let c_in = x[1];
        let c_out = output[1];
        let kernel = weight[2];
        let h_in = x[2];
        let h_out = output[2];
        workloads.push(InventoryWorkload {
            node: node_name,
            workload: Workload {
                domain: node.domain.clone(),
                op: node.op.clone(),
                inputs,
                outputs,
                attributes_digest: encode_bytes(&onnx_vulkan_core::tuning::attributes_digest(node)),
            },
            implementation_digest: encode_bytes(
                onnx_vulkan_core::shaders::conv::implementation_fingerprint().as_bytes(),
            ),
            runner: ConvRunnerGeometry {
                family: "conv-f32".into(),
                c_in,
                c_out,
                kernel,
                h_in,
                h_out,
                stride,
                pad,
            },
        });
    }
    workloads.sort_unstable();
    workloads.dedup_by(|left, right| left.workload == right.workload);
    skipped.sort_unstable();
    Ok(ModelInventory {
        schema_version: 1,
        profile: Profile {
            model_digest: model_digest.clone(),
            symbol_values,
        },
        model_digest,
        workloads,
        skipped,
    })
}

fn encode_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

pub fn resume_artifact(output: &Path, draft: Artifact) -> Result<Artifact, ArtifactError> {
    if !output.exists() {
        return Ok(draft);
    }
    let current_provenance = draft.created_by.clone();
    let previous = read_artifact(output)?;
    let mut resumed = merge_artifacts(&[previous, draft])?;
    resumed.created_by = current_provenance;
    Ok(resumed)
}

pub fn path_list(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn artifact() -> Artifact {
        Artifact {
            schema_version: 1,
            created_by: CreatedBy {
                onnx_vulkan_version: "0.0.1".into(),
                git_commit: "abc123".into(),
                dirty: false,
                command_line: vec!["onnx-vulkan-tune".into(), "build".into()],
                created_unix_seconds: 1,
            },
            device: Device {
                name: "gpu".into(),
                vendor_id: 1,
                device_id: 2,
                driver_version: 3,
                api_version: 4,
                pipeline_cache_uuid: "00".repeat(16),
                subgroup_size: 32,
                features: vec!["z".into(), "a".into(), "a".into()],
            },
            profile: Profile {
                model_digest: "11".repeat(32),
                symbol_values: BTreeMap::new(),
            },
            measurement_policy: MeasurementPolicy {
                warmup_iterations: 2,
                measured_iterations: 20,
                statistic: "median".into(),
                clock: "vulkan_timestamp".into(),
            },
            records: vec![Record {
                workload: Workload {
                    domain: String::new(),
                    op: "Conv".into(),
                    inputs: vec![Tensor {
                        dtype: 1,
                        dims: vec![1, 3, 224, 224],
                    }],
                    outputs: Vec::new(),
                    attributes_digest: "22".repeat(32),
                },
                implementation_digest: "33".repeat(32),
                tactic: Tactic {
                    family: "blocked".into(),
                    id: "tile64".into(),
                    parameters: BTreeMap::from([("tile".into(), 64)]),
                },
                measurement: Measurement {
                    samples: 20,
                    median_gpu_ns: 110,
                    min_gpu_ns: 100,
                    max_gpu_ns: 120,
                    diagnostic_samples_gpu_ns: Some(vec![100, 110, 120]),
                },
                correctness: Correctness {
                    kind: "max_abs_rel".into(),
                    passed: true,
                },
            }],
            diagnostics: Vec::new(),
            coverage: None,
        }
    }

    fn execution_plan() -> ExecutionPlanArtifact {
        let tuning = artifact();
        ExecutionPlanArtifact {
            schema_version: EXECUTION_PLAN_SCHEMA_VERSION,
            created_by: tuning.created_by,
            device: tuning.device,
            graph_digest: "55".repeat(32),
            implementation_digests: BTreeMap::from([("conv".into(), "33".repeat(32))]),
            profiles: vec![ConcreteProfile {
                id: "batch1".into(),
                inputs: BTreeMap::from([(
                    "image".into(),
                    Tensor {
                        dtype: 1,
                        dims: vec![1, 3, 224, 224],
                    },
                )]),
            }],
            resolved_tactics: vec![ResolvedTactic {
                workload: tuning.records[0].workload.clone(),
                implementation: "conv".into(),
                tactic: tuning.records[0].tactic.clone(),
            }],
            packed_weights: vec![PackedWeightMetadata {
                name: "conv.weight".into(),
                dtype: 1,
                dims: vec![64, 3, 7, 7],
                layout: "conv-blocked-v1".into(),
                bytes: 37_632,
                source_digest: "66".repeat(32),
            }],
            steps: vec![StepPlanMetadata {
                profile: "batch1".into(),
                dispatches: 12,
                host_nodes: 2,
                temporary_buffers: vec![TemporaryBufferMetadata {
                    name: "activation-arena".into(),
                    bytes: 4096,
                    alignment: 256,
                }],
            }],
        }
    }

    fn temporary_path(name: &str) -> PathBuf {
        let serial = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("test clock is after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("onnx-vulkan-tune-{name}-{serial}"))
    }

    #[test]
    fn canonical_round_trip_is_byte_exact() {
        let first = canonical_json(&artifact()).expect("valid test artifact");
        let parsed = parse_artifact(&first).expect("canonical JSON parses");
        let second = canonical_json(&parsed).expect("parsed artifact canonicalizes");
        assert_eq!(first, second);
        assert_eq!(parsed.device.features, ["a", "z"]);
    }

    #[test]
    fn execution_plan_round_trip_is_canonical_and_contains_metadata_only() {
        let plan = execution_plan();
        let first = canonical_execution_plan_json(&plan).expect("valid execution plan");
        let parsed = parse_execution_plan(&first).expect("execution plan parses");
        let second = canonical_execution_plan_json(&parsed).expect("plan canonicalizes");
        assert_eq!(first, second);
        assert_eq!(parsed.packed_weights[0].bytes, 37_632);
        assert!(
            !String::from_utf8(first)
                .expect("JSON is UTF-8")
                .contains("weight_bytes")
        );
    }

    #[test]
    fn execution_plan_identity_and_profile_references_are_exact() {
        let plan = execution_plan();
        validate_execution_plan_for(
            &plan,
            &plan.device,
            &plan.graph_digest,
            &plan.implementation_digests,
        )
        .expect("exact identity matches");

        let mut stale = plan.implementation_digests.clone();
        stale.insert("conv".into(), "77".repeat(32));
        assert!(matches!(
            validate_execution_plan_for(&plan, &plan.device, &plan.graph_digest, &stale),
            Err(ArtifactError::Incompatible(_))
        ));

        let mut unknown_profile = plan;
        unknown_profile.steps[0].profile = "dynamic".into();
        assert!(matches!(
            validate_execution_plan(&unknown_profile),
            Err(ArtifactError::Invalid(_))
        ));
    }

    #[test]
    fn conflicting_exact_key_is_rejected() {
        let mut value = artifact();
        let mut conflict = value.records[0].clone();
        conflict.tactic.id = "tile128".into();
        value.records.push(conflict);
        assert!(matches!(
            canonicalize(&mut value),
            Err(ArtifactError::ConflictingKey(_))
        ));
    }

    #[test]
    fn interrupted_atomic_write_keeps_previous_artifact() {
        let path = temporary_path("atomic");
        fs::write(&path, b"previous").expect("create previous artifact");
        let result = atomic_write_impl(&path, b"replacement", true);
        assert!(matches!(result, Err(ArtifactError::Io(_))));
        assert_eq!(
            fs::read(&path).expect("read previous artifact"),
            b"previous"
        );
        fs::remove_file(path).expect("remove test artifact");
    }

    #[test]
    fn merge_rejects_incompatible_device() {
        let left = artifact();
        let mut right = artifact();
        right.device.driver_version += 1;
        assert!(matches!(
            merge_artifacts(&[left, right]),
            Err(ArtifactError::Incompatible(_))
        ));
    }

    #[test]
    fn diff_separates_device_policy_source_tactic_and_timing() {
        let left = artifact();
        let mut right = left.clone();
        right.device.driver_version += 1;
        right.measurement_policy.warmup_iterations += 1;
        right.records[0].implementation_digest = "44".repeat(32);
        right.records[0].tactic.id = "tile128".into();
        right.records[0].measurement.median_gpu_ns += 1;

        let report = diff(&left, &right).expect("valid artifacts can always be compared");
        assert!(!report.compatible);
        assert!(report.metadata.device_changed);
        assert!(report.metadata.policy_changed);
        assert!(!report.metadata.profile_changed);
        assert_eq!(report.changed.len(), 1);
        assert!(report.changed[0].source_digest_changed);
        assert!(report.changed[0].tactic_changed);
        assert!(report.changed[0].timing_changed);
        assert!(!report.changed[0].correctness_changed);
        assert!(report.added.is_empty());
        assert!(report.removed.is_empty());
    }

    #[test]
    fn bisect_returns_first_divergent_node_in_graph_order() {
        let manifest = br#"{
          "schema_version": 1,
          "intermediates": [
            {"output_name":"later","dtype":"FLOAT","node_index":9,"node_name":"n9","domain":"","op":"Add","tactic_family":"ai.onnx::Add"},
            {"output_name":"first","dtype":"INT64","node_index":3,"node_name":"n3","domain":"","op":"Gather","tactic_family":"ai.onnx::Gather"}
          ]
        }"#;
        let report = br#"{
          "schema_version": 1,
          "outputs": [
            {"name":"later","passed":false},
            {"name":"first","passed":false}
          ]
        }"#;
        let output = bisect_reports(manifest, report).expect("valid diagnostic inputs");
        assert!(!output.all_passed);
        assert_eq!(output.compared_intermediates, 1);
        assert_eq!(output.first_divergence.expect("divergence").node_index, 3);
    }

    #[test]
    fn bisect_rejects_incomplete_backend_report() {
        let manifest = br#"{
          "schema_version": 1,
          "intermediates": [
            {"output_name":"missing","dtype":"FLOAT","node_index":0,"node_name":"","domain":"","op":"Add","tactic_family":"ai.onnx::Add"}
          ]
        }"#;
        let report = br#"{"schema_version":1,"outputs":[]}"#;
        assert!(matches!(
            bisect_reports(manifest, report),
            Err(ArtifactError::Invalid(_))
        ));
    }

    #[test]
    fn stale_replay_is_rejected_before_artifact_creation() {
        let recorded = artifact();
        let mut current = recorded.clone();
        current.records[0].implementation_digest = "44".repeat(32);
        let error = prepare_replay(&recorded, &current).expect_err("stale source must fail");
        assert!(matches!(error, ArtifactError::Incompatible(_)));
        assert!(error.to_string().contains("stale implementation digest"));

        let replay = prepare_replay(&recorded, &recorded).expect("exact replay is valid");
        assert_eq!(replay.records[0].tactic, recorded.records[0].tactic);
    }

    #[test]
    fn reduce_keeps_only_the_exact_failure_reproducer() {
        let mut source = artifact();
        source.diagnostics.push(Diagnostic {
            workload: source.records[0].workload.clone(),
            tactic_id: "broken-tile".into(),
            tactic: Some(Tactic {
                family: "blocked".into(),
                id: "broken-tile".into(),
                parameters: BTreeMap::from([("tile".into(), 7)]),
            }),
            implementation_digest: Some("33".repeat(32)),
            outcome: "numerical_failure".into(),
            detail: "NaN classification changed at output 4".into(),
        });
        let fixture = reduce_diagnostic(&source, 0).expect("diagnostic is reducible");
        assert_eq!(fixture.workload, source.records[0].workload);
        assert_eq!(fixture.tactic.expect("full tactic").parameters["tile"], 7);
        assert_eq!(fixture.failure.outcome, "numerical_failure");

        source.diagnostics[0].detail.clear();
        assert!(matches!(
            validate_artifact(&source),
            Err(ArtifactError::Invalid(_))
        ));
    }

    #[test]
    fn validated_artifact_converts_to_exact_runtime_table() {
        use onnx_vulkan_core::tuning::{
            DeviceFingerprint, ImplementationFingerprint, TensorSignature, TuningKey,
            WorkloadSignature,
        };

        let source = artifact();
        let table = runtime_table(&source).expect("valid artifact converts to runtime types");
        let key = TuningKey {
            device: DeviceFingerprint::new(
                "gpu".into(),
                1,
                2,
                3,
                4,
                [0; 16],
                32,
                vec!["a".into(), "z".into()],
            ),
            workload: WorkloadSignature {
                domain: String::new(),
                op: "Conv".into(),
                inputs: vec![TensorSignature {
                    dtype: onnx_vulkan_core::ElementType::Float32,
                    dimensions: vec![1, 3, 224, 224],
                }],
                outputs: Vec::new(),
                attributes_digest: [0x22; 32],
            },
            implementation: ImplementationFingerprint::from_bytes([0x33; 32]),
        };
        let selected = table.get(&key).expect("exact converted key exists");
        assert_eq!(selected.tactic().family(), "blocked");
        assert_eq!(selected.tactic().variant(), "tile64");
        assert_eq!(selected.parameters()["tile"], 64);
    }

    #[test]
    fn runtime_modes_preserve_fallback_and_require_an_artifact_explicitly() {
        use onnx_vulkan_core::tuning::TuningMode;

        assert_eq!(
            runtime_resolver("off", None).unwrap().mode(),
            TuningMode::Off
        );
        assert_eq!(
            runtime_resolver("cache-only", None).unwrap().mode(),
            TuningMode::CacheOnly
        );
        assert!(matches!(
            runtime_resolver("require-cache", None),
            Err(ArtifactError::Invalid(_))
        ));
        assert!(matches!(
            runtime_resolver("nearest-shape", None),
            Err(ArtifactError::Invalid(_))
        ));

        let path = temporary_path("partial-coverage");
        let mut partial = artifact();
        partial.coverage = Some(ModelCoverage {
            scope: "conv-f32-v1".into(),
            inventory_workloads: 2,
            tuned_workloads: 1,
            untunable_nodes: 1,
        });
        write_artifact(&path, &partial).expect("write partial artifact");
        assert_eq!(
            runtime_resolver("cache-only", Some(&path)).unwrap().mode(),
            TuningMode::CacheOnly
        );
        assert!(matches!(
            runtime_resolver("require-cache", Some(&path)),
            Err(ArtifactError::Incompatible(message))
                if message.contains("partial model coverage")
        ));
        fs::remove_file(path).expect("remove partial artifact");
    }
}
