use onnx_vulkan_tune::{
    ArtifactError, CreatedBy, ExecutionPlanDraft, OUTPUT_SCHEMA_VERSION, bisect_reports,
    build_execution_plan, diff, execution_plan_identity, inspect, inventory_model, merge_artifacts,
    prepare_replay, read_artifact, read_execution_plan, reduce_diagnostic, resume_artifact,
    sha256_file, validate_artifact, validate_execution_plan, validate_execution_plan_for,
    write_artifact, write_execution_plan, write_reduced_fixture,
};
use serde::Serialize;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const USAGE: &str = "\
usage:
  onnx-vulkan-tune build --input DRAFT --output ARTIFACT [--model MODEL] [--resume]
  onnx-vulkan-tune inspect ARTIFACT [--json]
  onnx-vulkan-tune validate ARTIFACT
  onnx-vulkan-tune diff LEFT RIGHT
  onnx-vulkan-tune bisect --manifest MANIFEST --report REPORT
  onnx-vulkan-tune replay --recorded ARTIFACT --target CURRENT --output REPLAY
  onnx-vulkan-tune reduce ARTIFACT --diagnostic INDEX --output FIXTURE
  onnx-vulkan-tune merge --output ARTIFACT INPUT...
  onnx-vulkan-tune model-inventory --model MODEL [--dim SYMBOL=N]...
  onnx-vulkan-tune plan-build --tuning ARTIFACT --model MODEL --inventory JSON --output PLAN
  onnx-vulkan-tune plan-inspect PLAN
  onnx-vulkan-tune plan-validate PLAN --tuning ARTIFACT --model MODEL
";

#[derive(Debug)]
enum AppError {
    Usage(String),
    Artifact(ArtifactError),
}

impl AppError {
    fn exit_code(&self) -> i32 {
        match self {
            Self::Usage(_) => 2,
            Self::Artifact(error) => error.exit_code(),
        }
    }
}

impl From<ArtifactError> for AppError {
    fn from(error: ArtifactError) -> Self {
        Self::Artifact(error)
    }
}

fn main() {
    if let Err(error) = run(env::args().collect()) {
        match &error {
            AppError::Usage(message) => eprintln!("onnx-vulkan-tune: {message}\n\n{USAGE}"),
            AppError::Artifact(source) => eprintln!("onnx-vulkan-tune: {source}"),
        }
        std::process::exit(error.exit_code());
    }
}

fn run(args: Vec<String>) -> Result<(), AppError> {
    let Some(command) = args.get(1).map(String::as_str) else {
        return Err(AppError::Usage("missing subcommand".into()));
    };
    match command {
        "build" => build(&args),
        "inspect" => inspect_command(&args),
        "validate" => validate_command(&args),
        "diff" => diff_command(&args),
        "bisect" => bisect_command(&args),
        "replay" => replay_command(&args),
        "reduce" => reduce_command(&args),
        "merge" => merge_command(&args),
        "model-inventory" => model_inventory_command(&args),
        "plan-build" => plan_build_command(&args),
        "plan-inspect" => plan_inspect_command(&args),
        "plan-validate" => plan_validate_command(&args),
        "-h" | "--help" | "help" => {
            print!("{USAGE}");
            Ok(())
        }
        other => Err(AppError::Usage(format!("unknown subcommand `{other}`"))),
    }
}

fn model_inventory_command(args: &[String]) -> Result<(), AppError> {
    let mut model = None;
    let mut symbols = std::collections::BTreeMap::new();
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--model" => model = Some(option_path(args, &mut index, "--model")?),
            "--dim" => {
                let raw = args
                    .get(index + 1)
                    .ok_or_else(|| AppError::Usage("--dim requires SYMBOL=N".into()))?;
                let (name, value) = raw
                    .split_once('=')
                    .ok_or_else(|| AppError::Usage("--dim requires SYMBOL=N".into()))?;
                if name.is_empty() {
                    return Err(AppError::Usage("--dim symbol is empty".into()));
                }
                let value = value
                    .parse::<u64>()
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| {
                        AppError::Usage(format!("--dim {name} requires a positive integer"))
                    })?;
                if symbols.insert(name.to_owned(), value).is_some() {
                    return Err(AppError::Usage(format!("duplicate --dim {name}")));
                }
                index += 2;
            }
            other => {
                return Err(AppError::Usage(format!(
                    "unknown model-inventory option `{other}`"
                )));
            }
        }
    }
    let model = required_path(model, "model-inventory requires --model")?;
    json_stdout(&inventory_model(&model, symbols)?)
}

fn plan_build_command(args: &[String]) -> Result<(), AppError> {
    let mut tuning = None;
    let mut model = None;
    let mut inventory = None;
    let mut output = None;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--tuning" => tuning = Some(option_path(args, &mut index, "--tuning")?),
            "--model" => model = Some(option_path(args, &mut index, "--model")?),
            "--inventory" => inventory = Some(option_path(args, &mut index, "--inventory")?),
            "--output" => output = Some(option_path(args, &mut index, "--output")?),
            other => {
                return Err(AppError::Usage(format!(
                    "unknown plan-build option `{other}`"
                )));
            }
        }
    }
    let tuning = read_artifact(&required_path(tuning, "plan-build requires --tuning")?)?;
    let model = required_path(model, "plan-build requires --model")?;
    let inventory = required_path(inventory, "plan-build requires --inventory")?;
    let output = required_path(output, "plan-build requires --output")?;
    let model = onnx_vulkan_frontend::load(&model).map_err(|error| {
        ArtifactError::Invalid(format!("load model {}: {error}", model.display()))
    })?;
    let bytes = std::fs::read(&inventory).map_err(|error| {
        ArtifactError::Io(format!("read inventory {}: {error}", inventory.display()))
    })?;
    let draft: ExecutionPlanDraft = serde_json::from_slice(&bytes).map_err(|error| {
        ArtifactError::Json(format!("inventory {}: {error}", inventory.display()))
    })?;
    let plan = build_execution_plan(&tuning, &model, draft, provenance(args)?)?;
    write_execution_plan(&output, &plan)?;
    json_stdout(&serde_json::json!({
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "plan-built",
        "path": output,
        "profiles": plan.profiles.len(),
        "resolved_tactics": plan.resolved_tactics.len(),
        "packed_weights": plan.packed_weights.len(),
        "steps": plan.steps.len(),
    }))
}

fn plan_inspect_command(args: &[String]) -> Result<(), AppError> {
    if args.len() != 3 {
        return Err(AppError::Usage("plan-inspect wants PLAN".into()));
    }
    let plan = read_execution_plan(Path::new(&args[2]))?;
    json_stdout(&serde_json::json!({
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "schema_version": plan.schema_version,
        "device": plan.device.name,
        "graph_digest": plan.graph_digest,
        "implementations": plan.implementation_digests,
        "profiles": plan.profiles,
        "resolved_tactics": plan.resolved_tactics,
        "packed_weights": plan.packed_weights,
        "steps": plan.steps,
    }))
}

fn plan_validate_command(args: &[String]) -> Result<(), AppError> {
    if args.len() != 7 || args[3] != "--tuning" || args[5] != "--model" {
        return Err(AppError::Usage(
            "plan-validate wants PLAN --tuning ARTIFACT --model MODEL".into(),
        ));
    }
    let plan = read_execution_plan(Path::new(&args[2]))?;
    validate_execution_plan(&plan)?;
    let tuning = read_artifact(Path::new(&args[4]))?;
    let model = Path::new(&args[6]);
    let graph = onnx_vulkan_frontend::load(model)
        .map_err(|error| {
            ArtifactError::Invalid(format!("load model {}: {error}", model.display()))
        })?
        .graph;
    let identity = execution_plan_identity(&tuning, &graph)?;
    validate_execution_plan_for(
        &plan,
        &identity.device,
        &identity.graph_digest,
        &identity.implementation_digests,
    )?;
    json_stdout(&serde_json::json!({
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "plan-valid",
        "profiles": plan.profiles.len(),
        "resolved_tactics": plan.resolved_tactics.len(),
    }))
}

fn build(args: &[String]) -> Result<(), AppError> {
    let mut input = None;
    let mut output = None;
    let mut model = None;
    let mut resume = false;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--input" => input = Some(option_path(args, &mut index, "--input")?),
            "--output" => output = Some(option_path(args, &mut index, "--output")?),
            "--model" => model = Some(option_path(args, &mut index, "--model")?),
            "--resume" => {
                resume = true;
                index += 1;
            }
            other => return Err(AppError::Usage(format!("unknown build option `{other}`"))),
        }
    }
    let input = input.ok_or_else(|| AppError::Usage("build requires --input".into()))?;
    let output = output.ok_or_else(|| AppError::Usage("build requires --output".into()))?;
    let mut artifact = read_draft(&input)?;
    artifact.created_by = provenance(args)?;
    if let Some(model) = model {
        artifact.profile.model_digest = sha256_file(&model)?;
    }
    artifact = if resume {
        resume_artifact(&output, artifact)?
    } else {
        artifact
    };
    write_artifact(&output, &artifact)?;
    json_stdout(&serde_json::json!({
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "built",
        "path": output,
        "records": artifact.records.len(),
        "resumed": resume,
    }))
}

fn read_draft(path: &Path) -> Result<onnx_vulkan_tune::Artifact, AppError> {
    let bytes = std::fs::read(path)
        .map_err(|error| ArtifactError::Io(format!("read draft {}: {error}", path.display())))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| ArtifactError::Json(format!("{}: {error}", path.display())).into())
}

fn inspect_command(args: &[String]) -> Result<(), AppError> {
    if !(args.len() == 3 || (args.len() == 4 && args[3] == "--json")) {
        return Err(AppError::Usage("inspect wants ARTIFACT [--json]".into()));
    }
    let artifact = read_artifact(Path::new(&args[2]))?;
    let output = inspect(&artifact)?;
    if args.len() == 4 {
        json_stdout(&output)
    } else {
        println!(
            "schema {} · {} record(s) · {} diagnostic(s)\ndevice: {}\nmodel: {}",
            output.schema_version,
            output.record_count,
            output.diagnostic_count,
            output.device_name,
            output.model_digest
        );
        for (operator, count) in output.operators {
            println!("  {operator}: {count}");
        }
        if let Some(coverage) = output.coverage {
            println!(
                "coverage: {} tuned {}/{} · untunable {} · strict {}",
                coverage.scope,
                coverage.tuned_workloads,
                coverage.inventory_workloads,
                coverage.untunable_nodes,
                coverage.strict_compatible()
            );
        }
        Ok(())
    }
}

fn validate_command(args: &[String]) -> Result<(), AppError> {
    if args.len() != 3 {
        return Err(AppError::Usage("validate wants ARTIFACT".into()));
    }
    let artifact = read_artifact(Path::new(&args[2]))?;
    validate_artifact(&artifact)?;
    json_stdout(&serde_json::json!({
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "valid",
        "schema_version": artifact.schema_version,
        "records": artifact.records.len(),
    }))
}

fn diff_command(args: &[String]) -> Result<(), AppError> {
    if args.len() != 4 {
        return Err(AppError::Usage("diff wants LEFT RIGHT".into()));
    }
    let left = read_artifact(Path::new(&args[2]))?;
    let right = read_artifact(Path::new(&args[3]))?;
    json_stdout(&diff(&left, &right)?)
}

fn bisect_command(args: &[String]) -> Result<(), AppError> {
    if args.len() != 6 || args[2] != "--manifest" || args[4] != "--report" {
        return Err(AppError::Usage(
            "bisect wants --manifest MANIFEST --report REPORT".into(),
        ));
    }
    let read = |label: &str, path: &str| {
        std::fs::read(path)
            .map_err(|error| ArtifactError::Io(format!("read {label} {path}: {error}")))
    };
    let manifest = read("manifest", &args[3])?;
    let report = read("report", &args[5])?;
    json_stdout(&bisect_reports(&manifest, &report)?)
}

fn replay_command(args: &[String]) -> Result<(), AppError> {
    if args.len() != 8 || args[2] != "--recorded" || args[4] != "--target" || args[6] != "--output"
    {
        return Err(AppError::Usage(
            "replay wants --recorded ARTIFACT --target CURRENT --output REPLAY".into(),
        ));
    }
    let recorded = read_artifact(Path::new(&args[3]))?;
    let target = read_artifact(Path::new(&args[5]))?;
    let replay = prepare_replay(&recorded, &target)?;
    let output = Path::new(&args[7]);
    write_artifact(output, &replay)?;
    json_stdout(&serde_json::json!({
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "replay-ready",
        "path": output,
        "records": replay.records.len(),
        "runtime_mode": "require-cache",
    }))
}

fn reduce_command(args: &[String]) -> Result<(), AppError> {
    if args.len() != 7 || args[3] != "--diagnostic" || args[5] != "--output" {
        return Err(AppError::Usage(
            "reduce wants ARTIFACT --diagnostic INDEX --output FIXTURE".into(),
        ));
    }
    let index = args[4]
        .parse::<usize>()
        .map_err(|_| AppError::Usage("--diagnostic must be a non-negative integer".into()))?;
    let artifact = read_artifact(Path::new(&args[2]))?;
    let fixture = reduce_diagnostic(&artifact, index)?;
    let output = Path::new(&args[6]);
    write_reduced_fixture(output, &fixture)?;
    json_stdout(&serde_json::json!({
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "reduced",
        "path": output,
        "diagnostic": index,
        "workload": fixture.workload,
        "tactic_id": fixture.tactic_id,
        "outcome": fixture.failure.outcome,
    }))
}

fn merge_command(args: &[String]) -> Result<(), AppError> {
    if args.len() < 5 || args[2] != "--output" {
        return Err(AppError::Usage(
            "merge wants --output ARTIFACT INPUT...".into(),
        ));
    }
    let output = PathBuf::from(&args[3]);
    let artifacts = args[4..]
        .iter()
        .map(|path| read_artifact(Path::new(path)))
        .collect::<Result<Vec<_>, _>>()?;
    let merged = merge_artifacts(&artifacts)?;
    write_artifact(&output, &merged)?;
    json_stdout(&serde_json::json!({
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "merged",
        "path": output,
        "inputs": artifacts.len(),
        "records": merged.records.len(),
    }))
}

fn option_path(args: &[String], index: &mut usize, name: &str) -> Result<PathBuf, AppError> {
    let value = args
        .get(*index + 1)
        .ok_or_else(|| AppError::Usage(format!("{name} requires a path")))?;
    *index += 2;
    Ok(PathBuf::from(value))
}

fn required_path(value: Option<PathBuf>, message: &str) -> Result<PathBuf, AppError> {
    value.ok_or_else(|| AppError::Usage(message.into()))
}

fn provenance(args: &[String]) -> Result<CreatedBy, AppError> {
    let created_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| ArtifactError::Io(format!("system clock predates Unix epoch: {error}")))?
        .as_secs();
    Ok(CreatedBy {
        onnx_vulkan_version: env!("CARGO_PKG_VERSION").into(),
        git_commit: git_output(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into()),
        dirty: git_dirty(),
        command_line: args.to_vec(),
        created_unix_seconds,
    })
}

fn git_output(arguments: &[&str]) -> Option<String> {
    let output = Command::new("git").args(arguments).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    Some(value.trim().to_owned())
}

fn git_dirty() -> bool {
    Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map_or(true, |output| {
            !output.status.success() || !output.stdout.is_empty()
        })
}

fn json_stdout(value: &impl Serialize) -> Result<(), AppError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ArtifactError::Json(format!("serialize command output: {error}")))?;
    println!("{}", String::from_utf8_lossy(&bytes));
    Ok(())
}
