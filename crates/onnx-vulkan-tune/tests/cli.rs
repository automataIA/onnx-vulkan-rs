use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_onnx-vulkan-tune")
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn frontend_model(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../onnx-vulkan-frontend/tests/models")
        .join(name)
}

fn run(arguments: &[&str]) -> Output {
    Command::new(binary())
        .args(arguments)
        .output()
        .expect("run onnx-vulkan-tune")
}

fn temporary_path(name: &str) -> PathBuf {
    let serial = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("test clock is after Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("onnx-vulkan-tune-cli-{name}-{serial}.json"))
}

#[test]
fn inspect_and_diff_are_gpu_free_json_contracts() {
    let valid = fixture("valid.json");
    let inspect = run(&[
        "inspect",
        valid.to_str().expect("UTF-8 fixture path"),
        "--json",
    ]);
    assert!(
        inspect.status.success(),
        "{}",
        String::from_utf8_lossy(&inspect.stderr)
    );
    let value: Value = serde_json::from_slice(&inspect.stdout).expect("inspect emits JSON");
    assert_eq!(value["output_schema_version"], 1);
    assert_eq!(value["record_count"], 1);

    let diff = run(&[
        "diff",
        valid.to_str().expect("UTF-8 fixture path"),
        valid.to_str().expect("UTF-8 fixture path"),
    ]);
    assert!(
        diff.status.success(),
        "{}",
        String::from_utf8_lossy(&diff.stderr)
    );
    let value: Value = serde_json::from_slice(&diff.stdout).expect("diff emits JSON");
    assert_eq!(value["compatible"], true);
    assert_eq!(value["added"].as_array().expect("added array").len(), 0);
}

#[test]
fn model_inventory_is_gpu_free_exact_and_reports_fallbacks() {
    let model = frontend_model("shapes.onnx");
    let output = run(&[
        "model-inventory",
        "--model",
        model.to_str().expect("UTF-8 model path"),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("inventory JSON");
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["workloads"].as_array().expect("workloads").len(), 1);
    assert_eq!(value["workloads"][0]["workload"]["op"], "Conv");
    assert_eq!(value["workloads"][0]["runner"]["family"], "conv-f32");
    assert_eq!(value["workloads"][0]["runner"]["h_in"], 32);
    assert!(
        value["skipped"]
            .as_array()
            .expect("skips")
            .iter()
            .any(|item| item["op"] == "Relu" && item["reason"] == "no model-level family runner")
    );

    let malformed = run(&[
        "model-inventory",
        "--model",
        model.to_str().expect("UTF-8 model path"),
        "--dim",
        "batch=0",
    ]);
    assert_eq!(malformed.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&malformed.stderr).contains("positive integer"));
}

#[test]
fn diff_reports_incompatibility_instead_of_hiding_it() {
    let output = run(&[
        "diff",
        fixture("valid.json").to_str().expect("UTF-8 fixture path"),
        fixture("incompatible.json")
            .to_str()
            .expect("UTF-8 fixture path"),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("diff emits JSON");
    assert_eq!(value["compatible"], false);
    assert_eq!(value["metadata"]["device_changed"], true);
    assert_eq!(value["metadata"]["policy_changed"], false);
}

#[test]
fn bisect_cli_reports_first_divergent_family() {
    let manifest = temporary_path("manifest");
    let report = temporary_path("report");
    fs::write(
        &manifest,
        r#"{"schema_version":1,"intermediates":[{"output_name":"x","dtype":"FLOAT","node_index":7,"node_name":"conv","domain":"","op":"Conv","tactic_family":"ai.onnx::Conv"}]}"#,
    )
    .expect("write manifest");
    fs::write(
        &report,
        r#"{"schema_version":1,"outputs":[{"name":"x","passed":false}]}"#,
    )
    .expect("write report");
    let output = run(&[
        "bisect",
        "--manifest",
        manifest.to_str().expect("UTF-8 manifest path"),
        "--report",
        report.to_str().expect("UTF-8 report path"),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("bisect emits JSON");
    assert_eq!(value["first_divergence"]["node_index"], 7);
    assert_eq!(value["first_divergence"]["tactic_family"], "ai.onnx::Conv");
    fs::remove_file(manifest).expect("remove manifest");
    fs::remove_file(report).expect("remove report");
}

#[test]
fn replay_cli_writes_only_after_compatibility_validation() {
    let valid = fixture("valid.json");
    let replay = temporary_path("replay");
    let output = run(&[
        "replay",
        "--recorded",
        valid.to_str().expect("UTF-8 fixture path"),
        "--target",
        valid.to_str().expect("UTF-8 fixture path"),
        "--output",
        replay.to_str().expect("UTF-8 replay path"),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let replay_value: Value =
        serde_json::from_slice(&fs::read(&replay).expect("read replay")).expect("parse replay");
    let valid_value: Value = serde_json::from_slice(&fs::read(&valid).expect("read valid fixture"))
        .expect("parse valid fixture");
    assert_eq!(replay_value, valid_value);
    fs::remove_file(replay).expect("remove replay");

    let rejected = temporary_path("replay-rejected");
    let output = run(&[
        "replay",
        "--recorded",
        valid.to_str().expect("UTF-8 fixture path"),
        "--target",
        fixture("incompatible.json")
            .to_str()
            .expect("UTF-8 fixture path"),
        "--output",
        rejected.to_str().expect("UTF-8 rejected path"),
    ]);
    assert_eq!(output.status.code(), Some(5));
    assert!(!rejected.exists());
}

#[test]
fn reduce_cli_emits_minimal_failure_fixture() {
    let source = temporary_path("reduce-source");
    let output = temporary_path("reduce-output");
    let mut artifact: Value =
        serde_json::from_slice(&fs::read(fixture("valid.json")).expect("read fixture"))
            .expect("parse fixture");
    let workload = artifact["records"][0]["workload"].clone();
    artifact["diagnostics"] = serde_json::json!([{
        "workload": workload,
        "tactic_id": "broken",
        "tactic": {"family":"blocked","id":"broken","parameters":{"tile":7}},
        "implementation_digest": "33".repeat(32),
        "outcome": "validation_failure",
        "detail": "mismatch at element 4"
    }]);
    fs::write(
        &source,
        serde_json::to_vec_pretty(&artifact).expect("serialize source"),
    )
    .expect("write source");
    let command = run(&[
        "reduce",
        source.to_str().expect("UTF-8 source path"),
        "--diagnostic",
        "0",
        "--output",
        output.to_str().expect("UTF-8 output path"),
    ]);
    assert!(
        command.status.success(),
        "{}",
        String::from_utf8_lossy(&command.stderr)
    );
    let reduced: Value = serde_json::from_slice(&fs::read(&output).expect("read reduced fixture"))
        .expect("parse reduced fixture");
    assert_eq!(reduced["schema_version"], 1);
    assert_eq!(reduced["tactic"]["parameters"]["tile"], 7);
    assert_eq!(reduced["failure"]["outcome"], "validation_failure");
    assert!(reduced.get("records").is_none());
    fs::remove_file(source).expect("remove source");
    fs::remove_file(output).expect("remove output");
}

#[test]
fn hostile_fixtures_have_stable_exit_classes() {
    for name in ["malformed.json", "old.json", "newer.json"] {
        let output = run(&[
            "validate",
            fixture(name).to_str().expect("UTF-8 fixture path"),
        ]);
        assert_eq!(
            output.status.code(),
            Some(3),
            "fixture {name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!output.stderr.is_empty());
        if name == "old.json" {
            assert!(String::from_utf8_lossy(&output.stderr).contains("obsolete"));
        }
        if name == "newer.json" {
            assert!(String::from_utf8_lossy(&output.stderr).contains("newer"));
        }
    }
    let duplicate = run(&[
        "validate",
        fixture("duplicate.json")
            .to_str()
            .expect("UTF-8 fixture path"),
    ]);
    assert_eq!(duplicate.status.code(), Some(5));
    let output = run(&[
        "merge",
        "--output",
        temporary_path("incompatible")
            .to_str()
            .expect("UTF-8 output path"),
        fixture("valid.json").to_str().expect("UTF-8 fixture path"),
        fixture("incompatible.json")
            .to_str()
            .expect("UTF-8 fixture path"),
    ]);
    assert_eq!(output.status.code(), Some(5));
}

#[test]
fn build_is_canonical_and_resume_preserves_records() {
    let draft = fixture("valid.json");
    let output = temporary_path("build");
    let first = run(&[
        "build",
        "--input",
        draft.to_str().expect("UTF-8 fixture path"),
        "--output",
        output.to_str().expect("UTF-8 output path"),
    ]);
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let first_bytes = fs::read(&output).expect("read built artifact");

    let validate = run(&["validate", output.to_str().expect("UTF-8 output path")]);
    assert!(validate.status.success());

    let second = run(&[
        "build",
        "--input",
        draft.to_str().expect("UTF-8 fixture path"),
        "--output",
        output.to_str().expect("UTF-8 output path"),
        "--resume",
    ]);
    assert!(
        second.status.success(),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );
    let second_bytes = fs::read(&output).expect("read resumed artifact");
    assert_ne!(first_bytes, second_bytes, "resume refreshes provenance");
    let value: Value = serde_json::from_slice(&second_bytes).expect("resumed artifact parses");
    assert_eq!(value["records"].as_array().expect("records array").len(), 1);
    fs::remove_file(output).expect("remove built artifact");
}

#[test]
fn execution_plan_cli_derives_identity_and_rejects_a_different_graph() {
    let inventory = temporary_path("plan-inventory");
    let plan = temporary_path("execution-plan");
    fs::write(
        &inventory,
        serde_json::to_vec_pretty(&serde_json::json!({
            "profiles": [{
                "id": "fixed",
                "inputs": {"x": {"dtype": 1, "dims": [2, 2]}}
            }],
            "packed_weights": [],
            "steps": []
        }))
        .expect("serialize inventory"),
    )
    .expect("write inventory");
    let tuning = fixture("valid.json");
    let model = frontend_model("mixed.onnx");
    let build = run(&[
        "plan-build",
        "--tuning",
        tuning.to_str().expect("UTF-8 tuning path"),
        "--model",
        model.to_str().expect("UTF-8 model path"),
        "--inventory",
        inventory.to_str().expect("UTF-8 inventory path"),
        "--output",
        plan.to_str().expect("UTF-8 plan path"),
    ]);
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let built: Value =
        serde_json::from_slice(&fs::read(&plan).expect("read plan")).expect("parse plan");
    assert_eq!(
        built["profiles"][0]["inputs"]["x"]["dims"],
        serde_json::json!([2, 2])
    );
    assert_eq!(built["resolved_tactics"].as_array().unwrap().len(), 1);
    assert!(built.get("nodes").is_none());

    let validate = run(&[
        "plan-validate",
        plan.to_str().expect("UTF-8 plan path"),
        "--tuning",
        tuning.to_str().expect("UTF-8 tuning path"),
        "--model",
        model.to_str().expect("UTF-8 model path"),
    ]);
    assert!(
        validate.status.success(),
        "{}",
        String::from_utf8_lossy(&validate.stderr)
    );

    let stale = run(&[
        "plan-validate",
        plan.to_str().expect("UTF-8 plan path"),
        "--tuning",
        tuning.to_str().expect("UTF-8 tuning path"),
        "--model",
        frontend_model("shapes.onnx")
            .to_str()
            .expect("UTF-8 stale model path"),
    ]);
    assert_eq!(stale.status.code(), Some(5));

    fs::write(
        &inventory,
        r#"{"profiles":[{"id":"wrong","inputs":{"x":{"dtype":1,"dims":[1,2]}}}]}"#,
    )
    .expect("write mismatched inventory");
    let rejected_plan = temporary_path("rejected-plan");
    let mismatched = run(&[
        "plan-build",
        "--tuning",
        tuning.to_str().expect("UTF-8 tuning path"),
        "--model",
        model.to_str().expect("UTF-8 model path"),
        "--inventory",
        inventory.to_str().expect("UTF-8 inventory path"),
        "--output",
        rejected_plan.to_str().expect("UTF-8 rejected plan path"),
    ]);
    assert_eq!(mismatched.status.code(), Some(3));
    assert!(!rejected_plan.exists());
    fs::remove_file(inventory).expect("remove inventory");
    fs::remove_file(plan).expect("remove plan");
}
