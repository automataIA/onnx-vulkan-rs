import json
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import onnx_vulkan_pipeline.pipeline as pipeline
from onnx_vulkan_pipeline import policy
from onnx_vulkan_pipeline.pipeline import seal, verify_bundle, _improvement, dimensions
from onnx_vulkan_pipeline.process import command, gpu_lock


@pytest.fixture(autouse=True)
def host_independent_policy(monkeypatch):
    """The execution policy reads the caller's environment; a host's own device
    or probe settings must not change what these tests expect."""
    for name in ("ONNX_VULKAN_ATTENTION_FP16", "VULKAN_POOL_BUDGET_MB", "VULKAN_EP_OPS", "VULKAN_EP_COMPILE",
                 "VULKAN_EP_STATS", "ONNX_VULKAN_TUNING_MODE", "ONNX_VULKAN_TUNING_ARTIFACT",
                 *policy.PROBES, *policy.DEVICE_SELECTION):
        monkeypatch.delenv(name, raising=False)


def test_tuning_priority_uses_measured_family_time():
    report = {"gpu": {"pareto": [
        {"op": "Conv16", "ms": 1.0}, {"op": "Conv_grouped", "ms": 2.0},
        {"op": "MatMul16", "ms": 4.0}, {"op": "Gemm", "ms": 5.0},
        {"op": "MatMulNBits", "ms": 100.0},
    ]}}
    assert pipeline._tuning_priority(report) == {"Conv": 3.0, "MatMul": 4.0, "Gemm": 5.0}


def test_external_weight_mutation_invalidates_bundle(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"model")
    (tmp_path / "weights.bin").write_bytes(b"weights")
    seal(tmp_path, {"schema_version": 1})
    verify_bundle(tmp_path)
    (tmp_path / "weights.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weights.bin"):
        verify_bundle(tmp_path)


def test_bundle_rejects_path_escape(tmp_path):
    seal(tmp_path, {"schema_version": 1})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"]["../outside"] = "irrelevant"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Stale"):
        verify_bundle(tmp_path)


def test_bundle_rejects_unsealed_extra_file(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"model")
    seal(tmp_path, {"schema_version": 1})
    (tmp_path / "unexpected.bin").write_bytes(b"new")
    with pytest.raises(ValueError, match="unsealed"):
        verify_bundle(tmp_path)


@pytest.mark.parametrize("field", ["model", "tactics"])
def test_bundle_rejects_unsealed_profile_artifact(tmp_path, field):
    model = tmp_path / "default" / "model.onnx"
    model.parent.mkdir()
    model.write_bytes(b"model")
    seal(tmp_path, {"schema_version": 1, "profiles": {"default": {
        "model": "default/model.onnx", "tactics": None,
    }}})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["profiles"]["default"][field] = f"default/unsealed-{field}.json"
    manifest["identity"] = pipeline._manifest_identity(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=f"Unsealed profile {field}"):
        verify_bundle(tmp_path)


def test_nested_manifest_mutation_invalidates_bundle(tmp_path):
    nested = tmp_path / "default" / "manifest.json"
    nested.parent.mkdir()
    nested.write_text('{"version": 1}\n')
    seal(tmp_path, {"schema_version": 1})
    nested.write_text('{"version": 2}\n')
    with pytest.raises(ValueError, match="default/manifest.json"):
        verify_bundle(tmp_path)


def test_no_promotion_without_gpu_and_noise_evidence():
    reference = [{"gpu_ms": 10, "wall_ms": 12}] * 20
    assert _improvement(reference, [{"gpu_ms": 7, "wall_ms": 8}] * 20)
    assert not _improvement(reference, [{"gpu_ms": None, "wall_ms": 8}] * 20)
    assert not _improvement(reference, [{"gpu_ms": 9.5, "wall_ms": 11.5}] * 20)
    assert not _improvement(reference[:19], reference[:19])
    assert not _improvement(reference, [{"gpu_ms": float("nan"), "wall_ms": 8}] * 20)


def _evidence(total=100, supported=100, claimed=12, **overrides):
    """What one profiled run of a graph reports, as `_diagnostic_evidence` builds it."""
    evidence = {"graph": {"model": "default/model.onnx", "digest": "d", "compatible": True, "total_nodes": total,
                          "supported_nodes": supported, "unsupported_nodes": total - supported},
                "tactics": None, "passed": True, "cpu_fallback_disabled": True, "flushes": 1, "upload_mb": 4,
                "download_mb": 2, "convex_blocks": 1, "nodes_claimed": claimed}
    return {**evidence, **overrides}


def test_faster_candidate_cannot_increase_transfers_or_partitioning():
    source = _evidence()
    assert pipeline._diagnostics_hold(source, _evidence()) is None
    for key, worse in (("upload_mb", 5), ("download_mb", 3), ("flushes", 2), ("convex_blocks", 2)):
        assert pipeline._diagnostics_hold(source, _evidence(**{key: worse})) == [
            f"{key} grew from {source[key]:g} to {worse:g}"]
    # An unmeasured transfer is not "no more than before".
    assert pipeline._diagnostics_hold(source, _evidence(upload_mb=None)) == ["upload_mb not measured"]


def test_a_smaller_fully_covered_graph_is_not_penalized_for_claiming_fewer_nodes():
    source = _evidence(total=1000, supported=1000, claimed=1000)
    rewritten = _evidence(total=400, supported=400, claimed=400)
    assert pipeline._diagnostics_hold(source, rewritten) is None
    # The same size with less of it supported is a loss of coverage, however many nodes it claims.
    partial = _evidence(total=400, supported=399, claimed=399)
    assert pipeline._diagnostics_hold(source, partial) == ["candidate covers 399/400 nodes, source 1000/1000"]
    # Evidence that it ran verified without the CPU fallback is required of both graphs.
    for missing in ({"cpu_fallback_disabled": False}, {"passed": False}, {"nodes_claimed": None},
                    {"nodes_claimed": 0}):
        assert pipeline._diagnostics_hold(source, _evidence(**missing)) == [
            "candidate graph has no verified Vulkan coverage evidence"]
        assert pipeline._diagnostics_hold(_evidence(**missing), source) == [
            "source graph has no verified Vulkan coverage evidence"]
    inconsistent = _evidence(total=400, supported=400)
    inconsistent["graph"]["unsupported_nodes"] = 1
    assert pipeline._diagnostics_hold(source, inconsistent)
    incompatible = _evidence()
    incompatible["graph"]["compatible"] = False
    assert pipeline._diagnostics_hold(source, incompatible)


def test_the_two_sides_are_inspected_on_their_own_graphs(tmp_path):
    (tmp_path / "default").mkdir()
    (tmp_path / "default" / "original.onnx").write_bytes(b"large")
    (tmp_path / "default" / "model.onnx").write_bytes(b"small")
    inspections = {"default/original.onnx": {"compatible": True, "execution": {
                       "total_nodes": 10, "supported_nodes": 10, "unsupported_nodes": 0}},
                   "default/model.onnx": {"compatible": True, "execution": {
                       "total_nodes": 4, "supported_nodes": 4, "unsupported_nodes": 0}}}
    source, candidate = (pipeline._graph_evidence(tmp_path, model, inspection)
                         for model, inspection in inspections.items())
    assert (source["model"], source["total_nodes"]) == ("default/original.onnx", 10)
    assert (candidate["model"], candidate["total_nodes"]) == ("default/model.onnx", 4)
    assert source["digest"] != candidate["digest"]
    # A missing inspection leaves the counts unknown, which `_coverage` refuses.
    unknown = pipeline._graph_evidence(tmp_path, "default/model.onnx", {"compatible": True})
    assert unknown["total_nodes"] is None
    assert pipeline._coverage({**_evidence(), "graph": unknown}) is None


def test_process_timeout_and_failure_are_not_success():
    with pytest.raises(subprocess.TimeoutExpired):
        command([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.05)
    with pytest.raises(RuntimeError, match="exited 3"):
        command([sys.executable, "-c", "raise SystemExit(3)"])
    # A long output is cut at a line boundary, not in the middle of a line.
    with pytest.raises(RuntimeError) as failure:
        command([sys.executable, "-c", "import sys; sys.stderr.write('x' * 9000 + '\\nError: cause\\n'); sys.exit(1)"])
    assert ": …Error: cause\n" in str(failure.value)
    # A signal from outside (earlyoom sends SIGTERM) is named as the cause.
    with pytest.raises(RuntimeError) as failure:
        command([sys.executable, "-c", "import os, signal; print('Error: not this'); os.kill(os.getpid(), signal.SIGTERM)"])
    assert pipeline.failure_reason(failure.value).startswith(
        f"Error: {Path(sys.executable).name} was killed by SIGTERM from outside onnx-vulkan")


def test_failed_run_preserves_report_outside_immutable_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    (bundle / "default").mkdir(parents=True)
    pipeline.write_json(bundle / "default" / "preparation.json", {})
    _dynamic_model(bundle / "default" / "model.onnx", quantized=False)
    manifest = {"identity": "fixture", "atol": 1e-5, "rtol": 1e-4, "profiles": {
        "default": {"model": "default/model.onnx", "dimensions": {}, "tactics": None},
    }}
    monkeypatch.setattr(pipeline, "binary", lambda _: Path("model-runner"))

    def fail(args, **kwargs):
        Path(args[args.index("--report-json") + 1]).write_text('{"passed": false}')
        raise RuntimeError("GPU fixture failed")

    monkeypatch.setattr(pipeline, "command", fail)
    with pytest.raises(RuntimeError, match="GPU fixture failed"):
        pipeline._run(bundle, manifest, "default", "standalone", 1, 1)
    evidence = tmp_path / ".bundle.run-failed"
    assert json.loads((evidence / "report.json").read_text())["passed"] is False
    assert (evidence / "error.log").read_text() == "GPU fixture failed"
    assert json.loads((evidence / "failure.json").read_text())["reason"] == "GPU fixture failed"
    assert json.loads((evidence / "failure.json").read_text())["identity"] == "fixture"
    failure = json.loads((evidence / "failure.json").read_text())
    assert failure["execution_policy"]["measurement"]["stats"] is True
    assert failure["environment"]["VULKAN_EP_STATS"] == "1"
    assert sorted(path.name for path in bundle.rglob("*")) == ["default", "model.onnx", "preparation.json"]
    # A second failure replaces the first instead of adding a directory.
    with pytest.raises(RuntimeError) as second:
        pipeline._run(bundle, manifest, "default", "standalone", 1, 1)
    assert [path.name for path in tmp_path.glob(".bundle.run-failed*")] == [".bundle.run-failed"]
    assert second.value.__notes__ == [f"Run evidence retained at {evidence}"]


def test_gpu_lane_is_exclusive_and_released(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    with gpu_lock(1):
        with pytest.raises(TimeoutError):
            with gpu_lock(0.01):
                pass
    with gpu_lock(0.1):
        pass


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_profile_fails_before_execution(value):
    with pytest.raises(ValueError):
        dimensions({"batch": value})


def _tuned_bundle(tmp_path, *, tools, device):
    bundle = tmp_path / "tuned"
    profile = bundle / "default"
    profile.mkdir(parents=True)
    (profile / "model.onnx").write_bytes(b"model")
    (profile / "tactics.json").write_text("{}\n")
    manifest = {
        "schema_version": 1,
        "tools": tools,
        "device": device,
        "atol": 1e-5,
        "rtol": 1e-4,
        "profiles": {
            "default": {
                "dimensions": {},
                "model": "default/model.onnx",
                "compatible": True,
                "tactics": "default/tactics.json",
            }
        },
    }
    seal(bundle, manifest)
    return bundle


@pytest.mark.parametrize(
    ("manifest_tools", "manifest_device", "current_tools", "current_device"),
    [
        ({"tool": "old"}, {"gpu": "same"}, {"tool": "new"}, {"gpu": "same"}),
        ({"tool": "same"}, {"gpu": "old"}, {"tool": "same"}, {"gpu": "new"}),
    ],
)
def test_run_bundle_rejects_stale_tactics_environment(
    tmp_path, monkeypatch, manifest_tools, manifest_device, current_tools, current_device
):
    bundle = _tuned_bundle(tmp_path, tools=manifest_tools, device=manifest_device)

    @contextmanager
    def no_gpu_lock(_timeout):
        yield

    monkeypatch.setattr(pipeline, "gpu_lock", no_gpu_lock)
    monkeypatch.setattr(pipeline, "tool_identity", lambda: current_tools)
    monkeypatch.setattr(pipeline, "binary", lambda name: Path(name))
    monkeypatch.setattr(
        pipeline,
        "command",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(current_device)),
    )
    monkeypatch.setattr(pipeline, "_run", lambda *args, **kwargs: pytest.fail("stale bundle ran"))

    with pytest.raises(ValueError, match="different tools or device"):
        pipeline.run_bundle(bundle, backend="ep", samples=1)


def _source_bundle(tmp_path, *, reference=False):
    source = tmp_path / "source"
    profile = source / "default"
    profile.mkdir(parents=True)
    (profile / "model.onnx").write_bytes(b"model")
    if reference:
        (profile / "reference").mkdir()
        (profile / "reference" / "input_0.pb").write_bytes(b"input")
    manifest = {
        "schema_version": 1,
        "tools": {"tool": "test"},
        "atol": 1e-5,
        "rtol": 1e-4,
        "profiles": {
            "default": {
                "dimensions": {},
                "model": "default/model.onnx",
                "compatible": True,
                "tactics": None,
            }
        },
    }
    seal(source, manifest)
    return source, verify_bundle(source)


def test_an_incompatible_profile_is_refused_with_its_unsupported_operators(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "default").mkdir(parents=True)
    (bundle / "default" / "model.onnx").write_bytes(b"model")
    pipeline.write_json(bundle / "default" / "compatibility.json", {"compatible": False, "issues": [
        {"op": op, "kind": "execution", "blocking": True} for op in ("NonZero", "RoiAlign", "NonZero")]
        + [{"op": "Reshape", "kind": "shape", "blocking": False}]})
    seal(bundle, {"schema_version": 1, "tools": {"tool": "test"}, "atol": 1e-5, "rtol": 1e-4, "profiles": {
        "default": {"dimensions": {}, "model": "default/model.onnx", "compatible": False, "tactics": None}}})
    with pytest.raises(ValueError, match="cannot run: Vulkan does not implement these operators, or these forms "
                                         "of them: NonZero ×2, RoiAlign ×1, and there is no CPU fallback"):
        pipeline.run_bundle(bundle)


def test_tuned_kernels_that_miss_the_tolerance_on_the_reference_inputs_are_dropped(tmp_path, monkeypatch):
    source, _ = _source_bundle(tmp_path, reference=True)

    @contextmanager
    def no_gpu_lock(_timeout):
        yield

    runs, tuner, tuner_env = [], [], []

    def run(bundle, manifest, profile, backend, samples, timeout, *, tactics=None, **kwargs):
        runs.append(tactics)
        return {"standalone_ms": [1.0], "passed": True}

    def command(argv, **kwargs):
        if any(str(arg).endswith("scripts/autotune/model.py") for arg in argv):
            tuner.append([str(arg) for arg in argv])
            tuner_env.append(kwargs["env"])
            # YOLOv4: the tuner's validation on the bundle's reference inputs.
            raise RuntimeError("uv exited 4: model autotune rejected: the tuned model misses the tolerance on "
                               "the validation inputs\n"
                               "  standalone vs EP   Identity:0   n=689520    max|Δ|=0.000e0  relative=0.00e0  "
                               "beyond tolerance: 0\n"
                               "  standalone vs CPU  Identity:0   n=689520    max|Δ|=2.009e-5  relative=1.10e-5  "
                               "beyond tolerance: 1\n")
        return SimpleNamespace(stdout="{}")

    monkeypatch.setattr(pipeline, "gpu_lock", no_gpu_lock)
    monkeypatch.setattr(pipeline, "tool_identity", lambda: {"tool": "test"})
    monkeypatch.setattr(pipeline, "binary", lambda name: Path(name))
    monkeypatch.setattr(pipeline, "command", command)
    monkeypatch.setattr(pipeline, "inspect_model",
                        lambda *args, **kwargs: {"compatible": False, "tuning": {"tunable_workloads": 1}})
    monkeypatch.setattr(pipeline, "_tuning_priority", lambda report: {})
    monkeypatch.setattr(pipeline, "_run", run)

    pipeline.optimize_bundle(source, tmp_path / "optimized")

    report = json.loads((tmp_path / "optimized" / "report.json").read_text())["default"]
    assert report["status"] == "compatible_no_measured_gain"
    assert report["tactics_rejected"] == "output Identity:0: max|Δ|=2.009e-5, 1 value(s) beyond tolerance"
    assert report["stopped"] == "identical"
    reference = tuner[0][tuner[0].index("--reference") + 1]
    assert Path(reference).parts[-2:] == ("default", "reference")
    assert None in runs and "default/tactics.json" not in runs
    optimized = verify_bundle(tmp_path / "optimized")
    assert optimized["profiles"]["default"]["tactics"] is None
    # The sealed manifest names the policy it was optimized under, and the tuner
    # ran under a recorded policy whose tuning variables the pipeline set.
    assert optimized["execution_policy"] == policy.pipeline_identity()
    assert optimized["tuner_execution_policy"] == policy.resolve()
    assert "default/tuning-policy.json" in optimized["files"]
    assert tuner_env[0]["ONNX_VULKAN_TUNING_MODE"] == "off" and tuner_env[0]["ONNX_VULKAN_TUNING_ARTIFACT"] == ""


def test_a_tuner_failure_that_is_not_a_tolerance_verdict_still_fails_optimize(tmp_path, monkeypatch):
    source, _ = _source_bundle(tmp_path)

    @contextmanager
    def no_gpu_lock(_timeout):
        yield

    def command(argv, **kwargs):
        if any(str(arg).endswith("scripts/autotune/model.py") for arg in argv):
            raise RuntimeError("uv exited 3: model autotune failed: Vulkan device lost")
        return SimpleNamespace(stdout="{}")

    monkeypatch.setattr(pipeline, "gpu_lock", no_gpu_lock)
    monkeypatch.setattr(pipeline, "tool_identity", lambda: {"tool": "test"})
    monkeypatch.setattr(pipeline, "binary", lambda name: Path(name))
    monkeypatch.setattr(pipeline, "command", command)
    monkeypatch.setattr(pipeline, "inspect_model",
                        lambda *args, **kwargs: {"compatible": False, "tuning": {"tunable_workloads": 1}})
    monkeypatch.setattr(pipeline, "_tuning_priority", lambda report: {})
    monkeypatch.setattr(pipeline, "_run", lambda *args, **kwargs: {"standalone_ms": [1.0], "passed": True})

    with pytest.raises(RuntimeError, match="device lost"):
        pipeline.optimize_bundle(source, tmp_path / "optimized")


def test_interrupted_optimization_workspace_resumes_same_identity(tmp_path, monkeypatch):
    source, manifest = _source_bundle(tmp_path)
    monkeypatch.setattr(pipeline, "tool_identity", lambda: {"tool": "test"})
    destination = tmp_path / "optimized"

    with pytest.raises(RuntimeError, match="interrupt"):
        with pipeline._optimization_workspace(source, destination, manifest, 20, True):
            raise RuntimeError("interrupt")

    staging = tmp_path / ".optimized.incomplete"
    assert staging.is_dir()
    assert json.loads((staging / "resume.json").read_text()) == {
        "source": manifest["identity"],
        "tools": {"tool": "test"},
        "samples": 20,
        "tune": True,
        "profiles": manifest["profiles"],
        "atol": manifest["atol"],
        "rtol": manifest["rtol"],
        "execution_policy": {
            "version": 1,
            "numerics": {"attention_fp16": False},
            "allocator": {"pool_budget_mb": "auto"},
            "ep": {"compile": True, "op_filter": None},
            "measurement": {"stats": True, "probes": []},
            "device_selection": {},
        },
        # How timings are taken: the profiler never times, the span probe does, and the
        # confirmation runs with nothing enabled.
        "measurement": {"version": 1, "modes": {
            "diagnostic": {"stats": True, "probes": []},
            "timing": {"stats": False, "probes": ["VULKAN_GPU_SPAN"]},
            "confirmation": {"stats": False, "probes": []},
        }},
        # The tuner subprocess runs under the caller's settings, unlike the pipeline's own runs.
        "tuner_execution_policy": {
            "version": 1,
            "numerics": {"attention_fp16": False},
            "allocator": {"pool_budget_mb": "auto"},
            "tuning": {"mode": "off", "artifact": None},
            "ep": {"compile": False, "op_filter": None},
            "measurement": {"stats": False, "probes": []},
            "device_selection": {},
        },
    }

    with pipeline._optimization_workspace(source, destination, manifest, 20, True):
        pass
    assert not staging.exists()


def test_interrupted_optimization_workspace_rejects_altered_source(tmp_path, monkeypatch):
    source, manifest = _source_bundle(tmp_path)
    monkeypatch.setattr(pipeline, "tool_identity", lambda: {"tool": "test"})
    destination = tmp_path / "optimized"
    with pytest.raises(RuntimeError):
        with pipeline._optimization_workspace(source, destination, manifest, 20, True):
            raise RuntimeError("interrupt")

    (source / "default" / "model.onnx").write_bytes(b"altered")
    seal(source, json.loads((source / "manifest.json").read_text()))
    altered_manifest = verify_bundle(source)
    with pytest.raises(ValueError, match="Stale experiment"):
        with pipeline._optimization_workspace(source, destination, altered_manifest, 20, True):
            pass


def test_interrupted_optimization_workspace_rejects_altered_tools(tmp_path, monkeypatch):
    source, manifest = _source_bundle(tmp_path)
    tool = {"tool": "test"}
    monkeypatch.setattr(pipeline, "tool_identity", lambda: tool)
    destination = tmp_path / "optimized"
    with pytest.raises(RuntimeError):
        with pipeline._optimization_workspace(source, destination, manifest, 20, True):
            raise RuntimeError("interrupt")

    monkeypatch.setattr(pipeline, "tool_identity", lambda: {"tool": "changed"})
    with pytest.raises(ValueError, match="Stale experiment"):
        with pipeline._optimization_workspace(source, destination, manifest, 20, True):
            pass


def test_quantized_model_uses_its_own_reference(tmp_path):
    profile = tmp_path / "default"
    (profile / "reference").mkdir(parents=True)
    assert pipeline._default_reference(tmp_path, "default", "default/model.onnx") == profile / "reference"
    (profile / "quantized-reference").mkdir()
    assert pipeline._default_reference(tmp_path, "default", "default/model.onnx") == profile / "quantized-reference"
    assert pipeline._default_reference(tmp_path, "default", "default/original.onnx") == profile / "reference"


def test_quality_is_not_claimed_for_fp32_fallback(tmp_path, monkeypatch):
    (tmp_path / "default" / "quality" / "0000").mkdir(parents=True)
    monkeypatch.setattr(pipeline, "_run", lambda *args, **kwargs: pytest.fail("fp32 fallback ran quality fixtures"))
    manifest = {"profiles": {"default": {"model": "default/original.onnx", "tactics": None}}}
    result = pipeline._verify_quality(tmp_path, manifest, "default", "standalone", 1)
    assert result["vulkan_verified"] is False
    assert result["samples"] == 0


def _observed(env):
    """The instrumentation a runtime would report for a child started with `env`
    (a None value removes the variable), each variable counting by presence."""
    present = {name for name, value in env.items() if value is not None}
    return {"stats": "VULKAN_EP_STATS" in present, "gpu_span": "VULKAN_GPU_SPAN" in present,
            "node_probe": "NODE_PROBE" in present, "alloc_probe": "ALLOC_PROBE" in present,
            "pool_probe": "POOL_PROBE" in present}


def test_a_dynamically_quantized_model_is_gated_by_parity_and_its_metric(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    (bundle / "default").mkdir(parents=True)
    pipeline.write_json(bundle / "default" / "preparation.json", {"quantization": {"mode": "dynamic"}})
    manifest = {"identity": "fixture", "atol": 1e-5, "rtol": 1e-4, "profiles": {
        "default": {"model": "default/model.onnx", "dimensions": {}, "tactics": None},
    }}
    monkeypatch.setattr(pipeline, "binary", lambda _: Path("model-runner"))
    commands = []

    def runner(args, **kwargs):
        commands.append([str(arg) for arg in args])
        Path(args[args.index("--report-json") + 1]).write_text(json.dumps({
            "passed": True, "gate": "parity", "cpu_fallback_disabled": True, "standalone_outputs": [{}],
            "outputs": [{"name": "y", "passed": False, "max_abs": 16.0}],
            "measurement": {"instrumentation": _observed(kwargs["env"])}}))
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(pipeline, "command", runner)
    report = pipeline._run(bundle, manifest, "default", "standalone", 1, 1)
    assert "--parity-gate" in commands[-1] and report["dynamic_quantization"] == "pipeline"
    # The fp32 original keeps the CPU tolerance gate.
    _dynamic_model(bundle / "default" / "original.onnx", quantized=False)
    pipeline._run(bundle, manifest, "default", "standalone", 1, 1, model="default/original.onnx")
    assert "--parity-gate" not in commands[-1]
    with pytest.raises(ValueError, match="--backend standalone"):
        pipeline._run(bundle, manifest, "default", "ep", 1, 1)
    with pytest.raises(RuntimeError, match="application metric"):
        pipeline._require_metric({"gate": "parity", "dynamic_quantization": "pipeline",
                                  "quality_evaluation": {"samples": 0, "vulkan_verified": False}})
    pipeline._require_metric({"gate": "parity", "dynamic_quantization": "pipeline",
                              "quality_evaluation": {"vulkan_verified": True}})
    pipeline._require_metric({"gate": "cpu", "quality_evaluation": {"vulkan_verified": False}})


def test_a_hopeless_candidate_stops_early_and_a_promising_one_does_not():
    from onnx_vulkan_pipeline.pipeline import FUTILITY_PAIRS, _futile
    baseline = [{"gpu_ms": 10.0 + i % 2 * 0.1, "wall_ms": 12.0 + i % 2 * 0.1} for i in range(FUTILITY_PAIRS)]
    same = [{"gpu_ms": 10.0, "wall_ms": 12.05} for _ in range(FUTILITY_PAIRS)]
    assert not _futile(baseline[:-1], same[:-1])
    assert _futile(baseline, same)
    # 12% faster on both: it may still pass at 20 pairs.
    faster = [{"gpu_ms": 8.8, "wall_ms": 10.5} for _ in range(FUTILITY_PAIRS)]
    assert not _futile(baseline, faster)
    # Equal medians but noise wide enough to hide a 10% gain: keep measuring.
    noisy = [{"gpu_ms": 10.0 + (-3 if i % 2 else 3), "wall_ms": 12.0 + (-3 if i % 2 else 3)}
             for i in range(FUTILITY_PAIRS)]
    assert not _futile(baseline, noisy)
    # A sample without GPU time can never be promoted.
    assert _futile(baseline, faster[:-1] + [{"gpu_ms": None, "wall_ms": 10.5}])


def _dynamic_model(path, *, quantized=True):
    """x → DynamicQuantizeLinear ×2 → DequantizeLinear → out, or x → Relu → out."""
    import onnx
    from onnx import TensorProto, helper

    nodes = ([helper.make_node("DynamicQuantizeLinear", ["x"], ["q", "s", "z"]),
              helper.make_node("DequantizeLinear", ["q", "s", "z"], ["d"]),
              helper.make_node("DynamicQuantizeLinear", ["d"], ["q2", "s2", "z2"]),
              helper.make_node("DequantizeLinear", ["q2", "s2", "z2"], ["out"])] if quantized
             else [helper.make_node("Relu", ["x"], ["out"])])
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                              [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def test_an_already_quantized_source_is_gated_by_its_first_differing_quantized_tensor(tmp_path, monkeypatch):
    import onnx

    bundle = tmp_path / "bundle"
    (bundle / "default").mkdir(parents=True)
    pipeline.write_json(bundle / "default" / "preparation.json", {})
    _dynamic_model(bundle / "default" / "model.onnx")
    manifest = {"profiles": {"default": {"model": "default/model.onnx", "dimensions": {}, "tactics": None}}}
    assert pipeline._dynamic_quantization(bundle, "default", "default/model.onnx") == "source"
    runs = []

    def run(bundle_, manifest_, profile, backend, samples, timeout, *, model, tactics, reference):
        outputs = [output.name for output in onnx.load(str(model)).graph.output]
        runs.append((outputs, reference))
        return {"outputs": [{"name": "out", "dtype": "float32", "candidate_dtype": "float32", "max_abs": 9.0},
                            {"name": "q", "dtype": "uint8", "candidate_dtype": "uint8", "max_abs": codes[0]},
                            {"name": "q2", "dtype": "uint8", "candidate_dtype": "uint8", "max_abs": codes[1]}]}

    monkeypatch.setattr(pipeline, "_run", run)
    verify = lambda: pipeline._verify_quantized_tensors(bundle, manifest, "default", "standalone", 1)  # noqa: E731
    # The first quantized tensor matches exactly; the second is the first that
    # differs and is gated. A cascade after it would not be.
    codes = (0.0, 1.0)
    tensors = verify()
    assert runs == [(["out", "q", "q2"], bundle / "default" / "reference")]
    assert tensors == {"total": 2, "identical": 1, "tensor": "q2", "max_abs": 1.0, "runs": 1,
                       "passed": True}
    report = {"gate": "parity", "dynamic_quantization": "source", "quantized_tensors": tensors,
              "quality_evaluation": {"samples": 0, "vulkan_verified": False}}
    pipeline._require_metric(report)
    codes = (0.0, 0.0)
    assert verify() == {"total": 2, "identical": 2, "tensor": None, "max_abs": 0.0, "runs": 1,
                        "passed": True}
    codes = (2.0, 0.0)
    report["quantized_tensors"] = verify()
    with pytest.raises(RuntimeError, match="`q` differs from the CPU EP by 2 codes"):
        pipeline._require_metric(report)


def test_quantized_tensors_are_exposed_in_bounded_batches_until_one_differs(tmp_path, monkeypatch):
    import onnx

    bundle = tmp_path / "bundle"
    (bundle / "default").mkdir(parents=True)
    _dynamic_model(bundle / "default" / "model.onnx")
    manifest = {"profiles": {"default": {"model": "default/model.onnx", "dimensions": {}, "tactics": None}}}
    runs, codes = [], {"q": 0.0, "q2": 0.0}

    def run(bundle_, manifest_, profile, backend, samples, timeout, *, model, tactics, reference):
        outputs = [output.name for output in onnx.load(str(model)).graph.output]
        runs.append(outputs)
        return {"outputs": [{"name": name, "dtype": "uint8", "candidate_dtype": "uint8", "max_abs": codes[name]}
                            for name in outputs[1:]]}

    monkeypatch.setattr(pipeline, "_run", run)
    # Each [1, 4] tensor fills a batch on its own.
    monkeypatch.setattr(pipeline, "PER_NODE_ELEMENTS", 4)
    verify = lambda: pipeline._verify_quantized_tensors(bundle, manifest, "default", "standalone", 1)  # noqa: E731
    assert verify() == {"total": 2, "identical": 2, "tensor": None, "max_abs": 0.0, "runs": 2, "passed": True}
    assert runs == [["out", "q"], ["out", "q2"]]
    runs.clear()
    codes["q"] = 1.0
    assert verify()["runs"] == 1
    assert runs == [["out", "q"]]


def test_identical_weight_files_of_a_profile_are_stored_once(tmp_path):
    from onnx_vulkan_pipeline.pipeline import _link_duplicates

    weights = bytes(range(256)) * 8192
    for name in ("model.onnx_data", "candidate.data"):
        (tmp_path / name).write_bytes(weights)
    (tmp_path / "other.data").write_bytes(weights[:-1] + b"x")
    (tmp_path / "small").write_bytes(b"1")
    (tmp_path / "small2").write_bytes(b"1")
    _link_duplicates(tmp_path)
    assert (tmp_path / "candidate.data").samefile(tmp_path / "model.onnx_data")
    assert (tmp_path / "candidate.data").read_bytes() == weights
    assert not (tmp_path / "other.data").samefile(tmp_path / "candidate.data")
    assert not (tmp_path / "small").samefile(tmp_path / "small2")
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "candidate.data", "model.onnx_data", "other.data", "small", "small2"]


# --- Execution-policy identity ------------------------------------------------

@contextmanager
def _no_gpu_lock(_timeout):
    yield


def _interrupted_experiment(tmp_path, monkeypatch):
    source, manifest = _source_bundle(tmp_path)
    monkeypatch.setattr(pipeline, "tool_identity", lambda: {"tool": "test"})
    destination = tmp_path / "optimized"
    with pytest.raises(RuntimeError):
        with pipeline._optimization_workspace(source, destination, manifest, 20, True):
            raise RuntimeError("interrupt")
    return source, manifest, destination


def test_changing_fp16_attention_invalidates_an_incomplete_resume(tmp_path, monkeypatch):
    source, manifest, destination = _interrupted_experiment(tmp_path, monkeypatch)
    staging = tmp_path / ".optimized.incomplete"
    assert json.loads((staging / "resume.json").read_text())["execution_policy"]["numerics"] == {
        "attention_fp16": False}

    monkeypatch.setenv("ONNX_VULKAN_ATTENTION_FP16", "1")
    with pytest.raises(ValueError, match=r"Stale experiment .*execution policy changed: "
                                         r"numerics\.attention_fp16: False → True"):
        with pipeline._optimization_workspace(source, destination, manifest, 20, True):
            pass
    assert staging.is_dir()


def test_a_value_the_runtime_treats_as_off_resumes_an_unset_experiment(tmp_path, monkeypatch):
    source, manifest, destination = _interrupted_experiment(tmp_path, monkeypatch)
    # The runtime enables the f16 kernel only for the exact value `1`.
    monkeypatch.setenv("ONNX_VULKAN_ATTENTION_FP16", "true")
    with pipeline._optimization_workspace(source, destination, manifest, 20, True):
        pass
    assert not (tmp_path / ".optimized.incomplete").exists()


@pytest.mark.parametrize("initial_tune", [False, True])
def test_changing_tuning_mode_rejects_resume_and_preserves_evidence(tmp_path, monkeypatch, initial_tune):
    source, manifest = _source_bundle(tmp_path)
    monkeypatch.setattr(pipeline, "tool_identity", lambda: {"tool": "test"})
    destination = tmp_path / "optimized"
    with pytest.raises(RuntimeError, match="interrupt"):
        with pipeline._optimization_workspace(source, destination, manifest, 20, initial_tune):
            raise RuntimeError("interrupt")
    resume = tmp_path / ".optimized.incomplete" / "resume.json"
    original = resume.read_bytes()
    with pytest.raises(ValueError, match="Stale experiment"):
        with pipeline._optimization_workspace(source, destination, manifest, 20, not initial_tune):
            pytest.fail("resumed with a different tuning mode")
    assert resume.read_bytes() == original


def test_an_incomplete_experiment_without_a_policy_is_not_resumed(tmp_path, monkeypatch):
    source, manifest, destination = _interrupted_experiment(tmp_path, monkeypatch)
    resume = tmp_path / ".optimized.incomplete" / "resume.json"
    recorded = json.loads(resume.read_text())
    del recorded["execution_policy"]
    resume.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="Stale experiment .*predates execution-policy identity"):
        with pipeline._optimization_workspace(source, destination, manifest, 20, True):
            pass


@pytest.mark.parametrize(("name", "value"), [
    ("VULKAN_POOL_BUDGET_MB", "256"),
    ("VULKAN_EP_OPS", "Conv,Relu"),
    ("VULKAN_GPU_SPAN", "1"),
    ("NODE_PROBE", ""),
    ("MESA_VK_DEVICE_SELECT", "10de:2786"),
    ("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json"),
])
def test_allocator_device_and_measurement_settings_change_the_resume_identity(tmp_path, monkeypatch, name, value):
    source, manifest, destination = _interrupted_experiment(tmp_path, monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="Stale experiment .*execution policy changed"):
        with pipeline._optimization_workspace(source, destination, manifest, 20, True):
            pass


@pytest.mark.parametrize("name", ["ONNX_VULKAN_ATTENTION_FP16", "VULKAN_POOL_BUDGET_MB", "VULKAN_EP_COMPILE",
                                  "ONNX_VULKAN_TUNING_MODE", "ONNX_VULKAN_TUNING_ARTIFACT", "VULKAN_EP_STATS"])
def test_ambient_values_the_pipeline_overrides_or_the_runtime_ignores_do_not_change_identity(monkeypatch, name):
    baseline = policy.pipeline_identity()
    values = {"ONNX_VULKAN_ATTENTION_FP16": ["0", "true", ""], "VULKAN_POOL_BUDGET_MB": ["abc", "", "-1", " 5"],
              "VULKAN_EP_COMPILE": ["0", ""], "ONNX_VULKAN_TUNING_MODE": ["require-cache"],
              "ONNX_VULKAN_TUNING_ARTIFACT": ["/elsewhere/tactics.json"], "VULKAN_EP_STATS": ["0", "1"]}
    for value in values[name]:
        monkeypatch.setenv(name, value)
        assert policy.pipeline_identity() == baseline, f"{name}={value!r}"


def test_pool_budget_is_canonicalized_as_the_runtime_parses_it(monkeypatch):
    def budget(value):
        monkeypatch.setenv("VULKAN_POOL_BUDGET_MB", value)
        return policy.pipeline_identity()["allocator"]["pool_budget_mb"]

    assert budget("128") == budget("0128") == budget("+128") == 128
    assert budget("0") == 0 and budget("0") != "auto"  # 0 frees everything; it is not the default
    assert budget("128") != budget("256")
    assert budget("") == budget("12MB") == budget("-1") == budget("18446744073709551616") == "auto"
    monkeypatch.delenv("VULKAN_POOL_BUDGET_MB")
    assert policy.pipeline_identity()["allocator"]["pool_budget_mb"] == "auto"


def test_diagnostic_probes_are_identified_by_presence_not_by_value(monkeypatch):
    unset = policy.pipeline_identity()
    for value in ("0", "", "1"):
        monkeypatch.setenv("VULKAN_GPU_SPAN", value)
        assert policy.pipeline_identity()["measurement"]["probes"] == ["VULKAN_GPU_SPAN"]
        assert policy.pipeline_identity() != unset
    assert policy.resolve(environ={"VULKAN_EP_STATS": "0"})["measurement"]["stats"] is True
    assert policy.resolve(environ={})["measurement"]["stats"] is False


def test_unrelated_environment_and_secrets_are_neither_identified_nor_reported(monkeypatch):
    baseline = policy.pipeline_identity()
    for name in ("HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "SOME_OTHER_SETTING", "RUST_LOG", "ORT_LOGGING_LEVEL"):
        monkeypatch.setenv(name, "s3cret-value")
    assert policy.pipeline_identity() == baseline
    assert "s3cret-value" not in json.dumps(policy.resolve(tactics="default/tactics.json"))
    assert set(policy.environment(policy.resolve())) <= {
        "ONNX_VULKAN_ATTENTION_FP16", "VULKAN_POOL_BUDGET_MB", "ONNX_VULKAN_TUNING_MODE",
        "ONNX_VULKAN_TUNING_ARTIFACT", "VULKAN_EP_COMPILE", "VULKAN_EP_OPS", "VULKAN_EP_STATS",
        *policy.PROBES, *policy.DEVICE_SELECTION}


def test_a_none_value_removes_an_inherited_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("POLICY_TEST_INHERITED", "yes")
    monkeypatch.setenv("POLICY_TEST_KEPT", "yes")
    probe = "import os; print(os.environ.get('POLICY_TEST_INHERITED'), os.environ.get('POLICY_TEST_KEPT'), " \
            "os.environ.get('POLICY_TEST_SET'))"
    result = command([sys.executable, "-c", probe], env={"POLICY_TEST_INHERITED": None, "POLICY_TEST_SET": "1"})
    assert result.stdout.split() == ["None", "yes", "1"]


def _stub_runner(tmp_path, monkeypatch):
    """A `model-runner` that records the environment it was started with and
    writes a passing report, so the real process path builds the child's env."""
    bundle = tmp_path / "bundle"
    (bundle / "default").mkdir(parents=True)
    pipeline.write_json(bundle / "default" / "preparation.json", {})
    _dynamic_model(bundle / "default" / "model.onnx", quantized=False)
    runner = tmp_path / "model-runner"
    child_environment = tmp_path / "child.env"
    runner.write_text(f"""#!/bin/sh
while [ $# -gt 0 ]; do [ "$1" = "--report-json" ] && report="$2"; shift; done
env > '{child_environment}'
# what the runtime reports as enabled: each variable counts by presence, as it does there
on() {{ if grep -q "^$1=" '{child_environment}'; then printf true; else printf false; fi; }}
echo '{{"passed": true, "cpu_fallback_disabled": true, "standalone_outputs": [{{}}], "outputs": [{{"passed": true}}],' \
  '"measurement": {{"instrumentation": {{"stats": '$(on VULKAN_EP_STATS)', "gpu_span": '$(on VULKAN_GPU_SPAN)',' \
  '"node_probe": '$(on NODE_PROBE)', "alloc_probe": '$(on ALLOC_PROBE)', "pool_probe": '$(on POOL_PROBE)'}}}}}}' > "$report"
""")
    runner.chmod(0o755)
    monkeypatch.setattr(pipeline, "binary", lambda _: runner)
    manifest = {"identity": "fixture", "atol": 1e-5, "rtol": 1e-4, "profiles": {
        "default": {"model": "default/model.onnx", "dimensions": {}, "tactics": None, "compatible": True}}}

    def child():
        return dict(line.split("=", 1) for line in child_environment.read_text().splitlines() if "=" in line)

    return bundle, manifest, child


def test_the_child_environment_realizes_the_reported_policy(tmp_path, monkeypatch):
    bundle, manifest, child = _stub_runner(tmp_path, monkeypatch)
    # Parent values the pipeline overrides, one the runtime ignores, and presence-based probes.
    monkeypatch.setenv("VULKAN_EP_COMPILE", "0")
    monkeypatch.setenv("VULKAN_EP_STATS", "0")
    monkeypatch.setenv("ONNX_VULKAN_TUNING_MODE", "require-cache")
    monkeypatch.setenv("ONNX_VULKAN_TUNING_ARTIFACT", "/elsewhere/tactics.json")
    monkeypatch.setenv("ONNX_VULKAN_ATTENTION_FP16", "true")
    monkeypatch.setenv("VULKAN_GPU_SPAN", "0")
    monkeypatch.setenv("NODE_PROBE", "1")
    monkeypatch.setenv("VULKAN_POOL_BUDGET_MB", "0512")
    monkeypatch.setenv("HF_TOKEN", "s3cret-value")

    report = pipeline._run(bundle, manifest, "default", "standalone", 1, 30)

    env = child()
    assert env["VULKAN_EP_COMPILE"] == "1" and env["VULKAN_EP_STATS"] == "1"
    assert env["ONNX_VULKAN_TUNING_MODE"] == "off" and env["ONNX_VULKAN_TUNING_ARTIFACT"] == ""
    assert "ONNX_VULKAN_ATTENTION_FP16" not in env  # `true` is off for the runtime; the child gets the canonical form
    # ambient probes never reach a measured child: the mode decides which are on
    assert "VULKAN_GPU_SPAN" not in env and "NODE_PROBE" not in env
    assert env["VULKAN_POOL_BUDGET_MB"] == "512"
    assert env["RUST_LOG"] == "info"
    assert report["execution_policy"] == {
        "version": 1,
        "numerics": {"attention_fp16": False},
        "allocator": {"pool_budget_mb": 512},
        "tuning": {"mode": "off", "artifact": None},
        "ep": {"compile": True, "op_filter": None},
        "measurement": {"stats": True, "probes": []},
        "device_selection": {},
    }
    assert report["measurement_mode"] == "diagnostic"
    assert "s3cret-value" not in json.dumps(report)


@pytest.mark.parametrize(("mode", "profiler", "span", "log"), [
    ("diagnostic", True, False, "info"),
    ("timing", False, True, "warn"),
    ("confirmation", False, False, "warn"),
])
def test_each_measurement_mode_runs_with_exactly_its_instrumentation(tmp_path, monkeypatch, mode, profiler, span, log):
    bundle, manifest, child = _stub_runner(tmp_path, monkeypatch)
    for name in ("VULKAN_EP_STATS", "VULKAN_GPU_SPAN", "NODE_PROBE", "ALLOC_PROBE", "POOL_PROBE"):
        monkeypatch.setenv(name, "1")
    report = pipeline._run(bundle, manifest, "default", "standalone", 1, 30, mode=mode)
    env = child()
    assert ("VULKAN_EP_STATS" in env, "VULKAN_GPU_SPAN" in env) == (profiler, span)
    assert not {"NODE_PROBE", "ALLOC_PROBE", "POOL_PROBE"} & set(env)
    assert env["RUST_LOG"] == log
    assert report["measurement_mode"] == mode
    assert report["execution_policy"]["measurement"] == {
        "stats": profiler, "probes": ["VULKAN_GPU_SPAN"] if span else []}
    assert report["measurement"]["instrumentation"] == {
        "stats": profiler, "gpu_span": span, "node_probe": False, "alloc_probe": False, "pool_probe": False}


def test_a_run_whose_runtime_reports_other_instrumentation_than_its_mode_is_rejected(tmp_path, monkeypatch):
    bundle, manifest, _ = _stub_runner(tmp_path, monkeypatch)
    runner = tmp_path / "model-runner"
    original = runner.read_text()
    # a runtime that reads a probe the pipeline did not enable
    runner.write_text(original.replace("$(on NODE_PROBE)", "true"))
    with pytest.raises(RuntimeError, match="differs from its measurement mode"):
        pipeline._run(bundle, manifest, "default", "standalone", 1, 30, mode="timing")
    # and one that reports none of its telemetry
    runner.write_text(original.replace('"measurement"', '"unrelated"'))
    with pytest.raises(RuntimeError, match="reported no instrumentation"):
        pipeline._run(bundle, manifest, "default", "standalone", 1, 30, mode="confirmation")


def test_a_tuned_run_names_its_artifact_in_the_command_and_the_report(tmp_path, monkeypatch):
    bundle, manifest, child = _stub_runner(tmp_path, monkeypatch)
    monkeypatch.setenv("ONNX_VULKAN_ATTENTION_FP16", "1")
    report = pipeline._run(bundle, manifest, "default", "standalone", 1, 30, tactics="default/tactics.json")
    env = child()
    assert env["ONNX_VULKAN_TUNING_MODE"] == "cache-only"
    assert env["ONNX_VULKAN_TUNING_ARTIFACT"] == str(bundle / "default/tactics.json")
    assert env["ONNX_VULKAN_ATTENTION_FP16"] == "1"
    assert report["execution_policy"]["tuning"] == {"mode": "cache-only", "artifact": "default/tactics.json"}
    assert report["execution_policy"]["numerics"] == {"attention_fp16": True}


def _tuned_with_policy(tmp_path, monkeypatch, recorded):
    device = {"gpu": "same"}
    bundle = _tuned_bundle(tmp_path, tools={"tool": "same"}, device=device)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if recorded is not None:
        manifest["execution_policy"] = recorded
    seal(bundle, manifest)
    monkeypatch.setattr(pipeline, "gpu_lock", _no_gpu_lock)
    monkeypatch.setattr(pipeline, "tool_identity", lambda: {"tool": "same"})
    monkeypatch.setattr(pipeline, "binary", lambda name: Path(name))
    monkeypatch.setattr(pipeline, "command", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(device)))
    return bundle


def test_a_tuned_bundle_is_reused_only_under_the_policy_it_was_optimized_under(tmp_path, monkeypatch):
    bundle = _tuned_with_policy(tmp_path, monkeypatch, policy.pipeline_identity())
    runs = []
    monkeypatch.setattr(pipeline, "_run", lambda *args, **kwargs: runs.append(kwargs["tactics"]) or {})
    monkeypatch.setattr(pipeline, "_verify_quality", lambda *args: {})

    # Measurement does not make a tuned artifact incompatible, but the report says it differs.
    monkeypatch.setenv("VULKAN_GPU_SPAN", "1")
    report = pipeline.run_bundle(bundle, backend="ep", samples=1)
    assert runs == ["default/tactics.json"]
    assert report["bundle_execution_policy"] == {
        "status": "different", "differences": ["measurement.probes: [] → ['VULKAN_GPU_SPAN']"]}
    monkeypatch.delenv("VULKAN_GPU_SPAN")

    assert pipeline.run_bundle(bundle, backend="ep", samples=1)["bundle_execution_policy"]["status"] == "match"

    for name, value, change in (("ONNX_VULKAN_ATTENTION_FP16", "1", "numerics.attention_fp16: False → True"),
                                ("VULKAN_POOL_BUDGET_MB", "64", "allocator.pool_budget_mb: 'auto' → 64"),
                                ("DRI_PRIME", "1", "device_selection.DRI_PRIME: None → '1'")):
        monkeypatch.setenv(name, value)
        runs.clear()
        with pytest.raises(ValueError, match="different execution policy") as rejected:
            pipeline.run_bundle(bundle, backend="ep", samples=1)
        assert change in str(rejected.value) and runs == []
        monkeypatch.delenv(name)


def test_a_tuned_bundle_without_a_policy_is_refused_but_an_untuned_one_still_runs(tmp_path, monkeypatch):
    bundle = _tuned_with_policy(tmp_path, monkeypatch, None)
    monkeypatch.setattr(pipeline, "_run", lambda *args, **kwargs: pytest.fail("legacy tuned bundle ran"))
    with pytest.raises(ValueError, match="records no execution policy"):
        pipeline.run_bundle(bundle, backend="ep", samples=1)

    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["profiles"]["default"]["tactics"] = None
    seal(bundle, manifest)
    monkeypatch.setattr(pipeline, "_run", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline, "_verify_quality", lambda *args: {})
    # Deliberate: a legacy bundle without tuned artifacts has nothing to be incompatible with.
    assert pipeline.run_bundle(bundle, backend="ep", samples=1)["bundle_execution_policy"] == {
        "status": "unrecorded", "differences": []}


def test_a_setting_the_pipeline_overrides_but_the_tuner_inherits_changes_a_tuned_resume(tmp_path, monkeypatch):
    source, manifest, destination = _interrupted_experiment(tmp_path, monkeypatch)
    staging = tmp_path / ".optimized.incomplete"
    pipeline_policy = policy.pipeline_identity()

    # The pipeline's own runs force statistics on, so its identity does not move...
    monkeypatch.setenv("VULKAN_EP_STATS", "1")
    assert policy.pipeline_identity() == pipeline_policy
    # ...but the tuner is launched with the caller's value, so tuned evidence would mix.
    with pytest.raises(ValueError, match=r"Stale experiment .*tuner execution policy changed: "
                                         r"measurement\.stats: False → True"):
        with pipeline._optimization_workspace(source, destination, manifest, 20, True):
            pass
    assert staging.is_dir()

    # Without tuning no tuner is launched, so the same ambient change is irrelevant.
    monkeypatch.delenv("VULKAN_EP_STATS")
    untuned, untuned_manifest = _source_bundle(tmp_path / "untuned")
    untuned_destination = tmp_path / "untuned" / "optimized"
    with pytest.raises(RuntimeError):
        with pipeline._optimization_workspace(untuned, untuned_destination, untuned_manifest, 20, False):
            raise RuntimeError("interrupt")
    monkeypatch.setenv("VULKAN_EP_STATS", "1")
    monkeypatch.setenv("VULKAN_EP_COMPILE", "1")
    with pipeline._optimization_workspace(untuned, untuned_destination, untuned_manifest, 20, False):
        pass
    assert not (tmp_path / "untuned" / ".optimized.incomplete").exists()


def test_the_tuner_is_launched_under_the_policy_the_experiment_recorded(tmp_path, monkeypatch):
    source, _ = _source_bundle(tmp_path)
    monkeypatch.setenv("VULKAN_EP_STATS", "1")
    launched = []

    def command(argv, **kwargs):
        if any(str(arg).endswith("scripts/autotune/model.py") for arg in argv):
            launched.append(kwargs["env"])
            raise RuntimeError("uv exited 4: model autotune rejected: the tuned model misses the tolerance on "
                               "the validation inputs\n"
                               "  standalone vs CPU  Identity:0   n=1    max|Δ|=1.000e-5  relative=1.00e-5  "
                               "beyond tolerance: 1\n")
        return SimpleNamespace(stdout="{}")

    monkeypatch.setattr(pipeline, "gpu_lock", _no_gpu_lock)
    monkeypatch.setattr(pipeline, "tool_identity", lambda: {"tool": "test"})
    monkeypatch.setattr(pipeline, "binary", lambda name: Path(name))
    monkeypatch.setattr(pipeline, "command", command)
    monkeypatch.setattr(pipeline, "inspect_model",
                        lambda *args, **kwargs: {"compatible": False, "tuning": {"tunable_workloads": 1}})
    monkeypatch.setattr(pipeline, "_tuning_priority", lambda report: {})
    monkeypatch.setattr(pipeline, "_run", lambda *args, **kwargs: {"standalone_ms": [1.0], "passed": True})
    pipeline.optimize_bundle(source, tmp_path / "optimized")

    assert launched[0]["VULKAN_EP_STATS"] == "1"
    optimized = verify_bundle(tmp_path / "optimized")
    assert optimized["tuner_execution_policy"]["measurement"]["stats"] is True
    recorded = json.loads((tmp_path / "optimized" / "default" / "tuning-policy.json").read_text())
    assert recorded == optimized["tuner_execution_policy"]


def test_optimize_stops_when_the_tuner_policy_changes_during_the_experiment(tmp_path, monkeypatch):
    source, _ = _source_bundle(tmp_path)

    def run(*args, **kwargs):
        # The pipeline's runs force this on, so only the tuner launch policy moves.
        monkeypatch.setenv("VULKAN_EP_STATS", "1")
        return {"standalone_ms": [1.0], "passed": True}

    monkeypatch.setattr(pipeline, "gpu_lock", _no_gpu_lock)
    monkeypatch.setattr(pipeline, "tool_identity", lambda: {"tool": "test"})
    monkeypatch.setattr(pipeline, "binary", lambda name: Path(name))
    monkeypatch.setattr(pipeline, "command", lambda *args, **kwargs: SimpleNamespace(stdout="{}"))
    monkeypatch.setattr(pipeline, "inspect_model", lambda *args, **kwargs: {"compatible": False})
    monkeypatch.setattr(pipeline, "_run", run)
    with pytest.raises(ValueError, match="Tuner execution policy changed during optimization"):
        pipeline.optimize_bundle(source, tmp_path / "optimized")
    assert not (tmp_path / "optimized").exists()

    # The same change is irrelevant when no tuner is launched.
    monkeypatch.delenv("VULKAN_EP_STATS")
    pipeline.optimize_bundle(source, tmp_path / "untuned", tune=False)
    assert verify_bundle(tmp_path / "untuned")["tuner_execution_policy"] is None


def test_optimize_stops_when_the_policy_changes_during_the_experiment(tmp_path, monkeypatch):
    source, _ = _source_bundle(tmp_path)

    def run(*args, **kwargs):
        monkeypatch.setenv("ONNX_VULKAN_ATTENTION_FP16", "1")
        return {"standalone_ms": [1.0], "passed": True}

    monkeypatch.setattr(pipeline, "gpu_lock", _no_gpu_lock)
    monkeypatch.setattr(pipeline, "tool_identity", lambda: {"tool": "test"})
    monkeypatch.setattr(pipeline, "binary", lambda name: Path(name))
    monkeypatch.setattr(pipeline, "command", lambda *args, **kwargs: SimpleNamespace(stdout="{}"))
    monkeypatch.setattr(pipeline, "inspect_model", lambda *args, **kwargs: {"compatible": False})
    monkeypatch.setattr(pipeline, "_run", run)
    with pytest.raises(ValueError, match="Execution policy changed during optimization"):
        pipeline.optimize_bundle(source, tmp_path / "optimized")
    assert not (tmp_path / "optimized").exists()
    assert (tmp_path / ".optimized.incomplete" / "failure.json").is_file()


# --- Measurement modes: timing, diagnostics and confirmation ---------------------

def _span(busy=5.0, **overrides):
    return {"flushes": 1, "command_buffers": 3, "submits": 3, "busy_ms": busy, "span_ms": busy + 0.5,
            "complete": True, **overrides}


def test_a_timing_row_takes_the_standalone_span_of_its_last_inference():
    report = {"standalone_ms": [30.0, 9.0], "standalone_storage_peak_bytes": 7, "measurement": {
        "standalone_gpu_span": [_span(20.0), _span(4.0)],
        # a different execution of the same process, never the subject of a standalone row
        "candidate_gpu_span": [_span(99.0), _span(99.0)]}}
    assert pipeline._timing_row(report) == {
        "wall_ms": 9.0, "gpu_ms": 4.0, "gpu_span": _span(4.0), "storage_peak_bytes": 7,
        "measurement_mode": "timing"}


@pytest.mark.parametrize("measurement", [
    None,
    {},
    {"standalone_gpu_span": []},
    {"standalone_gpu_span": [None]},
    {"standalone_gpu_span": [_span(complete=False)]},
    {"standalone_gpu_span": [_span(flushes=0)]},
    {"standalone_gpu_span": [_span(command_buffers=0)]},
    {"standalone_gpu_span": [_span(busy=float("nan"))]},
    {"standalone_gpu_span": [_span(busy=0.0)]},
    {"standalone_gpu_span": [{**_span(), "span_ms": 1.0}]},
])
def test_a_timing_without_valid_gpu_evidence_has_no_gpu_time_and_cannot_promote(measurement):
    report = {"standalone_ms": [9.0]}
    if measurement is not None:
        report["measurement"] = measurement
    row = pipeline._timing_row(report)
    assert row["gpu_ms"] is None and row["wall_ms"] == 9.0
    baseline = [{"gpu_ms": 10.0, "wall_ms": 12.0}] * 20
    assert not _improvement(baseline, [row] * 20)
    # a profiler-free wall gain alone is the same verdict
    assert not _improvement(baseline, [{**row, "wall_ms": 5.0}] * 20)
    assert _improvement(baseline, [{**row, "wall_ms": 5.0, "gpu_ms": 5.0}] * 20)


def test_only_the_uninstrumented_confirmation_wall_confirms():
    baseline = [{"wall_ms": 12.0}] * 20
    assert pipeline._confirmed(baseline, [{"wall_ms": 8.0}] * 20)
    assert not pipeline._confirmed(baseline, [{"wall_ms": 11.5}] * 20)
    assert not pipeline._confirmed(baseline, [{"wall_ms": None}] * 20)
    assert not pipeline._confirmed(baseline[:19], [{"wall_ms": 8.0}] * 19)


def test_rows_of_another_measurement_mode_are_never_mixed_in(tmp_path):
    path = tmp_path / "benchmark.json"
    assert pipeline._load_rows(path, "timing") == {"baseline": [], "candidate": []}
    pipeline.write_json(path, {"baseline": [{"wall_ms": 1.0, "measurement_mode": "timing"}] * 3,
                               "candidate": [{"wall_ms": 1.0, "measurement_mode": "timing"}] * 2})
    # an unfinished pair is dropped
    assert [len(v) for v in pipeline._load_rows(path, "timing").values()] == [2, 2]
    with pytest.raises(ValueError, match="rows measured as 'timing', not 'confirmation'"):
        pipeline._load_rows(path, "confirmation")
    # rows written before the modes existed carry none
    pipeline.write_json(path, {"baseline": [{"wall_ms": 1.0, "gpu_ms": 1.0}], "candidate": []})
    with pytest.raises(ValueError, match="measured as None"):
        pipeline._load_rows(path, "timing")


def test_interleaving_alternates_the_order_and_resumes_without_repeating_a_run(tmp_path):
    path, order = tmp_path / "rows.json", []

    def measure(label):
        order.append(label)
        return {"label": label, "measurement_mode": "timing"}

    rows = {"baseline": [], "candidate": []}
    assert pipeline._interleave(rows, path, 4, measure) is None
    assert order == ["baseline", "candidate", "candidate", "baseline"] * 1 + ["baseline", "candidate"] * 1 + [
        "candidate", "baseline"]
    order.clear()
    # a run interrupted after the first of a pair resumes with the second only
    rows = {"baseline": rows["baseline"][:2], "candidate": rows["candidate"][:1]}
    pipeline._interleave(rows, path, 2, measure)
    assert order == ["candidate"]
    assert pipeline._interleave({"baseline": [{}] * 5, "candidate": [{}] * 5}, path, 6, measure,
                                futile=lambda *_: True) == "futility"


def test_a_changed_measurement_plan_or_one_that_predates_modes_is_a_stale_resume(tmp_path, monkeypatch):
    source, manifest, destination = _interrupted_experiment(tmp_path, monkeypatch)
    resume = tmp_path / ".optimized.incomplete" / "resume.json"
    recorded = json.loads(resume.read_text())
    original = resume.read_bytes()

    # an experiment whose rows were timed before the modes were recorded
    del recorded["measurement"]
    resume.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="Stale experiment .*predates measurement-mode identity"):
        with pipeline._optimization_workspace(source, destination, manifest, 20, True):
            pytest.fail("resumed rows of an unknown measurement")

    # the timing mode changed: its rows measure something else now
    resume.write_bytes(original)
    monkeypatch.setitem(policy.MODES, "timing", {"stats": True, "probes": []})
    with pytest.raises(ValueError, match=r"Stale experiment .*measurement modes changed: modes\.timing: "):
        with pipeline._optimization_workspace(source, destination, manifest, 20, True):
            pytest.fail("resumed rows of another timing mode")
    assert resume.read_bytes() == original


def _optimize_scenario(tmp_path, monkeypatch, *, gpu=5.0, wall=8.0, confirmed=8.0, coverage=(4, 4), candidate=None):
    """`optimize_bundle` on a tunable profile whose rewrite is smaller than the source,
    with the runner replaced by one that answers per measurement mode."""
    source, _ = _source_bundle(tmp_path)
    (source / "default" / "original.onnx").write_bytes(b"source graph, larger")
    seal(source, json.loads((source / "manifest.json").read_text()))
    calls = []
    graphs = {"default/original.onnx": (10, 10), "default/model.onnx": coverage}

    def inspect(path, **kwargs):
        total, supported = graphs[f"default/{Path(path).name}"]
        return {"compatible": True, "tuning": {"tunable_workloads": 1},
                "execution": {"total_nodes": total, "supported_nodes": supported,
                              "unsupported_nodes": total - supported}}

    def run(bundle, manifest, profile, backend, samples, timeout, *, mode="diagnostic", model=None, tactics=None,
            **kwargs):
        side = "baseline" if model else "candidate"
        calls.append((mode, side, samples, tactics))
        if mode == "timing":
            busy, wall_ms = (10.0, 12.0) if side == "baseline" else (gpu, wall)
            span = None if busy is None else _span(busy)
            return {"standalone_ms": [wall_ms], "measurement": {"standalone_gpu_span": [span]}}
        if mode == "confirmation":
            return {"standalone_ms": [12.0 if side == "baseline" else confirmed]}
        evidence = {"flushes": 1, "upload_mb": 4, "download_mb": 2, "blocks": {"convex_blocks": 1, "nodes_claimed": 9}}
        if samples == 1 and (model or tactics):
            evidence = {**evidence, **((candidate or {}) if side == "candidate" else {})}
        return {"passed": True, "cpu_fallback_disabled": True, "gpu": evidence, "standalone_ms": [1.0]}

    def command(argv, **kwargs):
        if any(str(arg).endswith("scripts/autotune/model.py") for arg in argv):
            Path(argv[argv.index("--output") + 1]).write_text("{}")
        return SimpleNamespace(stdout="{}")

    monkeypatch.setattr(pipeline, "gpu_lock", _no_gpu_lock)
    monkeypatch.setattr(pipeline, "tool_identity", lambda: {"tool": "test"})
    monkeypatch.setattr(pipeline, "binary", lambda name: Path(name))
    monkeypatch.setattr(pipeline, "command", command)
    monkeypatch.setattr(pipeline, "inspect_model", inspect)
    monkeypatch.setattr(pipeline, "_tuning_priority", lambda report: {})
    monkeypatch.setattr(pipeline, "_run", run)
    return source, calls


def _outcome(tmp_path):
    return json.loads((tmp_path / "optimized" / "report.json").read_text())["default"]


def test_a_smaller_fully_covered_rewrite_is_promoted_after_timing_diagnostics_and_confirmation(tmp_path, monkeypatch):
    source, calls = _optimize_scenario(tmp_path, monkeypatch)
    pipeline.optimize_bundle(source, tmp_path / "optimized")
    outcome = _outcome(tmp_path)
    assert outcome["status"] == "accelerated" and outcome["pairs"] == 20
    timing = [call for call in calls if call[0] == "timing"]
    confirmation = [call for call in calls if call[0] == "confirmation"]
    # 20 interleaved pairs in both: AB, BA, AB, ...
    for runs in (timing, confirmation):
        assert [side for _, side, *_ in runs[:4]] == ["baseline", "candidate", "candidate", "baseline"]
        assert len(runs) == 40
    # the profiled runs are separate and few: one per graph, none of them a timing
    profiled = [call for call in calls if call[0] == "diagnostic" and call[2] == 1]
    # (the tuner's profile of the candidate model comes first, then one run per graph)
    assert profiled[1:] == [("diagnostic", "baseline", 1, None), ("diagnostic", "candidate", 1, "default/tactics.json")]
    # the timings come before the diagnostics, and these before the confirmation
    modes = [mode for mode, *_ in calls if mode != "diagnostic"]
    assert modes == ["timing"] * 40 + ["confirmation"] * 40
    assert outcome["measurement"]["diagnostics_hold"] is None
    assert outcome["measurement"]["gpu_metric"] == "gpu_span.busy_ms"
    assert outcome["measurement"]["confirmation"] == {"pairs": 20, "baseline": 12.0, "candidate": 8.0}
    # the evidence names each graph on its own
    graphs = outcome["measurement"]["diagnostics"]
    assert (graphs["baseline"]["graph"]["model"], graphs["baseline"]["graph"]["total_nodes"]) == (
        "default/original.onnx", 10)
    assert (graphs["candidate"]["graph"]["model"], graphs["candidate"]["graph"]["total_nodes"]) == (
        "default/model.onnx", 4)
    assert graphs["baseline"]["graph"]["digest"] != graphs["candidate"]["graph"]["digest"]
    assert (tmp_path / "optimized" / "default" / "diagnostics.json").is_file()


def test_a_rewrite_that_leaves_nodes_unsupported_is_not_promoted_however_fast(tmp_path, monkeypatch):
    source, calls = _optimize_scenario(tmp_path, monkeypatch, coverage=(4, 3))
    pipeline.optimize_bundle(source, tmp_path / "optimized")
    outcome = _outcome(tmp_path)
    assert outcome["status"] == "compatible_no_measured_gain"
    assert outcome["measurement"]["diagnostics_hold"] == ["candidate covers 3/4 nodes, source 10/10"]
    assert outcome["measurement"]["confirmation"] is None
    assert not [call for call in calls if call[0] == "confirmation"]


@pytest.mark.parametrize("worse", [{"download_mb": 3}, {"flushes": 2}, {"blocks": {"convex_blocks": 2, "nodes_claimed": 9}}])
def test_a_faster_rewrite_with_more_transfers_or_partitions_is_not_promoted(tmp_path, monkeypatch, worse):
    source, calls = _optimize_scenario(tmp_path, monkeypatch, candidate=worse)
    pipeline.optimize_bundle(source, tmp_path / "optimized")
    outcome = _outcome(tmp_path)
    assert outcome["status"] == "compatible_no_measured_gain" and outcome["measurement"]["diagnostics_hold"]
    assert not [call for call in calls if call[0] == "confirmation"]


def test_a_gain_the_uninstrumented_confirmation_does_not_show_is_not_promoted(tmp_path, monkeypatch):
    source, _ = _optimize_scenario(tmp_path, monkeypatch, confirmed=11.9)
    pipeline.optimize_bundle(source, tmp_path / "optimized")
    outcome = _outcome(tmp_path)
    assert outcome["status"] == "compatible_no_measured_gain"
    assert outcome["measurement"]["diagnostics_hold"] is None
    assert outcome["measurement"]["confirmation"]["candidate"] == 11.9


def test_a_wall_gain_without_gpu_evidence_is_not_promoted(tmp_path, monkeypatch):
    source, calls = _optimize_scenario(tmp_path, monkeypatch, gpu=None)
    pipeline.optimize_bundle(source, tmp_path / "optimized")
    outcome = _outcome(tmp_path)
    assert outcome["status"] == "compatible_no_measured_gain" and outcome["stopped"] == "futility"
    assert not [call for call in calls if call[0] in ("confirmation",)]
    assert outcome["candidate"]["median_gpu_ms"] is None
