import json
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import onnx_vulkan_pipeline.pipeline as pipeline
from onnx_vulkan_pipeline.pipeline import seal, verify_bundle, _improvement, dimensions
from onnx_vulkan_pipeline.process import command, gpu_lock


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


def test_faster_candidate_cannot_increase_transfers_or_partitioning():
    from copy import deepcopy
    row = {"gpu_ms": 10, "wall_ms": 12, "gpu": {
        "flushes": 1, "upload_mb": 4, "download_mb": 2,
        "blocks": {"convex_blocks": 1, "nodes_claimed": 12},
    }}
    fast = deepcopy(row)
    fast.update(gpu_ms=5, wall_ms=6)
    assert _improvement([row] * 20, [fast] * 20)
    fast["gpu"]["upload_mb"] = 5
    assert not _improvement([row] * 20, [fast] * 20)
    fast["gpu"]["upload_mb"] = 4
    fast["gpu"]["blocks"]["convex_blocks"] = 2
    assert not _improvement([row] * 20, [fast] * 20)


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

    runs, tuner = [], []

    def run(bundle, manifest, profile, backend, samples, timeout, *, tactics=None, **kwargs):
        runs.append(tactics)
        return {"standalone_ms": [1.0], "passed": True}

    def command(argv, **kwargs):
        if any(str(arg).endswith("scripts/autotune/model.py") for arg in argv):
            tuner.append([str(arg) for arg in argv])
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
    assert verify_bundle(tmp_path / "optimized")["profiles"]["default"]["tactics"] is None


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
            "outputs": [{"name": "y", "passed": False, "max_abs": 16.0}]}))
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
