import copy
import json

import pytest

from onnx_vulkan_pipeline import search, workload

RAW = {
    "schema_version": 1,
    "name": "demo",
    "source": "model.onnx",
    "adapter": "adapter.py",
    "profiles": {
        "short": {"dimensions": {"batch": 1, "sequence": 16}, "weight": 3, "max_regression": 0.02},
        "long": {"dimensions": {"batch": 1, "sequence": 128}, "weight": 1},
    },
    "tolerances": {"atol": 1e-5, "rtol": 1e-4},
    "quantization": {"config": "quantization.json", "grid": [{"mode": "dynamic"}, {"mode": "int4"}]},
    "budget": {"seconds": 3600, "attempts": 8, "top_k": 2, "samples": 20, "screen_samples": 3, "timeout": 120},
}


def _files(root, *, quantization=True):
    (root / "model.onnx").write_bytes(b"model")
    (root / "adapter.py").write_text("def evaluate(session, samples): return 0\n")
    (root / "ev").mkdir()
    (root / "ev" / "0.npz").write_text("x")
    if quantization:
        (root / "quantization.json").write_text(json.dumps({
            "evaluation": str(root / "ev"), "adapter": str(root / "adapter.py"), "metric": "m",
            "max_drop": 0.1, "higher_is_better": True}))


def _with(**changes):
    raw = copy.deepcopy(RAW)
    for key, value in changes.items():
        raw[key] = value
    return raw


def test_a_valid_workload_is_normalized_with_resolved_paths(tmp_path):
    validated = workload.validate(RAW, base=tmp_path)
    assert validated["source"] == tmp_path / "model.onnx"
    assert validated["profiles"] == {"short": {"batch": 1, "sequence": 16}, "long": {"batch": 1, "sequence": 128}}
    assert validated["weights"] == {"short": 3.0, "long": 1.0}
    assert validated["max_regression"] == {"short": 0.02, "long": 0.0}
    assert (validated["top_k"], validated["samples"], validated["attempts"]) == (2, 20, 8)
    assert validated["budget_seconds"] == 3600.0 and validated["timeout"] == 120.0


def test_defaults_are_bounded_and_the_budget_seconds_are_mandatory():
    minimal = {"schema_version": 1, "source": "m.onnx",
               "profiles": {"p": {"dimensions": {}, "weight": 1}}, "budget": {"seconds": 60}}
    validated = workload.validate(minimal)
    assert (validated["top_k"], validated["samples"], validated["screen_samples"], validated["timeout"]) == (3, 20, 5, 300.0)
    assert validated["attempts"] is None
    del minimal["budget"]["seconds"]
    with pytest.raises(ValueError, match=r"missing keys \['seconds'\]"):
        workload.validate(minimal)


@pytest.mark.parametrize("mutate, message", [
    (lambda w: w.update(schema_version=2), "unsupported workload schema_version"),
    (lambda w: w.update(schema_version=True), "unsupported workload schema_version"),
    (lambda w: w.update(extra=1), r"unknown keys \['extra'\]"),
    (lambda w: w.pop("source"), "missing keys"),
    (lambda w: w.update(backend="ep"), "backend must be one of standalone"),
    (lambda w: w.update(objective="throughput"), "objective must be"),
    (lambda w: w.update(profiles={}), "at least one profile"),
    (lambda w: w["profiles"].update({"bad name": {"dimensions": {}, "weight": 1}}), "profile names"),
    (lambda w: w["profiles"]["short"].update(weight=0), "weight must be > 0"),
    (lambda w: w["profiles"]["short"].update(weight=float("inf")), "weight must be a finite number"),
    (lambda w: w["profiles"]["short"].update(weight=True), "weight must be a finite number"),
    (lambda w: w["profiles"]["short"].update(max_regression=-0.1), "max_regression must be >= 0"),
    (lambda w: w["profiles"]["short"].update(colour="red"), "unknown keys"),
    (lambda w: w["profiles"]["short"]["dimensions"].update(batch=0), "dimension batch must be > 0"),
    (lambda w: w["profiles"]["short"]["dimensions"].update(batch=1.5), "dimension batch must be a finite integer"),
    (lambda w: w["budget"].update(seconds=0), "budget seconds must be > 0"),
    (lambda w: w["budget"].update(seconds=10 ** 9), "budget seconds must be > 0 and <="),
    (lambda w: w["budget"].update(attempts=0), "budget attempts must be >= 1"),
    (lambda w: w["budget"].update(attempts=10 ** 6), "budget attempts"),
    (lambda w: w["budget"].update(samples=19), "budget samples must be >= 20"),
    (lambda w: w["budget"].update(screen_samples=0), "screen_samples must be >= 1"),
    (lambda w: w["budget"].update(timeout=float("nan")), "timeout must be a finite number"),
    (lambda w: w["budget"].update(workers=2), "unknown keys"),
    (lambda w: w["tolerances"].update(atol=-1), "atol must be >= 0"),
    (lambda w: w["quantization"].update(grid=[]), "nonempty list of objects"),
    (lambda w: w["quantization"].pop("config"), "missing keys"),
    (lambda w: w.update(quality={"config": "q.json", "status": "quality_unmeasured"}), "exactly one of"),
    (lambda w: w.update(quality={}), "exactly one of"),
    (lambda w: w.update(quality={"status": "fine"}), "quality status must be"),
])
def test_invalid_workloads_are_rejected_before_any_work(mutate, message):
    raw = copy.deepcopy(RAW)
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        workload.validate(raw)


def test_load_names_the_file_and_resolves_relative_to_it(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(RAW))
    assert workload.load(path)["source"] == tmp_path / "model.onnx"
    path.write_text(json.dumps({**RAW, "budget": {"seconds": -1}}))
    with pytest.raises(ValueError, match="w.json: budget seconds"):
        workload.load(path)
    path.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        workload.load(path)


def test_the_cpu_workload_schema_is_not_accepted_here(tmp_path):
    cpu = {"schema_version": 1, "source": "model.onnx", "profiles": {"short": {"batch": 1}}, "samples": 20}
    with pytest.raises(ValueError):
        workload.validate(cpu)


def test_plan_reports_readiness_and_the_bounded_work(tmp_path):
    _files(tmp_path)
    plan = workload.plan(workload.validate(RAW, base=tmp_path))
    assert plan["ready"] is True and plan["missing"] == []
    assert plan["candidates"] == ["dynamic", "int4"]
    assert plan["profiles"]["short"]["weight"] == pytest.approx(0.75)
    work = plan["worst_case"]
    assert work["screening_candidates"] == 2 and work["screening_profile_runs"] == 4
    assert work["confirmations"] == 2 and work["attempts_needed"] == 4 and work["complete_within_attempts"] is True
    assert work["seconds_limit"] == 3600.0 and set(work["claims"]) == set(search.CLAIMS)


def test_plan_lists_every_missing_input_without_running_anything(tmp_path, monkeypatch):
    for name in ("prepare_bundle", "run_bundle", "optimize_bundle", "device_identity"):
        monkeypatch.setattr(search, name, lambda *a, **k: pytest.fail("the plan executed work"))
    plan = workload.plan(workload.validate(RAW, base=tmp_path))
    assert plan["ready"] is False
    assert any("source is not an existing .onnx file" in item for item in plan["missing"])
    assert any("adapter does not exist" in item for item in plan["missing"])
    assert any("quantization" in item and "quantization.json" in item for item in plan["missing"])


def test_plan_catches_a_quantization_grid_typo(tmp_path):
    _files(tmp_path)
    raw = _with(quantization={"config": "quantization.json", "grid": [{"mode": "int4", "block_size": 24}]})
    plan = workload.plan(workload.validate(raw, base=tmp_path))
    assert plan["ready"] is False and any("block_size" in item for item in plan["missing"])


def test_unmeasured_quality_blocks_lossy_candidates(tmp_path):
    _files(tmp_path)
    raw = _with(quality={"status": "quality_unmeasured"})
    plan = workload.plan(workload.validate(raw, base=tmp_path))
    assert plan["ready"] is False and plan["quality"]["status"] == "quality_unmeasured"
    assert any("quality_unmeasured" in item for item in plan["missing"])
    with pytest.raises(ValueError, match="quality_unmeasured"):
        workload.search_arguments(workload.validate(raw, base=tmp_path))
    # without a quantization section the workload is not blocked: nothing lossy is proposed
    raw.pop("quantization")
    assert workload.plan(workload.validate(raw, base=tmp_path))["ready"] is True


def test_plan_notes_when_attempts_cannot_cover_the_grid(tmp_path):
    _files(tmp_path)
    raw = _with(budget={"seconds": 60, "attempts": 2, "top_k": 2})
    plan = workload.plan(workload.validate(raw, base=tmp_path))
    assert plan["worst_case"]["complete_within_attempts"] is False
    assert "budget_exhausted" in plan["notes"][0]


def test_search_arguments_feed_the_search_exactly(tmp_path):
    _files(tmp_path)
    arguments = workload.search_arguments(workload.validate(RAW, base=tmp_path))
    assert arguments["base"]["metric"] == "m" and arguments["grid"] == [{"mode": "dynamic"}, {"mode": "int4"}]
    assert arguments["profile_weights"] == {"short": 3.0, "long": 1.0}
    assert arguments["max_profile_regression"] == {"short": 0.02, "long": 0.0}
    assert arguments["budget_seconds"] == 3600.0 and arguments["attempts"] == 8 and arguments["top_k"] == 2
    assert arguments["quality"] is None
    # every key is an argument of the real function
    import inspect

    accepted = set(inspect.signature(search.search_quantization).parameters)
    assert set(arguments) <= accepted


def test_the_tracked_example_is_a_valid_workload():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "python" / "examples" / "vulkan-workload.json"
    loaded = workload.load(example)
    assert loaded["quantization"] is not None and loaded["budget_seconds"] > 0
