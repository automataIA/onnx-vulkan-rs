import json

import pytest

from onnx_vulkan_pipeline import search
from onnx_vulkan_pipeline.pipeline import write_json

BASE = {"calibration": "cal", "evaluation": "ev", "adapter": "a.py", "metric": "m",
        "max_drop": 0.1, "higher_is_better": True, "format": "QOperator", "per_channel": True}


def test_default_grid_needs_calibration_for_static_forms():
    without = {key: value for key, value in BASE.items() if key != "calibration"}
    assert {item["mode"] for item in search.default_grid(BASE)} == {"static", "dynamic", "int4"}
    assert {item["mode"] for item in search.default_grid(without)} == {"dynamic", "int4"}


def test_candidate_config_drops_keys_the_mode_rejects():
    assert search.candidate_config(BASE, {"mode": "int4", "block_size": 32}) == {
        "evaluation": "ev", "adapter": "a.py", "metric": "m", "max_drop": 0.1,
        "higher_is_better": True, "mode": "int4", "block_size": 32}
    dynamic = search.candidate_config(BASE, {"mode": "dynamic"})
    assert dynamic["per_channel"] is True and "calibration" not in dynamic and "format" not in dynamic
    with pytest.raises(ValueError, match="mode must be"):
        search.candidate_config(BASE, {"mode": "fp8"})


def test_candidate_names_are_stable_directory_names():
    assert search.candidate_name({"mode": "static", "format": "QDQ", "per_channel": False}) == "QDQ-static-no_per_channel"
    assert search.candidate_name({"mode": "int4", "block_size": 32}) == "block_size32-int4"
    with pytest.raises(ValueError):
        search.candidate_name({"nodes_to_exclude": ["a"]})


def test_search_ranks_passing_candidates_and_optimizes_the_fastest(tmp_path, monkeypatch):
    walls = {"dynamic": [9.0, 5.0, 5.0], "int4": [9.0, 3.0, 3.0]}

    def prepare(source, bundle, *, quantization, **_):
        mode = quantization["mode"]
        if mode == "static":
            raise ValueError("Quality regression 0.5 exceeds 0.1")
        (bundle / "default").mkdir(parents=True)
        write_json(bundle / "default" / "preparation.json", {"quantization": {"drop": 0.01}})
        (bundle / "mode").write_text(mode)
        return {"profiles": {"default": {"model": "default/model.onnx"}}, "compatibility_obtained": True}

    def run(bundle, **_):
        mode = (bundle / "mode").read_text()
        return {"standalone_ms": walls[mode], "quality_evaluation": {"vulkan_verified": True, "drop": 0.02}}

    optimized = []

    def optimize(source, destination, **_):
        optimized.append(source)
        destination.mkdir()
        write_json(destination / "report.json", {"default": {"status": "accelerated"}})
        return {"acceleration_demonstrated": True}

    monkeypatch.setattr(search, "prepare_bundle", prepare)
    monkeypatch.setattr(search, "run_bundle", run)
    monkeypatch.setattr(search, "optimize_bundle", optimize)
    monkeypatch.setattr(search, "quantized_op_count", lambda model: 1)
    grid = [{"mode": "static", "format": "QDQ"}, {"mode": "dynamic"}, {"mode": "int4"}]
    report = search.search_quantization("m.onnx", tmp_path / "out", BASE, grid=grid)

    assert [item["status"] for item in report["candidates"]] == ["rejected", "passed", "passed"]
    assert "Quality regression" in report["candidates"][0]["reason"]
    # the first iteration is the warm-up and does not count
    assert report["candidates"][2]["median_wall_ms"] == 3.0
    assert report["selected"] == "int4"
    assert optimized == [str(tmp_path / "out" / "candidates" / "int4")]
    assert report["optimized"]["acceleration_demonstrated"] is True
    assert json.loads((tmp_path / "out" / "search.json").read_text()) == report


def test_a_candidate_that_quantizes_nothing_is_not_measured(tmp_path, monkeypatch):
    def prepare(source, bundle, **_):
        (bundle / "default").mkdir(parents=True)
        write_json(bundle / "default" / "preparation.json",
                   {"quantization": {"drop": 0.0, "options": {"op_types_to_quantize": ["MatMul"]}}})
        return {"profiles": {"default": {"model": "default/model.onnx"}}, "compatibility_obtained": True}

    monkeypatch.setattr(search, "prepare_bundle", prepare)
    monkeypatch.setattr(search, "quantized_op_count", lambda model: 0)
    monkeypatch.setattr(search, "run_bundle", lambda *a, **k: pytest.fail("measured an unquantized model"))
    report = search.search_quantization("m.onnx", tmp_path / "out", BASE, grid=[{"mode": "int4"}])
    assert report["candidates"][0]["reason"] == ("no operator was quantized: the model has no MatMul node this mode "
                                                  "quantizes, or all were excluded")
    assert report["selected"] is None


def test_the_reason_is_the_exception_line_not_trailing_logging():
    worker = RuntimeError("python exited 1: WARNING:root:pre-processing\nTraceback (most recent call last):\n"
                          "  File \"quantize.py\", line 408, in quantize_model\n"
                          "ValueError: Quality regression 0.5 exceeds 0.44\n\n"
                          "Number of histogram bins : 2048\nPercentile : (0.001,99.999)\n")
    assert search.failure_reason(worker) == "ValueError: Quality regression 0.5 exceeds 0.44"
    runner = RuntimeError("model-runner exited 1: Error: divergent outputs beyond atol=1.0e-5\n"
                          "  reference standalone output  max|Δ|=8.9e-1  beyond tolerance: 999")
    assert search.failure_reason(runner) == "model-runner exited 1: Error: divergent outputs beyond atol=1.0e-5"
    assert search.failure_reason(OSError("disk full")) == "disk full"
    divergence = RuntimeError("Error: divergent outputs beyond atol=1.0e-5\n"
                              "  output boxes   n=8  max|Δ|=2.0e-5  |ref|max=1.0e0  relative=2.00e-5  beyond tolerance: 1\n"
                              "  output scores  n=8  max|Δ|=1.598e1  |ref|max=6.398e2  relative=2.50e-2  beyond tolerance: 7\n"
                              "  standalone vs CPU  scores  n=8  max|Δ|=1.598e1  relative=9.00e-1  beyond tolerance: 7")
    assert search.divergence_detail(divergence) == "; worst output scores: max|Δ|=1.598e1, relative=2.50e-2"
    assert search.divergence_detail(OSError("disk full")) == ""
    assert search.failure_reason(ValueError()) == "ValueError"


def test_duplicate_or_empty_grids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        search.search_quantization("m.onnx", tmp_path / "a", BASE, grid=[{"mode": "int4"}, {"mode": "int4"}])
    with pytest.raises(ValueError, match="nonempty"):
        search.search_quantization("m.onnx", tmp_path / "b", BASE, grid=[])
