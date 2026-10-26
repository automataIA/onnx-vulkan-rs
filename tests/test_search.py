import contextlib
import json
import pathlib
import subprocess
import sys
import time

import pytest

from onnx_vulkan_pipeline import policy, process, search
from onnx_vulkan_pipeline.pipeline import write_json

BASE = {"calibration": "cal", "evaluation": "ev", "adapter": "a.py", "metric": "m",
        "max_drop": 0.1, "higher_is_better": True, "format": "QOperator", "per_channel": True}
GRID = [{"mode": "static", "format": "QDQ"}, {"mode": "dynamic"}, {"mode": "int4"}]


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


# ----------------------------------------------------------------------------------------------
# Pure decisions
# ----------------------------------------------------------------------------------------------

def _outcome(status, baseline, candidate, *, confirmed=None):
    outcome = {"status": status, "baseline": {"median_wall_ms": baseline}, "candidate": {"median_wall_ms": candidate}}
    if confirmed:
        outcome["measurement"] = {"confirmation": {"baseline": confirmed[0], "candidate": confirmed[1]}}
    return outcome


def test_the_uninstrumented_confirmation_decides_the_ratio_over_the_timing_medians():
    verdict = search.judge_confirmation(
        {"a": _outcome("accelerated", 10.0, 5.0, confirmed=(10.0, 8.0))}, {"a": 1.0}, {"a": 0.0}, {"a": True})
    assert verdict["status"] == "promoted"
    assert verdict["profiles"]["a"]["ratio"] == pytest.approx(0.8)
    assert verdict["profiles"]["a"]["ratio_source"] == "confirmation"


def test_a_weighted_objective_cannot_hide_a_regressed_profile():
    outcomes = {"hot": _outcome("accelerated", 10.0, 2.0, confirmed=(10.0, 2.0)),
                "rare": _outcome("compatible_no_measured_gain", 10.0, 14.0)}
    weights = {"hot": 0.9, "rare": 0.1}
    verdict = search.judge_confirmation(outcomes, weights, {"hot": 0.0, "rare": 0.0}, {"hot": True, "rare": True})
    assert verdict["status"] == "guard_failed"
    assert "rare: +40.0% latency against a bound of 0.0%" in verdict["guard_failures"]
    allowed = search.judge_confirmation(outcomes, weights, {"hot": 0.0, "rare": 0.5}, {"hot": True, "rare": True})
    assert allowed["status"] == "promoted"
    # The profile that does not accelerate keeps the baseline model: ratio 1 in the objective.
    assert allowed["weighted_ratio"] == pytest.approx(0.9 * 0.2 + 0.1 * 1.0)


def test_an_accelerated_profile_without_vulkan_quality_is_not_promoted():
    verdict = search.judge_confirmation({"a": _outcome("accelerated", 10.0, 5.0, confirmed=(10.0, 5.0))},
                                        {"a": 1.0}, {"a": 0.0}, {"a": False})
    assert verdict["status"] == "guard_failed"
    assert "application quality of the deployed model was not verified on Vulkan" in verdict["guard_failures"][0]


def test_a_candidate_without_acceleration_is_no_gain_not_a_failure():
    verdict = search.judge_confirmation({"a": _outcome("compatible_no_measured_gain", 10.0, 10.0)},
                                        {"a": 1.0}, {"a": 0.0}, {"a": True})
    assert verdict["status"] == "no_gain"


def test_a_missing_measurement_is_a_failure_never_a_pass():
    verdict = search.judge_confirmation({"a": {"status": "accelerated"}}, {"a": 1.0}, {"a": 0.0}, {"a": True})
    assert verdict["status"] == "guard_failed"
    assert verdict["profiles"]["a"]["ratio"] is None


def test_pareto_front_keeps_tradeoffs_and_never_invents_missing_memory():
    items = [{"n": "fast", "ms": 1.0, "drop": 0.05, "mem": None},
             {"n": "accurate", "ms": 2.0, "drop": 0.0, "mem": None},
             {"n": "dominated", "ms": 3.0, "drop": 0.06, "mem": None},
             {"n": "small", "ms": 2.5, "drop": 0.06, "mem": 10}]
    front = search.pareto_front(items, lambda i: (i["ms"], i["drop"], i["mem"]))
    # `small` would be dominated by `accurate` on latency and quality if memory were ignored,
    # but memory was measured only for it, so only the two shared objectives are compared.
    assert [i["n"] for i in front] == ["fast", "accurate"]
    both = [{"n": "a", "ms": 2.0, "drop": 0.0, "mem": 10}, {"n": "b", "ms": 2.0, "drop": 0.0, "mem": 20}]
    assert [i["n"] for i in search.pareto_front(both, lambda i: (i["ms"], i["drop"], i["mem"]))] == ["a"]


# ----------------------------------------------------------------------------------------------
# The search with explicit fakes for what needs a device or the Rust binaries
# ----------------------------------------------------------------------------------------------

def _model(path, *, extra=0):
    """A real ONNX file, so that its content identity is the real one."""
    import onnx
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(np.full([1], float(extra), dtype=np.float32), "w")
    graph = helper.make_graph([helper.make_node("Add", ["x", "w"], ["y"])], "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])], [weight])
    onnx.save(helper.make_model(graph), str(path))


class World:
    """Candidate outcomes for the faked pipeline services; every call is recorded."""

    def __init__(self, root):
        self.root = root
        self.source = root / "m.onnx"
        _model(self.source)
        (root / "cal").mkdir()
        (root / "cal" / "0.npz").write_text("calibration")
        (root / "ev").mkdir()
        (root / "ev" / "0.npz").write_text("evaluation")
        (root / "a.py").write_text("def evaluate(session, samples): return 0\n")
        self.base = {**BASE, "calibration": str(root / "cal"), "evaluation": str(root / "ev"),
                     "adapter": str(root / "a.py")}
        self.walls = {}       # candidate -> {profile: [ms, ...]}
        self.drops = {}       # candidate -> Vulkan metric drop
        self.peaks = {}       # candidate -> storage peak bytes (absent: not reported)
        self.reject = {}      # candidate -> error raised by prepare
        self.outcomes = {}    # candidate -> {profile: optimize_bundle outcome}
        self.during_optimize = {}  # candidate -> callable run while its confirmation is produced
        self.calls = []
        self.device = {"name": "gpu0"}
        self.tools = {"model-runner": "t1"}

    def install(self, monkeypatch):
        monkeypatch.setattr(search, "prepare_bundle", self.prepare)
        monkeypatch.setattr(search, "run_bundle", self.run)
        monkeypatch.setattr(search, "optimize_bundle", self.optimize)
        monkeypatch.setattr(search, "quantized_op_count", lambda model: 1)
        monkeypatch.setattr(search, "device_identity", self.query_device)
        monkeypatch.setattr(search, "tool_identity", lambda: self.tools)
        monkeypatch.setattr(search, "gpu_lock", lambda timeout: contextlib.nullcontext())
        monkeypatch.setattr(search, "verify_bundle", self.verify)
        return self

    def query_device(self, timeout):
        self.calls.append(("device",))
        return self.device

    def prepare(self, source, bundle, *, quantization, profiles, **_):
        name = bundle.name
        self.calls.append(("prepare", name))
        if name in self.reject:
            raise self.reject[name]
        if callable(self.reject.get("*")):
            self.reject["*"](name)
        for profile in profiles:
            (bundle / profile).mkdir(parents=True)
            write_json(bundle / profile / "preparation.json", {"quantization": {"drop": 0.01}})
        (bundle / "mode").write_text(name)
        return self.verify(bundle, profiles)

    def verify(self, bundle, profiles=None):
        bundle = pathlib.Path(bundle)
        profiles = profiles or [p.name for p in bundle.iterdir() if p.is_dir()]
        # content-derived, like the real manifest identity: moving a bundle does not change it
        return {"identity": f"id-{(bundle / 'mode').read_text()}", "compatibility_obtained": True,
                "profiles": {p: {"model": f"{p}/model.onnx"} for p in profiles}}

    def run(self, bundle, *, profile, samples, **_):
        self.calls.append(("run", bundle.name, profile, samples))
        name = bundle.name
        report = {"standalone_ms": self.walls[name][profile],
                  "quality_evaluation": {"vulkan_verified": True, "drop": self.drops.get(name, 0.02)}}
        if name in self.peaks:
            report["standalone_storage_peak_bytes"] = self.peaks[name]
        return report

    def optimize(self, source, destination, *, samples, **_):
        name = destination.name
        self.calls.append(("optimize", name, samples))
        destination.mkdir()
        (destination / "mode").write_text(f"confirmed-{name}")  # what the fake `verify` seals, as the real manifest does
        if name in self.during_optimize:
            self.during_optimize[name]()
        write_json(destination / "report.json", self.outcomes[name])
        for profile in self.outcomes[name]:
            (destination / profile).mkdir()
            write_json(destination / profile / "validation.json", {"quality_evaluation": {"vulkan_verified": True}})
        return {"acceleration_demonstrated": True}

    def search(self, destination="out", **kwargs):
        kwargs.setdefault("grid", GRID)
        kwargs.setdefault("samples", 20)
        return search.search_quantization(self.source, self.root / destination, self.base, **kwargs)

    def count(self, kind):
        return sum(call[0] == kind for call in self.calls)


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path).install(monkeypatch)


def _walls(**by_name):
    return {name: {"default": [9.0, ms, ms]} for name, ms in by_name.items()}


def test_the_runner_up_wins_when_the_fastest_screen_fails_confirmation(world):
    world.reject["QDQ-static"] = ValueError("Quality regression 0.5 exceeds 0.1")
    world.walls = _walls(dynamic=5.0, int4=3.0)
    world.outcomes = {"int4": {"default": _outcome("compatible_no_measured_gain", 10.0, 10.0)},
                      "dynamic": {"default": _outcome("accelerated", 10.0, 5.0, confirmed=(10.0, 5.0))}}
    report = world.search(top_k=2)

    assert [item["status"] for item in report["candidates"]] == ["rejected", "passed", "passed"]
    assert "Quality regression" in report["candidates"][0]["reason"]
    # the first iteration is the warm-up and does not count
    assert report["candidates"][2]["profiles"]["default"]["median_wall_ms"] == 3.0
    assert report["ranking"] == ["int4", "dynamic"]
    assert report["attempted"] == ["int4", "dynamic"]
    assert report["confirmations"]["int4"]["status"] == "no_gain"
    assert report["confirmations"]["dynamic"]["status"] == "promoted"
    # attempted and promoted are separate fields
    assert report["promoted"] == report["selected"] == "dynamic"
    assert report["status"] == "promoted"
    assert (world.root / "out" / "selected" / "report.json").is_file()
    assert report["optimized"]["bundle"] == str(world.root / "out" / "selected")
    assert json.loads((world.root / "out" / "search.json").read_text()) == report


def test_no_confirmed_gain_promotes_nothing_and_says_so(world):
    world.walls = _walls(**{"QDQ-static": 4.0, "dynamic": 5.0, "int4": 3.0})
    world.outcomes = {name: {"default": _outcome("compatible_no_measured_gain", 10.0, 10.0)}
                      for name in ("QDQ-static", "dynamic", "int4")}
    report = world.search(top_k=2)
    assert report["status"] == "no_measured_gain" and report["selected"] is None and report["optimized"] is None
    assert len(report["attempted"]) == 2
    assert not (world.root / "out" / "selected").exists()


def test_a_guard_failure_blocks_promotion_even_with_a_big_gain(world):
    profiles = {"hot": {}, "rare": {}}
    world.walls = {"int4": {"hot": [9.0, 2.0], "rare": [9.0, 2.0]}}
    world.outcomes = {"int4": {"hot": _outcome("accelerated", 10.0, 2.0, confirmed=(10.0, 2.0)),
                               "rare": _outcome("compatible_no_measured_gain", 10.0, 14.0)}}
    report = world.search(grid=[{"mode": "int4"}], profiles=profiles, profile_weights={"hot": 9, "rare": 1})
    assert report["confirmations"]["int4"]["status"] == "guard_failed"
    assert report["status"] == "no_measured_gain" and report["selected"] is None
    again = world.search("out2", grid=[{"mode": "int4"}], profiles=profiles, profile_weights={"hot": 9, "rare": 1},
                         max_profile_regression={"hot": 0.0, "rare": 0.5})
    assert again["status"] == "promoted"


def test_screening_keeps_quality_and_memory_tradeoffs_on_the_shortlist(world):
    world.walls = _walls(**{"QDQ-static": 4.0, "dynamic": 3.0, "int4": 6.0})
    world.drops = {"QDQ-static": 0.00, "dynamic": 0.08, "int4": 0.05}
    world.peaks = {"int4": 100}
    world.outcomes = {name: {"default": _outcome("compatible_no_measured_gain", 10.0, 10.0)} for name in world.walls}
    report = world.search(top_k=2)
    # int4 is slower and less accurate than QDQ-static: dominated on the shared objectives, and
    # it alone reports memory, which is never compared with a missing value.
    assert report["ranking"] == ["dynamic", "QDQ-static", "int4"]
    assert report["pareto"] == ["dynamic", "QDQ-static", "int4"] or report["pareto"] == ["dynamic", "QDQ-static"]
    assert report["shortlist"] == ["dynamic", "QDQ-static"]
    assert report["candidates"][2]["storage_peak_bytes"] == 100
    assert report["candidates"][0]["storage_peak_bytes"] is None


def test_screening_needs_few_iterations_but_confirmation_needs_twenty(world):
    world.walls = _walls(int4=3.0)
    world.outcomes = {"int4": {"default": _outcome("accelerated", 10.0, 5.0, confirmed=(10.0, 5.0))}}
    report = world.search(grid=[{"mode": "int4"}], screen_samples=2, samples=20)
    assert ("run", "int4", "default", 2) in world.calls and ("optimize", "int4", 20) in world.calls
    assert report["status"] == "promoted"
    for bad in ({"samples": 19}, {"screen_samples": 0}, {"top_k": 0}, {"attempts": 0}):
        with pytest.raises(ValueError):
            world.search("bad", grid=[{"mode": "int4"}], **bad)


def test_invalid_specifications_do_no_costly_work(world, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("costly work for an invalid specification")

    for name in ("prepare_bundle", "run_bundle", "optimize_bundle", "device_identity", "tool_identity"):
        monkeypatch.setattr(search, name, boom)
    bad = [{"budget_seconds": 0}, {"budget_seconds": float("nan")}, {"timeout": -1}, {"top_k": True},
           {"profile_weights": {"default": -1}}, {"profile_weights": {"other": 1}},
           {"max_profile_regression": {"nope": 0.0}}, {"max_profile_regression": -0.1},
           {"profiles": {"bad name": {}}}, {"profiles": {"p": {"batch": 0}}},
           {"grid": [{"mode": "static", "format": "BAD"}]}, {"grid": [{"mode": "dynamic", "block_size": 32}]},
           {"atol": float("inf")}]
    for kwargs in bad:
        with pytest.raises(ValueError):
            world.search("bad", **kwargs)
    with pytest.raises(ValueError, match="source does not exist"):
        search.search_quantization(world.root / "missing.onnx", world.root / "bad", world.base, grid=GRID)
    assert not (world.root / "bad").exists()
    assert world.calls == []


def test_an_existing_finished_destination_is_never_reused(world):
    world.walls = _walls(int4=3.0)
    world.outcomes = {"int4": {"default": _outcome("accelerated", 10.0, 5.0, confirmed=(10.0, 5.0))}}
    world.search(grid=[{"mode": "int4"}])
    with pytest.raises(FileExistsError):
        world.search(grid=[{"mode": "int4"}])
    (world.root / "other").mkdir()
    with pytest.raises(FileExistsError):
        world.search("other", grid=[{"mode": "int4"}])


def test_a_candidate_that_quantizes_nothing_is_not_measured(world, monkeypatch):
    def prepare(source, bundle, **_):
        (bundle / "default").mkdir(parents=True)
        write_json(bundle / "default" / "preparation.json",
                   {"quantization": {"drop": 0.0, "options": {"op_types_to_quantize": ["MatMul"]}}})
        return {"identity": "x", "profiles": {"default": {"model": "default/model.onnx"}},
                "compatibility_obtained": True}

    monkeypatch.setattr(search, "prepare_bundle", prepare)
    monkeypatch.setattr(search, "quantized_op_count", lambda model: 0)
    monkeypatch.setattr(search, "run_bundle", lambda *a, **k: pytest.fail("measured an unquantized model"))
    monkeypatch.setattr(search, "device_identity", lambda timeout: pytest.fail("queried the device"))
    report = world.search(grid=[{"mode": "int4"}])
    assert report["candidates"][0]["reason"] == ("no operator was quantized: the model has no MatMul node this mode "
                                                  "quantizes, or all were excluded")
    assert report["status"] == "no_passing_candidate" and report["selected"] is None
    # Rejected before any GPU work: no device was needed, so none is recorded.
    assert report["device"] is None


def test_a_device_is_recorded_only_once_it_is_measured(world):
    world.walls = _walls(int4=3.0)
    world.outcomes = {"int4": {"default": _outcome("accelerated", 10.0, 5.0, confirmed=(10.0, 5.0))}}
    report = world.search(grid=[{"mode": "int4"}])
    assert report["device"] == {"name": "gpu0"}
    assert json.loads((world.root / "out" / "resume.json").read_text())["device"] == {"name": "gpu0"}


# ----------------------------------------------------------------------------------------------
# Budget, attempts and resume
# ----------------------------------------------------------------------------------------------

def test_the_global_deadline_ends_a_nested_worker_and_reports_budget_exhausted(world):
    def sleepy(name):
        process.command([sys.executable, "-c", "import time; time.sleep(30)"], timeout=60)

    world.reject["*"] = sleepy
    started = time.monotonic()
    report = world.search(budget_seconds=1.0)
    assert time.monotonic() - started < 20
    assert report["status"] == "budget_exhausted" and report["exhausted"] == "seconds"
    assert report["resume"] == str(world.root / "out")
    assert report["selected"] is None
    assert json.loads((world.root / "out" / "search.json").read_text())["status"] == "budget_exhausted"
    # not a verdict on the candidate
    assert report["candidates"][0]["status"] == "screening"


def test_an_ordinary_worker_timeout_is_a_rejection_not_budget_exhaustion(world):
    world.reject["*"] = lambda name: process.command([sys.executable, "-c", "import time; time.sleep(30)"],
                                                     timeout=0.3)
    report = world.search(grid=[{"mode": "int4"}], budget_seconds=60)
    assert report["status"] == "no_passing_candidate"
    assert report["candidates"][0]["status"] == "rejected" and report["candidates"][0]["timed_out"]


def test_the_attempt_count_bounds_an_invocation_and_a_resume_reuses_completed_work(world):
    world.walls = _walls(**{"QDQ-static": 4.0, "dynamic": 5.0, "int4": 3.0})
    world.outcomes = {name: {"default": _outcome("compatible_no_measured_gain", 10.0, 10.0)}
                      for name in world.walls}
    world.outcomes["int4"] = {"default": _outcome("accelerated", 10.0, 5.0, confirmed=(10.0, 5.0))}
    first = world.search(attempts=2, top_k=1)
    assert first["status"] == "budget_exhausted" and first["exhausted"] == "attempts"
    assert first["unscreened"] == ["int4"]
    assert first["budget"]["attempts_used"] == 2 and world.count("prepare") == 2

    second = world.search(attempts=3, top_k=1)
    assert second["status"] == "promoted" and second["selected"] == "int4"
    # only the missing candidate was prepared; the first two were reused
    assert world.count("prepare") == 3 and world.count("run") == 3
    assert second["budget"]["attempts_used"] == 2 + 2 and second["budget"]["invocations"] == 2


def test_attempts_leave_room_to_stop_between_screening_and_confirmation(world):
    world.walls = _walls(**{"QDQ-static": 4.0, "dynamic": 5.0, "int4": 3.0})
    first = world.search(attempts=3)
    assert first["status"] == "budget_exhausted" and "confirmation" in first["exhausted_detail"]
    assert first["shortlist"] and not first["attempted"] and world.count("optimize") == 0


def test_a_timed_out_candidate_is_measured_again_on_resume(world):
    world.walls = _walls(**{"QDQ-static": 4.0, "dynamic": 5.0, "int4": 3.0})
    world.outcomes = {name: {"default": _outcome("compatible_no_measured_gain", 10.0, 10.0)} for name in world.walls}
    world.reject["QDQ-static"] = subprocess.TimeoutExpired(["worker"], 5)
    first = world.search(attempts=1)
    assert first["candidates"][0]["timed_out"] and first["status"] == "budget_exhausted"
    del world.reject["QDQ-static"]
    second = world.search(attempts=10)
    assert second["candidates"][0]["status"] == "passed" and "timed_out" not in second["candidates"][0]


def _interrupted(world):
    world.walls = _walls(**{"QDQ-static": 4.0, "dynamic": 5.0, "int4": 3.0})
    world.outcomes = {name: {"default": _outcome("accelerated", 10.0, 5.0, confirmed=(10.0, 5.0))}
                      for name in world.walls}
    assert world.search(attempts=1)["status"] == "budget_exhausted"


def _change_model(world):
    _model(world.source, extra=1)


def _change_dataset(world):
    (world.root / "ev" / "0.npz").write_text("different evaluation data")


def _change_calibration(world):
    (world.root / "cal" / "1.npz").write_text("another calibration sample")


def _change_adapter(world):
    (world.root / "a.py").write_text("def evaluate(session, samples): return 1\n")


def _change_policy(world):
    world.monkeypatch.setenv("ONNX_VULKAN_ATTENTION_FP16", "1")


def _change_tools(world):
    world.tools = {"model-runner": "t2"}


def _change_option(world):
    world.base["max_drop"] = 0.2


@pytest.mark.parametrize("change", [_change_model, _change_dataset, _change_calibration, _change_adapter,
                                    _change_policy, _change_tools, _change_option])
def test_a_resume_rejects_stale_evidence_and_keeps_it_untouched(world, monkeypatch, change):
    world.monkeypatch = monkeypatch
    monkeypatch.delenv("ONNX_VULKAN_ATTENTION_FP16", raising=False)
    _interrupted(world)
    before = (world.root / "out" / "search.json").read_text()
    prepared = world.count("prepare")
    change(world)
    with pytest.raises(search.StaleExperiment, match="Stale search"):
        world.search(attempts=10)
    assert world.count("prepare") == prepared
    assert (world.root / "out" / "search.json").read_text() == before


def test_a_resume_rejects_external_data_that_changed(world):
    import onnx

    world.walls = _walls(dynamic=4.0, int4=3.0)
    world.outcomes = {name: {"default": _outcome("accelerated", 10.0, 5.0, confirmed=(10.0, 5.0))}
                      for name in ("dynamic", "int4")}
    _model(world.source, extra=3)
    model = onnx.load(str(world.source))
    onnx.save_model(model, str(world.source), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="weights.bin", size_threshold=0)
    assert (world.root / "weights.bin").is_file()
    assert world.search(grid=[{"mode": "dynamic"}, {"mode": "int4"}], attempts=1)["status"] == "budget_exhausted"
    (world.root / "weights.bin").write_bytes(b"\x00" * (world.root / "weights.bin").stat().st_size)
    with pytest.raises(search.StaleExperiment, match="external_data"):
        world.search(grid=[{"mode": "dynamic"}, {"mode": "int4"}], attempts=5)


def test_a_resume_rejects_another_device_and_a_changed_bundle(world):
    _interrupted(world)
    # the interrupted invocation measured one candidate, so the device is bound
    assert json.loads((world.root / "out" / "resume.json").read_text())["device"] == {"name": "gpu0"}
    world.device = {"name": "gpu1"}
    with pytest.raises(search.StaleExperiment, match="measured on .*gpu0.*gpu1"):
        world.search(attempts=10)
    world.device = {"name": "gpu0"}
    (world.root / "out" / "candidates" / "QDQ-static" / "mode").write_text("tampered")
    with pytest.raises(search.StaleExperiment, match="bundle of candidate QDQ-static changed"):
        world.search(attempts=10)


def test_changing_only_the_invocation_limits_resumes(world):
    _interrupted(world)
    report = world.search(attempts=20, budget_seconds=600, top_k=1, timeout=120)
    assert report["status"] == "promoted"


def test_identity_changes_name_what_differs():
    old = {"source": {"sha256": "a" * 64}, "tune": True}
    new = {"source": {"sha256": "b" * 64}, "tune": True}
    assert search.identity_changes(old, new) == [f"source.sha256: {'a' * 24!r} → {'b' * 24!r}"]


def test_the_policy_is_part_of_the_identity(world, monkeypatch):
    first = search._identity(world.source, None, world.base, [], {}, {}, {}, 1e-5, 1e-4, 20, 5, True, None)
    monkeypatch.setenv("ONNX_VULKAN_ATTENTION_FP16", "1")
    second = search._identity(world.source, None, world.base, [], {}, {}, {}, 1e-5, 1e-4, 20, 5, True, None)
    assert policy.differences(first["execution_policy"], second["execution_policy"]) == [
        "numerics.attention_fp16: False → True"]


def test_duplicate_or_empty_grids_are_rejected(world):
    with pytest.raises(ValueError, match="duplicate"):
        world.search("a", grid=[{"mode": "int4"}, {"mode": "int4"}])
    with pytest.raises(ValueError, match="nonempty"):
        world.search("b", grid=[])


# ----------------------------------------------------------------------------------------------
# Completed evidence is reused only while its sealed artifacts are untouched
# ----------------------------------------------------------------------------------------------

def test_a_mutated_completed_candidate_stops_the_resume_before_any_work(world):
    world.walls = _walls(**{"QDQ-static": 4.0, "dynamic": 5.0, "int4": 3.0})
    world.outcomes = {name: {"default": _outcome("accelerated", 10.0, 5.0, confirmed=(10.0, 5.0))}
                      for name in world.walls}
    first = world.search(attempts=2, top_k=1)
    assert first["status"] == "budget_exhausted" and first["unscreened"] == ["int4"]
    bundle = world.root / "out" / "candidates" / "QDQ-static"
    (bundle / "mode").write_text("tampered")
    before = len(world.calls)

    with pytest.raises(search.StaleExperiment, match="bundle of candidate QDQ-static changed"):
        world.search(attempts=5, top_k=1)

    # no worker, device query or preparation began, nothing was promoted, the evidence is kept
    assert world.calls[before:] == []
    assert not (world.root / "out" / "selected").exists()
    assert (bundle / "mode").read_text() == "tampered"
    assert json.loads((world.root / "out" / "search.json").read_text())["status"] != "promoted"


def test_a_deleted_completed_candidate_is_stale_not_a_crash(world):
    import shutil

    world.walls = _walls(**{"QDQ-static": 4.0, "dynamic": 5.0, "int4": 3.0})
    world.search(attempts=2, top_k=1)
    shutil.rmtree(world.root / "out" / "candidates" / "dynamic")
    before = len(world.calls)
    with pytest.raises(search.StaleExperiment, match="candidate dynamic .* cannot be verified"):
        world.search(attempts=5, top_k=1)
    assert world.calls[before:] == []


def test_a_mutated_completed_confirmation_stops_the_resume_before_any_work(world):
    world.walls = _walls(**{"QDQ-static": 4.0, "dynamic": 5.0, "int4": 3.0})
    world.outcomes = {name: {"default": _outcome("accelerated", 10.0, 5.0, confirmed=(10.0, 5.0))}
                      for name in world.walls}
    # three screenings and one confirmation: the budget ends before the second shortlisted candidate
    first = world.search(attempts=4, top_k=2)
    assert first["status"] == "budget_exhausted" and first["attempted"] == ["int4"]
    confirmed = world.root / "out" / "confirmations" / "int4"
    assert first["confirmations"]["int4"]["bundle_identity"] == "id-confirmed-int4"
    (confirmed / "mode").write_text("tampered")
    before = len(world.calls)

    with pytest.raises(search.StaleExperiment, match="confirmed bundle of candidate int4 changed"):
        world.search(attempts=5, top_k=2)

    assert world.calls[before:] == []
    assert not (world.root / "out" / "selected").exists()
    assert (confirmed / "mode").read_text() == "tampered"


def test_a_confirmation_mutated_before_promotion_is_never_promoted(world):
    world.walls = _walls(**{"QDQ-static": 6.0, "dynamic": 5.0, "int4": 3.0})
    world.outcomes = {"int4": {"default": _outcome("accelerated", 10.0, 5.0, confirmed=(10.0, 5.0))},
                      "dynamic": {"default": _outcome("accelerated", 10.0, 8.0, confirmed=(10.0, 8.0))}}
    confirmed = world.root / "out" / "confirmations" / "int4"
    # the winner's bundle changes while the runner-up is being confirmed
    world.during_optimize["dynamic"] = lambda: (confirmed / "mode").write_text("tampered")
    with pytest.raises(search.StaleExperiment, match="confirmed bundle of candidate int4 changed"):
        world.search(top_k=2)
    assert not (world.root / "out" / "selected").exists()
    assert confirmed.exists()
    assert json.loads((world.root / "out" / "search.json").read_text())["status"] != "promoted"


def test_an_untouched_resume_still_verifies_and_promotes(world):
    world.walls = _walls(**{"QDQ-static": 6.0, "dynamic": 5.0, "int4": 3.0})
    world.outcomes = {name: {"default": _outcome("accelerated", 10.0, 5.0, confirmed=(10.0, 5.0))}
                      for name in ("int4", "dynamic")}
    world.search(attempts=4, top_k=2)
    second = world.search(attempts=5, top_k=2)
    assert second["status"] == "promoted"
    # moving the winner to selected/ does not change its sealed identity
    assert (world.root / "out" / "selected" / "mode").read_text() == f"confirmed-{second['selected']}"


@pytest.mark.parametrize("section", ["candidates", "confirmations"])
def test_completed_legacy_evidence_without_identity_is_refused(world, section):
    world.walls = _walls(**{"QDQ-static": 6.0, "dynamic": 5.0, "int4": 3.0})
    world.outcomes = {name: {"default": _outcome("accelerated", 10.0, 5.0, confirmed=(10.0, 5.0))}
                      for name in ("int4", "dynamic")}
    world.search(attempts=4, top_k=2)
    path = world.root / "out" / "search.json"
    report = json.loads(path.read_text())
    entry = report["candidates"][0] if section == "candidates" else report["confirmations"]["int4"]
    del entry["bundle_identity"]
    path.write_text(json.dumps(report))
    before = len(world.calls)
    with pytest.raises(search.StaleExperiment, match="no sealed bundle identity"):
        world.search(attempts=5, top_k=2)
    assert world.calls[before:] == []
    assert not (world.root / "out" / "selected").exists()
