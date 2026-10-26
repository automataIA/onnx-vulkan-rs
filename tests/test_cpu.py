"""CPU optimization: real tiny-model runs, plus injected clocks and workers for the bounds."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

onnx = pytest.importorskip("onnx")
numpy = pytest.importorskip("numpy")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

import onnx_vulkan_pipeline.cpu as cpu  # noqa: E402
import onnx_vulkan_pipeline._cpu_worker as worker  # noqa: E402

LEVEL_NAMES = {name: short for short, name in cpu.LEVELS.items()}


def write_model(path: Path, scale: float = 2.0, offset: float = 0.0) -> Path:
    """y = relu(x @ w) * scale + offset on a fixed [1, 8] input."""
    weights = numpy.linspace(-1, 1, 64, dtype=numpy.float32).reshape(8, 8)
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["m"]),
        helper.make_node("Relu", ["m"], ["r"]),
        helper.make_node("Mul", ["r", "s"], ["t"]),
        helper.make_node("Add", ["t", "o"], ["y"]),
    ]
    inits = [numpy_helper.from_array(weights, "w"),
             numpy_helper.from_array(numpy.array(scale, dtype=numpy.float32), "s"),
             numpy_helper.from_array(numpy.array(offset, dtype=numpy.float32), "o")]
    graph = helper.make_graph(
        nodes, "tiny", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.save(model, str(path))
    return path


def candidate_id(settings) -> str:
    return f"{settings['model']}-{LEVEL_NAMES[settings['graph_optimization_level']]}-t{settings['intra_op_num_threads']}"


def timings(samples: int, ratio: float = 1.0, *, correct: bool = True, failure: str | None = None,
            baseline_correct: bool = True) -> dict:
    return {"baseline_ms": [10.0] * samples, "candidate_ms": [10.0 * ratio] * samples,
            "orders": ["AB", "BA"] * (samples // 2), "setup_ms": {"baseline": 1.0, "candidate": 1.0},
            "correct": correct, "baseline_correct": baseline_correct, "failure": failure,
            "max_abs": 0.0, "max_rel": 0.0, "ort_version": "fake"}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeWorker:
    """Stands in for the measurement subprocess: records calls, answers from `behave`."""

    def __init__(self, behave=None, clock: Clock | None = None, step: float = 0.0):
        self.behave = behave or (lambda call: timings(20))
        self.clock, self.step = clock, step
        self.calls: list[dict] = []

    def __call__(self, argv, timeout):
        config = json.loads(Path(argv[-1]).read_text())
        assert config["mode"] == "measure", config["mode"]
        assert "model-runner" not in " ".join(map(str, argv))
        call = {"id": candidate_id(config["candidate"]), "profile": Path(config["directory"]).name,
                "timeout": timeout, "config": config}
        call["count"] = sum(c["id"] == call["id"] and c["profile"] == call["profile"] for c in self.calls) + 1
        self.calls.append(call)
        if self.clock:
            self.clock.now += self.step
        result = self.behave(call)
        Path(config["output"]).write_text(json.dumps(result))
        return subprocess.CompletedProcess(argv, 0, "", "")

    def ids(self):
        return [c["id"] for c in self.calls]


@pytest.fixture
def lab(monkeypatch, tmp_path):
    """Four usable CPUs and a preparation that copies the source instead of launching workers."""
    monkeypatch.setattr(cpu, "usable_cpus", lambda: [0, 1, 2, 3])

    def prepare(budget, staging, name, dims, source, adapter, atol, rtol):
        prepare.calls.append(name)
        directory = staging / "bundle" / name
        shutil.rmtree(directory, ignore_errors=True)  # like the real one, an unrecorded directory is rebuilt
        (directory / "reference").mkdir(parents=True)
        shutil.copy(source, directory / "original.onnx")
        (directory / "model.onnx").write_bytes(source.read_bytes() + b"\0")
        (directory / "reference/input_0.pb").write_bytes(b"input")
        (directory / "feeds.npz").write_bytes(b"feeds")
        (directory / "oracle.npz").write_bytes(b"outputs")
        (directory / "oracle.json").write_text("{}")

    prepare.calls = []
    monkeypatch.setattr(cpu, "_prepare_profile", prepare)
    return write_model(tmp_path / "model.onnx")


def run(lab, tmp_path, fake, **kwargs):
    kwargs.setdefault("threads", [1, 2])
    return cpu.optimize_cpu(lab, tmp_path / "out", runner=fake, **kwargs)


def test_thread_grid_is_bounded_by_usable_cpus():
    assert cpu.thread_grid(12) == [1, 2, 4, 8, 12]
    assert cpu.thread_grid(12, 5) == [1, 2, 4, 5]
    assert cpu.thread_grid(4, [3, 1, 3]) == [1, 3]
    assert cpu.thread_grid(1) == [1]
    for bad in (5, 0, [], [1, 5], True, [2.0]):
        with pytest.raises(ValueError):
            cpu.thread_grid(4, bad)


def test_judge_requires_gain_consistency_and_no_regression():
    def verdict(*ratios_by_profile, correct=True):
        results = {f"p{i}": timings(20, ratio, correct=correct) for i, ratio in enumerate(ratios_by_profile)}
        return cpu.judge(results, samples=20, min_gain=0.10, regression_limit=0.02)

    assert verdict(0.8)[0] == "eligible"
    assert verdict(1.0)[0] == "no_measured_gain"
    assert verdict(0.5, 1.1)[0] == "rejected_regression"  # a fast profile cannot pay for a slow one
    assert verdict(0.5, 1.01)[0] == "eligible"
    assert verdict(0.5, correct=False)[0] == "rejected_divergence"
    mostly_slower = timings(20) | {"candidate_ms": [8.0] * 11 + [50.0] * 9}  # median gain, but not consistent
    assert cpu.judge({"p": mostly_slower}, samples=20, min_gain=0.10, regression_limit=0.02)[0] == "no_measured_gain"
    short = timings(5, 0.5)
    assert cpu.judge({"p": short}, samples=20, min_gain=0.10, regression_limit=0.02)[0] == "invalid"


def test_selects_confirmed_candidate_and_writes_deployable_settings(lab, tmp_path):
    fake = FakeWorker(lambda call: timings(20, 0.5 if call["id"] == "prepared-all-t2" else 1.0))
    report = run(lab, tmp_path, fake)

    assert report["status"] == "optimized" and report["selected"] == "prepared-all-t2"
    out = tmp_path / "out"
    session = json.loads((out / "session.json").read_text())
    assert session["providers"] == ["CPUExecutionProvider"]
    assert session["session_options"] == {"graph_optimization_level": "ORT_ENABLE_ALL",
                                          "execution_mode": "ORT_SEQUENTIAL", "intra_op_num_threads": 2}
    assert session["models"] == {"default": "default/model.onnx"}
    assert (out / session["models"]["default"]).is_file() and (out / "default/original.onnx").is_file()
    stored = json.loads((out / "cpu-optimization.json").read_text())
    assert stored["policy_identity"] == session["policy_identity"] == report["policy_identity"]
    assert stored["baseline"] == cpu.BASELINE  # original graph, ENABLE_ALL, ORT default threads
    assert stored["baseline"]["graph_optimization_level"] == "ORT_ENABLE_ALL"
    assert stored["baseline"]["intra_op_num_threads"] == 0
    assert stored["claims"]["application_quality"] == "not measured"
    assert stored["confirmation"]["verdict"] == "eligible"
    assert set(stored["artifacts"]["default"]) == {"original.onnx", "model.onnx", "oracle.json", "feeds.npz",
                                                    "oracle.npz", "reference/input_0.pb"}
    # 12 candidates (2 models x 3 levels x 2 thread counts) plus one confirmation.
    assert len(fake.calls) == 13 and report["budget"]["attempts_used"] == 13
    assert all(c["config"]["baseline"] == cpu.BASELINE and c["config"]["samples"] == 20 for c in fake.calls)
    assert not (tmp_path / ".out.incomplete").exists()


def test_no_gain_keeps_baseline_settings(lab, tmp_path):
    report = run(lab, tmp_path, FakeWorker())
    assert report["status"] == "no_measured_gain" and report["selected"] is None
    session = json.loads((tmp_path / "out/session.json").read_text())
    assert session["session_options"] == {"graph_optimization_level": "ORT_ENABLE_ALL",
                                          "execution_mode": "ORT_SEQUENTIAL", "intra_op_num_threads": 0}
    assert session["models"] == {"default": "default/original.onnx"}


def test_regression_in_any_profile_rejects_candidate(lab, tmp_path):
    def behave(call):
        slow = call["profile"] == "long" and call["id"] == "original-all-t1"
        return timings(20, 1.2 if slow else 0.5)

    fake = FakeWorker(behave)
    report = run(lab, tmp_path, fake, profiles={"short": {}, "long": {}}, threads=[1])
    assert report["ledger"]["candidates"]["original-all-t1"]["verdict"] == "rejected_regression"
    assert report["ledger"]["candidates"]["prepared-all-t1"]["verdict"] == "eligible"
    assert report["selected"] == "prepared-all-t1"  # first of the equal gains


def test_unconfirmed_candidate_is_demoted(lab, tmp_path):
    def behave(call):
        if call["id"] == "prepared-all-t1":  # best on the search run, ordinary when measured again
            return timings(20, 0.4 if call["count"] == 1 else 1.0)
        return timings(20, 0.7 if call["id"] == "original-all-t1" else 1.0)

    report = run(lab, tmp_path, FakeWorker(behave), threads=[1])
    assert report["ledger"]["confirmations"]["prepared-all-t1"]["verdict"] == "no_measured_gain"
    assert report["selected"] == "original-all-t1" and report["status"] == "optimized"


def test_output_divergence_rejects_candidate_but_not_the_search(lab, tmp_path):
    def behave(call):
        if call["id"] == "prepared-all-t1":
            return timings(3, 0.1, correct=False, failure="candidate sample 2: output 0 beyond tolerance")
        return timings(20)

    report = run(lab, tmp_path, FakeWorker(behave), threads=[1])
    entry = report["ledger"]["candidates"]["prepared-all-t1"]
    assert entry["verdict"] == "rejected_divergence" and "beyond tolerance" in json.dumps(entry["summary"])
    assert report["status"] == "no_measured_gain"


def test_optimized_claims_name_the_confirmed_result(lab, tmp_path):
    fake = FakeWorker(lambda call: timings(20, 0.5 if call["id"] == "prepared-all-t2" else 1.0))
    report = run(lab, tmp_path, fake)
    claims = report["claims"]
    assert claims["cpu_correctness"].startswith("selected candidate prepared-all-t2: 20 confirmation and 20 search")
    assert "baseline was not validated" not in claims["cpu_correctness"]
    assert "independent confirmation" in claims["acceleration"] and "50.0%" in claims["acceleration"]
    assert "'default': 0.5" in claims["acceleration"] and "no tail percentile" in claims["acceleration"]
    assert claims["application_quality"] == "not measured"
    assert report["evidence"]["diverged"] == {} and report["evidence"]["failed"] == {}


def test_no_gain_claims_demonstrate_no_acceleration(lab, tmp_path):
    claims = run(lab, tmp_path, FakeWorker(), threads=[1])["claims"]
    assert claims["acceleration"].startswith("none demonstrated")
    assert "120 baseline and 120 candidate timed samples" in claims["cpu_correctness"]  # 6 candidates x 20
    assert "selected candidate" not in claims["cpu_correctness"] and "every" not in claims["cpu_correctness"]


def test_exhausted_claims_do_not_assert_a_complete_search(lab, tmp_path):
    fake = FakeWorker(lambda call: timings(20, 0.5 if call["id"] == "prepared-extended-t1" else 1.0))
    report = run(lab, tmp_path, fake, attempts=3, threads=[1])
    claims = report["claims"]
    assert report["status"] == "budget_exhausted" and report["selected"] is None
    assert claims["acceleration"].startswith("not established") and "3 candidate(s) unevaluated" in claims["acceleration"]
    assert "eligible but not independently confirmed: prepared-extended-t1" in claims["acceleration"]
    assert "60 baseline and 60 candidate timed samples" in claims["cpu_correctness"]
    assert "3 candidate(s) not evaluated" in claims["cpu_correctness"] and "every" not in claims["cpu_correctness"]
    assert json.loads((tmp_path / ".out.incomplete/report.json").read_text())["claims"] == claims


def test_divergence_claims_count_only_samples_that_passed(lab, tmp_path):
    def behave(call):
        if call["id"] == "prepared-all-t1":
            return timings(3, 0.1, correct=False, failure="candidate sample 2: output 0 beyond tolerance")
        return timings(20)

    report = run(lab, tmp_path, FakeWorker(behave), threads=[1])
    claims = report["claims"]
    # Five complete candidates, plus the two samples that passed before the third diverged.
    assert "103 baseline and 102 candidate timed samples" in claims["cpu_correctness"]
    assert "rejected for divergence: prepared-all-t1" in claims["cpu_correctness"]
    assert report["evidence"]["diverged"] == {"prepared-all-t1": {"default": "candidate sample 2: output 0 beyond tolerance"}}
    assert claims["acceleration"].startswith("none demonstrated")


def test_all_workers_crashing_validates_nothing(lab, tmp_path):
    def crash(call):
        raise RuntimeError("worker exited -9: killed")

    report = run(lab, tmp_path, FakeWorker(crash), threads=[1])
    claims = report["claims"]
    assert report["status"] == "no_measured_gain"  # the baseline settings are kept, by default and not by measurement
    assert claims["cpu_correctness"].startswith("not established")
    assert "baseline was not validated by any measurement" in claims["cpu_correctness"]
    assert "failed before completing" in claims["cpu_correctness"]
    assert claims["acceleration"].startswith("none measured")
    assert report["evidence"]["baseline_samples"] == report["evidence"]["candidate_samples"] == 0
    assert len(report["evidence"]["failed"]) == 6


def test_baseline_divergence_is_a_run_failure(lab, tmp_path):
    fake = FakeWorker(lambda call: timings(3, baseline_correct=False, failure="baseline sample 0: dtype"))
    with pytest.raises(RuntimeError, match="fair baseline disagrees with the oracle"):
        run(lab, tmp_path, fake, threads=[1])
    assert not (tmp_path / "out").exists()
    assert "fair baseline" in json.loads((tmp_path / ".out.incomplete/failure.json").read_text())["error"]


def test_worker_crash_fails_one_candidate_and_continues(lab, tmp_path):
    def behave(call):
        if call["id"] == "prepared-all-t1":
            raise RuntimeError("worker exited -9: killed by the out-of-memory killer")
        return timings(20)

    report = run(lab, tmp_path, FakeWorker(behave), threads=[1])
    entry = report["ledger"]["candidates"]["prepared-all-t1"]
    assert entry["verdict"] == "failed" and "out-of-memory" in entry["reason"]
    assert report["status"] == "no_measured_gain"


def test_attempt_budget_keeps_partial_evidence_and_resumes(lab, tmp_path):
    fake = FakeWorker(lambda call: timings(20, 0.5 if call["id"] == "prepared-extended-t1" else 1.0))
    first = run(lab, tmp_path, fake, attempts=3, threads=[1])
    staging = tmp_path / ".out.incomplete"
    assert first["status"] == "budget_exhausted" and first["exhausted"] == "attempts"
    assert not (tmp_path / "out").exists() and staging.is_dir()
    assert fake.ids() == ["prepared-all-t1", "original-all-t1", "prepared-extended-t1"]
    assert len(first["unevaluated_candidates"]) == 3
    assert json.loads((staging / "report.json").read_text())["status"] == "budget_exhausted"
    assert set(json.loads((staging / "ledger.json").read_text())["candidates"]) == set(fake.ids())

    # Resume with a larger budget: finished candidates are not measured again.
    again = FakeWorker(lambda call: timings(20, 0.5 if call["id"] == "prepared-extended-t1" else 1.0))
    second = run(lab, tmp_path, again, attempts=10, threads=[1])
    # The remaining candidates, then the confirmation of the one found before the stop.
    assert again.ids() == ["original-extended-t1", "prepared-basic-t1", "original-basic-t1", "prepared-extended-t1"]
    assert second["status"] == "optimized" and second["selected"] == "prepared-extended-t1"
    assert (tmp_path / "out/session.json").is_file() and not staging.exists()


def exhausted_run(lab, tmp_path):
    """An attempt-limited run that leaves completed candidates and prepared artifacts behind."""
    fake = FakeWorker()
    assert run(lab, tmp_path, fake, attempts=2, threads=[1])["status"] == "budget_exhausted"
    assert len(fake.calls) == 2
    return tmp_path / ".out.incomplete"


def test_unmodified_resume_reuses_prepared_artifacts_and_completed_work(lab, tmp_path):
    staging = exhausted_run(lab, tmp_path)
    assert cpu._prepare_profile.calls == ["default"] and (staging / "artifacts.json").is_file()
    again = FakeWorker()
    assert run(lab, tmp_path, again, threads=[1])["status"] == "no_measured_gain"
    assert cpu._prepare_profile.calls == ["default"]  # not prepared again
    assert again.ids() == ["prepared-extended-t1", "original-extended-t1", "prepared-basic-t1", "original-basic-t1"]


@pytest.mark.parametrize("damage", ["model", "original", "feeds", "oracle", "oracle_meta", "reference", "deleted",
                                    "replaced", "extra"])
def test_resume_rejects_changed_artifacts_before_any_worker_launches(lab, tmp_path, damage):
    staging = exhausted_run(lab, tmp_path)
    profile = staging / "bundle/default"
    ledger = (staging / "ledger.json").read_text()
    if damage == "deleted":
        (profile / "oracle.npz").unlink()
    elif damage == "replaced":  # same bytes would pass; a different file under the same name must not
        (profile / "model.onnx").unlink()
        (profile / "model.onnx").write_bytes(b"another model")
    elif damage == "extra":
        (profile / "stray.data").write_bytes(b"x")
    else:
        target = {"model": "model.onnx", "original": "original.onnx", "feeds": "feeds.npz", "oracle": "oracle.npz",
                  "oracle_meta": "oracle.json", "reference": "reference/input_0.pb"}[damage]
        with (profile / target).open("ab") as handle:
            handle.write(b"\0")
    again = FakeWorker()
    with pytest.raises(ValueError, match="Stale CPU experiment artifacts"):
        run(lab, tmp_path, again, threads=[1])
    assert again.calls == [] and cpu._prepare_profile.calls == ["default"]
    assert (staging / "ledger.json").read_text() == ledger and not (tmp_path / "out").exists()


def test_artifacts_changed_during_the_run_block_promotion(lab, tmp_path):
    staging = tmp_path / ".out.incomplete"
    fake = FakeWorker()

    def behave(call):
        if len(fake.calls) == 6:  # the last candidate: nothing is left to catch it but the final check
            (staging / "bundle/default/oracle.npz").write_bytes(b"swapped")
        return timings(20)

    fake.behave = behave
    with pytest.raises(ValueError, match="Stale CPU experiment artifacts.*oracle.npz modified"):
        run(lab, tmp_path, fake, threads=[1])
    assert not (tmp_path / "out").exists() and (staging / "ledger.json").is_file()


def test_interrupted_preparation_is_restartable(lab, tmp_path, monkeypatch):
    complete = cpu._prepare_profile

    def interrupted(budget, staging, name, *args):
        (staging / "bundle" / name).mkdir(parents=True)
        (staging / "bundle" / name / "model.onnx").write_bytes(b"half written")
        raise cpu.BudgetExhausted("seconds")

    monkeypatch.setattr(cpu, "_prepare_profile", interrupted)
    first = run(lab, tmp_path, FakeWorker(), threads=[1])
    assert first["status"] == "budget_exhausted" and not (tmp_path / ".out.incomplete/artifacts.json").exists()
    monkeypatch.setattr(cpu, "_prepare_profile", complete)
    assert run(lab, tmp_path, FakeWorker(), threads=[1])["status"] == "no_measured_gain"
    assert complete.calls == ["default"]


def test_deadline_bounds_every_subprocess_and_reports_exhaustion(lab, tmp_path):
    clock = Clock()
    fake = FakeWorker(clock=clock, step=40.0)
    report = run(lab, tmp_path, fake, budget_seconds=100.0, timeout=60.0, clock=clock, threads=[1])
    assert report["status"] == "budget_exhausted" and report["exhausted"] == "seconds"
    assert [c["timeout"] for c in fake.calls] == [60.0, 60.0, 20.0]  # cut to what is left of the budget
    assert report["budget"]["seconds_left"] == 0.0
    assert not (tmp_path / "out").exists() and (tmp_path / ".out.incomplete/ledger.json").is_file()


def test_timeout_caused_by_the_budget_is_not_a_candidate_failure(lab, tmp_path):
    clock = Clock()

    def runner(argv, timeout):
        clock.now += timeout
        raise subprocess.TimeoutExpired(argv, timeout)

    report = run(lab, tmp_path, runner, budget_seconds=30.0, timeout=60.0, clock=clock, threads=[1])
    assert report["status"] == "budget_exhausted"
    assert all(e["verdict"] != "failed" for e in report["ledger"]["candidates"].values())

    # With budget to spare the same timeout is the candidate's own failure.
    clock.now = 0.0
    shutil.rmtree(tmp_path / ".out.incomplete")
    report = run(lab, tmp_path, runner, budget_seconds=1e6, timeout=60.0, clock=clock, threads=[1])
    assert report["status"] == "no_measured_gain"
    assert {e["verdict"] for e in report["ledger"]["candidates"].values()} == {"failed"}


@pytest.mark.parametrize("change", ["atol", "rtol", "samples", "profile", "source", "env", "threads", "affinity"])
def test_stale_resume_is_rejected_when_identity_changes(lab, tmp_path, monkeypatch, change):
    kwargs = {"attempts": 1, "threads": [1]}
    assert run(lab, tmp_path, FakeWorker(), **kwargs)["status"] == "budget_exhausted"
    if change == "atol":
        kwargs["atol"] = 1e-3
    elif change == "rtol":
        kwargs["rtol"] = 1e-2
    elif change == "samples":
        kwargs["samples"] = 24
    elif change == "profile":
        kwargs["profiles"] = {"default": {}, "other": {}}
    elif change == "source":
        write_model(lab, scale=3.0)
    elif change == "env":
        monkeypatch.setenv("OMP_NUM_THREADS", "7")
    elif change == "threads":
        kwargs["threads"] = [1, 2]
    else:
        monkeypatch.setattr(cpu, "usable_cpus", lambda: [0, 1])
    with pytest.raises(ValueError, match="Stale CPU experiment"):
        run(lab, tmp_path, FakeWorker(), **kwargs)
    assert (tmp_path / ".out.incomplete/ledger.json").is_file()  # the old evidence is untouched


def test_budget_does_not_change_the_identity(lab, tmp_path):
    fake = FakeWorker()
    run(lab, tmp_path, fake, attempts=1, budget_seconds=500.0, timeout=100.0, threads=[1])
    resumed = run(lab, tmp_path, FakeWorker(), attempts=50, budget_seconds=900.0, timeout=200.0, threads=[1])
    assert resumed["status"] == "no_measured_gain"


def test_identity_records_environment_and_host(lab, tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("ORT_SOMETHING", "1")
    monkeypatch.setenv("UNRELATED_SETTING", "x")
    report = run(lab, tmp_path, FakeWorker(), attempts=1, threads=[1])
    identity = json.loads((tmp_path / ".out.incomplete/resume.json").read_text())
    assert report["identity"] == identity
    assert {"OMP_NUM_THREADS": "3", "ORT_SOMETHING": "1"}.items() <= identity["environment"].items()
    assert "UNRELATED_SETTING" not in identity["environment"]
    assert identity["host"]["affinity"] == [0, 1, 2, 3] and "isa_flags_sha256" in identity["host"]
    assert identity["tolerances"] == {"atol": 1e-5, "rtol": 1e-4} and identity["target"] == "cpu"
    assert {"onnxruntime", "numpy", "sources"} <= set(identity["tools"])
    assert len(identity["source"]["sha256"]) == 64 and identity["adapter"] is None
    assert identity["search"]["threads"] == [1] and identity["search"]["baseline"] == cpu.BASELINE
    import os
    assert os.environ["OMP_NUM_THREADS"] == "3" and "ORT_SOMETHING" in os.environ  # nothing was mutated


def test_never_overwrites_an_existing_destination(lab, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep").write_text("earlier result")
    with pytest.raises(FileExistsError):
        run(lab, tmp_path, FakeWorker())
    assert (out / "keep").read_text() == "earlier result"


def test_second_run_after_success_cannot_replace_it(lab, tmp_path):
    run(lab, tmp_path, FakeWorker(), threads=[1])
    before = (tmp_path / "out/cpu-optimization.json").read_text()
    with pytest.raises(FileExistsError):
        run(lab, tmp_path, FakeWorker(), threads=[1])
    assert (tmp_path / "out/cpu-optimization.json").read_text() == before


def test_no_vulkan_requirement(lab, tmp_path, monkeypatch):
    import onnx_vulkan_pipeline.pipeline as pipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("the CPU path must not touch Vulkan tooling")

    for name in ("binary", "inspect_model", "gpu_lock", "_device", "_run", "_require_compatible_policy"):
        monkeypatch.setattr(pipeline, name, forbidden, raising=False)
    monkeypatch.setenv("ONNX_VULKAN_ROOT", str(tmp_path / "no-binaries-here"))
    monkeypatch.setenv("VK_ICD_FILENAMES", str(tmp_path / "missing_icd.json"))
    report = run(lab, tmp_path, FakeWorker(), threads=[1])
    assert report["status"] == "no_measured_gain"
    stored = json.loads((tmp_path / "out/cpu-optimization.json").read_text())
    assert not {"device", "compatible", "compatibility"} & set(stored)
    assert "VK_ICD_FILENAMES" not in stored["identity"]["environment"]


def test_invalid_arguments_fail_before_any_work(lab, tmp_path):
    for kwargs in ({"samples": 19}, {"atol": -1.0}, {"budget_seconds": 0}, {"attempts": 0},
                   {"threads": [64]}, {"profiles": {"bad name": {}}}, {"profiles": {"p": {"n": 0}}}):
        with pytest.raises(ValueError):
            run(lab, tmp_path, FakeWorker(), **kwargs)
    assert not (tmp_path / ".out.incomplete").exists()
    with pytest.raises(ValueError, match="existing .onnx"):
        cpu.optimize_cpu(tmp_path / "missing.onnx", tmp_path / "out2")


def test_load_workload_resolves_paths_and_rejects_unknown_keys(tmp_path):
    (tmp_path / "w.json").write_text(json.dumps({"source": "m.onnx", "profiles": {"a": {"n": 1}}, "attempts": 4}))
    workload = cpu.load_workload(tmp_path / "w.json")
    assert workload == {"source": tmp_path / "m.onnx", "profiles": {"a": {"n": 1}}, "attempts": 4}
    (tmp_path / "bad.json").write_text(json.dumps({"source": "m.onnx", "attemps": 4}))
    with pytest.raises(ValueError, match="unknown keys"):
        cpu.load_workload(tmp_path / "bad.json")
    example = Path(__file__).parents[1] / "python/examples/cpu-workload.json"
    assert set(cpu.load_workload(example)) <= cpu.WORKFLOW_KEYS


# Real ONNX Runtime from here on: the worker, oracle and a complete subprocess run.

def profile_directory(tmp_path: Path, candidate_offset: float) -> Path:
    directory = tmp_path / "profile"
    (directory / "reference").mkdir(parents=True)
    write_model(directory / "original.onnx")
    write_model(directory / "model.onnx", offset=candidate_offset)
    feed = numpy.linspace(-1, 1, 8, dtype=numpy.float32).reshape(1, 8)
    (directory / "reference/input_0.pb").write_bytes(numpy_helper.from_array(feed, "x").SerializeToString())
    worker.oracle({"directory": str(directory)})
    return directory


def measure_config(directory: Path, candidate: dict, **overrides) -> dict:
    return {"directory": str(directory), "baseline": cpu.BASELINE, "candidate": candidate, "samples": 20,
            "warmup": 1, "atol": 1e-5, "rtol": 1e-4, **overrides}


def test_session_settings_reach_onnx_runtime(tmp_path):
    directory = profile_directory(tmp_path, 0.0)
    baseline = worker._session(ort, directory, cpu.BASELINE).get_session_options()
    assert baseline.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    assert baseline.intra_op_num_threads == 0 and baseline.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    candidate = {"model": "prepared", "graph_optimization_level": "ORT_ENABLE_BASIC",
                 "execution_mode": "ORT_SEQUENTIAL", "intra_op_num_threads": 2}
    options = worker._session(ort, directory, candidate).get_session_options()
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    assert options.intra_op_num_threads == 2
    oracle = json.loads((directory / "oracle.json").read_text())
    assert oracle["setting"]["graph_optimization_level"] == "ORT_DISABLE_ALL"
    assert oracle["setting"]["intra_op_num_threads"] == oracle["setting"]["inter_op_num_threads"] == 1
    assert oracle["input_names"] == ["x"] and oracle["output_names"] == ["y"]
    assert oracle["outputs"] == [{"dtype": "float32", "shape": [1, 8]}]


def test_worker_times_checked_interleaved_pairs(tmp_path):
    directory = profile_directory(tmp_path, 0.0)
    candidate = {"model": "prepared", "graph_optimization_level": "ORT_ENABLE_EXTENDED",
                 "execution_mode": "ORT_SEQUENTIAL", "intra_op_num_threads": 1}
    result = worker.measure(measure_config(directory, candidate))
    assert result["correct"] and result["baseline_correct"] and result["failure"] is None
    assert len(result["baseline_ms"]) == len(result["candidate_ms"]) == 20
    assert result["orders"] == ["AB", "BA"] * 10
    assert all(t > 0 for t in result["baseline_ms"] + result["candidate_ms"])
    assert set(result["setup_ms"]) == {"baseline", "candidate"}  # session creation is reported, not timed


def test_worker_rejects_output_beyond_fixed_tolerance(tmp_path):
    directory = profile_directory(tmp_path, 1e-3)  # candidate model differs from the original by +1e-3
    candidate = {"model": "prepared", "graph_optimization_level": "ORT_ENABLE_ALL",
                 "execution_mode": "ORT_SEQUENTIAL", "intra_op_num_threads": 1}
    strict = worker.measure(measure_config(directory, candidate))
    assert not strict["correct"] and strict["baseline_correct"]
    assert "sample 0" in strict["failure"] and "beyond tolerance" in strict["failure"]
    assert strict["max_abs"] == pytest.approx(1e-3, rel=1e-3)
    assert len(strict["candidate_ms"]) < 20  # stopped at the first divergent sample
    loose = worker.measure(measure_config(directory, candidate, atol=1e-2))
    assert loose["correct"] and len(loose["candidate_ms"]) == 20


def test_end_to_end_cpu_run_with_real_workers(tmp_path):
    source = write_model(tmp_path / "model.onnx")
    report = cpu.optimize_cpu(source, tmp_path / "out", threads=[1], timeout=120, budget_seconds=600)
    assert report["status"] in ("optimized", "no_measured_gain")
    out = tmp_path / "out"
    session = json.loads((out / "session.json").read_text())
    assert session["providers"] == ["CPUExecutionProvider"]
    assert (out / session["models"]["default"]).is_file()
    assert (out / "default/original.onnx").read_bytes() == source.read_bytes()  # the original is preserved
    for entry in report["ledger"]["candidates"].values():
        assert entry["verdict"] in ("eligible", "no_measured_gain", "rejected_regression"), entry
        assert all(r["correct"] for r in entry["profiles"].values())
        assert all(len(r["candidate_ms"]) == 20 for r in entry["profiles"].values())
    # The deployable settings run: load the recorded model with the recorded options.
    options = ort.SessionOptions()
    options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, session["session_options"]["graph_optimization_level"])
    options.intra_op_num_threads = session["session_options"]["intra_op_num_threads"]
    deployed = ort.InferenceSession(str(out / session["models"]["default"]), options, providers=session["providers"])
    feed = numpy.linspace(-1, 1, 8, dtype=numpy.float32).reshape(1, 8)
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, {"x": feed})[0]
    numpy.testing.assert_allclose(deployed.run(None, {"x": feed})[0], expected, atol=1e-5, rtol=1e-4)
