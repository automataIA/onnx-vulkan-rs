"""CPU deployment optimization: a bounded search over ONNX Runtime CPU session settings.

No Vulkan binary, device, graph-compatibility check or GPU lane is involved.
The model is prepared with `prepare_model` in a subprocess (original plus the
conservative transformed model), an *oracle* is computed from the original with
optimizations disabled on one thread, and candidates are timed against a *fair
baseline*: the original model under ORT_ENABLE_ALL with ORT's default
threading. Correctness (fixed tolerances against the oracle on every timed
sample) and acceleration (median paired latency ratio) are separate claims;
application quality is not measured here.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .pipeline import _link_duplicates, digest, write_json
from .process import command, failure_reason

SCHEMA_VERSION = 1
LEVELS = {"all": "ORT_ENABLE_ALL", "extended": "ORT_ENABLE_EXTENDED", "basic": "ORT_ENABLE_BASIC"}
# What a plain deployment gets: the original graph, full optimization, ORT's own threading.
BASELINE = {"model": "original", "graph_optimization_level": LEVELS["all"],
            "execution_mode": "ORT_SEQUENTIAL", "intra_op_num_threads": 0}
# Environment variables that change CPU threading or kernels. Only these (and ORT_*)
# are read; the whole environment is neither recorded nor modified.
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_DYNAMIC", "OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES",
    "GOMP_CPU_AFFINITY", "GOMP_SPINCOUNT", "KMP_AFFINITY", "KMP_BLOCKTIME", "KMP_HW_SUBSET",
    "MKL_NUM_THREADS", "MKL_DYNAMIC", "MKL_ENABLE_INSTRUCTIONS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
WORKFLOW_KEYS = {"source", "profiles", "adapter", "atol", "rtol", "samples", "timeout", "budget_seconds",
                 "attempts", "threads", "warmup", "min_gain", "regression_limit"}
# Fraction of paired samples that must favor a candidate, pooled over profiles.
MIN_FASTER_PAIRS = 0.7


class BudgetExhausted(Exception):
    """The invocation's deadline or attempt count ran out; evidence stays resumable."""

    def __init__(self, reason):
        super().__init__(f"budget exhausted ({reason})")
        self.reason = reason


def load_workload(path):
    """Read a CPU workload JSON file into `optimize_cpu` keyword arguments.

    Relative `source` and `adapter` paths resolve against the file's directory.
    Unknown keys are rejected so a typo cannot silently fall back to a default."""
    path = Path(path)
    workload = json.loads(path.read_text())
    unknown = sorted(set(workload) - WORKFLOW_KEYS - {"schema_version"})
    if unknown or "source" not in workload:
        raise ValueError(f"{path}: " + (f"unknown keys {unknown}" if unknown else "missing key 'source'"))
    if workload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported workload schema")
    workload.pop("schema_version", None)
    for key in ("source", "adapter"):
        if workload.get(key) is not None:
            workload[key] = path.parent / workload[key]
    return workload


def usable_cpus():
    """CPU ids this process may run on (affinity, not the machine's total)."""
    return sorted(os.sched_getaffinity(0))


def thread_grid(usable, threads=None):
    """Intra-op thread counts to try, bounded by the usable CPU count.

    `threads` is None (powers of two up to `usable`, plus `usable`), an int
    upper bound used the same way, or an explicit sequence of counts. Counts
    above `usable` would oversubscribe the CPUs and are rejected."""
    def valid(count):
        return isinstance(count, int) and not isinstance(count, bool) and 1 <= count <= usable

    if threads is None or isinstance(threads, int):
        bound = usable if threads is None else threads
        if not valid(bound):
            raise ValueError(f"threads must be between 1 and the {usable} usable CPUs")
        return sorted({2 ** k for k in range(bound.bit_length()) if 2 ** k <= bound} | {bound})
    counts = sorted(set(threads))
    if not counts or not all(valid(count) for count in counts):
        raise ValueError(f"threads must be integers between 1 and the {usable} usable CPUs")
    return counts


def _candidates(grid, models):
    """Deterministic candidate order: most promising level first, model preference next."""
    return {f"{model}-{level}-t{count}": {"model": model, "graph_optimization_level": name,
                                          "execution_mode": "ORT_SEQUENTIAL", "intra_op_num_threads": count}
            for level, name in LEVELS.items() for model in models for count in grid}


def _versions():
    versions = {}
    for name in ("onnx", "onnxruntime", "numpy", "onnxscript", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    import onnxruntime

    versions["ort_device"] = onnxruntime.get_device()
    versions["ort_providers"] = onnxruntime.get_available_providers()
    here = Path(__file__).parent
    versions["sources"] = {name: digest(here / name)
                           for name in ("cpu.py", "_cpu_worker.py", "prepare.py", "_prepare_worker.py")}
    return versions


def _host():
    model, flags = None, ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "model name" and model is None:
                model = value.strip()
            elif key in ("flags", "Features") and not flags:
                flags = value.strip()
    except OSError:
        pass
    return {"system": platform.system(), "machine": platform.machine(), "cpu_model": model,
            "isa_flags_sha256": hashlib.sha256(" ".join(sorted(flags.split())).encode()).hexdigest(),
            "logical_cpus": os.cpu_count(), "affinity": usable_cpus()}


def _source_identity(source):
    import onnx
    from .prepare import _external_locations

    model = onnx.load(str(source), load_external_data=False)
    return {"sha256": digest(source),
            "external_data": {str(location): digest(source.parent / location)
                              for location in _external_locations(model)}}


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten(value[key], f"{prefix}{key}.")
    else:
        yield prefix.rstrip("."), value


def _changes(recorded, current):
    old, new = dict(_flatten(recorded)), dict(_flatten(current))
    return [f"{key}: {old.get(key)!r} → {new.get(key)!r}" for key in sorted({*old, *new}) if old.get(key) != new.get(key)]


def _identity(source, adapter, profiles, atol, rtol, samples, warmup, min_gain, regression_limit, grid):
    """Everything the evidence depends on. Budget and attempt limits are absent on
    purpose: they bound an invocation and a resumed run may raise them."""
    identity = {
        "schema_version": SCHEMA_VERSION, "target": "cpu", "source": _source_identity(source),
        "adapter": digest(adapter) if adapter else None, "profiles": profiles,
        "tolerances": {"atol": atol, "rtol": rtol},
        "measurement": {"samples": samples, "warmup": warmup, "min_gain": min_gain,
                        "regression_limit": regression_limit, "pair_order": "AB/BA interleaved",
                        "faster_pair_fraction": MIN_FASTER_PAIRS},
        "search": {"baseline": BASELINE, "levels": list(LEVELS.values()), "threads": grid,
                   "oracle": {"graph_optimization_level": "ORT_DISABLE_ALL", "intra_op_num_threads": 1}},
        "tools": _versions(), "host": _host(),
        "environment": {name: os.environ[name] for name in sorted(os.environ)
                        if name in THREAD_ENVIRONMENT or name.startswith("ORT_")},
    }
    return json.loads(json.dumps(identity, allow_nan=False))


def _validate_profiles(profiles):
    profiles = profiles or {"default": {}}
    for name, dims in profiles.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name or ""):
            raise ValueError("Profile names must contain only letters, digits, underscores or hyphens")
        for dim, size in dims.items():
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError(f"profile {name}: {dim}={size}: sizes must be positive integers")
    return {name: dict(dims) for name, dims in profiles.items()}


class _Budget:
    """The invocation's deadline and attempt count, and the one place workers are launched."""

    def __init__(self, seconds, attempts, timeout, clock, runner):
        self.deadline = clock() + seconds
        self.attempts_left, self.attempts_used = attempts, 0
        self.timeout, self.clock, self.runner = timeout, clock, runner

    def remaining(self):
        return self.deadline - self.clock()

    def attempt(self):
        if self.attempts_left <= 0:
            raise BudgetExhausted("attempts")
        self.attempts_left -= 1
        self.attempts_used += 1

    def run(self, argv, *, serialize=False):
        """Run a subprocess limited by the smaller of its timeout and the remaining budget."""
        if self.remaining() <= 0:
            raise BudgetExhausted("seconds")
        if not serialize:
            return self._launch(argv)
        # Timing runs of concurrent invocations would disturb each other.
        path = Path(tempfile.gettempdir()) / f"onnx-vulkan-cpu-{os.getuid()}.lock"
        with path.open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if self.remaining() <= 0:
                        raise BudgetExhausted("seconds")
                    time.sleep(0.05)
            try:
                return self._launch(argv)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _launch(self, argv):
        remaining = self.remaining()
        if remaining <= 0:
            raise BudgetExhausted("seconds")
        try:
            return self.runner(argv, timeout=min(self.timeout, remaining))
        except subprocess.TimeoutExpired:
            if remaining < self.timeout:
                raise BudgetExhausted("seconds")
            raise RuntimeError(f"{argv[2]} timed out after {self.timeout}s")


def _worker(budget, config, path, *, serialize=False):
    write_json(path, config)
    budget.run([sys.executable, "-m", "onnx_vulkan_pipeline._cpu_worker", path], serialize=serialize)


def _prepare_profile(budget, staging, name, dims, source, adapter, atol, rtol):
    """Prepare one profile into `bundle/<name>`; the directory exists only when complete.

    The caller records the finished directory in the artifact manifest. A directory
    without a manifest entry (interrupted before that) is rebuilt from scratch."""
    bundle = staging / "bundle"
    final, temporary = bundle / name, bundle / f".{name}.preparing"
    shutil.rmtree(final, ignore_errors=True)
    shutil.rmtree(temporary, ignore_errors=True)
    bundle.mkdir(exist_ok=True)
    config = staging / f"prepare-{name}.json"
    write_json(config, {"source": str(source), "destination": str(temporary),
                        "adapter": str(adapter) if adapter else None, "profiles": dims,
                        "atol": atol, "rtol": rtol, "quantization": None})
    budget.run([sys.executable, "-m", "onnx_vulkan_pipeline._prepare_worker", config])
    _link_duplicates(temporary)
    _worker(budget, {"mode": "oracle", "directory": str(temporary)}, staging / f"oracle-{name}.json")
    temporary.rename(final)


def _profile_digests(directory):
    """Content digest of every file of a profile directory (reference tensors included)."""
    def entry(path):
        # A link is part of the artifact: record where it points as well as what it reaches.
        if path.is_symlink():
            return f"symlink {os.readlink(path)} {digest(path) if path.is_file() else 'dangling'}"
        return digest(path)

    return {path.relative_to(directory).as_posix(): entry(path)
            for path in sorted(directory.rglob("*")) if path.is_file() or path.is_symlink()}


def _verify_artifacts(staging):
    """The recorded profile artifacts, after checking them against the files on disk.

    Prepared models, external data, feeds, oracle outputs and reference tensors are
    cached across invocations and the ledger's timings depend on them, so any
    altered, deleted, added or replaced file rejects the experiment. This guards
    against accidental changes; the manifest and ledger themselves are not authenticated."""
    path = staging / "artifacts.json"
    recorded = json.loads(path.read_text()) if path.is_file() else {}
    for name, files in recorded.items():
        directory = staging / "bundle" / name
        actual = _profile_digests(directory) if directory.is_dir() else {}
        problems = [f"{rel} {'missing' if rel not in actual else 'modified'}"
                    for rel in sorted(files) if actual.get(rel) != files[rel]]
        problems += [f"{rel} unrecorded" for rel in sorted(set(actual) - set(files))]
        if problems:
            raise ValueError(f"Stale CPU experiment artifacts at {staging} (profile {name}: "
                             f"{'; '.join(problems)}); choose a new output directory")
    return recorded


def judge(results, *, samples, min_gain, regression_limit):
    """Verdict and per-profile summary for one candidate's paired measurements.

    `results` maps profile name to the worker's result. Rejections come first and
    are final: any output divergence, then any profile whose median candidate/
    baseline ratio regresses past `regression_limit`. A candidate is `eligible`
    only if the geometric mean of the per-profile median ratios gains at least
    `min_gain` and, pair by pair (the geometric mean over profiles of each
    pair's ratio), it was faster in at least MIN_FASTER_PAIRS of the pairs.
    Medians only; no tail percentiles are claimed."""
    if any(not result["correct"] for result in results.values()):
        # A diverging worker stops at the first bad sample, so its timings are incomplete.
        return "rejected_divergence", {name: result["failure"] for name, result in results.items()
                                       if not result["correct"]}, 0.0
    summary, ratios, per_pair = {}, [], []
    for name, result in results.items():
        base, cand = result["baseline_ms"], result["candidate_ms"]
        if (len(base) != samples or len(cand) != samples
                or any(not math.isfinite(v) or v <= 0 for v in base + cand)):
            return "invalid", {"reason": f"profile {name}: expected {samples} finite positive timings"}, 0.0
        paired = [c / b for b, c in zip(base, cand)]
        summary[name] = {"baseline_median_ms": statistics.median(base), "candidate_median_ms": statistics.median(cand),
                         "median_pair_ratio": statistics.median(paired), "faster_pairs": sum(r < 1 for r in paired),
                         "pairs": len(paired)}
        ratios.append(summary[name]["median_pair_ratio"])
        per_pair.append(paired)
    summary = {"profiles": summary}
    if any(ratio > 1 + regression_limit for ratio in ratios):
        return "rejected_regression", summary, 0.0
    gain = 1 - math.exp(statistics.fmean(math.log(ratio) for ratio in ratios))
    combined = [math.exp(statistics.fmean(math.log(r) for r in pair)) for pair in zip(*per_pair)]
    summary["faster_pair_fraction"] = sum(r < 1 for r in combined) / len(combined)
    eligible = gain >= min_gain and summary["faster_pair_fraction"] >= MIN_FASTER_PAIRS
    return ("eligible" if eligible else "no_measured_gain"), summary, gain


class _Search:
    def __init__(self, staging, budget, profiles, samples, warmup, atol, rtol, min_gain, regression_limit):
        self.staging, self.budget, self.profiles = staging, budget, profiles
        self.samples, self.warmup, self.atol, self.rtol = samples, warmup, atol, rtol
        self.min_gain, self.regression_limit = min_gain, regression_limit
        path = staging / "ledger.json"
        self.ledger = json.loads(path.read_text()) if path.is_file() else {"candidates": {}, "confirmations": {}}
        self.counter = 0

    def save(self):
        write_json(self.staging / "ledger.json", self.ledger)

    def _measure(self, entry, settings, label):
        """Time `settings` on every profile still missing from `entry`; True if it ran anything."""
        ran = False
        for name in self.profiles:
            if name in entry["profiles"]:
                continue
            if not ran:
                self.budget.attempt()
                ran = True
            self.counter += 1
            output = self.staging / "work" / f"measure-{self.counter}.json"
            output.parent.mkdir(exist_ok=True)
            try:
                _worker(self.budget, {"mode": "measure", "directory": str(self.staging / "bundle" / name),
                                      "baseline": BASELINE, "candidate": settings, "samples": self.samples,
                                      "warmup": self.warmup, "atol": self.atol, "rtol": self.rtol,
                                      "output": str(output)},
                        self.staging / "work" / f"measure-{self.counter}-config.json", serialize=True)
                result = json.loads(output.read_text())
            except BudgetExhausted:
                raise
            except Exception as error:  # a worker crash rejects the candidate, not the run
                entry["verdict"], entry["reason"] = "failed", f"{label} on {name}: {failure_reason(error)}"
                self.save()
                return ran
            if not result["baseline_correct"]:
                raise RuntimeError(f"the fair baseline disagrees with the oracle on profile {name}: {result['failure']}")
            entry["profiles"][name] = result
            self.save()
            if not result["correct"]:
                break
        return ran

    def _judge(self, entry):
        if entry.get("verdict") == "failed":
            return
        entry["verdict"], entry["summary"], entry["gain"] = judge(
            entry["profiles"], samples=self.samples, min_gain=self.min_gain, regression_limit=self.regression_limit)
        # Raw timings stay in the ledger; the summary is what reports read.
        self.save()

    def evaluate(self, cid, settings):
        entry = self.ledger["candidates"].setdefault(cid, {"settings": settings, "profiles": {}, "verdict": None})
        if entry["verdict"] is not None:
            return
        self._measure(entry, settings, cid)
        self._judge(entry)

    def winner(self):
        """Best eligible candidate whose independent re-measurement also passes."""
        # Equal gains keep search order, which lists the simpler settings first.
        eligible = sorted((-entry["gain"], index, cid) for index, (cid, entry) in enumerate(self.ledger["candidates"].items())
                          if entry["verdict"] == "eligible")
        for _, _, cid in eligible:
            entry = self.ledger["confirmations"].setdefault(
                cid, {"settings": self.ledger["candidates"][cid]["settings"], "profiles": {}, "verdict": None})
            if entry["verdict"] is None:
                self._measure(entry, entry["settings"], f"confirmation of {cid}")
                self._judge(entry)
            if entry["verdict"] == "eligible":
                return cid
        return None


def _checks(ledger):
    """Numerical checks that actually completed, counted from the worker results in the ledger.

    A worker that stops at a divergent sample timed it but did not pass it, and a crashed
    worker contributes nothing, so only results the parent received are counted."""
    checks = {"baseline_samples": 0, "candidate_samples": 0, "diverged": {}, "failed": {}}
    for kind in ("candidates", "confirmations"):
        for cid, entry in ledger[kind].items():
            label = cid if kind == "candidates" else f"confirmation of {cid}"
            for name, result in entry["profiles"].items():
                if result["baseline_correct"]:
                    checks["baseline_samples"] += len(result["baseline_ms"])
                matched = len(result["candidate_ms"])
                if not result["correct"]:
                    matched = max(matched - 1, 0)
                    checks["diverged"].setdefault(label, {})[name] = result["failure"]
                checks["candidate_samples"] += matched
            if entry["verdict"] == "failed":
                checks["failed"][label] = entry["reason"]
    return checks


def _samples(entry):
    return sum(len(result["candidate_ms"]) for result in entry["profiles"].values())


def _claims(identity, status, ledger, chosen, pending):
    """Claims limited to what the ledger's completed checks and measurements support."""
    tolerances = identity["tolerances"]
    oracle = (f"the oracle (original model, ORT_DISABLE_ALL, 1 thread) within "
              f"atol={tolerances['atol']}, rtol={tolerances['rtol']}")
    checks = _checks(ledger)
    parts = []
    if not checks["baseline_samples"] and not checks["candidate_samples"]:
        parts.append(f"not established: no worker completed a numerical check against {oracle}; "
                     "the baseline was not validated by any measurement")
    else:
        if chosen:
            confirmation = ledger["confirmations"][chosen]
            parts.append(f"selected candidate {chosen}: {_samples(confirmation)} confirmation and "
                         f"{_samples(ledger['candidates'][chosen])} search timed samples matched {oracle}")
        parts.append(f"{checks['baseline_samples']} baseline and {checks['candidate_samples']} candidate "
                     f"timed samples in total matched {oracle}")
        if not checks["baseline_samples"]:
            parts.append("the baseline was not validated by any measurement")
    if checks["diverged"]:
        parts.append(f"rejected for divergence: {', '.join(sorted(checks['diverged']))} "
                     "(samples after the first divergence were not checked)")
    if checks["failed"]:
        parts.append(f"failed before completing, not fully checked: {', '.join(sorted(checks['failed']))}")
    if pending:
        parts.append(f"{len(pending)} candidate(s) not evaluated")
    if chosen:
        confirmation = ledger["confirmations"][chosen]
        ratios = {name: round(entry["median_pair_ratio"], 4) for name, entry in confirmation["summary"]["profiles"].items()}
        acceleration = (f"selected candidate {chosen}: its independent confirmation re-measurement shows a geometric-mean "
                        f"median paired latency gain of {confirmation['gain']:.1%} over the fair baseline "
                        f"(search run: {ledger['candidates'][chosen]['gain']:.1%}), faster in "
                        f"{confirmation['summary']['faster_pair_fraction']:.0%} of pairs; median candidate/baseline "
                        f"ratio per profile {ratios}; session setup and warmup excluded; no tail percentile is claimed")
    elif status == "budget_exhausted":
        unconfirmed = [cid for cid, entry in ledger["candidates"].items()
                       if entry["verdict"] == "eligible" and ledger["confirmations"].get(cid, {}).get("verdict") is None]
        acceleration = (f"not established: the search stopped with {len(pending)} candidate(s) unevaluated and nothing "
                        "selected" + (f"; eligible but not independently confirmed: {', '.join(unconfirmed)}" if unconfirmed else ""))
    elif not checks["candidate_samples"]:
        acceleration = "none measured: no candidate completed a timed and checked run; the baseline settings are kept by default"
    else:
        acceleration = ("none demonstrated: no candidate passed both the search and an independent confirmation; "
                        "the baseline settings are kept")
    return {"cpu_correctness": "; ".join(parts), "acceleration": acceleration, "application_quality": "not measured"}


def _report(identity, status, search, budget, *, chosen=None, selected_settings=None, pending=(), extra=None):
    ledger = search.ledger
    return {
        "schema_version": SCHEMA_VERSION, "target": "cpu", "status": status,
        "claims": _claims(identity, status, ledger, chosen, pending),
        "evidence": _checks(ledger),
        "baseline": BASELINE, "selected": chosen, "selected_settings": selected_settings,
        "identity": identity, "ledger": ledger,
        "budget": {"attempts_used": budget.attempts_used, "attempts_left": budget.attempts_left,
                   "seconds_left": max(0.0, budget.remaining())},
        "limitations": [
            "baseline and candidate sessions share one process and interleave; idle thread pools may still spin",
            "sequential execution only; inter-op parallelism and spin controls are not searched",
            "no quantization: lossy changes need a dataset and metric",
            "settings are valid for the recorded ORT build, CPU/ISA, affinity and thread environment",
        ],
        **(extra or {}),
    }


def optimize_cpu(source, destination, *, profiles=None, adapter=None, atol=1e-5, rtol=1e-4, samples=20,
                 timeout=300, budget_seconds=1800.0, attempts=16, threads=None, warmup=3, min_gain=0.10,
                 regression_limit=0.02, clock=time.monotonic, runner=command):
    """Search ORT CPU session settings for `source` and write a deployable bundle to `destination`.

    Returns the report. `status` is `optimized` (a candidate beat the fair
    baseline and its independent confirmation), `no_measured_gain` (every
    candidate was evaluated; the bundle keeps the baseline settings) or
    `budget_exhausted` (the deadline or attempt count ended the search first:
    nothing is written to `destination`, the evidence stays in
    `.<destination>.incomplete` and calling again with the same inputs resumes it).
    `timeout` bounds each subprocess and is further cut to the remaining
    `budget_seconds`; `attempts` counts candidate (and confirmation) evaluations
    in this invocation. An existing `destination` is never overwritten.
    """
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
               for v in (atol, rtol, regression_limit)) or not math.isfinite(min_gain) or min_gain <= 0:
        raise ValueError("tolerances and regression_limit must be finite and nonnegative; min_gain positive")
    if samples < 20:
        raise ValueError("CPU optimization requires at least 20 interleaved sample pairs")
    if not (isinstance(warmup, int) and warmup >= 0 and isinstance(attempts, int) and attempts > 0
            and math.isfinite(budget_seconds) and budget_seconds > 0 and math.isfinite(timeout) and timeout > 0):
        raise ValueError("warmup must be >= 0; attempts, budget_seconds and timeout must be positive")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    adapter = Path(adapter).resolve() if adapter else None
    if source.suffix.lower() != ".onnx" or not source.is_file():
        raise ValueError(f"source must be an existing .onnx file: {source}")
    if adapter is not None and not adapter.is_file():
        raise ValueError(f"adapter does not exist: {adapter}")
    if destination.exists():
        raise FileExistsError(destination)
    profiles = _validate_profiles(profiles)
    grid = thread_grid(len(usable_cpus()), threads)
    identity = _identity(source, adapter, profiles, float(atol), float(rtol), samples, warmup,
                         float(min_gain), float(regression_limit), grid)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.incomplete")
    if staging.exists():
        recorded = json.loads((staging / "resume.json").read_text())
        if recorded != identity:
            raise ValueError(f"Stale CPU experiment at {staging} ({'; '.join(_changes(recorded, identity))}); "
                             "choose a new output directory")
    else:
        staging.mkdir()
        write_json(staging / "resume.json", identity)
    manifest = _verify_artifacts(staging)
    budget = _Budget(budget_seconds, attempts, timeout, clock, runner)
    search = _Search(staging, budget, profiles, samples, warmup, float(atol), float(rtol),
                     float(min_gain), float(regression_limit))
    candidates = {}
    try:
        for name, dims in profiles.items():
            if name in manifest:
                continue
            _prepare_profile(budget, staging, name, dims, source, adapter, float(atol), float(rtol))
            manifest[name] = _profile_digests(staging / "bundle" / name)
            write_json(staging / "artifacts.json", manifest)
        # A transformed model that is byte-identical everywhere is not a separate candidate.
        differs = any((staging / "bundle" / name / "model.onnx").read_bytes()
                      != (staging / "bundle" / name / "original.onnx").read_bytes() for name in profiles)
        models = ["prepared", "original"] if differs else ["original"]
        candidates = _candidates(grid, models)
        for cid, settings in candidates.items():
            search.evaluate(cid, settings)
        chosen = search.winner()
    except BudgetExhausted as stop:
        done = search.ledger["candidates"]
        pending = [cid for cid in candidates if cid not in done or done[cid]["verdict"] is None]
        report = _report(identity, "budget_exhausted", search, budget, pending=pending,
                         extra={"exhausted": stop.reason, "unevaluated_candidates": pending,
                                "resume": str(staging)})
        write_json(staging / "report.json", report)
        return report
    except BaseException as error:
        write_json(staging / "failure.json", {"reason": failure_reason(error), "error": str(error),
                                              "type": type(error).__name__})
        raise
    # The cached artifacts must still be what every recorded measurement used.
    manifest = _verify_artifacts(staging)
    selected = search.ledger["candidates"][chosen]["settings"] if chosen else BASELINE
    status = "optimized" if chosen else "no_measured_gain"
    bundle = staging / "bundle"
    suffix = {"prepared": "model.onnx", "original": "original.onnx"}[selected["model"]]
    session = {"schema_version": SCHEMA_VERSION, "target": "cpu", "providers": ["CPUExecutionProvider"],
               "session_options": {key: selected[key] for key in
                                   ("graph_optimization_level", "execution_mode", "intra_op_num_threads")},
               "models": {name: f"{name}/{suffix}" for name in profiles}, "status": status}
    session["policy_identity"] = hashlib.sha256(json.dumps(
        {"identity": identity, "session": session, "artifacts": manifest},
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    write_json(bundle / "session.json", session)
    report = _report(identity, status, search, budget, chosen=chosen, selected_settings=selected,
                     extra={"policy_identity": session["policy_identity"], "artifacts": manifest,
                            "confirmation": search.ledger["confirmations"].get(chosen) if chosen else None})
    write_json(bundle / "cpu-optimization.json", report)
    if destination.exists():
        raise FileExistsError(destination)
    bundle.rename(destination)
    shutil.rmtree(staging)
    return report


__all__ = ["BudgetExhausted", "judge", "load_workload", "optimize_cpu", "thread_grid", "usable_cpus"]
