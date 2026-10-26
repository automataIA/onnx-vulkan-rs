"""Bounded, resumable grid search over quantization options, gated by the application metric.

Phases (every one persists ``search.json`` after each completed step):

1. *Screening.* Each candidate goes through the ordinary services: ``prepare_bundle`` (CPU
   metric gate and Rust compatibility), then ``run_bundle`` on every profile with
   ``screen_samples`` timed iterations (Vulkan outputs against the quantized CPU reference and
   the metric on Vulkan outputs). Passing candidates are ranked by the weighted median
   standalone wall time of their profiles. Those runs are separate and few, so the ranking
   only orders candidates; it is never an acceleration claim.
2. *Confirmation.* The ``top_k`` best candidates (nondominated ones first) are each handed to
   ``optimize_bundle``, whose interleaved comparison against the fp32 baseline is the only
   acceleration claim. A candidate that fails confirmation does not end the search. A
   candidate is promoted only if some profile is accelerated, every profile stays within its
   regression bound and the application quality of the deployed model was verified on Vulkan.
   The best promoted candidate by weighted confirmed latency becomes ``selected/``.

A global monotonic deadline (``budget_seconds``) and an attempt count bound the invocation.
Exhausting either is ``budget_exhausted``, distinct from a rejection or ``no_measured_gain``;
the experiment directory stays resumable. A resume reuses completed work only when the
content identity of every input is unchanged (see ``_identity``); otherwise it fails.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import statistics
import subprocess
import time
from pathlib import Path

from . import policy, process
from .pipeline import (MIN_PAIRS, REQUIRED_GAIN, _device, digest, optimize_bundle, prepare_bundle, run_bundle,
                       tool_identity, verify_bundle, write_json)
from .process import failure_reason, gpu_lock
from .quantize import MODE_KEYS, _validate_config

# Op types that only a quantized graph contains: a candidate without any of
# them quantized nothing and would only measure the fp32 model again.
QUANTIZED_OPS = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "QLinearConv",
                 "QLinearMatMul", "QLinearAdd", "QLinearGlobalAveragePool", "ConvInteger",
                 "MatMulInteger", "MatMulNBits"}
SCHEMA_VERSION = 2
IDENTITY_VERSION = 1
# Quantization config keys that name files or directories: their content, not their path, is identity.
PATH_KEYS = ("calibration", "evaluation", "adapter")
TERMINAL = ("promoted", "no_measured_gain", "no_passing_candidate")
PROFILE_NAME = re.compile(r"[A-Za-z0-9_-]+")
CLAIMS = {
    "screening": "separate, non-interleaved runs of a few iterations: orders candidates, no speed claim",
    "acceleration": "only the interleaved optimize_bundle comparison against the fp32 baseline, per profile",
    "application_quality": "the metric recomputed on Vulkan outputs; a lossy candidate without it is never promoted",
}


class StaleExperiment(ValueError):
    """Recorded evidence does not belong to this invocation; it is never reused."""


def default_grid(base):
    """Static forms need the calibration set; dynamic and int4 never do."""
    grid = []
    if "calibration" in base:
        grid += [{"mode": "static", "format": format, "per_channel": per_channel}
                 for format in ("QOperator", "QDQ") for per_channel in (False, True)]
    grid += [{"mode": "dynamic", "per_channel": per_channel} for per_channel in (False, True)]
    grid += [{"mode": "int4", "block_size": block_size} for block_size in (32, 128)]
    return grid


def candidate_config(base, overrides):
    """Merge ``overrides`` into ``base`` without keys the chosen mode rejects."""
    mode = overrides.get("mode", base.get("mode", "static"))
    if mode not in MODE_KEYS:
        raise ValueError(f"mode must be one of {', '.join(MODE_KEYS)}")
    foreign = {key for keys in MODE_KEYS.values() for key in keys} - set(MODE_KEYS[mode])
    config = {key: value for key, value in base.items() if key not in foreign}
    config.update(overrides)
    config["mode"] = mode
    return config


def candidate_name(overrides):
    parts = []
    for key in sorted(overrides):
        value = overrides[key]
        if isinstance(value, bool):
            parts.append(key if value else f"no_{key}")
        elif isinstance(value, (str, int)):
            parts.append(str(value) if key in ("mode", "format", "calibrate_method") else f"{key}{value}")
        else:
            raise ValueError(f"Grid value for {key} must be a string, integer or boolean")
    name = "-".join(parts)
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError(f"Cannot derive a directory name from {overrides}")
    return name


def plan_grid(base, grid=None, *, atol=1e-5, rtol=1e-4):
    """The validated candidates of a search, without touching any model or device.

    Every candidate's merged configuration is checked by the quantizer's own validator, so a
    typo in the grid is an error before any work starts, not one rejection among many."""
    grid = default_grid(base) if grid is None else grid
    if not isinstance(grid, list) or not grid or not all(isinstance(item, dict) for item in grid):
        raise ValueError("The quantization grid must be a nonempty list of objects")
    names = [candidate_name(item) for item in grid]
    if len(set(names)) != len(names):
        raise ValueError("The quantization grid contains duplicate candidates")
    candidates = []
    for name, overrides in zip(names, grid):
        config = candidate_config(base, overrides)
        try:
            _validate_config(config, atol=atol, rtol=rtol)
        except ValueError as error:
            raise ValueError(f"candidate {name}: {error}") from error
        candidates.append({"name": name, "overrides": overrides, "config": config})
    return candidates


def divergence_detail(error):
    """The worst output of a CPU/Vulkan divergence, or an empty string."""
    rows = re.finditer(r"^\s*output (?P<name>\S+)\s.*?max\|Δ\|=(?P<delta>\S+).*?relative=(?P<relative>\S+)",
                       str(error), re.MULTILINE)
    worst = max(rows, key=lambda row: float(row["relative"]), default=None)
    return "" if worst is None else (f"; worst output {worst['name']}: max|Δ|={worst['delta']}, "
                                     f"relative={worst['relative']}")


def quantized_op_count(model):
    import onnx

    graph = onnx.load(str(model), load_external_data=False).graph
    return sum(node.op_type in QUANTIZED_OPS for node in graph.node)


def device_identity(timeout):
    """Identity of the device the runtime picks. Queried only once GPU work is due."""
    return _device(timeout)


# ----------------------------------------------------------------------------------------------
# Validation of the search specification (before any directory, digest or subprocess)
# ----------------------------------------------------------------------------------------------

def _count(value, label, *, minimum=1, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _seconds(value, label, *, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number of seconds")
    return float(value)


def normalized_weights(profiles, weights):
    """Per-profile weights summing to one. Equal when ``weights`` is None."""
    if weights is None:
        weights = {name: 1.0 for name in profiles}
    if not isinstance(weights, dict) or set(weights) != set(profiles):
        raise ValueError(f"profile_weights must name exactly the profiles {sorted(profiles)}")
    for name, weight in weights.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"weight of profile {name} must be finite and positive")
    total = sum(weights.values())
    return {name: weights[name] / total for name in profiles}


def regression_bounds(profiles, bound):
    """Largest tolerated relative latency regression of each profile (0.0: none)."""
    bounds = bound if isinstance(bound, dict) else {name: bound for name in profiles}
    if set(bounds) != set(profiles):
        raise ValueError(f"max_profile_regression must name exactly the profiles {sorted(profiles)}")
    for name, value in bounds.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"max_profile_regression of profile {name} must be finite and nonnegative")
    return {name: float(bounds[name]) for name in profiles}


def _profiles(profiles):
    from .pipeline import dimensions

    profiles = profiles or {"default": {}}
    if not isinstance(profiles, dict):
        raise ValueError("profiles must map names to dimension objects")
    for name, dims in profiles.items():
        if not isinstance(name, str) or not PROFILE_NAME.fullmatch(name):
            raise ValueError("Profile names must contain only letters, digits, underscores or hyphens")
        dimensions(dims)
    return {name: dict(dims) for name, dims in profiles.items()}


# ----------------------------------------------------------------------------------------------
# Content identity of a search
# ----------------------------------------------------------------------------------------------

def _tree_digest(path):
    """Digest of a file, or of a directory's relative file names and contents."""
    path = Path(path)
    if path.is_file():
        return digest(path)
    if path.is_dir():
        listing = [f"{item.relative_to(path)}\0{digest(item)}" for item in sorted(path.rglob("*")) if item.is_file()]
        return hashlib.sha256("\n".join(listing).encode()).hexdigest()
    raise ValueError(f"{path} does not exist")


def _source_identity(source):
    source = Path(source)
    if not source.is_file():
        raise ValueError(f"source does not exist: {source}")
    identity = {"sha256": digest(source), "external_data": {}}
    if source.suffix == ".onnx":
        import onnx
        from .prepare import _external_locations

        model = onnx.load(str(source), load_external_data=False)
        identity["external_data"] = {str(location): digest(source.parent / location)
                                     for location in _external_locations(model)}
    return identity


def _identity(source, adapter, base, candidates, profiles, weights, bounds, atol, rtol, samples, screen_samples,
              tune, quality):
    """Everything the evidence of a search depends on, as JSON.

    Contents, not paths: the ONNX file and its external data, the adapter, the datasets
    and adapter the quantization config names, the options, the grid, the profiles with their
    weights and bounds, tolerances, sample counts and thresholds, the toolchain and the
    execution policies. The device is bound separately, when it is first measured. Invocation
    limits (budget, attempts, timeout, top_k) are absent on purpose: a resume may change them.
    """
    inputs = {key: _tree_digest(base[key]) for key in PATH_KEYS if key in base}
    identity = {
        "version": IDENTITY_VERSION,
        "source": _source_identity(source),
        "adapter": digest(adapter) if adapter else None,
        "quantization": {"options": {k: v for k, v in sorted(base.items()) if k not in PATH_KEYS}, "inputs": inputs},
        "quality": None if quality is None else {
            "options": {k: v for k, v in sorted(quality.items()) if k not in ("evaluation", "adapter")},
            "inputs": {key: _tree_digest(quality[key]) for key in ("evaluation", "adapter")}},
        "grid": [{"name": c["name"], "overrides": c["overrides"]} for c in candidates],
        "profiles": profiles, "weights": weights, "max_profile_regression": bounds,
        "tolerances": {"atol": atol, "rtol": rtol},
        "measurement": {"samples": samples, "screen_samples": screen_samples, "required_gain": REQUIRED_GAIN,
                        "min_pairs": MIN_PAIRS},
        "tune": tune, "tools": tool_identity(), "execution_policy": policy.pipeline_identity(),
        "measurement_plan": policy.measurement_plan(),
        "tuner_execution_policy": policy.resolve() if tune else None,
    }
    return json.loads(json.dumps(identity, allow_nan=False))


def _flatten(value, prefix=""):
    if isinstance(value, dict) and value:
        for key in sorted(value):
            yield from _flatten(value[key], f"{prefix}{key}.")
    else:
        yield prefix.rstrip("."), value


def identity_changes(recorded, current):
    """``component.key: old → new`` for every difference between two identities."""
    old, new = dict(_flatten(recorded)), dict(_flatten(current))
    return [f"{key}: {str(old.get(key))[:24]!r} → {str(new.get(key))[:24]!r}"
            for key in sorted({*old, *new}) if old.get(key) != new.get(key)]


# ----------------------------------------------------------------------------------------------
# Ranking, Pareto front and the confirmation verdict (pure functions)
# ----------------------------------------------------------------------------------------------

def _dominates(a, b):
    """``a`` is no worse on every objective both measured and better on one. Lower is better.
    A missing objective (None) is never compared, never invented."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return bool(pairs) and all(x <= y for x, y in pairs) and any(x < y for x, y in pairs)


def pareto_front(items, objectives):
    """The nondominated ``items`` in their original order."""
    scored = [(item, objectives(item)) for item in items]
    return [item for item, score in scored if not any(_dominates(other, score) for _, other in scored)]


def _screening_objectives(record):
    return record["weighted_wall_ms"], record["vulkan_drop"], record.get("storage_peak_bytes")


def _deployed_ratio(outcome):
    """Candidate/baseline median wall of a profile, from the uninstrumented confirmation when
    it ran, else from the timing pairs; None when neither was measured."""
    confirmation = (outcome.get("measurement") or {}).get("confirmation") or {}
    for base, candidate, source in (
            (confirmation.get("baseline"), confirmation.get("candidate"), "confirmation"),
            ((outcome.get("baseline") or {}).get("median_wall_ms"), (outcome.get("candidate") or {}).get(
                "median_wall_ms"), "timing")):
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
               for v in (base, candidate)):
            return candidate / base, source
    return None, None


def judge_confirmation(outcomes, weights, bounds, quality_verified):
    """The verdict of one candidate's ``optimize_bundle`` report.

    ``promoted``: some profile accelerated, every profile within its bound, quality verified
    on every accelerated profile. ``guard_failed``: some profile accelerated but another
    regressed past its bound or lacks evidence; the weighted objective never hides it.
    ``no_gain``: nothing accelerated. A profile that does not accelerate keeps the baseline
    model in the output bundle, so it contributes ratio 1 to the weighted objective; its
    measured candidate ratio is still held against its bound.
    """
    profiles, failures, weighted = {}, [], 0.0
    for name, weight in weights.items():
        outcome = outcomes.get(name) or {}
        accelerated = outcome.get("status") == "accelerated"
        ratio, source = _deployed_ratio(outcome)
        row = profiles[name] = {"status": outcome.get("status"), "weight": weight, "ratio": ratio,
                                "ratio_source": source, "bound": bounds[name]}
        if ratio is None:
            failures.append(f"{name}: no valid wall-clock medians")
        elif ratio - 1 > bounds[name]:
            failures.append(f"{name}: {ratio - 1:+.1%} latency against a bound of {bounds[name]:.1%}")
        if accelerated and not quality_verified.get(name):
            failures.append(f"{name}: application quality of the deployed model was not verified on Vulkan")
        row["deployed_ratio"] = ratio if accelerated else 1.0
        weighted += weight * row["deployed_ratio"] if ratio is not None or not accelerated else 0.0
    accelerated_any = any(row["status"] == "accelerated" for row in profiles.values())
    status = "no_gain" if not accelerated_any else "guard_failed" if failures else "promoted"
    return {"status": status, "weighted_ratio": weighted, "profiles": profiles, "guard_failures": failures}


# ----------------------------------------------------------------------------------------------
# The experiment
# ----------------------------------------------------------------------------------------------

def _new_report(identity, weights, bounds, limits):
    return {"schema_version": SCHEMA_VERSION, "status": "in_progress", "exhausted": None,
            "identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            "claims": CLAIMS, "weights": weights, "max_profile_regression": bounds, "device": None,
            "candidates": [], "unscreened": [], "ranking": [], "pareto": [], "shortlist": [],
            "confirmations": {}, "attempted": [], "proven_pareto": [], "promoted": None, "selected": None,
            "optimized": None,
            "budget": {**limits, "attempts_used": 0, "seconds_used": 0.0, "invocations": 0}}


class _Experiment:
    def __init__(self, destination, resume, report, candidates, params):
        self.destination, self.resume, self.report = destination, resume, report
        self.candidates, self.params = candidates, params
        self.records = {record["name"]: record for record in report["candidates"]}
        self.used = 0
        self.device_bound = False
        self.started = time.monotonic()
        self.seconds_before = report["budget"]["seconds_used"]
        report["budget"]["invocations"] += 1

    # -- persistence -----------------------------------------------------------------------
    def save(self):
        budget = self.report["budget"]
        budget["seconds_used"] = self.seconds_before + time.monotonic() - self.started
        budget["attempts_this_invocation"] = self.used
        self.report["unscreened"] = [c["name"] for c in self.candidates if not self._final(c["name"])]
        write_json(self.destination / "search.json", self.report)

    def _final(self, name):
        return name in self.records and self.records[name]["status"] in ("passed", "rejected")

    def _spend(self):
        limit = self.params["attempts"]
        if limit is not None and self.used >= limit:
            return False
        self.used += 1
        self.report["budget"]["attempts_used"] += 1
        return True

    def _bind_device(self):
        """Record the device before the first GPU work; refuse a different one on resume."""
        if self.device_bound:
            return
        timeout = self.params["timeout"]
        with gpu_lock(timeout):
            device = device_identity(timeout)
        recorded = self.resume.get("device")
        if recorded is not None and recorded != device:
            raise StaleExperiment(f"Stale search at {self.destination}: measured on {recorded}, "
                                  f"this device is {device}; choose a new output directory")
        if recorded is None:
            self.resume["device"] = device
            write_json(self.destination / "resume.json", self.resume)
        self.report["device"] = device
        self.device_bound = True

    # -- screening -------------------------------------------------------------------------
    def _sealed(self, label, path, expected):
        """The real `verify_bundle` of `path`, bound to the identity recorded when it was produced."""
        try:
            manifest = verify_bundle(path)
        except (ValueError, OSError) as error:
            raise StaleExperiment(f"Stale search: {label} at {path} cannot be verified ({error}); "
                                  "its evidence is not reused") from error
        if manifest["identity"] != expected:
            raise StaleExperiment(f"Stale search: {label} changed since it was recorded")
        return manifest

    def _unchanged(self, record):
        """The recorded bundle, which must still be the one that was measured."""
        return self._sealed(f"bundle of candidate {record['name']}", record["bundle"], record["bundle_identity"])

    def _unchanged_confirmation(self, name, entry):
        """The confirmed bundle: where it was written, or `selected/` if a promotion already moved it."""
        paths = [Path(entry["bundle"]), self.destination / "selected"]
        path = next((path for path in paths if path.exists()), paths[0])
        return self._sealed(f"confirmed bundle of candidate {name}", path, entry["bundle_identity"])

    def _verify_completed(self):
        """Completed evidence is reused only if its sealed artifacts are untouched.

        Runs before any attempt, device query or promotion, so a mutated artifact stops the resume
        before GPU work. Nothing is deleted or moved: the evidence stays where it is."""
        for record in self.records.values():
            if record["status"] == "passed" and "bundle_identity" not in record:
                raise StaleExperiment("Completed screening has no sealed bundle identity; start a new search")
            if record["status"] in ("passed", "rejected") and "bundle_identity" in record:
                self._unchanged(record)
        for name, entry in self.report["confirmations"].items():
            if entry.get("status") not in ("running", "failed") and "bundle_identity" not in entry:
                raise StaleExperiment("Completed confirmation has no sealed bundle identity; start a new search")
            if "bundle_identity" in entry:
                self._unchanged_confirmation(name, entry)

    def _prepared(self, record, bundle, config):
        if "bundle_identity" in record:
            return self._unchanged(record)
        if bundle.exists():
            raise StaleExperiment(f"{bundle} exists but its preparation was never recorded; move it aside to resume")
        manifest = prepare_bundle(self.params["source"], bundle, adapter=self.params["adapter"],
                                  profiles=self.params["profiles"], atol=self.params["atol"],
                                  rtol=self.params["rtol"], timeout=self.params["timeout"], quantization=config,
                                  **({} if self.params["quality"] is None else {"quality": self.params["quality"]}))
        record["bundle"] = str(bundle)
        record["bundle_identity"] = manifest["identity"]
        self.save()
        return manifest

    def screen(self, candidate):
        name = candidate["name"]
        record = self.records.get(name)
        if record is None:
            record = {"name": name, "overrides": candidate["overrides"], "status": "screening", "profiles": {}}
            self.records[name] = record
            self.report["candidates"].append(record)
        record["status"] = "screening"
        bundle = self.destination / "candidates" / name
        try:
            manifest = self._prepared(record, bundle, candidate["config"])
            preparation = {profile: json.loads((bundle / profile / "preparation.json").read_text())["quantization"]
                           for profile in manifest["profiles"]}
            record["cpu_drop"] = max(item["drop"] for item in preparation.values())
            if not manifest["compatibility_obtained"]:
                record["status"], record["reason"] = "rejected", "incompatible with the Vulkan pipeline; see compatibility.json"
                return
            if not all(quantized_op_count(bundle / profile["model"]) for profile in manifest["profiles"].values()):
                targets = sorted({op for item in preparation.values() for op in item["options"]["op_types_to_quantize"]})
                record["status"] = "rejected"
                record["reason"] = (f"no operator was quantized: the model has no {'/'.join(targets)} "
                                    "node this mode quantizes, or all were excluded")
                return
            for profile in manifest["profiles"]:
                if profile in record["profiles"]:
                    continue
                self._bind_device()
                result = run_bundle(bundle, profile=profile, samples=self.params["screen_samples"],
                                    timeout=self.params["timeout"])
                quality = result["quality_evaluation"]
                if not quality.get("vulkan_verified"):
                    raise RuntimeError("Vulkan application quality was not verified")
                peak = result.get("standalone_storage_peak_bytes")
                # The first iteration includes pipeline creation and uploads.
                record["profiles"][profile] = {
                    "median_wall_ms": statistics.median(result["standalone_ms"][1:] or result["standalone_ms"]),
                    "vulkan_drop": quality["drop"],
                    "storage_peak_bytes": peak if isinstance(peak, (int, float)) and not isinstance(peak, bool) else None}
                self.save()
            measured = record["profiles"]
            record["weighted_wall_ms"] = sum(self.params["weights"][p] * measured[p]["median_wall_ms"] for p in measured)
            record["vulkan_drop"] = max(item["vulkan_drop"] for item in measured.values())
            peaks = [item["storage_peak_bytes"] for item in measured.values()]
            record["storage_peak_bytes"] = None if any(p is None for p in peaks) else max(peaks)
            record["status"] = "passed"
        except StaleExperiment:
            raise
        except subprocess.TimeoutExpired as error:
            # Operational, not a verdict on the candidate: a resume measures it again.
            record["status"], record["timed_out"] = "rejected", True
            record["reason"] = failure_reason(error) + divergence_detail(error)
        except (ValueError, RuntimeError, OSError) as error:
            record["status"] = "rejected"
            record["reason"] = failure_reason(error) + divergence_detail(error)
        finally:
            self.save()

    # -- ranking ---------------------------------------------------------------------------
    def rank(self):
        passed = sorted((r for r in self.records.values() if r["status"] == "passed"),
                        key=lambda r: (r["weighted_wall_ms"], r["vulkan_drop"], r["name"]))
        front = {r["name"] for r in pareto_front(passed, _screening_objectives)}
        ordered = [r for r in passed if r["name"] in front] + [r for r in passed if r["name"] not in front]
        self.report["ranking"] = [r["name"] for r in passed]
        self.report["pareto"] = [r["name"] for r in passed if r["name"] in front]
        self.report["shortlist"] = [r["name"] for r in ordered[:self.params["top_k"]]]
        self.save()

    # -- confirmation ----------------------------------------------------------------------
    def confirm(self, name):
        destination = self.destination / "confirmations" / name
        entry = self.report["confirmations"].get(name)
        if entry is None:
            entry = self.report["confirmations"][name] = {"status": "running", "bundle": str(destination)}
            self.report["attempted"].append(name)
        elif destination.exists():
            raise StaleExperiment(f"{destination} exists but its result was never recorded; move it aside to resume")
        self._unchanged(self.records[name])
        entry["status"] = "running"
        self.save()
        self._bind_device()
        destination.parent.mkdir(exist_ok=True)
        try:
            optimize_bundle(self.records[name]["bundle"], destination, samples=self.params["samples"],
                            timeout=self.params["timeout"], tune=self.params["tune"])
            entry["bundle_identity"] = verify_bundle(destination)["identity"]
            outcomes = json.loads((destination / "report.json").read_text())
            verified = {}
            for profile in self.params["weights"]:
                path = destination / profile / "validation.json"
                evaluation = json.loads(path.read_text()).get("quality_evaluation") if path.is_file() else None
                verified[profile] = bool((evaluation or {}).get("vulkan_verified"))
            verdict = judge_confirmation(outcomes, self.params["weights"], self.params["bounds"], verified)
            entry.update(verdict, outcomes=outcomes)
        except StaleExperiment:
            raise
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            entry.update(status="failed", reason=failure_reason(error) + divergence_detail(error))
        finally:
            self.save()

    # -- outcome ---------------------------------------------------------------------------
    def stop(self, reason, detail=None):
        self.report["status"], self.report["exhausted"] = "budget_exhausted", reason
        if detail:
            self.report["exhausted_detail"] = detail
        self.report["resume"] = str(self.destination)
        self.save()
        return self.report

    def run(self):
        self._verify_completed()
        for candidate in self.candidates:
            if self._final(candidate["name"]):
                continue
            if not self._spend():
                return self.stop("attempts", "attempts ended screening; resume with more attempts")
            self.screen(candidate)
        self.rank()
        for name in self.report["shortlist"]:
            if self.report["confirmations"].get(name, {}).get("status") not in (None, "running"):
                continue
            if not self._spend():
                return self.stop("attempts", "attempts ended confirmation; resume with more attempts")
            self.confirm(name)
        return self.finish()

    def finish(self):
        confirmations = self.report["confirmations"]
        proven = [self.records[n] | {"weighted_ratio": confirmations[n]["weighted_ratio"]}
                  for n in self.report["shortlist"] if confirmations[n]["status"] == "promoted"]
        self.report["proven_pareto"] = [r["name"] for r in pareto_front(
            proven, lambda r: (r["weighted_ratio"], r["vulkan_drop"], r.get("storage_peak_bytes")))]
        if proven:
            winner = min(proven, key=lambda r: (r["weighted_ratio"], r["vulkan_drop"], r["name"]))["name"]
            if tool_identity() != self.resume["identity"]["tools"]:
                raise StaleExperiment("Tools changed during the search; rerun in a new directory")
            # The bundle is about to become the result: it must be the one that was measured.
            self._unchanged(self.records[winner])
            self._unchanged_confirmation(winner, confirmations[winner])
            self._promote(winner)
            self.report["status"] = "promoted"
        else:
            self.report["status"] = "no_measured_gain" if self.report["ranking"] else "no_passing_candidate"
        self.save()
        return self.report

    def _promote(self, name):
        source, selected = self.destination / "confirmations" / name, self.destination / "selected"
        if source.exists():
            if selected.exists():
                raise StaleExperiment(f"{selected} already exists; move it aside to resume")
            source.rename(selected)
        elif not selected.exists():
            raise StaleExperiment(f"confirmed bundle of {name} is missing")
        entry = self.report["confirmations"][name]
        entry["bundle"] = str(selected)
        self.report["promoted"] = self.report["selected"] = name
        self.report["optimized"] = {"bundle": str(selected), "outcomes": entry["outcomes"],
                                    "acceleration_demonstrated": True}


def _open(destination, identity, new_report):
    """A new experiment directory, or the unfinished one with this identity."""
    resume_path, report_path = destination / "resume.json", destination / "search.json"
    if not destination.exists():
        destination.mkdir(parents=True)
        resume = {"identity": identity, "device": None}
        write_json(resume_path, resume)
        return resume, new_report
    if not (resume_path.is_file() and report_path.is_file()):
        raise FileExistsError(destination)
    report = json.loads(report_path.read_text())
    if report.get("status") in TERMINAL:
        raise FileExistsError(destination)
    resume = json.loads(resume_path.read_text())
    if resume.get("identity") != identity:
        raise StaleExperiment(f"Stale search at {destination} ("
                              + "; ".join(identity_changes(resume.get("identity", {}), identity)[:8])
                              + "); choose a new output directory")
    # A worker timeout is operational, not a verdict: the candidate is measured again.
    for record in report["candidates"]:
        if record.pop("timed_out", False):
            record["status"] = "screening"
            record.pop("reason", None)
    report["status"], report["exhausted"] = "in_progress", None
    report.pop("exhausted_detail", None)
    return resume, report


def search_quantization(source, destination, base, *, grid=None, adapter=None, profiles=None,
                        atol=1e-5, rtol=1e-4, samples=20, timeout=300, tune=True, top_k=3, attempts=None,
                        budget_seconds=None, screen_samples=5, profile_weights=None, max_profile_regression=0.0,
                        quality=None):
    """Search quantization options for ``source`` under a bounded, resumable budget.

    ``samples`` (>= 20) are the interleaved pairs of each confirmation; ``screen_samples`` the
    few timed iterations of a screening run. ``top_k`` candidates are confirmed. ``attempts``
    counts screened candidates plus confirmations in this invocation; ``budget_seconds`` is a
    monotonic deadline over the whole invocation, preparation, tuning, validation and GPU-lane
    waits included. ``quality`` is an independent quality config (see ``quality.validate_config``)
    that every candidate is also held to. ``profile_weights`` (default equal) define the objective;
    ``max_profile_regression`` (a number, or one per profile; default 0: none tolerated) bounds
    each profile's latency against the fp32 baseline. Returns the report; ``status`` is
    ``promoted``, ``no_measured_gain``, ``no_passing_candidate`` or ``budget_exhausted`` (the
    destination then stays resumable by calling again with the same inputs).
    """
    samples = _count(samples, "samples", minimum=MIN_PAIRS)
    screen_samples = _count(screen_samples, "screen_samples")
    top_k = _count(top_k, "top_k")
    attempts = _count(attempts, "attempts", allow_none=True)
    budget_seconds = _seconds(budget_seconds, "budget_seconds", allow_none=True)
    timeout = _seconds(timeout, "timeout")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    candidates = plan_grid(base, grid, atol=atol, rtol=rtol)
    profiles = _profiles(profiles)
    weights = normalized_weights(profiles, profile_weights)
    bounds = regression_bounds(profiles, max_profile_regression)
    destination = Path(destination).resolve()
    for key in PATH_KEYS:
        if key in base and not Path(base[key]).exists():
            raise ValueError(f"quantization {key} does not exist: {base[key]}")
    if adapter is not None and not Path(adapter).is_file():
        raise ValueError(f"adapter does not exist: {adapter}")
    if quality is not None:
        from . import quality as quality_module

        quality = quality_module.validate_config(quality)
        quality_module.check_quantization_compatible(quality, base)
    identity = _identity(source, adapter, base, candidates, profiles, weights, bounds, float(atol), float(rtol),
                         samples, screen_samples, tune, quality)
    limits = {"attempts_limit": attempts, "seconds_limit": budget_seconds}
    resume, report = _open(destination, identity, _new_report(identity, weights, bounds, limits))
    report["budget"].update(limits)
    experiment = _Experiment(destination, resume, report, candidates, {
        "source": source, "adapter": adapter, "profiles": profiles, "atol": atol, "rtol": rtol,
        "samples": samples, "screen_samples": screen_samples, "timeout": timeout, "tune": tune, "top_k": top_k,
        "attempts": attempts, "weights": weights, "bounds": bounds,
        "quality": quality})
    scope = process.deadline(budget_seconds) if budget_seconds is not None else contextlib.nullcontext()
    try:
        with scope:
            return experiment.run()
    except process.BudgetExhausted as stop:
        return experiment.stop("seconds", str(stop))
    finally:
        experiment.save()
