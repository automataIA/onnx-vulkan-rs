"""Versioned Vulkan workload contract: what an automatic search may optimize, and its limits.

A workload file names the model, the profiles it must serve with their weights and
regression bounds, how application quality is measured, the quantization options to
try and the budget of the search. ``validate`` rejects anything unknown or unbounded
before costly work. ``plan`` reports readiness and the bounded work a search would do,
without running a model or touching a device. ``search_arguments`` turns a validated
workload into the keyword arguments of ``search_quantization``.

The CPU session-settings search has its own file schema (``cpu.load_workload``); the two
are deliberately separate commands and schemas.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from . import search

SCHEMA_VERSION = 1
BACKENDS = ("standalone",)
OBJECTIVES = ("weighted_median_wall",)
QUALITY_STATUSES = ("quality_unmeasured",)
KEYS = {"schema_version", "name", "source", "adapter", "backend", "objective", "profiles", "tolerances", "quality",
        "quantization", "budget"}
PROFILE_KEYS = {"dimensions", "weight", "max_regression"}
BUDGET_KEYS = {"seconds", "attempts", "top_k", "samples", "screen_samples", "timeout"}
QUANTIZATION_KEYS = {"config", "grid"}
# Upper limits keep an automatic search bounded even with a typo such as an extra zero.
MAX_SECONDS = 7 * 24 * 3600
MAX_ATTEMPTS = 1000
MAX_SAMPLES = 1000
MAX_TIMEOUT = 24 * 3600
DEFAULT_BUDGET = {"top_k": 3, "samples": 20, "screen_samples": 5, "timeout": 300}


def _object(value: Any, label: str, allowed: set[str], required: set[str] = frozenset()) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label}: unknown keys {unknown}")
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{label}: missing keys {missing}")
    return value


def _number(value: Any, label: str, *, minimum: float, maximum: float | None = None, integer: bool = False,
            strict_minimum: bool = False) -> float:
    kind = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kind) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite {'integer' if integer else 'number'}")
    if value < minimum or (strict_minimum and value == minimum) or (maximum is not None and value > maximum):
        bound = f"> {minimum:g}" if strict_minimum else f">= {minimum:g}"
        raise ValueError(f"{label} must be {bound}" + ("" if maximum is None else f" and <= {maximum:g}"))
    return value


def _path(value: Any, label: str, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty path")
    return (base / value).resolve()


def validate(raw: Any, *, base: Path | str = ".") -> dict[str, Any]:
    """Check a workload object and return a normalized copy; relative paths resolve against ``base``.

    Only the shape and bounds are checked here (no file is opened): ``plan`` reports whether the
    referenced files are usable."""
    base = Path(base)
    workload = _object(raw, "workload", KEYS, {"schema_version", "source", "profiles", "budget"})
    if workload["schema_version"] != SCHEMA_VERSION or isinstance(workload["schema_version"], bool):
        raise ValueError(f"unsupported workload schema_version {workload['schema_version']!r}; expected {SCHEMA_VERSION}")
    backend = workload.get("backend", "standalone")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}; the search measures the standalone runtime")
    objective = workload.get("objective", OBJECTIVES[0])
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {', '.join(OBJECTIVES)}")
    name = workload.get("name")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("name must be a nonempty string")

    profiles = workload["profiles"]
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("profiles must name at least one profile")
    dimensions, weights, bounds = {}, {}, {}
    for profile, spec in profiles.items():
        label = f"profile {profile}"
        if not isinstance(profile, str) or not search.PROFILE_NAME.fullmatch(profile):
            raise ValueError("profile names must contain only letters, digits, underscores or hyphens")
        spec = _object(spec, label, PROFILE_KEYS, {"dimensions", "weight"})
        dims = spec["dimensions"]
        if not isinstance(dims, Mapping):
            raise ValueError(f"{label} dimensions must be an object")
        for symbol, size in dims.items():
            _number(size, f"{label} dimension {symbol}", minimum=0, integer=True, strict_minimum=True)
        dimensions[profile] = dict(dims)
        weights[profile] = float(_number(spec["weight"], f"{label} weight", minimum=0, strict_minimum=True))
        bounds[profile] = float(_number(spec.get("max_regression", 0.0), f"{label} max_regression", minimum=0.0))

    tolerances = _object(workload.get("tolerances", {}), "tolerances", {"atol", "rtol"})
    atol = float(_number(tolerances.get("atol", 1e-5), "atol", minimum=0.0))
    rtol = float(_number(tolerances.get("rtol", 1e-4), "rtol", minimum=0.0))

    budget = _object(workload["budget"], "budget", BUDGET_KEYS, {"seconds"})
    limits = {
        "budget_seconds": float(_number(budget["seconds"], "budget seconds", minimum=0, maximum=MAX_SECONDS,
                                        strict_minimum=True)),
        "attempts": None if budget.get("attempts") is None else _number(
            budget["attempts"], "budget attempts", minimum=1, maximum=MAX_ATTEMPTS, integer=True),
        "top_k": _number(budget.get("top_k", DEFAULT_BUDGET["top_k"]), "budget top_k", minimum=1,
                         maximum=MAX_ATTEMPTS, integer=True),
        "samples": _number(budget.get("samples", DEFAULT_BUDGET["samples"]), "budget samples",
                           minimum=search.MIN_PAIRS, maximum=MAX_SAMPLES, integer=True),
        "screen_samples": _number(budget.get("screen_samples", DEFAULT_BUDGET["screen_samples"]),
                                  "budget screen_samples", minimum=1, maximum=MAX_SAMPLES, integer=True),
        "timeout": float(_number(budget.get("timeout", DEFAULT_BUDGET["timeout"]), "budget timeout", minimum=0,
                                 maximum=MAX_TIMEOUT, strict_minimum=True)),
    }

    quality = workload.get("quality")
    if quality is not None:
        quality = _object(quality, "quality", {"config", "status"})
        if ("config" in quality) == ("status" in quality):
            raise ValueError("quality names exactly one of config (a quality config file) or status "
                             f"({', '.join(QUALITY_STATUSES)})")
        if "status" in quality and quality["status"] not in QUALITY_STATUSES:
            raise ValueError(f"quality status must be {', '.join(QUALITY_STATUSES)}")
        quality = ({"config": _path(quality["config"], "quality config", base)} if "config" in quality
                   else {"status": quality["status"]})

    quantization = workload.get("quantization")
    if quantization is not None:
        quantization = _object(quantization, "quantization", QUANTIZATION_KEYS, {"config"})
        grid = quantization.get("grid")
        if grid is not None and (not isinstance(grid, list) or not grid
                                 or not all(isinstance(item, Mapping) for item in grid)):
            raise ValueError("quantization grid must be a nonempty list of objects")
        quantization = {"config": _path(quantization["config"], "quantization config", base),
                        "grid": None if grid is None else [dict(item) for item in grid]}

    return {
        "schema_version": SCHEMA_VERSION, "name": name,
        "source": _path(workload["source"], "source", base),
        "adapter": None if workload.get("adapter") is None else _path(workload["adapter"], "adapter", base),
        "backend": backend, "objective": objective, "profiles": dimensions, "weights": weights,
        "max_regression": bounds, "atol": atol, "rtol": rtol, "quality": quality, "quantization": quantization,
        **limits,
    }


def load(path: Path | str) -> dict[str, Any]:
    """Read and validate a workload file; relative paths resolve against its directory."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON: {error}") from error
    try:
        return validate(raw, base=path.resolve().parent)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text())
    except OSError as error:
        raise ValueError(f"{label} {path}: {error.strerror or error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} {path}: not valid JSON: {error}") from error


def _check_files(workload: Mapping[str, Any]) -> list[str]:
    """Problems that would stop a search before its first candidate, as messages."""
    problems = []
    source = workload["source"]
    if source.suffix.lower() != ".onnx" or not source.is_file():
        problems.append(f"source is not an existing .onnx file: {source}")
    adapter = workload["adapter"]
    if adapter is not None and not adapter.is_file():
        problems.append(f"adapter does not exist: {adapter}")
    return problems


def search_arguments(workload: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments of ``search_quantization`` for a validated workload with a quantization section.

    Reads the quantization config (and the optional quality config) files; opens nothing else."""
    quantization = workload["quantization"]
    if quantization is None:
        raise ValueError("the workload has no quantization section: there is nothing to search")
    quality = workload["quality"]
    if quality is not None and "status" in quality:
        raise ValueError("quality is quality_unmeasured: a lossy quantization candidate cannot be promoted "
                         "without a measured application metric")
    base = _read_json(quantization["config"], "quantization config")
    if not isinstance(base, dict):
        raise ValueError(f"quantization config {quantization['config']}: must be an object")
    return {
        "source": workload["source"], "base": base, "grid": quantization["grid"], "adapter": workload["adapter"],
        "profiles": workload["profiles"], "atol": workload["atol"], "rtol": workload["rtol"],
        "samples": workload["samples"], "screen_samples": workload["screen_samples"],
        "timeout": workload["timeout"], "top_k": workload["top_k"], "attempts": workload["attempts"],
        "budget_seconds": workload["budget_seconds"], "profile_weights": workload["weights"],
        "max_profile_regression": workload["max_regression"],
        "quality": None if quality is None else _read_json(quality["config"], "quality config"),
    }


def plan(workload: Mapping[str, Any]) -> dict[str, Any]:
    """Readiness and the bounded work of a search, without running any model or touching a device.

    ``ready`` is true only when ``missing`` is empty. The plan never promises a result:
    ``worst_case`` is the most a search may spend (screenings, confirmations, seconds)."""
    missing = _check_files(workload)
    quality = workload["quality"]
    quantization = workload["quantization"]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "name": workload["name"], "backend": workload["backend"],
        "objective": workload["objective"], "source": str(workload["source"]),
        "profiles": {name: {"dimensions": workload["profiles"][name], "weight": workload["weights"][name] / sum(
            workload["weights"].values()), "max_regression": workload["max_regression"][name]}
                     for name in workload["profiles"]},
        "quality": {"status": "quality_unmeasured" if quality and "status" in quality else
                    "unspecified" if quality is None else "configured"},
        "candidates": [], "worst_case": None,
    }
    candidates = []
    if quantization is None:
        result["notes"] = ["no quantization section: nothing to search; the profiles apply to prepare/optimize"]
    else:
        try:
            base = _read_json(quantization["config"], "quantization config")
            if not isinstance(base, dict):
                raise ValueError(f"quantization config {quantization['config']}: must be an object")
            candidates = search.plan_grid(base, quantization["grid"], atol=workload["atol"], rtol=workload["rtol"])
            for key in search.PATH_KEYS:
                if key in base and not Path(base[key]).exists():
                    missing.append(f"quantization {key} does not exist: {base[key]}")
            if quality is not None and "config" in quality:
                from . import quality as quality_module

                config = quality_module.validate_config(_read_json(quality["config"], "quality config"))
                quality_module.check_quantization_compatible(config, base)
        except (ValueError, OSError) as error:
            missing.append(str(error))
        if quality is not None and "status" in quality:
            missing.append("quality is quality_unmeasured: lossy quantization candidates need a measured "
                           "application metric and would not be promoted")
    result["candidates"] = [candidate["name"] for candidate in candidates]
    if candidates:
        profiles, top_k = len(workload["profiles"]), min(workload["top_k"], len(candidates))
        screening = len(candidates)
        attempts_needed = screening + top_k
        limit = workload["attempts"]
        result["worst_case"] = {
            "screening_candidates": screening, "screening_profile_runs": screening * profiles,
            "screening_timed_iterations": workload["screen_samples"],
            "confirmations": top_k, "confirmation_pairs_per_profile": workload["samples"],
            "attempts_needed": attempts_needed, "attempts_limit": limit,
            "complete_within_attempts": limit is None or limit >= attempts_needed,
            "seconds_limit": workload["budget_seconds"], "per_process_timeout": workload["timeout"],
            "claims": search.CLAIMS,
        }
        if limit is not None and limit < attempts_needed:
            result["notes"] = [f"attempts ({limit}) cannot cover all {screening} candidates and {top_k} "
                               "confirmations: the first invocation ends budget_exhausted and a resume continues"]
    result["missing"] = missing
    result["ready"] = not missing
    return result
