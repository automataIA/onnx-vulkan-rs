"""Grid search over quantization options, gated by the application metric.

Every candidate goes through the ordinary services: ``prepare_bundle``
(CPU metric gate and Rust compatibility), then ``run_bundle`` (Vulkan outputs
against the quantized CPU reference and the metric on Vulkan outputs). Passing
candidates are ranked by their median standalone wall time. Those runs are not
interleaved, so the ranking only orders candidates; the winner is handed to
``optimize_bundle``, whose interleaved comparison against the fp32 baseline is
the only acceleration claim.
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
from pathlib import Path

from .pipeline import optimize_bundle, prepare_bundle, run_bundle, write_json
from .process import failure_reason
from .quantize import MODE_KEYS

# Op types that only a quantized graph contains: a candidate without any of
# them quantized nothing and would only measure the fp32 model again.
QUANTIZED_OPS = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "QLinearConv",
                 "QLinearMatMul", "QLinearAdd", "QLinearGlobalAveragePool", "ConvInteger",
                 "MatMulInteger", "MatMulNBits"}


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


def search_quantization(source, destination, base, *, grid=None, adapter=None, profiles=None,
                        atol=1e-5, rtol=1e-4, samples=20, timeout=300, tune=True):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    grid = default_grid(base) if grid is None else grid
    if not isinstance(grid, list) or not grid or not all(isinstance(item, dict) for item in grid):
        raise ValueError("The quantization grid must be a nonempty list of objects")
    names = [candidate_name(item) for item in grid]
    if len(set(names)) != len(names):
        raise ValueError("The quantization grid contains duplicate candidates")
    destination.mkdir(parents=True)
    report = {"candidates": [], "selected": None, "optimized": None}
    for name, overrides in zip(names, grid):
        record = {"name": name, "overrides": overrides, "status": "rejected"}
        report["candidates"].append(record)
        bundle = destination / "candidates" / name
        try:
            manifest = prepare_bundle(source, bundle, adapter=adapter, profiles=profiles, atol=atol,
                                      rtol=rtol, timeout=timeout,
                                      quantization=candidate_config(base, overrides))
            record["bundle"] = str(bundle)
            preparation = {profile: json.loads((bundle / profile / "preparation.json").read_text())["quantization"]
                           for profile in manifest["profiles"]}
            record["cpu_drop"] = max(item["drop"] for item in preparation.values())
            if not manifest["compatibility_obtained"]:
                record["reason"] = "incompatible with the Vulkan pipeline; see compatibility.json"
                continue
            if not all(quantized_op_count(bundle / profile["model"]) for profile in manifest["profiles"].values()):
                targets = sorted({op for item in preparation.values()
                                  for op in item["options"]["op_types_to_quantize"]})
                record["reason"] = (f"no operator was quantized: the model has no {'/'.join(targets)} "
                                    "node this mode quantizes, or all were excluded")
                continue
            wall = []
            for profile in manifest["profiles"]:
                result = run_bundle(bundle, profile=profile, samples=samples, timeout=timeout)
                quality = result["quality_evaluation"]
                if not quality.get("vulkan_verified"):
                    raise RuntimeError("Vulkan application quality was not verified")
                record["vulkan_drop"] = max(record.get("vulkan_drop", quality["drop"]), quality["drop"])
                # The first iteration includes pipeline creation and uploads.
                wall.append(statistics.median(result["standalone_ms"][1:] or result["standalone_ms"]))
            record["median_wall_ms"] = sum(wall)
            record["status"] = "passed"
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            record["reason"] = failure_reason(error) + divergence_detail(error)
        finally:
            write_json(destination / "search.json", report)
    passed = [item for item in report["candidates"] if item["status"] == "passed"]
    if passed:
        # Wall time orders candidates; equal times prefer the smaller quality drop.
        best = min(passed, key=lambda item: (item["median_wall_ms"], item["vulkan_drop"]))
        report["selected"] = best["name"]
        write_json(destination / "search.json", report)
        optimized = optimize_bundle(best["bundle"], destination / "selected", samples=samples,
                                    timeout=timeout, tune=tune)
        outcomes = json.loads((destination / "selected" / "report.json").read_text())
        report["optimized"] = {"bundle": str(destination / "selected"), "outcomes": outcomes,
                               "acceleration_demonstrated": optimized["acceleration_demonstrated"]}
        write_json(destination / "search.json", report)
    return report
