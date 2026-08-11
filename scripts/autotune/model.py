#!/usr/bin/env -S uv run --quiet --script
"""Tune every currently supported exact workload in one concrete ONNX profile."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TUNE_BIN = ROOT / "target" / "release" / "onnx-vulkan-tune"
CONV_BIN = ROOT / "target" / "release" / "examples" / "conv_blocked"


def command(arguments: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )


def build() -> None:
    command(["cargo", "build", "--release", "-p", "onnx-vulkan-tune"], capture=False)
    command(
        ["cargo", "build", "--release", "-p", "onnx-vulkan-core", "--example", "conv_blocked"],
        capture=False,
    )


def geometry_key(geometry: dict[str, Any]) -> str:
    names = ("c_in", "c_out", "kernel", "h_in", "h_out", "stride", "pad")
    return ",".join(str(geometry[name]) for name in names)


def read_json_output(arguments: list[str]) -> dict[str, Any]:
    result = command(arguments)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON from {' '.join(arguments)}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError(f"unsupported response from {' '.join(arguments)}")
    return value


def append_state(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())


def load_state(
    path: Path, model_digest: str, device: dict[str, Any], implementation: str
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{path}:{number}: malformed JSONL: {error}") from error
        if (
            row.get("schema_version") == 1
            and row.get("model_digest") == model_digest
            and row.get("device") == device
            and row.get("implementation_digest") == implementation
            and isinstance(row.get("geometry_key"), str)
            and isinstance(row.get("result"), dict)
        ):
            rows[row["geometry_key"]] = row["result"]
    return rows


def tune_missing(
    geometries: list[dict[str, Any]],
    state: Path,
    model_digest: str,
    device: dict[str, Any],
    implementation: str,
) -> dict[str, dict[str, Any]]:
    results = load_state(state, model_digest, device, implementation)
    missing = [geometry for geometry in geometries if geometry_key(geometry) not in results]
    if not missing:
        return results
    arguments = [str(CONV_BIN)]
    for geometry in missing:
        arguments.extend(["--tune-shape", geometry_key(geometry)])
    process = subprocess.Popen(
        arguments,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("Conv runner stdout is unavailable")
    try:
        for line in process.stdout:
            result = json.loads(line)
            if result.get("schema_version") != 1 or result.get("family") != "conv-f32":
                raise RuntimeError("Conv runner returned an unsupported response")
            if result.get("device") != device or result.get("implementation_digest") != implementation:
                raise RuntimeError("Conv runner identity changed during the tuning session")
            key = geometry_key(result["geometry"])
            if key not in {geometry_key(item) for item in missing}:
                raise RuntimeError(f"Conv runner returned unexpected geometry {key}")
            results[key] = result
            append_state(
                state,
                {
                    "schema_version": 1,
                    "model_digest": model_digest,
                    "device": device,
                    "implementation_digest": implementation,
                    "geometry_key": key,
                    "result": result,
                },
            )
    finally:
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Conv runner exited with status {return_code}")
    absent = [geometry_key(item) for item in missing if geometry_key(item) not in results]
    if absent:
        raise RuntimeError(f"Conv runner omitted geometries: {', '.join(absent)}")
    return results


def artifact_from(
    inventory: dict[str, Any],
    device: dict[str, Any],
    results: dict[str, dict[str, Any]],
    untunable_conv_nodes: int = 0,
) -> dict[str, Any]:
    records = []
    for item in inventory["workloads"]:
        result = results[geometry_key(item["runner"])]
        records.append(
            {
                "workload": item["workload"],
                "implementation_digest": item["implementation_digest"],
                "tactic": result["tactic"],
                "measurement": result["measurement"],
                "correctness": {
                    "kind": result["correctness"]["kind"],
                    "passed": result["correctness"]["passed"],
                },
            }
        )
    return {
        "schema_version": 1,
        "created_by": {},
        "device": device,
        "profile": inventory["profile"],
        "measurement_policy": {
            "warmup_iterations": 1,
            "measured_iterations": 20,
            "statistic": "median",
            "clock": "vulkan_timestamp",
        },
        "records": records,
        "diagnostics": [],
        "coverage": {
            "scope": "conv-f32-v1",
            "inventory_workloads": len(records) + untunable_conv_nodes,
            "tuned_workloads": len(records),
            "untunable_nodes": untunable_conv_nodes,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="concrete ONNX model")
    parser.add_argument("--output", type=Path, required=True, help="production artifact JSON")
    parser.add_argument("--dim", action="append", default=[], metavar="SYMBOL=N")
    parser.add_argument("--state", type=Path, help="resumable JSONL journal")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if any Conv node is outside the current tunable slice",
    )
    parser.add_argument("--no-build", action="store_true", help="reuse existing release binaries")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = args.model.resolve()
    output = args.output.resolve()
    state = (args.state or output.with_suffix(".jsonl")).resolve()
    try:
        if not args.no_build:
            build()
        inventory_args = [str(TUNE_BIN), "model-inventory", "--model", str(model)]
        for dimension in args.dim:
            inventory_args.extend(["--dim", dimension])
        inventory = read_json_output(inventory_args)
        skipped_conv = [item for item in inventory["skipped"] if item["op"] == "Conv"]
        if args.strict and skipped_conv:
            reasons = "; ".join(f"{item['node']}: {item['reason']}" for item in skipped_conv)
            raise RuntimeError(f"strict profile has untunable Conv nodes: {reasons}")
        if not inventory["workloads"]:
            raise RuntimeError("model contains no Conv workload supported by the current tuner")
        identity = read_json_output([str(CONV_BIN), "--device-json"])
        device = identity["device"]
        implementation = identity["implementation_digest"]
        if any(item["implementation_digest"] != implementation for item in inventory["workloads"]):
            raise RuntimeError("inventory and Conv runner implementation fingerprints differ")
        unique = {
            geometry_key(item["runner"]): item["runner"] for item in inventory["workloads"]
        }
        results = tune_missing(
            list(unique.values()),
            state,
            inventory["model_digest"],
            device,
            implementation,
        )
        draft = artifact_from(inventory, device, results, len(skipped_conv))
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", dir=output.parent, delete=False
        ) as temporary:
            json.dump(draft, temporary, sort_keys=True)
            temporary.write("\n")
            draft_path = Path(temporary.name)
        try:
            command(
                [
                    str(TUNE_BIN),
                    "build",
                    "--input",
                    str(draft_path),
                    "--output",
                    str(output),
                    "--model",
                    str(model),
                ]
            )
            command([str(TUNE_BIN), "validate", str(output)])
        finally:
            draft_path.unlink(missing_ok=True)
        summary = {
            "schema_version": 1,
            "status": "built",
            "artifact": str(output),
            "state": str(state),
            "device": device["name"],
            "records": len(inventory["workloads"]),
            "distinct_geometries": len(unique),
            "untunable_conv_nodes": len(skipped_conv),
            "other_fallback_nodes": len(inventory["skipped"]) - len(skipped_conv),
            "runtime_mode": "require-cache" if not skipped_conv else "cache-only",
        }
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"model autotune failed: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
