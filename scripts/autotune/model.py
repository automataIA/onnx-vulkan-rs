#!/usr/bin/env -S uv run --quiet --script
"""Tune every currently supported exact workload in one concrete ONNX profile."""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TUNE_BIN = ROOT / "target" / "release" / "onnx-vulkan-tune"
CONV_BIN = ROOT / "target" / "release" / "examples" / "conv_blocked"
MATRIX_BIN = ROOT / "target" / "release" / "examples" / "matrix_tune"
MODEL_RUNNER = ROOT / "target" / "release" / "model-runner"
DEFAULT_RESPONSE_TIMEOUT = 30.0
DEFAULT_PROCESS_TIMEOUT = 900.0
# Exit status when the artifact is built but the tuned model misses the
# tolerance on the validation inputs; callers keep the default kernels.
VALIDATION_REJECTED = 4


def terminate_process(process: subprocess.Popen[Any], timeout: float = 2.0) -> None:
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait(timeout=timeout)


def command(
    arguments: list[str],
    *,
    capture: bool = True,
    timeout: float = DEFAULT_PROCESS_TIMEOUT,
    env: dict[str, str] | None = None,
    input_data: str | None = None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        arguments,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        stdin=subprocess.PIPE if input_data is not None else None,
        env=env,
        start_new_session=(os.name == "posix"),
    )
    try:
        stdout, stderr = process.communicate(input=input_data, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        terminate_process(process)
        process.communicate()
        raise RuntimeError(f"process timed out after {timeout:g}s: {' '.join(arguments)}") from error
    result = subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode, arguments, output=result.stdout, stderr=result.stderr
        )
    return result


def build() -> None:
    command(["cargo", "build", "--release", "-p", "onnx-vulkan-tune"], capture=False)
    command(
        ["cargo", "build", "--release", "-p", "onnx-vulkan-core", "--example", "conv_blocked"],
        capture=False,
    )
    command(
        ["cargo", "build", "--release", "-p", "onnx-vulkan-core", "--example", "matrix_tune"],
        capture=False,
    )
    command(["cargo", "build", "--release", "-p", "model-runner"], capture=False)


def geometry_key(geometry: dict[str, Any]) -> str:
    if geometry.get("family") == "matrix-f32":
        # Matrix runner geometries are deliberately keyed by their complete
        # canonical JSON object.  This keeps future matrix axes resumable
        # without silently aliasing two workloads.
        return json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False)
    # Keep the original key stable for resumable journals. Extended runners
    # expose the full rectangular/grouped geometry and use all of these axes.
    extended = any(name in geometry for name in ("group", "w_in", "w_out", "kernel_h", "kh"))
    if not extended:
        names = ("c_in", "c_out", "kernel", "h_in", "h_out", "stride", "pad")
        return ",".join(str(geometry[name]) for name in names)

    def value(*names: str, default: int = 1) -> Any:
        for name in names:
            if name in geometry:
                return geometry[name]
        return default

    values = (
        value("c_in"),
        value("c_out"),
        value("group"),
        value("kernel_h", "kh", "kernel"),
        value("kernel_w", "kw", "kernel_h", "kh", "kernel"),
        value("h_in"),
        value("w_in", default=value("h_in")),
        value("h_out"),
        value("w_out", default=value("h_out")),
        value("stride_h", "sh", "stride"),
        value("stride_w", "sw", "stride_h", "sh", "stride"),
        value("pad_top", "phb", "pad"),
        value("pad_bottom", "phe", "pad_top", "phb", "pad"),
        value("pad_left", "pwb", "pad"),
        value("pad_right", "pwe", "pad_left", "pwb", "pad"),
        value("dilation_h", "dh", default=1),
        value("dilation_w", "dw", "dilation_h", "dh", default=1),
    )
    if (
        values[2] == 1
        and values[3] == values[4]
        and values[5] == values[6]
        and values[7] == values[8]
        and values[9] == values[10]
        and values[11] == values[12] == values[13] == values[14]
        and values[15] == values[16] == 1
    ):
        return ",".join(str(values[index]) for index in (0, 1, 3, 5, 7, 9, 11))
    return ",".join(str(item) for item in values)


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
    path: Path,
    model_digest: str,
    device: dict[str, Any],
    implementation: str,
    family: str | None = None,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for number, line in enumerate(lines, 1):
        truncated = not line.endswith("\n")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            # Recover only a final record cut off by interruption. A malformed
            # completed record is retained as a hard failure for safe resume.
            if truncated and number == len(lines):
                continue
            raise RuntimeError(f"{path}:{number}: malformed JSONL: {error}") from error
        row_family = row.get("family", "conv-f32")
        if (
            row.get("schema_version") == 1
            and (family is None or row_family == family)
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
    *,
    family: str = "conv-f32",
    runner: Path | None = None,
    response_timeout: float = DEFAULT_RESPONSE_TIMEOUT,
    process_timeout: float = DEFAULT_PROCESS_TIMEOUT,
) -> dict[str, dict[str, Any]]:
    if response_timeout <= 0 or process_timeout <= 0:
        raise ValueError("tuning timeouts must be positive")
    results = load_state(state, model_digest, device, implementation, family)
    missing = [geometry for geometry in geometries if geometry_key(geometry) not in results]
    if not missing:
        return results
    runner = Path(runner or (CONV_BIN if family == "conv-f32" else MATRIX_BIN))
    arguments = [str(runner)]
    input_lines: list[bytes] | None = None
    if family == "conv-f32":
        for geometry in missing:
            arguments.extend(["--tune-shape", geometry_key(geometry)])
    elif family == "matrix-f32":
        input_lines = [
            (
                json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            ).encode("utf-8")
            for geometry in missing
        ]
    else:
        raise RuntimeError(f"unsupported tuning family {family!r}")
    process = subprocess.Popen(
        arguments,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE if input_lines is not None else None,
        bufsize=0,
        start_new_session=(os.name == "posix"),
    )
    if process.stdout is None:
        raise RuntimeError(f"{family} runner stdout is unavailable")
    writer_errors: list[BaseException] = []
    writer: threading.Thread | None = None
    if input_lines is not None and process.stdin is not None:
        def write_requests() -> None:
            try:
                for line in input_lines:
                    process.stdin.write(line)
                process.stdin.close()
            except (BrokenPipeError, ValueError):
                # The response loop reports missing geometries or the runner's
                # exit status with the useful context.  A runner that exits
                # after all responses is already complete.
                try:
                    process.stdin.close()
                except OSError:
                    pass
            except BaseException as error:  # pragma: no cover - OS-specific I/O
                writer_errors.append(error)

        writer = threading.Thread(target=write_requests, name="matrix-input", daemon=True)
        writer.start()
    started = time.monotonic()
    buffer = bytearray()
    expected = {geometry_key(item) for item in missing}

    def terminate() -> None:
        terminate_process(process)

    def read_line() -> bytes:
        deadline = min(started + process_timeout, time.monotonic() + response_timeout)
        fd = process.stdout.fileno()
        while True:
            newline = buffer.find(b"\n")
            if newline >= 0:
                line = bytes(buffer[: newline + 1])
                del buffer[: newline + 1]
                return line
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminate()
                raise RuntimeError(f"{family} runner response/process timeout")
            readable, _, _ = select.select([fd], [], [], remaining)
            if not readable:
                terminate()
                raise RuntimeError(f"{family} runner response/process timeout")
            chunk = os.read(fd, 65536)
            if chunk:
                buffer.extend(chunk)
            else:
                raise RuntimeError(f"{family} runner closed stdout before completing responses")
    try:
        while True:
            line = read_line()
            result = json.loads(line.decode("utf-8"))
            if result.get("schema_version") != 1 or result.get("family") != family:
                raise RuntimeError(f"{family} runner returned an unsupported response")
            if result.get("device") != device or result.get("implementation_digest") != implementation:
                raise RuntimeError(f"{family} runner identity changed during the tuning session")
            key = geometry_key(result["geometry"])
            if key not in expected:
                raise RuntimeError(f"{family} runner returned unexpected geometry {key}")
            results[key] = result
            append_state(
                state,
                {
                    "schema_version": 1,
                    "family": family,
                    "model_digest": model_digest,
                    "device": device,
                    "implementation_digest": implementation,
                    "geometry_key": key,
                    "result": result,
                },
            )
            if expected.issubset(results):
                break
    finally:
        if writer is not None:
            if writer.is_alive() and expected.issubset(results):
                # A misbehaving runner may emit all responses before consuming
                # the remaining request stream. Closing stdin lets the writer
                # unwind instead of waiting for a full pipe.
                try:
                    if process.stdin is not None:
                        process.stdin.close()
                except OSError:
                    pass
            if writer.is_alive() and process.poll() is None and not expected.issubset(results):
                terminate()
            writer.join(timeout=2.0)
        if process.poll() is None:
            try:
                process.wait(timeout=min(2.0, process_timeout))
            except subprocess.TimeoutExpired:
                terminate()
        if process.stdout is not None:
            process.stdout.close()
    return_code = process.returncode
    if writer_errors:
        raise RuntimeError("Matrix runner input failed") from writer_errors[0]
    if return_code != 0:
        raise RuntimeError(f"{family} runner exited with status {return_code}")
    absent = [geometry_key(item) for item in missing if geometry_key(item) not in results]
    if absent:
        raise RuntimeError(f"{family} runner omitted geometries: {', '.join(absent)}")
    return results


def artifact_from(
    inventory: dict[str, Any],
    device: dict[str, Any],
    results: dict[str, dict[str, Any]],
    untunable_conv_nodes: int = 0,
    untunable_nodes: int | None = None,
    family_devices: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    records_by_key: dict[str, dict[str, Any]] = {}
    for item in inventory["workloads"]:
        result = results[geometry_key(item["runner"])]
        record = {
            "workload": item["workload"],
            "implementation_digest": item["implementation_digest"],
            "tactic": result["tactic"],
            "measurement": result["measurement"],
            "correctness": {
                "kind": result["correctness"]["kind"],
                "passed": result["correctness"]["passed"],
            },
        }
        key = json.dumps(
            [record["workload"], record["implementation_digest"]],
            sort_keys=True,
            separators=(",", ":"),
        )
        previous = records_by_key.setdefault(key, record)
        if previous != record:
            raise RuntimeError(f"conflicting tuning results for exact workload {key}")
    records = list(records_by_key.values())
    artifact = {
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
        # Coverage counts distinct exact workloads, the unit of `records`;
        # counting occurrences would make a fully tuned model with a repeated
        # workload look partial, and require-cache would refuse it.
        "coverage": {
            "scope": "fp32-vision-v1",
            "inventory_workloads": len(records) + (
                untunable_conv_nodes if untunable_nodes is None else untunable_nodes
            ),
            "tuned_workloads": len(records),
            "untunable_nodes": untunable_conv_nodes if untunable_nodes is None else untunable_nodes,
        },
    }
    if family_devices:
        artifact["family_devices"] = family_devices
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="concrete ONNX model")
    parser.add_argument("--output", type=Path, required=True, help="production artifact JSON")
    parser.add_argument("--dim", action="append", default=[], metavar="SYMBOL=N")
    parser.add_argument(
        "--reference",
        type=Path,
        help="test_data_set directory (input_*.pb, output_*.pb) that validates the tuned model; "
        "default: generated inputs, compared with the CPU EP only",
    )
    parser.add_argument("--state", type=Path, help="resumable JSONL journal")
    parser.add_argument(
        "--priority-json",
        type=Path,
        help="optional JSON object mapping operation names to measured priority times",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if any Conv, MatMul, or Gemm node is outside the current tunable slice",
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=DEFAULT_RESPONSE_TIMEOUT,
        help="seconds allowed for each runner response",
    )
    parser.add_argument(
        "--process-timeout",
        type=float,
        default=DEFAULT_PROCESS_TIMEOUT,
        help="maximum seconds for each runner and validation process",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--no-build", action="store_true", help="reuse existing release binaries")
    return parser.parse_args()


def _family_for_item(item: dict[str, Any]) -> str:
    runner = item.get("runner", {})
    family = runner.get("family")
    if family:
        return str(family)
    # Legacy inventories predate the explicit family field.
    if item.get("workload", {}).get("op") == "Conv":
        return "conv-f32"
    if item.get("workload", {}).get("op") in {"MatMul", "Gemm"}:
        return "matrix-f32"
    raise RuntimeError(f"inventory workload has no supported runner family: {item!r}")


def _runner_for_family(family: str) -> Path:
    if family == "conv-f32":
        return CONV_BIN
    if family == "matrix-f32":
        return MATRIX_BIN
    raise RuntimeError(f"unsupported inventory runner family {family!r}")


def _ordered_families(
    grouped: dict[str, list[dict[str, Any]]], priority: dict[str, float] | None
) -> list[str]:
    names = list(grouped)
    if not priority:
        return names

    def score(family: str) -> tuple[int, float]:
        # Priorities are family-level GPU Pareto weights.  Count each operation
        # once: a model with many duplicate nodes should not reorder a family
        # merely because its graph repeats the same operator.  Families with
        # no measured operation are kept after measured families.
        operations = {
            item.get("workload", {}).get("op")
            for item in grouped[family]
            if item.get("workload", {}).get("op") in priority
        }
        return (bool(operations), sum(priority[operation] for operation in operations))

    # Python's sort is stable, so families with equal or absent priority retain
    # the inventory order that makes default runs reproducible.
    return sorted(names, key=score, reverse=True)


def _read_priority(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid priority JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("priority JSON must be an object mapping operation to time")
    result: dict[str, float] = {}
    for operation, duration in value.items():
        if not isinstance(operation, str) or not operation:
            raise RuntimeError("priority JSON operation names must be non-empty strings")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)):
            raise RuntimeError(f"priority for {operation!r} must be a finite number")
        result[operation] = float(duration)
    return result


def main() -> int:
    args = parse_args()
    model = args.model.resolve()
    output = args.output.resolve()
    state = (args.state or output.with_suffix(".jsonl")).resolve()
    try:
        if not args.no_build:
            build()
        if (
            not math.isfinite(args.atol)
            or not math.isfinite(args.rtol)
            or args.atol < 0
            or args.rtol < 0
        ):
            raise ValueError("--atol and --rtol must be finite and nonnegative")
        priority = _read_priority(args.priority_json)
        inventory_args = [str(TUNE_BIN), "model-inventory", "--model", str(model)]
        for dimension in args.dim:
            inventory_args.extend(["--dim", dimension])
        inventory = read_json_output(inventory_args)
        relevant_ops = {"Conv", "MatMul", "Gemm"}
        skipped_relevant = [item for item in inventory["skipped"] if item.get("op") in relevant_ops]
        skipped_conv = [item for item in skipped_relevant if item.get("op") == "Conv"]
        skipped_matrix = [item for item in skipped_relevant if item.get("op") in {"MatMul", "Gemm"}]
        if args.strict and skipped_relevant:
            reasons = "; ".join(f"{item['node']}: {item['reason']}" for item in skipped_relevant)
            raise RuntimeError(f"strict profile has untunable relevant nodes: {reasons}")
        if not inventory["workloads"]:
            raise RuntimeError("model contains no supported Conv, MatMul, or Gemm workload")

        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in inventory["workloads"]:
            grouped.setdefault(_family_for_item(item), []).append(item)
        family_order = _ordered_families(grouped, priority)
        results: dict[str, dict[str, Any]] = {}
        family_devices: dict[str, dict[str, Any]] = {}
        device: dict[str, Any] | None = None
        unique_total: dict[str, dict[str, Any]] = {}
        for family in family_order:
            runner = _runner_for_family(family)
            identity = read_json_output([str(runner), "--device-json"])
            family_device = identity["device"]
            family_implementation = identity["implementation_digest"]
            family_devices[family] = family_device
            if device is None:
                device = family_device
            elif device != family_device:
                raise RuntimeError("tuning runners reported different devices")
            family_items = grouped[family]
            if any(item["implementation_digest"] != family_implementation for item in family_items):
                raise RuntimeError(f"inventory and {family} runner implementation fingerprints differ")
            unique = {
                geometry_key(item["runner"]): item["runner"] for item in family_items
            }
            unique_total.update(unique)
            family_results = tune_missing(
                list(unique.values()),
                state,
                inventory["model_digest"],
                family_device,
                family_implementation,
                family=family,
                runner=runner,
                response_timeout=args.response_timeout,
                process_timeout=args.process_timeout,
            )
            results.update(family_results)
        if device is None:
            raise RuntimeError("model contains no supported workload family")
        draft = artifact_from(
            inventory,
            device,
            results,
            len(skipped_conv),
            untunable_nodes=len(skipped_relevant),
            family_devices=family_devices,
        )
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

        # Schema validation proves that the artifact is internally coherent;
        # this separate run proves that the finished model still executes and
        # that the standalone facade agrees with ORT. Keep its report beside
        # the artifact for diagnosis, but do not turn it into tuning evidence.
        validation_report = output.with_suffix(".validation.json")
        validation_arguments = [
            str(MODEL_RUNNER),
            str(model),
            "--standalone",
            "--require-vulkan",
            "--no-opt",
            "--iters",
            "20",
            # The CPU EP is the reference here; its timing is not evidence.
            "--cpu-budget",
            "30",
            "--tol",
            str(args.atol),
            "--rtol",
            str(args.rtol),
            "--report-json",
            str(validation_report),
        ]
        for dimension in args.dim:
            validation_arguments.extend(["--dim", dimension])
        if args.reference:
            validation_arguments.extend(["--reference", str(args.reference)])
        validation_env = os.environ.copy()
        validation_mode = "require-cache" if not skipped_relevant else "cache-only"
        validation_env["ONNX_VULKAN_TUNING_MODE"] = validation_mode
        validation_env["ONNX_VULKAN_TUNING_ARTIFACT"] = str(output)
        try:
            command(
                validation_arguments,
                timeout=args.process_timeout,
                env=validation_env,
            )
        except subprocess.CalledProcessError as error:
            # A report that says `passed: false` is a numerical verdict on the
            # tuned kernels, not a failure of the tuner.
            try:
                verdict = json.loads(validation_report.read_text(encoding="utf-8")).get("passed")
            except (OSError, json.JSONDecodeError):
                verdict = None
            if verdict is not False:
                raise
            print(
                "model autotune rejected: the tuned model misses the tolerance on the validation inputs",
                file=sys.stderr,
            )
            # model-runner's comparison lines name the outputs beyond tolerance.
            print(
                "\n".join(line for line in (error.stdout or "").splitlines() if "beyond tolerance" in line),
                file=sys.stderr,
            )
            return VALIDATION_REJECTED
        if not validation_report.exists():
            raise RuntimeError("model-runner did not produce its validation report")
        try:
            report = json.loads(validation_report.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"model-runner wrote invalid validation report: {error}") from error
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            raise RuntimeError("model-runner wrote an unsupported validation report")
        if report.get("passed") is not True:
            raise RuntimeError("model-runner did not confirm full-model correctness")
        summary = {
            "schema_version": 1,
            "status": "built",
            "artifact": str(output),
            "state": str(state),
            "device": device["name"],
            "records": len(draft["records"]),
            "workload_occurrences": len(inventory["workloads"]),
            "distinct_geometries": len(unique_total),
            "untunable_conv_nodes": len(skipped_conv),
            "untunable_matrix_nodes": len(skipped_matrix),
            "other_fallback_nodes": len(inventory["skipped"]) - len(skipped_relevant),
            "runtime_mode": validation_mode,
            "validation": str(validation_report),
        }
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"model autotune failed: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
