#!/usr/bin/env -S uv run --quiet --script
"""Gemm16 research gates for the bounded, exact-shape experiment contract.

The research supervisor runs this file from a candidate worktree.  Every
subcommand gets its paths and experiment contract from the environment, uses
argument arrays for child processes, and leaves logs/results in the gate
output directory for later review.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
INTERP = Path("crates/onnx-vulkan-core/src/interp.rs")
DEFAULT_TIMEOUT = 900.0


def _path_from(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty path")
    path = Path(value)
    return (base / path if not path.is_absolute() else path).resolve()


def _context() -> tuple[dict[str, Any], Path, Path]:
    raw_problem = os.environ.get("VULKAN_RESEARCH_PROBLEM")
    raw_output = os.environ.get("VULKAN_RESEARCH_OUTPUT")
    if not raw_problem or not raw_output:
        raise ValueError("VULKAN_RESEARCH_PROBLEM and VULKAN_RESEARCH_OUTPUT are required")
    problem_path = Path(raw_problem).resolve()
    try:
        problem = json.loads(problem_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read research problem {problem_path}") from error
    if not isinstance(problem, dict):
        raise ValueError("research problem must be a JSON object")
    output = Path(raw_output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return problem, problem_path.parent, output


def _timeout(problem: dict[str, Any]) -> float:
    value = problem.get("timeout", os.environ.get("VULKAN_RESEARCH_TIMEOUT", DEFAULT_TIMEOUT))
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("research timeout must be a finite positive number") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError("research timeout must be a finite positive number")
    return result


def _geometry(problem: dict[str, Any]) -> dict[str, Any]:
    value = problem.get("geometry")
    if not isinstance(value, dict):
        raise ValueError("research problem geometry must be an object")
    result = dict(value)
    if result.get("op", "Gemm") != "Gemm":
        raise ValueError("Gemm16 experiment geometry must have op=Gemm")
    for name in ("m", "k", "n"):
        number = result.get(name)
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise ValueError(f"geometry.{name} must be a positive integer")
    result.setdefault("family", "matrix-f32")
    result["op"] = "Gemm"
    return result


def _fixture(problem: dict[str, Any], base: Path) -> Path:
    return _path_from(problem.get("fixture"), base=base, label="fixture")


def _write_log(output: Path, name: str, text: str) -> None:
    # The supervisor owns <gate>.log; keep detailed subprocess logs separate.
    path = output / f"driver-{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _command(argv: Sequence[Path | str], *, cwd: Path, output: Path, label: str,
             timeout: float, input_data: str | None = None):
    # Import lazily so proposing a source patch does not import the full Python
    # pipeline or require optional runtime packages.
    python_root = cwd / "python"
    if python_root.is_dir() and str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    from onnx_vulkan_pipeline.process import command

    arguments = [str(item) for item in argv]
    try:
        if input_data is None:
            result = command(arguments, cwd=cwd, timeout=timeout)
        else:
            # process.command intentionally has no stdin parameter.  The
            # matrix runner is the one bounded exception: it consumes one
            # JSON line and must be allowed to stream its response.
            import subprocess

            process = subprocess.Popen(
                arguments, cwd=cwd, env=os.environ.copy(), text=True,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=(os.name == "posix"),
            )
            try:
                stdout, stderr = process.communicate(input=input_data, timeout=timeout)
            except subprocess.TimeoutExpired as error:
                process.kill()
                process.communicate()
                raise TimeoutError(f"{arguments[0]} timed out after {timeout:g}s") from error
            if process.returncode:
                raise RuntimeError(f"{arguments[0]} exited {process.returncode}: {stderr[-8000:]}")
            result = subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)
    except BaseException as error:
        _write_log(output, label, f"{type(error).__name__}: {error}\n")
        raise
    _write_log(output, label, result.stdout + result.stderr)
    return result


def _link_dependencies(problem: dict[str, Any], problem_base: Path, candidate: Path) -> Path:
    root = _path_from(problem.get("dependencies_root"), base=problem_base,
                      label="dependencies_root")
    if not root.is_dir():
        raise ValueError(f"dependencies_root is not a directory: {root}")

    def ensure_link(target: Path, source: Path) -> None:
        if not source.is_dir():
            raise ValueError(f"dependency directory is missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() != source:
                raise ValueError(f"existing link points to {target.resolve()}, expected {source}")
            return
        if target.exists():
            if target.is_dir() and not any(target.iterdir()):
                target.rmdir()
            else:
                raise ValueError(f"refusing to replace existing dependency path: {target}")
        target.symlink_to(source, target_is_directory=True)

    ensure_link(candidate / "third_party", root / "third_party")
    ensure_link(candidate / "models", root / "models")
    return candidate / "models"


def _propose(problem: dict[str, Any], problem_base: Path, output: Path) -> None:
    candidate = Path.cwd().resolve()
    geometry = _geometry(problem)
    _link_dependencies(problem, problem_base, candidate)
    path = candidate / INTERP
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read candidate interpreter source: {path}") from error

    marker = "let use_gemm16 = selected.as_deref() == Some(\"gemm16\")"
    if marker in source:
        _write_log(output, "propose", "Gemm16 exact-shape routing is already present.\n")
        return
    old = '        let (label, source, tile) = if selected.as_deref() == Some("gemm16") {'
    if source.count(old) != 1:
        raise ValueError("candidate interpreter has no unique Gemm tactic selection site")
    m, k, n = (geometry[name] for name in ("m", "k", "n"))
    new = (
        f'        let use_gemm16 = selected.as_deref() == Some("gemm16")\n'
        f"            || (selected.is_none() && m == {m} && ka == {k} && n == {n});\n"
        '        let (label, source, tile) = if use_gemm16 {'
    )
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    _write_log(
        output,
        "propose",
        f"Patched {path} for Gemm16 only when selected.is_none() and shape is {m}x{k}x{n}.\n",
    )


def _local(problem: dict[str, Any], output: Path, timeout: float) -> None:
    candidate = Path.cwd().resolve()
    commands = [
        ("fmt", ["cargo", "fmt", "--all", "--", "--check"]),
        ("clippy", ["cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"]),
        ("matrix-tests", ["cargo", "test", "-p", "onnx-vulkan-core", "--test", "matrix_tuning", "--", "--test-threads=1"]),
        ("build", ["cargo", "build", "--release", "--workspace", "--lib", "--bins", "--examples"]),
    ]
    for label, argv in commands:
        _command(argv, cwd=candidate, output=output, label=label, timeout=timeout)


def _kernel(problem: dict[str, Any], problem_base: Path, output: Path, timeout: float) -> None:
    geometry = _geometry(problem)
    candidate = Path.cwd().resolve()
    runner = candidate / "target" / "release" / "examples" / "matrix_tune"
    request = json.dumps(geometry, sort_keys=True, separators=(",", ":")) + "\n"
    process = _command([runner], cwd=candidate, output=output, label="kernel",
                       timeout=timeout, input_data=request)
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("matrix_tune returned no result")
    response = json.loads(lines[-1])
    if response.get("family") != "matrix-f32" or response.get("geometry") != geometry:
        raise ValueError("matrix_tune returned a different matrix geometry")
    correctness = response.get("correctness")
    medians = (response.get("interleaved") or {}).get("median_gpu_ns")
    tactic = response.get("tactic")
    if (not isinstance(correctness, dict) or correctness.get("passed") is not True
            or not isinstance(tactic, dict) or tactic.get("id") != "gemm16"):
        raise ValueError("Gemm16 kernel correctness did not pass")
    if (not isinstance(medians, list) or len(medians) < 2
            or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in medians)
            or not medians[1] < medians[0]):
        raise ValueError("Gemm16 median must be lower than Gemm64 median")
    (output / "kernel.json").write_text(json.dumps(response, indent=2, sort_keys=True) + "\n")


def _pipeline_module(candidate: Path):
    python_root = candidate / "python"
    if python_root.is_dir() and str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    from onnx_vulkan_pipeline import pipeline

    return pipeline


def _metric_row(report: dict[str, Any]) -> dict[str, Any]:
    gpu = report.get("gpu") or {}
    samples = report.get("standalone_ms")
    wall = samples[-1] if isinstance(samples, list) and samples else None
    return {
        "wall_ms": wall,
        "gpu_ms": gpu.get("compute_ms"),
        "gpu": gpu,
        "storage_peak_bytes": report.get("standalone_storage_peak_bytes"),
    }


def _model(problem: dict[str, Any], problem_base: Path, output: Path, timeout: float) -> None:
    candidate = Path.cwd().resolve()
    baseline = _path_from(problem.get("baseline_root"), base=problem_base, label="baseline_root")
    if not baseline.is_dir():
        raise ValueError(f"baseline_root is not a directory: {baseline}")
    fixture = _fixture(problem, problem_base)
    profile = problem.get("profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("profile must be a nonempty string")
    pipeline = _pipeline_module(candidate)
    manifest = pipeline.verify_bundle(fixture)
    if manifest.get("status") != "verified" or profile not in manifest.get("profiles", {}):
        raise ValueError("model gate requires a verified fixture and selected profile")
    original_root = pipeline.ROOT
    original_env = {key: os.environ.get(key) for key in ("ORT_DYLIB_PATH", "VULKAN_EP_PATH")}

    def select(root):
        pipeline.ROOT = root
        os.environ["ORT_DYLIB_PATH"] = str(root / "third_party/onnxruntime/linux-x64/lib/libonnxruntime.so")
        os.environ["VULKAN_EP_PATH"] = str(root / "target/release/libonnxruntime_ep_vulkan.so")

    rows: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    try:
        for index in range(20):
            labels = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            for label in labels:
                select(baseline if label == "baseline" else candidate)
                report = pipeline._run(fixture, manifest, profile, "standalone", 1, timeout)
                (output / f"{label}-{index:02d}.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                rows[label].append(_metric_row(report))
        for label, root in (("baseline", baseline), ("candidate", candidate)):
            select(root)
            report = pipeline._run(fixture, manifest, profile, "standalone", 20, timeout)
            (output / f"final-{label}.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    finally:
        pipeline.ROOT = original_root
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    metrics = {"interleaved": True, "baseline": rows["baseline"], "candidate": rows["candidate"]}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _regression(problem: dict[str, Any], problem_base: Path, output: Path, timeout: float) -> None:
    candidate = Path.cwd().resolve()
    argv: list[Path | str] = [candidate / "scripts" / "testsuite.sh", "-M", "standalone", "-n",
                               "-i", "20", "--keep", "9999", "--tag", "research"]
    if problem.get("regression_baseline") is not None:
        argv.extend(["--baseline", _path_from(problem["regression_baseline"], base=problem_base,
                                                label="regression_baseline")])
    _command(argv, cwd=candidate, output=output, label="regression", timeout=timeout)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("propose", "local", "kernel", "model", "regression"))
    args = parser.parse_args(argv)
    try:
        problem, problem_base, output = _context()
        timeout = _timeout(problem)
        if args.command == "propose":
            _propose(problem, problem_base, output)
        elif args.command == "local":
            _local(problem, output, timeout)
        elif args.command == "kernel":
            _kernel(problem, problem_base, output, timeout)
        elif args.command == "model":
            _model(problem, problem_base, output, timeout)
        else:
            _regression(problem, problem_base, output, timeout)
    except BaseException as error:
        print(f"gemm16 {args.command} failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
