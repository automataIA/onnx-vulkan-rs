#!/usr/bin/env -S uv run --quiet --with optuna --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["optuna"]
# ///
"""Persistent-process tuner for Rust-owned Vulkan candidate spaces."""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN = ROOT / "target" / "release" / "examples" / "matmulnbits"


def build() -> None:
    subprocess.run(
        ["cargo", "build", "--release", "-p", "onnx-vulkan-core", "--example", "matmulnbits"],
        cwd=ROOT,
        check=True,
    )


@dataclass(frozen=True, order=True)
class Candidate:
    lanes: int
    vec: int
    wg: int
    sub: int

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> Candidate:
        return cls(*(require_int(value, name) for name in ("lanes", "vec", "wg", "sub")))

    def request(self) -> dict[str, Any]:
        return {"op": "compare", **self.parameters()}

    def parameters(self) -> dict[str, int]:
        return {"lanes": self.lanes, "vec": self.vec, "wg": self.wg, "sub": self.sub}


def require_int(value: dict[str, Any], name: str) -> int:
    field = value.get(name)
    if not isinstance(field, int) or isinstance(field, bool) or field < 0:
        raise RuntimeError(f"Rust response field {name!r} is not an unsigned integer")
    return field


class Runner:
    def __init__(self, binary: Path, geometry: str) -> None:
        self.process = subprocess.Popen(
            [str(binary), "--serve", "--geom", geometry],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.closed = False

    def request(self, value: dict[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("runner pipes are unavailable")
        self.process.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise RuntimeError(f"Rust runner exited without a response (exit {code})")
        response = json.loads(line)
        if response.get("schema_version") != 1:
            raise RuntimeError("unsupported Rust runner response schema")
        if response.get("ok") is False:
            raise RuntimeError(str(response.get("error", "runner request failed")))
        return response

    def shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.poll() is None:
                self.request({"op": "shutdown"})
        except (BrokenPipeError, json.JSONDecodeError, RuntimeError):
            pass
        finally:
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)

    def __enter__(self) -> Runner:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()


def load_state(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{path}:{number}: invalid JSONL state: {error}") from error
        if row.get("schema_version") != 1:
            raise RuntimeError(f"{path}:{number}: unsupported state schema")
        rows.append(row)
    return rows


def append_state(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())


def ordered_candidates(mode: str, candidates: list[Candidate], seed: int) -> list[Candidate]:
    result = list(candidates)
    if mode == "random":
        random.Random(seed).shuffle(result)
    return result


def create_study(path: Path, geometry: str, seed: int) -> optuna.Study:
    path.parent.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        study_name=f"matmul_nbits-{geometry.replace(',', 'x')}",
        storage=f"sqlite:///{path.resolve()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        load_if_exists=True,
    )


def ask_candidate(study: optuna.Study, axes: dict[str, list[int]]) -> tuple[optuna.Trial, Candidate]:
    trial = study.ask()
    return trial, Candidate(
        trial.suggest_categorical("lanes", axes["lanes"]),
        trial.suggest_categorical("vec", axes["vec"]),
        trial.suggest_categorical("wg", axes["wg"]),
        trial.suggest_categorical("sub", axes["sub"]),
    )


def tune_geometry(
    args: argparse.Namespace, geometry: str, state: Path
) -> tuple[list[dict[str, Any]], int]:
    previous = [
        row
        for row in load_state(state)
        if row.get("geometry") == geometry and row.get("mode") == args.mode
    ]
    already = {
        Candidate.from_json(row["candidate"])
        for row in previous
        if isinstance(row.get("candidate"), dict) and args.mode != "tpe"
    }
    started = time.monotonic()
    new_rows: list[dict[str, Any]] = []
    with Runner(args.binary, geometry) as runner:
        metadata = runner.request({"op": "list"})
        if metadata.get("family") != "matmul_nbits":
            raise RuntimeError("Rust runner did not expose the MatMulNBits family")
        raw_candidates = metadata.get("candidates")
        axes = metadata.get("axes")
        if not isinstance(raw_candidates, list) or not isinstance(axes, dict):
            raise RuntimeError("Rust runner omitted authoritative candidates or axes")
        for name in ("lanes", "vec", "wg", "sub"):
            values = axes.get(name)
            if not isinstance(values, list) or not values or not all(
                isinstance(value, int) and not isinstance(value, bool) for value in values
            ):
                raise RuntimeError(f"Rust runner axis {name!r} is invalid")
        candidates = [Candidate.from_json(value) for value in raw_candidates]
        candidate_set = set(candidates)
        if len(candidate_set) != len(candidates):
            raise RuntimeError("Rust runner exposed duplicate candidates")
        resident_bytes = require_int(metadata, "resident_bytes")
        if args.vram_bytes and resident_bytes > args.vram_bytes:
            raise RuntimeError(
                f"resident workload needs {resident_bytes} bytes, VRAM budget is {args.vram_bytes}"
            )

        study = create_study(args.study, geometry, args.seed) if args.mode == "tpe" else None
        ordered = ordered_candidates(args.mode, candidates, args.seed)
        cursor = 0
        completed = 0
        remaining = max(0, args.trials - len(previous))
        while completed < remaining:
            elapsed = time.monotonic() - started
            if args.wall_seconds and elapsed >= args.wall_seconds:
                break
            trial = None
            if study is not None:
                trial, candidate = ask_candidate(study, axes)
                if candidate not in candidate_set:
                    study.tell(trial, state=TrialState.PRUNED)
                    row = result_row(
                        args.mode,
                        geometry,
                        candidate,
                        "pruned",
                        elapsed,
                        None,
                        "not_in_rust_space",
                    )
                    append_state(state, row)
                    new_rows.append(row)
                    completed += 1
                    continue
            else:
                while cursor < len(ordered) and ordered[cursor] in already:
                    cursor += 1
                if cursor >= len(ordered):
                    break
                candidate = ordered[cursor]
                cursor += 1

            response = runner.request(candidate.request())
            if response.get("eligible") is True:
                score = response.get("gb_s")
                if not isinstance(score, (int, float)) or isinstance(score, bool) or score <= 0:
                    raise RuntimeError("eligible Rust result has no positive gb_s")
                status, reason = "scored", None
                if study is not None and trial is not None:
                    study.tell(trial, float(score))
            else:
                score = None
                diagnostic = response.get("diagnostic")
                reason = (
                    str(diagnostic.get("code"))
                    if isinstance(diagnostic, dict) and diagnostic.get("code")
                    else "rejected"
                )
                status = "pruned"
                if study is not None and trial is not None:
                    study.tell(trial, state=TrialState.PRUNED)
            row = result_row(
                args.mode,
                geometry,
                candidate,
                status,
                time.monotonic() - started,
                float(score) if score is not None else None,
                reason,
            )
            row["runner"] = response
            append_state(state, row)
            new_rows.append(row)
            already.add(candidate)
            completed += 1
    return previous + new_rows, len(candidates)


def result_row(
    mode: str,
    geometry: str,
    candidate: Candidate,
    status: str,
    elapsed: float,
    score: float | None,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": mode,
        "geometry": geometry,
        "candidate": candidate.parameters(),
        "status": status,
        "elapsed_seconds": elapsed,
        "score_gb_s": score,
        "reason": reason,
    }


def points(rows: list[dict[str, Any]]) -> tuple[list[float], float]:
    best = 0.0
    curve = []
    for row in rows:
        score = row.get("score_gb_s")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            best = max(best, float(score))
        curve.append(best)
    return curve, best


def polyline(values: list[float], x: int, y: int, width: int, height: int, maximum: float) -> str:
    if not values or maximum <= 0:
        return ""
    denominator = max(1, len(values) - 1)
    coords = [
        f"{x + width * index / denominator:.1f},{y + height * (1 - value / maximum):.1f}"
        for index, value in enumerate(values)
    ]
    return f'<polyline fill="none" stroke="#68d391" stroke-width="2" points="{" ".join(coords)}"/>'


def write_report(path: Path, all_rows: dict[str, list[dict[str, Any]]], spaces: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"schema_version": 1, "geometries": {}}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="700" viewBox="0 0 1000 700">',
           '<rect width="100%" height="100%" fill="#111827"/>',
           '<style>text{fill:#e5e7eb;font:14px monospace}.muted{fill:#9ca3af}</style>',
           '<text x="40" y="35" font-size="20">onnx-vulkan autotune diagnostics</text>']
    colors = ["#68d391", "#63b3ed", "#f6ad55", "#fc8181"]
    for geometry_index, (geometry, rows) in enumerate(all_rows.items()):
        curve, best = points(rows)
        scored = sum(row.get("status") == "scored" for row in rows)
        causes = Counter(str(row.get("reason")) for row in rows if row.get("status") != "scored")
        seen: set[str] = set()
        coverage_curve = []
        for row in rows:
            seen.add(json.dumps(row.get("candidate"), sort_keys=True))
            coverage_curve.append(len(seen))
        coverage = len(seen)
        regret = [0.0 if best == 0 else 1 - value / best for value in curve]
        summary["geometries"][geometry] = {
            "trials": len(rows), "scored": scored, "coverage": coverage,
            "space": spaces.get(geometry, 0), "best_gb_s": best, "pruned_causes": dict(causes),
            "best_so_far_gb_s": curve,
            "coverage_curve": coverage_curve,
            "retrospective_regret": regret,
        }
        color = colors[geometry_index % len(colors)]
        y = 80 + geometry_index * 140
        safe_geometry = html.escape(geometry)
        svg.append(f'<text x="40" y="{y}">{safe_geometry}: best {best:.2f} GB/s · coverage {coverage}/{spaces.get(geometry, 0)} · scored {scored}/{len(rows)}</text>')
        line = polyline(curve, 40, y + 20, 300, 75, best)
        svg.append(line.replace("#68d391", color))
        svg.append('<text class="muted" x="40" y="{}">best-so-far</text>'.format(y + 115))
        svg.append(polyline([float(value) for value in coverage_curve], 365, y + 20, 170, 75, float(max(spaces.get(geometry, 1), 1))).replace("#68d391", "#63b3ed"))
        svg.append('<text class="muted" x="365" y="{}">coverage</text>'.format(y + 115))
        svg.append(polyline(regret, 560, y + 20, 170, 75, 1.0).replace("#68d391", "#f6ad55"))
        svg.append('<text class="muted" x="560" y="{}">retrospective regret</text>'.format(y + 115))
        cause_text = ", ".join(f"{key}:{value}" for key, value in sorted(causes.items())) or "none"
        cause_max = max(causes.values(), default=1)
        for cause_index, (cause, count) in enumerate(causes.most_common(3)):
            bar_y = y + 22 + cause_index * 24
            width = 160 * count / cause_max
            svg.append(f'<rect x="760" y="{bar_y}" width="{width:.1f}" height="13" fill="#fc8181"/>')
            svg.append(f'<text class="muted" x="925" y="{bar_y + 12}">{html.escape(cause)}:{count}</text>')
        if not causes:
            svg.append(f'<text class="muted" x="760" y="{y + 45}">pruned: none</text>')
        svg.append(f'<text class="muted" x="760" y="{y + 115}">causes · {html.escape(cause_text)}</text>')
    svg.append("</svg>\n")
    path.with_suffix(".svg").write_text("".join(svg), encoding="utf-8")
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geom", action="append", default=[], help="K,N; repeat per geometry")
    parser.add_argument("--mode", choices=("exhaustive", "random", "tpe"), default="exhaustive")
    parser.add_argument("--trials", type=int, default=60, help="maximum trials per geometry")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wall-seconds", type=float, default=0.0, help="budget per geometry; 0 disables")
    parser.add_argument("--vram-bytes", type=int, default=0, help="resident workload ceiling; 0 disables")
    parser.add_argument("--bin", type=Path)
    parser.add_argument("--state", type=Path, default=ROOT / "runs" / "autotune" / "study.jsonl")
    parser.add_argument("--study", type=Path, default=ROOT / "runs" / "autotune" / "optuna.sqlite3")
    parser.add_argument("--report", type=Path, default=ROOT / "runs" / "autotune" / "report.json")
    args = parser.parse_args()
    if args.trials <= 0 or args.wall_seconds < 0 or args.vram_bytes < 0:
        parser.error("budgets must be positive, or zero where documented")
    args.geom = args.geom or ["1152,6912"]
    return args


def main() -> int:
    args = parse_args()
    if args.bin is None:
        build()
        args.binary = DEFAULT_BIN
    else:
        args.binary = args.bin.resolve()
    if not args.binary.exists():
        print(f"missing runner {args.binary}", file=sys.stderr)
        return 2
    results: dict[str, list[dict[str, Any]]] = {}
    spaces: dict[str, int] = {}
    try:
        for geometry in args.geom:
            rows, candidate_count = tune_geometry(args, geometry, args.state)
            results[geometry] = rows
            spaces[geometry] = candidate_count
    except KeyboardInterrupt:
        print("autotune cancelled; runner shutdown requested and state flushed", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"autotune failed: {error}", file=sys.stderr)
        return 3
    write_report(args.report, results, spaces)
    for geometry, rows in results.items():
        _, best = points(rows)
        print(f"{geometry}: {len(rows)} recorded trials, best {best:.2f} GB/s")
    print(f"state: {args.state}\nreport: {args.report}\nplot: {args.report.with_suffix('.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
