#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Turns a run's logs into `<model>/<mode>.json` (plan-test-suite.md §6).

    parse_run.py --model yolov8n --mode compile --runner model-runner --iters 50 \\
        --stdout run.log [--stats-log stats.log] [--metrics run.csv] \\
        --validate outputs --exit-code 0 --out compile.json

Three sources, kept separate because they have different reliability: the wall
time comes from the pass **without** the profiler (per-dispatch timestamps
falsify it), the GPU breakdown from the pass with `VULKAN_EP_STATS=1`, and the
system metrics from the sampler CSV.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import peaks as peaks_mod  # noqa: E402

# --- model-runner -----------------------------------------------------------
RE_CPU_EP = re.compile(r"^CPU EP:\s+([\d.]+) ms")
# min/max are missing on binaries predating their introduction: keep them optional
RE_SECOND = re.compile(
    r"^(?:Vulkan EP|CPU no-opt):\s+([\d.]+) ms(?:.*?\[min ([\d.]+) max ([\d.]+)\])?"
)
RE_OUTPUT = re.compile(
    r"output (\S+)\s+n=(\d+)\s+max\|Δ\|=([\d.e+-]+)\s+\|ref\|max=([\d.e+-]+)\s+"
    r"relative=([\d.e+-]+)\s+beyond tolerance: (\d+)"
)
RE_OK = re.compile(r"^OK: within atol=.*worst relative error ([\d.e+-]+)\)")
RE_DIVERGE = re.compile(r"divergent outputs")

# --- standalone parity (`--standalone`) -------------------------------------
# The two hosts run the same kernels on the same graph, so their outputs are
# expected **bit for bit** identical, not merely close: anything else is a bug
# in one of the two paths rather than model sensitivity.
RE_STANDALONE = re.compile(
    r"^standalone:\s+([\d.]+) ms(?:.*?\[min ([\d.]+) max ([\d.]+)\])?"
)
RE_PARITY = re.compile(
    r"parity: standalone vs EP relative=([\d.e+-]+), vs CPU EP relative=([\d.e+-]+)"
)
RE_PARITY_ROW = re.compile(
    r"standalone vs (EP|CPU)\s+(\S+)\s+n=(\d+)\s+max\|Δ\|=([\d.e+-]+)\s+"
    r"relative=([\d.e+-]+)\s+beyond tolerance: (\d+)"
)

# --- external reference backend (`--webgpu`) --------------------------------
# ORT's standalone WebGPU plugin EP, run in the same process on the same inputs.
# It is a reference, not a gated path: `report.py` exempts this mode from the
# wall gate, because its performance is not ours to regress.
RE_WEBGPU = re.compile(r"^webgpu:\s+([\d.]+) ms(?:.*?\[min ([\d.]+) max ([\d.]+)\])?")
# ORT's own partitioning verdict, from a separate `ORT_LOG=verbose` pass. It is
# the fairness caveat made into data: unlike our all-or-nothing contract, the
# WebGPU EP claims what it supports and leaves the rest on the CPU EP, so a wall
# compared without this is not necessarily a GPU wall.
RE_PLACEMENTS = re.compile(r"Node placements")
RE_PLACED_ALL = re.compile(r"All nodes placed on \[(\w+)\]\. Number of nodes: (\d+)")
RE_PLACED_ONE = re.compile(r"Node\(s\) placed on \[(\w+)\]\. Number of nodes: (\d+)")
WEBGPU_EP_NAME = "WebGpuExecutionProvider"

# --- stt-app ----------------------------------------------------------------
RE_ENC_ITER = re.compile(r"encoder iter (\d+)/(\d+): ([\d.]+) ms")

# --- EP log -----------------------------------------------------------------
RE_DEVICE = re.compile(r"Vulkan device: (.+?) \(vendor")
RE_CLAIMED = re.compile(r"claimed (\d+)/(\d+) nodes")
RE_BLOCKS = re.compile(r"compile: (\d+) nodes in (\d+) convex blocks")

# --- profiler ---------------------------------------------------------------
RE_PARETO_HEAD = re.compile(r"VulkanEP profile")
# `gflop`/`gb` are the analytic work of the nodes charged to the kernel
# (`onnx-vulkan-core::work`). Optional: absent from logs predating them, and
# from any op the work model does not cover.
RE_PARETO_ROW = re.compile(
    r"^\s*(\S+)\s+([\d.]+) ms\s+([\d.]+)%\s+\((\d+) dispatch\)"
    r"(?:\s+gflop=([\d.]+)\s+gb=([\d.]+))?"
)
RE_GPU_TOTAL = re.compile(r"TOTAL GPU compute\s+([\d.]+) ms")
RE_SYNC = re.compile(
    r"sync/overhead ~([\d.]+) ms across (\d+) flushes; transfer up ([\d.]+) MB / down ([\d.]+) MB"
)
# Counted, not timed: these do not move between two identical runs, which is
# what makes them usable as the keep/discard signal of an experiment loop
# (`plan.md` §9.2). Absent from logs produced before the counters existed —
# the fields are then simply missing, and the gate does not read them.
RE_COUNTERS = re.compile(
    r"counters: submit=(\d+) upload=(\d+) download=(\d+) alloc=(\d+) pool-hit=(\d+)"
)

# --- official reference data (`test_data_set_*` of the model zoo) -----------
RE_REFERENCE = re.compile(
    r"reference (\S+)\s+(\S+)\s+max\|Δ\|=(\S+)\s+\|ref\|max=\S+\s+"
    r"beyond tolerance: (\d+)\s+argmax Some\((\d+)\)→Some\((\d+)\)"
)

SOFTWARE_DEVICES = ("llvmpipe", "lavapipe", "swiftshader", "software")


def parse_reference(lines: list[str]) -> dict | None:
    """Comparison against the output expected by the model's authors.

    Kept separate from CPU-vs-Vulkan accuracy because it answers a different
    question: not "the two backends agree" but "the answer is the right one".
    The `cpu` backend serves to validate the reference itself — if it diverges,
    the comparison is inconclusive and it is not our fault.
    """
    entries: dict[str, list[dict]] = {}
    for line in lines:
        if m := RE_REFERENCE.search(line):
            entries.setdefault(m.group(1), []).append(
                {
                    "output": m.group(2),
                    "worst_abs": float(m.group(3)),
                    "mismatches": int(m.group(4)),
                    "argmax_expected": int(m.group(5)),
                    "argmax_got": int(m.group(6)),
                }
            )
    if not entries:
        return None
    def ok(rows):
        return all(
            r["mismatches"] == 0 and r["argmax_expected"] == r["argmax_got"] for r in rows
        )
    backends = {name: {"outputs": rows, "ok": ok(rows)} for name, rows in entries.items()}
    tested = [name for name in backends if name != "cpu"]
    return {
        "backends": backends,
        # the verdict is the one of the backend under test, not the CPU EP
        "ok": all(backends[name]["ok"] for name in tested) if tested else None,
        "cpu_agrees": backends.get("cpu", {}).get("ok"),
    }


def read(path: Path | None) -> list[str]:
    if not path or not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def steady(times: list[float]) -> dict | None:
    """Median/min/max discarding the first iteration (pipeline + first upload)."""
    if not times:
        return None
    tail = times[1:] if len(times) > 1 else times
    return {
        "median": round(statistics.median(tail), 3),
        "min": round(min(tail), 3),
        "max": round(max(tail), 3),
        "iters_used": len(tail),
    }


def parse_parity(lines: list[str]) -> dict | None:
    """standalone-vs-EP agreement, the verdict of the `standalone` mode.

    `max_abs_delta` and not just the relative figure: parity is expected to be
    exactly zero, and a relative error rounds to `0.00e0` well before an
    absolute one does.
    """
    outputs = []
    for line in lines:
        if m := RE_PARITY_ROW.search(line):
            outputs.append(
                {
                    "against": m.group(1).lower(),
                    "name": m.group(2),
                    "n": int(m.group(3)),
                    "max_abs_delta": float(m.group(4)),
                    "relative": float(m.group(5)),
                    "mismatches": int(m.group(6)),
                }
            )
    totals = next((m for m in map(RE_PARITY.search, lines) if m), None)
    if not totals and not outputs:
        return None
    vs_ep = [o for o in outputs if o["against"] == "ep"]
    return {
        "vs_ep": float(totals.group(1)) if totals else None,
        "vs_cpu": float(totals.group(2)) if totals else None,
        "outputs": vs_ep,
        # the property the gate is on: same kernels, same graph, same bytes out
        "bit_exact": bool(vs_ep) and all(o["max_abs_delta"] == 0.0 for o in vs_ep),
        "mismatches": sum(o["mismatches"] for o in vs_ep),
    }


def parse_placement(lines: list[str], ep: str) -> dict | None:
    """How ORT partitioned the *last* session in the log, which is `ep`'s.

    Three sessions are created in a `webgpu` run — CPU EP, our EP, then the
    WebGPU EP — and each logs its own placement, so only the last block answers
    the question being asked.
    """
    starts = [i for i, line in enumerate(lines) if RE_PLACEMENTS.search(line)]
    if not starts:
        return None
    tail = lines[starts[-1] :]
    # ORT logs one line per EP when the graph is split, and a single
    # "All nodes placed on" line when it is not
    for line in tail:
        if m := RE_PLACED_ALL.search(line):
            return {
                "by_provider": {m.group(1): int(m.group(2))},
                "nodes": int(m.group(2)),
                "fully_on": m.group(1) == ep,
            }
    by_provider: dict[str, int] = {}
    for line in tail:
        if m := RE_PLACED_ONE.search(line):
            by_provider[m.group(1)] = by_provider.get(m.group(1), 0) + int(m.group(2))
    if not by_provider:
        return None
    return {
        "by_provider": by_provider,
        "nodes": sum(by_provider.values()),
        "fully_on": list(by_provider) == [ep],
    }


def parse_wall(
    lines: list[str], runner: str, mode: str = ""
) -> tuple[dict | None, float | None, float | None]:
    """(steady-state wall of the path under test, CPU EP time, our EP's own time).

    The third figure is only distinct from the first in the modes that put a
    second path under test in the same process (`standalone`, `webgpu`): there
    the log carries our EP's wall as well, and it is what the reference has to
    be read against.
    """
    if runner == "stt-app":
        times = [float(m.group(3)) for m in map(RE_ENC_ITER.search, lines) if m]
        return steady(times), None, None
    cpu_ms = None
    wall = None
    ep_ms = None
    # in `standalone` and `webgpu` the path under test is not the EP: the same
    # log carries both timings and picking the wrong one would benchmark the
    # thing the mode exists to avoid
    second = {"standalone": RE_STANDALONE, "webgpu": RE_WEBGPU}.get(mode, RE_SECOND)
    for line in lines:
        if m := RE_CPU_EP.match(line):
            cpu_ms = float(m.group(1))
            continue
        if m := RE_SECOND.match(line):
            ep_ms = float(m.group(1))
        if m := second.match(line):
            wall = {
                "median": float(m.group(1)),
                "min": float(m.group(2)) if m.group(2) else None,
                "max": float(m.group(3)) if m.group(3) else None,
                "iters_used": None,
            }
    return wall, cpu_ms, ep_ms


def parse_accuracy(lines: list[str], kind: str, expect: str, exposed: list[str]) -> dict:
    if kind == "none":
        return {"kind": "none", "tolerated": None}

    if kind == "argmax":
        # the criterion is the classifier's answer, not the logits: in QDQ
        # graphs the golden is produced with integer-arithmetic fusion, which
        # the literal graph does not reproduce element-by-element (see
        # docs/testsuite.md)
        reference = parse_reference(lines) or {}
        rows = [
            row
            for name, backend in reference.get("backends", {}).items()
            if name != "cpu"
            for row in backend["outputs"]
        ]
        return {
            "kind": "argmax",
            "outputs": rows,
            "tolerated": bool(rows)
            and all(r["argmax_expected"] == r["argmax_got"] for r in rows),
            "criterion": "argmax equal to the official reference",
        }

    if kind == "transcript":
        # the transcript is the only line printed to stdout without a log prefix
        text = next(
            (
                ln.strip()
                for ln in reversed(lines)
                if ln.strip() and not ln.startswith("[") and "ms" not in ln
            ),
            "",
        )
        return {
            "kind": "transcript",
            "text": text,
            "expect": expect,
            "tolerated": bool(expect) and expect.lower() in text.lower(),
        }

    outputs = [
        {
            "name": m.group(1),
            "n": int(m.group(2)),
            "max_abs_delta": float(m.group(3)),
            "ref_scale": float(m.group(4)),
            "relative": float(m.group(5)),
            "mismatches": int(m.group(6)),
        }
        for m in map(RE_OUTPUT.search, lines)
        if m
    ]
    if kind == "per-node":
        # ±1 LSB criterion on the **first exposed intermediate**: downstream, the
        # scales of DynamicQuantizeLinear amplify any rounding tie.
        # It is not `outputs[0]`: in the instrumented graph outputs the original
        # ones come first, and they are exactly the ones that can't be compared.
        by_name = {o["name"]: o for o in outputs}
        # the file carries `name<TAB>dtype`; quantized tensors are the (U)INT8 ones
        pairs = [(ln.split("\t")[0], ln.split("\t")[-1]) for ln in exposed if ln.strip()]
        quantized = [n for n, dtype in pairs if dtype in ("UINT8", "INT8")]
        order = quantized or [n for n, _ in pairs]
        first = next((by_name[n] for n in order if n in by_name), None)
        return {
            "kind": "per-node",
            "first_output": first,
            "outputs": outputs,
            "tolerated": bool(first) and first["max_abs_delta"] <= 1.0,
            "criterion": (
                "max|Δ| ≤ 1 LSB on the first quantized intermediate"
                if quantized
                else "max|Δ| ≤ 1 LSB on the first exposed intermediate "
                "(no quantized intermediate among the exposed ones)"
            ),
        }

    worst = max((o["relative"] for o in outputs), default=None)
    ok = any(RE_OK.match(ln) for ln in lines)
    diverged = any(RE_DIVERGE.search(ln) for ln in lines)
    return {
        "kind": "outputs",
        "worst_relative": worst,
        "outputs": outputs,
        "tolerated": ok and not diverged,
    }


# Below this fraction of the attainable roof a kernel is not limited by the
# machine but by its own launch shape: not enough workgroups, not enough work
# per thread. `CLAUDE.md` calls it "latency-bound" and the lever there is grid
# shape (split-K, occupancy), never the tile. The threshold is a reading aid,
# not a measurement: `pct_peak_*` and `efficiency` are the numbers.
LATENCY_EFFICIENCY = 0.10


def roofline(entry: dict, peaks: dict) -> None:
    """Adds the Roofline fields to a Pareto entry, in place.

    Achieved throughput is the analytic work over the measured GPU time; the
    class is decided by *arithmetic intensity against the ridge point*, not by
    which percentage is larger — a kernel can sit at 19% of the compute roof and
    still be compute-bound, with the missing 81% being its own inefficiency.
    That distinction is what `efficiency` carries.
    """
    ms, gflop, gb = entry.get("ms"), entry.get("gflop"), entry.get("gb")
    if ms is None or gflop is None or gb is None or ms <= 0:
        return
    seconds = ms / 1e3
    entry["gflops"] = round(gflop / seconds, 2)
    entry["gb_s"] = round(gb / seconds, 2)
    if gb <= 0:
        return
    intensity = gflop / gb  # FLOP/B
    entry["intensity"] = round(intensity, 4)
    ridge = peaks_mod.ridge(peaks)
    if ridge is None:
        return
    peak_gflops = peaks["peak_tflops_fp32"] * 1e3
    entry["pct_peak_compute"] = round(entry["gflops"] / peak_gflops * 100, 2)
    entry["pct_peak_bw"] = round(entry["gb_s"] / peaks["peak_gb_s"] * 100, 2)
    attainable = min(peak_gflops, peaks["peak_gb_s"] * intensity)
    efficiency = entry["gflops"] / attainable if attainable > 0 else 0.0
    entry["efficiency"] = round(efficiency, 4)
    entry["class"] = (
        "latency"
        if efficiency < LATENCY_EFFICIENCY
        else ("memory" if intensity < ridge else "compute")
    )


def parse_gpu(lines: list[str], peaks: dict | None = None) -> dict | None:
    """Last Pareto block in the log: that is the steady-state one."""
    starts = [i for i, ln in enumerate(lines) if RE_PARETO_HEAD.search(ln)]
    if not starts:
        return None
    block = lines[starts[-1] :]
    gpu: dict = {"pareto": []}
    for line in block:
        if m := RE_GPU_TOTAL.search(line):
            gpu["compute_ms"] = float(m.group(1))
        elif m := RE_SYNC.search(line):
            gpu["sync_ms"] = float(m.group(1))
            gpu["flushes"] = int(m.group(2))
            gpu["upload_mb"] = float(m.group(3))
            gpu["download_mb"] = float(m.group(4))
        elif m := RE_COUNTERS.search(line):
            gpu["submits"] = int(m.group(1))
            gpu["uploads"] = int(m.group(2))
            gpu["downloads"] = int(m.group(3))
            gpu["allocs"] = int(m.group(4))
            gpu["pool_hits"] = int(m.group(5))
        elif m := RE_PARETO_ROW.search(line.split("] ", 1)[-1]):
            entry = {
                "op": m.group(1),
                "ms": float(m.group(2)),
                "pct": float(m.group(3)),
                "dispatches": int(m.group(4)),
            }
            if m.group(5):
                entry["gflop"] = float(m.group(5))
                entry["gb"] = float(m.group(6))
                roofline(entry, peaks or {})
            gpu["pareto"].append(entry)
    return gpu


def parse_blocks(lines: list[str]) -> dict:
    blocks: dict = {}
    for line in lines:
        if m := RE_CLAIMED.search(line):
            blocks["nodes_claimed"] = int(m.group(1))
            blocks["nodes_total"] = int(m.group(2))
        elif m := RE_BLOCKS.search(line):
            blocks["nodes_claimed"] = int(m.group(1))
            blocks["convex_blocks"] = int(m.group(2))
    return blocks


def parse_metrics(path: Path | None) -> dict | None:
    lines = read(path)
    idle_fb = None
    rows = []
    for line in lines:
        if line.startswith("#"):
            if m := re.search(r"idle_fb_mb=(-?\d+)", line):
                idle_fb = int(m.group(1))
            continue
        if line.startswith("t_ms"):
            continue
        parts = line.split(",")
        if len(parts) != 6:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    if not rows:
        return None

    def col(i: int) -> list[float]:
        return [r[i] for r in rows if r[i] >= 0]

    sm, fb, rss, cpu = col(1), col(3), col(4), col(5)
    system: dict = {"samples": len(rows)}
    if sm:
        ordered = sorted(sm)
        system["sm_pct"] = {
            "mean": round(statistics.mean(sm), 1),
            "max": max(sm),
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        }
    if fb:
        system["fb_peak_mb"] = max(fb)
        if idle_fb is not None and idle_fb >= 0:
            system["vram_delta_mb"] = max(fb) - idle_fb
            system["vram_idle_mb"] = idle_fb
            # nvidia-smi memory is global: under WDDM a Vulkan process may not
            # appear in --query-compute-apps (plan-test-suite.md §5.2)
            system["vram_source"] = "global-delta"
    if rss:
        system["rss_mb"] = max(rss)
    if cpu:
        system["cpu_pct"] = {"mean": round(statistics.mean(cpu), 1), "max": max(cpu)}
    return system


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--stdout", type=Path, required=True)
    ap.add_argument("--stats-log", type=Path)
    ap.add_argument(
        "--placement-log",
        type=Path,
        help="an ORT_LOG=verbose log; the last 'Node placements' block is read "
        "to record whether the external EP took the whole graph",
    )
    ap.add_argument("--metrics", type=Path)
    ap.add_argument("--validate", default="outputs")
    ap.add_argument("--expect", default="")
    ap.add_argument(
        "--exposed",
        type=Path,
        help="file with the names of the exposed intermediates (expose-intermediates.py), in order",
    )
    ap.add_argument("--size-mb", type=float, default=0)
    ap.add_argument("--perf-tol", type=float, default=10.0)
    # golden models validate correctness, not product performance
    ap.add_argument("--golden", action="store_true")
    ap.add_argument("--exit-code", type=int, default=0)
    ap.add_argument(
        "--env",
        type=Path,
        help="run's env.json: source of the device peaks for the Roofline "
        "(falls back to the table in peaks.py keyed by the device name)",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    clean = read(args.stdout)
    stats = read(args.stats_log)
    wall, cpu_ms, ep_ms = parse_wall(clean, args.runner, args.mode)
    device = next((m.group(1) for m in map(RE_DEVICE.search, clean + stats) if m), None)

    # env.json first: it names the GPU the run was launched on, which on the
    # standalone path is not the one the log reports
    env = {}
    if args.env and args.env.exists():
        env = json.loads(args.env.read_text(encoding="utf-8"))
    device_peaks = {k: env[k] for k in ("peak_tflops_fp32", "peak_gb_s") if k in env}
    if not device_peaks:
        device_peaks = peaks_mod.resolve(device or env.get("gpu"))

    parity = parse_parity(clean)
    if args.mode == "standalone":
        # The verdict of this mode is parity, not agreement with ORT. A
        # dynamically-quantized graph diverges from the CPU EP by construction
        # (`plan.md` Phase 1.5) and the runner exits ≠ 0 for it; that question
        # belongs to the `compile` mode, which asks it on the real GPU. Here the
        # question is whether our two hosts compute the same thing, and its
        # answer must not be shadowed by the other one.
        ok = wall is not None and bool(parity and parity.get("bit_exact"))
    else:
        ok = (args.exit_code == 0 or args.validate in ("per-node", "argmax")) and wall is not None

    result = {
        "model": args.model,
        "mode": args.mode,
        "runner": args.runner,
        "iters": args.iters,
        # on `per-node` the runner exits ≠ 0 by construction: the final outputs of
        # a dynamic-quantization graph always diverge, and that's why we look at
        # the first quantized tensor instead of them
        # `argmax` like `per-node`: the runner exits ≠ 0 by construction,
        # because the element-by-element comparison is not the right criterion
        "ok": ok,
        "exit_code": args.exit_code,
        "model_size_mb": args.size_mb,
        "perf_tol": args.perf_tol,
        "golden": args.golden,
        "device": device,
        # on a software rasterizer timings say nothing: the data is still
        # emitted, but flagged (plan-test-suite.md §9)
        "perf_valid": args.mode == "cpu"
        or bool(device and not any(s in device.lower() for s in SOFTWARE_DEVICES)),
        "wall_ms": wall,
        "cpu_ep_ms": cpu_ms,
        # only set where the mode measures something other than our EP in the
        # same process: it is the number the reference has to be read against
        "ep_ms": ep_ms if args.mode in ("standalone", "webgpu") else None,
        "placement": parse_placement(read(args.placement_log), WEBGPU_EP_NAME)
        if args.mode == "webgpu"
        else None,
        "accuracy": parse_accuracy(clean, args.validate, args.expect, read(args.exposed)),
        "parity": parity,
        "reference": parse_reference(clean),
        "gpu": parse_gpu(stats or clean, device_peaks),
        # recorded next to the numbers derived from them: a Roofline read
        # against the wrong peaks is worse than no Roofline
        "peaks": device_peaks or None,
        # in `webgpu` mode our EP runs too and logs its own partitioning, but
        # attributing it to this row would duplicate the `compile` row's number
        # under a label that means something else
        "blocks": None if args.mode == "webgpu" else parse_blocks(stats or clean),
        "system": parse_metrics(args.metrics),
        "raw": {
            "stdout": args.stdout.name,
            "stats": args.stats_log.name if args.stats_log else None,
            "metrics": args.metrics.name if args.metrics and args.metrics.exists() else None,
        },
    }
    if result["accuracy"].get("tolerated") is False:
        result["ok"] = False
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
