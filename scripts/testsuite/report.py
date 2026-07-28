#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Aggregates a run's results into `summary.json` + `summary.md`, and applies the gates.

    report.py runs/2026-07-25T18-04-11 [--baseline runs/baseline.json]

`summary.md` is the table you paste into `cronologia.md` without retouching. With
`--baseline` the exit code is ≠ 0 if a gate trips (plan-test-suite.md §7): it
serves as a regression gate, so a failure must name the model and the quantity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GATE_PERF_DEFAULT = 10.0  # tolerated worsening on the median wall time, %


def collect(run_dir: Path) -> dict:
    results = []
    for path in sorted(run_dir.glob("*/*.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    env_path = run_dir / "env.json"
    return {
        "tag": run_dir.name,
        "env": json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {},
        "results": results,
    }


def fmt(value, spec: str = ".1f", dash: str = "—") -> str:
    return dash if value is None else format(value, spec)


def markdown(summary: dict) -> str:
    rows = []
    for r in summary["results"]:
        if r.get("status") == "skip":
            continue
        wall = r.get("wall_ms") or {}
        gpu = r.get("gpu") or {}
        system = r.get("system") or {}
        blocks = r.get("blocks") or {}
        acc = r.get("accuracy") or {}
        cpu_ms = r.get("cpu_ep_ms")
        median = wall.get("median")
        ratio = f"{cpu_ms / median:.2f}×" if cpu_ms and median else "—"
        if acc.get("kind") == "outputs":
            err = fmt(acc.get("worst_relative"), ".2e")
        elif acc.get("kind") == "per-node":
            first = acc.get("first_output") or {}
            err = f"±{fmt(first.get('max_abs_delta'), '.0f')} LSB ({first.get('mismatches', '?')})"
        elif acc.get("kind") == "argmax":
            err = "argmax" if acc.get("tolerated") else "ARGMAX≠"
        elif acc.get("kind") == "transcript":
            err = "ok" if acc.get("tolerated") else "DIFFERENT"
        else:
            err = "—"
        # official reference: the class expected by the model authors,
        # or "—" for models that do not have one
        ref = r.get("reference") or {}
        if ref.get("ok") is None:
            rif = "—"
        else:
            tested = [
                o
                for name, b in ref.get("backends", {}).items()
                if name != "cpu"
                for o in b["outputs"]
            ]
            top = tested[0] if tested else {}
            same_class = bool(tested) and all(
                o["argmax_expected"] == o["argmax_got"] for o in tested
            )
            # `~` = the answer is the right one but the values are out of tolerance:
            # that's the QDQ case, where the golden comes from integer fusion
            mark = "✓" if ref["ok"] else ("~" if same_class else "✗")
            rif = f"{top.get('argmax_expected', '?')}{mark}"

        rows.append(
            "| {model} | {mode} | {wall} | {cpu} | {ratio} | {blocks} | {flush} | {up}/{down} "
            "| {err} | {rif} | {sm} | {vram} | {ok} |".format(
                model=r["model"],
                mode=r["mode"],
                wall=fmt(median),
                cpu=fmt(cpu_ms),
                ratio=ratio,
                blocks=blocks.get("convex_blocks", "—"),
                flush=gpu.get("flushes", "—"),
                up=fmt(gpu.get("upload_mb")),
                down=fmt(gpu.get("download_mb")),
                err=err,
                rif=rif,
                sm=fmt((system.get("sm_pct") or {}).get("p95"), ".0f"),
                vram=fmt(system.get("vram_delta_mb"), ".0f"),
                ok="ok" if r.get("ok") else "**FAIL**",
            )
        )

    env = summary.get("env", {})
    head = [
        f"# Test suite — {summary['tag']}",
        "",
        f"host `{env.get('host', '?')}` · commit `{env.get('commit', '?')}`"
        f"{' (dirty)' if env.get('dirty') else ''} · ORT {env.get('ort', '?')}"
        f" · driver {env.get('driver', '?')} · GPU {env.get('gpu', '?')}",
        "",
        "| model | mode | wall ms | CPU EP ms | ratio | blocks | flushes | up/down MB "
        "| error | ref | sm p95 % | ΔVRAM MB | outcome |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    notes = [
        "",
        "Wall = median of iterations after the first. `sm` is the p95, not the mean,",
        "since the runner dilutes it by also running the reference session on the CPU EP.",
        "ΔVRAM is a **global** delta (`vram_source: global-delta`), not a per-process",
        "measurement. `ref` is the class expected by the model zoo's official data",
        "(`test_data_set_*`), with ✓ if the backend reproduces it exactly and `~` if it",
        "reproduces the class but not the values (see docs/testsuite.md).",
    ]
    invalid = sorted({r["model"] for r in summary["results"] if r.get("perf_valid") is False})
    if invalid:
        notes.append(
            f"\n⚠ perf invalid (software or unknown device): {', '.join(invalid)}."
        )
    return "\n".join(head + rows + notes) + "\n"


def key(r: dict) -> tuple[str, str]:
    return (r["model"], r["mode"])


def gate(summary: dict, baseline: dict) -> list[str]:
    """Comparison against the baseline. Returns the violations, one line each."""
    base = {key(r): r for r in baseline.get("results", [])}
    failures = []
    for cur in summary["results"]:
        old = base.get(key(cur))
        if old is None:
            continue
        tag = f"{cur['model']}/{cur['mode']}"

        if old.get("ok") and not cur.get("ok"):
            failures.append(f"crash/regression: {tag} completed in the baseline, now fails")

        old_ref, new_ref = old.get("reference") or {}, cur.get("reference") or {}
        if old_ref.get("ok") and new_ref.get("ok") is False:
            failures.append(f"correctness: {tag} no longer passes the official reference")

        old_acc, new_acc = old.get("accuracy") or {}, cur.get("accuracy") or {}
        if old_acc.get("tolerated") and new_acc.get("tolerated") is False:
            failures.append(f"correctness: {tag} was within tolerance, now it is not")
        ow, nw = old_acc.get("worst_relative"), new_acc.get("worst_relative")
        if ow and nw and nw >= 10 * ow:
            failures.append(f"correctness: {tag} relative error {ow:.2e} → {nw:.2e} (≥ 10×)")

        tol = cur.get("perf_tol", GATE_PERF_DEFAULT)
        om = (old.get("wall_ms") or {}).get("median")
        nm = (cur.get("wall_ms") or {}).get("median")
        # the `cpu` mode runs without the plugin (`STT_NO_VULKAN=1`): no change
        # to the EP can affect it, so comparing it spends the threshold on host
        # noise. It stays in the report as a reference, not in the gate.
        # and golden models do not measure the product: they are there for
        # correctness, with low `iters` and inputs fixed by the reference
        if cur["mode"] == "cpu" or cur.get("golden"):
            om = nm = None
        if om and nm and cur.get("perf_valid") and old.get("perf_valid"):
            delta = 100 * (nm - om) / om
            if delta > tol:
                failures.append(
                    f"performance: {tag} wall {om:.1f} → {nm:.1f} ms (+{delta:.1f}%, threshold {tol:.0f}%)"
                )

        # Golden models are exempt from the *wall* gate above — low `iters` and
        # fixed inputs make the milliseconds noisy — but not from these. Flushes
        # and bytes transferred are deterministic, they do not care how many
        # iterations ran, and they lead the wall clock. Exempting them too is how
        # resnet50-qdq went 11.4 → 21.5 ms (128 MB re-uploaded per run) under a
        # gate that reported "no regression".
        og, ng = old.get("gpu") or {}, cur.get("gpu") or {}
        for field, label in (
            ("flushes", "flushes"),
            ("upload_mb", "MB uploaded"),
            ("download_mb", "MB downloaded"),
        ):
            o, n = og.get(field), ng.get(field)
            if o is not None and n is not None and n > o:
                failures.append(f"fragmentation: {tag} {label} {o} → {n}")

        ob, nb = old.get("blocks") or {}, cur.get("blocks") or {}
        if (ob.get("convex_blocks"), nb.get("convex_blocks")) != (None, None):
            o, n = ob.get("convex_blocks"), nb.get("convex_blocks")
            if o is not None and n is not None and n > o:
                failures.append(f"coverage: {tag} convex blocks {o} → {n}")
        o, n = ob.get("nodes_claimed"), nb.get("nodes_claimed")
        if o is not None and n is not None and n < o:
            failures.append(f"coverage: {tag} claimed nodes {o} → {n}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--baseline", type=Path)
    args = ap.parse_args()

    summary = collect(args.run_dir)
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = markdown(summary)
    (args.run_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)

    failed = [r for r in summary["results"] if not r.get("ok") and r.get("status") != "skip"]
    if failed:
        print("failed runs: " + ", ".join(f"{r['model']}/{r['mode']}" for r in failed))

    if args.baseline:
        if not args.baseline.exists():
            print(f"baseline missing: {args.baseline}")
            return 2
        violations = gate(summary, json.loads(args.baseline.read_text(encoding="utf-8")))
        print("\n## Regression gate\n")
        if violations:
            for v in violations:
                print(f"- ✗ {v}")
            return 1
        print("- ✓ no regression against the baseline")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
