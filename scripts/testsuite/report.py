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
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import peaks as peaks_mod  # noqa: E402

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
    notes += roofline_table(summary)
    invalid = sorted({r["model"] for r in summary["results"] if r.get("perf_valid") is False})
    if invalid:
        notes.append(
            f"\n⚠ perf invalid (software or unknown device): {', '.join(invalid)}."
        )
    return "\n".join(head + rows + notes) + "\n"


def roofline_table(summary: dict) -> list[str]:
    """Roofline of the Pareto head, one row per job that has one.

    Its purpose is to answer "is this kernel worth touching" before anyone
    touches it: `class` names the roof that binds and `efficiency` how far the
    kernel is from it. A `memory` head near 100% of bandwidth is finished work;
    a `latency` head is starved of parallelism and the lever is the grid, not
    the tile (`plan.md` §9.3).
    """
    rows = []
    skipped = 0
    for r in summary["results"]:
        if r.get("status") == "skip":
            continue
        pareto = (r.get("gpu") or {}).get("pareto") or []
        head = pareto[0] if pareto else {}
        if "class" not in head:
            continue
        # a Roofline needs a device the peaks describe: on a software
        # rasterizer every kernel classifies `latency` and says nothing
        if r.get("perf_valid") is False:
            skipped += 1
            continue
        rows.append(
            "| {model} | {mode} | {op} | {ms} | {gflops} | {gbs} | {ai} | {pc}/{pb} | {cls} |".format(
                model=r["model"],
                mode=r["mode"],
                op=head["op"],
                ms=fmt(head.get("ms"), ".3f"),
                gflops=fmt(head.get("gflops") / 1e3 if head.get("gflops") else None, ".2f"),
                gbs=fmt(head.get("gb_s"), ".0f"),
                ai=fmt(head.get("intensity"), ".2f"),
                pc=fmt(head.get("pct_peak_compute"), ".0f"),
                pb=fmt(head.get("pct_peak_bw"), ".0f"),
                cls=head.get("class", "—"),
            )
        )
    if not rows:
        return (
            ["", f"Roofline: {skipped} job(s) had one, all on a non-benchmarkable device."]
            if skipped
            else []
        )
    peaks = next((r.get("peaks") for r in summary["results"] if r.get("peaks")), {}) or {}
    return [
        "",
        "## Roofline (Pareto head)",
        "",
        f"Peaks: {peaks.get('peak_tflops_fp32', '?')} TFLOP/s fp32, "
        f"{peaks.get('peak_gb_s', '?')} GB/s → ridge "
        f"{peaks_mod.ridge(peaks):.1f} FLOP/B." if peaks else "",
        "FLOPs are analytic; bytes are **compulsory** traffic (each tensor once),",
        "so a percentage is distance from the best any implementation could do.",
        "",
        "| model | mode | op | ms | TFLOP/s | GB/s | FLOP/B | % peak c/b | class |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
    ]


def key(r: dict) -> tuple[str, str]:
    return (r["model"], r["mode"])


def parity_failures(summary: dict) -> list[str]:
    """Standalone-vs-EP parity, checked **without** a baseline.

    Every other gate is a comparison against a previous run, because it asks
    whether something got worse. This one is absolute: the two hosts run the
    same kernels over the same graph, so their outputs are identical or one of
    them is wrong. There is no tolerance to spend and no baseline to promote.
    """
    failures = []
    for r in summary["results"]:
        if r.get("mode") != "standalone" or r.get("status") == "skip":
            continue
        tag = f"{r['model']}/standalone"
        parity = r.get("parity")
        if not parity or not parity.get("outputs"):
            # a mode whose only verdict is missing must fail, not pass quietly
            failures.append(f"parity: {tag} produced no standalone-vs-EP comparison")
            continue
        if not parity.get("bit_exact"):
            worst = max(o["max_abs_delta"] for o in parity["outputs"])
            failures.append(
                f"parity: {tag} standalone and EP disagree (max|Δ|={worst:.3e}, "
                f"relative={parity.get('vs_ep')}) — same kernels, so one path is wrong"
            )
    return failures


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


TSV_COLUMNS = (
    "when",
    "commit",
    "tag",
    "model",
    "mode",
    "wall_ms",
    "compute_ms",
    "target_op",
    "target_ms",
    "blocks",
    "flushes",
    "status",
    "description",
)


def target_op(result: dict) -> tuple[str | None, float | None]:
    """The op the experiment is judged on.

    Default is the Pareto head, because the head is what moves a model and it
    *changes* as kernels are optimized — pinning it in the manifest would mean
    editing the manifest every time an optimization succeeds. `TESTSUITE_TARGET_OP`
    overrides it while hunting a specific op that is not (or no longer) the head.
    """
    pareto = (result.get("gpu") or {}).get("pareto") or []
    if not pareto:
        return None, None
    wanted = os.environ.get("TESTSUITE_TARGET_OP")
    if wanted:
        entry = next((e for e in pareto if e.get("op") == wanted), None)
        # asked for an op this model does not run: report the miss as a zero,
        # not as the head, or the loop would compare two different ops
        return (wanted, entry["ms"] if entry else 0.0)
    head = pareto[0]
    return head.get("op"), head.get("ms")


def append_tsv(path: Path, summary: dict, description: str, tag: str) -> int:
    """One line per job, appended to a file that outlives `--keep` rotation.

    This is the loop's history: `runs/<tag>/` is deleted after N runs, so the
    log cannot live inside it (AutoKernel's `results.tsv`, `plan.md` §9.5 P0).
    """
    env = summary.get("env") or {}
    rows = []
    for r in summary["results"]:
        wall = r.get("wall_ms") or {}
        gpu = r.get("gpu") or {}
        blocks = r.get("blocks") or {}
        op, op_ms = target_op(r)
        rows.append(
            {
                "when": env.get("when", ""),
                "commit": (env.get("commit", "") or "") + ("+dirty" if env.get("dirty") else ""),
                "tag": tag,
                "model": r.get("model", ""),
                "mode": r.get("mode", ""),
                "wall_ms": fmt(wall.get("median"), ".3f", ""),
                "compute_ms": fmt(gpu.get("compute_ms"), ".3f", ""),
                "target_op": op or "",
                "target_ms": fmt(op_ms, ".3f", ""),
                "blocks": fmt(blocks.get("convex_blocks"), "d", ""),
                "flushes": fmt(gpu.get("flushes"), "d", ""),
                "status": r.get("status") or ("ok" if r.get("ok") else "KO"),
                "description": description,
            }
        )
    new = not path.exists()
    with path.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("\t".join(TSV_COLUMNS) + "\n")
        for row in rows:
            # tabs and newlines in a free-text description would break the format
            fh.write("\t".join(str(row[c]).replace("\t", " ").replace("\n", " ") for c in TSV_COLUMNS) + "\n")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--desc", default="", help="description recorded in results.tsv")
    ap.add_argument(
        "--results-tsv",
        type=Path,
        help="append one line per job here (default: <run_dir>/../results.tsv)",
    )
    args = ap.parse_args()

    summary = collect(args.run_dir)
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = markdown(summary)
    (args.run_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)

    tsv = args.results_tsv or (args.run_dir.parent / "results.tsv")
    # resolved: invoked on `runs/latest` the tag would otherwise be recorded as
    # "latest", and a row that cannot name its run directory is not history
    tag = args.run_dir.resolve().name
    n = append_tsv(tsv, summary, args.desc or tag, tag)
    print(f"\n{n} line(s) appended to {tsv}")

    failed = [r for r in summary["results"] if not r.get("ok") and r.get("status") != "skip"]
    if failed:
        print("failed runs: " + ", ".join(f"{r['model']}/{r['mode']}" for r in failed))

    # absolute, so it runs with or without a baseline
    parity = parity_failures(summary)
    if parity:
        print("\n## Standalone parity\n")
        for v in parity:
            print(f"- ✗ {v}")
    elif any(r.get("mode") == "standalone" for r in summary["results"]):
        n = sum(
            1
            for r in summary["results"]
            if r.get("mode") == "standalone" and (r.get("parity") or {}).get("bit_exact")
        )
        print(f"\n## Standalone parity\n\n- ✓ {n} model(s) bit-exact against the EP")

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
    return 1 if (failed or parity) else 0


if __name__ == "__main__":
    raise SystemExit(main())
