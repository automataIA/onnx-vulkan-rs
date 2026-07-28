#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Reads `tests/models.toml` and produces the execution matrix.

Three subcommands, all meant to be consumed by `scripts/testsuite.sh`:

    manifest.py jobs  [-m NAME]... [-M MODE]... [--iters N]   → one job per line
    manifest.py stage [-m NAME]...                            → paths to propagate
    manifest.py fetch [-m NAME]...                            → download what is missing

The fields are fixed (see `JOB_FIELDS`) and separated by US (`\\x1f`): bash reads
them with `IFS=$'\\x1f'` without interpreting either TOML or JSON. The separator
is not tab because tab is whitespace for `read`, which would merge adjacent
empty fields.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "tests" / "models.toml"
MODES = ("cpu", "registry", "compile")
SEP = "\x1f"  # field separator: non-whitespace, so empty fields are preserved

JOB_FIELDS = (
    "name",
    "mode",
    "runner",
    "iters",
    "path",
    "args",  # JSON list
    "stats",  # 0/1
    "validate",
    "expect",
    "instrument",  # JSON dict or ""
    "perf_tol",
    "size_mb",
    "reference",  # directory test_data_set_* or ""
    "golden",  # 0/1: correctness gate, not performance
    "status",  # run | skip
    "reason",
)


def load(manifest: Path) -> list[dict]:
    with manifest.open("rb") as fh:
        return tomllib.load(fh).get("model", [])


def select(models: list[dict], names: list[str]) -> list[dict]:
    if not names:
        return [m for m in models if m.get("default")]
    by_name = {m["name"]: m for m in models}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        sys.exit(f"unknown model in manifest: {', '.join(unknown)}")
    return [by_name[n] for n in names]


def job_rows(model: dict, mode: str, iters: int | None) -> dict:
    """One job, already resolved: `status = skip` carries the reason, it does not disappear."""
    row = {
        "name": model["name"],
        "mode": mode,
        "runner": model.get("runner", "model-runner"),
        "iters": str(iters if iters else model.get("iters", 1)),
        "path": model.get("path", ""),
        "args": json.dumps(model.get("args", [])),
        # the VULKAN_EP_STATS=1 pass makes no sense without the plugin
        "stats": "0" if mode == "cpu" else ("1" if model.get("stats", True) else "0"),
        "validate": model.get("validate", "outputs"),
        "expect": model.get("expect", ""),
        "instrument": json.dumps(model["instrument"]) if "instrument" in model else "",
        "perf_tol": str(model.get("perf_tol", 10.0)),
        "size_mb": str(model.get("size_mb", 0)),
        "reference": model.get("reference", ""),
        "golden": "1" if model.get("golden") else "0",
        "status": "run",
        "reason": "",
    }
    if "skip" in model:
        row["status"] = "skip"
        row["reason"] = model["skip"]
    elif mode == "cpu" and row["runner"] == "model-runner":
        # model-runner already runs a reference session on CPU EP only at every
        # run: a separate `cpu` mode would be the same measurement, twice.
        row["status"] = "skip"
        row["reason"] = "model-runner already measures the CPU EP anyway"
    return row


def fetch(name: str, spec: dict) -> None:
    """Downloads and extracts `member` from the archive into `dir`, if missing.

    The ONNX model zoo archives contain **model and `test_data_set_*` together**:
    a single download serves both, and extracting the whole member avoids
    downloading the same file twice.

    Idempotent by construction: if the directory exists, no network is touched.
    That is what makes it acceptable to call it at every suite startup.
    """
    target = ROOT / spec["dir"]
    if target.exists():
        return
    url, member = spec["url"], spec["member"]
    print(f"== {name}: downloading {url}", file=sys.stderr)

    import tarfile
    import tempfile
    import urllib.request

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "reference.tar.gz"
        try:
            urllib.request.urlretrieve(url, archive)
        except OSError as e:
            # without network the suite must continue: the reference is an
            # additional check, not a precondition to measure
            print(f"   ! download failed ({e}); {name} will run without it", file=sys.stderr)
            return
        with tarfile.open(archive) as tar:
            wanted = [m for m in tar.getmembers() if m.name.startswith(member)]
            if not wanted:
                print(f"   ! '{member}' missing from the archive", file=sys.stderr)
                return
            tar.extractall(Path(tmp), members=wanted, filter="data")
        target.parent.mkdir(parents=True, exist_ok=True)
        (Path(tmp) / member).rename(target)
    print(f"   → {spec['dir']}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("jobs", "stage", "fields", "fetch"))
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("-m", "--model", action="append", default=[])
    ap.add_argument("-M", "--mode", action="append", default=[], choices=MODES)
    ap.add_argument("--iters", type=int)
    args = ap.parse_args()

    if args.command == "fields":
        print(SEP.join(JOB_FIELDS))
        return 0

    models = select(load(args.manifest), args.model)

    if args.command == "fetch":
        for model in models:
            if "skip" not in model and "fetch" in model:
                fetch(model["name"], model["fetch"])
        return 0

    if args.command == "stage":
        for model in models:
            path = model.get("path")
            if not path or "skip" in model:
                continue
            print(path)
            for extra in model.get("args", []):
                # arguments that are files (e.g. the stt-app wav) must be staged
                if (ROOT / extra).exists():
                    print(extra)
            # the reference data is needed where the model runs, not here
            reference = model.get("reference")
            if reference and (ROOT / reference).exists():
                print(reference)
        return 0

    modes = args.mode or ["cpu", "compile"]
    for model in models:
        for mode in modes:
            row = job_rows(model, mode, args.iters)
            print(SEP.join(row[f] for f in JOB_FIELDS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
