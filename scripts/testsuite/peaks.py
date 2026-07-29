#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Device peaks for the Roofline: fp32 TFLOP/s and GB/s.

One table, used from two places — `testsuite.sh` writes the resolved peaks into
`runs/<tag>/env.json`, `parse_run.py` reads them back (or falls back to this
table using the device name the log reports). Called as a script it prints the
two numbers as JSON:

    peaks.py "NVIDIA GeForce RTX 4070"   # {"peak_tflops_fp32": 29.15, ...}

Numbers are vendor specifications, not measurements: `pct_peak_*` is therefore
a fraction of a theoretical maximum no kernel reaches. Override per run with
`TESTSUITE_PEAK_TFLOPS` / `TESTSUITE_PEAK_GB_S` — needed on any device absent
from the table, which resolves to nothing rather than guessing.
"""

from __future__ import annotations

import json
import os
import sys

# substring of the device name → (fp32 TFLOP/s, GB/s)
TABLE: dict[str, tuple[float, float]] = {
    "rtx 4070 ti": (40.09, 504.2),
    "rtx 4070": (29.15, 504.2),
    "rtx 4090": (82.58, 1008.0),
    "rtx 3060": (12.74, 360.0),
    "rx 7900 xtx": (61.42, 960.0),
    "rx 7800 xt": (37.32, 624.1),
}


def resolve(device: str | None) -> dict:
    """Peaks for a device name, `{}` when unknown.

    A software rasterizer is deliberately absent from the table: its Roofline
    would be meaningless in exactly the way `perf_valid: false` already says.
    """
    tflops = os.environ.get("TESTSUITE_PEAK_TFLOPS")
    gb_s = os.environ.get("TESTSUITE_PEAK_GB_S")
    if tflops and gb_s:
        return {"peak_tflops_fp32": float(tflops), "peak_gb_s": float(gb_s)}
    name = (device or "").lower()
    # longest match first, so "rtx 4070 ti" wins over "rtx 4070"
    for key in sorted(TABLE, key=len, reverse=True):
        if key in name:
            peak_tflops, peak_gb_s = TABLE[key]
            return {"peak_tflops_fp32": peak_tflops, "peak_gb_s": peak_gb_s}
    return {}


def ridge(peaks: dict) -> float | None:
    """FLOP/B at which the two roofs meet: below it a kernel is bandwidth-bound."""
    if not peaks:
        return None
    return peaks["peak_tflops_fp32"] * 1e12 / (peaks["peak_gb_s"] * 1e9)


if __name__ == "__main__":
    print(json.dumps(resolve(sys.argv[1] if len(sys.argv) > 1 else None)))
