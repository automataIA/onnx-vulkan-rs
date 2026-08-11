#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""System metrics sampler for the test suite (`scripts/testsuite.sh`).

    sample_metrics.py sample -p model-runner -o metrics.csv --interval-ms 100 \\
        --stop-file .testsuite-stop
    sample_metrics.py gpu                       # {"gpu": ..., "driver": ...}

`sample` writes a CSV `t_ms,sm,mem_pct,fb_mb,proc_rss_mb,proc_cpu_pct` preceded
by a comment line carrying the VRAM idle baseline. It stops when the observed
process exits or when the stop file appears. The column contract is the one
`parse_run.py` reads, and a column that cannot be measured is written as `-1`,
which that consumer drops.

Deliberate limitations, declared in the data (see `docs/testsuite.md`):
  - `fb_mb` is the **global** GPU memory, not the process's. The consumer uses
    the delta against `idle_fb_mb` and marks it `vram_source: global-delta`.
  - `proc_cpu_pct` is in units of "100% = one saturated core".

GPU counters come from NVML, loaded directly rather than through `nvidia-smi`:
the sampler runs at 10 Hz inside the timed window, and forking a process per
sample would perturb the measurement it is taking. On a machine with no NVIDIA
driver the library simply does not load and the GPU columns stay `-1` — the
process columns are still measured, so the run is degraded, never invalid.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
import time
from pathlib import Path

CLK_TCK = os.sysconf("SC_CLK_TCK")


class Nvml:
    """The three NVML calls this sampler needs, or nothing at all.

    `ok` is false whenever the driver is absent or refuses to initialize; every
    accessor then answers with the sentinel the CSV contract expects.
    """

    def __init__(self) -> None:
        self.ok = False
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError:
            return
        if self.lib.nvmlInit_v2() != 0:
            return
        handle = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
            self.lib.nvmlShutdown()
            return
        self.handle = handle
        self.ok = True

    def utilization(self) -> tuple[int, int]:
        """(compute %, memory-traffic %) — the `sm` and `mem_pct` columns."""
        if not self.ok:
            return -1, -1

        class Util(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

        util = Util()
        if self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util)) != 0:
            return -1, -1
        return util.gpu, util.memory

    def used_mb(self) -> int:
        if not self.ok:
            return -1

        class Memory(ctypes.Structure):
            _fields_ = [
                ("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong),
            ]

        mem = Memory()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem)) != 0:
            return -1
        return mem.used // (1024 * 1024)

    def identity(self) -> tuple[str, str]:
        if not self.ok:
            return "unknown", "unknown"
        name = ctypes.create_string_buffer(96)
        driver = ctypes.create_string_buffer(96)
        got_name = self.lib.nvmlDeviceGetName(self.handle, name, 96) == 0
        got_driver = self.lib.nvmlSystemGetDriverVersion(driver, 96) == 0
        return (
            name.value.decode() if got_name else "unknown",
            driver.value.decode() if got_driver else "unknown",
        )


def pids_of(name: str) -> list[int]:
    """Every live PID whose comm is exactly `name`.

    `comm` and not the command line: the suite names a binary, and matching a
    command line would also catch the shell that launched it and this sampler's
    own arguments.
    """
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / "comm").read_text().strip() == name:
                found.append(int(entry.name))
        except OSError:
            continue  # the process exited between listing and reading
    return found


def proc_sample(pids: list[int]) -> tuple[int, float] | None:
    """(RSS MiB, CPU seconds) summed over the pids, or None if all are gone."""
    rss_kb = 0
    jiffies = 0
    alive = False
    for pid in pids:
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            continue
        alive = True
        if m := re.search(r"^VmRSS:\s+(\d+) kB", status, re.M):
            rss_kb += int(m.group(1))
        # fields 14 and 15 (1-based) are utime and stime, but the second field
        # is the comm in parentheses and may itself contain spaces
        fields = stat[stat.rindex(")") + 2 :].split()
        jiffies += int(fields[11]) + int(fields[12])
    if not alive:
        return None
    return rss_kb // 1024, jiffies / CLK_TCK


def sample(args: argparse.Namespace) -> int:
    nvml = Nvml()
    stop = Path(args.stop_file) if args.stop_file else None

    def stopped() -> bool:
        return stop is not None and stop.exists()

    idle_fb = nvml.used_mb()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(f"# idle_fb_mb={idle_fb} interval_ms={args.interval_ms} process={args.process}\n")
        fh.write("t_ms,sm,mem_pct,fb_mb,proc_rss_mb,proc_cpu_pct\n")
        fh.flush()

        # 1) wait for the process to appear: the suite starts the sampler first
        deadline = time.monotonic() + args.wait_seconds
        while not (pids := pids_of(args.process)):
            if stopped() or time.monotonic() > deadline:
                return 0
            time.sleep(0.02)

        # 2) sample until the process ends or the stop signal arrives
        t0 = time.monotonic()
        prev_cpu: float | None = None
        prev_t = 0.0
        while True:
            # rescanned every sample and not held: the runner may fork, and a
            # stale pid list would report the parent's memory as the whole run's
            pids = pids_of(args.process)
            current = proc_sample(pids) if pids else None
            if current is None:
                break
            rss_mb, cpu_sec = current
            now = (time.monotonic() - t0) * 1000.0
            cpu_pct = 0
            if prev_cpu is not None:
                dt = (now - prev_t) / 1000.0
                if dt > 0:
                    cpu_pct = int(100 * (cpu_sec - prev_cpu) / dt)
            prev_cpu, prev_t = cpu_sec, now
            sm, mem_pct = nvml.utilization()
            fh.write(f"{now:.0f},{sm},{mem_pct},{nvml.used_mb()},{rss_mb},{cpu_pct}\n")
            fh.flush()
            if stopped():
                break
            time.sleep(args.interval_ms / 1000.0)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="sample a running process until it exits")
    s.add_argument("-p", "--process", required=True, help="process name (comm)")
    s.add_argument("-o", "--out", required=True, help="destination CSV")
    s.add_argument("--interval-ms", type=int, default=100)
    s.add_argument("--stop-file", default="")
    s.add_argument("--wait-seconds", type=int, default=300)

    sub.add_parser("gpu", help="print the GPU name and driver version as JSON")

    args = ap.parse_args()
    if args.command == "gpu":
        name, driver = Nvml().identity()
        json.dump({"gpu": name, "driver": driver}, sys.stdout)
        print()
        return 0
    return sample(args)


if __name__ == "__main__":
    raise SystemExit(main())
