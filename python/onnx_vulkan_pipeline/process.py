"""Bounded process trees and a host-wide serialized GPU lane (Linux)."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time


def command(argv, *, cwd=None, env=None, timeout=300):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive")
    process = subprocess.Popen(
        [str(arg) for arg in argv], cwd=cwd,
        env={**os.environ, **(env or {})}, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        # Some runners start their own sessions. Include descendants even when
        # they left our process group before terminating the parent.
        def descendants(pid):
            try:
                children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            except OSError:
                return []
            return [child for raw in children for child in [*descendants(int(raw)), int(raw)]]
        for pid in [*descendants(process.pid), process.pid]:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.communicate()
        raise
    if process.returncode:
        message = f"{argv[0]} exited {process.returncode}: {_tail(stderr, 8000)}\n{_tail(stdout, 2000)}"
        if process.returncode < 0:
            # Our own timeout raises above; a signal here came from outside.
            message += (f"\nError: {Path(str(argv[0])).name} was killed by "
                        f"{signal.Signals(-process.returncode).name} from outside onnx-vulkan, usually an "
                        "out-of-memory killer (earlyoom, systemd-oomd or the kernel's); see `journalctl -k` "
                        "and the free memory")
        error = RuntimeError(message)
        # The message keeps the tails; evidence and verdicts need the whole
        # output, since a long log pushes the failing comparison out of a tail.
        error.output = f"{stderr}\n{stdout}"
        raise error
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)



def _tail(text, limit):
    """The end of a long output, starting at a whole line."""
    return text if len(text) <= limit else "…" + text[-limit:].split("\n", 1)[-1]


def failure_reason(error):
    """The exception line of a failed worker's output.

    Workers report their whole output, and logging may follow the traceback
    (ORT's calibrator prints its histogram settings after the error), so the
    last line is not necessarily the cause.
    """
    lines = [line.strip() for line in str(error).strip().splitlines() if line.strip()]
    for line in reversed(lines):
        if re.search(r"(?:^|\s)[\w.]*(?:Error|Exception):", line):
            return line
    return lines[-1] if lines else type(error).__name__


@contextmanager
def gpu_lock(timeout=300):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    path = Path(tempfile.gettempdir()) / f"onnx-vulkan-gpu-{os.getuid()}.lock"
    with path.open("a") as lock:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("GPU lane is busy")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
