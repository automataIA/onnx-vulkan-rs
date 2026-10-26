"""Bounded process trees and a host-wide serialized GPU lane (Linux)."""
from __future__ import annotations

from contextlib import contextmanager
import contextvars
import fcntl
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time

# Monotonic instants after which no subprocess may run or be waited for. The
# innermost scope of `deadline` is the earliest one, so the variable holds the
# effective limit. It is scoped to the context (thread or task), never global.
_DEADLINE = contextvars.ContextVar("onnx_vulkan_deadline", default=None)


class BudgetExhausted(Exception):
    """A `deadline` scope ran out. Not a worker failure and not a `timeout`:
    work interrupted by it is incomplete, not wrong, and may be resumed."""


@contextmanager
def deadline(seconds, *, clock=time.monotonic):
    """Cap every `command` and `gpu_lock` wait inside the block, descendants of
    the commands included, at `seconds` from now (or at the enclosing deadline,
    if that is earlier). Exhaustion raises `BudgetExhausted`; a worker that
    merely exceeds its own `timeout` still raises `TimeoutExpired`."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("deadline must be a positive number of seconds")
    enclosing = _DEADLINE.get()
    limit = clock() + seconds if enclosing is None else min(enclosing[0], clock() + seconds)
    token = _DEADLINE.set((limit, clock))
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def remaining():
    """Seconds left in the innermost `deadline` scope, or None outside any."""
    scope = _DEADLINE.get()
    return None if scope is None else scope[0] - scope[1]()


def _capped(timeout):
    """`timeout` limited by the deadline: (effective timeout, whether the deadline binds)."""
    left = remaining()
    if left is None:
        return timeout, False
    if left <= 0:
        raise BudgetExhausted("the time budget is spent")
    return (left, True) if left < timeout else (timeout, False)


def command(argv, *, cwd=None, env=None, timeout=300):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive")
    timeout, bound = _capped(timeout)
    # A None value removes the variable the child would inherit.
    environment = {k: v for k, v in {**os.environ, **(env or {})}.items() if v is not None}
    process = subprocess.Popen(
        [str(arg) for arg in argv], cwd=cwd,
        env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException as interruption:
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
        if bound and isinstance(interruption, subprocess.TimeoutExpired):
            raise BudgetExhausted(f"the time budget ended {Path(str(argv[0])).name} (its own timeout "
                                  "had not expired)") from interruption
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
    wait, bound = _capped(timeout)
    path = Path(tempfile.gettempdir()) / f"onnx-vulkan-gpu-{os.getuid()}.lock"
    with path.open("a") as lock:
        give_up = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= give_up:
                    if bound:
                        raise BudgetExhausted("the time budget ended while waiting for the GPU lane")
                    raise TimeoutError("GPU lane is busy")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
