import fcntl
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from onnx_vulkan_pipeline import process

SLEEP = [sys.executable, "-c", "import time; time.sleep(30)"]


def test_a_deadline_ends_a_worker_whose_own_timeout_has_not_expired():
    started = time.monotonic()
    with pytest.raises(process.BudgetExhausted, match="its own timeout had not expired"):
        with process.deadline(0.5):
            process.command(SLEEP, timeout=60)
    assert time.monotonic() - started < 10


def test_a_worker_over_its_own_timeout_is_still_a_timeout_inside_a_deadline():
    with process.deadline(60):
        with pytest.raises(subprocess.TimeoutExpired):
            process.command(SLEEP, timeout=0.3)


def test_the_deadline_kills_descendants_that_left_the_session(tmp_path):
    marker = tmp_path / "pid"
    code = ("import os, subprocess, sys, time\n"
            f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)\n"
            f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
            "time.sleep(30)\n")
    with pytest.raises(process.BudgetExhausted):
        with process.deadline(1.0):
            process.command([sys.executable, "-c", code], timeout=60)
    pid = int(marker.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        # a zombie of a killed grandchild is reaped by init shortly
        time.sleep(0.1)
    else:
        pytest.fail("a descendant survived the deadline")


def test_no_work_starts_after_the_deadline_is_spent():
    with process.deadline(0.05):
        time.sleep(0.1)
        with pytest.raises(process.BudgetExhausted, match="spent"):
            process.command([sys.executable, "-c", "pass"])


def test_scopes_nest_to_the_earliest_deadline_and_do_not_leak():
    assert process.remaining() is None
    with process.deadline(100):
        outer = process.remaining()
        with process.deadline(1000):
            assert process.remaining() <= outer
        with process.deadline(5):
            assert process.remaining() <= 5
        assert process.remaining() > 5
    assert process.remaining() is None


def test_a_deadline_does_not_apply_to_another_thread():
    seen = []
    with process.deadline(100):
        thread = threading.Thread(target=lambda: seen.append(process.remaining()))
        thread.start()
        thread.join()
    assert seen == [None]


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), True, "5"])
def test_invalid_deadlines_are_rejected(seconds):
    with pytest.raises(ValueError):
        with process.deadline(seconds):
            pass


def test_a_deadline_ends_the_wait_for_the_gpu_lane(monkeypatch, tmp_path):
    # An isolated lock file: the host's real GPU lane is never touched.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    path = Path(tempfile.gettempdir()) / f"onnx-vulkan-gpu-{os.getuid()}.lock"
    with path.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        try:
            started = time.monotonic()
            with pytest.raises(process.BudgetExhausted, match="GPU lane"):
                with process.deadline(0.3):
                    with process.gpu_lock(60):
                        pass
            assert time.monotonic() - started < 10
            # without a deadline the lane's own timeout is an ordinary failure
            with pytest.raises(TimeoutError, match="busy"):
                with process.gpu_lock(0.2):
                    pass
        finally:
            fcntl.flock(held, fcntl.LOCK_UN)
