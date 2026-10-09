"""Research snapshots preserve the user's staged/unstaged split."""
import importlib.util
from pathlib import Path
import subprocess

SPEC = importlib.util.spec_from_file_location("research_snapshot", Path(__file__).parents[1] / "scripts/research/snapshot.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_snapshot_keeps_head_index_and_private_untracked_files(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()
    git("init", "-q")
    git("config", "user.name", "Snapshot test")
    git("config", "user.email", "test@example.invalid")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base")
    git("add", ".")
    git("commit", "-qm", "base")
    head = git("rev-parse", "HEAD")
    tracked.write_text("staged")
    git("add", ".")
    tracked.write_text("working")
    (tmp_path / "python").mkdir()
    (tmp_path / "python/new.py").write_text("print('source')")
    (tmp_path / ".env").write_text("private")
    before = git("status", "--porcelain")
    revision = MODULE.snapshot(tmp_path)
    assert git("rev-parse", "HEAD") == head
    assert git("status", "--porcelain") == before
    assert git("show", ":tracked.txt") == "staged"
    assert git("show", f"{revision}:tracked.txt") == "working"
    assert git("show", f"{revision}:python/new.py") == "print('source')"
    assert ".env" not in git("ls-tree", "--name-only", revision).splitlines()
