"""Create a local research baseline without changing HEAD or the user's index."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def snapshot(root: Path) -> str:
    """Snapshot tracked edits and untracked project sources in an isolated index."""
    def git(*args: str, env=None, input=None) -> str:
        return subprocess.run(["git", *args], cwd=root, env=env, input=input,
                              text=True, capture_output=True, check=True, timeout=60).stdout.strip()

    with tempfile.TemporaryDirectory(prefix="vulkan-research-index-") as temporary:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}
        parent = git("rev-parse", "HEAD")
        git("read-tree", parent, env=env)
        git("add", "-u", "--", ".", env=env)
        sources = git("ls-files", "--others", "--exclude-standard", "-z").split("\0")
        selected = [name for name in sources if name and (
            Path(name).parts[0] in {"crates", "python", "scripts", "tests"}
            or name in {"pyproject.toml", "uv.lock"})]
        if selected:
            git("add", "--", *selected, env=env)
        tree = git("write-tree", env=env)
        return git("commit-tree", tree, "-p", parent, env=env,
                   input="Research baseline snapshot of current project sources\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New baseline worktree directory")
    args = parser.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    destination = args.output.resolve()
    if destination.exists():
        parser.error("output already exists")
    revision = snapshot(root)
    subprocess.run(["git", "worktree", "add", "--detach", str(destination), revision],
                   cwd=root, check=True, timeout=60, capture_output=True, text=True)
    print(json.dumps({"base_commit": revision, "baseline_root": str(destination)}))


if __name__ == "__main__":
    main()
