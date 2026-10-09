"""Research supervisor tests using real git worktrees and explicit fake commands."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys

import pytest

from onnx_vulkan_pipeline.pipeline import seal
import onnx_vulkan_pipeline.research as research_module


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Tests")
    (repo / "README").write_text("base\n")
    _git(repo, "add", "README")
    _git(repo, "commit", "-qm", "base")
    return repo


def _fixture(repo: Path, *, status: str = "verified", profile: str = "p", name: str = "fixture") -> Path:
    fixture = repo / name
    fixture.mkdir()
    (fixture / "model.onnx").write_bytes(b"fixture model")
    seal(fixture, {"schema_version": 1, "status": status,
                   "profiles": {profile: {"compatible": True, "dimensions": {}}}})
    return fixture


def _problem(path: Path, fixture: Path, proposer: list[str], gates: dict[str, list[str]], profile: str = "p") -> Path:
    problem = {
        "fixture": str(fixture), "shape": [1, 3, 8, 8], "dtype": "float32",
        "profile": profile, "reference": "reference", "metric": "latency",
        "proposer": proposer, "gates": gates,
    }
    path.write_text(json.dumps(problem))
    return path


def _commands(*, metrics: bool = True, failing_gate: str | None = None,
              proposal_edit: str = "tracked"):
    proposal_code = {
        "tracked": "from pathlib import Path; p=Path('README'); p.write_text(p.read_text() + 'candidate\\n')",
        "untracked": "from pathlib import Path; Path('candidate.txt').write_text('candidate\\n')",
        "noop": "pass",
    }[proposal_edit]
    proposer = [sys.executable, "-c", proposal_code]
    if metrics:
        metric_code = (
            "import json, os; "
            "x={'interleaved': True, 'baseline': [{'wall_ms': 12.0, 'gpu_ms': 10.0}]*20, "
            "'candidate': [{'wall_ms': 8.0, 'gpu_ms': 7.0}]*20}; "
            "json.dump(x, open(os.path.join(os.environ['VULKAN_RESEARCH_OUTPUT'], 'metrics.json'), 'w'))"
        )
    else:
        metric_code = "raise SystemExit(9)"
    failed_gate_code = (
        "import sys; print('gate stdout'); print('gate stderr', file=sys.stderr); "
        "raise SystemExit(9)"
    )
    gates = {
        gate: [sys.executable, "-c", failed_gate_code if gate == failing_gate else metric_code]
        for gate in ("local", "kernel", "model", "regression")
    }
    return proposer, gates


@pytest.fixture()
def isolated_research(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    repo = _repo(tmp_path)
    monkeypatch.setattr(research_module, "ROOT", repo)
    # Research tests use a private context instead of the host-wide production
    # lock, so concurrent agents cannot make these tests flaky.
    monkeypatch.setattr(research_module, "gpu_lock", lambda timeout: nullcontext())
    return repo


def test_success_runs_all_four_gates_and_keeps_worktree(isolated_research: Path, tmp_path: Path):
    proposer, gates = _commands()
    problem = _problem(tmp_path / "problem.json", _fixture(isolated_research), proposer, gates)
    result = research_module.research(problem, tmp_path / "run", attempts=1, budget_seconds=30, timeout=5)

    row = result["attempts"][0]
    assert row["status"] == "proposed_for_review"
    assert row["gates"] == {name: "passed" for name in ("local", "kernel", "model", "regression")}
    assert "candidate" in row["diff"]
    assert (tmp_path / "run" / "attempt-0").is_dir()
    assert json.loads((tmp_path / "run" / "journal.json").read_text())["attempts"][0]["status"] == row["status"]


def test_failed_gate_is_journaled(isolated_research: Path, tmp_path: Path):
    proposer, gates = _commands(failing_gate="kernel")
    problem = _problem(tmp_path / "problem.json", _fixture(isolated_research), proposer, gates)
    result = research_module.research(problem, tmp_path / "run", attempts=1, budget_seconds=30, timeout=5)

    row = result["attempts"][0]
    assert row["status"] == "failed"
    assert row["failed_gate"] == "kernel"
    assert row["gates"]["local"] == "passed"
    assert row["gates"]["kernel"] == "failed"
    assert "model" not in row["gates"]
    kernel_log = (tmp_path / "run" / "result-0" / "kernel.log").read_text()
    assert "gate stdout" in kernel_log
    assert "gate stderr" in kernel_log
    assert "candidate" in row["diff"]


def test_failed_proposer_keeps_output_and_patch(isolated_research: Path, tmp_path: Path):
    proposer = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; "
        "p=Path('README'); p.write_text(p.read_text() + 'candidate\\n'); "
        "print('proposer stdout'); print('proposer stderr', file=sys.stderr); "
        "raise SystemExit(9)",
    ]
    _, gates = _commands()
    problem = _problem(tmp_path / "problem.json", _fixture(isolated_research), proposer, gates)
    result = research_module.research(problem, tmp_path / "run", attempts=1, budget_seconds=30, timeout=5)

    row = result["attempts"][0]
    assert row["status"] == "failed"
    proposer_log = (tmp_path / "run" / "result-0" / "proposer.log").read_text()
    assert "proposer stdout" in proposer_log
    assert "proposer stderr" in proposer_log
    assert "candidate" in row["diff"]


def test_noop_proposal_is_rejected_for_review(isolated_research: Path, tmp_path: Path):
    proposer, gates = _commands(proposal_edit="noop")
    problem = _problem(tmp_path / "problem.json", _fixture(isolated_research), proposer, gates)
    result = research_module.research(problem, tmp_path / "run", attempts=1, budget_seconds=30, timeout=5)

    row = result["attempts"][0]
    assert row["status"] == "failed"
    assert "no tracked source diff" in row["error"]
    assert row["diff"] == ""
    assert row["untracked"] == []


def test_untracked_only_proposal_is_rejected_with_explicit_reason(isolated_research: Path, tmp_path: Path):
    proposer, gates = _commands(proposal_edit="untracked")
    problem = _problem(tmp_path / "problem.json", _fixture(isolated_research), proposer, gates)
    result = research_module.research(problem, tmp_path / "run", attempts=1, budget_seconds=30, timeout=5)

    row = result["attempts"][0]
    assert row["status"] == "failed"
    assert "only untracked files" in row["error"]
    assert row["diff"] == ""
    assert "candidate.txt" in row["untracked"]


def test_budget_timeout_is_recorded(isolated_research: Path, tmp_path: Path):
    proposer = [sys.executable, "-c", "import time; time.sleep(10)"]
    _, gates = _commands()
    problem = _problem(tmp_path / "problem.json", _fixture(isolated_research), proposer, gates)
    result = research_module.research(problem, tmp_path / "run", attempts=3, budget_seconds=0.05, timeout=5)

    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["status"] == "failed"


def test_budget_timeout_keeps_tracked_patch(isolated_research: Path, tmp_path: Path):
    proposer = [
        sys.executable,
        "-c",
        "from pathlib import Path; import time; "
        "p=Path('README'); p.write_text(p.read_text() + 'candidate\\n'); time.sleep(10)",
    ]
    _, gates = _commands()
    problem = _problem(tmp_path / "problem.json", _fixture(isolated_research), proposer, gates)
    result = research_module.research(problem, tmp_path / "run", attempts=1, budget_seconds=0.05, timeout=5)

    row = result["attempts"][0]
    assert row["status"] == "failed"
    assert "candidate" in row["diff"]


def test_interrupted_run_resumes_with_remaining_total_budget(isolated_research: Path, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch):
    proposer, gates = _commands()
    problem = _problem(tmp_path / "problem.json", _fixture(isolated_research), proposer, gates)
    original_command = research_module.command
    interrupted = {"value": True}

    def interrupt_once(argv, **kwargs):
        if interrupted["value"] and list(argv) == proposer:
            interrupted["value"] = False
            raise KeyboardInterrupt()
        return original_command(argv, **kwargs)

    monkeypatch.setattr(research_module, "command", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        research_module.research(problem, tmp_path / "run", attempts=2, budget_seconds=30, timeout=5)

    resumed = research_module.research(problem, tmp_path / "run", attempts=5, budget_seconds=300, timeout=5)
    assert len(resumed["attempts"]) == 2
    assert resumed["attempts"][0]["status"] == "interrupted"
    assert resumed["attempts"][1]["status"] == "proposed_for_review"
    assert (tmp_path / "run" / "attempt-0").is_dir()
    assert (tmp_path / "run" / "attempt-1").is_dir()


def test_stale_resume_and_unverified_fixture_are_rejected(isolated_research: Path, tmp_path: Path):
    proposer, gates = _commands()
    fixture = _fixture(isolated_research)
    problem = _problem(tmp_path / "problem.json", fixture, proposer, gates)
    research_module.research(problem, tmp_path / "run", attempts=1, budget_seconds=30, timeout=5)

    changed = json.loads(problem.read_text())
    changed["metric"] = "different"
    problem.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Stale research resume"):
        research_module.research(problem, tmp_path / "run", attempts=1, budget_seconds=30, timeout=5)

    bad_fixture = _fixture(isolated_research, status="prepared", name="bad-fixture")
    bad_problem = _problem(tmp_path / "bad-problem.json", bad_fixture, proposer, gates)
    with pytest.raises(ValueError, match="verified status"):
        research_module.research(bad_problem, tmp_path / "bad-run", attempts=1, budget_seconds=30, timeout=5)
