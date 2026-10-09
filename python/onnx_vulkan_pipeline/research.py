"""Bounded EP experiments in disposable worktrees, never automatic integration."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from .pipeline import ROOT, _improvement, digest, verify_bundle, write_json
from .process import command, gpu_lock


_REQUIRED_FIELDS = (
    "fixture", "shape", "dtype", "profile", "reference", "metric", "proposer", "gates",
)
_MANDATORY_GATES = ("local", "kernel", "model", "regression")


def _contract_digest(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_command(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a nonempty explicit argument array")
    return list(value)


def _load_contract(problem: Path) -> tuple[dict[str, Any], Path, dict[str, Any], str, str]:
    try:
        raw = json.loads(problem.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read experiment problem: {problem}") from error
    if not isinstance(raw, dict):
        raise ValueError("Experiment problem must be a JSON object")
    for field in _REQUIRED_FIELDS:
        if field not in raw:
            raise ValueError(f"Missing experiment field: {field}")
    gates = raw["gates"]
    if not isinstance(gates, dict):
        raise ValueError("Experiment gates must be an object of argument arrays")
    for gate in _MANDATORY_GATES:
        if gate not in gates:
            raise ValueError(f"Missing mandatory gate: {gate}")
    proposer = _validate_command(raw["proposer"], "proposer")
    validated_gates = {
        gate: _validate_command(gates[gate], f"gate {gate!r}")
        for gate in _MANDATORY_GATES
    }
    profile = raw["profile"]
    if not isinstance(profile, str) or not profile:
        raise ValueError("Experiment profile must be a nonempty profile name")
    base_request = raw.get("base_commit", "HEAD")
    if not isinstance(base_request, str) or not base_request:
        raise ValueError("base_commit must be a nonempty git revision")
    fixture = Path(raw["fixture"])
    if not fixture.is_absolute():
        fixture = problem.parent / fixture
    fixture = fixture.resolve()
    manifest = verify_bundle(fixture)
    if manifest.get("status") != "verified":
        raise ValueError("Research fixture must have verified status")
    profiles = manifest.get("profiles")
    if not isinstance(profiles, dict) or profile not in profiles:
        raise ValueError(f"Research fixture has no selected profile: {profile}")
    base = command(["git", "rev-parse", base_request], cwd=ROOT).stdout.strip()
    if not base:
        raise ValueError(f"Could not resolve base_commit: {base_request}")
    contract = copy.deepcopy(raw)
    contract["fixture"] = str(fixture)
    contract["base_commit"] = base
    contract["proposer"] = proposer
    contract["gates"] = validated_gates
    fixture_hash = digest(fixture / "manifest.json")
    return contract, fixture, manifest, base, fixture_hash


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read research {label}: {path}") from error


def _new_or_resume(problem: Path, destination: Path, contract: dict[str, Any], manifest: dict[str, Any],
                   base: str, fixture_hash: str, attempts: int, budget_seconds: float
                   ) -> tuple[list[dict[str, Any]], dict[str, Any], int, float, str]:
    contract_hash = _contract_digest(contract)
    problem_path = destination / "problem.json"
    state_path = destination / "state.json"
    journal_path = destination / "journal.json"
    if not destination.exists():
        destination.mkdir(parents=True)
        write_json(problem_path, contract)
        problem_hash = digest(problem_path)
        state = {
            "schema_version": 1,
            "contract": contract,
            "contract_digest": contract_hash,
            "problem_digest": problem_hash,
            "fixture_hash": fixture_hash,
            "fixture_identity": manifest.get("identity"),
            "base_commit": base,
            "started_at": time.time(),
            "budget_seconds": budget_seconds,
            "attempt_budget": attempts,
        }
        write_json(state_path, state)
        rows: list[dict[str, Any]] = []
        write_json(journal_path, {"schema_version": 1, "contract_digest": contract_hash,
                                  "problem_digest": problem_hash, "attempts": rows})
        return rows, state, attempts, budget_seconds, contract_hash

    if not destination.is_dir():
        raise ValueError(f"Research destination is not a directory: {destination}")
    state = _read_json(state_path, "state")
    journal = _read_json(journal_path, "journal")
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("Unsupported or stale research state")
    if not isinstance(journal, dict) or journal.get("schema_version") != 1:
        raise ValueError("Unsupported or stale research journal")
    if state.get("contract_digest") != contract_hash or state.get("contract") != contract:
        raise ValueError("Stale research resume: problem contract changed")
    if state.get("base_commit") != base or state.get("fixture_hash") != fixture_hash:
        raise ValueError("Stale research resume: base commit or fixture changed")
    if state.get("fixture_identity") != manifest.get("identity"):
        raise ValueError("Stale research resume: fixture identity changed")
    if journal.get("contract_digest") != contract_hash or journal.get("problem_digest") != state.get("problem_digest"):
        raise ValueError("Stale research resume: journal contract changed")
    if not problem_path.is_file() or digest(problem_path) != state.get("problem_digest"):
        raise ValueError("Stale research resume: problem record changed")
    rows = journal.get("attempts")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Malformed research journal")
    indexes = [row.get("attempt") for row in rows]
    if indexes != list(range(len(rows))):
        raise ValueError("Malformed research journal attempt sequence")
    stored_attempts = state.get("attempt_budget")
    stored_budget = state.get("budget_seconds")
    if not isinstance(stored_attempts, int) or stored_attempts < 1 or not isinstance(stored_budget, (int, float)):
        raise ValueError("Malformed research budget state")
    # Resumes never extend the original total attempt/time budget.
    return rows, state, min(attempts, stored_attempts), float(stored_budget), contract_hash


def _write_journal(destination: Path, contract_hash: str, problem_hash: str, rows: list[dict[str, Any]]) -> None:
    write_json(destination / "journal.json", {"schema_version": 1, "contract_digest": contract_hash,
                                               "problem_digest": problem_hash, "attempts": rows})


def _write_command_log(path: Path, *, completed=None, error: BaseException | None = None) -> None:
    """Persist command output even when ``command`` raises before returning."""
    if completed is not None:
        text = f"{completed.stdout}{completed.stderr}"
    else:
        stdout = getattr(error, "stdout", None) if error is not None else None
        stderr = getattr(error, "stderr", None) if error is not None else None
        if stdout is not None or stderr is not None:
            text = f"{stdout or ''}{stderr or ''}"
            if not text and error is not None:
                text = str(error)
        else:
            text = str(error) if error is not None else ""
    path.write_text(text)


def _capture_patch(base: str, worktree: Path, result: dict[str, Any]) -> None:
    """Record the tracked patch and untracked paths, including after failures."""
    previous_diff = result.get("diff", "")
    try:
        # Cleanup reads must remain possible after the experiment budget expires.
        result["diff"] = command(
            ["git", "diff", "--binary", base], cwd=worktree, timeout=10
        ).stdout
    except BaseException as error:
        result["diff"] = previous_diff
        result["patch_error"] = f"Could not capture git diff: {error}"
    previous_untracked = result.get("untracked", [])
    try:
        result["untracked"] = command(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=worktree, timeout=10
        ).stdout.splitlines()
    except BaseException as error:
        result["untracked"] = previous_untracked
        result["patch_error"] = f"Could not capture untracked files: {error}"


def _promotion_refusal(result: dict[str, Any]) -> str | None:
    """Return a reason when metrics cannot be backed by a reviewable patch."""
    if result.get("diff"):
        return None
    if result.get("untracked"):
        return (
            "Review promotion blocked: only untracked files are present; their contents "
            "are not included in the review patch"
        )
    return "Review promotion blocked: candidate has no tracked source diff"


def research(problem, destination, *, attempts=3, budget_seconds=1800, timeout=300):
    """Run a bounded proposer and four mandatory gates in disposable worktrees."""
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    if (isinstance(budget_seconds, bool) or not isinstance(budget_seconds, (int, float))
            or not math.isfinite(float(budget_seconds)) or budget_seconds <= 0):
        raise ValueError("budget_seconds must be a finite positive number")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout)) or timeout <= 0):
        raise ValueError("timeout must be a finite positive number")
    problem = Path(problem).resolve()
    destination = Path(destination).resolve()
    contract, fixture, manifest, base, fixture_hash = _load_contract(problem)
    rows, state, total_attempts, total_budget, contract_hash = _new_or_resume(
        problem, destination, contract, manifest, base, fixture_hash, attempts, float(budget_seconds))
    problem_hash = state["problem_digest"]
    resumed_running = False
    for row in rows:
        if row.get("status") == "running":
            row["status"] = "interrupted"
            row["error"] = "resumed after an incomplete attempt"
            resumed_running = True
    if resumed_running:
        _write_journal(destination, contract_hash, problem_hash, rows)
    started_at = state.get("started_at")
    if not isinstance(started_at, (int, float)) or not math.isfinite(float(started_at)):
        raise ValueError("Malformed research start time")
    deadline = float(started_at) + total_budget

    for index in range(len(rows), total_attempts):
        if time.time() >= deadline:
            break
        branch = f"autoresearch/{uuid.uuid4().hex[:12]}-{index}"
        worktree = destination / f"attempt-{index}"
        result: dict[str, Any] = {
            "attempt": index, "branch": branch, "base_commit": base,
            "fixture_hash": fixture_hash, "contract_digest": contract_hash,
            "problem_digest": problem_hash, "status": "running", "gates": {},
            "diff": "", "untracked": [],
        }
        rows.append(result)
        _write_journal(destination, contract_hash, problem_hash, rows)

        def invoke(argv, **kwargs):
            remaining = min(float(timeout), deadline - time.time())
            if remaining <= 0:
                raise TimeoutError("Experiment time budget exhausted")
            return command(argv, timeout=remaining, **kwargs)

        output = destination / f"result-{index}"
        try:
            invoke(["git", "worktree", "add", "-b", branch, worktree, base], cwd=ROOT)
            output.mkdir()
            env = {"VULKAN_RESEARCH_PROBLEM": str(destination / "problem.json"),
                   "VULKAN_RESEARCH_OUTPUT": str(output), "VULKAN_RESEARCH_ATTEMPT": str(index),
                   "VULKAN_RESEARCH_FIXTURE": str(fixture)}
            try:
                proposal = invoke(contract["proposer"], cwd=worktree, env=env)
            except BaseException as error:
                _write_command_log(output / "proposer.log", error=error)
                raise
            _write_command_log(output / "proposer.log", completed=proposal)
            lock_timeout = min(float(timeout), max(0.01, deadline - time.time()))
            with gpu_lock(lock_timeout):
                for gate in _MANDATORY_GATES:
                    result["gates"][gate] = "running"
                    _write_journal(destination, contract_hash, problem_hash, rows)
                    try:
                        checked = invoke(contract["gates"][gate], cwd=worktree, env=env)
                    except BaseException as error:
                        _write_command_log(output / f"{gate}.log", error=error)
                        result["gates"][gate] = "failed"
                        result["failed_gate"] = gate
                        result["error"] = str(error)
                        _write_journal(destination, contract_hash, problem_hash, rows)
                        raise
                    _write_command_log(output / f"{gate}.log", completed=checked)
                    result["gates"][gate] = "passed"
                    _write_journal(destination, contract_hash, problem_hash, rows)
            metrics = _read_json(output / "metrics.json", "metrics")
            if not isinstance(metrics, dict) or metrics.get("interleaved") is not True:
                raise ValueError("Metrics must attest interleaved baseline/candidate sampling")
            baseline, candidate = metrics.get("baseline"), metrics.get("candidate")
            if (not isinstance(baseline, list) or not isinstance(candidate, list)
                    or len(baseline) < 20 or len(candidate) < 20):
                raise ValueError("Metrics require at least 20 baseline and candidate samples")
            improved = _improvement(baseline, candidate)
            result["metrics"] = metrics
            _capture_patch(base, worktree, result)
            if improved:
                refusal = _promotion_refusal(result)
                if refusal is None:
                    result["status"] = "proposed_for_review"
                else:
                    result["status"] = "failed"
                    result["error"] = refusal
            else:
                result["status"] = "no_measured_gain"
        except BaseException as error:
            result["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            result["error"] = str(error)
            if not isinstance(error, Exception):
                raise
        finally:
            if worktree.is_dir():
                _capture_patch(base, worktree, result)
            _write_journal(destination, contract_hash, problem_hash, rows)
    return {"schema_version": 1, "base_commit": base, "fixture_hash": fixture_hash,
            "contract_digest": contract_hash, "attempts": rows, "automatic_integration": False}


__all__ = ["research"]
