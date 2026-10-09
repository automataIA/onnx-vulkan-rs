"""Check the bounded proposer preserves explicit tactic selection and sources."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "research_gemm16", Path(__file__).parents[1] / "scripts/research/gemm16.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_proposer_changes_only_default_for_exact_shape(tmp_path, monkeypatch):
    deps = tmp_path / "dependencies"
    for name in ("models", "third_party"):
        (deps / name).mkdir(parents=True)
    candidate = tmp_path / "candidate"
    target = candidate / driver.INTERP
    target.parent.mkdir(parents=True)
    target.write_text('before\n        let (label, source, tile) = if selected.as_deref() == Some("gemm16") {\nafter\n')
    output = tmp_path / "result"
    output.mkdir()
    (output / "propose.log").write_text("supervisor output")
    monkeypatch.chdir(candidate)
    problem = {"dependencies_root": str(deps), "geometry": {"m": 1, "k": 1280, "n": 1000}}
    driver._propose(problem, tmp_path, output)
    assert (output / "propose.log").read_text() == "supervisor output"
    assert (output / "driver-propose.log").is_file()
    patched = target.read_text()
    assert 'selected.as_deref() == Some("gemm16")' in patched
    assert 'selected.is_none() && m == 1 && ka == 1280 && n == 1000' in patched
    assert patched.startswith('before\n') and patched.endswith('after\n')
    assert (candidate / "models").resolve() == deps / "models"
    assert (candidate / "third_party").resolve() == deps / "third_party"
    driver._propose(problem, tmp_path, output)
    assert target.read_text() == patched


def test_proposer_refuses_unknown_selection_site(tmp_path, monkeypatch):
    for name in ("models", "third_party"):
        (tmp_path / "dependencies" / name).mkdir(parents=True)
    target = tmp_path / "candidate" / driver.INTERP
    target.parent.mkdir(parents=True)
    target.write_text("unrecognized source\n")
    monkeypatch.chdir(tmp_path / "candidate")
    with pytest.raises(ValueError, match="unique Gemm"):
        driver._propose({"dependencies_root": str(tmp_path / "dependencies"),
                         "geometry": {"m": 1, "k": 1280, "n": 1000}}, tmp_path, tmp_path)
    assert target.read_text() == "unrecognized source\n"
