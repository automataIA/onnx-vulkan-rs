"""Precision-independent quality contract: CPU-only, recorded outputs stand in for Vulkan."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from onnx_vulkan_pipeline import _quality_worker, pipeline, quality  # noqa: E402
from onnx_vulkan_pipeline.quantize import quantize_model  # noqa: E402

SHAPE = [1, 1, 3, 3]


def _model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, SHAPE)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, SHAPE)
    weights = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), "weights")
    bias = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), "bias")
    graph = helper.make_graph([helper.make_node("Conv", ["x", "weights", "bias"], ["conv"]),
                               helper.make_node("Relu", ["conv"], ["y"])], "tiny", [x], [y], [weights, bias])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


ADAPTER = (
    "import numpy as np\n"
    "def evaluate(session, samples):\n"
    "    return float(np.mean([session.run(None, sample)[0] for sample in samples]))\n"
)


@pytest.fixture()
def adapter(tmp_path: Path) -> Path:
    path = tmp_path / "adapter.py"
    path.write_text(ADAPTER)
    return path


@pytest.fixture()
def evaluation(tmp_path: Path) -> Path:
    root = tmp_path / "evaluation"
    root.mkdir()
    for index, level in enumerate((0.25, 0.5, 0.75)):
        np.savez(root / f"{index:03d}.npz", x=np.full(SHAPE, level, dtype=np.float32))
    return root


@pytest.fixture()
def profile(tmp_path: Path) -> Path:
    """A prepared fp32 profile: model.onnx is a conservative rewrite of original.onnx."""
    directory = tmp_path / "bundle" / "default"
    directory.mkdir(parents=True)
    _model(directory / "model.onnx")
    shutil.copyfile(directory / "model.onnx", directory / "original.onnx")
    (directory / "preparation.json").write_text("{}")
    return directory


def _config(evaluation_dir: Path, adapter_file: Path, **extra):
    config = {"dataset": {"id": "tiny-set", "split": "heldout", "seed": 7}, "evaluation": evaluation_dir,
              "adapter": adapter_file, "metric": "mean_output", "higher_is_better": True, "max_drop": 0.05,
              "preprocessing": "identity-v1", "postprocessing": "mean-v1"}
    return {**config, **extra}


def _manifest(model: str = "default/model.onnx") -> dict:
    return {"profiles": {"default": {"model": model, "tactics": None, "dimensions": {}}}}


def _fixture_dirs(directory: Path) -> list[Path]:
    return sorted(path for path in (directory / "quality").glob("*") if path.is_dir())


def _output(reference: Path, shift: float = 0.0) -> dict:
    tensor = onnx.load_tensor(str(reference / "output_0.pb"))
    values = numpy_helper.to_array(tensor) + np.float32(shift)
    return {"name": tensor.name, "dtype": "float32", "shape": list(values.shape),
            "values": values.reshape(-1).tolist()}


def _records(references, *, shift: float = 0.0, backend: str = "standalone") -> list[dict]:
    return [{"reference": str(reference),
             "result": {"backend": backend, "outputs": [_output(reference, shift)]}} for reference in references]


def test_fp32_contract_is_attached_and_evaluated_on_measured_outputs(profile, adapter, evaluation, monkeypatch):
    summary = quality.attach_quality(profile, _config(evaluation, adapter))

    document = json.loads((profile / "quality.json").read_text())
    assert document["schema_version"] == 1
    assert summary["contract_digest"] == document["contract_digest"]
    assert document["reference"]["score"] == pytest.approx(0.5)
    assert document["reference"]["model"]["sha256"] == pipeline.digest(profile / "original.onnx")
    assert document["heldout"]["status"] == "no-calibration"
    assert document["spec"]["dataset"] == {
        "id": "tiny-set", "split": "heldout", "seed": 7, "sample_count": 3,
        "files": document["spec"]["dataset"]["files"]}
    assert [item["sha256"] for item in document["spec"]["dataset"]["files"]] == [
        pipeline.digest(path) for path in sorted(evaluation.glob("*.npz"))]
    assert document["spec"]["adapter"]["sha256"] == pipeline.digest(adapter)
    assert len(_fixture_dirs(profile)) == 3

    seen = []

    def run(bundle, manifest, name, backend, samples, timeout, *, tactics, reference, output_data):
        seen.append(reference)
        assert output_data
        return {"output_data": {"backend": "standalone", "outputs": [_output(reference)]}}

    monkeypatch.setattr(pipeline, "_run", run)
    report = quality.verify_quality(profile.parent, _manifest(), "default", "standalone", 120)

    assert report["measured"] and report["vulkan_verified"] and report["samples"] == 3
    assert report["backend"] == "standalone" and report["requested_backend"] == "standalone"
    assert report["reference_source"] == "fixtures" and seen == _fixture_dirs(profile)
    assert report["before"] == pytest.approx(report["after"]) and report["drop"] == pytest.approx(0)
    assert report["contract"]["digest"] == document["contract_digest"]
    assert report["contract"]["dataset"] == {"id": "tiny-set", "split": "heldout", "seed": 7}
    assert report["selected_model"] == document["fixtures"]["model"]["digest"]


@pytest.mark.parametrize("higher_is_better, shift, passes", [
    (True, -0.1, False), (True, 0.1, True), (False, 0.1, False), (False, -0.1, True)])
def test_metric_direction_decides_what_a_drop_is(profile, adapter, evaluation, higher_is_better, shift, passes):
    quality.attach_quality(profile, _config(evaluation, adapter, higher_is_better=higher_is_better))
    records = _records(_fixture_dirs(profile), shift=shift)
    if passes:
        assert _quality_worker.evaluate(profile, records)["measured"]
    else:
        with pytest.raises(ValueError, match="application quality regression"):
            _quality_worker.evaluate(profile, records)


@pytest.mark.parametrize("change, message", [
    ({"max_drop": float("nan")}, "max_drop"), ({"max_drop": -0.1}, "max_drop"),
    ({"max_drop": True}, "max_drop"), ({"max_drop": "0.1"}, "max_drop"),
    ({"higher_is_better": "yes"}, "higher_is_better"), ({"metric": " "}, "metric"),
    ({"dataset": {"id": "a", "split": "b"}}, "exactly id, split and seed"),
    ({"dataset": {"id": "a", "split": "b", "seed": True}}, "seed"),
    ({"dataset": {"id": "", "split": "b", "seed": 1}}, "dataset id"),
    ({"preprocessing": ""}, "preprocessing"), ({"postprocessing": None}, "postprocessing"),
    ({"schema_version": 2}, "schema_version"), ({"surprise": 1}, "Unknown quality config keys"),
    ({"evaluation": "/nonexistent"}, "directory"), ({"adapter": "/nonexistent.py"}, ".py file"),
    ({"adapter": 5}, "adapter must be a path"),
])
def test_invalid_specs_are_rejected_before_any_work(profile, adapter, evaluation, change, message):
    with pytest.raises(ValueError, match=message):
        quality.validate_config(_config(evaluation, adapter, **change))
    with pytest.raises(ValueError, match=message):
        quality.attach_quality(profile, _config(evaluation, adapter, **change))
    assert not (profile / "quality.json").exists() and not (profile / "quality").exists()


def test_missing_keys_empty_data_and_bad_adapters_are_rejected(profile, adapter, evaluation, tmp_path):
    config = _config(evaluation, adapter)
    for key in config:
        with pytest.raises(ValueError, match=key):
            quality.validate_config({k: v for k, v in config.items() if k != key})
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="NPZ"):
        quality.validate_config({**config, "evaluation": empty})
    np.savez(evaluation / "003.npz", other=np.zeros(SHAPE, dtype=np.float32))
    with pytest.raises(ValueError, match="different input names"):
        quality.validate_config(config)
    (evaluation / "003.npz").unlink()
    no_evaluate = tmp_path / "no_evaluate.py"
    no_evaluate.write_text("def score(session, samples):\n    return 1.0\n")
    with pytest.raises(ValueError, match="define evaluate"):
        quality.validate_config({**config, "adapter": no_evaluate})
    broken = tmp_path / "broken.py"
    broken.write_text("def evaluate(:\n")
    with pytest.raises(ValueError, match="not valid Python"):
        quality.validate_config({**config, "adapter": broken})


def test_unknown_input_names_and_nonfinite_metrics_leave_nothing_behind(profile, adapter, evaluation, tmp_path):
    nan = tmp_path / "nan.py"
    nan.write_text("def evaluate(session, samples):\n    return float('nan')\n")
    with pytest.raises(ValueError, match="non-finite"):
        quality.attach_quality(profile, _config(evaluation, nan))
    np.savez(evaluation / "003.npz", labels=np.zeros(SHAPE, dtype=np.float32))
    for path in sorted(evaluation.glob("00[012].npz")):
        path.unlink()
    with pytest.raises(ValueError, match="input names differ"):
        quality.attach_quality(profile, _config(evaluation, adapter))
    assert not (profile / "quality.json").exists() and not (profile / "quality").exists()
    assert not (profile / "quality_adapter.py").exists()


def test_cpu_regression_of_the_candidate_is_rejected(profile, adapter, evaluation):
    # A candidate that scales its output by half fails the metric even though no tensor tolerance is involved.
    model = onnx.load(profile / "model.onnx")
    model.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.full((1, 1, 1, 1), 0.5, np.float32), "weights"))
    onnx.save(model, profile / "model.onnx")
    with pytest.raises(ValueError, match="Quality regression"):
        quality.attach_quality(profile, _config(evaluation, adapter))
    assert not (profile / "quality").exists()


def test_contract_binds_dataset_adapter_and_fixture_content(profile, adapter, evaluation):
    quality.attach_quality(profile, _config(evaluation, adapter))
    contract = quality.load_contract(profile)
    quality.check_fixtures(profile, contract)

    # Samples that are not the contract's are not "the evaluation".
    other = _fixture_dirs(profile)[0]
    with pytest.raises(ValueError, match="differ from the evaluation samples"):
        _quality_worker.evaluate(profile, _records([other]))

    tensor = _fixture_dirs(profile)[1] / "input_0.pb"
    original = tensor.read_bytes()
    tensor.write_bytes(numpy_helper.from_array(np.zeros(SHAPE, np.float32), "x").SerializeToString())
    with pytest.raises(ValueError, match="missing or modified"):
        quality.check_fixtures(profile, contract)
    tensor.write_bytes(original)
    quality.check_fixtures(profile, contract)

    (profile / "quality" / "9999").mkdir()
    with pytest.raises(ValueError, match="does not record"):
        quality.check_fixtures(profile, contract)
    (profile / "quality" / "9999").rmdir()

    (profile / "quality_adapter.py").write_text(ADAPTER + "# edited\n")
    with pytest.raises(ValueError, match="differs from the contract"):
        quality.load_contract(profile)


def test_edited_manifest_is_detected(profile, adapter, evaluation):
    quality.attach_quality(profile, _config(evaluation, adapter))
    document = json.loads((profile / "quality.json").read_text())
    document["reference"]["score"] = 0.9
    (profile / "quality.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="digest mismatch"):
        quality.load_contract(profile)


def test_existing_contracts_and_fixtures_are_never_overwritten(profile, adapter, evaluation):
    quality.attach_quality(profile, _config(evaluation, adapter))
    before = {p: p.read_bytes() for p in profile.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        quality.attach_quality(profile, _config(evaluation, adapter))
    (profile / "quality.json").unlink()
    with pytest.raises(FileExistsError, match="fixture directory"):
        quality.attach_quality(profile, _config(evaluation, adapter))
    assert {p: p.read_bytes() for p in profile.rglob("*") if p.is_file() and p.name != "quality.json"} == {
        p: b for p, b in before.items() if p.name != "quality.json"}


def test_no_contract_or_no_fixtures_is_unmeasured_not_an_error(profile, monkeypatch):
    monkeypatch.setattr(pipeline, "_run", lambda *args, **kwargs: pytest.fail("nothing to measure"))
    report = quality.verify_quality(profile.parent, _manifest(), "default", "ep", 120)
    assert report == {"samples": 0, "measured": False, "vulkan_verified": False, "backend": None,
                      "requested_backend": "ep", "contract": None,
                      "reason": "no quality contract for this profile"}
    # Legacy metadata without recorded fixtures is the same: no data, no claim.
    (profile / "preparation.json").write_text(json.dumps({"quantization": {"mode": "static"}}))
    assert not quality.verify_quality(profile.parent, _manifest(), "default", "ep", 120)["measured"]


def test_recorded_session_never_infers_and_rejects_foreign_records(profile, adapter, evaluation, monkeypatch):
    quality.attach_quality(profile, _config(evaluation, adapter))
    references = _fixture_dirs(profile)

    def forbidden(*args, **kwargs):
        raise AssertionError("the metric must not run a CPU session")

    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    assert _quality_worker.evaluate(profile, _records(references))["vulkan_verified"]
    session = _quality_worker.RecordedSession(_records(references))
    with pytest.raises(ValueError, match="without measured Vulkan outputs"):
        session.run(None, {"x": np.full(SHAPE, 9, dtype=np.float32)})
    with pytest.raises(ValueError, match="unmeasured outputs"):
        session.run(["nope"], session.samples[0])
    for backend in ("ort-cpu", "cpu", "onnxruntime_cpu"):
        with pytest.raises(ValueError, match="requires Vulkan outputs"):
            _quality_worker.evaluate(profile, _records(references, backend=backend))
    mixed = _records(references)
    mixed[1]["result"]["backend"] = "vulkan-ep"
    with pytest.raises(ValueError, match="mix backends"):
        _quality_worker.evaluate(profile, mixed)


def test_incomplete_conflicting_or_nonfinite_measurements_are_rejected(profile, adapter, evaluation, tmp_path):
    quality.attach_quality(profile, _config(evaluation, adapter))
    references = _fixture_dirs(profile)
    with pytest.raises(ValueError, match="differ from the evaluation samples"):
        _quality_worker.evaluate(profile, _records(references[:2]))
    with pytest.raises(ValueError, match="differ from the evaluation samples"):
        _quality_worker.evaluate(profile, _records([references[0], *references]))
    with pytest.raises(ValueError, match="no measured outputs"):
        records = _records(references)
        records[0]["result"]["outputs"] = []
        _quality_worker.evaluate(profile, records)
    records = _records(references)
    records[2]["result"]["outputs"][0]["values"][0] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        _quality_worker.evaluate(profile, records)
    # The same inputs measured twice with different outputs: one of them is wrong.
    conflicting = _records([references[0], references[0]])
    conflicting[1]["result"]["outputs"][0]["values"][0] += 1
    with pytest.raises(ValueError, match="conflicting measured outputs"):
        _quality_worker.RecordedSession(conflicting)
    # An adapter that scores a subset would make `samples` describe something it did not evaluate.
    lazy = tmp_path / "lazy.py"
    lazy.write_text("import numpy as np\n"
                    "def evaluate(session, samples):\n    return float(np.mean(session.run(None, samples[0])[0]))\n")
    shutil.copyfile(lazy, profile / "quality_adapter.py")
    document = json.loads((profile / "quality.json").read_text())
    document["spec"]["adapter"]["sha256"] = pipeline.digest(lazy)
    document["spec_digest"] = quality._canonical_digest(document["spec"])
    document["contract_digest"] = quality._canonical_digest({
        "spec": document["spec_digest"], "score": document["reference"]["score"],
        "reference": document["reference"]["model"]["digest"]})
    (profile / "quality.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="did not evaluate every measured sample"):
        _quality_worker.evaluate(profile, _records(references))


def _quantized_profile(tmp_path: Path, adapter: Path) -> tuple[Path, dict, Path]:
    directory = tmp_path / "bundle" / "default"
    directory.mkdir(parents=True)
    _model(directory / "model.onnx")
    calibration, evaluation = tmp_path / "calibration", tmp_path / "quant-evaluation"
    calibration.mkdir()
    evaluation.mkdir()
    np.savez(calibration / "000.npz", x=np.full(SHAPE, -1.0, dtype=np.float32))
    np.savez(calibration / "001.npz", x=np.full(SHAPE, 1.0, dtype=np.float32))
    np.savez(evaluation / "000.npz", x=np.full(SHAPE, 0.5, dtype=np.float32))
    result = quantize_model(directory / "model.onnx", {
        "calibration": calibration, "evaluation": evaluation, "adapter": adapter, "metric": "mean_output",
        "max_drop": 0.25, "higher_is_better": True, "format": "QDQ"}, atol=0.2, rtol=0.01)
    (directory / "preparation.json").write_text(json.dumps({"quantization": result}))
    return directory, result, evaluation


def test_legacy_quantization_contract_is_normalized_and_verified(tmp_path, adapter, monkeypatch):
    directory, result, _ = _quantized_profile(tmp_path, adapter)
    contract = quality.load_contract(directory)
    assert contract["source"] == "legacy-quantization" and contract["reference_score"] == result["before"]
    assert contract["heldout"]["status"] == "disjoint-from-calibration"
    seen = []

    def run(bundle, manifest, name, backend, samples, timeout, *, tactics, reference, output_data):
        seen.append(reference)
        return {"output_data": {"backend": "vulkan-ep", "outputs": [_output(reference)]}}

    monkeypatch.setattr(pipeline, "_run", run)
    report = quality.verify_quality(directory.parent, _manifest(), "default", "ep", 120)
    assert report["measured"] and report["backend"] == "vulkan-ep" and report["reference_source"] == "fixtures"
    assert report["contract"]["source"] == "legacy-quantization" and seen == _fixture_dirs(directory)
    assert report["metric"] == "mean_output" and report["samples"] == 1


def test_fp32_fallback_is_evaluated_against_cpu_outputs_of_the_selected_model(tmp_path, adapter, monkeypatch):
    directory, _, _ = _quantized_profile(tmp_path, adapter)
    fixtures = _fixture_dirs(directory)
    seen = []

    def run(bundle, manifest, name, backend, samples, timeout, *, tactics, reference, output_data):
        # The temporary reference vanishes after verify_quality: capture what it holds.
        seen.append((reference, {p.name: p.read_bytes() for p in reference.glob("*.pb")}))
        return {"output_data": {"backend": "standalone", "outputs": [_output(reference)]}}

    monkeypatch.setattr(pipeline, "_run", run)
    sealed = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = quality.verify_quality(directory.parent, _manifest("default/fp32.onnx"), "default", "standalone", 120)

    assert report["measured"] and report["reference_source"] == "regenerated-cpu"
    assert report["before"] == pytest.approx(report["after"])
    assert report["selected_model"] == quality.model_identity(directory / "fp32.onnx")["digest"]
    assert [reference for reference, _ in seen] and all(reference not in fixtures for reference, _ in seen)
    expected = ort.InferenceSession(str(directory / "fp32.onnx"), providers=["CPUExecutionProvider"]).run(
        None, {"x": np.full(SHAPE, 0.5, dtype=np.float32)})[0]
    regenerated = seen[0][1]
    np.testing.assert_array_equal(numpy_helper.to_array(onnx.TensorProto.FromString(regenerated["output_0.pb"])), expected)
    assert regenerated["input_0.pb"] == (fixtures[0] / "input_0.pb").read_bytes()
    # The sealed bundle was not touched, and no CPU outputs leaked into it.
    assert sealed == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_quality_config_reuses_compatible_quantization_data_and_rejects_conflicts(tmp_path, adapter):
    directory, result, evaluation = _quantized_profile(tmp_path, adapter)
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    config = _config(evaluation, adapter, max_drop=0.25)

    for change, message in [({"max_drop": 0.5}, "max_drop"), ({"higher_is_better": False}, "higher_is_better"),
                            ({"metric": "other"}, "metric")]:
        with pytest.raises(ValueError, match=message):
            quality.attach_quality(directory, {**config, **change})
    other_adapter = tmp_path / "other.py"
    other_adapter.write_text(ADAPTER + "# other\n")
    with pytest.raises(ValueError, match="adapter"):
        quality.attach_quality(directory, {**config, "adapter": other_adapter})
    other_data = tmp_path / "other-data"
    other_data.mkdir()
    np.savez(other_data / "000.npz", x=np.full(SHAPE, 0.6, dtype=np.float32))
    with pytest.raises(ValueError, match="differs from the quantization evaluation data"):
        quality.attach_quality(directory, {**config, "evaluation": other_data})
    assert not (directory / "quality.json").exists()

    summary = quality.attach_quality(directory, config)
    document = json.loads((directory / "quality.json").read_text())
    assert summary["heldout"] == "disjoint-from-calibration"
    assert document["reference"]["score"] == result["before"]
    assert document["reference"]["model"]["sha256"] == result["original_sha256"]
    assert document["fixtures"]["samples"] == result["quality_fixtures"]
    assert {p: p.read_bytes() for p in directory.rglob("*") if p.is_file() and p.name != "quality.json"} == before
    assert quality.load_contract(directory)["source"] == "quality.json"


def test_evaluation_samples_that_overlap_calibration_are_rejected(tmp_path, adapter):
    directory, result, evaluation = _quantized_profile(tmp_path, adapter)
    result["calibration_content_hashes"].append(result["evaluation_content_hashes"][0])
    (directory / "preparation.json").write_text(json.dumps({"quantization": result}))
    with pytest.raises(ValueError, match="overlap the calibration"):
        quality.attach_quality(directory, _config(evaluation, adapter, max_drop=0.25))
    assert not (directory / "quality.json").exists()


def test_prepare_worker_validates_quality_before_preparing_and_attaches_it(tmp_path, adapter, evaluation, monkeypatch):
    from onnx_vulkan_pipeline import _prepare_worker

    source = tmp_path / "source.onnx"
    _model(source)

    def run_worker(quality_config):
        config = {"source": str(source), "destination": str(tmp_path / "out"), "adapter": None,
                  "profiles": {}, "atol": 1e-5, "rtol": 1e-4, "quantization": None, "quality": quality_config}
        (tmp_path / "config.json").write_text(json.dumps(config))
        monkeypatch.setattr("sys.argv", ["worker", str(tmp_path / "config.json")])
        _prepare_worker.main()

    prepared = []
    monkeypatch.setattr(_prepare_worker, "prepare_model", lambda *a, **k: prepared.append(a) or pytest.fail("ran"))
    bad = {k: str(v) if isinstance(v, Path) else v for k, v in _config(evaluation, adapter, max_drop=-1).items()}
    with pytest.raises(ValueError, match="max_drop"):
        run_worker(bad)
    assert not prepared

    from onnx_vulkan_pipeline.prepare import prepare_model
    monkeypatch.setattr(_prepare_worker, "prepare_model", prepare_model)
    run_worker({k: str(v) if isinstance(v, Path) else v for k, v in _config(evaluation, adapter).items()})
    out = tmp_path / "out"
    preparation = json.loads((out / "preparation.json").read_text())
    assert preparation["quality"]["file"] == "quality.json" and (out / "quality.json").is_file()
    assert quality.load_contract(out)["reference_score"] == pytest.approx(0.5)
