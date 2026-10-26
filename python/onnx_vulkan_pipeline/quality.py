"""Precision-independent application quality contract (audit A4).

A bundle profile may carry a *quality contract*: a dataset, an adapter that
turns ``session.run`` outputs into one scalar metric, the metric's direction
and the largest accepted drop from the original model's score. The contract is
about the workload, not about quantization: it applies to fp32 conservative
transforms, tuning and quantized candidates alike, and it never widens a
tensor tolerance.

Input configuration (``prepare_bundle(..., quality=...)``), validated by
:func:`validate_config`::

    {"dataset": {"id": "coco-val-2017-subset", "split": "heldout", "seed": 7},
     "evaluation": "path/to/dir-of-named-input-npz-files",
     "adapter": "path/to/adapter.py",          # defines evaluate(session, samples) -> float
     "metric": "mAP", "higher_is_better": true, "max_drop": 0.01,
     "preprocessing": "letterbox-640-v2",      # identifiers; the code they name must
     "postprocessing": "nms-0.5-v1"}           # live in (or be hashed through) the adapter

Stored manifest ``<profile>/quality.json`` (schema version 1)::

    {"schema_version": 1,
     "spec": {"dataset": {"id", "split", "seed", "sample_count",
                          "files": [{"name", "sha256", "content_sha256"}, ...]},
              "adapter": {"file": "quality_adapter.py", "sha256"},
              "metric", "higher_is_better", "max_drop", "preprocessing", "postprocessing"},
     "reference": {"score", "backend": "ort-cpu", "model": <identity>},
     "candidate": {"cpu_score", "backend": "ort-cpu", "model": <identity>},
     "fixtures": {"backend": "ort-cpu", "model": <identity>, "samples": [...]},
     "heldout": {"status", "calibration_samples", "overlap"},
     "spec_digest", "contract_digest"}

``<identity>`` is ``{"file", "sha256", "external_data": [{"file", "sha256"}],
"digest"}``: the model proto and every external-data file it points at.
Fixture inputs are the evaluation samples; fixture outputs are the CPU outputs
of the fixture model, used only as ``model-runner`` references. The metric
itself is always computed from *measured Vulkan outputs* (see
``_quality_worker``): this module never substitutes CPU outputs for them.

Legacy quantization bundles (``preparation.json["quantization"]``) have no
``quality.json``; :func:`load_contract` normalizes them to the same shape.

Imports of ``pipeline``/``quantize`` are lazy: ``pipeline`` may import this
module, and ``quantize`` imports ``pipeline``.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

SCHEMA_VERSION = 1
QUALITY_FILE = "quality.json"
ADAPTER_FILE = "quality_adapter.py"
FIXTURE_DIR = "quality"
_CONFIG_KEYS = ("dataset", "evaluation", "adapter", "metric", "higher_is_better", "max_drop",
                "preprocessing", "postprocessing")
_SHA256 = frozenset("0123456789abcdef")


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return float(value)


def _path(value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{label} must be a path")
    return Path(value).resolve()


def _defines_evaluate(source: str) -> bool:
    """Static check: the adapter binds the name `evaluate` at module level (not executed)."""
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "evaluate":
            return True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "evaluate"
                                                 for t in node.targets):
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (alias.asname or alias.name) == "evaluate" for alias in node.names):
            return True
    return False


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a quality config without running the adapter or loading sample values.

    Cheap enough to call before any preparation work. Returns a JSON-safe,
    normalized copy (resolved string paths) that is safe to hand to the worker.
    """
    import numpy as np

    if not isinstance(config, Mapping):
        raise ValueError("Quality config must be an object")
    if config.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported quality schema_version (expected {SCHEMA_VERSION})")
    unknown = sorted(set(config) - set(_CONFIG_KEYS) - {"schema_version"})
    if unknown:
        raise ValueError(f"Unknown quality config keys: {', '.join(map(str, unknown))}")
    missing = [key for key in _CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"Quality config requires {', '.join(missing)}")
    dataset = config["dataset"]
    if not isinstance(dataset, Mapping) or set(dataset) != {"id", "split", "seed"}:
        raise ValueError("dataset must be an object with exactly id, split and seed")
    if isinstance(dataset["seed"], bool) or not isinstance(dataset["seed"], int) or dataset["seed"] < 0:
        raise ValueError("dataset seed must be a nonnegative integer")
    if not isinstance(config["higher_is_better"], bool):
        raise ValueError("higher_is_better must be a boolean")
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {"id": _nonempty(dataset["id"], "dataset id"),
                    "split": _nonempty(dataset["split"], "dataset split"), "seed": dataset["seed"]},
        "metric": _nonempty(config["metric"], "metric"),
        "higher_is_better": config["higher_is_better"],
        "max_drop": _finite(config["max_drop"], "max_drop", nonnegative=True),
        "preprocessing": _nonempty(config["preprocessing"], "preprocessing"),
        "postprocessing": _nonempty(config["postprocessing"], "postprocessing"),
    }
    evaluation = _path(config["evaluation"], "evaluation")
    adapter = _path(config["adapter"], "adapter")
    if not evaluation.is_dir():
        raise ValueError(f"Quality evaluation must be a directory: {evaluation}")
    paths = sorted(evaluation.glob("*.npz"))
    if not paths:
        raise ValueError("Quality evaluation directory must contain named-input NPZ files")
    names = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            files = set(archive.files)
        if not files:
            raise ValueError(f"Quality sample {path} is empty")
        if names is not None and files != names:
            raise ValueError(f"Quality sample {path} has different input names from {paths[0]}")
        names = files
    if not adapter.is_file() or adapter.suffix != ".py":
        raise ValueError(f"Quality adapter must be an existing .py file: {adapter}")
    try:
        defined = _defines_evaluate(adapter.read_text())
    except (SyntaxError, UnicodeDecodeError) as error:
        raise ValueError(f"Quality adapter is not valid Python: {error}") from error
    if not defined:
        raise ValueError("Quality adapter must define evaluate(session, samples)")
    return {**normalized, "evaluation": str(evaluation), "adapter": str(adapter)}


def check_quantization_compatible(config: Mapping[str, Any], quantization: Mapping[str, Any]) -> None:
    """Reject, before costly work, a quality config that contradicts the quantization config.

    Only the cheap, certain conflicts: metric, direction, drop and adapter code.
    Evaluation content is compared exactly by :func:`attach_quality`.
    """
    from .pipeline import digest

    conflicts = [key for key in ("metric", "higher_is_better") if config[key] != quantization.get(key)]
    if float(quantization.get("max_drop", math.nan)) != config["max_drop"]:
        conflicts.append("max_drop")
    try:
        # The quantization config names an adapter path; its result metadata records the hash.
        adapter = quantization.get("adapter")
        recorded = adapter["sha256"] if isinstance(adapter, Mapping) else digest(_path(adapter, "adapter"))
        if digest(config["adapter"]) != recorded:
            conflicts.append("adapter")
    except (ValueError, OSError, KeyError):
        conflicts.append("adapter")
    if conflicts:
        raise ValueError("Quality config conflicts with the quantization config on "
                         f"{', '.join(conflicts)}; they share one quality contract, make them identical")


def validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the stored ``quality.json["spec"]`` (pure data, no filesystem)."""
    if not isinstance(spec, Mapping):
        raise ValueError("Quality spec must be an object")
    dataset, adapter = spec.get("dataset"), spec.get("adapter")
    if not isinstance(dataset, Mapping) or not isinstance(adapter, Mapping):
        raise ValueError("Quality spec lacks dataset or adapter")
    _nonempty(dataset.get("id"), "dataset id")
    _nonempty(dataset.get("split"), "dataset split")
    if isinstance(dataset.get("seed"), bool) or not isinstance(dataset.get("seed"), int):
        raise ValueError("dataset seed must be an integer")
    files = dataset.get("files")
    if (not isinstance(files, list) or not files or dataset.get("sample_count") != len(files)
            or not all(isinstance(item, Mapping) and _is_sha256(item.get("sha256"))
                       and _is_sha256(item.get("content_sha256")) for item in files)):
        raise ValueError("dataset files must list a sha256 and content_sha256 for each sample")
    if not _is_sha256(adapter.get("sha256")) or adapter.get("file") != ADAPTER_FILE:
        raise ValueError("adapter must record its file and sha256")
    _nonempty(spec.get("metric"), "metric")
    if not isinstance(spec.get("higher_is_better"), bool):
        raise ValueError("higher_is_better must be a boolean")
    _finite(spec.get("max_drop"), "max_drop", nonnegative=True)
    _nonempty(spec.get("preprocessing"), "preprocessing")
    _nonempty(spec.get("postprocessing"), "postprocessing")
    return dict(spec)


def model_identity(path: Path, *, root: Path | None = None, hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Identity of an ONNX file plus the external-data files it references.

    ``hashes`` maps paths relative to ``root`` to known sha256 values (a
    sealed manifest's ``files``); anything else is hashed from disk.
    """
    import onnx
    from onnx import TensorProto

    from .pipeline import digest

    path = Path(path)
    root = Path(root) if root is not None else path.parent

    def sha(item: Path) -> str:
        known = None
        if hashes is not None and item.is_relative_to(root):
            known = hashes.get(str(item.relative_to(root)))
        if known is None:
            if not item.is_file():
                raise ValueError(f"Model file is missing: {item}")
            known = digest(item)
        return known

    def tensors(graph):
        yield from graph.initializer
        for node in graph.node:
            for attribute in node.attribute:
                yield from attribute.tensors
                if attribute.HasField("t"):
                    yield attribute.t
                if attribute.HasField("g"):
                    yield from tensors(attribute.g)
                for sub in attribute.graphs:
                    yield from tensors(sub)

    model = onnx.load(str(path), load_external_data=False)
    locations = sorted({entry.value for tensor in tensors(model.graph)
                        if tensor.data_location == TensorProto.EXTERNAL
                        for entry in tensor.external_data if entry.key == "location"})
    external = [{"file": location, "sha256": sha(path.parent / location)} for location in locations]
    model_sha = sha(path)
    return {"file": path.name, "sha256": model_sha, "external_data": external,
            "digest": _canonical_digest({"sha256": model_sha,
                                         "external_data": [[e["file"], e["sha256"]] for e in external]})}


def _cpu_session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # Same options as quantization: no arena keeping each session's peak.
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _check_input_names(samples: list[dict[str, Any]], session, label: str) -> None:
    names = {item.name for item in session.get_inputs()}
    for index, sample in enumerate(samples):
        if set(sample) != names:
            raise ValueError(f"quality sample {index} input names differ from the {label} model: "
                             f"missing {sorted(names - set(sample))}, unexpected {sorted(set(sample) - names)} "
                             "(labels belong in the quality adapter)")


def _heldout(content: list[str], calibration: list[str] | None) -> dict[str, Any]:
    if not calibration:
        return {"status": "no-calibration", "calibration_samples": 0, "overlap": 0}
    overlap = len(set(content) & set(calibration))
    return {"status": "overlaps-calibration" if overlap else "disjoint-from-calibration",
            "calibration_samples": len(calibration), "overlap": overlap}


def attach_quality(directory: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Create ``<directory>/quality.json`` for a prepared (and maybe quantized) profile.

    ``directory`` holds ``model.onnx`` (the candidate), the reference model
    (``fp32.onnx`` when quantized, else ``original.onnx``) and optionally
    ``preparation.json``. Returns a small summary, or None without a config
    (a legacy quantization contract needs no attachment). Raises, leaving no
    ``quality.json``/fixtures behind, when the contract cannot be established;
    existing fixtures and manifests are never overwritten.

    CPU scores computed here are recorded as ``ort-cpu`` provenance only; a
    Vulkan claim comes from :func:`verify_quality`.
    """
    if config is None:
        return None
    from .pipeline import digest, write_json
    from .quantize import _adapter_metric, _load_adapter, _load_samples, _write_quality_fixtures

    directory = Path(directory).resolve()
    config = validate_config(config)
    if (directory / QUALITY_FILE).exists():
        raise FileExistsError(f"{directory / QUALITY_FILE} already exists")
    preparation_path = directory / "preparation.json"
    quantization = (json.loads(preparation_path.read_text()).get("quantization")
                    if preparation_path.is_file() else None)

    adapter_path = Path(config["adapter"])
    paths = sorted(Path(config["evaluation"]).glob("*.npz"))
    samples, content = _load_samples(paths, label="quality evaluation")
    adapter_sha = digest(adapter_path)
    dataset_files = [{"name": path.name, "sha256": digest(path), "content_sha256": identity}
                     for path, identity in zip(paths, content, strict=True)]
    candidate_path = directory / "model.onnx"
    if not candidate_path.is_file():
        raise ValueError(f"Quality needs {candidate_path}")

    if quantization is not None:
        check_quantization_compatible(config, quantization)
        if quantization.get("evaluation_content_hashes") != content:
            raise ValueError("Quality evaluation data differs from the quantization evaluation data; "
                             "they share one quality contract, use the same dataset")
        fixtures = quantization.get("quality_fixtures")
        reference_path = directory / "fp32.onnx"
        if not fixtures or not reference_path.is_file():
            raise ValueError("Quantization recorded no quality fixtures or fp32 reference to attach")
        if digest(candidate_path) != quantization.get("candidate_sha256") \
                or digest(reference_path) != quantization.get("original_sha256"):
            raise ValueError("Quantization metadata does not describe the model files present")
        if not (directory / ADAPTER_FILE).is_file() or digest(directory / ADAPTER_FILE) != adapter_sha:
            raise ValueError("Quantization quality adapter differs from the quality config adapter")
        heldout = _heldout(content, quantization.get("calibration_content_hashes"))
        if heldout["overlap"]:
            raise ValueError("Quality evaluation samples overlap the calibration data")
        reference_score = _finite(quantization["before"], "reference score")
        candidate_score = _finite(quantization["after"], "candidate score")
    else:
        reference_path = directory / "original.onnx"
        if not reference_path.is_file():
            raise ValueError(f"Quality needs the original model {reference_path} as its reference")
        if (directory / FIXTURE_DIR).exists():
            raise FileExistsError(f"Quality fixture directory already exists: {directory / FIXTURE_DIR}")
        existing_adapter = directory / ADAPTER_FILE
        if existing_adapter.exists() and digest(existing_adapter) != adapter_sha:
            raise FileExistsError(f"{existing_adapter} exists with different code")
        heldout = _heldout(content, None)
        adapter = _load_adapter(adapter_path)
        reference = _cpu_session(reference_path)
        _check_input_names(samples, reference, "original")
        reference_score = _adapter_metric(adapter, reference, samples)
        candidate = _cpu_session(candidate_path)
        _check_input_names(samples, candidate, "candidate")
        candidate_score = _adapter_metric(adapter, candidate, samples)
        drop = (reference_score - candidate_score if config["higher_is_better"]
                else candidate_score - reference_score)
        if drop > config["max_drop"]:
            raise ValueError(f"Quality regression {drop} on CPU exceeds {config['max_drop']}")
        outputs = [candidate.run(None, sample) for sample in samples]
        fixtures = _write_quality_fixtures(directory / FIXTURE_DIR, samples, outputs,
                                           [item.name for item in candidate.get_outputs()])
    try:
        if quantization is None:
            shutil.copyfile(adapter_path, directory / ADAPTER_FILE)
        spec = {"dataset": {**config["dataset"], "sample_count": len(paths), "files": dataset_files},
                "adapter": {"file": ADAPTER_FILE, "sha256": adapter_sha},
                "metric": config["metric"], "higher_is_better": config["higher_is_better"],
                "max_drop": config["max_drop"], "preprocessing": config["preprocessing"],
                "postprocessing": config["postprocessing"]}
        reference_identity = model_identity(reference_path)
        candidate_identity = model_identity(candidate_path)
        document = {
            "schema_version": SCHEMA_VERSION, "spec": spec,
            "reference": {"score": reference_score, "backend": "ort-cpu", "model": reference_identity},
            "candidate": {"cpu_score": candidate_score, "backend": "ort-cpu", "model": candidate_identity},
            "fixtures": {"backend": "ort-cpu", "model": candidate_identity, "samples": fixtures},
            "heldout": heldout,
        }
        document["spec_digest"] = _canonical_digest(spec)
        document["contract_digest"] = _canonical_digest(
            {"spec": document["spec_digest"], "score": reference_score,
             "reference": reference_identity["digest"]})
        validate_spec(spec)
        write_json(directory / QUALITY_FILE, document)
    except BaseException:
        if quantization is None:
            shutil.rmtree(directory / FIXTURE_DIR, ignore_errors=True)
            (directory / ADAPTER_FILE).unlink(missing_ok=True)
        raise
    return {"file": QUALITY_FILE, "contract_digest": document["contract_digest"],
            "reference_score": reference_score, "samples": len(paths), "heldout": heldout["status"],
            "backend": "ort-cpu"}


def load_contract(directory: Path) -> dict[str, Any] | None:
    """Normalized contract of a profile directory, or None when it has none.

    Reads ``quality.json``, else normalizes legacy quantization metadata.
    Raises when a contract exists but is malformed or its adapter was altered.
    """
    from .pipeline import digest

    directory = Path(directory)
    path = directory / QUALITY_FILE
    if path.is_file():
        document = json.loads(path.read_text())
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported quality.json schema")
        spec = validate_spec(document.get("spec"))
        reference, fixtures = document.get("reference"), document.get("fixtures")
        try:
            score = _finite(reference["score"], "reference score")
            expected = _canonical_digest({"spec": document["spec_digest"], "score": score,
                                          "reference": reference["model"]["digest"]})
            fixtures["model"]["digest"]
        except (KeyError, TypeError) as error:
            raise ValueError("quality.json lacks a reference score or model identity") from error
        if document["spec_digest"] != _canonical_digest(spec) or document.get("contract_digest") != expected:
            raise ValueError("quality.json digest mismatch")
        if not fixtures or len(fixtures.get("samples", [])) != spec["dataset"]["sample_count"]:
            raise ValueError("quality.json fixtures do not match the dataset sample count")
        contract = {
            "source": QUALITY_FILE, "digest": document["contract_digest"], "metric": spec["metric"],
            "higher_is_better": spec["higher_is_better"], "max_drop": float(spec["max_drop"]),
            "reference_score": score, "adapter_sha256": spec["adapter"]["sha256"],
            "content_hashes": [item["content_sha256"] for item in spec["dataset"]["files"]],
            "fixtures": fixtures["samples"], "fixture_model_sha256": fixtures["model"]["sha256"],
            "fixture_model_digest": fixtures["model"]["digest"],
            "dataset": {key: spec["dataset"][key] for key in ("id", "split", "seed")},
            "preprocessing": spec["preprocessing"], "postprocessing": spec["postprocessing"],
            "heldout": document.get("heldout"),
        }
    else:
        preparation = directory / "preparation.json"
        quantization = (json.loads(preparation.read_text()).get("quantization")
                        if preparation.is_file() else None)
        if not quantization or not quantization.get("quality_fixtures"):
            return None
        try:
            hashes = quantization["evaluation_content_hashes"]
            contract = {
                "source": "legacy-quantization",
                "digest": _canonical_digest({"config": quantization["config_digest"],
                                             "adapter": quantization["adapter_digest"],
                                             "before": quantization["before"]}),
                "metric": quantization["metric"], "higher_is_better": quantization["higher_is_better"],
                "max_drop": float(quantization["max_drop"]), "reference_score": quantization["before"],
                "adapter_sha256": quantization["adapter_digest"], "content_hashes": hashes,
                "fixtures": quantization["quality_fixtures"],
                "fixture_model_sha256": quantization["candidate_sha256"], "fixture_model_digest": None,
                "dataset": None, "preprocessing": None, "postprocessing": None,
                "heldout": _heldout(hashes, quantization.get("calibration_content_hashes")),
            }
        except KeyError as error:
            raise ValueError(f"Legacy quantization metadata lacks {error}") from error
        _finite(contract["reference_score"], "reference score")
        _finite(contract["max_drop"], "max_drop", nonnegative=True)
        if not isinstance(contract["higher_is_better"], bool) or len(hashes) != len(contract["fixtures"]):
            raise ValueError("Legacy quantization metadata is inconsistent")
    adapter = directory / ADAPTER_FILE
    if not adapter.is_file() or digest(adapter) != contract["adapter_sha256"]:
        raise ValueError("Quality adapter is missing or differs from the contract")
    return contract


def contract_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a contract that reports carry to say what was measured."""
    return {key: contract[key] for key in ("source", "digest", "metric", "dataset", "preprocessing",
                                           "postprocessing", "heldout")}


def check_fixtures(directory: Path, contract: Mapping[str, Any]) -> list[Path]:
    """Fixture sample directories, after checking files, hashes and the sample content."""
    from .pipeline import digest
    from .quantize import _reference_feeds, _sample_digest

    directory = Path(directory).resolve()
    paths = []
    for index, fixture in enumerate(contract["fixtures"]):
        sample = (directory / fixture["directory"]).resolve()
        if not sample.is_relative_to(directory) or not sample.is_dir():
            raise ValueError(f"Quality fixture is missing: {fixture['directory']}")
        for item in (*fixture["inputs"], *fixture["outputs"]):
            file = (directory / item["path"]).resolve()
            if not file.is_relative_to(sample) or not file.is_file() or digest(file) != item["sha256"]:
                raise ValueError(f"Quality fixture file is missing or modified: {item['path']}")
        if _sample_digest(_reference_feeds(sample)) != contract["content_hashes"][index]:
            raise ValueError(f"Quality fixture {fixture['directory']} inputs differ from the recorded dataset")
        paths.append(sample)
    present = {path for path in (directory / FIXTURE_DIR).glob("*") if path.is_dir()}
    if present != set(paths):
        raise ValueError("Quality fixture directory holds samples the contract does not record")
    return paths


def _unmeasured(reason: str, backend: str, contract: Mapping[str, Any] | None) -> dict[str, Any]:
    return {"samples": 0, "measured": False, "vulkan_verified": False, "reason": reason,
            "backend": None, "requested_backend": backend,
            "contract": contract_identity(contract) if contract else None}


def verify_quality(bundle: Path, manifest: Mapping[str, Any], profile: str, backend: str,
                   timeout: float) -> dict[str, Any]:
    """Measure the application metric of the *selected* model on Vulkan outputs.

    The contract's score for the original model is the reference. Fixture
    outputs describe the model that made them; when the selected model is a
    different one (fp32 fallback, a rewritten candidate) a CPU reference is
    regenerated for it on the same quality inputs in a temporary directory,
    never in the sealed bundle. Without a contract or fixtures the result is
    unmeasured, not an error and not a success claim.
    """
    import numpy as np

    from . import pipeline

    bundle = Path(bundle)
    directory = bundle / profile
    contract = load_contract(directory)
    if contract is None:
        return _unmeasured("no quality contract for this profile", backend, None)
    fixtures = check_fixtures(directory, contract)
    selected = manifest["profiles"][profile]
    model = bundle / selected["model"]
    identity = model_identity(model, root=bundle, hashes=manifest.get("files"))
    same = (identity["digest"] == contract["fixture_model_digest"]
            if contract["fixture_model_digest"] else identity["sha256"] == contract["fixture_model_sha256"])
    with tempfile.TemporaryDirectory(prefix="vulkan-quality-") as temporary:
        temporary = Path(temporary)
        if same:
            references, source = fixtures, "fixtures"
        else:
            regenerated = temporary / "regenerated"
            request, response = temporary / "reference-request.json", temporary / "reference-response.json"
            pipeline.write_json(request, {"mode": "generate_reference", "model": str(model),
                                          "samples": [str(path) for path in fixtures],
                                          "destination": str(regenerated), "response": str(response)})
            pipeline.command([sys.executable, "-m", "onnx_vulkan_pipeline._quality_worker", request],
                             timeout=timeout)
            references, source = sorted(path for path in regenerated.glob("*") if path.is_dir()), "regenerated-cpu"
            if len(references) != len(fixtures):
                raise RuntimeError("Reference regeneration did not cover every quality sample")
        records = []
        for index, reference in enumerate(references):
            result = pipeline._run(bundle, manifest, profile, backend, 1, timeout, tactics=selected["tactics"],
                                   reference=reference, output_data=True)
            # Kept as .npy files, not JSON lists: SAM 3's detector writes 17M
            # values a sample, and holding every sample as Python floats (then
            # serializing them again for the worker) exhausted 15 GB.
            data = result["output_data"]
            for position, item in enumerate(data["outputs"]):
                path = temporary / f"{index:04d}_{position}.npy"
                np.save(path, np.asarray(item.pop("values"), dtype=item["dtype"]).reshape(item["shape"]))
                item["file"] = str(path)
            records.append({"reference": str(reference), "result": data})
            del result
        request, response = temporary / "request.json", temporary / "response.json"
        pipeline.write_json(request, {"directory": str(directory), "records": records,
                                      "response": str(response)})
        pipeline.command([sys.executable, "-m", "onnx_vulkan_pipeline._quality_worker", request],
                         timeout=timeout)
        report = json.loads(response.read_text())
    return {**report, "requested_backend": backend, "reference_source": source,
            "selected_model": identity["digest"]}
