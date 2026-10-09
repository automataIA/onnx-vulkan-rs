"""Prepare an ONNX model and record enough evidence to use the result.

The module keeps heavyweight imports lazy so package discovery remains cheap.
Verified preparation requires the project's ONNX, ONNX Runtime, NumPy, and
onnxscript dependencies; PyTorch is needed only when an adapter is supplied.
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


class PreparationError(RuntimeError):
    """Raised when the source or adapter cannot be prepared safely."""


class VerificationError(PreparationError):
    """Raised when original and candidate models disagree on validation data."""

    def __init__(self, message: str, metadata: dict[str, Any]):
        super().__init__(message)
        self.metadata = metadata


def _import(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError as exc:  # pragma: no cover - wording is user-facing
        how = "run with `uv run --extra torch`" if name == "torch" else "install it in the uv environment"
        raise PreparationError(f"{name} is required for this preparation step; {how}") from exc


def _load_onnx(path: Path, *, external: bool) -> Any:
    onnx = _import("onnx")
    try:
        return onnx.load(str(path), load_external_data=external)
    except TypeError:  # Older ONNX versions do not expose the keyword.
        return onnx.load(str(path), external_data=external)


def _json_path(path: Path) -> str:
    return str(path.resolve())


def _bundle_path(path: Path, destination: Path) -> str:
    """Render an artifact path relative to the preparation bundle."""
    try:
        return path.resolve().relative_to(destination.resolve()).as_posix()
    except ValueError:
        # The source/adapter may intentionally live outside the bundle. Keep
        # that provenance absolute while every generated artifact is relative.
        return _json_path(path)


def _safe_external_location(location: str) -> Path:
    """Return a safe relative external-data path, rejecting path escapes."""

    location_path = Path(location)
    if location_path.is_absolute() or ".." in location_path.parts:
        raise PreparationError(f"unsafe ONNX external-data location: {location!r}")
    if not location_path.parts or any(part in ("", ".") for part in location_path.parts):
        raise PreparationError(f"invalid ONNX external-data location: {location!r}")
    if location_path.name in {"model.onnx", "original.onnx", "candidate.data"}:
        raise PreparationError(f"external-data location collides with a bundle model: {location!r}")
    return location_path


def _external_locations(model: Any) -> list[Path]:
    locations: list[Path] = []

    def tensor_location(tensor: Any) -> None:
        for entry in tensor.external_data:
            if entry.key == "location":
                location = _safe_external_location(entry.value)
                if location not in locations:
                    locations.append(location)
                break

    def sparse_locations(sparse_tensor: Any) -> None:
        if sparse_tensor.HasField("values"):
            tensor_location(sparse_tensor.values)
        if sparse_tensor.HasField("indices"):
            tensor_location(sparse_tensor.indices)

    def graphs(graph: Any):
        yield graph
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == 5 and attribute.HasField("g"):
                    yield from graphs(attribute.g)
                elif attribute.type == 10:
                    for child in attribute.graphs:
                        yield from graphs(child)

    for graph in graphs(model.graph):
        for tensor in graph.initializer:
            tensor_location(tensor)
        for sparse_tensor in graph.sparse_initializer:
            sparse_locations(sparse_tensor)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == 4 and attribute.HasField("t"):
                    tensor_location(attribute.t)
                elif attribute.type == 9:
                    for tensor in attribute.tensors:
                        tensor_location(tensor)
                elif attribute.type == 11 and attribute.HasField("sparse_tensor"):
                    sparse_locations(attribute.sparse_tensor)
                elif attribute.type == 12:
                    for sparse_tensor in attribute.sparse_tensors:
                        sparse_locations(sparse_tensor)
    return locations


def _copy_source_tree(source: Path, destination: Path, model: Any) -> list[str]:
    """Copy the model and external files while preserving relative locations."""

    destination.mkdir(parents=True, exist_ok=True)
    original = destination / "original.onnx"
    candidate = destination / "model.onnx"
    if source.resolve() != original.resolve():
        shutil.copy2(source, original)
    else:
        shutil.copy2(source, original.with_suffix(".onnx.copy"))
        shutil.move(original.with_suffix(".onnx.copy"), original)

    copied: list[str] = []
    for location in _external_locations(model):
        from_path = source.parent / location
        if not from_path.is_file():
            raise PreparationError(
                f"ONNX external data referenced by {source} is missing: {from_path}"
            )
        to_path = destination / location
        to_path.parent.mkdir(parents=True, exist_ok=True)
        if from_path.resolve() != to_path.resolve():
            shutil.copy2(from_path, to_path)
        copied.append(_json_path(to_path))

    # Keep a byte-for-byte candidate copy until the transformed model is saved.
    if source.resolve() != candidate.resolve():
        shutil.copy2(source, candidate)
    return copied


def _load_adapter(path: Path, required=("load_model", "example_inputs")) -> ModuleType:
    if not path.is_file():
        raise PreparationError(f"adapter does not exist: {path}")
    module_name = f"onnx_vulkan_adapter_{abs(hash(path.resolve())):x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PreparationError(f"cannot load adapter module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for name in required:
        if not callable(getattr(module, name, None)):
            raise PreparationError(f"adapter must define callable {name}(): {path}")
    return module


def _adapter_examples(module, profiles):
    for name, size in (profiles or {}).items():
        if not isinstance(name, str) or not name or isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("Profile dimensions must have names and positive integer sizes")
    if "profile" in inspect.signature(module.example_inputs).parameters:
        return module.example_inputs(profile=dict(profiles or {}))
    if profiles:
        raise PreparationError("An adapter with explicit profiles must accept example_inputs(profile=...)")
    return module.example_inputs()


def _call_adapter_export(adapter_path: Path, output: Path, destination: Path, profiles=None) -> dict[str, Any]:
    """Export an adapter model using the newest exporter options available."""

    module = _load_adapter(adapter_path)
    torch = _import("torch")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = module.load_model()
    examples = _adapter_examples(module, profiles)
    if not isinstance(examples, Mapping) or not examples:
        raise PreparationError("adapter example_inputs() must return a non-empty mapping")
    example_map = dict(examples)
    names = [str(name) for name in example_map]
    if any(not name for name in names):
        raise PreparationError("adapter input names must be non-empty")

    if callable(getattr(model, "eval", None)):
        model.eval()

    with torch.no_grad() if hasattr(torch, "no_grad") else _null_context():
        try:
            reference = model(**example_map)
        except TypeError:
            reference = model(*tuple(example_map.values()))
    output_names = _output_names(reference)
    common = dict(
        input_names=names,
        output_names=output_names or None,
        opset_version=21,
        report=True,
        artifacts_dir=str(destination),
    )

    try:
        torch.onnx.export(
            model,
            args=(),
            f=str(output),
            kwargs=example_map,
            **dict(common, dynamo=True),
        )
    except (TypeError, RuntimeError, ValueError) as exc:
        raise PreparationError(f"PyTorch dynamo ONNX export failed: {exc}") from exc

    reports = sorted(
        p for p in destination.glob("*.md") if "report" in p.name.lower()
    )
    return {
        "path": _json_path(adapter_path),
        "sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "exported": True,
        "dynamo": True,
        "report": _json_path(reports[-1]) if reports else None,
        "input_names": names,
        "output_names": output_names,
        "has_calibration_data": callable(getattr(module, "calibration_data", None)),
        "has_evaluation_data": callable(getattr(module, "evaluation_data", None)),
        "has_evaluate": callable(getattr(module, "evaluate", None)),
        "_module": module,
        "_model": model,
        "_examples": example_map,
        "_reference": reference,
    }


def _load_adapter_for_validation(adapter_path: Path, profiles=None) -> dict[str, Any]:
    """Load an adapter without exporting over an existing ONNX source.

    Over an ONNX source only `example_inputs()` is required. With `load_model()`
    the model it returns is also run on those inputs and must match the ONNX
    model; that needs torch.
    """
    module = _load_adapter(adapter_path, required=("example_inputs",))
    examples = _adapter_examples(module, profiles)
    if not isinstance(examples, Mapping) or not examples:
        raise PreparationError("adapter example_inputs() must return a non-empty mapping")
    example_map = dict(examples)
    public = {
        "path": _json_path(adapter_path),
        "sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "exported": False,
        "dynamo": False,
        "report": None,
        "input_names": [str(name) for name in example_map],
        "has_calibration_data": callable(getattr(module, "calibration_data", None)),
        "has_evaluation_data": callable(getattr(module, "evaluation_data", None)),
        "has_evaluate": callable(getattr(module, "evaluate", None)),
    }
    if not callable(getattr(module, "load_model", None)):
        return {**public, "output_names": [], "_module": module, "_examples": example_map}
    torch = _import("torch")
    model = module.load_model()
    if not callable(model):
        raise PreparationError("adapter load_model() must return a callable model")
    if callable(getattr(model, "eval", None)):
        model.eval()
    with torch.no_grad() if hasattr(torch, "no_grad") else _null_context():
        try:
            reference = model(**example_map)
        except TypeError:
            reference = model(*tuple(example_map.values()))
    return {
        **public,
        "output_names": _output_names(reference),
        "_module": module,
        "_model": model,
        "_examples": example_map,
        "_reference": reference,
    }


class _null_context:
    def __enter__(self) -> "_null_context":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def _output_names(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [str(key) for key in value]
    if isinstance(value, (tuple, list)):
        return [f"output_{i}" for i in range(len(value))]
    return ["output_0"] if value is not None else []


def _shape_dims(value_info: Any) -> list[Any]:
    tensor_type = value_info.type.tensor_type
    result: list[Any] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(str(dim.dim_param))
        else:
            result.append(None)
    return result


def _value_metadata(values: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in values:
        result.append(
            {
                "name": str(value.name),
                "elem_type": int(value.type.tensor_type.elem_type),
                "shape": _shape_dims(value),
            }
        )
    return result


def _specialize(model: Any, profiles: Mapping[str, int] | None) -> tuple[list[str], list[str]]:
    if profiles is None:
        return [], []
    normalized: dict[str, int] = {}
    for name, size in profiles.items():
        if not isinstance(name, str) or not name:
            raise ValueError("profile dimension names must be non-empty strings")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"profile dimension {name!r} must be a positive integer")
        normalized[name] = int(size)

    all_names: set[str] = set()
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                all_names.add(str(dim.dim_param))
    unknown = sorted(set(normalized) - all_names)
    if unknown:
        raise ValueError(f"profile dimensions are not present in the model: {', '.join(unknown)}")

    changed: list[str] = []
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_param") and dim.dim_param in normalized:
                dim.dim_value = normalized[dim.dim_param]
                dim.ClearField("dim_param")
                changed.append(str(dim.dim_value))
    return sorted(normalized), changed


def _initializer_map(model: Any) -> dict[str, Any]:
    return {str(init.name): init for init in model.graph.initializer}


def _fold_conv_batchnorm(model: Any) -> int:
    """Fold inference Conv -> BatchNormalization pairs with exact formulas."""

    onnx = _import("onnx")
    numpy = _import("numpy")
    from onnx import numpy_helper

    initializers = _initializer_map(model)
    consumers: dict[str, list[Any]] = {}
    producers: dict[str, Any] = {}
    for node in model.graph.node:
        for output_name in node.output:
            if output_name:
                producers[str(output_name)] = node
        for input_name in node.input:
            if input_name:
                consumers.setdefault(str(input_name), []).append(node)
    folded = 0
    remove: set[int] = set()
    for index, bn in enumerate(model.graph.node):
        if bn.op_type != "BatchNormalization" or len(bn.input) < 5 or not bn.input[0]:
            continue
        # Training-mode BatchNormalization is intentionally not folded.
        training_mode = next((a.i for a in bn.attribute if a.name == "training_mode"), 0)
        if int(training_mode) != 0:
            continue
        conv = producers.get(str(bn.input[0]))
        if conv is None or conv.op_type != "Conv":
            continue
        if len(consumers.get(str(bn.input[0]), [])) != 1:
            continue
        if len(conv.output) != 1 or not conv.output[0] or not bn.output or not bn.output[0]:
            continue
        if any(name not in initializers for name in bn.input[1:5]):
            continue
        if conv.input[1] not in initializers:
            continue
        weight = numpy_helper.to_array(initializers[conv.input[1]])
        if weight.ndim < 1:
            continue
        scale = numpy_helper.to_array(initializers[bn.input[1]]).astype(weight.dtype, copy=False)
        bias_bn = numpy_helper.to_array(initializers[bn.input[2]]).astype(weight.dtype, copy=False)
        mean = numpy_helper.to_array(initializers[bn.input[3]]).astype(weight.dtype, copy=False)
        variance = numpy_helper.to_array(initializers[bn.input[4]]).astype(weight.dtype, copy=False)
        if scale.size != weight.shape[0] or any(x.size != scale.size for x in (bias_bn, mean, variance)):
            continue
        epsilon = next((float(a.f) for a in bn.attribute if a.name == "epsilon"), 1e-5)
        factor = scale / numpy.sqrt(variance + epsilon)
        reshaped = factor.reshape((factor.size,) + (1,) * (weight.ndim - 1))
        folded_weight = weight * reshaped

        if len(conv.input) >= 3 and conv.input[2]:
            if conv.input[2] not in initializers:
                continue
            bias = numpy_helper.to_array(initializers[conv.input[2]]).astype(weight.dtype, copy=False)
        else:
            bias = numpy.zeros(weight.shape[0], dtype=weight.dtype)
        folded_bias = (bias - mean) * factor + bias_bn

        weight_name = conv.input[1]
        initializers[weight_name].CopyFrom(numpy_helper.from_array(folded_weight, weight_name))
        bias_name = conv.input[2] if len(conv.input) >= 3 and conv.input[2] else f"{conv.name or 'conv'}_folded_bias"
        if len(conv.input) < 3:
            conv.input.append(bias_name)
        initializers[bias_name] = numpy_helper.from_array(folded_bias, bias_name)
        # The initializer list is a protobuf repeated field; append only when
        # the Conv previously had no bias.
        if not any(init.name == bias_name for init in model.graph.initializer):
            model.graph.initializer.append(initializers[bias_name])
        conv.output[0] = bn.output[0]
        remove.add(index)
        folded += 1

    if remove:
        kept = [node for i, node in enumerate(model.graph.node) if i not in remove]
        del model.graph.node[:]
        model.graph.node.extend(kept)
    return folded


def _save_candidate(model: Any, path: Path, had_external: bool) -> None:
    onnx = _import("onnx")
    if had_external:
        # A fresh sidecar avoids stale offsets from the source's external file;
        # onnx appends to an existing one, so a second save must not find it.
        # Small tensors stay inline: ORT's load-time shape inference cannot
        # read shape operands from external data.
        (path.parent / "candidate.data").unlink(missing_ok=True)
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="candidate.data",
            size_threshold=1024,
        )
    else:
        onnx.save_model(model, str(path))


def _ort_inputs(model: Any, profiles: Mapping[str, int] | None) -> tuple[dict[str, Any] | None, str | None]:
    numpy = _import("numpy")
    type_map = {
        1: numpy.float32,
        2: numpy.uint8,
        3: numpy.int8,
        4: numpy.uint16,
        5: numpy.int16,
        6: numpy.int32,
        7: numpy.int64,
        9: numpy.bool_,
        10: numpy.float16,
        11: numpy.float64,
        12: numpy.uint32,
        13: numpy.uint64,
        14: numpy.complex64,
        15: numpy.complex128,
    }
    feeds: dict[str, Any] = {}
    initializer_names = {str(init.name) for init in model.graph.initializer}
    for value in model.graph.input:
        if value.name in initializer_names:
            continue
        shape: list[int] = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                size = int(dim.dim_value)
            elif dim.HasField("dim_param") and profiles and dim.dim_param in profiles:
                size = int(profiles[dim.dim_param])
            elif dim.HasField("dim_param"):
                return None, (f"input {value.name!r} has symbolic dimension {dim.dim_param!r}; "
                              f"pass its size, e.g. --dim {dim.dim_param}=1")
            else:
                return None, f"input {value.name!r} has a dimension without a value or a symbol name"
            if size < 0:
                return None, f"input {value.name!r} has an invalid dimension"
            shape.append(size)
        dtype = type_map.get(int(value.type.tensor_type.elem_type))
        if dtype is None:
            return None, f"input {value.name!r} has unsupported element type"
        count = math.prod(shape) if shape else 1
        if count > 2_000_000:
            return None, (f"input {value.name!r} has {count} elements, too many to generate for verification; "
                          "pass example inputs with --adapter, an adapter whose example_inputs() returns them "
                          "(python/README.md, PyTorch adapter)")
        if numpy.issubdtype(dtype, numpy.bool_):
            feeds[str(value.name)] = numpy.zeros(shape, dtype=dtype)
        elif numpy.issubdtype(dtype, numpy.integer):
            feeds[str(value.name)] = numpy.arange(count, dtype=dtype).reshape(shape or ()) % 3
        else:
            feeds[str(value.name)] = (numpy.arange(count, dtype=numpy.float64).reshape(shape or ()) % 7 - 3) / 5
            feeds[str(value.name)] = feeds[str(value.name)].astype(dtype)
    return feeds, None


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _compare_ort(original: Path, candidate: Path, feeds: Mapping[str, Any], atol: float, rtol: float) -> dict[str, Any]:
    ort = _import("onnxruntime")
    numpy = _import("numpy")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        old = ort.InferenceSession(str(original), opts, providers=["CPUExecutionProvider"])
        new = ort.InferenceSession(str(candidate), opts, providers=["CPUExecutionProvider"])
        old_names = [x.name for x in old.get_outputs()]
        new_names = [x.name for x in new.get_outputs()]
        if old_names != new_names:
            return {"ok": False, "reason": "output names changed", "original": old_names, "candidate": new_names}
        old_values = old.run(None, {k: _to_numpy(v) for k, v in feeds.items()})
        new_values = new.run(None, {k: _to_numpy(v) for k, v in feeds.items()})
        max_abs = 0.0
        max_rel = 0.0
        for lhs, rhs in zip(old_values, new_values):
            left = numpy.asarray(lhs)
            right = numpy.asarray(rhs)
            if left.shape != right.shape:
                return {"ok": False, "reason": "output shape changed", "shape": [list(left.shape), list(right.shape)]}
            if left.dtype != right.dtype:
                return {
                    "ok": False,
                    "reason": "output dtype changed",
                    "dtype": [str(left.dtype), str(right.dtype)],
                }
            if numpy.issubdtype(left.dtype, numpy.floating):
                if not numpy.all(numpy.isfinite(left)) or not numpy.all(numpy.isfinite(right)):
                    return {"ok": False, "reason": "non-finite numeric output"}
                diff = numpy.abs(left.astype(numpy.float64) - right.astype(numpy.float64))
                max_abs = max(max_abs, float(diff.max()) if diff.size else 0.0)
                denom = numpy.maximum(numpy.abs(left.astype(numpy.float64)), 1e-12)
                max_rel = max(max_rel, float((diff / denom).max()) if diff.size else 0.0)
                numpy.testing.assert_allclose(left, right, atol=atol, rtol=rtol, equal_nan=False)
            elif numpy.issubdtype(left.dtype, numpy.complexfloating):
                if not numpy.all(numpy.isfinite(left)) or not numpy.all(numpy.isfinite(right)):
                    return {"ok": False, "reason": "non-finite numeric output"}
                if not numpy.array_equal(left, right):
                    return {"ok": False, "reason": "integer or complex output mismatch"}
            elif not numpy.array_equal(left, right):
                return {"ok": False, "reason": "non-numeric output mismatch"}
        return {"ok": True, "outputs": len(old_values), "max_abs": max_abs, "max_rel": max_rel}
    except Exception as exc:  # ORT has several exception classes across versions.
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def _compare_torch(candidate: Path, adapter: dict[str, Any], atol: float, rtol: float) -> dict[str, Any]:
    try:
        ort = _import("onnxruntime")
        numpy = _import("numpy")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(candidate), opts, providers=["CPUExecutionProvider"])
        feeds = {name: _to_numpy(value) for name, value in adapter["_examples"].items()}
        actual = session.run(None, feeds)
        expected = adapter["_reference"]
        if isinstance(expected, Mapping):
            expected_values = list(expected.values())
        elif isinstance(expected, (tuple, list)):
            expected_values = list(expected)
        else:
            expected_values = [expected]
        if len(actual) != len(expected_values):
            return {"ok": False, "reason": "PyTorch/export output count differs"}
        max_abs = max_rel = 0.0
        for lhs, rhs in zip(expected_values, actual):
            left = numpy.asarray(_to_numpy(lhs))
            right = numpy.asarray(rhs)
            if left.shape != right.shape:
                return {"ok": False, "reason": "PyTorch/export output shape differs"}
            if left.dtype != right.dtype:
                return {"ok": False, "reason": "PyTorch/export output dtype differs", "dtype": [str(left.dtype), str(right.dtype)]}
            if numpy.issubdtype(left.dtype, numpy.floating):
                if not numpy.all(numpy.isfinite(left)) or not numpy.all(numpy.isfinite(right)):
                    return {"ok": False, "reason": "non-finite PyTorch/export output"}
                diff = numpy.abs(left.astype(numpy.float64) - right.astype(numpy.float64))
                max_abs = max(max_abs, float(diff.max()) if diff.size else 0.0)
                denom = numpy.maximum(numpy.abs(left.astype(numpy.float64)), 1e-12)
                max_rel = max(max_rel, float((diff / denom).max()) if diff.size else 0.0)
                numpy.testing.assert_allclose(left, right, atol=atol, rtol=rtol, equal_nan=False)
            elif numpy.issubdtype(left.dtype, numpy.complexfloating):
                if not numpy.all(numpy.isfinite(left)) or not numpy.all(numpy.isfinite(right)):
                    return {"ok": False, "reason": "non-finite PyTorch/export output"}
                if not numpy.array_equal(left, right):
                    return {"ok": False, "reason": "integer or complex PyTorch/export output mismatch"}
            elif not numpy.array_equal(left, right):
                return {"ok": False, "reason": "integer or non-numeric PyTorch/export output mismatch"}
        return {"ok": True, "outputs": len(actual), "max_abs": max_abs, "max_rel": max_rel}
    except Exception as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def _write_reference(
    original: Path,
    directory: Path,
    feeds: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any] | None:
    """Write ORT's original-model tensors in model-runner's TensorProto format."""
    try:
        ort = _import("onnxruntime")
        numpy = _import("numpy")
        onnx = _import("onnx")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(original), opts, providers=["CPUExecutionProvider"])
        normalized = {name: _to_numpy(value) for name, value in feeds.items()}
        outputs = session.run(None, normalized)
        directory.mkdir(parents=True, exist_ok=True)
        input_paths: list[str] = []
        for index, input_meta in enumerate(session.get_inputs()):
            if input_meta.name not in normalized:
                return None
            tensor = onnx.numpy_helper.from_array(
                numpy.asarray(normalized[input_meta.name]), name=input_meta.name
            )
            path = directory / f"input_{index}.pb"
            path.write_bytes(tensor.SerializeToString())
            input_paths.append(_json_path(path))
        output_paths: list[str] = []
        for index, (output_meta, value) in enumerate(zip(session.get_outputs(), outputs)):
            tensor = onnx.numpy_helper.from_array(numpy.asarray(value), name=output_meta.name)
            path = directory / f"output_{index}.pb"
            path.write_bytes(tensor.SerializeToString())
            output_paths.append(_json_path(path))
        return {
            "directory": _bundle_path(directory, destination),
            "inputs": [_bundle_path(Path(path), destination) for path in input_paths],
            "outputs": [_bundle_path(Path(path), destination) for path in output_paths],
            "backend": "onnxruntime_cpu",
        }
    except Exception:
        # Reference generation is a convenience artifact. Numerical
        # verification still reports its own detailed error.
        return None


def _public_adapter(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if data is None:
        return None
    return {key: value for key, value in data.items() if not key.startswith("_")}


def prepare_model(
    source: Path,
    destination: Path,
    *,
    adapter: Path | None = None,
    profiles: dict[str, int] | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Copy, conservatively optimise, and verify an ONNX model.

    ``source`` is an existing ONNX file unless ``adapter`` is supplied and the
    source does not yet exist; in that case the adapter exports it at opset 21.
    Profile values resolve only dimensions explicitly named in ``profiles``.
    The return value contains JSON-compatible strings, numbers, lists, and
    dictionaries and does not claim that arbitrary ONNX operators are supported
    by the Vulkan execution provider.
    """

    source = Path(source)
    provenance_source = source
    destination = Path(destination)
    if (
        isinstance(atol, bool)
        or isinstance(rtol, bool)
        or not isinstance(atol, (int, float))
        or not isinstance(rtol, (int, float))
        or not math.isfinite(float(atol))
        or not math.isfinite(float(rtol))
        or atol < 0
        or rtol < 0
    ):
        raise ValueError("atol and rtol must be finite non-negative numbers")
    adapter_data: dict[str, Any] | None = None
    exported = False
    if adapter is not None:
        adapter = Path(adapter)
        # The adapter owns checkpoint loading. Never send a non-ONNX source to
        # the ONNX parser, and always write an adapter export into the bundle.
        if source.is_file() and source.suffix.lower() == ".onnx":
            adapter_data = _load_adapter_for_validation(adapter, profiles)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            exported_source = destination / "exported.onnx"
            adapter_data = _call_adapter_export(adapter, exported_source, destination, profiles)
            source = exported_source
            exported = True
    elif not source.is_file() or source.suffix.lower() != ".onnx":
        raise PreparationError(f"source must be an existing .onnx file or use an adapter: {source}")
    if not source.is_file() or source.suffix.lower() != ".onnx":
        raise PreparationError(f"ONNX source does not exist: {source}")

    onnx = _import("onnx")
    raw_model = _load_onnx(source, external=False)
    had_external = bool(_external_locations(raw_model))
    external_paths = _copy_source_tree(source, destination, raw_model)
    original_model = _load_onnx(destination / "original.onnx", external=True)
    candidate_model = copy.deepcopy(original_model)
    input_metadata = _value_metadata(original_model.graph.input)
    output_metadata = _value_metadata(original_model.graph.output)
    # Export is static for the representative inputs of each explicit profile.
    # Its symbols have already been consumed by example_inputs(profile=...).
    profile_names, _ = _specialize(candidate_model, None if exported else profiles)
    if exported:
        profile_names = sorted(profiles or {})

    transformations: list[dict[str, Any]] = []
    folded = _fold_conv_batchnorm(candidate_model)
    transformations.append({"name": "conv_batchnorm_fold", "count": folded, "status": "applied"})

    optimizer_status = "skipped"
    optimizer_error: str | None = None
    try:
        optimizer = _import("onnxscript").optimizer
        optimized = optimizer.optimize(
            candidate_model,
            num_iterations=1,
            onnx_shape_inference=True,
            stop_if_no_change=True,
            inline=False,
        )
        if hasattr(optimized, "to_proto"):
            optimized = optimized.to_proto()
        if hasattr(optimized, "SerializeToString"):
            candidate_model = optimized
            optimizer_status = "applied"
        else:
            optimizer_status = "skipped"
            optimizer_error = "onnxscript returned an unsupported model representation"
    except (ImportError, AttributeError) as exc:
        optimizer_error = str(exc)
    except Exception as exc:
        optimizer_status = "failed"
        optimizer_error = f"{type(exc).__name__}: {exc}"
    transformations.append({"name": "onnxscript_optimize", "status": optimizer_status, **({"error": optimizer_error} if optimizer_error else {})})

    # Optimisers may legally alter value_info but must not alter the public API.
    original_inputs = [str(value.name) for value in original_model.graph.input]
    original_outputs = [str(value.name) for value in original_model.graph.output]
    candidate_inputs = [str(value.name) for value in candidate_model.graph.input]
    candidate_outputs = [str(value.name) for value in candidate_model.graph.output]
    names_ok = original_inputs == candidate_inputs and original_outputs == candidate_outputs
    if not names_ok:
        transformations.append({"name": "public_name_check", "status": "failed"})
    _save_candidate(candidate_model, destination / "model.onnx", had_external)

    verification: dict[str, Any]
    reference_data: dict[str, Any] | None = None
    if not names_ok:
        verification = {"status": "failed", "ok": False, "reason": "public input/output names changed"}
    else:
        feeds: dict[str, Any] | None = None
        reason: str | None = None
        if adapter_data and "_examples" in adapter_data:
            feeds = adapter_data["_examples"]
        else:
            feeds, reason = _ort_inputs(original_model, profiles)
        if feeds is None:
            verification = {"status": "skipped", "ok": False, "reason": reason}
        else:
            try:
                reference_data = _write_reference(
                    destination / "original.onnx", destination / "reference", feeds, destination
                )
                original_candidate = _compare_ort(
                    destination / "original.onnx", destination / "model.onnx", feeds, float(atol), float(rtol)
                )
                if not original_candidate.get("ok") and optimizer_status == "applied":
                    # An optimizer that changes the outputs is not kept: the
                    # candidate falls back to the model without it (RF-DETR:
                    # max |Δ| 7e-5 in fp32, a third of the int8 outputs).
                    entry = next(t for t in transformations if t["name"] == "onnxscript_optimize")
                    entry.update(status="rejected",
                                 reason=f"outputs differ from the original model: {original_candidate.get('reason')}")
                    candidate_model = copy.deepcopy(original_model)
                    _specialize(candidate_model, None if exported else profiles)
                    _fold_conv_batchnorm(candidate_model)
                    _save_candidate(candidate_model, destination / "model.onnx", had_external)
                    original_candidate = _compare_ort(
                        destination / "original.onnx", destination / "model.onnx", feeds, float(atol), float(rtol)
                    )
                verification = {"status": "passed" if original_candidate.get("ok") else "failed", "ok": bool(original_candidate.get("ok")), "original_vs_candidate": original_candidate}
                if reference_data is None:
                    verification["status"] = "failed"
                    verification["ok"] = False
                    verification["reference_error"] = "could not write original-model reference tensors"
            except PreparationError as exc:
                verification = {"status": "skipped", "ok": False, "reason": str(exc)}
            if adapter_data and "_reference" in adapter_data:
                torch_check = _compare_torch(destination / "original.onnx", adapter_data, float(atol), float(rtol))
                verification["pytorch_vs_original"] = torch_check
                if not torch_check.get("ok"):
                    verification["status"] = "failed"
                    verification["ok"] = False

    metadata = {
        "paths": {
            "source": _bundle_path(provenance_source, destination),
            "effective_source": _bundle_path(source, destination),
            "original": "original.onnx",
            "model": "model.onnx",
            "candidate": "model.onnx",
            "external_data": [
                _bundle_path(Path(path), destination) for path in external_paths
            ],
            "reference": "reference"
            if reference_data is not None
            else None,
        },
        "source": _json_path(provenance_source),
        "effective_source": _bundle_path(source, destination),
        "original": "original.onnx",
        "model": "model.onnx",
        "candidate": "model.onnx",
        "reference": reference_data,
        "profiles": {str(k): int(v) for k, v in (profiles or {}).items()},
        "profile_dimensions_specialized": profile_names,
        "inputs": input_metadata,
        "outputs": output_metadata,
        "transformations": transformations,
        "verification": verification,
        "candidate_accepted": bool(verification.get("ok")),
        "support": {
            "status": "unverified",
            "reason": "Preparation verifies graph equivalence; Vulkan operator coverage is decided by the runtime.",
        },
        "adapter": _public_adapter(adapter_data),
    }
    # Ensure future additions cannot accidentally leak protobuf/tensor objects.
    json.dumps(metadata)
    if verification.get("status") != "passed" or reference_data is None:
        detail = verification.get("original_vs_candidate") or verification.get("reason") or verification
        raise VerificationError(f"candidate verification did not pass: {detail}", metadata)
    return metadata


__all__ = ["PreparationError", "VerificationError", "prepare_model"]
