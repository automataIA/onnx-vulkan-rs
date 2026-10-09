"""Opt-in quantization (static, dynamic or int4 weight-only) with separate,
auditable quality data.

Quantization is deliberately a CPU operation.  The output of this module is
only accepted after the original and quantized models have both run through
ONNX Runtime on every evaluation sample.  Vulkan compatibility and quality
are reported as unverified until the Rust pipeline consumes the generated
fixtures.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import tempfile
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from .pipeline import digest

# Op types whose QOperator form the Rust interpreter implements: QLinearConv,
# QLinearMatMul, QLinearAdd, QLinearGlobalAveragePool and MaxPool on uint8/int8.
# Relu and Clip are absorbed into the preceding output range. ONNX Runtime's
# default list also emits QLinearSigmoid/QLinearMul/QLinearConcat, which
# model-check refuses.
VULKAN_OP_TYPES = ("Conv", "MatMul", "Add", "GlobalAveragePool", "MaxPool", "Relu", "Clip")
# Dynamic quantization emits DynamicQuantizeLinear + ConvInteger/MatMulInteger;
# ONNX Runtime's default also covers Attention, LSTM and Gather forms.
DYNAMIC_OP_TYPES = ("Conv", "MatMul")
# Weight-only int4 emits MatMulNBits; "Gather" (GatherBlockQuantized) is opt-in.
INT4_OP_TYPES = ("MatMul",)
CALIBRATION_METHODS = ("MinMax", "Entropy", "Percentile")
# Keys that only one mode reads; any other mode rejects them instead of
# silently ignoring them.
MODE_KEYS = {"static": ("calibration", "format", "per_channel", "calibrate_method"),
             "dynamic": ("per_channel",),
             "int4": ("block_size",)}


def _config_digest(config: Mapping[str, Any]) -> str:
    """Hash the effective JSON configuration, including path values."""

    def normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    encoded = json.dumps(normalize(config), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_digest(sample: Mapping[str, Any]) -> str:
    """Return an identity for values, independent of NPZ compression/layout."""

    import numpy as np

    result = hashlib.sha256()
    for name in sorted(sample):
        value = np.asarray(sample[name])
        if value.dtype.hasobject:
            raise ValueError(f"NPZ sample {name!r} has an object dtype")
        array = np.ascontiguousarray(value)
        # Length prefixes keep names, dtypes, dimensions and bytes unambiguous.
        for field in (name.encode("utf-8"), array.dtype.str.encode("ascii"),
                      repr(tuple(int(dim) for dim in array.shape)).encode("ascii"),
                      array.tobytes(order="C")):
            result.update(len(field).to_bytes(8, "big"))
            result.update(field)
    return result.hexdigest()


def _load_samples(paths: list[Path], *, label: str) -> tuple[list[dict[str, Any]], list[str]]:
    import numpy as np

    samples: list[dict[str, Any]] = []
    identities: list[str] = []
    seen: dict[str, Path] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            names = tuple(archive.files)
            if not names:
                raise ValueError(f"{label} sample {path} is empty")
            sample = {name: np.array(archive[name], copy=True) for name in names}
        identity = _sample_digest(sample)
        if identity in seen:
            raise ValueError(f"duplicate sample content in {label} dataset: {path} and {seen[identity]}")
        seen[identity] = path
        samples.append(sample)
        identities.append(identity)
    return samples, identities


def _validate_config(config: Mapping[str, Any], *, atol: float, rtol: float) -> None:
    mode = config.get("mode", "static")
    if mode not in MODE_KEYS:
        raise ValueError(f"mode must be one of {', '.join(MODE_KEYS)}")
    unused = sorted(key for keys in MODE_KEYS.values() for key in keys
                    if key in config and key not in MODE_KEYS[mode])
    if unused:
        raise ValueError(f"{', '.join(unused)} does not apply to mode {mode}")
    required = ["evaluation", "adapter", "metric", "max_drop", "higher_is_better"]
    if mode == "static":
        required = ["calibration", *required, "format"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Quantization requires {', '.join(missing)}")
    if not isinstance(config["higher_is_better"], bool):
        raise ValueError("higher_is_better must be a boolean")
    if not isinstance(config["metric"], str) or not config["metric"].strip():
        raise ValueError("A named metric is required")
    if (isinstance(config["max_drop"], bool) or not isinstance(config["max_drop"], Real)
            or not math.isfinite(float(config["max_drop"])) or config["max_drop"] < 0):
        raise ValueError("max_drop must be finite and nonnegative")
    if mode == "static" and config["format"] not in ("QDQ", "QOperator"):
        raise ValueError("Quantization format must be QDQ or QOperator")
    block_size = config.get("block_size", 128)
    # ONNX Runtime requires a power of two >= 16, which also keeps the Vulkan
    # kernel's packed words inside one block.
    if (isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 16
            or block_size & (block_size - 1)):
        raise ValueError("block_size must be a power of two >= 16")
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value))
           or value < 0 for value in (atol, rtol, config.get("atol", 0), config.get("rtol", 0))):
        raise ValueError("Tolerances must be finite and nonnegative")
    for key in ("per_channel", "preprocess"):
        if not isinstance(config.get(key, False), bool):
            raise ValueError(f"{key} must be a boolean")
    if config.get("calibrate_method", "MinMax") not in CALIBRATION_METHODS:
        raise ValueError(f"calibrate_method must be one of {', '.join(CALIBRATION_METHODS)}")
    for key in ("op_types_to_quantize", "nodes_to_exclude"):
        value = config.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            raise ValueError(f"{key} must be a list of names")
    op_types = config.get("op_types_to_quantize", [])
    if "ConvTranspose" in op_types and (mode != "dynamic" or "MatMul" not in op_types):
        # ConvTranspose is quantized through its MatMul rewrite, dynamic only
        raise ValueError("ConvTranspose is quantized as a MatMul: it needs mode dynamic and MatMul")


def _load_adapter(path: Path):
    if not path.is_file():
        raise ValueError(f"Quality adapter does not exist: {path}")
    module_name = f"onnx_vulkan_quality_adapter_{abs(hash(path.resolve())):x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load quality adapter: {path}")
    adapter = importlib.util.module_from_spec(spec)
    # Bundle contents are sealed: importing must not add a __pycache__ artifact.
    exec(compile(path.read_bytes(), str(path), "exec"), adapter.__dict__)
    if not callable(getattr(adapter, "evaluate", None)):
        raise ValueError("Quality adapter must define evaluate(session, samples)")
    return adapter


def _fixture_name(index: int, prefix: str) -> str:
    # model-runner reads numbered files. Binding names remain in TensorProto.
    return f"{prefix}_{index}.pb"


def _write_quality_fixtures(root: Path, evaluation: list[dict[str, Any]], outputs: list[list[Any]],
                            output_names: list[str]) -> list[dict[str, Any]]:
    from onnx import numpy_helper

    root.mkdir(parents=True, exist_ok=False)
    fixture_metadata: list[dict[str, Any]] = []
    try:
        for sample_index, (sample, result) in enumerate(zip(evaluation, outputs, strict=True)):
            directory = root / f"{sample_index:04d}"
            directory.mkdir()
            input_paths: list[str] = []
            for index, name in enumerate(sorted(sample)):
                tensor = numpy_helper.from_array(sample[name], name=name)
                path = directory / _fixture_name(index, "input")
                path.write_bytes(tensor.SerializeToString())
                input_paths.append(str(path))
            output_paths: list[str] = []
            if len(result) != len(output_names):
                raise ValueError("Reference output count changed while writing quality fixtures")
            for index, (name, value) in enumerate(zip(output_names, result, strict=True)):
                tensor = numpy_helper.from_array(value, name=name)
                path = directory / _fixture_name(index, "output")
                path.write_bytes(tensor.SerializeToString())
                output_paths.append(str(path))
            fixture_metadata.append({
                "directory": str(directory.relative_to(root.parent)),
                "inputs": [{"path": str(Path(path).relative_to(root.parent)), "sha256": digest(path)} for path in input_paths],
                "outputs": [{"path": str(Path(path).relative_to(root.parent)), "sha256": digest(path)} for path in output_paths],
            })
    except Exception:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
        raise
    return fixture_metadata


def _numeric_report(reference: list[Any], candidate: list[Any], *, atol: float, rtol: float) -> dict[str, float]:
    import numpy as np

    if len(reference) != len(candidate):
        raise ValueError("Quantization changed output arity")
    max_absolute = 0.0
    max_relative = 0.0
    for expected, actual in zip(reference, candidate, strict=True):
        expected = np.asarray(expected)
        actual = np.asarray(actual)
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise ValueError("Quantization changed output shape or dtype")
        if not np.issubdtype(expected.dtype, np.number):
            if not np.array_equal(expected, actual):
                raise ValueError("Quantization changed a non-numeric output")
            continue
        if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
            raise ValueError("Quantization produced a non-finite output")
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        max_absolute = max(max_absolute, float(np.max(difference, initial=0.0)))
        denominator = np.maximum(np.abs(expected.astype(np.float64)), np.finfo(np.float64).tiny)
        max_relative = max(max_relative, float(np.max(difference / denominator, initial=0.0)))
        try:
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
        except AssertionError as error:
            raise ValueError(
                f"Quantized outputs exceed atol={atol}, rtol={rtol} (max absolute "
                f"{max_absolute}); declare the accepted quantization error as 'atol'/'rtol' "
                "in the quantization config") from error
    return {"max_absolute": max_absolute, "max_relative": max_relative,
            "atol": float(atol), "rtol": float(rtol)}


def _adapter_metric(adapter: Any, session: Any, samples: list[dict[str, Any]]) -> float:
    import numpy as np

    # Copies prevent a mutating adapter from changing the next evaluation.
    isolated = [{name: np.array(value, copy=True) for name, value in sample.items()}
                for sample in samples]
    value = adapter.evaluate(session, isolated)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Quality adapter evaluate(session, samples) must return a numeric scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Quality metric returned a non-finite value")
    return value


def _reference_feeds(directory: Path) -> dict[str, Any]:
    from onnx import load_tensor, numpy_helper

    feeds = {}
    for path in sorted(directory.glob("input_*.pb")):
        tensor = load_tensor(str(path))
        feeds[tensor.name] = numpy_helper.to_array(tensor)
    return feeds


def _convtranspose_as_matmul(model, exclude) -> dict[str, list[str]]:
    """Rewrite each non-overlapping ConvTranspose of ``model`` as a MatMul, in place.

    ORT's dynamic quantizer has no integer ConvTranspose. When the kernel
    equals the stride (no pads, groups or dilation), every output pixel takes
    one kernel tap: y[n, co, h*kh+i, w*kw+j] = sum_ci x[n, ci, h, w] *
    W[ci, co, i, j] + b[co], which is Transpose -> MatMul -> Add -> Reshape ->
    Transpose -> Reshape, exact in fp32, and the MatMul quantizes like any
    other. The layout serves the Vulkan runtime and changes no value:

    - the bias is added per MatMul column, right after it, so the quantized
      MatMul, its dequantization and the bias fuse into one kernel;
    - the columns are ordered (i, co, j), so the pixel shuffle moves pairs of
      kw values as a 2D transpose of (w, co) instead of interleaving single
      elements;
    - an input that is itself the NHWC -> NCHW Transpose of a value is read
      as that NHWC value, without transposing it back.

    Other ConvTransposes, excluded ones and those without a constant fp32
    weight or a static input shape stay as they are.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}
    shapes = {value.name: [dim.dim_value if dim.HasField("dim_value") else None
                           for dim in value.type.tensor_type.shape.dim]
              for value in (*graph.input, *graph.value_info)}
    to_nchw = {node.output[0]: node.input[0] for node in graph.node
               if node.op_type == "Transpose" and node.domain in ("", "ai.onnx")
               and [helper.get_attribute_value(item) for item in node.attribute if item.name == "perm"]
               == [[0, 3, 1, 2]]}
    rewritten: list[str] = []
    kept: list[str] = []
    nodes = []
    for node in graph.node:
        if node.op_type != "ConvTranspose" or node.domain not in ("", "ai.onnx"):
            nodes.append(node)
            continue
        attrs = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        weight = initializers.get(node.input[1])
        bias = initializers.get(node.input[2]) if len(node.input) > 2 and node.input[2] else None
        shape = shapes.get(node.input[0], [])
        kernel = list(weight.dims[2:]) if weight is not None else []
        eligible = (node.name not in exclude and weight is not None and len(kernel) == 2
                    and weight.data_type == TensorProto.FLOAT
                    and (len(node.input) < 3 or not node.input[2] or bias is not None)
                    and len(shape) == 4 and all(shape)
                    and list(attrs.get("strides", [1, 1])) == kernel
                    and list(attrs.get("kernel_shape", kernel)) == kernel
                    and not any(attrs.get("pads", [0])) and not any(attrs.get("output_padding", [0]))
                    and set(attrs.get("dilations", [1])) == {1} and attrs.get("group", 1) == 1
                    and attrs.get("auto_pad", b"NOTSET") in (b"NOTSET", b"VALID")
                    and "output_shape" not in attrs)
        if not eligible:
            kept.append(node.name)
            nodes.append(node)
            continue
        c_in, c_out, (kh, kw) = weight.dims[0], weight.dims[1], kernel
        n, _, h, w = shape
        name = node.name or f"convtranspose_{len(rewritten)}"
        columns = numpy_helper.to_array(weight).transpose(0, 2, 1, 3).reshape(c_in, kh * c_out * kw)
        graph.initializer.extend([
            numpy_helper.from_array(columns, f"{name}/weight"),
            numpy_helper.from_array(np.array([n, h, w, kh, c_out, kw], np.int64), f"{name}/split"),
            numpy_helper.from_array(np.array([n, c_out, h * kh, w * kw], np.int64), f"{name}/merge"),
        ])
        nhwc = to_nchw.get(node.input[0])
        if nhwc is None:
            nhwc = f"{name}/nhwc"
            nodes.append(helper.make_node("Transpose", [node.input[0]], [nhwc], nhwc, perm=[0, 2, 3, 1]))
        product = f"{name}/mm"
        nodes.append(helper.make_node("MatMul", [nhwc, f"{name}/weight"], [product], product))
        if bias is not None:
            per_column = np.broadcast_to(numpy_helper.to_array(bias)[None, :, None], (kh, c_out, kw))
            graph.initializer.append(numpy_helper.from_array(
                np.ascontiguousarray(per_column).reshape(-1), f"{name}/bias"))
            nodes.append(helper.make_node("Add", [product, f"{name}/bias"], [f"{name}/biased"], f"{name}/bias"))
            product = f"{name}/biased"
        nodes += [
            helper.make_node("Reshape", [product, f"{name}/split"], [f"{name}/split_out"], f"{name}/split"),
            helper.make_node("Transpose", [f"{name}/split_out"], [f"{name}/shuffled"], f"{name}/shuffle",
                             perm=[0, 4, 1, 3, 2, 5]),
            helper.make_node("Reshape", [f"{name}/shuffled", f"{name}/merge"], [node.output[0]], f"{name}/merge"),
        ]
        rewritten.append(node.name)
    if rewritten:
        del graph.node[:]
        graph.node.extend(nodes)
        used = {name for node in graph.node for name in node.input}
        unused = [item for item in graph.initializer if item.name not in used]
        for item in unused:
            graph.initializer.remove(item)
    return {"rewritten": rewritten, "kept": kept}


def quantize_model(model, config, *, atol, rtol):
    """Quantize ``model`` in place; ``atol``/``rtol`` are the preparation tolerances.

    ``mode`` selects static (calibrated activations), dynamic (activations
    quantized at run time) or int4 (weight-only MatMulNBits). The config may
    declare its own ``atol``/``rtol`` for the fp32-vs-quantized comparison.
    Otherwise the preparation tolerances apply unchanged.
    """
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_dynamic, quantize_static)

    if not isinstance(config, Mapping):
        raise ValueError("Quantization config must be an object")
    _validate_config(config, atol=atol, rtol=rtol)
    tolerance_source = "quantization" if "atol" in config or "rtol" in config else "preparation"
    atol, rtol = config.get("atol", atol), config.get("rtol", rtol)
    mode = config.get("mode", "static")
    default_op_types = {"static": VULKAN_OP_TYPES, "dynamic": DYNAMIC_OP_TYPES,
                        "int4": INT4_OP_TYPES}[mode]
    options = {"op_types_to_quantize": list(config.get("op_types_to_quantize", default_op_types)),
               "nodes_to_exclude": list(config.get("nodes_to_exclude", [])),
               "preprocess": config.get("preprocess", False)}
    if mode in ("static", "dynamic"):
        options["per_channel"] = config.get("per_channel", False)
    if mode == "static":
        options["calibrate_method"] = config.get("calibrate_method", "MinMax")
    if mode == "int4":
        options["block_size"] = config.get("block_size", 128)
    model = Path(model).resolve()
    try:
        calibration_dir = Path(config["calibration"]).resolve() if mode == "static" else None
        evaluation_dir = Path(config["evaluation"]).resolve()
    except (TypeError, ValueError) as error:
        raise ValueError("Calibration and evaluation must be path-like directories") from error
    if (calibration_dir is not None and not calibration_dir.is_dir()) or not evaluation_dir.is_dir():
        raise ValueError("Calibration and evaluation must be directories")
    calibration_paths = sorted(calibration_dir.glob("*.npz")) if calibration_dir else []
    evaluation_paths = sorted(evaluation_dir.glob("*.npz"))
    if (calibration_dir is not None and not calibration_paths) or not evaluation_paths:
        raise ValueError("Calibration and evaluation directories must contain named-input NPZ files")
    calibration, calibration_content = _load_samples(calibration_paths, label="calibration")
    evaluation, evaluation_content = _load_samples(evaluation_paths, label="evaluation")
    if set(calibration_content) & set(evaluation_content):
        raise ValueError("calibration and evaluation datasets contain duplicate sample content")
    calibration_hashes = [digest(path) for path in calibration_paths]
    evaluation_hashes = [digest(path) for path in evaluation_paths]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # The reference and candidate sessions live together; an arena keeps each
    # one's activation peak after its run, which for large encoders exceeds
    # the worker's memory bound. Values do not depend on these options.
    session_options.enable_cpu_mem_arena = False
    session_options.enable_mem_pattern = False
    reference = ort.InferenceSession(str(model), sess_options=session_options,
                                     providers=["CPUExecutionProvider"])
    input_names = {item.name for item in reference.get_inputs()}
    for label, samples in (("calibration", calibration), ("evaluation", evaluation)):
        for index, sample in enumerate(samples):
            if set(sample) != input_names:
                missing, extra = sorted(input_names - set(sample)), sorted(set(sample) - input_names)
                details = []
                if missing:
                    details.append(f"missing {', '.join(missing)}")
                if extra:
                    details.append(f"unexpected {', '.join(extra)} (labels belong in the quality adapter)")
                raise ValueError(f"{label} sample {index} input names differ: {'; '.join(details)}")

    try:
        adapter_path = Path(config["adapter"]).resolve()
    except (TypeError, ValueError) as error:
        raise ValueError("Quality adapter must be a path") from error
    adapter = _load_adapter(adapter_path)
    output = model.with_name("quantized.onnx")
    if output.exists():
        raise ValueError(f"Quantization candidate already exists: {output}")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = iter(calibration)

        def get_next(self):
            return next(self.items, None)

    quantized_reference = model.parent / "quantized-reference"
    if quantized_reference.exists():
        raise ValueError(f"Quantized reference already exists: {quantized_reference}")
    try:
        with tempfile.TemporaryDirectory(prefix="quantize-", dir=model.parent) as temporary:
            source = model
            if options["preprocess"]:
                # Shape inference and ORT's basic graph cleanup, as ORT recommends
                # before static quantization. Kept outside the bundle.
                from onnxruntime.quantization.shape_inference import quant_pre_process
                source = Path(temporary) / "preprocessed.onnx"
                quant_pre_process(str(model), str(source), save_as_external_data=True,
                                  all_tensors_to_one_file=True,
                                  external_data_location="preprocessed.onnx.data")
            if mode == "static" and config["format"] == "QDQ" and options["per_channel"]:
                import onnx
                loaded = onnx.load(str(source))
                opset = next((item.version for item in loaded.opset_import if item.domain in ("", "ai.onnx")), 0)
                if opset < 13:
                    # Per-channel DequantizeLinear needs `axis`, added in opset
                    # 13; ORT's quantizer emits it without upgrading older
                    # models, which then fail to load. Upgrade the copy.
                    source = Path(temporary) / "opset13.onnx"
                    onnx.save_model(onnx.version_converter.convert_version(loaded, 13), str(source),
                                    save_as_external_data=True, all_tensors_to_one_file=True,
                                    location=f"{source.name}.data")
            if mode == "static":
                quantize_static(str(source), str(output), Reader(),
                                quant_format=getattr(QuantFormat, config["format"]),
                                op_types_to_quantize=options["op_types_to_quantize"],
                                nodes_to_exclude=options["nodes_to_exclude"],
                                calibrate_method=getattr(CalibrationMethod, options["calibrate_method"]),
                                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                                per_channel=options["per_channel"], use_external_data_format=True)
            elif mode == "dynamic":
                import onnx
                # ORT's dynamic quantizer rewrites Gemm(transB=1) as MatMul by
                # transposing the weight initializer in place, but keeps its
                # value_info, so the shape inference that follows rejects the
                # stale shape. Prepared models carry value_info for
                # initializers; their shapes are known without it.
                stripped = onnx.load(str(source))
                op_types = options["op_types_to_quantize"]
                if "ConvTranspose" in op_types:
                    rewrite = _convtranspose_as_matmul(stripped, set(options["nodes_to_exclude"]))
                    options["convtranspose_as_matmul"] = rewrite
                    op_types = [op for op in op_types if op != "ConvTranspose"]
                initializers = {item.name for item in stripped.graph.initializer}
                kept = [item for item in stripped.graph.value_info if item.name not in initializers]
                rewritten = options.get("convtranspose_as_matmul", {}).get("rewritten")
                if len(kept) != len(stripped.graph.value_info) or rewritten:
                    del stripped.graph.value_info[:]
                    stripped.graph.value_info.extend(kept)
                    source = Path(temporary) / "without-initializer-shapes.onnx"
                    onnx.save_model(stripped, str(source), save_as_external_data=True,
                                    all_tensors_to_one_file=True, location=f"{source.name}.data")
                quantize_dynamic(str(source), str(output),
                                 op_types_to_quantize=op_types,
                                 nodes_to_exclude=options["nodes_to_exclude"],
                                 per_channel=options["per_channel"], weight_type=QuantType.QInt8,
                                 use_external_data_format=True)
            else:
                import onnx
                from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
                # Asymmetric only: the Vulkan MatMulNBits kernel binds the
                # zero-point input, which symmetric quantization omits.
                quantizer = MatMulNBitsQuantizer(
                    onnx.load(str(source)), block_size=options["block_size"], is_symmetric=False,
                    nodes_to_exclude=options["nodes_to_exclude"],
                    op_types_to_quantize=tuple(options["op_types_to_quantize"]))
                quantizer.process()
                quantizer.model.save_model_to_file(str(output), use_external_data_format=True)
        candidate = ort.InferenceSession(str(output), sess_options=session_options,
                                         providers=["CPUExecutionProvider"])
        candidate_outputs: list[list[Any]] = []
        numeric = {"max_absolute": 0.0, "max_relative": 0.0,
                   "atol": float(atol), "rtol": float(rtol), "tolerance_source": tolerance_source}
        for sample in evaluation:
            expected = reference.run(None, sample)
            actual = candidate.run(None, sample)
            report = _numeric_report(expected, actual, atol=float(atol), rtol=float(rtol))
            numeric["max_absolute"] = max(numeric["max_absolute"], report["max_absolute"])
            numeric["max_relative"] = max(numeric["max_relative"], report["max_relative"])
            candidate_outputs.append(actual)
        before = _adapter_metric(adapter, reference, evaluation)
        after = _adapter_metric(adapter, candidate, evaluation)
        drop = before - after if config["higher_is_better"] else after - before
        if drop > float(config["max_drop"]):
            raise ValueError(f"Quality regression {drop} exceeds {config['max_drop']}")

        quality_root = model.parent / "quality"
        if quality_root.exists():
            raise ValueError(f"Quality fixture directory already exists: {quality_root}")
        # Fixture outputs are the quantized model's own CPU outputs: model-runner
        # compares both backends against them, and the application metric (not
        # this comparison) bounds the quantization error.
        fixtures = _write_quality_fixtures(quality_root, evaluation, candidate_outputs,
                                           [item.name for item in reference.get_outputs()])
        reference_metadata = None
        if (model.parent / "reference").is_dir():
            # `reference/` holds fp32 outputs for original.onnx; the quantized
            # model gets its own outputs on the same inputs.
            from .prepare import _write_reference
            reference_metadata = _write_reference(output, quantized_reference,
                                                  _reference_feeds(model.parent / "reference"),
                                                  model.parent)
            if reference_metadata is None:
                raise ValueError("Could not write quantized-model reference tensors")
        backup = model.with_name("fp32.onnx")
        if backup.exists():
            raise ValueError(f"Original model backup already exists: {backup}")
        import shutil
        shutil.copyfile(adapter_path, model.parent / "quality_adapter.py")
        model.replace(backup)
        output.replace(model)
    except Exception:
        if output.exists():
            output.unlink()
        if quantized_reference.exists():
            import shutil
            shutil.rmtree(quantized_reference)
        raise

    config_digest = _config_digest(config)
    adapter_sha = digest(adapter_path)
    return {
        "mode": mode,
        "format": config.get("format"),
        "options": options,
        "reference": reference_metadata,
        "metric": config["metric"],
        "before": before,
        "after": after,
        "drop": drop,
        "max_drop": float(config["max_drop"]),
        "higher_is_better": config["higher_is_better"],
        "calibration_hashes": calibration_hashes,
        "evaluation_hashes": evaluation_hashes,
        "calibration_content_hashes": calibration_content,
        "evaluation_content_hashes": evaluation_content,
        "config_digest": config_digest,
        "adapter": {"path": "quality_adapter.py", "sha256": adapter_sha,
                    "metric": config["metric"], "evaluate": "evaluate(session, samples)"},
        "adapter_digest": adapter_sha,
        "numeric_comparison": numeric,
        "quality_fixtures": fixtures,
        "quality_backend": "ort-cpu",
        "vulkan_quality_verified": False,
        "rust_pipeline_check": {"status": "pending", "verified": False},
        "candidate_sha256": digest(model),
        "original_sha256": digest(backup),
    }
