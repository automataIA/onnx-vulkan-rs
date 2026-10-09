"""Static quantization checks use a real ONNX Runtime graph, without Vulkan."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from onnx_vulkan_pipeline.quantize import VULKAN_OP_TYPES, quantize_model  # noqa: E402


def _model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3, 3])
    weights = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), "weights")
    bias = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), "bias")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "weights", "bias"], ["conv"]),
         helper.make_node("Relu", ["conv"], ["y"])],
        "tiny_quant", [x], [y], [weights, bias],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def _dataset(root: Path) -> tuple[Path, Path]:
    calibration, evaluation = root / "calibration", root / "evaluation"
    calibration.mkdir()
    evaluation.mkdir()
    np.savez(calibration / "000.npz", x=np.full((1, 1, 3, 3), -1.0, dtype=np.float32))
    np.savez(calibration / "001.npz", x=np.full((1, 1, 3, 3), 1.0, dtype=np.float32))
    np.savez(evaluation / "000.npz", x=np.full((1, 1, 3, 3), 0.5, dtype=np.float32))
    return calibration, evaluation


@pytest.fixture()
def adapter(tmp_path: Path) -> Path:
    path = tmp_path / "adapter.py"
    path.write_text(
        "import numpy as np\n"
        "def evaluate(session, samples):\n"
        "    return float(np.mean([session.run(None, sample)[0] for sample in samples]))\n"
    )
    return path


@pytest.mark.parametrize("quant_format", ["QDQ", "QOperator"])
def test_quantizes_tiny_onnx_graph_and_writes_model_runner_fixtures(
    tmp_path: Path, adapter: Path, quant_format: str,
):
    model = tmp_path / "model.onnx"
    _model(model)
    calibration, evaluation = _dataset(tmp_path)
    result = quantize_model(
        model,
        {"calibration": calibration, "evaluation": evaluation, "adapter": adapter,
         "metric": "mean_output", "max_drop": 0.25, "higher_is_better": True,
         "format": quant_format},
        atol=0.2,
        rtol=0.01,
    )

    assert model.is_file()
    assert (tmp_path / "fp32.onnx").is_file()
    assert result["format"] == quant_format
    assert result["quality_backend"] == "ort-cpu"
    assert result["vulkan_quality_verified"] is False
    assert result["rust_pipeline_check"] == {"status": "pending", "verified": False}
    assert result["numeric_comparison"]["max_absolute"] <= 0.2
    fixture = tmp_path / "quality" / "0000"
    input_proto = onnx.load_tensor(str(fixture / "input_0.pb"))
    output_proto = onnx.load_tensor(str(fixture / "output_0.pb"))
    assert input_proto.name == "x"
    assert output_proto.name == "y"
    np.testing.assert_allclose(numpy_helper.to_array(output_proto), 0.5, atol=0.2)
    assert result["quality_fixtures"][0]["directory"] == "quality/0000"


def test_duplicate_values_are_rejected_even_when_npz_bytes_differ(tmp_path: Path, adapter: Path):
    model = tmp_path / "model.onnx"
    _model(model)
    calibration, evaluation = _dataset(tmp_path)
    # Distinct ZIP metadata/compression, identical named tensor and values.
    np.savez_compressed(evaluation / "001.npz", x=np.full((1, 1, 3, 3), -1.0, dtype=np.float32))
    np.savez(evaluation / "002.npz", x=np.full((1, 1, 3, 3), -1.0, dtype=np.float32))
    with pytest.raises(ValueError, match="duplicate sample content"):
        quantize_model(
            model,
            {"calibration": calibration, "evaluation": evaluation, "adapter": adapter,
             "metric": "mean_output", "max_drop": 1, "higher_is_better": True,
             "format": "QDQ"},
            atol=0.2,
            rtol=0.01,
        )
    assert model.is_file()
    assert not (tmp_path / "quality").exists()


def test_invalid_tolerance_is_rejected(tmp_path: Path, adapter: Path):
    model = tmp_path / "model.onnx"
    _model(model)
    calibration, evaluation = _dataset(tmp_path)
    with pytest.raises(ValueError, match="Tolerances"):
        quantize_model(
            model,
            {"calibration": calibration, "evaluation": evaluation, "adapter": adapter,
             "metric": "mean_output", "max_drop": 1, "higher_is_better": True,
             "format": "QDQ"},
            atol=float("nan"),
            rtol=0.01,
        )


def test_application_gate_uses_measured_vulkan_outputs(tmp_path: Path, adapter: Path):
    import json
    from onnx_vulkan_pipeline._quality_worker import evaluate, RecordedSession

    model = tmp_path / "model.onnx"
    _model(model)
    calibration, evaluation = _dataset(tmp_path)
    result = quantize_model(model, {
        "calibration": calibration, "evaluation": evaluation, "adapter": adapter,
        "metric": "mean_output", "max_drop": 0.01, "higher_is_better": True,
        "format": "QDQ",
    }, atol=0.2, rtol=0.01)
    (tmp_path / "preparation.json").write_text(json.dumps({"quantization": result}))
    records = [{"reference": str(tmp_path / "quality/0000"), "result": {
        "backend": "standalone", "outputs": [{"name": "y", "dtype": "float32",
        "shape": [1, 1, 3, 3], "values": [0.5] * 9}]}}]
    files_before = {p.relative_to(tmp_path): p.read_bytes()
                    for p in tmp_path.rglob("*") if p.is_file()}
    assert evaluate(tmp_path, records)["vulkan_verified"]
    assert evaluate(tmp_path, records)["vulkan_verified"]
    assert files_before == {p.relative_to(tmp_path): p.read_bytes()
                            for p in tmp_path.rglob("*") if p.is_file()}
    # This difference passes the numerical atol=0.2, but fails the metric gate.
    records[0]["result"]["outputs"][0]["values"] = [0.4] * 9
    with pytest.raises(ValueError, match="application quality regression"):
        evaluate(tmp_path, records)
    with pytest.raises(ValueError, match="without measured Vulkan outputs"):
        RecordedSession(records).run(None, {"x": np.zeros((1, 1, 3, 3), dtype=np.float32)})
    # The pipeline records large outputs as .npy files instead of JSON lists.
    recorded = tmp_path / "recorded"
    recorded.mkdir()
    np.save(recorded / "y.npy", np.full((1, 1, 3, 3), 0.5, dtype=np.float32))
    output = {"name": "y", "dtype": "float32", "shape": [1, 1, 3, 3], "file": str(recorded / "y.npy")}
    records[0]["result"]["outputs"] = [output]
    assert evaluate(tmp_path, records)["vulkan_verified"]
    output["shape"] = [1, 9]
    with pytest.raises(ValueError, match="does not match its dtype/shape"):
        evaluate(tmp_path, records)


def _silu_model(path: Path, opset: int = 17) -> None:
    """Conv -> SiLU -> Concat, the block ORT's default op list turns into
    QLinearSigmoid/QLinearMul/QLinearConcat."""
    rng = np.random.default_rng(0)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 6, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 6, 6])
    weights = numpy_helper.from_array(rng.normal(size=(4, 4, 3, 3)).astype(np.float32), "weights")
    bias = numpy_helper.from_array(rng.normal(size=(4,)).astype(np.float32), "bias")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "weights", "bias"], ["conv"], pads=[1, 1, 1, 1]),
         helper.make_node("Sigmoid", ["conv"], ["sigmoid"]),
         helper.make_node("Mul", ["conv", "sigmoid"], ["silu"]),
         helper.make_node("Concat", ["silu", "conv"], ["y"], axis=1)],
        "silu_quant", [x], [y], [weights, bias],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)]), path)


def _silu_dataset(root: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(1)
    calibration, evaluation = root / "calibration", root / "evaluation"
    calibration.mkdir()
    evaluation.mkdir()
    for index in range(4):
        np.savez(calibration / f"{index:03d}.npz", x=rng.normal(size=(1, 4, 6, 6)).astype(np.float32))
    np.savez(evaluation / "000.npz", x=rng.normal(size=(1, 4, 6, 6)).astype(np.float32))
    return calibration, evaluation


def _silu_config(root: Path, adapter: Path, **extra):
    calibration, evaluation = _silu_dataset(root)
    return {"calibration": calibration, "evaluation": evaluation, "adapter": adapter,
            "metric": "mean_output", "max_drop": 10.0, "higher_is_better": True,
            "format": "QOperator", **extra}


def test_preparation_tolerances_reject_quantization_error_with_guidance(tmp_path: Path, adapter: Path):
    model = tmp_path / "model.onnx"
    _silu_model(model)
    with pytest.raises(ValueError, match="declare the accepted quantization error"):
        quantize_model(model, _silu_config(tmp_path, adapter), atol=1e-5, rtol=1e-4)
    assert model.is_file()
    assert not (tmp_path / "quantized.onnx").exists()
    assert not (tmp_path / "quality").exists()


def test_quantization_tolerances_are_separate_from_preparation(tmp_path: Path, adapter: Path):
    model = tmp_path / "model.onnx"
    _silu_model(model)
    result = quantize_model(model, _silu_config(tmp_path, adapter, atol=1.0, rtol=0.5),
                            atol=1e-5, rtol=1e-4)
    assert result["numeric_comparison"]["tolerance_source"] == "quantization"
    assert result["numeric_comparison"]["atol"] == 1.0


def test_default_op_types_emit_only_vulkan_qoperator_forms(tmp_path: Path, adapter: Path):
    model = tmp_path / "model.onnx"
    _silu_model(model)
    result = quantize_model(model, _silu_config(tmp_path, adapter, atol=1.0, rtol=0.5),
                            atol=1e-5, rtol=1e-4)
    ops = {node.op_type for node in onnx.load(str(model)).graph.node}
    assert "QLinearConv" in ops
    assert not ops & {"QLinearSigmoid", "QLinearMul", "QLinearConcat"}
    assert result["options"]["op_types_to_quantize"] == list(VULKAN_OP_TYPES)


@pytest.mark.parametrize("extra", [
    {"per_channel": True},
    {"calibrate_method": "Percentile"},
    {"preprocess": True},
    {"nodes_to_exclude": ["missing_node"]},
])
def test_quantization_options_are_applied_and_recorded(tmp_path: Path, adapter: Path, extra):
    model = tmp_path / "model.onnx"
    _silu_model(model)
    result = quantize_model(model, _silu_config(tmp_path, adapter, atol=1.0, rtol=0.5, **extra),
                            atol=1e-5, rtol=1e-4)
    for key, value in extra.items():
        assert result["options"][key] == value
    assert not any(path.name.startswith("quantize-") for path in tmp_path.iterdir())


def test_per_channel_weights_are_symmetric(tmp_path: Path, adapter: Path):
    model = tmp_path / "model.onnx"
    _silu_model(model)
    quantize_model(model, _silu_config(tmp_path, adapter, atol=1.0, rtol=0.5, per_channel=True),
                   atol=1e-5, rtol=1e-4)
    graph = onnx.load(str(model)).graph
    initializers = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    conv = next(node for node in graph.node if node.op_type == "QLinearConv")
    assert initializers[conv.input[4]].size == 4
    assert not initializers[conv.input[5]].any()


@pytest.mark.parametrize("extra, message", [
    ({"per_channel": "yes"}, "per_channel"),
    ({"calibrate_method": "Distribution"}, "calibrate_method"),
    ({"op_types_to_quantize": "Conv"}, "op_types_to_quantize"),
    ({"nodes_to_exclude": [""]}, "nodes_to_exclude"),
    ({"atol": -1.0}, "Tolerances"),
])
def test_invalid_quantization_options_are_rejected(tmp_path: Path, adapter: Path, extra, message):
    model = tmp_path / "model.onnx"
    _silu_model(model)
    with pytest.raises(ValueError, match=message):
        quantize_model(model, _silu_config(tmp_path, adapter, **extra), atol=1e-5, rtol=1e-4)


def test_quantized_model_gets_its_own_reference_and_fixtures(tmp_path: Path, adapter: Path):
    import onnxruntime as ort

    model = tmp_path / "model.onnx"
    _silu_model(model)
    reference = tmp_path / "reference"
    reference.mkdir()
    feed = np.random.default_rng(2).normal(size=(1, 4, 6, 6)).astype(np.float32)
    (reference / "input_0.pb").write_bytes(numpy_helper.from_array(feed, "x").SerializeToString())
    config = _silu_config(tmp_path, adapter, atol=1.0, rtol=0.5)
    result = quantize_model(model, config, atol=1e-5, rtol=1e-4)

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    expected = session.run(None, {"x": feed})[0]
    written = numpy_helper.to_array(onnx.load_tensor(str(tmp_path / "quantized-reference/output_0.pb")))
    np.testing.assert_array_equal(written, expected)
    assert result["reference"]["directory"] == "quantized-reference"

    sample = dict(np.load(next(Path(config["evaluation"]).glob("*.npz"))))
    fixture = numpy_helper.to_array(onnx.load_tensor(str(tmp_path / "quality/0000/output_0.pb")))
    np.testing.assert_array_equal(fixture, session.run(None, sample)[0])


def _without_calibration(config, **extra):
    config = {key: value for key, value in config.items() if key not in ("calibration", "format")}
    return {**config, **extra}


def test_dynamic_mode_needs_no_calibration_and_emits_integer_ops(tmp_path: Path, adapter: Path):
    model = tmp_path / "model.onnx"
    _silu_model(model)
    config = _without_calibration(_silu_config(tmp_path, adapter), mode="dynamic", atol=1.0, rtol=0.5)
    result = quantize_model(model, config, atol=1e-5, rtol=1e-4)
    ops = {node.op_type for node in onnx.load(str(model)).graph.node}
    assert {"DynamicQuantizeLinear", "ConvInteger"} <= ops
    assert result["mode"] == "dynamic"
    assert result["calibration_hashes"] == []
    assert result["options"] == {"op_types_to_quantize": ["Conv", "MatMul"], "nodes_to_exclude": [],
                                 "preprocess": False, "per_channel": False}


def _matmul_model(path: Path) -> None:
    rng = np.random.default_rng(3)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16])
    weights = numpy_helper.from_array(rng.normal(size=(64, 16)).astype(np.float32), "weights")
    graph = helper.make_graph([helper.make_node("MatMul", ["x", "weights"], ["y"])],
                              "matmul_quant", [x], [y], [weights])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def test_dynamic_mode_accepts_a_transposed_gemm_with_initializer_shapes(tmp_path: Path, adapter: Path):
    # Prepared models carry value_info for initializers; ORT's dynamic
    # quantizer transposes a transB Gemm weight in place and kept the old shape.
    model = tmp_path / "model.onnx"
    rng = np.random.default_rng(5)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16])
    weights = numpy_helper.from_array(rng.normal(size=(16, 64)).astype(np.float32), "weights")
    bias = numpy_helper.from_array(rng.normal(size=(16,)).astype(np.float32), "bias")
    graph = helper.make_graph([helper.make_node("Gemm", ["x", "weights", "bias"], ["y"], transB=1)],
                              "gemm_quant", [x], [y], [weights, bias],
                              value_info=[helper.make_tensor_value_info("weights", TensorProto.FLOAT, [16, 64])])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), model)
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    np.savez(evaluation / "000.npz", x=rng.normal(size=(1, 64)).astype(np.float32))
    config = {"mode": "dynamic", "evaluation": evaluation, "adapter": adapter, "metric": "mean_output",
              "max_drop": 10.0, "higher_is_better": True, "atol": 2.0, "rtol": 1.0}
    quantize_model(model, config, atol=1e-5, rtol=1e-4)
    assert "MatMulInteger" in {node.op_type for node in onnx.load(str(model)).graph.node}


def _convtranspose_model(path: Path) -> None:
    # SAM 3's neck upsampling (2×2, stride 2, bias) next to one with
    # overlapping taps (3×3, stride 2), which has no MatMul form
    rng = np.random.default_rng(7)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 5, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6, 10, 12])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4, 11, 13])
    weights = [numpy_helper.from_array(rng.normal(size=shape).astype(np.float32), name)
               for name, shape in (("w2", (8, 6, 2, 2)), ("b2", (6,)), ("w3", (8, 4, 3, 3)))]
    graph = helper.make_graph(
        [helper.make_node("ConvTranspose", ["x", "w2", "b2"], ["y"], "up", strides=[2, 2]),
         helper.make_node("ConvTranspose", ["x", "w3"], ["z"], "overlap", strides=[2, 2])],
        "convtranspose_quant", [x], [y, z], weights)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def test_non_overlapping_convtranspose_is_rewritten_exactly_as_a_matmul(tmp_path: Path):
    import onnxruntime as ort

    from onnx_vulkan_pipeline.quantize import _convtranspose_as_matmul

    original = tmp_path / "original.onnx"
    _convtranspose_model(original)
    model = onnx.load(str(original))
    assert _convtranspose_as_matmul(model, set()) == {"rewritten": ["up"], "kept": ["overlap"]}
    onnx.checker.check_model(model)
    rewritten = tmp_path / "rewritten.onnx"
    onnx.save(model, rewritten)
    feed = {"x": np.random.default_rng(8).normal(size=(1, 8, 5, 6)).astype(np.float32)}
    expected, got = (ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feed)
                     for path in (original, rewritten))
    np.testing.assert_allclose(got[0], expected[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(got[1], expected[1])

    excluded = onnx.load(str(original))
    assert _convtranspose_as_matmul(excluded, {"up"})["rewritten"] == []


def test_convtranspose_rewrite_biases_columns_and_reads_an_nhwc_source(tmp_path: Path):
    # SAM 3's neck: an NHWC activation transposed to NCHW feeds the upsampling
    import onnxruntime as ort

    from onnx_vulkan_pipeline.quantize import _convtranspose_as_matmul

    rng = np.random.default_rng(10)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 5, 6, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6, 10, 12])
    weights = [numpy_helper.from_array(rng.normal(size=shape).astype(np.float32), name)
               for name, shape in (("w", (8, 6, 2, 2)), ("b", (6,)))]
    graph = helper.make_graph(
        [helper.make_node("Transpose", ["x"], ["nchw"], "to_nchw", perm=[0, 3, 1, 2]),
         helper.make_node("ConvTranspose", ["nchw", "w", "b"], ["y"], "up", strides=[2, 2])],
        "convtranspose_nhwc", [x], [y], weights,
        value_info=[helper.make_tensor_value_info("nchw", TensorProto.FLOAT, [1, 8, 5, 6])])
    original = tmp_path / "original.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), original)
    model = onnx.load(str(original))
    assert _convtranspose_as_matmul(model, set())["rewritten"] == ["up"]
    nodes = {node.name: node for node in model.graph.node}
    # no transpose back to NHWC; the bias follows the product, before the shuffle
    assert [node.op_type for node in model.graph.node] == [
        "Transpose", "MatMul", "Add", "Reshape", "Transpose", "Reshape"]
    assert list(nodes["up/mm"].input) == ["x", "up/weight"]
    assert list(nodes["up/bias"].input) == ["up/mm", "up/bias"]
    rewritten = tmp_path / "rewritten.onnx"
    onnx.save(model, rewritten)
    feed = {"x": rng.normal(size=(1, 5, 6, 8)).astype(np.float32)}
    expected, got = (ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feed)
                     for path in (original, rewritten))
    np.testing.assert_allclose(got[0], expected[0], rtol=1e-5, atol=1e-5)


def test_dynamic_mode_quantizes_convtranspose_through_its_matmul(tmp_path: Path, adapter: Path):
    model = tmp_path / "model.onnx"
    _convtranspose_model(model)
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    np.savez(evaluation / "000.npz", x=np.random.default_rng(9).normal(size=(1, 8, 5, 6)).astype(np.float32))
    config = {"mode": "dynamic", "evaluation": evaluation, "adapter": adapter, "metric": "mean_output",
              "max_drop": 10.0, "higher_is_better": True, "atol": 2.0, "rtol": 1.0,
              "op_types_to_quantize": ["ConvTranspose", "MatMul"]}
    result = quantize_model(model, config, atol=1e-5, rtol=1e-4)
    ops = [node.op_type for node in onnx.load(str(model)).graph.node]
    assert ops.count("MatMulInteger") == 1 and ops.count("ConvTranspose") == 1
    assert result["options"]["convtranspose_as_matmul"] == {"rewritten": ["up"], "kept": ["overlap"]}


@pytest.mark.parametrize("extra", [{"op_types_to_quantize": ["ConvTranspose"]},
                                   {"mode": "int4", "op_types_to_quantize": ["ConvTranspose", "MatMul"]}])
def test_convtranspose_needs_dynamic_mode_and_matmul(tmp_path: Path, adapter: Path, extra):
    model = tmp_path / "model.onnx"
    _convtranspose_model(model)
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    config = {"mode": "dynamic", "evaluation": evaluation, "adapter": adapter, "metric": "mean_output",
              "max_drop": 10.0, "higher_is_better": True, **extra}
    with pytest.raises(ValueError, match="ConvTranspose is quantized as a MatMul"):
        quantize_model(model, config, atol=1e-5, rtol=1e-4)


def test_int4_mode_emits_asymmetric_matmulnbits(tmp_path: Path, adapter: Path):
    model = tmp_path / "model.onnx"
    _matmul_model(model)
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    np.savez(evaluation / "000.npz", x=np.random.default_rng(4).normal(size=(1, 64)).astype(np.float32))
    config = {"mode": "int4", "block_size": 32, "evaluation": evaluation, "adapter": adapter,
              "metric": "mean_output", "max_drop": 10.0, "higher_is_better": True,
              "atol": 2.0, "rtol": 1.0}
    result = quantize_model(model, config, atol=1e-5, rtol=1e-4)
    node = next(node for node in onnx.load(str(model)).graph.node if node.op_type == "MatMulNBits")
    attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
    assert attributes["bits"] == 4 and attributes["block_size"] == 32
    # the Vulkan kernel binds the zero point unconditionally
    assert len(node.input) > 3 and node.input[3]
    assert result["options"]["block_size"] == 32


@pytest.mark.parametrize("extra, message", [
    ({"mode": "fp8"}, "mode must be one of"),
    ({"mode": "dynamic"}, "calibration, format does not apply to mode dynamic"),
    ({"mode": "int4", "per_channel": True}, "per_channel does not apply to mode int4"),
    ({"block_size": 32}, "block_size does not apply to mode static"),
])
def test_mode_specific_keys_are_rejected(tmp_path: Path, adapter: Path, extra, message):
    model = tmp_path / "model.onnx"
    _silu_model(model)
    config = _silu_config(tmp_path, adapter, **extra)
    if extra.get("mode") == "int4":
        config = _without_calibration(config)
    with pytest.raises(ValueError, match=message):
        quantize_model(model, config, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("block_size", [24, 8, True])
def test_int4_block_size_must_be_a_power_of_two(tmp_path: Path, adapter: Path, block_size):
    model = tmp_path / "model.onnx"
    _matmul_model(model)
    config = _without_calibration(_silu_config(tmp_path, adapter), mode="int4", block_size=block_size)
    with pytest.raises(ValueError, match="block_size"):
        quantize_model(model, config, atol=1e-5, rtol=1e-4)


def test_per_channel_qdq_upgrades_an_opset_12_model_to_a_loadable_graph(tmp_path: Path, adapter: Path):
    # Per-channel DequantizeLinear needs `axis`, which opset 13 introduced;
    # ORT's quantizer emits it without upgrading the model.
    import onnxruntime as ort
    model = tmp_path / "model.onnx"
    _silu_model(model, opset=12)
    quantize_model(model, _silu_config(tmp_path, adapter, atol=1.0, rtol=0.5, format="QDQ", per_channel=True),
                   atol=1e-5, rtol=1e-4)
    loaded = onnx.load(str(model))
    assert next(item.version for item in loaded.opset_import if item.domain in ("", "ai.onnx")) >= 13
    ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
