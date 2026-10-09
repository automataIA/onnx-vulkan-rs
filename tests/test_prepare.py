"""Small preparation fixtures: graph rewrites, provenance, and safe skipping."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

onnx = pytest.importorskip("onnx")
numpy = pytest.importorskip("numpy")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from onnx_vulkan_pipeline import prepare_model  # noqa: E402
import onnx_vulkan_pipeline.prepare as prepare_module  # noqa: E402


def _conv_bn_model(dynamic: bool = False):
    shape = [1, 1, 2, 2] if not dynamic else ["batch", 1, 2, 2]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)
    initializers = [
        numpy_helper.from_array(numpy.ones((1, 1, 1, 1), dtype=numpy.float32), "w"),
        numpy_helper.from_array(numpy.array([2], dtype=numpy.float32), "scale"),
        numpy_helper.from_array(numpy.array([1], dtype=numpy.float32), "bias"),
        numpy_helper.from_array(numpy.array([0.5], dtype=numpy.float32), "mean"),
        numpy_helper.from_array(numpy.array([0.25], dtype=numpy.float32), "var"),
    ]
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["conv_out"]),
        helper.make_node("BatchNormalization", ["conv_out", "scale", "bias", "mean", "var"], ["y"]),
    ]
    return helper.make_model(
        helper.make_graph(nodes, "conv_bn", [x], [y], initializers),
        opset_imports=[helper.make_opsetid("", 21)],
    )


def test_conv_bn_fold_preserves_original_and_verifies(tmp_path: Path):
    source = tmp_path / "source.onnx"
    onnx.save(_conv_bn_model(), source)
    destination = tmp_path / "prepared"
    metadata = prepare_model(source, destination)

    assert (destination / metadata["original"]).read_bytes() == source.read_bytes()
    assert metadata["verification"]["ok"] is True
    assert metadata["candidate_accepted"] is True
    assert metadata["paths"]["original"] == "original.onnx"
    assert metadata["paths"]["model"] == "model.onnx"
    assert metadata["paths"]["reference"] == "reference"
    candidate = onnx.load(destination / metadata["candidate"])
    assert [node.op_type for node in candidate.graph.node].count("BatchNormalization") == 0
    assert metadata["reference"]["inputs"]
    assert metadata["reference"]["outputs"]


def test_external_weights_are_copied(tmp_path: Path):
    source = tmp_path / "source.onnx"
    onnx.save_model(
        _conv_bn_model(),
        source,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    destination = tmp_path / "prepared"
    metadata = prepare_model(source, destination)

    assert (destination / "original.onnx").is_file()
    assert (destination / "weights.bin").is_file()
    assert metadata["paths"]["external_data"]
    assert metadata["verification"]["ok"]


def test_a_candidate_sidecar_holds_large_tensors_once_and_keeps_shapes_inline(tmp_path: Path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4096])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [64, 64])
    initializers = [
        numpy_helper.from_array(numpy.ones((1, 4096), dtype=numpy.float32), "w"),
        numpy_helper.from_array(numpy.array([64, 64], dtype=numpy.int64), "shape"),
    ]
    nodes = [helper.make_node("Mul", ["x", "w"], ["m"]), helper.make_node("Reshape", ["m", "shape"], ["y"])]
    model = helper.make_model(helper.make_graph(nodes, "reshape", [x], [y], initializers),
                              opset_imports=[helper.make_opsetid("", 21)])
    path = tmp_path / "model.onnx"

    # The optimizer-rejected fallback saves the candidate a second time.
    prepare_module._save_candidate(onnx.ModelProto.FromString(model.SerializeToString()), path, True)
    prepare_module._save_candidate(model, path, True)

    assert (tmp_path / "candidate.data").stat().st_size < 2 * 4096 * 4
    saved = {t.name: t.data_location for t in onnx.load(str(path), load_external_data=False).graph.initializer}
    assert saved == {"w": TensorProto.EXTERNAL, "shape": TensorProto.DEFAULT}


def test_unresolved_shapes_fail_strict_preparation(tmp_path: Path):
    source = tmp_path / "source.onnx"
    onnx.save(_conv_bn_model(dynamic=True), source)
    with pytest.raises(prepare_module.VerificationError) as error:
        prepare_model(source, tmp_path / "prepared")

    # The message names the flag that resolves the symbol.
    assert "e.g. --dim batch=1" in str(error.value)
    metadata = error.value.metadata
    assert metadata["verification"]["status"] == "skipped"
    assert metadata["candidate_accepted"] is False
    assert metadata["support"]["status"] == "unverified"


def test_mismatch_is_reported_and_candidate_not_accepted(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    onnx.save(_conv_bn_model(), source)

    monkeypatch.setattr(
        prepare_module,
        "_compare_ort",
        lambda *args, **kwargs: {"ok": False, "reason": "fixture mismatch"},
    )
    with pytest.raises(prepare_module.VerificationError) as error:
        prepare_model(source, tmp_path / "prepared")

    metadata = error.value.metadata
    assert metadata["verification"]["status"] == "failed"
    assert metadata["verification"]["ok"] is False
    assert metadata["candidate_accepted"] is False


def test_torch_export_checks_original_with_explicit_profiles(tmp_path: Path):
    pytest.importorskip("torch")
    adapter = tmp_path / "adapter.py"
    adapter.write_text('''import torch
class Vision(torch.nn.Module):
    def forward(self, images):
        return images.relu() + 0.25
def load_model():
    return Vision().eval()
def example_inputs(profile):
    batch = profile["batch"]
    return {"images": torch.linspace(-1, 1, batch * 12).reshape(batch, 3, 2, 2)}
''')
    for batch in (1, 2):
        destination = tmp_path / f"batch{batch}"
        result = prepare_model(tmp_path / "checkpoint.pt", destination,
                               adapter=adapter, profiles={"batch": batch})
        assert result["verification"]["pytorch_vs_original"]["ok"]
        tensor = onnx.load_tensor(str(destination / "reference/input_0.pb"))
        assert tensor.dims[0] == batch


def test_an_inputs_only_adapter_feeds_an_onnx_source_without_torch(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.onnx"
    onnx.save(_conv_bn_model(dynamic=True), source)
    adapter = tmp_path / "adapter.py"
    adapter.write_text('''import numpy
def example_inputs():
    return {"x": numpy.full((2, 1, 2, 2), 0.75, dtype=numpy.float32)}
''')
    real_import = prepare_module._import

    def no_torch(name):
        if name == "torch":
            raise prepare_module.PreparationError("torch is not available")
        return real_import(name)

    monkeypatch.setattr(prepare_module, "_import", no_torch)
    destination = tmp_path / "prepared"
    metadata = prepare_model(source, destination, adapter=adapter)

    assert metadata["verification"]["ok"] is True
    assert "pytorch_vs_original" not in metadata["verification"]
    assert metadata["adapter"]["input_names"] == ["x"]
    tensor = numpy_helper.to_array(onnx.load_tensor(str(destination / "reference/input_0.pb")))
    assert tensor.shape == (2, 1, 2, 2) and (tensor == 0.75).all()


def test_an_adapter_without_example_inputs_is_refused(tmp_path: Path):
    source = tmp_path / "source.onnx"
    onnx.save(_conv_bn_model(), source)
    adapter = tmp_path / "adapter.py"
    adapter.write_text("def load_model():\n    return None\n")
    with pytest.raises(prepare_module.PreparationError, match="example_inputs"):
        prepare_model(source, tmp_path / "prepared", adapter=adapter)


def test_an_optimizer_that_changes_the_outputs_is_rejected(tmp_path: Path, monkeypatch):
    class Optimizer:
        @staticmethod
        def optimize(model, **_):
            changed = onnx.ModelProto()
            changed.CopyFrom(model)
            weight = next(init for init in changed.graph.initializer if init.name == "w")
            weight.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(weight) * 1.5, "w"))
            return changed

    real_import = prepare_module._import
    monkeypatch.setattr(prepare_module, "_import",
                        lambda name: type("onnxscript", (), {"optimizer": Optimizer})
                        if name == "onnxscript" else real_import(name))
    source = tmp_path / "source.onnx"
    onnx.save(_conv_bn_model(), source)
    destination = tmp_path / "prepared"
    metadata = prepare_model(source, destination)

    assert metadata["verification"]["ok"] is True
    optimizer = next(t for t in metadata["transformations"] if t["name"] == "onnxscript_optimize")
    assert optimizer["status"] == "rejected"
    assert optimizer["reason"].startswith("outputs differ from the original model")
    candidate = onnx.load(destination / "model.onnx")
    # The Conv–BN fold is still applied; only the optimizer's change is gone.
    assert [node.op_type for node in candidate.graph.node] == ["Conv"]
    weight = next(init for init in candidate.graph.initializer if init.name == "w")
    assert numpy_helper.to_array(weight).item() == pytest.approx(2 / numpy.sqrt(0.25 + 1e-5), rel=1e-6)
