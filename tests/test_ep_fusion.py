"""Run against a built Vulkan plugin with ONNX_VULKAN_TEST_EP set to its path."""

import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper


def test_fusion_respects_implicit_if_input(tmp_path, monkeypatch):
    plugin = os.environ.get("ONNX_VULKAN_TEST_EP")
    if not plugin:
        pytest.skip("set ONNX_VULKAN_TEST_EP to the built Vulkan plugin")

    def tensor(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [4])

    branch = helper.make_graph(
        [helper.make_node("Identity", ["activated"], ["branch_output"])],
        "branch", [], [tensor("branch_output")],
    )
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["x"], ["activated"]),
            helper.make_node(
                "If", ["condition"], ["selected"], then_branch=branch, else_branch=branch,
            ),
            helper.make_node("Mul", ["activated", "selected"], ["y"]),
        ],
        "implicit_if_input", [tensor("x")], [tensor("y")],
        [helper.make_tensor("condition", TensorProto.BOOL, [], [True])],
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 19)])
    onnx.checker.check_model(model)
    model_bytes = model.SerializeToString()
    feed = {"x": np.array([-2, -1, 1, 2], dtype=np.float32)}

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    cpu = ort.InferenceSession(model_bytes, sess_options=options, providers=["CPUExecutionProvider"])
    (expected,) = cpu.run(None, feed)
    del cpu

    monkeypatch.setenv("VULKAN_EP_COMPILE", "1")
    monkeypatch.setenv("VULKAN_EP_OPS", "Relu,Mul")
    ort.register_execution_provider_library("VulkanEP", str(Path(plugin).resolve()))
    try:
        devices = [device for device in ort.get_ep_devices() if device.ep_name == "VulkanEP"]
        assert devices, "the Vulkan plugin did not expose a Vulkan device"
        options.add_provider_for_devices(devices[:1], {})
        options.enable_profiling = True
        options.profile_file_prefix = str(tmp_path / "fusion")
        session = ort.InferenceSession(model_bytes, sess_options=options)
        try:
            (actual,) = session.run(None, feed)
            np.testing.assert_array_equal(actual, expected)
            profile = Path(session.end_profiling())
        finally:
            del session
    finally:
        del options
        ort.unregister_execution_provider_library("VulkanEP")

    events = json.loads(profile.read_text())
    # The profile also attributes host-to-device copies to the Vulkan EP.
    vulkan_partitions = [
        event for event in events
        if event.get("cat") == "Node"
        and event.get("name", "").endswith("_kernel_time")
        and event.get("args", {}).get("provider") == "VulkanEP"
        and event["args"].get("op_name", "").startswith("VulkanEP_")
    ]
    assert len(vulkan_partitions) == 2, "expected separate Vulkan partitions on either side of If"
