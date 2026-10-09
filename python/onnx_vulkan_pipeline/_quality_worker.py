"""Evaluate the declared application metric on measured Vulkan outputs.

The adapter sees a session with run(output_names, input_feed). It may only
request the recorded evaluation inputs; any unmeasured inference is rejected.
This module runs in a bounded subprocess because adapters are user code.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import numpy_helper

from .pipeline import write_json
from .quantize import _adapter_metric, _load_adapter, _sample_digest


def _values(item):
    """A recorded output: inline JSON values, or a .npy file the pipeline wrote."""
    if "file" in item:
        values = np.load(item["file"], allow_pickle=False)
        if values.dtype != np.dtype(item["dtype"]) or list(values.shape) != list(item["shape"]):
            raise ValueError(f"Recorded output {item['name']} does not match its dtype/shape")
        return values
    return np.asarray(item["values"], dtype=item["dtype"]).reshape(item["shape"])


class RecordedSession:
    def __init__(self, records):
        self.outputs = {}
        self.samples = []
        for record in records:
            sample = {}
            for path in sorted(Path(record["reference"]).glob("input_*.pb")):
                tensor = onnx.load_tensor(str(path))
                sample[tensor.name] = numpy_helper.to_array(tensor)
            result = record["result"]
            if result["backend"] not in ("standalone", "vulkan-ep"):
                raise ValueError("Quality evaluation requires Vulkan outputs")
            outputs = {item["name"]: _values(item) for item in result["outputs"]}
            self.outputs[_sample_digest(sample)] = outputs
            self.samples.append(sample)

    def run(self, output_names, input_feed, run_options=None):
        outputs = self.outputs.get(_sample_digest(input_feed))
        if outputs is None:
            raise ValueError("Quality adapter requested inputs without measured Vulkan outputs")
        return [outputs[name].copy() for name in (output_names if output_names is not None else outputs)]


def evaluate(directory, records):
    directory = Path(directory)
    quantization = json.loads((directory / "preparation.json").read_text())["quantization"]
    session = RecordedSession(records)
    if not session.samples:
        raise ValueError("Quality evaluation requires measured samples")
    adapter = _load_adapter(directory / "quality_adapter.py")
    after = _adapter_metric(adapter, session, session.samples)
    before = quantization["before"]
    drop = before - after if quantization["higher_is_better"] else after - before
    if drop > quantization["max_drop"]:
        raise ValueError(f"Vulkan application quality regression {drop} exceeds {quantization['max_drop']}")
    return {"samples": len(session.samples), "vulkan_verified": True,
            "metric": quantization["metric"], "before": before, "after": after,
            "drop": drop, "max_drop": quantization["max_drop"],
            "backend": records[0]["result"]["backend"]}


if __name__ == "__main__":
    config = json.loads(Path(sys.argv[1]).read_text())
    write_json(config["response"], evaluate(config["directory"], config["records"]))
