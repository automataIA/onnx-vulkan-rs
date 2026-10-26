"""Evaluate the declared application metric on measured Vulkan outputs.

The adapter sees a session with run(output_names, input_feed). It may only
request the recorded evaluation inputs; any unmeasured inference is rejected.
This module runs in a bounded subprocess because adapters are user code.

The contract comes from ``quality.json`` or, for older bundles, from
``preparation.json["quantization"]`` (see ``quality.load_contract``). A second
mode, ``generate_reference``, writes CPU outputs of a given model on the
contract's inputs as ``model-runner`` references; it never feeds the metric.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import numpy_helper

from .pipeline import write_json
from .quality import ADAPTER_FILE, _check_input_names, _cpu_session, contract_identity, load_contract
from .quantize import (_adapter_metric, _load_adapter, _reference_feeds, _sample_digest,
                       _write_quality_fixtures)


def _values(item):
    """A recorded output: inline JSON values, or a .npy file the pipeline wrote."""
    if "file" in item:
        values = np.load(item["file"], allow_pickle=False)
        if values.dtype != np.dtype(item["dtype"]) or list(values.shape) != list(item["shape"]):
            raise ValueError(f"Recorded output {item['name']} does not match its dtype/shape")
        return values
    return np.asarray(item["values"], dtype=item["dtype"]).reshape(item["shape"])


class RecordedSession:
    """Serves only recorded Vulkan outputs; there is no inference behind it."""

    def __init__(self, records):
        self.outputs = {}
        self.samples = []
        self.backend = None
        self._requested = set()
        for record in records:
            sample = {}
            for path in sorted(Path(record["reference"]).glob("input_*.pb")):
                tensor = onnx.load_tensor(str(path))
                sample[tensor.name] = numpy_helper.to_array(tensor)
            if not sample:
                raise ValueError("Quality sample has no recorded inputs")
            result = record["result"]
            if result["backend"] not in ("standalone", "vulkan-ep"):
                raise ValueError("Quality evaluation requires Vulkan outputs")
            if self.backend not in (None, result["backend"]):
                raise ValueError("Quality records mix backends "
                                 f"({self.backend} and {result['backend']}); measure one backend")
            self.backend = result["backend"]
            if not result["outputs"]:
                raise ValueError("Quality record has no measured outputs")
            outputs = {item["name"]: _values(item) for item in result["outputs"]}
            for name, values in outputs.items():
                if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
                    raise ValueError(f"Measured output {name} is not finite")
            key = _sample_digest(sample)
            previous = self.outputs.get(key)
            if previous is not None and (previous.keys() != outputs.keys() or any(
                    not np.array_equal(previous[name], outputs[name]) for name in outputs)):
                raise ValueError("Duplicate quality inputs have conflicting measured outputs")
            self.outputs[key] = outputs
            self.samples.append(sample)

    def run(self, output_names, input_feed, run_options=None):
        key = _sample_digest(input_feed)
        outputs = self.outputs.get(key)
        if outputs is None:
            raise ValueError("Quality adapter requested inputs without measured Vulkan outputs")
        self._requested.add(key)
        names = list(output_names) if output_names is not None else list(outputs)
        missing = [name for name in names if name not in outputs]
        if missing:
            raise ValueError(f"Quality adapter requested unmeasured outputs: {', '.join(missing)}")
        return [outputs[name].copy() for name in names]

    def require_all_requested(self):
        if {_sample_digest(sample) for sample in self.samples} - self._requested:
            raise ValueError("Quality adapter did not evaluate every measured sample")


def evaluate(directory, records):
    directory = Path(directory)
    contract = load_contract(directory)
    if contract is None:
        raise ValueError("Profile has no quality contract")
    session = RecordedSession(records)
    if not session.samples:
        raise ValueError("Quality evaluation requires measured samples")
    # Every contract sample, exactly once: a subset or foreign inputs would
    # make `samples` and the score describe another dataset.
    if sorted(_sample_digest(sample) for sample in session.samples) != sorted(contract["content_hashes"]):
        raise ValueError("Measured samples differ from the evaluation samples of the quality contract")
    adapter = _load_adapter(directory / ADAPTER_FILE)
    after = _adapter_metric(adapter, session, session.samples)
    session.require_all_requested()
    before = contract["reference_score"]
    drop = before - after if contract["higher_is_better"] else after - before
    if drop > contract["max_drop"]:
        raise ValueError(f"Vulkan application quality regression {drop} exceeds {contract['max_drop']}")
    return {"samples": len(session.samples), "measured": True, "vulkan_verified": True,
            "metric": contract["metric"], "before": before, "after": after,
            "drop": drop, "max_drop": contract["max_drop"],
            "higher_is_better": contract["higher_is_better"],
            "backend": session.backend, "contract": contract_identity(contract)}


def generate_reference(model, samples, destination):
    """CPU outputs of ``model`` on the inputs of the ``samples`` directories."""
    feeds = [_reference_feeds(Path(sample)) for sample in samples]
    session = _cpu_session(Path(model))
    _check_input_names(feeds, session, "selected")
    outputs = [session.run(None, feed) for feed in feeds]
    _write_quality_fixtures(Path(destination), feeds, outputs, [item.name for item in session.get_outputs()])
    return len(feeds)


if __name__ == "__main__":
    config = json.loads(Path(sys.argv[1]).read_text())
    if config.get("mode") == "generate_reference":
        write_json(config["response"], {"samples": generate_reference(
            config["model"], config["samples"], config["destination"])})
    else:
        write_json(config["response"], evaluate(config["directory"], config["records"]))
