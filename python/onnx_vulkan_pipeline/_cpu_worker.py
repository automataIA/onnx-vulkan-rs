"""Private process boundary for CPU optimization: oracle generation and paired timing.

Both modes run ONNX Runtime's CPU provider only; nothing here touches Vulkan.
The parent passes a JSON config and reads the JSON result at ``config["output"]``
(``oracle`` writes into the profile directory instead).
"""
import gc
import json
from pathlib import Path
import sys
import time


def _session(ort, directory, setting):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, setting["graph_optimization_level"])
    options.execution_mode = getattr(ort.ExecutionMode, setting["execution_mode"])
    # 0 leaves ONNX Runtime's own default, which is what a plain deployment gets.
    options.intra_op_num_threads = setting["intra_op_num_threads"]
    if "inter_op_num_threads" in setting:
        options.inter_op_num_threads = setting["inter_op_num_threads"]
    model = directory / ("original.onnx" if setting["model"] == "original" else "model.onnx")
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


def _reference_inputs(directory):
    import onnx
    from onnx import numpy_helper

    paths = sorted((directory / "reference").glob("input_*.pb"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise RuntimeError(f"no recorded inputs in {directory / 'reference'}")
    tensors = []
    for path in paths:
        tensor = onnx.TensorProto()
        tensor.ParseFromString(path.read_bytes())
        tensors.append((tensor.name, numpy_helper.to_array(tensor)))
    return tensors


def oracle(config):
    """Reference outputs of the original model: optimizations disabled, one thread."""
    import numpy
    import onnxruntime as ort

    directory = Path(config["directory"])
    inputs = _reference_inputs(directory)
    setting = {"model": "original", "graph_optimization_level": "ORT_DISABLE_ALL",
               "execution_mode": "ORT_SEQUENTIAL", "intra_op_num_threads": 1, "inter_op_num_threads": 1}
    session = _session(ort, directory, setting)
    expected = [meta.name for meta in session.get_inputs()]
    if [name for name, _ in inputs] != expected:
        raise RuntimeError(f"recorded inputs {[n for n, _ in inputs]} do not match model inputs {expected}")
    outputs = [numpy.asarray(value) for value in session.run(None, {name: array for name, array in inputs})]
    numpy.savez(directory / "feeds.npz", **{f"i{i}": array for i, (_, array) in enumerate(inputs)})
    numpy.savez(directory / "oracle.npz", **{f"o{i}": array for i, array in enumerate(outputs)})
    metadata = {
        "input_names": expected,
        "output_names": [meta.name for meta in session.get_outputs()],
        "outputs": [{"dtype": str(array.dtype), "shape": list(array.shape)} for array in outputs],
        "setting": setting,
    }
    # Written last: its presence marks a complete profile directory.
    (directory / "oracle.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _load(directory):
    import numpy

    metadata = json.loads((directory / "oracle.json").read_text())
    with numpy.load(directory / "feeds.npz") as stored:
        feeds = {name: stored[f"i{i}"] for i, name in enumerate(metadata["input_names"])}
    with numpy.load(directory / "oracle.npz") as stored:
        expected = [stored[f"o{i}"] for i in range(len(metadata["outputs"]))]
    return metadata, feeds, expected


def _divergence(numpy, outputs, expected, atol, rtol):
    """Return (failure or None, max_abs, max_rel) of one run against the oracle."""
    max_abs = max_rel = 0.0
    if len(outputs) != len(expected):
        return f"output count {len(outputs)} != {len(expected)}", max_abs, max_rel
    for index, (actual, wanted) in enumerate(zip(outputs, expected)):
        actual = numpy.asarray(actual)
        if actual.shape != wanted.shape:
            return f"output {index} shape {list(actual.shape)} != {list(wanted.shape)}", max_abs, max_rel
        if actual.dtype != wanted.dtype:
            return f"output {index} dtype {actual.dtype} != {wanted.dtype}", max_abs, max_rel
        if numpy.issubdtype(wanted.dtype, numpy.floating):
            if not (numpy.all(numpy.isfinite(actual)) and numpy.all(numpy.isfinite(wanted))):
                return f"output {index} is not finite", max_abs, max_rel
            left, right = actual.astype(numpy.float64), wanted.astype(numpy.float64)
            diff = numpy.abs(left - right)
            if diff.size:
                max_abs = max(max_abs, float(diff.max()))
                max_rel = max(max_rel, float((diff / numpy.maximum(numpy.abs(right), 1e-12)).max()))
            if not numpy.all(diff <= atol + rtol * numpy.abs(right)):
                return f"output {index} beyond tolerance (max|Δ|={max_abs:.3g})", max_abs, max_rel
        elif not numpy.array_equal(actual, wanted):
            return f"output {index} differs exactly", max_abs, max_rel
    return None, max_abs, max_rel


def measure(config):
    """Interleaved AB/BA timing of ``baseline`` against ``candidate`` on one profile.

    Only ``session.run`` sits between the two ``perf_counter`` reads: session
    creation and warmup happen before, and outputs are compared with the oracle
    after both runs of a pair, so every timed sample is also a checked sample.
    """
    import numpy
    import onnxruntime as ort

    directory = Path(config["directory"])
    metadata, feeds, expected = _load(directory)
    atol, rtol = float(config["atol"]), float(config["rtol"])
    sides = {"baseline": config["baseline"], "candidate": config["candidate"]}
    sessions, setup_ms = {}, {}
    for side, setting in sides.items():
        start = time.perf_counter()
        sessions[side] = _session(ort, directory, setting)
        setup_ms[side] = (time.perf_counter() - start) * 1e3
    result = {"baseline_ms": [], "candidate_ms": [], "orders": [], "setup_ms": setup_ms,
              "correct": True, "baseline_correct": True, "failure": None,
              "max_abs": 0.0, "max_rel": 0.0, "ort_version": ort.__version__}
    for side, session in sessions.items():
        names = [meta.name for meta in session.get_outputs()]
        if names != metadata["output_names"]:
            result["baseline_correct" if side == "baseline" else "correct"] = False
            result["failure"] = f"{side} output names {names} != {metadata['output_names']}"
            return result
    for _ in range(int(config["warmup"])):
        for session in sessions.values():
            session.run(None, feeds)
    gc.collect()
    gc.disable()
    try:
        for index in range(int(config["samples"])):
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            outputs = {}
            for side in order:
                start = time.perf_counter()
                outputs[side] = sessions[side].run(None, feeds)
                result[f"{side}_ms"].append((time.perf_counter() - start) * 1e3)
            result["orders"].append("AB" if index % 2 == 0 else "BA")
            for side in ("baseline", "candidate"):
                failure, max_abs, max_rel = _divergence(numpy, outputs[side], expected, atol, rtol)
                if side == "candidate":
                    result["max_abs"], result["max_rel"] = max(result["max_abs"], max_abs), max(result["max_rel"], max_rel)
                if failure:
                    result["baseline_correct" if side == "baseline" else "correct"] = False
                    result["failure"] = f"{side} sample {index}: {failure}"
                    return result
    finally:
        gc.enable()
    return result


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    if config["mode"] == "oracle":
        oracle(config)
    elif config["mode"] == "measure":
        Path(config["output"]).write_text(json.dumps(measure(config)))
    else:
        raise ValueError(f"unknown mode {config['mode']!r}")


if __name__ == "__main__":
    main()
