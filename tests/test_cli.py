"""Human-facing summaries printed after the JSON results."""
from argparse import Namespace
import subprocess

from onnx_vulkan_pipeline import cli
from onnx_vulkan_pipeline.pipeline import write_json


def test_a_divergence_ends_with_the_cause_the_offending_outputs_and_a_hint():
    error = RuntimeError(
        "model-runner exited 1: [INFO] flush 3/overhead ~0.059 ms\n"
        "Error: divergent outputs beyond atol=1.0e-5 + rtol=1.0e-4·|ref|\n"
        "  output output   n=1000  max|Δ|=3.052e-5  relative=2.78e-6  beyond tolerance: 4\n"
        "  standalone vs EP   output   n=1000  max|Δ|=0.000e0  beyond tolerance: 0\n")
    summary = cli.failure_summary(error)
    assert summary[0] == "Error: divergent outputs beyond atol=1.0e-5 + rtol=1.0e-4·|ref|"
    assert summary[1].endswith("beyond tolerance: 4")
    assert len(summary) == 3 and summary[2].startswith("hint: ") and "--atol" in summary[2]


def _model(path, shape):
    """Model with two inputs of the given shape, for the inspect summaries."""
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Add", ["ids", "mask"], ["y"])], "g",
        [helper.make_tensor_value_info("ids", TensorProto.FLOAT, shape),
         helper.make_tensor_value_info("mask", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    return path


def test_inspect_names_only_the_input_dimensions_to_choose(tmp_path):
    _model(tmp_path / "m.onnx", ["batch_size", "seq_len"])
    # Derived symbols in the issues are not something --dim can set.
    reason = "shape of `{}` is unresolved (symbolic dimension `floor(seq_len/2)` has no --dim value); runtime resolves it"
    result = {"compatible": True, "execution": {"supported_nodes": 3, "total_nodes": 3},
              "tuning": {"tunable_workloads": 2},
              "issues": [{"node": f"n{i}", "op": "Conv", "kind": "shape", "reason": reason.format(i)} for i in range(7)]}
    lines = cli.success_summary("inspect", Namespace(source=tmp_path / "m.onnx", dim=[]), result)
    assert lines[:4] == ["compatible: yes (3/3 nodes supported, tunable workloads counted only with every "
                         "--dim given)",
                         "  input ids: float [batch_size, seq_len]",
                         "  input mask: float [batch_size, seq_len]",
                         "  input dimensions without a value: batch_size, seq_len; prepare needs the size of your "
                         "real inputs for each, as --dim batch_size=<size> --dim seq_len=<size>"]
    assert lines[4].startswith("  7 node(s) have shapes known only at run time")
    lines = cli.success_summary("inspect", Namespace(source=tmp_path / "m.onnx", dim=["batch_size=1"]), result)
    assert lines[3].startswith("  input dimensions without a value: seq_len;")
    lines = cli.success_summary("inspect", Namespace(source=tmp_path / "m.onnx", dim=["batch_size=1", "seq_len=4"]),
                                result)
    assert lines[0] == "compatible: yes (3/3 nodes supported, 2 tunable workloads)"


def test_optimize_reports_the_medians_behind_its_decision(tmp_path):
    write_json(tmp_path / "report.json", {"default": {
        "status": "compatible_no_measured_gain", "tuned": True, "required_gain": 0.1,
        "baseline": {"median_wall_ms": 2.0, "median_gpu_ms": 1.2},
        "candidate": {"median_wall_ms": 1.9, "median_gpu_ms": None}}})
    lines = cli.success_summary("optimize", Namespace(output=tmp_path, no_tune=False), {})
    assert lines[:2] == ["profile default: compatible_no_measured_gain",
                         "  original model → prepared model with tuned kernels, both on Vulkan: "
                         "wall 2.000 ms → 1.900 ms, GPU 1.200 ms → n/a (promotion needs ≥10% on both, beyond noise)"]
    assert "`run` compares Vulkan with the CPU" in lines[2]


def test_optimize_says_why_it_did_not_tune(tmp_path):
    write_json(tmp_path / "report.json", {"default": {
        "status": "compatible_no_measured_gain", "tuned": False, "required_gain": 0.1,
        "baseline": {"median_wall_ms": 2.0, "median_gpu_ms": 1.2},
        "candidate": {"median_wall_ms": 1.9, "median_gpu_ms": 1.1}}})
    lines = cli.success_summary("optimize", Namespace(output=tmp_path, no_tune=False), {})
    assert lines[0] == "profile default: compatible_no_measured_gain (no tuning: the model has no tunable workloads)"
    assert "prepared model with default kernels" in lines[1]
    lines = cli.success_summary("optimize", Namespace(output=tmp_path, no_tune=True), {})
    assert lines[0].endswith("(no tuning: --no-tune)")


def test_optimize_says_when_it_dropped_the_tuned_kernels(tmp_path):
    write_json(tmp_path / "report.json", {"default": {
        "status": "compatible_no_measured_gain", "tuned": True, "required_gain": 0.1,
        "tactics_rejected": "Error: divergent outputs", "stopped": "identical"}})
    lines = cli.success_summary("optimize", Namespace(output=tmp_path, no_tune=False), {})
    assert lines == ["profile default: compatible_no_measured_gain",
                     "  tuned kernels rejected: on the bundle's reference inputs they miss its tolerance "
                     "(Error: divergent outputs); default kernels kept",
                     "  not measured: the prepared model is the original one, with default kernels",
                     f"next: uv run onnx-vulkan run {tmp_path}"]


def test_a_gross_divergence_is_not_blamed_on_the_tolerance():
    error = RuntimeError(
        "model-runner exited 1:\n"
        "Error: divergent outputs beyond atol=1.0e-4 + rtol=1.0e-4·|ref|\n"
        "  output output   n=1000  max|Δ|=1.853e1  |ref|max=6.459e0  relative=2.87e0  beyond tolerance: 1000\n")
    summary = cli.failure_summary(error)
    assert summary[1].endswith("beyond tolerance: 1000")
    assert summary[-1].startswith("hint: differences this large are wrong results")
    assert not any("prepare again" in line for line in summary)


def test_run_compares_vulkan_with_the_optimized_cpu_ep_when_it_was_timed(tmp_path):
    (tmp_path / "default").mkdir()
    write_json(tmp_path / "default" / "preparation.json", {"adapter": None})
    args = Namespace(source=tmp_path, profile="default")
    result = {"atol": 1e-4, "rtol": 1e-4, "cpu_ms": [50, 9.0, 9.0, 9.0], "executed_backends": ["ort-cpu", "vulkan-ep"],
              "cpu_optimized_ms": [40, 1.5, 1.4, 1.6], "candidate_ms": [30, 1.9, 1.9, 1.9],
              "outputs": [{"max_abs": 6.7e-4, "max_relative": None, "max_tolerance_ratio": 0.07},
                          {"max_abs": 0, "max_tolerance_ratio": 0.0}]}
    lines = cli.success_summary("run", args, result)
    assert lines[0] == ("vulkan-ep outputs match the CPU reference within atol=0.0001, rtol=0.0001: every element "
                        "has |Δ| ≤ atol + rtol·|reference|, the worst at 7% of that allowance")
    assert lines[1].startswith("on deterministic generated inputs; ") and "not application quality" in lines[1]
    assert lines[2] == "validation timings, not a benchmark: CPU EP 1.50 ms, Vulkan EP 1.90 ms"
    del result["cpu_optimized_ms"]
    assert "CPU EP without graph optimizations 9.00 ms" in cli.success_summary("run", args, result)[2]
    result.update(gate="parity", executed_backends=["ort-cpu", "vulkan-ep", "standalone"],
                  outputs=[{"max_abs": 15.98}, {"max_abs": 0.5}])
    assert cli.success_summary("run", args, result)[0].startswith(
        "vulkan-ep and standalone outputs are identical; the CPU EP differs by up to max|Δ|=16, ")


def test_quantize_search_lists_every_candidate_with_its_outcome(tmp_path):
    result = {"selected": "int8", "optimized": None, "candidates": [
        {"name": "int8", "status": "passed", "median_wall_ms": 1.814, "vulkan_drop": 0.0},
        {"name": "int4", "status": "rejected", "reason": "no operator was quantized: " + "x" * 200}]}
    lines = cli.success_summary("quantize-search", Namespace(output=tmp_path), result)
    assert lines[0] == "  int8: passed — standalone 1.81 ms, Vulkan metric drop 0"
    assert lines[1].startswith("  int4: rejected — no operator was quantized") and lines[1].endswith("…")
    assert lines[2] == "selected candidate: int8"


def test_inspect_folds_shapes_known_only_at_run_time_into_one_line(tmp_path):
    reason = "shape of `472` is unresolved; runtime resolves it from concrete tensors"
    result = {"compatible": True, "execution": {"supported_nodes": 2, "total_nodes": 2},
              "tuning": {"tunable_workloads": 1},
              "issues": [{"node": node, "op": op, "kind": "shape", "reason": reason}
                         for node, op in (("Reshape_103", "Reshape"), ("Gemm_104", "Gemm"))]}
    lines = cli.success_summary("inspect", Namespace(source=_model(tmp_path / "m.onnx", [1, 4]), dim=[]), result)
    assert lines[3:] == ["  2 node(s) have shapes known only at run time (Gemm_104, Reshape_103): "
                         "informational, prepare checks them on real inputs"]


def test_inspect_groups_unsupported_nodes_by_operator_and_says_the_model_cannot_run(tmp_path):
    reason = "operator or operator form is not implemented by the interpreter"
    result = {"compatible": False, "execution": {"supported_nodes": 2, "total_nodes": 5},
              "tuning": {"tunable_workloads": 0},
              "issues": [{"node": str(i), "op": op, "kind": "execution", "blocking": True, "reason": reason}
                         for i, op in enumerate(["NonZero", "RoiAlign", "NonZero"])]}
    lines = cli.success_summary("inspect", Namespace(source=_model(tmp_path / "m.onnx", [1, 4]), dim=[]), result)
    assert lines[3:] == ["  not implemented on Vulkan, as operators or in these forms: NonZero ×2, RoiAlign ×1",
                         "  so the model cannot run on Vulkan end to end: there is no CPU fallback, and `run` and "
                         "`optimize` refuse it"]


def test_an_argmax_change_is_not_blamed_on_the_tolerance():
    error = RuntimeError(
        "model-runner exited 1:\n"
        "  output logits   n=1000  max|Δ|=1.2e-5  relative=1.1e-6  beyond tolerance: 0\n"
        "  reference vulkan    logits   max|Δ|=1.2e-5  |ref|max=1.1e1  beyond tolerance: 0  argmax Some(3)→Some(7)  (different answer)\n"
        "Error: divergent outputs: values beyond atol=1.0e-5 + rtol=1.0e-4·|ref|, or a different argmax (see the lines above)\n")
    summary = cli.failure_summary(error)
    assert summary[1].endswith("(different answer)")
    assert summary[-1].startswith("hint: every value is within tolerance")
    assert not any("--atol" in line for line in summary)


def test_prepare_says_when_it_rejected_the_optimizer(tmp_path):
    (tmp_path / "default").mkdir()
    write_json(tmp_path / "default" / "preparation.json", {"transformations": [
        {"name": "conv_batchnorm_fold", "status": "applied"},
        {"name": "onnxscript_optimize", "status": "rejected", "reason": "outputs differ"}]})
    result = {"compatibility_obtained": True, "profiles": {"default": {}}}
    lines = cli.success_summary("prepare", Namespace(output=tmp_path, adapter=None), result)
    assert lines[2].startswith("onnxscript_optimize rejected (default): it changed the CPU outputs")
    assert lines[3].startswith("next: uv run onnx-vulkan run ")


def test_prepare_says_which_operators_make_a_bundle_unrunnable(tmp_path):
    for profile, compatible in (("small", True), ("large", False)):
        (tmp_path / profile).mkdir()
        write_json(tmp_path / profile / "preparation.json", {})
        write_json(tmp_path / profile / "compatibility.json", {"compatible": compatible, "issues": [
            {"op": "NonZero", "kind": "execution", "blocking": True}] if not compatible else []})
    result = {"compatibility_obtained": False, "profiles": {"small": {}, "large": {}}}
    lines = cli.success_summary("prepare", Namespace(output=tmp_path, adapter=None), result)
    assert lines[2:] == [f"`run` and `optimize` refuse profile large: Vulkan does not implement NonZero ×1; "
                         f"details in {tmp_path / 'large' / 'compatibility.json'}"]


def test_a_failure_with_evidence_prints_the_verdict_and_the_log_path(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        error = RuntimeError("model-runner exited 1: [INFO] stats line\n" * 50
                             + "  output y   n=4  max|Δ|=3.0e-4  relative=1.0e-4  beyond tolerance: 2\n"
                             + "Error: divergent outputs: values beyond atol=1.0e-5 + rtol=1.0e-4·|ref|\n")
        error.add_note(f"Run evidence retained at {tmp_path / '.b.run-failed'}")
        raise error

    monkeypatch.setattr(cli, "run_bundle", fail)
    monkeypatch.setattr("sys.argv", ["onnx-vulkan", "run", str(tmp_path / "b")])
    assert cli.main() == 1
    err = capsys.readouterr().err
    assert "stats line" not in err
    assert err.splitlines()[0] == "onnx-vulkan: run failed: Error: divergent outputs: values beyond atol=1.0e-5 + rtol=1.0e-4·|ref|"
    assert err.splitlines()[-1] == f"onnx-vulkan: full output: {tmp_path / '.b.run-failed' / 'error.log'}"


def test_a_timeout_names_the_stage_and_the_limit_not_the_command_line(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        error = subprocess.TimeoutExpired(["/repo/target/release/model-runner", "/long/model.onnx", "--iters"], 300)
        error.add_note(f"Run evidence retained at {tmp_path / '.b.run-failed'}")
        error.add_note("Stage: the per-node check, which exposes the quantized tensors as outputs")
        raise error

    monkeypatch.setattr(cli, "run_bundle", fail)
    monkeypatch.setattr("sys.argv", ["onnx-vulkan", "run", str(tmp_path / "b")])
    assert cli.main() == 1
    err = capsys.readouterr().err.splitlines()
    assert err[0] == ("onnx-vulkan: run failed during the per-node check, which exposes the quantized tensors "
                      "as outputs: model-runner did not finish within 300 s")
    assert err[1].startswith("onnx-vulkan: hint: raise --timeout") and err[1].endswith("fewer --samples shorten `run`")
    assert err[2] == f"onnx-vulkan: evidence: {tmp_path / '.b.run-failed'}"
    assert "--iters" not in "\n".join(err)


def test_run_says_what_gates_an_already_quantized_source(tmp_path):
    (tmp_path / "default").mkdir()
    write_json(tmp_path / "default" / "preparation.json", {"adapter": None})
    result = {"atol": 1e-5, "rtol": 1e-4, "gate": "parity", "dynamic_quantization": "source",
              "cpu_ms": [9.0, 9.0], "candidate_ms": [2.0, 2.0], "standalone_ms": [2.0, 2.0],
              "executed_backends": ["ort-cpu", "vulkan-ep", "standalone"], "outputs": [{"max_abs": 3.3}],
              "quantized_tensors": {"total": 9, "identical": 4, "tensor": "q", "max_abs": 1.0, "passed": True}}
    lines = cli.success_summary("run", Namespace(source=tmp_path, profile="default"), result)
    assert lines[0].endswith("so the quantized tensors below are the gate")
    assert lines[1] == ("quantized tensors, in graph order: 4 of 9 identical to the CPU EP; the first that differs, "
                        "q, is within ±1 code (max|Δ|=1)")
    assert lines[2].startswith("application quality not measured: the source was already quantized")


def test_a_model_that_cannot_run_on_vulkan_exits_2_after_a_successful_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "success_summary", lambda *args: [])
    for compatible, status in ((False, 2), (True, 0)):
        monkeypatch.setattr(cli, "inspect_model", lambda *args, **kwargs: {"compatible": compatible})
        monkeypatch.setattr(cli, "prepare_bundle", lambda *args, **kwargs: {"compatibility_obtained": compatible})
        monkeypatch.setattr("sys.argv", ["onnx-vulkan", "inspect", str(tmp_path / "m.onnx")])
        assert cli.main() == status
        monkeypatch.setattr("sys.argv", ["onnx-vulkan", "prepare", str(tmp_path / "m.onnx"),
                                         "--output", str(tmp_path / "b")])
        assert cli.main() == status
    assert capsys.readouterr().out.count('"compatib') == 4  # the JSON still comes first


def test_inspect_prints_each_issue_once_and_counts_the_runtime_shapes():
    unsupported = "operator or operator form is not implemented by the interpreter"
    result = {"compatible": False, "issues": [
        {"node": "a", "op": "NonZero", "kind": "execution", "reason": unsupported, "blocking": True},
        {"node": "b", "op": "NonZero", "kind": "execution", "reason": unsupported, "blocking": True},
        {"node": "c", "op": "Gather", "kind": "shape", "reason": "shape of `x` is unresolved", "blocking": False},
        {"node": "c", "op": "Gather", "kind": "shape", "reason": "shape of `y` is unresolved", "blocking": False},
        {"node": "d", "op": "Slice", "kind": "shape", "reason": "shape of `x` is unresolved", "blocking": False}]}
    printed = cli._inspect_json(result)
    assert printed["issues"] == [{"kind": "execution", "op": "NonZero", "reason": unsupported, "blocking": True,
                                  "nodes": ["a", "b"]}]
    assert printed["runtime_shape_nodes"] == 2
    assert len(result["issues"]) == 5


def _decoder(path):
    """Decoder inputs with a KV cache, as Gemma exports them."""
    import onnx
    from onnx import TensorProto, helper

    cache = ["batch_size", 1, "past_sequence_length", 8]
    inputs = [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
              helper.make_tensor_value_info("attention_mask", TensorProto.INT64,
                                            ["batch_size", "total_sequence_length"])]
    inputs += [helper.make_tensor_value_info(f"past_key_values.{layer}.{part}", TensorProto.FLOAT, cache)
               for layer in range(3) for part in ("key", "value")]
    graph = helper.make_graph([helper.make_node("Identity", ["input_ids"], ["y"])], "g", inputs,
                              [helper.make_tensor_value_info("y", TensorProto.INT64, ["batch_size", "sequence_length"])])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    return path


def test_inspect_folds_numbered_inputs_and_explains_a_kv_cache(tmp_path):
    _decoder(tmp_path / "d.onnx")
    result = {"compatible": True, "execution": {"supported_nodes": 1, "total_nodes": 1}, "issues": []}
    lines = cli.success_summary("inspect", Namespace(source=tmp_path / "d.onnx", dim=[]), result)
    assert lines[1:5] == [
        "  input input_ids: int64 [batch_size, sequence_length]",
        "  input attention_mask: int64 [batch_size, total_sequence_length]",
        "  3 inputs past_key_values.#.key: float [batch_size, 1, past_sequence_length, 8] (# from 0 to 2)",
        "  3 inputs past_key_values.#.value: float [batch_size, 1, past_sequence_length, 8] (# from 0 to 2)"]
    assert lines[6] == ("  a decoder with a KV cache: total_sequence_length = past_sequence_length + sequence_length, "
                        "all ≥ 1 (a past of 0 is a zero-size tensor), e.g. --dim batch_size=1 --dim sequence_length=4 "
                        "--dim total_sequence_length=5 --dim past_sequence_length=1")


def test_prepare_checks_the_dimensions_before_leaving_any_directory(tmp_path, monkeypatch, capsys):
    _decoder(tmp_path / "d.onnx")
    for dims, expected in (
            (["batch_size=1"], "onnx-vulkan: input dimensions without a value: sequence_length, "
                               "total_sequence_length, past_sequence_length; pass the size of your real inputs for "
                               "each, a decoder with a KV cache: "),
            (["batch_size=1", "sequence_length=4", "total_sequence_length=4", "past_sequence_length=0"],
             "onnx-vulkan: --dim past_sequence_length=0: sizes must be positive integers; "
             "the runtime has no zero-size tensors")):
        monkeypatch.setattr("sys.argv", ["onnx-vulkan", "prepare", str(tmp_path / "d.onnx"),
                                         "--output", str(tmp_path / "b"), *(f"--dim={d}" for d in dims)])
        assert cli.main() == 1
        assert capsys.readouterr().err.startswith(expected)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["d.onnx"]


def test_a_failure_line_drops_the_exception_class_path():
    assert cli._without_class("onnx_vulkan_pipeline.prepare.VerificationError: candidate verification did not "
                              "pass: x") == "candidate verification did not pass: x"
    assert cli._without_class("RuntimeError: x") == "RuntimeError: x"


def test_quantized_bundles_say_what_was_quantized_and_how_the_metric_compares(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("QuantizeLinear", ["x", "s", "z"], ["q"]),
         helper.make_node("QLinearConv", ["q", "s", "z", "w", "s", "z", "s", "z"], ["c"]),
         helper.make_node("QLinearConv", ["c", "s", "z", "w", "s", "z", "s", "z"], ["d"]),
         helper.make_node("DequantizeLinear", ["d", "s", "z"], ["y"]), helper.make_node("Relu", ["y"], ["r"])],
        "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("r", TensorProto.FLOAT, [1])])
    (tmp_path / "default").mkdir()
    onnx.save(helper.make_model(graph), tmp_path / "default" / "model.onnx")
    write_json(tmp_path / "default" / "preparation.json", {"adapter": None, "transformations": [], "quantization": {
        "mode": "static", "format": "QOperator", "metric": "mAP", "before": 0.5, "after": 0.48, "drop": 0.02,
        "max_drop": 0.01, "higher_is_better": True}})
    lines = cli.success_summary("prepare", Namespace(output=tmp_path, adapter=None),
                                {"compatibility_obtained": True, "profiles": {"default": {}}})
    assert lines[1:3] == ["static QOperator quantization (default): QLinearConv ×2, QuantizeLinear ×1, "
                          "DequantizeLinear ×1",
                          "  application metric mAP 0.5 → 0.48 on the CPU EP (higher is better): drop 0.02, "
                          "beyond the limit max_drop=0.01"]
    write_json(tmp_path / "report.json", {"default": {
        "status": "compatible_no_measured_gain", "tuned": False, "required_gain": 0.1,
        "baseline_model": "default/original.onnx",
        "baseline": {"median_wall_ms": 10.0, "median_gpu_ms": 5.0},
        "candidate": {"median_wall_ms": 14.0, "median_gpu_ms": 5.0}}})
    lines = cli.success_summary("optimize", Namespace(output=tmp_path, no_tune=False), {})
    assert lines[1].startswith("  original model → prepared model (static QOperator quantization) with default kernels")
    write_json(tmp_path / "default" / "preparation.json", {"quantization": {"mode": "dynamic", "format": None}})
    assert cli._quantization(tmp_path, "default") == "dynamic quantization"
    assert lines[3] == ("  the output bundle runs the original model, not the quantized one (static QOperator quantization), "
                        "which was not faster")


def test_the_json_result_is_for_redirected_stdout_not_for_a_terminal(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "success_summary", lambda *args: ["verdict"])
    monkeypatch.setattr(cli, "inspect_model", lambda *args, **kwargs: {"compatible": True})
    monkeypatch.setattr("sys.argv", ["onnx-vulkan", "inspect", str(tmp_path / "m.onnx")])
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert cli.main() == 0
    out, err = capsys.readouterr()
    assert out == "" and "redirect stdout to keep it" in err and err.endswith("onnx-vulkan: verdict\n")


def test_a_failing_comparison_is_named_even_when_the_log_is_long(tmp_path):
    import sys

    from onnx_vulkan_pipeline.pipeline import _retain_evidence
    from onnx_vulkan_pipeline.process import command

    script = ("import sys\n"
              "print('  reference vulkan-ep   logits  n=1048576  max|Δ|=1.715e-3  relative=7.3e0  beyond tolerance: 8')\n"
              "for index in range(3000): print(f'  reference vulkan-ep   present.{index}  beyond tolerance: 0')\n"
              "sys.stderr.write('Error: divergent outputs\\n'); sys.exit(1)\n")
    try:
        command([sys.executable, "-c", script])
    except RuntimeError as raised:
        error = raised
    assert "logits" not in str(error)
    assert any("logits" in line and "beyond tolerance: 8" in line for line in cli.failure_summary(error))
    (tmp_path / "source").mkdir()
    _retain_evidence(error, tmp_path / "source", tmp_path / "failed", "Run", {})
    assert "logits" in (tmp_path / "failed" / "error.log").read_text()
