#!/usr/bin/env -S uv run --quiet --script
"""CPU-only contract tests for the model-level autotune orchestrator."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("model.py")
SPEC = importlib.util.spec_from_file_location("model_autotune", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


class ModelAutotuneTests(unittest.TestCase):
    def test_geometry_key_is_canonical(self) -> None:
        geometry = {
            "pad": 1,
            "stride": 2,
            "h_out": 16,
            "h_in": 32,
            "kernel": 3,
            "c_out": 8,
            "c_in": 3,
        }
        self.assertEqual(model.geometry_key(geometry), "3,8,3,32,16,2,1")

    def test_resume_rejects_stale_identity(self) -> None:
        device = {"name": "gpu", "driver_version": 1}
        row = {
            "schema_version": 1,
            "model_digest": "aa",
            "device": device,
            "implementation_digest": "bb",
            "geometry_key": "1,2,3,4,5,1,0",
            "result": {"eligible": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.jsonl"
            state.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(
                len(model.load_state(state, "aa", device, "bb")),
                1,
            )
            self.assertEqual(model.load_state(state, "cc", device, "bb"), {})
            self.assertEqual(
                model.load_state(state, "aa", {"name": "other"}, "bb"),
                {},
            )
            self.assertEqual(model.load_state(state, "aa", device, "cc"), {})

    def test_resume_recovers_only_truncated_final_record(self) -> None:
        device = {"name": "gpu", "driver_version": 1}
        row = {
            "schema_version": 1,
            "model_digest": "aa",
            "device": device,
            "implementation_digest": "bb",
            "geometry_key": "1,2,3,4,5,1,0",
            "result": {"eligible": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.jsonl"
            state.write_text(json.dumps(row) + "\n{\"schema_version\":", encoding="utf-8")
            self.assertEqual(len(model.load_state(state, "aa", device, "bb")), 1)
            state.write_text(json.dumps(row) + "\n{\"schema_version\": 1\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                model.load_state(state, "aa", device, "bb")

    def test_geometry_key_includes_rectangular_grouped_axes(self) -> None:
        geometry = {
            "c_in": 32,
            "c_out": 32,
            "group": 32,
            "kernel_h": 3,
            "kernel_w": 5,
            "h_in": 17,
            "w_in": 23,
            "h_out": 9,
            "w_out": 12,
            "stride_h": 2,
            "stride_w": 2,
            "pad_top": 1,
            "pad_bottom": 1,
            "pad_left": 2,
            "pad_right": 2,
            "dilation_h": 1,
            "dilation_w": 1,
        }
        self.assertEqual(model.geometry_key(geometry), "32,32,32,3,5,17,23,9,12,2,2,1,1,2,2,1,1")

    def test_matrix_geometry_key_is_canonical_json(self) -> None:
        first = {
            "family": "matrix-f32",
            "op": "Gemm",
            "m": 16,
            "k": 32,
            "n": 8,
            "trans_a": False,
            "trans_b": True,
            "alpha_bits": 1065353216,
            "beta_bits": 0,
            "c_rows": 16,
            "c_cols": 8,
        }
        second = dict(reversed(list(first.items())))
        self.assertEqual(model.geometry_key(first), model.geometry_key(second))
        self.assertTrue(model.geometry_key(first).startswith('{"alpha_bits":'))

    def test_state_identity_is_scoped_to_family(self) -> None:
        device = {"name": "gpu"}
        matrix_geometry = {
            "family": "matrix-f32", "op": "MatMul", "m": 2, "k": 3, "n": 4,
            "trans_a": False, "trans_b": False, "alpha_bits": 0,
            "beta_bits": 0, "c_rows": 0, "c_cols": 0,
        }
        conv_row = {
            "schema_version": 1, "family": "conv-f32", "model_digest": "aa",
            "device": device, "implementation_digest": "same",
            "geometry_key": "1,2,3,4,5,1,0", "result": {"family": "conv-f32"},
        }
        matrix_row = {
            "schema_version": 1, "family": "matrix-f32", "model_digest": "aa",
            "device": device, "implementation_digest": "same",
            "geometry_key": model.geometry_key(matrix_geometry), "result": {"family": "matrix-f32"},
        }
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.jsonl"
            state.write_text(json.dumps(conv_row) + "\n" + json.dumps(matrix_row) + "\n", encoding="utf-8")
            self.assertEqual(
                model.load_state(state, "aa", device, "same", "matrix-f32"),
                {matrix_row["geometry_key"]: matrix_row["result"]},
            )

    def test_matrix_runner_receives_json_lines_on_stdin(self) -> None:
        geometry = {
            "family": "matrix-f32", "op": "MatMul", "m": 2, "k": 3, "n": 4,
            "trans_a": False, "trans_b": False, "alpha_bits": 1065353216,
            "beta_bits": 0, "c_rows": 0, "c_cols": 0,
        }
        response = {
            "schema_version": 1, "family": "matrix-f32", "device": {"name": "gpu"},
            "implementation_digest": "matrix-impl", "geometry": geometry,
            "tactic": {"family": "matrix-f32", "id": "reference", "parameters": {}},
            "measurement": {"samples": 20, "median_gpu_ns": 3},
            "correctness": {"kind": "max_rel", "passed": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            runner = Path(directory) / "matrix-runner"
            runner.write_text(
                "#!/bin/sh\n"
                "cat >/dev/null\n"
                f"printf '%s\\n' '{json.dumps(response, separators=(',', ':'))}'\n",
                encoding="utf-8",
            )
            runner.chmod(runner.stat().st_mode | 0o111)
            state = Path(directory) / "state.jsonl"
            results = model.tune_missing(
                [geometry], state, "model", {"name": "gpu"}, "matrix-impl",
                family="matrix-f32", runner=runner, response_timeout=2, process_timeout=2,
            )
            self.assertEqual(results[model.geometry_key(geometry)], response)
            journal = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(journal["family"], "matrix-f32")

    def test_resume_waits_for_every_missing_matrix_response(self) -> None:
        geometry = {"family": "matrix-f32", "op": "MatMul", "m": 2, "k": 3, "n": 4}
        device = {"name": "gpu"}
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.jsonl"
            cached = dict(geometry, n=5)
            model.append_state(state, {"schema_version": 1, "family": "matrix-f32",
                "model_digest": "model", "device": device, "implementation_digest": "impl",
                "geometry_key": model.geometry_key(cached), "result": {"cached": True}})
            runner = Path(directory) / "runner"
            responses = [{"schema_version": 1, "family": "matrix-f32", "device": device,
                "implementation_digest": "impl", "geometry": dict(geometry, n=n)} for n in (4, 6)]
            runner.write_text("#!/bin/sh\ncat >/dev/null\n" + "\n".join(
                f"printf '%s\\n' '{json.dumps(row)}'\nsleep 0.02" for row in responses) + "\n")
            runner.chmod(0o700)
            result = model.tune_missing([cached, geometry, dict(geometry, n=6)], state,
                "model", device, "impl", family="matrix-f32", runner=runner,
                response_timeout=2, process_timeout=2)
            self.assertEqual(len(result), 3)

    def test_command_accepts_explicit_stdin(self) -> None:
        result = model.command(
            [sys.executable, "-c", "import sys; print(sys.stdin.read(), end='')"],
            input_data="matrix\n",
        )
        self.assertEqual(result.stdout, "matrix\n")

    def test_family_priority_sums_unique_operations_and_keeps_default_order(self) -> None:
        grouped = {
            "conv-f32": [
                {"workload": {"op": "Conv"}},
                {"workload": {"op": "Conv"}},
            ],
            "matrix-f32": [
                {"workload": {"op": "MatMul"}},
                {"workload": {"op": "Gemm"}},
                {"workload": {"op": "MatMul"}},
            ],
        }
        self.assertEqual(
            model._ordered_families(
                grouped, {"MatMul": 5.0, "Gemm": 3.0, "Conv": 6.0}
            ),
            ["matrix-f32", "conv-f32"],
        )
        self.assertEqual(model._ordered_families(grouped, None), ["conv-f32", "matrix-f32"])

    def test_artifact_uses_exact_workload_and_runner_evidence(self) -> None:
        geometry = {
            "family": "conv-f32",
            "c_in": 3,
            "c_out": 8,
            "kernel": 3,
            "h_in": 32,
            "h_out": 16,
            "stride": 2,
            "pad": 1,
        }
        inventory = {
            "profile": {"model_digest": "aa", "symbol_values": {}},
            "workloads": [
                {
                    "runner": geometry,
                    "workload": {"domain": "", "op": "Conv"},
                    "implementation_digest": "bb",
                }
            ],
        }
        result = {
            "tactic": {"family": "conv-f32", "id": "blocked", "parameters": {}},
            "measurement": {
                "samples": 20,
                "median_gpu_ns": 2,
                "min_gpu_ns": 1,
                "max_gpu_ns": 3,
            },
            "correctness": {"kind": "max_rel", "passed": True, "max_rel": 0.0},
        }
        artifact = model.artifact_from(
            inventory,
            {"name": "gpu"},
            {model.geometry_key(geometry): result},
        )
        self.assertEqual(artifact["records"][0]["workload"]["op"], "Conv")
        self.assertTrue(artifact["records"][0]["correctness"]["passed"])
        self.assertNotIn("max_rel", artifact["records"][0]["correctness"])

    def test_artifact_deduplicates_repeated_exact_workloads(self) -> None:
        geometry = {
            "family": "conv-f32",
            "c_in": 3,
            "c_out": 8,
            "kernel": 3,
            "h_in": 32,
            "h_out": 16,
            "stride": 2,
            "pad": 1,
        }
        workload = {"domain": "", "op": "Conv", "inputs": [1, 3, 32, 32]}
        item = {
            "runner": geometry,
            "workload": workload,
            "implementation_digest": "bb",
        }
        result = {
            "tactic": {"family": "conv-f32", "id": "blocked", "parameters": {}},
            "measurement": {
                "samples": 20,
                "median_gpu_ns": 2,
                "min_gpu_ns": 1,
                "max_gpu_ns": 3,
            },
            "correctness": {"kind": "max_rel", "passed": True},
        }
        artifact = model.artifact_from(
            {
                "profile": {"model_digest": "aa", "symbol_values": {}},
                "workloads": [item, item],
            },
            {"name": "gpu"},
            {model.geometry_key(geometry): result},
        )
        self.assertEqual(len(artifact["records"]), 1)
        # A repeated workload is still fully covered by its single record.
        self.assertEqual(artifact["coverage"]["inventory_workloads"], 1)
        self.assertEqual(artifact["coverage"]["tuned_workloads"], 1)

    def test_artifact_combines_matrix_and_conv_and_counts_relevant_fallbacks(self) -> None:
        conv = {
            "family": "conv-f32", "c_in": 3, "c_out": 8, "kernel": 3,
            "h_in": 32, "h_out": 16, "stride": 2, "pad": 1,
        }
        matrix = {
            "family": "matrix-f32", "op": "MatMul", "m": 2, "k": 3, "n": 4,
            "trans_a": False, "trans_b": False, "alpha_bits": 0,
            "beta_bits": 0, "c_rows": 0, "c_cols": 0,
        }
        def result(family: str) -> dict[str, object]:
            return {
                "tactic": {"family": family, "id": "reference", "parameters": {}},
                "measurement": {"samples": 20, "median_gpu_ns": 2},
                "correctness": {"kind": "max_rel", "passed": True},
            }
        inventory = {
            "profile": {"model_digest": "aa", "symbol_values": {}},
            "workloads": [
                {"runner": conv, "workload": {"domain": "", "op": "Conv"}, "implementation_digest": "conv"},
                {"runner": matrix, "workload": {"domain": "", "op": "MatMul"}, "implementation_digest": "matrix"},
            ],
        }
        artifact = model.artifact_from(
            inventory,
            {"name": "gpu"},
            {model.geometry_key(conv): result("conv-f32"), model.geometry_key(matrix): result("matrix-f32")},
            untunable_nodes=2,
            family_devices={"conv-f32": {"name": "gpu"}, "matrix-f32": {"name": "gpu"}},
        )
        self.assertEqual(artifact["coverage"]["scope"], "fp32-vision-v1")
        self.assertEqual(artifact["coverage"]["inventory_workloads"], 4)
        self.assertEqual(len(artifact["records"]), 2)
        self.assertEqual(set(artifact["family_devices"]), {"conv-f32", "matrix-f32"})


if __name__ == "__main__":
    unittest.main()
