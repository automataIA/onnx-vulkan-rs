#!/usr/bin/env -S uv run --quiet --script
"""CPU-only contract tests for the model-level autotune orchestrator."""

from __future__ import annotations

import importlib.util
import json
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


if __name__ == "__main__":
    unittest.main()
