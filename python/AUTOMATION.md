# Bounded automation: workloads, searches and budgets

This document covers the commands that run many measured experiments without
supervision: `plan`, `quantize-search` (Vulkan) and `cpu-optimize` (ONNX
Runtime CPU). Preparing a single bundle is described in [README.md](README.md).

A Vulkan workload file requires a time budget. Searches retain content-identified
evidence and support resume; flag-based Vulkan searches should also pass
`--budget-seconds` and `--attempts` to bound unattended work. Keep these four
claims separate:

| Claim | Established by |
| --- | --- |
| Numerical correctness | Fixed `atol`/`rtol` against the recorded oracle; CPU tuning checks both its fair baseline and candidates |
| EP/standalone parity | Exact comparison where both Vulkan paths run; not established by CPU tuning |
| Application quality | A representative dataset and metric, recomputed on the deployed Vulkan outputs; CPU tuning alone leaves it unmeasured |
| Acceleration | Interleaved baseline/candidate measurements and an independent confirmation; Vulkan also requires valid GPU spans and diagnostic gates |

Screening timings rank candidates and are labelled "ranking only"; they are
never an acceleration result.

## Commands

```bash
# Validate a workload and show the bounded plan. Runs no model and touches no GPU.
uv run onnx-vulkan plan python/examples/vulkan-workload.json

# Vulkan: screen every candidate cheaply, confirm the best top_k against fp32.
uv run onnx-vulkan quantize-search --workload workload.json --output runs/search1 > search.json

# Vulkan, without a workload file (flags replace the file; the two cannot be mixed).
uv run onnx-vulkan quantize-search model.onnx --quantization q.json --adapter adapter.py \
  --profiles profiles.json --weights weights.json --max-regression 0.02 \
  --top-k 2 --attempts 12 --budget-seconds 7200 --output runs/search1

# ONNX Runtime CPU session settings (threads, graph optimization level, model variant).
uv run onnx-vulkan cpu-optimize --workload python/examples/cpu-workload.json --output runs/cpu1
uv run onnx-vulkan cpu-optimize model.onnx --profiles profiles.json --threads 1,2,4 \
  --samples 30 --attempts 16 --budget-seconds 1800 --output runs/cpu1
```

Passing `--workload` together with the model argument or with any option the file
defines (`--samples`, `--budget-seconds`, `--profiles`, …) is an error: a run has
one source of truth. `--no-tune` (search only) and `--output` are always flags.
`prepare` also accepts `--quality quality.json` for an independent application
quality check on the Vulkan outputs of the deployed model.

Exit codes: `0` finished, `1` failed, `2` model not Vulkan-runnable or `plan`
not ready, `3` budget exhausted (not a failure: run the same command again).

The same entry points exist as a library: `onnx_vulkan_pipeline.search_quantization`,
`optimize_cpu`, `load_cpu_workload`, `workload.validate/load/plan`,
`BudgetExhausted`.

## Vulkan workload schema (version 1)

Strict: unknown keys, wrong types, non-finite numbers and out-of-range values are
rejected before any process starts. Relative paths resolve against the file's
directory (except those *inside* the quantization config, which resolve against the
working directory, as for `prepare`).

| Key | Meaning |
| --- | --- |
| `schema_version` | `1` |
| `name` | Optional label |
| `source` | `.onnx` model |
| `adapter` | Optional Python file for real inputs / export |
| `backend` | `"standalone"` (the only value) |
| `objective` | `"weighted_median_wall"` (the only value) |
| `profiles` | Name → `{dimensions, weight > 0, max_regression >= 0}` |
| `tolerances` | `{atol, rtol}` for every CPU/Vulkan comparison |
| `quality` | Optional, exactly one of `{config: path}` or `{status: "quality_unmeasured"}` |
| `quantization` | `{config: path, grid: [option overrides]}`; omit when nothing lossy is searched |
| `budget` | `seconds` (required, ≤ 7 days), `attempts` (≤ 1000), `top_k`, `samples` (≥ 20), `screen_samples`, `timeout` |

The weights are normalized. `max_regression` is the largest latency regression of
that profile against fp32 a promotion may carry (default `0`). The objective is the
weighted mean of per-profile candidate/baseline ratios; the per-profile guard is
checked separately, so a fast average cannot hide one slower profile.

`quality: {"status": "quality_unmeasured"}` records that no application metric
exists. `plan` then reports the workload as **not ready** for lossy candidates, and
the search refuses to run: nothing lossy is promoted without a measured metric.

CPU workloads use their own flat schema (`python/examples/cpu-workload.json`,
loaded by `load_cpu_workload`); the two schemas are intentionally not merged.

## How the Vulkan search decides

1. **Screening.** Each grid candidate is prepared (the quantization quality gate
   applies) and run for `screen_samples` iterations per profile. Candidates that
   are incompatible, quantize nothing or lose quality are `rejected` with the reason.
2. **Shortlist.** Candidates rank by weighted median wall time; non-dominated ones
   (weighted wall, Vulkan metric drop, storage peak) come first. The Pareto set is
   kept in `search.json`. Unmeasured objectives are never invented.
3. **Confirmation.** The top `top_k` go through `optimize_bundle` (so they inherit its
   interleaved measurement and promotion gates). The fastest screening candidate
   failing to gain does not stop the search; the next is confirmed.
4. **Verdict.** `promoted` (accelerated on a profile, every profile within its bound,
   Vulkan quality verified), `guard_failed`, or `no_gain`. A missing measurement is a
   failure. The best promoted candidate becomes `selected/`.

Final `status`: `promoted`, `no_measured_gain`, `no_passing_candidate` (terminal)
or `budget_exhausted` (resumable).

## Budget and resume

- `budget.seconds` / `--budget-seconds` caps the **whole invocation**: nested worker
  processes (and their descendants) and the wait for the GPU lane are cut at the same
  deadline. Ending it is `budget_exhausted`, distinct from a worker exceeding its own
  `timeout`, which only fails that candidate.
- `attempts` limits new screenings plus confirmations **per invocation**.
  Completed steps cost no new attempt on resume. `budget.attempts_used` records
  the cumulative count across invocations.
- The output directory holds `resume.json` (the identity) and `search.json`. Running
  the same command on an unfinished directory continues it. The identity hashes
  *contents*, not paths: the model and its external data, adapter, quantization /
  evaluation / calibration trees, options, grid, profiles, weights, bounds,
  tolerances, thresholds, tool binaries, policy and measurement plan. The Vulkan
  device is bound at the first GPU measurement and must match on resume.
- Any difference is refused as stale (`StaleExperiment`); evidence is never silently
  reused. Budget, attempts, timeout and `top_k` are invocation limits and may change.
- A terminal result is not overwritten: use a new `--output`.

## Limitations

- Standalone runtime only; the ORT Vulkan EP is not searched here.
- Only quantization is searched on Vulkan; kernel tuning stays inside `optimize`.
- `plan` checks files and the grid but cannot know that the device, driver or model
  will pass; compatibility is found at screening.
- The GPU lane is serialized across processes; a long wait counts against the budget.
- Screening uses few iterations: it can mis-order close candidates. Only the
  confirmation is evidence of speed.
- The CPU search covers session settings only (no quantization) and is valid for the
  recorded ORT build, CPU and affinity.
- Lavapipe validates correctness only; performance claims need a real GPU.
  The full shortlist search has not yet demonstrated an acceleration on a
  representative application dataset.

## Independent quality contract

`prepare --quality quality.json` attaches a task metric to any supported model,
including fp32. The preparation gate compares original and prepared CPU scores;
`run` and `optimize` recompute that metric from the actual deployed Vulkan
outputs. Falling back to fp32 still evaluates its quality when this contract is
present. Without a contract, numerical agreement does not establish task quality.

```json
{
  "schema_version": 1,
  "dataset": {"id": "my-evaluation-set-v1", "split": "holdout", "seed": 42},
  "evaluation": "data/evaluation",
  "adapter": "quality.py",
  "metric": "accuracy",
  "higher_is_better": true,
  "max_drop": 0.01,
  "preprocessing": "tokenizer-v1",
  "postprocessing": "argmax-v1"
}
```

The evaluation directory holds `.npz` samples mapping model input names to
arrays. The self-contained adapter implements `evaluate(session, samples)` and
returns a finite numeric scalar for the configured metric. It calls
`session.run(output_names, input_feed)` for every supplied sample; during the
Vulkan check this session replays recorded Vulkan results and cannot silently
compute new inputs on CPU. Include labels in the adapter, or in the recorded
inputs when the model accepts them. Dataset ID/split/seed and preprocessing /
postprocessing names document provenance; the adapter must implement those
operations. Use a separate final holdout if repeatedly selecting against an
evaluation set.

`max_drop` is an absolute score difference in the metric's units, not a relative
percentage. Calibration and evaluation samples must be disjoint. If both
`--quality` and quantization quality are supplied, they must describe the same
dataset, adapter, metric and threshold; conflicting contracts are refused.
Adapter bytes, input files and fixtures participate in integrity checks.
Paths inside this config resolve from the command's working directory.

## CPU deployment

`cpu-optimize` searches ORT graph optimization levels and intra-op thread counts
for the original and conservatively prepared model. It uses an unoptimized
original-model CPU oracle, and compares candidates against a fair optimized
original-model baseline. Warm-up, session construction and correctness checks
are outside `session.run` timing. At least 20 interleaved pairs and a separate
confirmation are required; every profile has a regression guard. If the fair
baseline already misses the oracle at the declared tolerances, the experiment
stops and keeps the failure evidence.

The result is `optimized`, `no_measured_gain`, or `budget_exhausted`. The last
keeps `.OUTPUT.incomplete` and resumes with identical source, external weights,
profile inputs, adapter, numerical policy, CPU ISA/affinity and ORT build.
Time and attempts apply per invocation. Application quality remains unmeasured:
this service optimizes numerically checked CPU execution without a task dataset.

A completed output contains `session.json` and `cpu-optimization.json`.
Load the selected profile model from `session.json["models"]`, configure
`ort.SessionOptions` from `session.json["session_options"]` using
`graph_optimization_level`, `execution_mode` and `intra_op_num_threads`, and use `CPUExecutionProvider`. The option names are
ORT enum member names (for example `ORT_ENABLE_ALL`, `ORT_SEQUENTIAL`). Retain
the model's external data and preprocessing with the deployment. Re-measure on
a different host or affinity instead of treating this session recipe as portable
performance evidence.

## Report fields

| Record | Evidence to inspect |
| --- | --- |
| Vulkan `search.json` | `schema_version: 2`, content identity digest, `candidates`, screening `ranking`/`pareto`, `shortlist`, `confirmations`, per-profile `guard_failures`, `selected`, budget counters |
| Selected Vulkan bundle `report.json` | Per-profile timing, diagnostics, uninstrumented confirmation and actual deployed-model status |
| CPU `cpu-optimization.json` | `status`, `claims`, numerical check evidence, candidate ledger, independent confirmation, host identity and artifact hashes |
| CPU `session.json` | Selected model paths, explicit session options, providers and policy identity |

An exhausted search has incomplete evidence. A rejected candidate and a
compatible candidate with no measured gain are different outcomes. Retain both;
neither authorizes an acceleration claim.

A worked real-model boundary is recorded in the [Piper results](../BENCHMARKS.md#piper-tts-serena-and-ryan):
Vulkan synthesis runs with CPU fallback disabled, but strict numerical checks
still fail and the experimental models are not promoted production bundles.
