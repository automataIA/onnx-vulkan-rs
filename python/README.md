# Verified Vulkan model pipeline

The pipeline runs from a checkout of this repository on Linux and drives the
release tools built there; `ONNX_VULKAN_ROOT` points an installed package at
another checkout. The Rust runtime itself does not need Python.

```bash
SKIP_MODEL=1 ./scripts/fetch-deps.sh     # ONNX Runtime 1.29.0 (~11 MB)
cargo build --release --workspace --lib --bins --examples   # `optimize` uses the tuning examples
uv run onnx-vulkan inspect models/zoo/mobilenetv2/mobilenetv2-12.onnx
uv run onnx-vulkan prepare model.onnx --output runs/prepared
uv run onnx-vulkan optimize runs/prepared --output runs/optimized
uv run onnx-vulkan run runs/optimized --backend standalone
uv run onnx-vulkan run runs/optimized --backend ep
```

Every command ends with a short verdict on stderr (`onnx-vulkan: ...`); a
failure ends with its cause and, when there is one, a hint. The JSON result
runs to hundreds of lines (see the [glossary](#json-result-glossary)), so it
goes to stdout only when stdout is redirected or piped (`> result.json`,
`| jq`); on a terminal it is not printed.

Exit status 0 means success, 1 failure. `inspect` and `prepare` exit 2 when
they worked but the model cannot run on Vulkan (`inspect`:
`"compatible": false`; `prepare`: `"compatibility_obtained": false`), so a
script chaining `inspect`, `prepare` and `run` stops there. `run` exits 1 when
an output misses the tolerance, the standalone/EP parity or the quality limit.
`optimize` exits 0 when it keeps the untuned model
(`compatible_no_measured_gain`): that is a result, not a failure.

When a worker dies, the verdict says how. A signal from outside, usually an
out-of-memory killer such as earlyoom, is named as such. A worker stopped at
`--timeout`, the limit for each worker process, is named with that limit. The
stage that failed is named too when it is not the main run. The
[README quickstart](../README.md#quickstart-your-onnx-model-on-the-gpu) shows a
complete example with its output.

`inspect` creates no Vulkan context. Its JSON keeps runtime compatibility
apart from tuning coverage. The verdict lists the graph inputs with their
element types and dimensions, counts the unsupported nodes per operator, says
when the model cannot run end to end (there is no CPU fallback), and names the
symbolic input dimensions that have no value. Symbols that shape inference
derives from those inside the graph are not listed: they follow from the input
sizes.

Each printed issue appears once, with the `nodes` it affects and `blocking`.
Shapes known only at run time are informational and only counted
(`runtime_shape_nodes`), so FasterRCNN prints 4.6 kB instead of 1.3 MB. The
`inspect_model()` API and a bundle's `compatibility.json` keep one issue per
node. Unresolved dimensions are diagnostics, not proof that a concrete run will
succeed.

`prepare` needs concrete inputs for its numerical gate: pass one `--dim NAME=N` per symbol (`--dim batch_size=1` for the zoo
MobileNetV2), or `--profiles profiles.json` containing, for example,
`{"batch1":{"batch_size":1},"batch4":{"batch_size":4}}`.

`prepare` preserves `original.onnx` and its external tensor files, exports or
transforms `model.onnx`, and compares both with ORT CPU on identical inputs,
whose named TensorProto inputs and outputs it writes under `reference/`. When
the `onnxscript` optimizer pass moves the CPU outputs beyond `--atol`/`--rtol`,
the pass is rejected (`"status": "rejected"` in `preparation.json`, also
printed) and `model.onnx` is the model without it. Any other numerical failure
stops preparation. No pass raises the model's opset ceiling, removes NMS, or
changes tolerances after a failure.

A source with external data gets a fresh `candidate.data` beside `model.onnx`.
Tensors of 1 KiB or less stay inline, because ORT's load-time shape inference
cannot read shape operands from an external file. Generated reference inputs
stop at 2M elements per input; above that, pass real ones with `--adapter` (see
[PyTorch adapter](#pytorch-adapter)).

Preparation runs in a bounded subprocess, and `--timeout` applies to each
external operation. A failed preparation keeps its evidence in
`.OUTPUT.failed` next to the destination, replaced by the next failure; the CLI
prints a short verdict and the path of the complete output, its `error.log`.
These hidden directories hold a copy of the bundle (1.6 GB for SAM 3), so
delete them once the failure is understood.

The manifest hashes every artifact, including external weights, reports and
references. Altered files or manifest parameters invalidate the bundle. Keep
the whole directory when distributing a model. Identical files of 1 MB or more
within a profile are hardlinked: a source with external data whose candidate
keeps every weight would otherwise hold the same bytes twice (Gemma 3 1B int4:
the bundle reports 864 MB and takes 825 MB on disk). Copy it with a tool that
preserves hardlinks, or it doubles. A prepared bundle states
`status: prepared`: GPU correctness is established by `run` or `optimize`.

`run` is a validation runner. `--backend ep` runs ORT CPU and the Vulkan EP;
`--backend standalone` also runs the pure Rust facade on the same inputs. The
JSON lists the backends that actually ran. Standalone/EP parity must be exact;
CPU/Vulkan comparisons use the `--atol` and `--rtol` that `prepare` recorded in
the manifest. `run` and `optimize` take no tolerance flags, so a different
tolerance means preparing a new bundle.

Its timings are those of a validation run, not a benchmark. The CPU reference
runs without ORT graph optimizations, so that both sides execute the same
graph; `run` therefore also times an optimized CPU EP session
(`cpu_optimized_ms`) and compares Vulkan with that one. The standalone library
itself stays independent of ORT. The CPU sessions stop timing once their
samples add up to 30 s (at least one warm-up and one timed run), and the
summary then says how many timed runs they got. SAM 3 takes about 11 s per CPU
sample, so it gets two, and its default `run` finishes in about 3.5 minutes
instead of exceeding the 300 s `--timeout` per worker.

`optimize` tunes eligible workloads and benchmarks the original compatible
model against the prepared candidate with its selected tactics. If the
original is incompatible, the reference is the prepared compatible candidate
without tuning.

The tuner validates its tactics on the bundle's reference inputs with the
bundle's tolerance. Tactics that miss it are dropped (`"tactics_rejected"`
names the output, and the verdict prints it), and the candidate keeps the
default kernels. YOLOv4 lost its tactics this way, over one value at 2.0e-5
against an atol of 1e-5. The tuner's validation has no standalone/EP parity
gate, so a dynamically quantized model with tunable fp32 workloads loses its
tactics the same way.

The benchmark collects at least 20 baseline/candidate pairs, alternates their
order, and discards one compilation warm-up per process. Reports include wall
samples, GPU compute/transfer counters and tracked standalone storage peaks
(not total process RSS or total device VRAM). Promotion requires GPU and
end-to-end improvements above 10% and above twice the larger sample standard
deviation. The gate is deliberately conservative, following the repository's
rule that wall time alone cannot justify promotion. Missing profiler evidence,
more flushes, transfers or convex blocks, and fewer claimed nodes also prevent
promotion.

Sampling stops early only to reject; promotion still needs every sample. If
the candidate is the original model with default kernels, nothing is measured
(`"stopped": "identical"`). After 5 pairs, a median gain that would miss the
threshold on GPU or wall time, even with two standard deviations in the
candidate's favor, ends it (`"stopped": "futility"`, with `"pairs"`).

Without a demonstrated improvement, the compatible reference is selected and
the report says `compatible_no_measured_gain`. That is the usual outcome
(MobileNetV2, YOLOv4, YOLOv8n): on this GPU the default kernels are already as
fast as the tuned ones. The output bundle is a verified Vulkan bundle running
the original model, and `run` measures it against the CPU. To look for a gain
anyway, prepare at the batch size you will deploy (`--dim batch_size=8`):
larger convolutions leave tactics more room. For a quantized bundle the
fallback drops the quantization too, and the verdict says so. The quantized
model was not faster than the fp32 original on Vulkan, which is common for int8
on a GPU (MobileNetV2 int8 runs at 0.44× the CPU EP in
[Performance](../README.md#performance)). `report.json` records, per profile,
whether tuning ran and the median wall and GPU times of the baseline and
candidate behind the decision.

Model tuning currently has two bounded fp32 families. `conv-f32` covers the
eligible profile-resolved Conv geometries. `matrix-f32` covers rank-2 MatMul
with `M > 1` and the shipped 64x64 and 16x16 MatMul routes, plus rank-2 Gemm
with transpose flags, finite alpha/beta values, and broadcastable bias on the
Gemm64 and Gemm16 routes. Batched MatMul, GEMV (`M = 1`), unsupported ranks or
dtypes, and unsupported attributes remain explicit inventory fallback reasons.
Gemm attribute identities normalize ORT-materialized defaults, so both
frontends resolve the same persistent tactic. The pipeline measures a
per-profile GPU Pareto report before tuning, persists it as
`tuning-profile.json` and its operation priorities as `tuning-priority.json`,
then schedules families by those priorities. This
coverage description makes no GPU performance claim.

GPU operations are serialized across pipeline processes for the current user.
An interrupted optimization retains `.OUTPUT.incomplete` next to the target;
rerunning the same command resumes completed pairs and tuner state. Changed
source content, binaries, device, profile or search parameters reject resume.
Choose a new output directory to start a different experiment. Existing
completed bundles are never overwritten.

## PyTorch adapter

Install the optional exporter dependencies with `uv sync --extra torch` and use
`uv run --extra torch onnx-vulkan prepare checkpoint.pt --adapter adapter.py --output runs/prepared`.
An adapter is executable project code and explicitly owns model construction
and checkpoint loading. Its required functions are:

```python
def load_model():
    # Construct architecture and load the checkpoint explicitly.
    return model.eval()

def example_inputs(profile=None):
    # Choose representative tensors for this explicit symbol/value profile.
    return {"images": representative_tensor}
```

Export uses `torch.onnx.export(..., dynamo=True, opset_version=21)` and checks
PyTorch against exported ONNX before subsequent transformations. Named inputs
must match the forward signature. Explicit profiles require
`example_inputs(profile=...)`; each profile produces a separate static export,
at the sizes of the returned tensors.

An existing ONNX source can use an adapter too, to supply domain-valid inputs
instead of deterministic generated ones. Such an adapter needs only
`example_inputs()`, returning numpy arrays keyed by input name. Their shapes fix
the symbolic dimensions, so no `--dim` is needed and torch is not imported:

```python
from pathlib import Path
import onnx
from onnx import numpy_helper

def example_inputs():
    data = Path("models/zoo/golden/roberta-base-11/test_data_set_0")
    return {"input_ids": numpy_helper.to_array(onnx.load_tensor(str(data / "input_0.pb")))}
```

If it also defines `load_model()`, the returned model is run on the same inputs
and must match the ONNX model; that check needs `uv run --extra torch`.

## Optional quantization

Quantization is disabled unless `prepare --quantization config.json` supplies
all of the following (static mode shown):

```json
{
  "calibration": "/data/calibration",
  "evaluation": "/data/evaluation",
  "adapter": "/project/quality.py",
  "metric": "accuracy",
  "max_drop": 0.01,
  "higher_is_better": true,
  "format": "QOperator"
}
```

`atol`/`rtol` (below) are required in practice: without them the CPU
comparison of the fp32 and quantized models uses the bundle tolerance and fails
(MobileNetV2 dynamic: `Quantized outputs exceed atol=1e-05, rtol=0.0001 (max
absolute 1.92...)`). The [worked example](#worked-example-dynamic-int8-in-three-files)
is a complete, tested starting point.

Optional keys:

- `mode`: `static` (default, calibrated activations), `dynamic`
  (`DynamicQuantizeLinear` + `ConvInteger`/`MatMulInteger`, no calibration) or
  `int4` (weight-only asymmetric `MatMulNBits`). `dynamic` and `int4` require no
  `calibration` or `format`, and reject keys that only another mode reads.
- `block_size` (`int4` only, default 128): power of two >= 16.
- `atol`/`rtol`: accepted fp32-vs-quantized error on the evaluation set. Without
  them the preparation tolerances apply unchanged, which quantization error
  normally exceeds; the error message then asks for an explicit value.
- `op_types_to_quantize`: in `static` mode defaults to the forms the Vulkan
  interpreter implements (`Conv`, `MatMul`, `Add`, `GlobalAveragePool`,
  `MaxPool`, `Relu`, `Clip`). ONNX Runtime's own default emits
  `QLinearSigmoid`, `QLinearMul` and `QLinearConcat`, which `inspect` refuses.
  `dynamic` defaults to `Conv`, `MatMul`; `int4` to `MatMul` (`Gather` is
  opt-in). `ConvTranspose` (`dynamic`, with `MatMul`): each one whose kernel
  equals its stride, with no pads, groups or dilation, is first rewritten as
  Transpose → MatMul → Add → Reshape → Transpose → Reshape (exact in fp32),
  and its MatMul is quantized; the others stay fp32. The bias is added per
  MatMul column so it fuses with the dequantization, and the columns are
  ordered so the pixel shuffle is a 2D transpose; an NHWC → NCHW Transpose
  in front is skipped instead of undone. The preparation report
  lists both under `options.convtranspose_as_matmul`.
- `per_channel` (`static`/`dynamic`, default `false`): per-channel weights.
- `calibrate_method` (`static`): `MinMax` (default), `Entropy` or `Percentile`.
- `nodes_to_exclude`: node names kept in fp32.
- `preprocess` (default `false`): ONNX Runtime's `quant_pre_process` before
  calibration.

`QOperator` runs as integer convolution/matmul on Vulkan. `QDQ` is compatible
but executes fp32 operators between quantize/dequantize pairs, because no
QDQ-to-QLinear fusion exists yet.

Each dataset is a directory of `.npz` samples whose keys are exactly the model
input names; any other key, such as a label, is rejected. Calibration and
evaluation must be disjoint. `quality.py` implements
`evaluate(session, samples) -> float`: `samples` is a list of
`{input name: numpy array}` dicts in the evaluation directory's file-name
order, and `session.run(None, sample)` returns the model outputs as on an ORT
session. Labels and postprocessing live in `quality.py`, for example as a list
indexed like `samples`. `metric` only names the returned value in reports, and
`max_drop` is the accepted decrease (increase when `higher_is_better` is
`false`) from the fp32 model's value.

Two tolerance pairs exist and are not interchangeable. The config's
`atol`/`rtol` bound fp32 against quantized outputs on the CPU, which
quantization error dominates (`1e-1` is common for int8 logits). The CLI's
`--atol`/`--rtol` bound the quantized model's CPU outputs against Vulkan's and
stay small (default `1e-5`/`1e-4`): a Vulkan integer kernel reproduces ORT's
quantized arithmetic, so a large difference there is a bug, not quantization
error.

The exception is `mode: dynamic`. Each `DynamicQuantizeLinear` takes its scale
from its input's extremes, so a one-ulp fp32 difference upstream (GPU `Sigmoid`
against MLAS, for example) moves a whole tensor by one step, and the difference
grows layer by layer (YOLOv8n: max |Δ| 16 on outputs up to 640). No tolerance
holds there, and none is needed: a dynamic bundle's Vulkan runs are gated by
bit-exact standalone/EP parity and by the application metric on Vulkan outputs.
The CPU difference is only reported.

A source that already holds `DynamicQuantizeLinear` (RF-DETR int8, for
example) has no metric, so it gets the same parity gate plus the per-node
contract: every quantized tensor is exposed as an output and compared with the
CPU EP in graph order, and the first that differs must be within one code
(`quantized_tensors` in the report). The tensors are exposed in batches of at
most 1G elements, one run each, in graph order until a batch holds the first
that differs (`"runs"`); a tensor of unknown size gets a run of its own.
model-runner keeps outputs in their own element type, one byte per quantized
element per backend: all 137 tensors of SAM 3 (997M elements) fit one batch at
6.6 GB peak, where a copy widened to 16 bytes per element took more than 20 GB.
Bundles under the parity gate run with `--backend standalone` only.

The original and quantized models must satisfy both the explicit numerical
tolerances and the quality threshold, and Vulkan still decides whether the
produced QDQ/QOperator graph is supported. `run`/`optimize` later check every
evaluation sample on Vulkan, recomputing the same application metric from
recorded Vulkan outputs in a bounded subprocess. The copied quality adapter
must be self-contained (labels included) and call
`session.run(output_names, input_feed)` only on the provided evaluation inputs;
new or altered inputs without a measured output are rejected. A CPU quality
metric alone is not marked as Vulkan quality verification.

Quality fixtures and `quantized-reference/` store the quantized model's CPU
outputs; `reference/` keeps the fp32 outputs used when a profile runs
`original.onnx`. If `optimize` falls back to the fp32 baseline, quality is not
claimed. Quantized representations are not assumed to be faster.

### Worked example: dynamic int8 in three files

`mode: dynamic` needs no calibration set, so it is the cheapest first try.
Three files, here for the zoo MobileNetV2 (input `input`, 1000 logits), in a
directory `runs/q8-example/`; the commands run from the repository root, and the
config's relative paths resolve against it.

`make_data.py` writes the evaluation samples, one `.npz` each, keyed by the
model's input name (`inspect` prints it). Replace the random arrays with your
preprocessed inputs; `static` mode also needs a `calibration` directory of the
same form, disjoint from the evaluation one:

```python
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
for directory, count in (("calibration", 16), ("evaluation", 8)):
    Path(directory).mkdir(exist_ok=True)
    for index in range(count):
        image = rng.random((1, 3, 224, 224), dtype=np.float32)
        np.savez(f"{directory}/{index:04d}.npz", input=image)
```

`quality.py` computes the metric. It is copied into the bundle, so the labels
live in it, one per evaluation file in file-name order:

```python
import numpy as np

LABELS = [0, 1, 2, 3, 4, 5, 6, 7]


def evaluate(session, samples):
    hits = [int(np.argmax(session.run(None, sample)[0])) == label for sample, label in zip(samples, LABELS)]
    return sum(hits) / len(hits)
```

`dynamic.json` declares the accepted metric drop and the accepted fp32-vs-int8
difference on the CPU (`atol` 3 on these logits, whose int8 error reached 1.92):

```json
{
  "mode": "dynamic",
  "evaluation": "runs/q8-example/evaluation",
  "adapter": "runs/q8-example/quality.py",
  "metric": "top1",
  "max_drop": 0.02,
  "atol": 3.0,
  "rtol": 0.1,
  "higher_is_better": true
}
```

```bash
(cd runs/q8-example && uv run make_data.py)
uv run onnx-vulkan prepare models/zoo/mobilenetv2/mobilenetv2-12.onnx --dim batch_size=1 --atol 1e-4 \
  --quantization runs/q8-example/dynamic.json --output runs/mnv2-q8
uv run onnx-vulkan run runs/mnv2-q8
uv run onnx-vulkan optimize runs/mnv2-q8 --output runs/mnv2-q8-opt
```

`--atol 1e-4` is the bundle tolerance of the [README quickstart](../README.md#quickstart-your-onnx-model-on-the-gpu):
`optimize` also runs the fp32 original on Vulkan against its CPU reference,
and MobileNetV2 fp32 misses the default `1e-5` by 3e-5.

Verdicts on 2026-10-06 (RTX 4070; the random inputs and labels make top-1 0):

```
onnx-vulkan: bundle written to runs/mnv2-q8 (37.6 MB); Vulkan compatible: yes
onnx-vulkan: dynamic quantization (default): DynamicQuantizeLinear ×53, ConvInteger ×52, MatMulInteger ×1
onnx-vulkan:   application metric top1 0 → 0 on the CPU EP (higher is better): drop 0, within the limit max_drop=0.02
...
onnx-vulkan: vulkan-ep and standalone outputs are identical; the CPU EP differs by up to max|Δ|=9.54e-07, expected for dynamic quantization, so the application metric below is the gate
onnx-vulkan: quantized tensors, in graph order: 53 of 53 identical to the CPU EP
onnx-vulkan: validation timings, not a benchmark: CPU EP 60.61 ms, Vulkan EP 4.44 ms, standalone 4.10 ms
onnx-vulkan: application metric top1 0 → 0 on Vulkan (higher is better): drop 0, within the limit max_drop=0.02
```

Then `optimize` decides whether the int8 bundle beats the fp32 original on
Vulkan; if not, its output runs the original. For MobileNetV2 it does not
(GPU 1.19 → 2.09 ms per run), and the verdict ends with `the output bundle runs
the original model, not the quantized one (dynamic quantization), which was not
faster`.

`static` mode adds `"calibration"` and `"format": "QOperator"`;
`quantize-search` tries the modes for you from the same config.

### Searching quantization options

`uv run onnx-vulkan quantize-search model.onnx --output runs/qsearch
--quantization base.json [--dim batch_size=1] [--grid grid.json] [--no-tune]`
prepares one bundle per candidate under `candidates/<name>`. `--dim` and
`--profiles` work as for `prepare`. `--adapter` is `prepare`'s export/input
adapter, not the quality adapter, which `base.json` names under `adapter`.
`base.json` is a quantization config; each grid entry is an object of option
overrides, and keys the candidate's `mode` does not read are dropped from the
base. Without `--grid`, the search
tries static QOperator/QDQ with and without per-channel weights (only when the
base has `calibration`), dynamic with and without per-channel weights, and
int4 with block sizes 32 and 128.

A candidate passes when `prepare` accepts it (numerical tolerances and the CPU
metric), the profile is compatible, at least one operator was quantized, and
`run` (standalone) matches its quantized reference and keeps the metric on
Vulkan outputs. Rejections are recorded with their reason in `search.json`.
Passing candidates are ranked by their median standalone wall time, excluding
the first iteration. Those runs are not interleaved, so the ranking is only an
ordering: the fastest candidate is handed to `optimize` (output `selected/`),
whose interleaved comparison against the fp32 `original.onnx` alone decides
`acceleration_demonstrated`.

## EP research experiments

`uv run onnx-vulkan research problem.json --output runs/research-001
--attempts 3 --budget-seconds 1800` creates branches named `autoresearch/*` from
`base_commit` (default HEAD). Uncommitted checkout edits are not the baseline.
The problem JSON contains `fixture` (a verified bundle path), `shape`, `dtype`,
`profile`, `reference`, `metric`, `proposer` (an argument array), and `gates` with
four command arrays: `local`, `kernel`, `model`, `regression`. No shell parsing
is performed. Supply an agent executable as the proposer when automatic code
generation is desired; the pipeline does not embed an API credential or choose
a hosted model.

Commands receive `VULKAN_RESEARCH_PROBLEM`, `VULKAN_RESEARCH_FIXTURE`,
`VULKAN_RESEARCH_OUTPUT` and `VULKAN_RESEARCH_ATTEMPT`. Gates must build and test
the candidate worktree, run the kernel benchmark, validate the whole model and
run the regression matrix. They write `metrics.json` in the output directory:
`{"interleaved":true,"baseline":[...],"candidate":[...]}`, with at least 20
`{"wall_ms":...,"gpu_ms":...}` samples per side. All four gates must exit zero
and the numerical performance gate must pass before `proposed_for_review`.
Failures and no-gain attempts are recorded in `journal.json`. Worktrees and
logs remain available for review; nothing is merged or published automatically.

## JSON result glossary

Fields a first reader meets; the code defines the rest.

| field | command | meaning |
|---|---|---|
| `compatible` | inspect | every node has a Vulkan kernel; there is no CPU fallback, so false means the model cannot run |
| `runtime_shape_nodes` | inspect | nodes whose shapes depend on input values; informational |
| `tuning.tunable_workloads` | inspect | Conv/MatMul/Gemm shapes `optimize` can tune, counted only with every `--dim` given |
| `compatibility_obtained` | prepare, optimize | the prepared profiles are Vulkan compatible (`false` ⇒ exit 2) |
| `acceleration_demonstrated` | prepare, optimize | `optimize` promoted tuned kernels over the original; always `false` after `prepare`, which measures nothing |
| `status` | prepare, optimize | `prepared`, or `optimize`'s decision such as `compatible_no_measured_gain` |
| `profiles.NAME.model` | optimize | the model the bundle runs; `original.onnx` after a fallback |
| `profiles.NAME.tactics` | optimize | the tuned kernel file, `null` when none was kept |
| `atol`, `rtol` | all | the bundle tolerance: an element passes when \|Δ\| ≤ atol + rtol·\|reference\| |
| `executed_backends` | run | what actually ran: `ort-cpu`, `vulkan-ep`, `standalone` |
| `gate` | run | `cpu`: outputs compared with the CPU under the tolerance; `parity`: dynamic quantization, gated by bit-exact standalone/EP parity and the metric |
| `outputs`, `standalone_outputs` | run | one entry per model output, Vulkan EP and standalone against the CPU |
| `max_abs` | run | the largest \|Δ\| of that output |
| `max_relative` | run | the largest \|Δ\|/\|reference\|; `null` means infinite, a nonzero Δ where the reference is 0 |
| `max_tolerance_ratio` | run | the worst element's \|Δ\| / (atol + rtol·\|reference\|): ≤ 1 passes; `null` means infinite |
| `mismatches` | run | elements beyond the tolerance |
| `cpu_ms`, `cpu_optimized_ms`, `candidate_ms`, `standalone_ms` | run | per-iteration wall times, the first being warm-up: CPU EP without and with graph optimizations, Vulkan EP, standalone |
| `cpu_fallback_disabled` | run | the Vulkan session could not move nodes to the CPU |
| `quality_evaluation` | run | the quantization metric recomputed on Vulkan outputs (`samples` 0 without one) |
| `gpu` | run | profiler counters: `compute_ms`, `flushes`, `uploads`, `pareto` (time per kernel) |

`optimize` writes its measurements to `OUTPUT/report.json`, per profile:
`baseline`/`candidate` medians, `pairs` (one original and one prepared sample
each), `stopped` (`identical`, `futility`), `tactics_rejected` and `tuned`.

## API and tests

`onnx_vulkan_pipeline` exports `inspect_model`, `prepare_model`,
`prepare_bundle`, `optimize_bundle` and `run_bundle`. Their keyword arguments
match the services called by the CLI. `research` is available from
`onnx_vulkan_pipeline.research`, `search_quantization` from
`onnx_vulkan_pipeline.search`.

```bash
uv run --extra test pytest -q
uv run --no-project scripts/autotune/test_model.py
cargo test --workspace -- --test-threads=1
cargo clippy --workspace --all-targets -- -D warnings
```

Use RTX 4070 for the initial performance target and lavapipe for correctness.
Results do not imply AMD performance. The initial vision corpus is ResNet50,
MobileNetV2 and YOLOv8; application quality still needs representative task data.

## Delivery status (2026-10-06)

A dated summary; none of this is rechecked automatically.

`inspect`, `prepare`, `run`, `optimize`, `quantize-search` and `research` are
implemented, with strict CPU/EP/standalone validation and opt-in quantization
(`static` QOperator/QDQ, `dynamic` int8, `int4` `MatMulNBits`) gated by an
application metric recomputed on Vulkan outputs. Tuning covers the Conv and
rank-2 fp32 MatMul/Gemm cases described above; the rest are explicit inventory
fallback reasons. The matrix tuning passed bit-exact standalone/EP parity on
the 11-model local corpus (2026-09-08, `runs/matrix-tuning-20260908`, STT
runner excluded).

On RTX 4070, these went through `prepare` and `run`:

- MobileNetV2 fp32 at `--atol 1e-4`, worst element at 20% of its allowance
  (2026-10-06). At the default `1e-5` it still fails on 4/1000 outputs (max
  absolute error 3.052e-5); the default was not widened.
- MobileNetV2 dynamic int8, the [worked example](#worked-example-dynamic-int8-in-three-files):
  53 of 53 quantized tensors identical to the CPU EP (2026-10-06).
- Gemma 3 1B int4 with KV-cache dimensions at `--atol 2e-3 --rtol 1e-3`, worst
  element at 59% (2026-10-06). One forward pass, not text generation.
- RoBERTa-base, RF-DETR fp32 (`--rtol 1e-3`) and int8 (parity and per-node
  contract), and SAM 3's pre-quantized vision encoder (2026-10-05/06).
  FasterRCNN is refused, its unsupported operators listed.

`optimize` kept the untuned model (`compatible_no_measured_gain`) on every real
model tried: MobileNetV2, YOLOv4, YOLOv8n, RoBERTa, and the MobileNetV2 int8
bundle (GPU 1.19 → 2.09 ms against the fp32 original). No tuned or quantized
candidate has demonstrated a gain on a real model. Application quality on
representative data is still unmeasured for ResNet50, MobileNetV2 and YOLOv8,
and no AMD performance claim is made.

The research controller accepts an external proposer and four explicit gates.
A bounded Gemm16 experiment completed all four on 2026-09-09 and produced a
proposal for review, unintegrated: a 23.8% complete-latency reduction on an
extracted MobileNet classification head, not the full network.

Checks on this checkout, 2026-10-06: 113 Python tests, 13 autotune tests,
308 serialized workspace Rust tests, workspace Clippy with warnings denied.
Failed runs keep their reports, replay command and complete output
(`error.log`) under `.BUNDLE.run-failed` beside the bundle, replaced by the
next failure.

## Bounded Gemm routing experiment

`scripts/research/snapshot.py --output NEW_DIRECTORY` creates a detached local
baseline from tracked working changes and untracked project sources using a
temporary Git index. It preserves the current branch, HEAD and staging area.
Build the baseline independently; the candidate gets its own target directory.

`scripts/research/gemm16.py` supplies `propose`, `local`, `kernel`, `model` and
`regression` commands for a single exact-shape Gemm experiment. The contract
adds `geometry` (matrix inventory JSON), `baseline_root`, `dependencies_root`
(the source checkout containing models and third_party), and optionally
`regression_baseline` (a testsuite summary) to the research API fields.
The deterministic proposer changes the default tile only for the requested
M/K/N and preserves explicit tactic choices. It is not an LLM code generator.
The model gate compares independent builds in 20 alternating AB/BA pairs and
runs final CPU/EP/standalone validation. Kernel improvement alone cannot
promote the patch. Failed attempts retain diagnostic logs and the tracked
binary diff, including when the experiment budget expires. Untracked-only
or empty proposals cannot be promoted.
