# onnx-vulkan-tune

Offline persistence and inspection boundary for Vulkan tactic measurements.
The executable never creates a Vulkan context: `inspect`, `validate`, `diff`,
and `merge` operate entirely on JSON artifacts, while `build` canonicalizes a
validated draft produced by an in-process family runner.

The user-facing ONNX-to-artifact workflow is documented in
[`../../docs/autotune-model.md`](../../docs/autotune-model.md). Its GPU-free
inventory boundary is:

```text
onnx-vulkan-tune model-inventory --model MODEL [--dim SYMBOL=N]...
```

## Dependency decision

Schema version 1 is nested, externally supplied JSON with unknown additive
fields, canonical maps, and actionable parse errors. The CLI therefore uses
`serde` 1 plus `serde_json` 1 rather than a custom parser. Both were already in
the resolved dependency graph, are MIT/Apache-2.0, have no native build, and
their MSRVs (1.56 and 1.68 respectively) are below the workspace's Rust 1.87.
`serde_json`'s default map is deliberately not used for persistent parameter
ordering: artifact maps are `BTreeMap`s and record arrays are sorted explicitly.

A custom parser was rejected because correctly handling Unicode escapes,
duplicate/unknown fields, integer bounds, nesting depth, and useful source
locations would create substantially more security-sensitive code than the
artifact logic itself. `clap`, `anyhow`, `chrono`, and `tempfile` are not added:
the command grammar, typed application errors, Unix timestamp, and same-folder
atomic temporary file need only `std`.

## Stable process contract

- Exit `0`: command completed and emitted the requested result.
- Exit `2`: command-line usage error.
- Exit `3`: malformed or semantically invalid artifact.
- Exit `4`: filesystem or process-I/O failure.
- Exit `5`: merge conflict or incompatible artifacts.

Machine-readable output is one JSON value on stdout. Human diagnostics and
progress use stderr. `inspect --json` and `diff` have stable schema version 1
outputs. Unknown additive artifact fields are accepted in schema version 1;
schema 0 and newer schemas are rejected explicitly.

## Runtime cache-only contract

Standalone and Vulkan EP hosts share two environment variables:

- `ONNX_VULKAN_TUNING_MODE=off|cache-only|require-cache` (default `off`);
- `ONNX_VULKAN_TUNING_ARTIFACT=/path/to/artifact.json`.

`cache-only` without an artifact preserves committed routing. `require-cache`
requires a readable artifact and rejects concrete signatures without an exact,
known record. The artifact is read once when the standalone session or ORT
model is built; ordinary inference never benchmarks candidates.
For the Vulkan EP, a non-`off` tuning mode automatically selects the shared
compiled-subgraph path even when `VULKAN_EP_COMPILE` was not set.

## Intermediate bisect

`scripts/expose-intermediates.py` writes an instrumented model plus an ordered
`.outputs.json` manifest. Run that model through `model-runner --no-opt
--report-json REPORT`; then locate the first divergent node with:

```text
onnx-vulkan-tune bisect --manifest MODEL.outputs.json --report REPORT
```

The instrumented model disables or changes graph fusion and is therefore a
correctness-localization tool, never a performance benchmark. The report keeps
integer outputs exact and accounts for NaN/infinity classes explicitly.

## Fingerprint-checked replay

To force previously recorded tactics, first produce a current target inventory
for the same concrete profile, then run:

```text
onnx-vulkan-tune replay --recorded OLD.json --target CURRENT.json --output REPLAY.json
```

The command writes `REPLAY.json` atomically only when device, model/profile,
workloads, and implementation digests match exactly. Run it with
`ONNX_VULKAN_TUNING_MODE=require-cache` and
`ONNX_VULKAN_TUNING_ARTIFACT=REPLAY.json`. A stale replay exits with the
incompatibility class and leaves no output artifact.

## Reduced failure fixtures

Rejected candidates live in the artifact `diagnostics` array. Reduce one to a
standalone, atomic JSON fixture with:

```text
onnx-vulkan-tune reduce ARTIFACT --diagnostic INDEX --output FIXTURE.json
```

The fixture contains only the exact device/profile, workload signature, tactic
and implementation digest (when recorded), plus the failure outcome/detail. It
does not copy unrelated winners or diagnostics.
