# Persistent Vulkan autotune orchestration

For the end-to-end ONNX model workflow, including coverage inspection,
artifact generation and runtime use, see
[`../../docs/autotune-model.md`](../../docs/autotune-model.md):

```bash
uv run scripts/autotune/model.py model.onnx --output tactics.json --strict
```

The lower-level command below remains the research harness for explicit
MatMulNBits geometries; it does not itself produce a runtime artifact.

Run the single entry point through `uv`; it builds the Rust runner unless
`--bin` is supplied:

```bash
uv run scripts/autotune/run.py --mode exhaustive --trials 336 --geom 1152,6912
uv run scripts/autotune/run.py --mode random --trials 60 --seed 0
uv run scripts/autotune/run.py --mode tpe --trials 60 --seed 0
```

One Rust child process remains alive per geometry. The candidate list,
viability decision, shader compilation, correctness gate, and GPU timestamps
all remain Rust-owned. Python only orders candidate IDs returned by `list` and
uses `compare` over JSON Lines.

`--state` is an fsync'd JSONL journal independent of production tactic
artifacts. TPE additionally uses `--study` as an Optuna SQLite database.
`--report` writes JSON plus a sibling SVG covering best-so-far, coverage,
rejection causes, and retrospective per-geometry regret. `--trials`,
`--wall-seconds`, and `--vram-bytes` enforce budgets. Ctrl-C flushes the current
journal row and asks the Rust process to shut down before exiting 130.
