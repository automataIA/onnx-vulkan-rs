#!/bin/sh
# Runs the runner on every model in models/zoo with the compiling EP enabled and
# summarizes: claimed nodes, outcome of the Vulkan vs CPU EP comparison.
#
#   ./scripts/sweep-models.sh [extra flags for model-runner...]
#
# Before each model it runs a `--self-check` (optimized CPU vs non-optimized
# CPU): if even that diverges, the model is **numerically unstable** and the
# comparison on final outputs says nothing about the backend. This happens with
# dynamic-quantization graphs, where scales come from min/max of activations and
# any perturbation shifts downstream buckets.
#
# Note: `--iters 1`. With more iterations `--no-mem-pattern` is needed (see
# cronologia.md).
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="$ROOT/target/release/model-runner"
LOG="${TMPDIR:-/tmp}/sweep-$$.log"

for model in "$ROOT"/models/zoo/*/*.onnx; do
    dir="$(basename "$(dirname "$model")")"
    name="$dir/$(basename "$model")"
    printf '%-52s ' "$name"

    # dynamic dimensions that the model requires to be consistent: with the
    # default of 1 the graph fails for reasons unrelated to the backend
    case "$dir" in
    rfdetr) dims="--dim height=560 --dim width=560" ;;
    *) dims="" ;;
    esac

    # shellcheck disable=SC2086
    if ! timeout 1800 "$RUNNER" "$model" --iters 1 --self-check $dims "$@" >"$LOG" 2>&1; then
        if grep -q 'divergent' "$LOG"; then
            echo "UNSTABLE  already diverges CPU-vs-CPU: output comparison not valid"
            continue
        fi
    fi

    # shellcheck disable=SC2086
    if RUST_LOG=info VULKAN_EP_COMPILE=1 timeout 1800 "$RUNNER" "$model" --iters 1 $dims "$@" >"$LOG" 2>&1; then
        claimed=$(grep -o '[0-9]* nodes in [0-9]* convex blocks' "$LOG" | head -1)
        delta=$(grep -o 'worst relative error [0-9.e+-]*' "$LOG" | head -1)
        echo "OK   ${claimed:-no fused block}  ${delta:-}"
    else
        echo "FAIL $(grep -m1 -E 'Error|error:|panicked|Status Message' "$LOG" | cut -c1-110)"
    fi
done
rm -f "$LOG"
