#!/usr/bin/env bash
# Measures encoder wall time for the three EP paths, on the local GPU.
# lavapipe is NOT representative: use a real GPU.
#
# Usage:  STT_BENCH=6 scripts/bench.sh
# Each path prints per-iteration ms (the 1st includes pipeline compilation;
# steady-state ms are the subsequent iterations).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="${STT_BENCH:-6}"
WAV="$ROOT/models/en-sample.wav"
MODEL="$ROOT/models/parakeet-tdt-0.6b-v3-onnx"
export RUST_LOG=info STT_BENCH="$BENCH"
export ORT_DYLIB_PATH="$ROOT/third_party/onnxruntime/linux-x64/lib/libonnxruntime.so"
export VULKAN_EP_PATH="$ROOT/target/release/libonnxruntime_ep_vulkan.so"

[ -x "$ROOT/target/release/stt-app" ] ||
    { echo "build first: cargo build --release -p stt-app -p vulkan-ep" >&2; exit 2; }

run() {
    local name="$1"
    shift
    echo ""
    echo "=================================================================="
    echo "== $name   (STT_BENCH=$BENCH)"
    echo "=================================================================="
    env "$@" "$ROOT/target/release/stt-app" "$WAV" "$MODEL" 2>&1 |
        grep -iE "encoder iter|convex blocks|claimed|nodes in|Pareto|GPU |flush|transfer|Well," ||
        echo "(no filtered output — check errors)"
}

run "CPU baseline (pure CPU EP)"        STT_NO_VULKAN=1
run "Vulkan kernel-registry (default)"
run "Vulkan compiling EP (1 block)"     VULKAN_EP_COMPILE=1

echo ""
echo "== Detailed GPU profile (compiling EP, VULKAN_EP_STATS=1) =="
env VULKAN_EP_COMPILE=1 VULKAN_EP_STATS=1 "$ROOT/target/release/stt-app" "$WAV" "$MODEL" 2>&1 |
    grep -iE "encoder iter|Pareto|GPU |flush|transfer|op " || true
