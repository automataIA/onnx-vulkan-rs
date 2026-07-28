#!/usr/bin/env bash
# Measures encoder wall time (native RTX) for the three EP paths, launching the
# Windows build via cmd.exe interop from WSL. lavapipe is NOT representative:
# use a real GPU. Assumes staging is already done (stt-app.exe, onnxruntime.dll,
# onnxruntime_ep_vulkan.dll, models/) in the Windows directory below.
#
# Usage:  STT_BENCH=6 scripts/bench-windows.sh
# Each path prints per-iteration ms (the 1st includes pipeline compilation;
# steady-state ms are the subsequent iterations).
set -uo pipefail

WINDIR="${WINDIR:?set WINDIR to the Windows staging dir, e.g. 'C:\Users\<you>\Downloads\onnx-vulkan-rs'}"
BENCH="${STT_BENCH:-6}"
WAV='models\en-sample.wav'
MODEL='models\parakeet-tdt-0.6b-v3-onnx'
COMMON="set RUST_LOG=info&& set ORT_DYLIB_PATH=onnxruntime.dll&& set STT_BENCH=$BENCH"

run() {
    local name="$1" extra="$2"
    echo ""
    echo "=================================================================="
    echo "== $name   (STT_BENCH=$BENCH)"
    echo "=================================================================="
    cmd.exe /c "cd /d $WINDIR&& $COMMON&& $extra stt-app.exe $WAV $MODEL" 2>&1 |
        grep -iE "encoder iter|convex blocks|claimed|nodes in|Pareto|GPU |flush|transfer|Well," ||
        echo "(no filtered output — check errors)"
}

run "CPU baseline (pure CPU EP)"        "set STT_NO_VULKAN=1&&"
run "Vulkan kernel-registry (default)"  ""
run "Vulkan compiling EP (1 block)"     "set VULKAN_EP_COMPILE=1&&"

echo ""
echo "== Detailed GPU profile (compiling EP, VULKAN_EP_STATS=1) =="
cmd.exe /c "cd /d $WINDIR&& $COMMON&& set VULKAN_EP_COMPILE=1&& set VULKAN_EP_STATS=1&& stt-app.exe $WAV $MODEL" 2>&1 |
    grep -iE "encoder iter|Pareto|GPU |flush|transfer|op " || true
