#!/usr/bin/env bash
# Runs the suite's models under the Khronos validation layer with synchronization
# validation on, and fails on any `SYNC-HAZARD-*`.
#
# Why this exists as its own script and not inside `testsuite.sh`: it is the only
# pass that ever caught the missing upload barrier (`runs/sync-fix-1`), and it
# caught it **statically** — the corruption it fixed had never reproduced in 37
# runs. A hazard is a defect whether or not it has fired yet, so waiting for a
# wrong number is the wrong test. The layer costs an order of magnitude in wall
# time, though, which is why it cannot live inside the timed matrix: it belongs
# beside the gate, run per model, not in it.
#
# Correctness of the *result* is not checked here — `testsuite.sh` does that.
# What is checked is whether the command stream is well synchronized.
#
#   scripts/sync-check.sh [-m MODEL]... [--no-build] [--windows]
#
# `--windows` runs the staged `.exe` on the Windows host through `cmd.exe`,
# where the Vulkan SDK lives, instead of the local Linux binary. The models and
# binaries must already be staged there — `scripts/testsuite.sh --stage-only`
# does that. This is usually the only mode that can actually run: the layer
# ships with the SDK, and a WSL2 checkout rarely has it.
#
# Exit codes: 0 clean · 1 hazards found · 2 cannot run the check.

set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
HELPERS=scripts/testsuite
MANIFEST=tests/models.toml
OUT=${SYNC_CHECK_OUT:-runs/sync-check}
WIN_DIR="${TESTSUITE_WIN_DIR:-}"

MODELS=()
BUILD=1
WINDOWS=0
while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model) MODELS+=(-m "$2"); shift 2 ;;
        --no-build) BUILD=0; shift ;;
        --windows) WINDOWS=1; BUILD=0; shift ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# --- precondition: the layer must actually be loadable ---------------------
#
# Without it the run succeeds and reports nothing, which reads exactly like a
# clean result. Refusing to start is the only honest behaviour.
if [ "$WINDOWS" = 1 ]; then
    [ -n "$WIN_DIR" ] || { echo "sync-check: set TESTSUITE_WIN_DIR to the Windows staging dir (WSL side)" >&2; exit 2; }
    WIN_PATH=$(wslpath -w "$WIN_DIR" 2>/dev/null) || {
        echo "sync-check: $WIN_DIR is not reachable from WSL" >&2; exit 2; }
    if ! reg.exe query 'HKLM\SOFTWARE\Khronos\Vulkan\ExplicitLayers' 2>/dev/null |
            grep -qi VkLayer_khronos_validation; then
        echo "sync-check: VkLayer_khronos_validation is not registered on the Windows host." >&2
        echo "  A run without the layer reports zero hazards whether or not there are any." >&2
        echo "  Install the Vulkan SDK on the host." >&2
        exit 2
    fi
    for exe in model-runner.exe stt-app.exe; do
        [ -x "$WIN_DIR/$exe" ] || {
            echo "sync-check: $exe is not staged in $WIN_DIR" >&2
            echo "  run scripts/testsuite.sh --stage-only first" >&2
            exit 2; }
    done
else
    if ! command -v vulkaninfo >/dev/null 2>&1; then
        echo "sync-check: vulkaninfo not found — install the Vulkan SDK / vulkan-tools" >&2
        echo "  (on WSL2 the layer usually lives on the Windows host: use --windows)" >&2
        exit 2
    fi
    if ! vulkaninfo --summary 2>/dev/null | grep -q VK_LAYER_KHRONOS_validation; then
        echo "sync-check: VK_LAYER_KHRONOS_validation is not installed." >&2
        echo "  A run without the layer reports zero hazards whether or not there are any." >&2
        echo "  Install vulkan-validationlayers, or use --windows to run on the host." >&2
        exit 2
    fi
fi

if [ "$BUILD" = 1 ]; then
    echo "== build"
    cargo build --release -p model-runner -p stt-app -p vulkan-ep ||
        { echo "build failed" >&2; exit 2; }
fi

mkdir -p "$OUT"
SEP=$'\x1f'
FAILED=0
CHECKED=0

# One `compile`-mode job per model: the per-node registry path records a
# different command stream, but it is not the path anyone runs.
while IFS="$SEP" read -r name mode runner iters path args stats validate expect \
        instrument perf_tol size_mb reference golden status reason; do
    [ "$status" = run ] || { echo "-- $name: skipped ($reason)"; continue; }
    base_dir=$([ "$WINDOWS" = 1 ] && echo "$WIN_DIR" || echo "$ROOT")
    if [ ! -e "$base_dir/$path" ]; then
        echo "-- $name: skipped (model not staged: $path)"
        continue
    fi

    extra=$(printf '%s' "$args" | python3 -c 'import json,sys; print(" ".join(json.load(sys.stdin)))' 2>/dev/null)
    log="$OUT/$name.log"
    echo "== $name"
    # One iteration either way: a hazard is recorded while the command buffer is
    # built, so repeating the run adds time and no coverage. `stt-app` takes its
    # extra arguments (the wav) *before* the model directory and its iteration
    # count from the environment, so the two runners differ in more than a name.
    if [ "$runner" = stt-app ]; then
        win_cmd="set STT_BENCH=1&& stt-app.exe ${extra//\//\\} ${path//\//\\}"
        nix_cmd=(env STT_BENCH=1 ./target/release/stt-app $extra "$path")
    else
        win_cmd="model-runner.exe ${path//\//\\} --iters 1 $extra"
        nix_cmd=(./target/release/model-runner "$path" --iters 1 $extra)
    fi
    if [ "$WINDOWS" = 1 ]; then
        # </dev/null: cmd.exe would otherwise eat the stdin of this loop
        cmd.exe /c "cd /d $WIN_PATH\
&& set RUST_LOG=warn&& set ORT_DYLIB_PATH=onnxruntime.dll\
&& set VULKAN_EP_COMPILE=1\
&& set VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation\
&& set VK_LAYER_VALIDATE_SYNC=1\
&& $win_cmd" >"$log" 2>&1 </dev/null
    else
        VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
        VK_LAYER_VALIDATE_SYNC=1 \
        VULKAN_EP_COMPILE=1 \
        RUST_LOG=warn \
            "${nix_cmd[@]}" >"$log" 2>&1
    fi
    CHECKED=$((CHECKED + 1))

    hazards=$(grep -c 'SYNC-HAZARD-' "$log")
    if [ "$hazards" -gt 0 ]; then
        FAILED=$((FAILED + 1))
        echo "   $hazards SYNC-HAZARD in $log"
        grep -o 'SYNC-HAZARD-[A-Z-]*' "$log" | sort | uniq -c | sed 's/^/     /'
    else
        echo "   clean"
    fi
done < <("$HELPERS/manifest.py" jobs --manifest "$MANIFEST" -M compile "${MODELS[@]+"${MODELS[@]}"}")

echo
if [ "$CHECKED" = 0 ]; then
    echo "sync-check: no model was actually run" >&2
    exit 2
fi
if [ "$FAILED" -gt 0 ]; then
    echo "sync-check: $FAILED of $CHECKED models report hazards (logs in $OUT)"
    exit 1
fi
echo "sync-check: $CHECKED models clean"
