#!/usr/bin/env bash
# Automated test suite: builds, runs the model × EP path matrix natively on
# Linux, samples metrics, writes structured logs in `runs/` and prints a
# summary pasteable into `cronologia.md`.
#
#   scripts/testsuite.sh                              # default matrix
#   scripts/testsuite.sh -m yolov8n -M compile -i 50  # one model, one path
#   scripts/testsuite.sh -n -m rfdetr                 # reuse the existing build
#   scripts/testsuite.sh --baseline runs/baseline.json
#   scripts/testsuite.sh --metric rfdetr.compile gpu.compute_ms   # read a number
#
# The reference plan is `plan-test-suite.md`; known limitations of the
# measurements are in `docs/testsuite.md` and must be read before trusting a
# number.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HELPERS="$ROOT/scripts/testsuite"
MANIFEST="$ROOT/tests/models.toml"
BIN_DIR="$ROOT/target/release"
ORT_LIB="$ROOT/third_party/onnxruntime/linux-x64/lib/libonnxruntime.so"
WEBGPU_LIB="$ROOT/third_party/webgpu-ep/libonnxruntime_providers_webgpu.so"

MODELS=() MODES=() ITERS="" BUILD=1 SAMPLE_MS=100
BASELINE="" TAG="" KEEP=20 DRY=0 DESC="" METRIC_SEL="" METRIC_PATH=""

die() {
    echo "testsuite: $*" >&2
    exit 2
}

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

  -m, --model NAME    only this model (repeatable; default: the manifest's `default` entries)
  -M, --mode MODE     cpu | registry | compile | standalone | webgpu
                      (repeatable; default: cpu,compile)
  -i, --iters N       iterations per run (default: from the manifest)
  -n, --no-build      skip the build and reuse target/release as it is
      --sample MS     metric sampling period (0 = disable; default 100)
      --baseline FILE compare with a previous run and apply the gates
      --tag NAME      label of the output folder (default: timestamp)
      --desc TEXT     description recorded in runs/results.tsv (default: the tag)
      --keep N        how many runs to keep in runs/ (default 20)
      --dry-run       print the commands without executing them

      --metric SEL PATH   read a number out of an existing run and exit: SEL is
                      <model>.<mode>, PATH a jq path into that result
                      (gpu.compute_ms, gpu.flushes, wall_ms.median,
                      gpu.pareto[0].ms). Reads runs/<--tag>/summary.json,
                      default runs/latest. Runs nothing, builds nothing.
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
    -m | --model) MODELS+=("$2"); shift 2 ;;
    -M | --mode) MODES+=("$2"); shift 2 ;;
    -i | --iters) ITERS="$2"; shift 2 ;;
    -n | --no-build) BUILD=0; shift ;;
    --sample) SAMPLE_MS="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --desc) DESC="$2"; shift 2 ;;
    --metric)
        [ $# -ge 3 ] || die "--metric wants <model>.<mode> and a jq path"
        METRIC_SEL="$2"; METRIC_PATH="$3"; shift 3 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h | --help) usage ;;
    *) die "unknown option: $1 (--help)" ;;
    esac
done

# ------------------------------------------------------- 0. --metric (query)
# Reads an existing run and exits. Deliberately before every check below: a
# query touches no GPU and no toolchain, so it must
# work anywhere the repo is checked out. Extraction stays separate from
# execution — a loop chains them (`testsuite.sh -m X --tag e42 &&
# testsuite.sh --metric X.compile gpu.compute_ms --tag e42`).
if [ -n "$METRIC_SEL" ]; then
    summary="$ROOT/runs/${TAG:-latest}/summary.json"
    [ -f "$summary" ] || die "no summary to read: $summary"
    case "$METRIC_SEL" in
    *.*) : ;;
    *) die "--metric wants <model>.<mode>, got: $METRIC_SEL" ;;
    esac
    # captured and not streamed: `jq -e` still prints `null` before failing on
    # an absent path, and the contract of this mode is a number or nothing
    metric_value="$(jq -e -r --arg m "${METRIC_SEL%%.*}" --arg mode "${METRIC_SEL#*.}" \
        "[.results[] | select(.model == \$m and .mode == \$mode)][0] // error(\"no result for \" + \$m + \".\" + \$mode) | .$METRIC_PATH" \
        "$summary")" ||
        die "metric not found: $METRIC_SEL $METRIC_PATH (in $summary)"
    printf '%s\n' "$metric_value"
    exit 0
fi

[ -f "$MANIFEST" ] || die "manifest missing: $MANIFEST"

[ -f "$ORT_LIB" ] || die "ONNX Runtime missing: $ORT_LIB (run ./scripts/fetch-deps.sh)"

run_cmd() { # executes, or just prints with --dry-run
    if [ "$DRY" = 1 ]; then
        echo "+ $*"
        return 0
    fi
    "$@"
}

manifest() { "$HELPERS/manifest.py" "$@" --manifest "$MANIFEST"; }

# ---------------------------------------------------------------- 1. build
if [ "$BUILD" = 1 ]; then
    echo "== build"
    run_cmd cargo build --release -p model-runner -p stt-app -p vulkan-ep ||
        die "build failed"
fi

# ------------------------------------------------- 1b. artifact freshness
# A run loads two artifacts built from the same tree — the runner and the
# plugin — and nothing in the process checks that they agree. When they do not,
# the mismatch does not crash: it reports numbers, and they are wrong. A
# half-built pair once produced a 1.5e-5 parity divergence that looked exactly
# like a kernel regression and cost a bisection to attribute.
#
# The invariant is per artifact, not between them: every artifact must be newer
# than every source it could have been built from.
stale_source() { # artifact -> prints the first source newer than it, or nothing
    find "$ROOT/crates" \
        \( -name '*.rs' -o -name '*.wgsl' -o -name '*.toml' \) \
        -newer "$1" -print -quit 2>/dev/null
}

check_fresh() { # kind, artifacts…
    local kind=$1 artifact newer
    shift
    for artifact in "$@"; do
        [ -f "$artifact" ] || die "missing $kind artifact: $artifact (drop -n?)"
        newer="$(stale_source "$artifact")"
        [ -z "$newer" ] || die "stale $kind artifact: ${artifact#"$ROOT/"} is older than \
${newer#"$ROOT/"}. Rebuild — a runner and a plugin from different trees measure nothing."
    done
}

if [ "$DRY" != 1 ]; then
    check_fresh release \
        "$BIN_DIR/model-runner" \
        "$BIN_DIR/stt-app" \
        "$BIN_DIR/libonnxruntime_ep_vulkan.so"
fi

# The WebGPU EP is deliberately outside `check_fresh`: it is a third-party
# artifact with no sources under `crates/`, so the staleness invariant says
# nothing about it. What does apply is that it must be there *before* the matrix
# starts rather than half way through it.
if [ "$DRY" != 1 ] && [[ " $(printf '%s ' ${MODES+"${MODES[@]}"})" == *" webgpu "* ]]; then
    [ -f "$WEBGPU_LIB" ] ||
        die "WebGPU EP missing: $WEBGPU_LIB (run ./scripts/fetch-webgpu-ep.sh)"
fi

MODEL_ARGS=()
for m in ${MODELS+"${MODELS[@]}"}; do MODEL_ARGS+=(-m "$m"); done
MODE_ARGS=()
for m in ${MODES+"${MODES[@]}"}; do MODE_ARGS+=(-M "$m"); done
[ -n "$ITERS" ] && MODE_ARGS+=(--iters "$ITERS")

# reference data must be fetched before the matrix runs, which consumes it
if [ "$DRY" = 0 ]; then
    manifest fetch ${MODEL_ARGS+"${MODEL_ARGS[@]}"} || true
fi

# ---------------------------------------------------------------- 3. context
TAG="${TAG:-$(date +%Y-%m-%dT%H-%M-%S)}"
RUN_DIR="$ROOT/runs/$TAG"
mkdir -p "$RUN_DIR"

# NVML, not nvidia-smi: the same helper that samples the run reads the identity,
# so the two cannot disagree about which device produced the numbers.
gpu_json="$("$HELPERS/sample_metrics.py" gpu 2>/dev/null || echo '{}')"
gpu_name="$(jq -r '.gpu // ""' <<<"$gpu_json")"
driver="$(jq -r '.driver // ""' <<<"$gpu_json")"
jq -n \
    --arg host "$(hostname)" \
    --arg commit "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)" \
    --argjson dirty "$([ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ] && echo true || echo false)" \
    --arg ort "$(basename "$(dirname "$(dirname "$ORT_LIB")")" 2>/dev/null)" \
    --arg gpu "${gpu_name:-unknown}" \
    --arg driver "${driver:-unknown}" \
    --arg sample_ms "$SAMPLE_MS" \
    --argjson peaks "$("$HELPERS/peaks.py" "${gpu_name:-unknown}")" \
    '{host:$host, commit:$commit, dirty:$dirty, ort:$ort, gpu:$gpu, driver:$driver,
      sample_ms:($sample_ms|tonumber), when:(now|todate)} + $peaks' \
    >"$RUN_DIR/env.json"

# ---------------------------------------------------------------- 4. matrix
# Instrumentation for `validate = "per-node"`: promotes intermedi to outputs.
# Costs minutes on a large graph, so the result is cached on disk.
instrument_model() { # $1 path, $2 name, $3 spec JSON → prints the relative path
    local src="$1" name="$2" spec="$3"
    local out="target/testsuite/instr-$name.onnx"
    local from every limit
    from=$(echo "$spec" | jq -r '.from // 0')
    every=$(echo "$spec" | jq -r '.every // 1')
    limit=$(echo "$spec" | jq -r '.limit // 0')
    mkdir -p "$ROOT/target/testsuite"
    if [ ! -f "$ROOT/$out" ] || [ ! -f "$ROOT/${out%.onnx}.outputs.txt" ] ||
        [ "$ROOT/$src" -nt "$ROOT/$out" ]; then
        echo "   instrumenting $name (expose-intermediates.py)" >&2
        run_cmd "$ROOT/scripts/expose-intermediates.py" "$ROOT/$src" "$ROOT/$out" \
            --from "$from" --every "$every" --limit "$limit" >&2 || return 1
    fi
    printf '%s' "$out"
}

# Starts the sampler on a process. The PID goes into SAMPLER_PID and not stdout:
# in a command substitution the process would be a child of the subshell, and
# `wait` in the parent could not wait on it.
SAMPLER_PID=""
STOP_FILE="$ROOT/target/testsuite/.testsuite-stop"
start_sampler() { # $1 process name, $2 destination csv
    SAMPLER_PID=""
    [ "$SAMPLE_MS" = 0 ] && return 0
    [ "$DRY" = 1 ] && { echo "+ sample_metrics.py sample -p $1" >&2; return 0; }
    mkdir -p "$(dirname "$STOP_FILE")"
    rm -f "$STOP_FILE" "$2"
    "$HELPERS/sample_metrics.py" sample -p "$1" -o "$2" \
        --interval-ms "$SAMPLE_MS" --stop-file "$STOP_FILE" \
        >/dev/null 2>&1 </dev/null &
    SAMPLER_PID=$!
}

stop_sampler() {
    [ -z "$SAMPLER_PID" ] && return 0
    touch "$STOP_FILE"
    wait "$SAMPLER_PID" 2>/dev/null
    SAMPLER_PID=""
    rm -f "$STOP_FILE"
}

mode_env() { # $1 mode, $2 runner
    case "$1" in
    cpu) [ "$2" = stt-app ] && printf 'STT_NO_VULKAN=1 ' ;;
    registry) : ;;
    compile) printf 'VULKAN_EP_COMPILE=1 ' ;;
    esac
}

# Runs a command from the repo root, with the mode's environment. Every job goes
# through here: the suite drives the release binaries of this same tree, on this
# same machine, so a run needs no toolchain beyond the one that built them.
native_exec() { # $1 `VAR=v` env prefix (space separated), $2 command line
    local env_prefix="$1" cmdline="$2"
    local common="RUST_LOG=info ORT_DYLIB_PATH=$ORT_LIB VULKAN_EP_PATH=$BIN_DIR/libonnxruntime_ep_vulkan.so"
    if [ "$DRY" = 1 ]; then
        echo "+ (cd $ROOT && env $common $env_prefix$cmdline)" >&2
        return 0
    fi
    (cd "$ROOT" && eval "env $common $env_prefix$cmdline") 2>&1 </dev/null
}

failures=0
jobs="$(manifest jobs ${MODEL_ARGS+"${MODEL_ARGS[@]}"} ${MODE_ARGS+"${MODE_ARGS[@]}"})" ||
    die "manifest unreadable"

while IFS=$'\x1f' read -r name mode runner iters path args stats validate expect \
    instrument perf_tol size_mb reference golden status reason; do
    [ -z "$name" ] && continue
    out_dir="$RUN_DIR/$name"
    mkdir -p "$out_dir"

    if [ "$status" = skip ]; then
        echo "-- $name/$mode: skipped ($reason)"
        jq -n --arg model "$name" --arg mode "$mode" --arg reason "$reason" \
            '{model:$model, mode:$mode, status:"skip", ok:true, reason:$reason}' \
            >"$out_dir/$mode.json"
        continue
    fi

    echo "== $name/$mode  ($runner, iters=$iters)"
    model_path="$path"
    exposed_arg=()
    if [ "$validate" = per-node ] && [ -n "$instrument" ]; then
        model_path="$(instrument_model "$path" "$name" "$instrument")" ||
            { echo "   instrumentation failed" >&2; failures=$((failures + 1)); continue; }
        exposed_arg=(--exposed "$ROOT/${model_path%.onnx}.outputs.txt")
    fi

    # runner command line
    mapfile -t extra < <(echo "$args" | jq -r '.[]')
    if [ "$mode" = webgpu ]; then
        # `--webgpu` adds a fourth backend: ORT's standalone WebGPU plugin EP
        # (Dawn -> Vulkan on Linux), in the same process and on the same
        # generated tensors. Our EP still runs, which is what keeps the
        # `Vulkan device:` line — and therefore `perf_valid` — in this log.
        #
        # No VULKAN_EP_STATS pass: the number under test here is not ours, and
        # our Pareto is already recorded by the `compile` row of the same model.
        cmdline="$BIN_DIR/model-runner $model_path --iters $iters --webgpu --webgpu-lib $WEBGPU_LIB"
        for a in ${extra+"${extra[@]}"}; do cmdline+=" $a"; done
        [ -n "$reference" ] && cmdline+=" --reference $reference"
        start_sampler model-runner "$out_dir/$mode.metrics.csv"
        native_exec "VULKAN_EP_COMPILE=1 " "$cmdline" >"$out_dir/$mode.stdout.log"
        code=$?
        stop_sampler
        # second pass, one iteration, purely to read ORT's partitioning verdict:
        # the WebGPU EP claims what it supports and leaves the rest on the CPU
        # EP, and a wall compared without knowing that is not a GPU wall. Kept
        # out of the timed pass because verbose ORT logging is not free.
        native_exec "VULKAN_EP_COMPILE=1 ORT_LOG=verbose " \
            "${cmdline/--iters $iters/--iters 1}" >"$out_dir/$mode.placement.log"
        if [ "$DRY" = 0 ]; then
            "$HELPERS/parse_run.py" --model "$name" --mode "$mode" --runner "$runner" \
                --iters "$iters" --stdout "$out_dir/$mode.stdout.log" \
                --placement-log "$out_dir/$mode.placement.log" \
                --metrics "$out_dir/$mode.metrics.csv" \
                --validate "$validate" --expect "$expect" \
                --size-mb "${size_mb:-0}" --perf-tol "${perf_tol:-10}" --exit-code "$code" \
                --env "$RUN_DIR/env.json" \
                $([ "$golden" = 1 ] && echo --golden) --out "$out_dir/$mode.json" ||
                die "parsing failed for $name/$mode"
            jq -r '"   webgpu \(.wall_ms.median // "—") ms · EP \(.ep_ms // "—") ms · CPU EP \(.cpu_ep_ms // "—") ms · placement \((.placement.providers // ["?"]) | join("+")) · result \(if .ok then "ok" else "KO" end)"' \
                "$out_dir/$mode.json"
            jq -e '.ok' "$out_dir/$mode.json" >/dev/null || failures=$((failures + 1))
        fi
        continue
    fi
    if [ "$mode" = standalone ]; then
        # `--standalone` adds the third backend: the same generated tensors go
        # through ORT, the EP and the facade in one process. Two separate
        # commands would compare two different inputs.
        cmdline="$BIN_DIR/model-runner $model_path --iters $iters --standalone"
        for a in ${extra+"${extra[@]}"}; do cmdline+=" $a"; done
        [ -n "$reference" ] && cmdline+=" --reference $reference"
        native_exec "VULKAN_EP_COMPILE=1 " "$cmdline" >"$out_dir/$mode.stdout.log"
        code=$?
        stats_arg=()
        if [ "$stats" = 1 ]; then
            native_exec "VULKAN_EP_COMPILE=1 VULKAN_EP_STATS=1 " "$cmdline" \
                >"$out_dir/$mode.stats.log"
            stats_arg=(--stats-log "$out_dir/$mode.stats.log")
        fi
        if [ "$DRY" = 0 ]; then
            "$HELPERS/parse_run.py" --model "$name" --mode "$mode" --runner "$runner" \
                --iters "$iters" --stdout "$out_dir/$mode.stdout.log" \
                "${stats_arg[@]}" --validate "$validate" --expect "$expect" \
                --size-mb "${size_mb:-0}" --perf-tol "${perf_tol:-10}" --exit-code "$code" \
                --env "$RUN_DIR/env.json" \
                $([ "$golden" = 1 ] && echo --golden) --out "$out_dir/$mode.json" ||
                die "parsing failed for $name/$mode"
            jq -r '"   parity vs EP \(.parity.vs_ep // "—") · wall \(.wall_ms.median // "—") ms · result \(if .ok then "ok" else "KO" end)"' \
                "$out_dir/$mode.json"
            jq -e '.ok' "$out_dir/$mode.json" >/dev/null || failures=$((failures + 1))
        fi
        continue
    fi
    if [ "$runner" = stt-app ]; then
        # stt-app takes the wav *before* the model directory and its iteration
        # count from the environment, so the two runners differ in more than a name
        cmdline="STT_BENCH=$iters $BIN_DIR/stt-app"
        for a in ${extra+"${extra[@]}"}; do cmdline+=" $a"; done
        cmdline+=" $model_path"
        proc=stt-app
    else
        cmdline="$BIN_DIR/model-runner $model_path --iters $iters"
        for a in ${extra+"${extra[@]}"}; do cmdline+=" $a"; done
        # official reference data: inputs and outputs expected by the model
        # authors, not generated by us
        if [ -n "$reference" ]; then
            cmdline+=" --reference $reference"
        fi
        proc=model-runner
    fi
    env_prefix="$(mode_env "$mode" "$runner")"

    # "clean" pass: wall times are measured without the profiler, which inserts
    # a timestamp after every dispatch
    start_sampler "$proc" "$out_dir/$mode.metrics.csv"
    native_exec "$env_prefix" "$cmdline" >"$out_dir/$mode.stdout.log"
    code=$?
    stop_sampler

    # pass with profiler: Pareto, flushes, MB transferred
    stats_arg=()
    if [ "$stats" = 1 ]; then
        native_exec "${env_prefix}VULKAN_EP_STATS=1 " "$cmdline" \
            >"$out_dir/$mode.stats.log"
        stats_arg=(--stats-log "$out_dir/$mode.stats.log")
    fi

    [ "$DRY" = 1 ] && continue
    "$HELPERS/parse_run.py" --model "$name" --mode "$mode" --runner "$runner" \
        --iters "$iters" --stdout "$out_dir/$mode.stdout.log" \
        "${stats_arg[@]}" ${exposed_arg+"${exposed_arg[@]}"} --metrics "$out_dir/$mode.metrics.csv" \
        --validate "$validate" --expect "$expect" --size-mb "${size_mb:-0}" \
        --perf-tol "${perf_tol:-10}" --exit-code "$code" --env "$RUN_DIR/env.json" \
        $([ "$golden" = 1 ] && echo --golden) --out "$out_dir/$mode.json" ||
        die "parsing failed for $name/$mode"
    jq -r '"   wall \(.wall_ms.median // "—") ms · CPU EP \(.cpu_ep_ms // "—") ms · result \(if .ok then "ok" else "KO" end)"' \
        "$out_dir/$mode.json"
    jq -e '.ok' "$out_dir/$mode.json" >/dev/null || failures=$((failures + 1))
done <<<"$jobs"

if [ "$DRY" = 1 ]; then
    rm -rf "$RUN_DIR"
    exit 0
fi

# ---------------------------------------------------------------- 5. report
ln -sfn "$TAG" "$ROOT/runs/latest"
report_args=()
[ -n "$BASELINE" ] && report_args=(--baseline "$BASELINE")
[ -n "$DESC" ] && report_args+=(--desc "$DESC")
"$HELPERS/report.py" "$RUN_DIR" ${report_args+"${report_args[@]}"}
gate=$?

# rotation: raw logs of twenty runs weigh little, but don't grow unbounded
if [ "$KEEP" -gt 0 ]; then
    # shellcheck disable=SC2012
    ls -1dt "$ROOT"/runs/*/ 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r old; do
        [ "$(basename "$old")" = "$TAG" ] || rm -rf "$old"
    done
fi

echo
echo "results in runs/$TAG (runs/latest)"
[ "$failures" -gt 0 ] && exit 1
exit "$gate"
