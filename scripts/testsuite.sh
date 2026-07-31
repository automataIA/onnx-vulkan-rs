#!/usr/bin/env bash
# Automated test suite: builds, stages to Windows, runs the model × EP path
# matrix, samples metrics, writes structured logs in `runs/` and prints a
# summary pasteable into `cronologia.md`.
#
#   scripts/testsuite.sh                              # default matrix
#   scripts/testsuite.sh -m yolov8n -M compile -i 50  # one model, one path
#   scripts/testsuite.sh -n -m rfdetr                 # reuse existing staging
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
TARGET=x86_64-pc-windows-msvc
BIN_DIR="$ROOT/target/$TARGET/release"
# Windows staging directory (WSL side); overridable from the environment
WIN_DIR="${TESTSUITE_WIN_DIR:-}"

MODELS=() MODES=() ITERS="" BUILD=1 STAGE_ONLY=0 SAMPLE_MS=100
BASELINE="" TAG="" KEEP=20 DRY=0 DESC="" METRIC_SEL="" METRIC_PATH=""

die() {
    echo "testsuite: $*" >&2
    exit 2
}

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

  -m, --model NAME    only this model (repeatable; default: the manifest's `default` entries)
  -M, --mode MODE     cpu | registry | compile (repeatable; default: cpu,compile)
  -i, --iters N       iterations per run (default: from the manifest)
  -n, --no-build      skip build and staging
      --stage-only    build and stage, does not run
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
    --stage-only) STAGE_ONLY=1; shift ;;
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
# query touches no Windows host, no staging dir and no toolchain, so it must
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

# A run made only of `standalone` jobs never touches Windows: no cross-build, no
# staging, no interop. That is the whole point of the mode — it turns a 3–5 min
# cycle into seconds — so its requirements must not be the Windows ones.
HAS_NATIVE=0 HAS_WIN=1 NATIVE_ONLY=0
if [ ${#MODES[@]} -gt 0 ]; then
    HAS_WIN=0
    for m in "${MODES[@]}"; do
        if [ "$m" = standalone ]; then HAS_NATIVE=1; else HAS_WIN=1; fi
    done
    [ "$HAS_NATIVE" = 1 ] && [ "$HAS_WIN" = 0 ] && NATIVE_ONLY=1
fi

WIN_PATH=""
if [ "$NATIVE_ONLY" = 0 ]; then
    command -v wslpath >/dev/null || die "WSL required: the suite drives the Windows binaries via interop"
    [[ -n "$WIN_DIR" ]] || die "set TESTSUITE_WIN_DIR to the Windows staging dir (WSL side)"
    WIN_PATH="$(wslpath -w "$WIN_DIR" 2>/dev/null)" || die "WIN_DIR not convertible: $WIN_DIR"
fi

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
    if [ "$HAS_NATIVE" = 1 ]; then
        echo "== build (native, for the standalone mode)"
        run_cmd cargo build --release -p model-runner -p vulkan-ep || die "native build failed"
    fi
    if [ "$HAS_WIN" = 1 ]; then
        echo "== build ($TARGET)"
        run_cmd cargo xwin build --release --target "$TARGET" \
            -p model-runner -p stt-app -p vulkan-ep || die "build failed"
    fi
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

if [ "$DRY" != 1 ] && [ "$HAS_NATIVE" = 1 ]; then
    check_fresh native \
        "$ROOT/target/release/model-runner" \
        "$ROOT/target/release/libonnxruntime_ep_vulkan.so"
fi
if [ "$DRY" != 1 ] && [ "$HAS_WIN" = 1 ]; then
    check_fresh "$TARGET" \
        "$BIN_DIR/model-runner.exe" \
        "$BIN_DIR/onnxruntime_ep_vulkan.dll"
fi

# ---------------------------------------------------------------- 2. staging
stage() {
    echo "== staging → $WIN_DIR"
    mkdir -p "$WIN_DIR"
    for f in model-runner.exe stt-app.exe onnxruntime_ep_vulkan.dll; do
        [ -f "$BIN_DIR/$f" ] || die "missing binary: $BIN_DIR/$f (drop -n?)"
        run_cmd cp "$BIN_DIR/$f" "$WIN_DIR/"
    done
    run_cmd cp "$ROOT/scripts/sample-metrics.ps1" "$WIN_DIR/"
    if [ ! -f "$WIN_DIR/onnxruntime.dll" ]; then
        run_cmd cp "$ROOT/third_party/onnxruntime/win-x64/lib/onnxruntime.dll" "$WIN_DIR/" ||
            die "onnxruntime.dll missing: run ./scripts/fetch-deps.sh"
    fi
    # the models weigh 4.2 GB in total: copy only what the matrix needs, and
    # only if missing or older than the source (--update)
    local paths
    paths="$(manifest stage "${MODEL_ARGS[@]}")" || die "manifest unreadable"
    [ -n "$paths" ] || return 0
    # shellcheck disable=SC2086
    echo "$paths" | while read -r p; do
        [ -e "$ROOT/$p" ] || { echo "  ! missing source: $p" >&2; continue; }
        # models with external weights carry a .onnx_data sidecar
        run_cmd rsync -a --update --relative "$ROOT/./$p" "$ROOT/./$p"_data "$WIN_DIR/" 2>/dev/null ||
            run_cmd rsync -a --update --relative "$ROOT/./$p" "$WIN_DIR/"
    done
}

MODEL_ARGS=()
for m in ${MODELS+"${MODELS[@]}"}; do MODEL_ARGS+=(-m "$m"); done
MODE_ARGS=()
for m in ${MODES+"${MODES[@]}"}; do MODE_ARGS+=(-M "$m"); done
[ -n "$ITERS" ] && MODE_ARGS+=(--iters "$ITERS")

# reference data must be fetched before staging, which propagates it
if [ "$DRY" = 0 ]; then
    manifest fetch ${MODEL_ARGS+"${MODEL_ARGS[@]}"} || true
fi

[ "$BUILD" = 1 ] && [ "$HAS_WIN" = 1 ] && stage
if [ "$STAGE_ONLY" = 1 ]; then
    echo "== stage-only: done"
    exit 0
fi

# ---------------------------------------------------------------- 3. context
TAG="${TAG:-$(date +%Y-%m-%dT%H-%M-%S)}"
RUN_DIR="$ROOT/runs/$TAG"
mkdir -p "$RUN_DIR"

gpu_name="$(/mnt/c/Windows/System32/nvidia-smi.exe --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r')"
driver="$(/mnt/c/Windows/System32/nvidia-smi.exe --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r')"
jq -n \
    --arg host "$(hostname)" \
    --arg commit "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)" \
    --argjson dirty "$([ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ] && echo true || echo false)" \
    --arg ort "$(basename "$(dirname "$(dirname "$ROOT/third_party/onnxruntime/win-x64/lib/onnxruntime.dll")")" 2>/dev/null)" \
    --arg gpu "${gpu_name:-unknown}" \
    --arg driver "${driver:-unknown}" \
    --arg sample_ms "$SAMPLE_MS" \
    --argjson peaks "$("$HELPERS/peaks.py" "${gpu_name:-unknown}")" \
    '{host:$host, commit:$commit, dirty:$dirty, ort:$ort, gpu:$gpu, driver:$driver,
      sample_ms:($sample_ms|tonumber), when:(now|todate)} + $peaks' \
    >"$RUN_DIR/env.json"

# ---------------------------------------------------------------- 4. matrix
win_rel() { printf '%s' "${1//\//\\}"; } # models/zoo/x.onnx → models\zoo\x.onnx

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
    run_cmd rsync -a --update --relative "$ROOT/./$out" "$WIN_DIR/" >&2
    printf '%s' "$out"
}

# Starts the sampler on a process. The PID goes into SAMPLER_PID and not stdout:
# in a command substitution the process would be a child of the subshell, and
# `wait` in the parent could not wait on it.
SAMPLER_PID=""
start_sampler() { # $1 process name (without .exe), $2 destination csv (WSL)
    SAMPLER_PID=""
    [ "$SAMPLE_MS" = 0 ] && return 0
    [ "$DRY" = 1 ] && { echo "+ sample-metrics.ps1 -ProcessName $1" >&2; return 0; }
    rm -f "$WIN_DIR/.testsuite-stop" "$2"
    powershell.exe -NoProfile -ExecutionPolicy Bypass \
        -File "$WIN_PATH\\sample-metrics.ps1" \
        -ProcessName "$1" -Out "$WIN_PATH\\$(basename "$2")" \
        -IntervalMs "$SAMPLE_MS" -StopFile "$WIN_PATH\\.testsuite-stop" \
        >/dev/null 2>&1 </dev/null &
    SAMPLER_PID=$!
}

stop_sampler() {
    [ -z "$SAMPLER_PID" ] && return 0
    touch "$WIN_DIR/.testsuite-stop"
    wait "$SAMPLER_PID" 2>/dev/null
    SAMPLER_PID=""
    rm -f "$WIN_DIR/.testsuite-stop"
}

# Runs a Windows command in the staging directory, with the mode's environment.
win_exec() { # $1 `set VAR=…&&` prefix, $2 command line → stdout+stderr
    local env_prefix="$1" cmdline="$2"
    local common="set RUST_LOG=info&& set ORT_DYLIB_PATH=onnxruntime.dll"
    if [ "$DRY" = 1 ]; then
        echo "+ cmd.exe /c \"cd /d $WIN_PATH&& $common&& $env_prefix$cmdline\"" >&2
        return 0
    fi
    # </dev/null: without it, cmd.exe consumes the stdin of the matrix loop
    cmd.exe /c "cd /d $WIN_PATH&& $common&& $env_prefix$cmdline" 2>&1 </dev/null
}

mode_env() { # $1 mode, $2 runner
    case "$1" in
    cpu) [ "$2" = stt-app ] && printf 'set STT_NO_VULKAN=1&& ' ;;
    registry) : ;;
    compile) printf 'set VULKAN_EP_COMPILE=1&& ' ;;
    esac
}

# Runs a native Linux command from the repo root. The `standalone` mode's whole
# reason to exist: no cross-build, no rsync to /mnt/c, no cmd.exe, no Windows
# host held hostage — the cycle is seconds. The Vulkan device here is whatever
# WSL exposes (usually lavapipe), so `parse_run.py` marks these runs
# `perf_valid: false` on its own and only their **parity** verdict is a gate.
native_exec() { # $1 `VAR=v` env prefix (space separated), $2 command line
    local env_prefix="$1" cmdline="$2"
    local common="RUST_LOG=info ORT_DYLIB_PATH=$ROOT/third_party/onnxruntime/linux-x64/lib/libonnxruntime.so VULKAN_EP_PATH=$ROOT/target/release/libonnxruntime_ep_vulkan.so"
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
    if [ "$mode" = standalone ]; then
        # native paths, native binary, and `--standalone` adds the third backend:
        # the same generated tensors go through ORT, the EP and the facade in one
        # process. Two separate commands would compare two different inputs.
        cmdline="./target/release/model-runner $model_path --iters $iters --standalone"
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
        cmdline="set STT_BENCH=$iters&& stt-app.exe"
        for a in ${extra+"${extra[@]}"}; do cmdline+=" $(win_rel "$a")"; done
        cmdline+=" $(win_rel "$model_path")"
        proc=stt-app
    else
        cmdline="model-runner.exe $(win_rel "$model_path") --iters $iters"
        for a in ${extra+"${extra[@]}"}; do cmdline+=" $a"; done
        # official reference data: inputs and outputs expected by the model
        # authors, not generated by us
        if [ -n "$reference" ]; then
            cmdline+=" --reference $(win_rel "$reference")"
        fi
        proc=model-runner
    fi
    env_prefix="$(mode_env "$mode" "$runner")"

    # "clean" pass: wall times are measured without the profiler, which inserts
    # a timestamp after every dispatch
    metrics_tmp="$WIN_DIR/.testsuite-metrics.csv"
    start_sampler "$proc" "$metrics_tmp"
    win_exec "$env_prefix" "$cmdline" >"$out_dir/$mode.stdout.log"
    code=${PIPESTATUS[0]}
    stop_sampler
    # copy and don't move: on /mnt/c an still-open Windows handle makes mv fail
    [ -f "$metrics_tmp" ] && cp "$metrics_tmp" "$out_dir/$mode.metrics.csv" && rm -f "$metrics_tmp"

    # pass with profiler: Pareto, flushes, MB transferred
    stats_arg=()
    if [ "$stats" = 1 ]; then
        win_exec "${env_prefix}set VULKAN_EP_STATS=1&& " "$cmdline" \
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
