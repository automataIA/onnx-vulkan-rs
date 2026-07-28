#!/bin/bash
# Samples resource usage during a command, to understand **where** the
# computation happens: process and system CPU, RSS, and NVIDIA GPU (utilization
# and memory).
#
#   ./scripts/profile-run.sh ./target/release/model-runner models/zoo/... --iters 3
#
# Reading note: if the Vulkan device is lavapipe, the NVIDIA GPU stays idle by
# construction — lavapipe is a software rasterizer, so "VRAM" is system RAM and
# the compute units are CPU cores. A GPU usage ~0 with many saturated CPU cores
# is exactly what you'd expect, not an anomaly.
set -u

INTERVAL="${PROFILE_INTERVAL:-0.5}"
NCPU=$(nproc)
have_nvidia=0
command -v nvidia-smi >/dev/null 2>&1 && have_nvidia=1

gpu_sample() { # → "util_pct mem_used_MiB"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null |
        head -1 | tr -d ' ' | tr ',' ' '
}

if [ "$have_nvidia" = 1 ]; then
    read -r gpu_util0 gpu_mem0 <<<"$(gpu_sample)"
    echo "NVIDIA GPU at rest: utilization ${gpu_util0}%, memory ${gpu_mem0} MiB"
fi

"$@" &
PID=$!

# starting CPU times (jiffies)
read_proc_cpu() { awk '{print $14 + $15}' "/proc/$1/stat" 2>/dev/null; }
HZ=$(getconf CLK_TCK)
prev_cpu=$(read_proc_cpu "$PID")
prev_t=$(date +%s.%N)

peak_rss=0 peak_cpu=0 peak_gpu=0 peak_gpu_mem=0 sum_cpu=0 sum_gpu=0 samples=0

while kill -0 "$PID" 2>/dev/null; do
    sleep "$INTERVAL"
    rss=$(awk '/VmRSS/{print $2}' "/proc/$PID/status" 2>/dev/null)
    cur_cpu=$(read_proc_cpu "$PID")
    cur_t=$(date +%s.%N)
    [ -z "$rss" ] || [ -z "$cur_cpu" ] && continue

    # process CPU percentage: 100% = one saturated core
    cpu=$(awk -v c="$cur_cpu" -v p="$prev_cpu" -v t="$cur_t" -v pt="$prev_t" -v hz="$HZ" \
        'BEGIN { d = t - pt; if (d <= 0) print 0; else printf "%.0f", 100 * (c - p) / hz / d }')
    prev_cpu=$cur_cpu prev_t=$cur_t

    gpu=0 gpu_mem=0
    if [ "$have_nvidia" = 1 ]; then
        read -r gpu gpu_mem <<<"$(gpu_sample)"
        gpu=${gpu:-0} gpu_mem=${gpu_mem:-0}
    fi

    [ "$rss" -gt "$peak_rss" ] && peak_rss=$rss
    [ "$cpu" -gt "$peak_cpu" ] && peak_cpu=$cpu
    [ "$gpu" -gt "$peak_gpu" ] && peak_gpu=$gpu
    [ "$gpu_mem" -gt "$peak_gpu_mem" ] && peak_gpu_mem=$gpu_mem
    sum_cpu=$((sum_cpu + cpu)) sum_gpu=$((sum_gpu + gpu)) samples=$((samples + 1))
done
wait "$PID"
status=$?

echo
echo "── resources (${samples} samples every ${INTERVAL}s, ${NCPU} cores available)"
if [ "$samples" -gt 0 ]; then
    printf '   process CPU    avg %s%%  peak %s%%   (100%% = 1 core; max %s%%)\n' \
        "$((sum_cpu / samples))" "$peak_cpu" "$((NCPU * 100))"
    printf '   RSS            peak %s MiB\n' "$((peak_rss / 1024))"
    if [ "$have_nvidia" = 1 ]; then
        printf '   NVIDIA GPU     avg %s%%  peak %s%%   memory peak %s MiB (at rest %s MiB)\n' \
            "$((sum_gpu / samples))" "$peak_gpu" "$peak_gpu_mem" "${gpu_mem0:-0}"
    else
        echo "   NVIDIA GPU     nvidia-smi not available"
    fi
fi
exit "$status"
