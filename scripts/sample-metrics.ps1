# System metrics sampler for the test suite (scripts/testsuite.sh).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sample-metrics.ps1 `
#       -ProcessName model-runner -Out metrics.csv -IntervalMs 100 -StopFile stop.flag
#
# Writes a CSV `t_ms,sm,mem_pct,fb_mb,proc_rss_mb,proc_cpu_pct` preceded by a
# comment line with the VRAM idle baseline. It stops when the observed process
# exits or when $StopFile appears.
#
# Deliberate limitations, declared in the data (see plan-test-suite.md §5.2):
#   - `fb_mb` is the **global** GPU memory, not the process's: under WDDM a Vulkan
#     app may not appear in --query-compute-apps. The consumer uses the delta
#     against `idle_fb_mb` and marks it `vram_source: global-delta`.
#   - `proc_cpu_pct` is in units of "100% = one saturated core".
param(
    [Parameter(Mandatory = $true)][string]$ProcessName,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$IntervalMs = 100,
    [string]$StopFile = "",
    # how long to wait for the process to appear before giving up
    [int]$WaitSeconds = 300
)

$ErrorActionPreference = "Continue"

function Get-Gpu {
    # utilization.gpu = % of time with at least one kernel active (the `sm`
    # column of dmon), utilization.memory = % of time with traffic toward VRAM.
    $raw = & nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used `
        --format=csv,noheader,nounits 2>$null
    if (-not $raw) { return $null }
    $p = ($raw -split "`n")[0] -split ','
    if ($p.Count -lt 3) { return $null }
    [pscustomobject]@{
        Sm     = [int]$p[0].Trim()
        MemPct = [int]$p[1].Trim()
        FbMb   = [int]$p[2].Trim()
    }
}

function Get-Target {
    $procs = @(Get-Process -Name $ProcessName -ErrorAction SilentlyContinue)
    if ($procs.Count -eq 0) { return $null }
    $rss = 0; $cpu = 0.0
    foreach ($p in $procs) {
        $p.Refresh()
        $rss += $p.WorkingSet64
        $cpu += $p.TotalProcessorTime.TotalSeconds
    }
    [pscustomobject]@{ RssMb = [int]($rss / 1MB); CpuSec = $cpu }
}

$idle = Get-Gpu
$writer = [System.IO.StreamWriter]::new($Out, $false)
$writer.AutoFlush = $true
$idleFb = if ($idle) { $idle.FbMb } else { -1 }
$writer.WriteLine("# idle_fb_mb=$idleFb interval_ms=$IntervalMs process=$ProcessName")
$writer.WriteLine("t_ms,sm,mem_pct,fb_mb,proc_rss_mb,proc_cpu_pct")

function Test-Stop { return ($StopFile -ne "" -and (Test-Path -LiteralPath $StopFile)) }

# 1) wait for the process to appear (the suite starts the sampler first)
$clock = [System.Diagnostics.Stopwatch]::StartNew()
while (-not (Get-Target)) {
    if ((Test-Stop) -or ($clock.Elapsed.TotalSeconds -gt $WaitSeconds)) {
        $writer.Close(); exit 0
    }
    Start-Sleep -Milliseconds 20
}

# 2) sampling until the process ends or the stop signal arrives
$t0 = [System.Diagnostics.Stopwatch]::StartNew()
$prevCpu = $null; $prevT = 0.0
while ($true) {
    $target = Get-Target
    if (-not $target) { break }
    $gpu = Get-Gpu
    $now = $t0.Elapsed.TotalMilliseconds
    $cpuPct = 0
    if ($null -ne $prevCpu) {
        $dt = ($now - $prevT) / 1000.0
        if ($dt -gt 0) { $cpuPct = [int](100 * ($target.CpuSec - $prevCpu) / $dt) }
    }
    $prevCpu = $target.CpuSec; $prevT = $now
    $sm = if ($gpu) { $gpu.Sm } else { -1 }
    $memPct = if ($gpu) { $gpu.MemPct } else { -1 }
    $fb = if ($gpu) { $gpu.FbMb } else { -1 }
    # the parentheses are required: `-f` binds tighter than the comma, and
    # without them the format receives only one argument
    $writer.WriteLine("{0:F0},{1},{2},{3},{4},{5}" -f @($now, $sm, $memPct, $fb, $target.RssMb, $cpuPct))
    if (Test-Stop) { break }
    Start-Sleep -Milliseconds $IntervalMs
}
$writer.Close()
