# Samples VRAM and GPU usage while a command runs, on Windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts\profile-run.ps1 `
#       -Command ".\model-runner.exe models\zoo\yolov8\yolov8n.onnx --iters 5"
#
# Windows equivalent of scripts/profile-run.sh (which uses /proc and is for Linux).
# NOTE: written but not run on this machine — see plan-test.md §3.
param(
    [Parameter(Mandatory = $true)][string]$Command,
    [double]$IntervalSeconds = 0.25
)

function Get-GpuSample {
    $raw = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
    if (-not $raw) { return $null }
    $parts = ($raw -split "`n")[0] -split ','
    [pscustomobject]@{ Util = [int]$parts[0].Trim(); MemMiB = [int]$parts[1].Trim() }
}

$idle = Get-GpuSample
if ($idle) { "GPU at rest: utilization $($idle.Util)%, memory $($idle.MemMiB) MiB" }

$proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $Command -PassThru -NoNewWindow
$samples = @()
while (-not $proc.HasExited) {
    Start-Sleep -Seconds $IntervalSeconds
    $gpu = Get-GpuSample
    if ($gpu) {
        # WorkingSet64 must be re-read: the property is cached
        $proc.Refresh()
        $samples += [pscustomobject]@{
            Util   = $gpu.Util
            MemMiB = $gpu.MemMiB
            RssMiB = [int]($proc.WorkingSet64 / 1MB)
            CpuSec = $proc.TotalProcessorTime.TotalSeconds
        }
    }
}
$proc.WaitForExit()

if ($samples.Count -eq 0) { "no samples collected"; exit $proc.ExitCode }
$peakUtil = ($samples | Measure-Object Util -Maximum).Maximum
$avgUtil = [int]($samples | Measure-Object Util -Average).Average
$peakMem = ($samples | Measure-Object MemMiB -Maximum).Maximum
$peakRss = ($samples | Measure-Object RssMiB -Maximum).Maximum
$cores = [Environment]::ProcessorCount
$cpuPct = [int](100 * $samples[-1].CpuSec / ($samples.Count * $IntervalSeconds))

""
"-- resources ($($samples.Count) samples every ${IntervalSeconds}s, $cores cores)"
"   GPU utilization avg $avgUtil%  peak $peakUtil%"
"   GPU memory      peak $peakMem MiB (at rest $($idle.MemMiB) MiB, delta $($peakMem - $idle.MemMiB) MiB)"
"   process CPU     ~$cpuPct%  (100% = 1 core)"
"   RSS             peak $peakRss MiB"
exit $proc.ExitCode
