$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logDirectory = Join-Path $projectRoot "logs"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python interpreter not found: $pythonPath"
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
Set-Location -LiteralPath $projectRoot

$steps = @(
    @{ Module = "scripts.download_510300_15m"; Log = "download_510300_15m_latest.log" },
    @{ Module = "scripts.download_510300_minute_auxiliary"; Log = "download_510300_minute_auxiliary_latest.log" },
    @{ Module = "scripts.quality_check_510300_15m"; Log = "quality_510300_15m_latest.log" },
    @{ Module = "scripts.cross_check_510300_minute_sources"; Log = "cross_510300_minute_latest.log" }
)

foreach ($step in $steps) {
    $logPath = Join-Path $logDirectory $step.Log
    $errorLogPath = "$logPath.stderr"
    Write-Host "Running: $($step.Module)"
    $process = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList "-m", $step.Module `
        -Wait `
        -NoNewWindow `
        -PassThru `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError $errorLogPath
    if ($process.ExitCode -ne 0) {
        throw "Minute data pipeline failed: $($step.Module); log: $logPath"
    }
}

Write-Host "Minute data refresh and audit completed."
