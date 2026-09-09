param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $projectRoot ".venv\Scripts\pythonw.exe"
$runner = Join-Path $projectRoot "scripts\run_priority_forward_supervisor.py"
$statusFile = Join-Path $projectRoot "reports\audit\priority_forward_supervisor_status.json"

if (-not (Test-Path -LiteralPath $pythonw)) {
    throw "pythonw.exe not found: $pythonw"
}
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Supervisor runner not found: $runner"
}

if (Test-Path -LiteralPath $statusFile) {
    try {
        $status = Get-Content -LiteralPath $statusFile -Raw -Encoding utf8 | ConvertFrom-Json
        if ($null -ne $status.pid) {
            $existing = Get-Process -Id ([int]$status.pid) -ErrorAction SilentlyContinue
            if ($null -ne $existing) {
                Write-Output "Supervisor already running. PID=$($status.pid)"
                exit 0
            }
        }
    }
    catch {
        Write-Output "Existing status could not be verified; lock file remains authoritative."
    }
}

$arguments = "`"$runner`""
$process = Start-Process `
    -FilePath $pythonw `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -PassThru
Start-Sleep -Seconds 2
if ($process.HasExited) {
    throw "Supervisor exited during startup. ExitCode=$($process.ExitCode)"
}
Write-Output "Supervisor started. PID=$($process.Id)"

