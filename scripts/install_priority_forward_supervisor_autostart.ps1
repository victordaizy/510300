param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $projectRoot "scripts\start_priority_forward_supervisor.ps1"
$runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$valueName = "CodexPriorityForwardSupervisorV1"

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Supervisor launcher not found: $launcher"
}

$command = "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$launcher`""
New-Item -Path $runKey -Force | Out-Null
New-ItemProperty -Path $runKey -Name $valueName -Value $command -PropertyType String -Force | Out-Null
$installed = Get-ItemPropertyValue -Path $runKey -Name $valueName
if ($installed -ne $command) {
    throw "Login autostart registry value verification failed."
}

& $launcher
if ($LASTEXITCODE -ne 0) {
    throw "Supervisor launcher failed with exit code $LASTEXITCODE"
}
Write-Output "Login autostart installed: $valueName"
Write-Output "Automation scope: current user login session only"
Write-Output "Trading authorization: false"

