param(
    [int]$CurrentShares = -1,
    [double]$AvailableCash = -1,
    [int]$SellableShares = -1,
    [int]$TodayBoughtShares = -1
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$projectRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $projectRoot "scripts\run_510300_primary_market_forward.ps1"
$logDirectory = Join-Path $projectRoot "output\primary_market_forward_logs"
$dateStamp = Get-Date -Format "yyyyMMdd"
$standardOutput = Join-Path $logDirectory "$dateStamp.stdout.log"
$standardError = Join-Path $logDirectory "$dateStamp.stderr.log"

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

$runnerArguments = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    "`"$runner`"",
    "-Watch"
)
if ($CurrentShares -ge 0) {
    $runnerArguments += @("-CurrentShares", $CurrentShares)
}
if ($AvailableCash -ge 0) {
    $runnerArguments += @("-AvailableCash", $AvailableCash)
}
if ($SellableShares -ge 0) {
    $runnerArguments += @("-SellableShares", $SellableShares)
}
if ($TodayBoughtShares -ge 0) {
    $runnerArguments += @("-TodayBoughtShares", $TodayBoughtShares)
}

$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $runnerArguments `
    -WindowStyle Hidden `
    -RedirectStandardOutput $standardOutput `
    -RedirectStandardError $standardError `
    -PassThru

Write-Output "510300前向采集已在后台启动，进程ID：$($process.Id)"
Write-Output "标准日志：$standardOutput"
Write-Output "错误日志：$standardError"
