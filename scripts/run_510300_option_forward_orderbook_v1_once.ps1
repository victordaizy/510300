param(
    [Parameter(Mandatory = $false)]
    [ValidatePattern('^\d{4}-\d{2}-\d{2}$')]
    [string]$TradeDate = (Get-Date -Format 'yyyy-MM-dd')
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$CollectorPath = Join-Path $ProjectRoot 'scripts\collect_510300_option_forward_orderbook_v1.py'
$AuditorPath = Join-Path $ProjectRoot 'scripts\audit_510300_option_forward_orderbook_v1.py'
$LogDirectory = Join-Path $ProjectRoot 'reports\forward\510300_option_orderbook_v1\runner_logs'
$ParsedDate = [datetime]::ParseExact(
    $TradeDate,
    'yyyy-MM-dd',
    [System.Globalization.CultureInfo]::InvariantCulture
)
$TargetTime = $ParsedDate.AddHours(14).AddMinutes(59)
$WindowEnd = $ParsedDate.AddHours(15).AddMinutes(5).AddSeconds(30)
$LogPath = Join-Path $LogDirectory ($ParsedDate.ToString('yyyyMMdd') + '_runner.log')

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

function Write-RunnerLog {
    param([string]$Message)
    $Line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss.fffK'), $Message
    Add-Content -LiteralPath $LogPath -Value $Line -Encoding utf8
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    Write-RunnerLog "NO_VIEW: Python runtime is missing: $PythonPath"
    exit 10
}

$CurrentTime = Get-Date
if ($CurrentTime -gt $WindowEnd) {
    Write-RunnerLog 'NO_VIEW: current time is later than the frozen capture window; backfill is forbidden.'
    exit 11
}

if ($CurrentTime -lt $TargetTime) {
    $Seconds = [math]::Ceiling(($TargetTime - $CurrentTime).TotalSeconds)
    Write-RunnerLog "Waiting $Seconds seconds; target capture time is $($TargetTime.ToString('yyyy-MM-dd HH:mm:ss'))."
    Start-Sleep -Seconds $Seconds
}

Write-RunnerLog "Starting collection; expected trade date=$TradeDate."
& $PythonPath $CollectorPath --expected-trade-date $TradeDate *>> $LogPath
$CollectorExit = $LASTEXITCODE
Write-RunnerLog "Collector exit code=$CollectorExit."

& $PythonPath $AuditorPath *>> $LogPath
$AuditorExit = $LASTEXITCODE
Write-RunnerLog "Auditor exit code=$AuditorExit."

if ($CollectorExit -ne 0 -or $AuditorExit -ne 0) {
    exit 12
}
exit 0
