param(
    [switch]$Watch,
    [int]$MaxSamples = 0,
    [int]$CurrentShares = -1,
    [double]$AvailableCash = -1,
    [int]$SellableShares = -1,
    [int]$TodayBoughtShares = -1,
    [switch]$SkipR5InputRefresh,
    [switch]$SkipPaperSignal
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$collector = Join-Path $projectRoot "scripts\collect_510300_primary_market.py"
$readinessAnalyzer = Join-Path $projectRoot "scripts\analyze_510300_primary_market_readiness.py"
$r5SignalRefresher = Join-Path $projectRoot "scripts\refresh_r5_daily_signal.py"
$signalGenerator = Join-Path $projectRoot "scripts\generate_510300_small_account_paper_signal.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到项目Python环境：$python"
}

$collectorArguments = @($collector)
if ($Watch) {
    $collectorArguments += "--watch"
}
if ($MaxSamples -gt 0) {
    $collectorArguments += @("--max-samples", $MaxSamples)
}

& $python @collectorArguments
if ($LASTEXITCODE -ne 0) {
    throw "PCF/IOPV采集失败，退出码：$LASTEXITCODE"
}

& $python $readinessAnalyzer
if ($LASTEXITCODE -ne 0) {
    throw "前向样本成熟度分析失败，退出码：$LASTEXITCODE"
}

if ($SkipPaperSignal) {
    Write-Output "研究采集模式：已跳过R5刷新与小账户纸面信号，不生成仓位映射或订单。"
}
else {
    $r5Arguments = @($r5SignalRefresher)
    if ($SkipR5InputRefresh) {
        $r5Arguments += "--skip-refresh"
    }
    & $python @r5Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "R5基础信号刷新失败，退出码：$LASTEXITCODE"
    }

    $signalArguments = @($signalGenerator)
    if ($CurrentShares -ge 0) {
        $signalArguments += @("--current-shares", $CurrentShares)
    }
    if ($AvailableCash -ge 0) {
        $signalArguments += @("--available-cash", $AvailableCash)
    }
    if ($SellableShares -ge 0) {
        $signalArguments += @("--sellable-shares", $SellableShares)
    }
    if ($TodayBoughtShares -ge 0) {
        $signalArguments += @("--today-bought-shares", $TodayBoughtShares)
    }

    & $python @signalArguments
    if ($LASTEXITCODE -ne 0) {
        throw "2万元纸面信号生成失败，退出码：$LASTEXITCODE"
    }
}
