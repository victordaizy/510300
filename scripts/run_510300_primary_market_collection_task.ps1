param(
    [string]$TaskName = "Codex-510300-Primary-Market-Collector"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$projectRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $projectRoot "scripts\run_510300_primary_market_forward.ps1"
$readinessFile = Join-Path $projectRoot "reports\data_quality\510300_primary_market_readiness.json"
$taskStatusFile = Join-Path $projectRoot "reports\data_quality\510300_primary_market_task_status.json"
$taskReceiptDirectory = Join-Path $projectRoot "reports\data_quality\primary_market_task_runs"
$tradingCalendarFile = Join-Path $projectRoot "data\reference\sse_trade_calendar_2026.csv"
$logDirectory = Join-Path $projectRoot "output\primary_market_forward_logs"
$dateStamp = Get-Date -Format "yyyyMMdd"
$logFile = Join-Path $logDirectory "$dateStamp.task.log"
$startedAt = Get-Date
$receiptId = "{0}_{1}" -f $startedAt.ToUniversalTime().ToString("yyyyMMddTHHmmssfffffffZ"), $PID
$taskReceiptFile = Join-Path $taskReceiptDirectory "$receiptId.json"

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $taskReceiptDirectory -Force | Out-Null

$runStatus = "FAILED"
$collectionStatus = "PROGRAM_FAILED"
$taskExitCode = 1
$errorText = $null

try {
    & $runner -Watch -SkipPaperSignal *>> $logFile
    $taskExitCode = $LASTEXITCODE
    if ($taskExitCode -ne 0) {
        throw "研究采集运行器退出码为$taskExitCode"
    }
    $runStatus = "SUCCESS"
    $collectionStatus = "COMPLETED_OR_MARKET_CLOSED"
}
catch {
    $errorText = "{0}: {1}" -f $_.Exception.GetType().Name, $_.Exception.Message
    $errorText | Out-File -LiteralPath $logFile -Append -Encoding utf8
    $taskExitCode = 1
}
finally {
    $readiness = $null
    if (Test-Path -LiteralPath $readinessFile) {
        $readiness = Get-Content -LiteralPath $readinessFile -Raw -Encoding utf8 | ConvertFrom-Json
    }
    $today = (Get-Date).ToString("yyyy-MM-dd")
    if ($runStatus -eq "SUCCESS") {
        if ($null -eq $readiness -or $readiness.last_observed_trade_date -ne $today) {
            if (-not (Test-Path -LiteralPath $tradingCalendarFile)) {
                $runStatus = "FAILED"
                $collectionStatus = "PROGRAM_FAILED_MISSING_TRADING_CALENDAR"
                $taskExitCode = 1
                $errorText = "交易日历缺失：$tradingCalendarFile"
            }
            else {
                $tradingDates = @(Import-Csv -LiteralPath $tradingCalendarFile | ForEach-Object { $_.trade_date })
                if ($tradingDates -contains $today) {
                    $runStatus = "FAILED"
                    $collectionStatus = "FAILED_NO_CURRENT_TRADING_DAY_DATA"
                    $taskExitCode = 1
                    $errorText = "交易日$today没有当天PCF/IOPV证据；不得按休市跳过。"
                }
                else {
                    $collectionStatus = "SKIPPED_NON_TRADING_DAY"
                }
            }
        }
        else {
            $todayQuality = $readiness.daily_quality | Where-Object { $_.trade_date -eq $today }
            if ($null -ne $todayQuality -and $todayQuality.complete_quality_day) {
                $collectionStatus = "COMPLETE_QUALITY_DAY"
            }
            else {
                $collectionStatus = "COLLECTED_INCOMPLETE_QUALITY_DAY"
            }
        }
    }

    $status = [ordered]@{
        schema_version = "1.0.0"
        receipt_id = $receiptId
        immutable_receipt = $true
        task_name = $TaskName
        run_status = $runStatus
        collection_status = $collectionStatus
        started_at = $startedAt.ToString("o")
        ended_at = (Get-Date).ToString("o")
        task_exit_code = $taskExitCode
        error = $errorText
        latest_observed_trade_date = if ($null -ne $readiness) { $readiness.last_observed_trade_date } else { $null }
        full_coverage_days = if ($null -ne $readiness) { $readiness.full_coverage_days } else { $null }
        readiness_status = if ($null -ne $readiness) { $readiness.status } else { $null }
        position_mapping_enabled = $false
        order_generation_enabled = $false
        broker_connection_enabled = $false
        live_trading_enabled = $false
        receipt_file = $taskReceiptFile.Substring($projectRoot.Length + 1).Replace("\", "/")
        log_file = $logFile.Substring($projectRoot.Length + 1).Replace("\", "/")
    }
    $json = $status | ConvertTo-Json -Depth 5
    if (Test-Path -LiteralPath $taskReceiptFile) {
        throw "任务运行收据已存在，禁止覆盖：$taskReceiptFile"
    }
    $receiptTemporary = "$taskReceiptFile.tmp"
    [System.IO.File]::WriteAllText($receiptTemporary, $json, (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $receiptTemporary -Destination $taskReceiptFile
    $temporary = "$taskStatusFile.tmp"
    [System.IO.File]::WriteAllText($temporary, $json, (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $taskStatusFile -Force
}

exit $taskExitCode
