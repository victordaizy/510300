param(
    [string]$TaskName = "Codex-Industry-Expectation-Gap-Forward-Operations-V1"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$outcomeRefresher = Join-Path $projectRoot "scripts\refresh_industry_expectation_gap_outcome_inputs.py"
$runner = Join-Path $projectRoot "scripts\run_industry_expectation_gap_forward_operations_v1.py"
$outcomeStatusFile = Join-Path $projectRoot "reports\data_quality\industry_expectation_gap_outcome_refresh_status.json"
$operationsStatusFile = Join-Path $projectRoot "reports\forward\industry_expectation_gap_v1_evaluation\operations_status.json"
$taskStatusFile = Join-Path $projectRoot "reports\forward\industry_expectation_gap_v1_evaluation\operations_task_status.json"
$taskReceiptDirectory = Join-Path $projectRoot "reports\forward\industry_expectation_gap_v1_evaluation\task_runs"
$logDirectory = Join-Path $projectRoot "output\industry_expectation_gap_forward_logs"
$dateStamp = Get-Date -Format "yyyyMMdd"
$logFile = Join-Path $logDirectory "$dateStamp.task.log"
$startedAt = Get-Date
$receiptId = "{0}_{1}" -f $startedAt.ToUniversalTime().ToString("yyyyMMddTHHmmssfffffffZ"), $PID
$taskReceiptFile = Join-Path $taskReceiptDirectory "$receiptId.json"

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $taskReceiptDirectory -Force | Out-Null

$runStatus = "FAILED"
$taskExitCode = 1
$errorText = $null

try {
    & $python $outcomeRefresher *>> $logFile
    $outcomeExitCode = $LASTEXITCODE
    if ($outcomeExitCode -ne 0) {
        throw "行业结果输入刷新器退出码为$outcomeExitCode"
    }
    & $python $runner --evaluate-if-ready *>> $logFile
    $taskExitCode = $LASTEXITCODE
    if ($taskExitCode -ne 0) {
        throw "行业预期差运行器退出码为$taskExitCode"
    }
    $runStatus = "SUCCESS"
}
catch {
    $errorText = "{0}: {1}" -f $_.Exception.GetType().Name, $_.Exception.Message
    $errorText | Out-File -LiteralPath $logFile -Append -Encoding utf8
    $taskExitCode = 1
}
finally {
    $outcomeStatus = $null
    if (Test-Path -LiteralPath $outcomeStatusFile) {
        $outcomeStatus = Get-Content -LiteralPath $outcomeStatusFile -Raw -Encoding utf8 | ConvertFrom-Json
    }
    $operationsStatus = $null
    if (Test-Path -LiteralPath $operationsStatusFile) {
        $operationsStatus = Get-Content -LiteralPath $operationsStatusFile -Raw -Encoding utf8 | ConvertFrom-Json
    }
    $status = [ordered]@{
        schema_version = "1.0.0"
        receipt_id = $receiptId
        immutable_receipt = $true
        task_name = $TaskName
        run_status = $runStatus
        started_at = $startedAt.ToString("o")
        ended_at = (Get-Date).ToString("o")
        task_exit_code = $taskExitCode
        error = $errorText
        outcome_refresh_run_status = if ($null -ne $outcomeStatus) { $outcomeStatus.run_status } else { $null }
        outcome_refresh_collection_status = if ($null -ne $outcomeStatus) { $outcomeStatus.collection_status } else { $null }
        outcome_refresh_target_date = if ($null -ne $outcomeStatus) { $outcomeStatus.target_date } else { $null }
        research_status = if ($null -ne $operationsStatus) { $operationsStatus.status } else { $null }
        origin_collection_status = if ($null -ne $operationsStatus) { $operationsStatus.origin_collection.status } else { $null }
        outcome_collection_status = if ($null -ne $operationsStatus) { $operationsStatus.outcome_collection.status } else { $null }
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
