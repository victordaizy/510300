param(
    [string]$TaskName = "Codex-Industry-Expectation-Gap-Forward-Operations-V1.2"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$outcomeRefresher = Join-Path $projectRoot "scripts\refresh_industry_expectation_gap_outcome_inputs_v1_2.py"
$snapshotRunner = Join-Path $projectRoot "scripts\freeze_industry_expectation_gap_outcome_inputs_v1_2.py"
$operationsRunner = Join-Path $projectRoot "scripts\run_industry_expectation_gap_forward_operations_v1_2.py"
$outcomeStatusFile = Join-Path $projectRoot "reports\data_quality\industry_expectation_gap_outcome_refresh_status_v1_2.json"
$operationsStatusFile = Join-Path $projectRoot "reports\forward\industry_expectation_gap_v1_evaluation\operations_status_v1_2.json"
$taskStatusFile = Join-Path $projectRoot "reports\forward\industry_expectation_gap_v1_evaluation\operations_task_status_v1_2.json"
$taskReceiptDirectory = Join-Path $projectRoot "reports\forward\industry_expectation_gap_v1_evaluation\task_runs_v1_2"
$logDirectory = Join-Path $projectRoot "output\industry_expectation_gap_forward_logs_v1_2"
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
$externalSourceBlocked = $false

try {
    & $python $outcomeRefresher *>> $logFile
    $outcomeExitCode = $LASTEXITCODE
    if ($outcomeExitCode -eq 3) {
        $externalSourceBlocked = $true
    }
    elseif ($outcomeExitCode -ne 0) {
        throw "行业结果输入零付费闸门程序失败，退出码为$outcomeExitCode"
    }

    & $python $snapshotRunner *>> $logFile
    if ($LASTEXITCODE -ne 0) {
        throw "行业结果输入内容寻址快照失败，退出码为$LASTEXITCODE"
    }

    $operationsArguments = @($operationsRunner)
    if (-not $externalSourceBlocked) {
        $operationsArguments += "--evaluate-if-ready"
    }
    & $python @operationsArguments *>> $logFile
    if ($LASTEXITCODE -ne 0) {
        throw "行业预期差 V1.2 状态运行失败，退出码为$LASTEXITCODE"
    }

    if ($externalSourceBlocked) {
        $runStatus = "FAILED"
        $collectionStatus = "EXTERNAL_FREE_SOURCE_FAILED"
        $taskExitCode = 3
        $errorText = "零付费调整后成分股结果源未通过资格门槛；未调用付费接口。"
    }
    else {
        $runStatus = "SUCCESS"
        $collectionStatus = "COMPLETED_ZERO_PAID_INPUT_VERIFIED"
        $taskExitCode = 0
    }
}
catch {
    $errorText = "{0}: {1}" -f $_.Exception.GetType().Name, $_.Exception.Message
    $errorText | Out-File -LiteralPath $logFile -Append -Encoding utf8
    $runStatus = "FAILED"
    $collectionStatus = "PROGRAM_FAILED"
    $taskExitCode = 1
}
finally {
    $outcomeStatus = $null
    if (Test-Path -LiteralPath $outcomeStatusFile -PathType Leaf) {
        try {
            $outcomeStatus = Get-Content -LiteralPath $outcomeStatusFile -Raw -Encoding utf8 | ConvertFrom-Json
        }
        catch {
            $runStatus = "FAILED"
            $collectionStatus = "PROGRAM_FAILED_INVALID_OUTCOME_STATUS"
            $taskExitCode = 1
            $errorText = "行业结果输入状态无法解析：$($_.Exception.Message)"
        }
    }
    $operationsStatus = $null
    if (Test-Path -LiteralPath $operationsStatusFile -PathType Leaf) {
        try {
            $operationsStatus = Get-Content -LiteralPath $operationsStatusFile -Raw -Encoding utf8 | ConvertFrom-Json
        }
        catch {
            $runStatus = "FAILED"
            $collectionStatus = "PROGRAM_FAILED_INVALID_OPERATIONS_STATUS"
            $taskExitCode = 1
            $errorText = "行业运行状态无法解析：$($_.Exception.Message)"
        }
    }
    $status = [ordered]@{
        schema_version = "1.2.0"
        receipt_id = $receiptId
        immutable_receipt = $true
        task_name = $TaskName
        run_status = $runStatus
        collection_status = $collectionStatus
        started_at = $startedAt.ToString("o")
        ended_at = (Get-Date).ToString("o")
        task_exit_code = $taskExitCode
        error = $errorText
        outcome_refresh_run_status = if ($null -ne $outcomeStatus) { $outcomeStatus.run_status } else { $null }
        outcome_refresh_collection_status = if ($null -ne $outcomeStatus) { $outcomeStatus.collection_status } else { $null }
        outcome_refresh_failure_class = if ($null -ne $outcomeStatus) { $outcomeStatus.failure_class } else { $null }
        outcome_refresh_target_date = if ($null -ne $outcomeStatus) { $outcomeStatus.target_date } else { $null }
        paid_provider_call_enabled = $false
        external_request_attempted = if ($null -ne $outcomeStatus) { $outcomeStatus.external_request_attempted } else { $false }
        research_status = if ($null -ne $operationsStatus) { $operationsStatus.status } else { $null }
        origin_collection_status = if ($null -ne $operationsStatus) { $operationsStatus.origin_collection.status } else { $null }
        outcome_collection_status = if ($null -ne $operationsStatus) { $operationsStatus.outcome_collection.status } else { $null }
        input_snapshot_id = if ($null -ne $operationsStatus) { $operationsStatus.input_snapshot.snapshot_id } else { $null }
        shared_latest_consumed_by_evaluation = if ($null -ne $operationsStatus) { $operationsStatus.input_snapshot.shared_latest_consumed_by_evaluation } else { $null }
        position_mapping_enabled = $false
        order_generation_enabled = $false
        broker_connection_enabled = $false
        live_trading_enabled = $false
        receipt_file = $taskReceiptFile.Substring($projectRoot.Length + 1).Replace("\", "/")
        log_file = $logFile.Substring($projectRoot.Length + 1).Replace("\", "/")
    }
    $json = $status | ConvertTo-Json -Depth 8
    if (Test-Path -LiteralPath $taskReceiptFile) {
        throw "行业 V1.2 任务收据已存在，禁止覆盖：$taskReceiptFile"
    }
    $receiptTemporary = "$taskReceiptFile.$PID.tmp"
    [System.IO.File]::WriteAllText($receiptTemporary, $json, (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $receiptTemporary -Destination $taskReceiptFile
    $statusTemporary = "$taskStatusFile.$PID.tmp"
    [System.IO.File]::WriteAllText($statusTemporary, $json, (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $statusTemporary -Destination $taskStatusFile -Force
}

exit $taskExitCode
