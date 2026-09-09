param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-f0-9]{64}$")]
    [string]$ExpectedManifestSha256,
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$manifest = Join-Path $projectRoot "config\priority_forward_research_operations_v1_11_1_manifest.json"
$config = Join-Path $projectRoot "config\priority_forward_supervisor_v1_7.yaml"
$taskName = "Codex-510300-Primary-Market-Collector"
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if ([string]$task.State -eq "Running") {
    throw "采集任务正在运行，本次不改变运行中的任务。"
}
if (@($task.Actions).Count -ne 1) {
    throw "既有任务动作数量与预期不一致，停止更新。"
}
Push-Location -LiteralPath $projectRoot
try {
    $verification = & $python -m scripts.run_priority_forward_codex_automation_v1_11_1 --verify-only --expected-manifest-sha256 $ExpectedManifestSha256
    if ($LASTEXITCODE -ne 0) {
        throw "新版本冻结校验未通过，计划任务未更新。"
    }
}
finally {
    Pop-Location
}
$arguments = "-m scripts.run_priority_forward_codex_automation_v1_11_1 --phase morning --config `"$config`" --manifest `"$manifest`" --expected-manifest-sha256 $ExpectedManifestSha256"
$preview = [ordered]@{
    status = "VERIFIED_DEPLOYMENT_PREVIEW"
    task_name = $taskName
    before_execute = [string]$task.Actions[0].Execute
    before_arguments = [string]$task.Actions[0].Arguments
    proposed_execute = $python
    proposed_arguments = $arguments
    proposed_working_directory = $projectRoot
    schedule_and_principal_preserved = $true
    collection_triggered = $false
    data_purchase_budget_cny = 0
    position_impact = 0
}
if (-not $Apply) {
    $preview | ConvertTo-Json -Depth 6
    exit 0
}
$directory = Join-Path $projectRoot "reports\audit\priority_forward_v1_11_1_deployment"
New-Item -ItemType Directory -Path $directory -Force | Out-Null
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffffffZ")
$backup = Join-Path $directory ($stamp + ".previous_task.xml")
[System.IO.File]::WriteAllText($backup, (Export-ScheduledTask -TaskName $taskName -TaskPath $task.TaskPath), (New-Object System.Text.UTF8Encoding($false)))
$action = New-ScheduledTaskAction -Execute $python -Argument $arguments -WorkingDirectory $projectRoot
Set-ScheduledTask -TaskName $taskName -TaskPath $task.TaskPath -Action $action | Out-Null
$updated = Get-ScheduledTask -TaskName $taskName -TaskPath $task.TaskPath
$newAction = @($updated.Actions)[0]
if ([string]$newAction.Arguments -ne $arguments -or [string]$newAction.Execute -ne $python -or [string]$newAction.WorkingDirectory -ne $projectRoot) {
    Set-ScheduledTask -TaskName $taskName -TaskPath $task.TaskPath -Action @($task.Actions) | Out-Null
    throw "注册后的入口不匹配，已恢复原动作。"
}
$info = Get-ScheduledTaskInfo -TaskName $taskName -TaskPath $task.TaskPath
$preview.status = "DEPLOYED_AWAITING_NEXT_ELIGIBLE_WINDOW"
$preview["deployed_at"] = (Get-Date).ToString("o")
$preview["previous_task_xml"] = $backup
$preview["next_run_time"] = $info.NextRunTime.ToString("o")
$preview["manifest_sha256"] = $ExpectedManifestSha256
$preview["principal_logon_type"] = [string]$updated.Principal.LogonType
$preview["restart_count"] = [int]$updated.Settings.RestartCount
$preview["start_when_available"] = [bool]$updated.Settings.StartWhenAvailable
$receipt = Join-Path $directory ($stamp + ".deployment.json")
[System.IO.File]::WriteAllText($receipt, ($preview | ConvertTo-Json -Depth 6), (New-Object System.Text.UTF8Encoding($false)))
$preview | ConvertTo-Json -Depth 6
exit 0
