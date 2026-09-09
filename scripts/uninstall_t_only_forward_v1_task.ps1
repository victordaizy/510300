param(
    [string]$TaskName = "Codex-510300-T-Only-Forward-V1"
)

$ErrorActionPreference = "Stop"
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $existing) {
    Write-Output "计划任务不存在：$TaskName"
    exit 0
}
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Output "计划任务已移除：$TaskName"
Write-Output "历史影子账本、报告和日志未删除，可继续审计。"
