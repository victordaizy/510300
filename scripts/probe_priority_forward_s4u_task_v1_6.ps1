param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$projectRoot = Split-Path -Parent $PSScriptRoot
$target = [System.IO.Path]::GetFullPath($OutputPath)
$targetDirectory = Split-Path -Parent $target
New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null
$startedAt = Get-Date
$probeStatus = "FAILED"
$failure = $null
$responseCount = 0
$securityProtocol = [Net.ServicePointManager]::SecurityProtocol

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $parameters = @{
        sqlId = "COMMON_SSE_CP_GPJCTPZ_GPLB_CJGK_MRGK_C"
        SEC_CODE = "510300"
        TX_DATE = "2026-08-25"
    }
    $headers = @{
        Referer = "https://www.sse.com.cn/assortment/fund/etf/price/"
        "User-Agent" = "Mozilla/5.0 Codex-Research-Source-Probe-V1.6"
    }
    $response = Invoke-RestMethod -Method Get -Uri "https://query.sse.com.cn/commonQuery.do" -Body $parameters -Headers $headers -TimeoutSec 30
    $rows = @($response.result)
    $responseCount = $rows.Count
    if ($rows.Count -ne 1 -or $rows[0].SEC_CODE -ne "510300" -or $rows[0].TX_DATE -ne "20260825") {
        throw "上交所端点返回行数或主键不符合资格探针约束"
    }
    $probeStatus = "PASS_S4U_WORKSPACE_AND_HTTPS_ENDPOINT"
}
catch {
    $failure = "{0}: {1}" -f $_.Exception.GetType().Name, $_.Exception.Message
}
finally {
    [Net.ServicePointManager]::SecurityProtocol = $securityProtocol
    $payload = [ordered]@{
        schema_version = "1.0.0"
        probe_status = $probeStatus
        started_at = $startedAt.ToString("o")
        ended_at = (Get-Date).ToString("o")
        windows_identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        session_name = $env:SESSIONNAME
        user_interactive = [Environment]::UserInteractive
        project_root = $projectRoot
        project_config_sha256 = (Get-FileHash -LiteralPath (Join-Path $projectRoot "config\priority_forward_supervisor_v1_2.yaml") -Algorithm SHA256).Hash.ToLowerInvariant()
        endpoint = "https://query.sse.com.cn/commonQuery.do"
        qualification_date = "2026-08-25"
        response_count = $responseCount
        failure = $failure
        changes_research_state = $false
    }
    $json = $payload | ConvertTo-Json -Depth 5
    $temporary = "$target.$PID.tmp"
    [System.IO.File]::WriteAllText($temporary, $json, (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $target -Force
}

if ($probeStatus -ne "PASS_S4U_WORKSPACE_AND_HTTPS_ENDPOINT") {
    exit 1
}
exit 0
