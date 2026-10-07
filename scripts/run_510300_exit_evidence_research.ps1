param(
    [ValidateSet('Run', 'Status')]
    [string]$Mode = 'Run'
)

$ErrorActionPreference = 'Stop'
$researchRoot = Split-Path -Parent $PSScriptRoot
$researchPython = Join-Path $researchRoot '.venv\Scripts\python.exe'
$researchStudies = @(
    'selected_mix_support_envelope_daily_v1',
    'selected_mix_exit_usage_factorial_v1',
    'selected_mix_support_earlier_diagnostic_v1'
)

if (-not (Test-Path -LiteralPath $researchPython -PathType Leaf)) {
    throw '项目 Python 环境不存在，请恢复项目原有 .venv。'
}

Push-Location -LiteralPath $researchRoot
try {
    foreach ($researchStudy in $researchStudies) {
        $researchEntry = Join-Path $researchRoot ('research\' + $researchStudy + '.py')
        $researchOutput = Join-Path $researchRoot ('reports\research\510300_' + $researchStudy)
        $researchResultPath = Join-Path $researchOutput 'result.json'
        if ($Mode -eq 'Run' -and -not (Test-Path -LiteralPath $researchResultPath)) {
            if (-not (Test-Path -LiteralPath (Join-Path $researchOutput 'freeze.json'))) {
                & $researchPython -X utf8 $researchEntry freeze
                if ($LASTEXITCODE -ne 0) { throw ('固定协议失败：' + $researchStudy) }
            }
            if (Test-Path -LiteralPath (Join-Path $researchOutput 'RUN_STARTED.json')) {
                throw ('已有启动记录但无完成结果，需定位中断，禁止覆盖重跑：' + $researchStudy)
            }
            & $researchPython -X utf8 $researchEntry run
            if ($LASTEXITCODE -ne 0) { throw ('研究运行失败：' + $researchStudy) }
        }
        if (Test-Path -LiteralPath $researchResultPath) {
            $researchResult = Get-Content -LiteralPath $researchResultPath -Raw -Encoding utf8 | ConvertFrom-Json
            $researchMetric = $researchResult.primary
            Write-Output ('{0}：压力夏普 {1:F3}，年化 {2:P2}，最大回撤 {3:P2}，状态 {4}' -f
                $researchStudy, $researchMetric.sharpe, $researchMetric.annual_return,
                [Math]::Abs($researchMetric.max_drawdown), $researchResult.status)
        }
        else {
            Write-Output ($researchStudy + '：尚未完成。')
        }
    }
    if ($Mode -eq 'Run') {
        & $researchPython -X utf8 (Join-Path $researchRoot 'research\selected_mix_exit_evidence_delivery_v1.py')
        if ($LASTEXITCODE -ne 0) { throw '学习退出证据整理失败。' }
    }
    Write-Output '本入口复用已完成本地研究，不覆盖中断，不发订单。目标以实际验收为准。'
}
finally {
    Pop-Location
}
