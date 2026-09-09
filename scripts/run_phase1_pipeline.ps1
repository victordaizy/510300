param(
    [switch]$RefreshData
)

$ErrorActionPreference = "Stop"
$Utf8Encoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $Utf8Encoding
$OutputEncoding = $Utf8Encoding
$env:PYTHONIOENCODING = "utf-8"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "未找到项目虚拟环境：$Python"
}

Push-Location $ProjectRoot
try {
    if ($RefreshData) {
        Write-Host "刷新沪深300特征预热行情……"
        & $Python "scripts\download_000300_feature_warmup.py"
        if ($LASTEXITCODE -ne 0) { throw "沪深300行情下载失败" }

        Write-Host "刷新沪深300历史估值……"
        & $Python "scripts\download_000300_valuation.py"
        if ($LASTEXITCODE -ne 0) { throw "沪深300估值下载失败" }

        Write-Host "刷新沪深300全收益指数……"
        & $Python "scripts\download_h00300_total_return.py"
        if ($LASTEXITCODE -ne 0) { throw "沪深300全收益指数下载失败" }
    }

    Write-Host "运行全部回归测试……"
    & $Python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "回归测试失败" }

    Write-Host "审计510300日线……"
    & $Python "scripts\quality_check_daily.py"
    if ($LASTEXITCODE -ne 0) { throw "510300日线质量审计失败" }

    Write-Host "审计第一阶段辅助数据……"
    & $Python "scripts\quality_check_phase1_auxiliary.py"
    if ($LASTEXITCODE -ne 0) { throw "辅助数据质量审计失败" }

    Write-Host "构建点时市场状态特征……"
    & $Python "research\build_market_state_features.py"
    if ($LASTEXITCODE -ne 0) { throw "市场状态特征构建失败" }

    Write-Host "运行基础特征研究……"
    & $Python "research\research_base_features.py"
    if ($LASTEXITCODE -ne 0) { throw "基础特征研究失败" }

    Write-Host "运行第一阶段五年回测……"
    & $Python -m backtest.run_phase1_backtest
    if ($LASTEXITCODE -ne 0) { throw "五年回测失败" }

    Write-Host "第一阶段流水线执行完成。"
}
finally {
    Pop-Location
}
