param(
    [switch]$Watch,
    [int]$MaxSamples = 0,
    [string]$PythonExecutable = "",
    [string]$CollectorScript = "",
    [string]$AnalyzerScript = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    Join-Path $projectRoot ".venv\Scripts\python.exe"
}
else {
    [System.IO.Path]::GetFullPath($PythonExecutable)
}
$collector = if ([string]::IsNullOrWhiteSpace($CollectorScript)) {
    Join-Path $projectRoot "scripts\collect_510300_primary_market_v1_2.py"
}
else {
    [System.IO.Path]::GetFullPath($CollectorScript)
}
$analyzer = if ([string]::IsNullOrWhiteSpace($AnalyzerScript)) {
    Join-Path $projectRoot "scripts\analyze_510300_primary_market_readiness_v1_2.py"
}
else {
    [System.IO.Path]::GetFullPath($AnalyzerScript)
}

foreach ($requiredFile in @($python, $collector, $analyzer)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        [Console]::Error.WriteLine("缺少研究运行文件：$requiredFile")
        exit 1
    }
}

$collectorArguments = @($collector)
if ($Watch) {
    $collectorArguments += "--watch"
}
if ($MaxSamples -gt 0) {
    $collectorArguments += @("--max-samples", $MaxSamples)
}

& $python @collectorArguments
$collectorExitCode = $LASTEXITCODE
if ($collectorExitCode -eq 3) {
    [Console]::Error.WriteLine("PCF/IOPV 免费外部来源失败，退出码：3")
    exit 3
}
if ($collectorExitCode -ne 0) {
    [Console]::Error.WriteLine("PCF/IOPV 采集程序失败，退出码：$collectorExitCode")
    exit 1
}

& $python $analyzer
$analyzerExitCode = $LASTEXITCODE
if ($analyzerExitCode -ne 0) {
    [Console]::Error.WriteLine("PCF/IOPV V1.2 成熟度分析失败，退出码：$analyzerExitCode")
    exit 1
}

Write-Output "PCF/IOPV V1.2 研究采集与成熟度分析完成；未生成信号、仓位映射或订单。"
exit 0
