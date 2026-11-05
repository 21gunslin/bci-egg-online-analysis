<#
一键按顺序执行：逐份转换与提取 → 按人汇总 → 随机森林评价 → 生成报告。
数据、标签和结果留在本机；不会提交Git，也不会改写原始BDF。
#>
param(
    [Parameter(Mandatory=$true)][string]$InputRoot,
    [Parameter(Mandatory=$true)][string]$Baseline,
    [string]$OutputDirectory,
    [int]$Workers = 2,
    [string]$PythonExecutable
)
$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
if (-not $PythonExecutable) {
    $PythonExecutable = Join-Path $project '.venv-mne-review\Scripts\python.exe'
    # 同一项目在工作区根目录和Git整理目录各有一份；优先用已有独立环境。
    $workspaceEnvironment = Join-Path $project 'github\bci-egg-online-analysis\.venv-mne-review\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $PythonExecutable) -and (Test-Path -LiteralPath $workspaceEnvironment)) {
        $PythonExecutable = $workspaceEnvironment
    }
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $project ('results\batch_fif_rf_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
if (-not (Test-Path -LiteralPath $InputRoot -PathType Container)) { throw '找不到输入文件夹' }
if (-not (Test-Path -LiteralPath $Baseline -PathType Leaf)) { throw '找不到基线表' }
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw '找不到Python环境，请指定-PythonExecutable' }
$InputRoot = (Resolve-Path -LiteralPath $InputRoot).Path
$Baseline = (Resolve-Path -LiteralPath $Baseline).Path
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
$env:PYTHONIOENCODING = 'utf-8'
$complete = Join-Path $OutputDirectory 'files_complete.json'
if (Test-Path -LiteralPath $complete) { Remove-Item -LiteralPath $complete }

# 独立的复制器在转换后核对指纹，再把fdata.fif放回原data.bdf目录。
# Windows启动外部程序时用引号保留路径内空格；文件路径本身不允许双引号。
$brokerScript = Join-Path $project 'tools\copy_fif_to_source.ps1'
$brokerArguments = @('-NoProfile', '-File', ('"{0}"' -f $brokerScript),
    '-OutputDirectory', ('"{0}"' -f $OutputDirectory), '-InputRoot', ('"{0}"' -f $InputRoot))
$broker = Start-Process -FilePath 'powershell.exe' -ArgumentList $brokerArguments -WindowStyle Hidden -PassThru
try {
    Write-Host '第1步：BDF转FIF，并逐文件提取特征。可查看结果目录中的progress.json。'
    & $PythonExecutable (Join-Path $project 'tools\run_bdf_fif_features.py') --input-root $InputRoot --output $OutputDirectory --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw '第1步失败，请查看结果目录的日志' }
    Write-Host '第2步：按记录汇总，再按产妇配对镇痛前后。'
    & $PythonExecutable (Join-Path $project 'tools\aggregate_subject_features.py') --output $OutputDirectory --baseline $Baseline --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw '第2步失败' }
    & $PythonExecutable (Join-Path $project 'tools\audit_batch_outputs.py') --output $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw '源文件/FIF核验发现问题，停止建立模型' }
    Write-Host '第3步：训练和评价随机森林。'
    & $PythonExecutable (Join-Path $project 'tools\train_random_forest.py') --output $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw '第3步失败' }
    & $PythonExecutable (Join-Path $project 'tools\build_batch_report.py') --output $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw '报告生成失败' }
    & $PythonExecutable (Join-Path $project 'tools\validate_batch_results.py') --output $OutputDirectory --baseline $Baseline
    if ($LASTEXITCODE -ne 0) { throw '最终标签或指标复核失败，请检查日志' }
    Write-Host ('已完成，打开：' + (Join-Path $OutputDirectory '随机森林探索性报告.html'))
} finally {
    # 只结束本脚本自己启动的复制器，不影响网站或其他Python任务。
    if (-not $broker.HasExited) {
        if (-not $broker.WaitForExit(10000)) { $broker | Stop-Process }
    }
}
