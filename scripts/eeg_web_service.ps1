param([switch]$Stop)
$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir
$outputDir = Join-Path $root 'results\eeg_web_output'
$pidFile = Join-Path $outputDir 'web_server.pid'
$serverScript = Join-Path $root 'backend\eeg_web_server.py'
$url = 'http://127.0.0.1:8765/'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

function Test-LocalEegService {
    try {
        $request = [System.Net.WebRequest]::Create('http://127.0.0.1:8765/health')
        $request.Proxy = $null
        $request.Timeout = 1200
        $response = $request.GetResponse()
        try { return ($response.StatusCode -eq [System.Net.HttpStatusCode]::OK) }
        finally { $response.Close() }
    } catch { return $false }
}

if ($Stop) {
    if (Test-Path -LiteralPath $pidFile) {
        $savedPid = 0
        if ([int]::TryParse((Get-Content -LiteralPath $pidFile -Raw).Trim(), [ref]$savedPid)) {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$savedPid" -ErrorAction SilentlyContinue
            if ($proc -and $proc.CommandLine -and $proc.CommandLine.Contains($serverScript)) {
                Stop-Process -Id $savedPid -Force
                Write-Host '本机脑电网页服务已停止。'
            } else {
                Write-Host '记录的服务进程已不存在。'
            }
        }
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    } else {
        Write-Host '没有找到服务进程记录。'
    }
    exit 0
}

if (-not (Test-LocalEegService)) {
    $pythonCandidates = @(
        (Join-Path $env:USERPROFILE '.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'),
        (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
    )
    $pythonExe = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $pythonExe) {
        $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($pythonCommand) { $pythonExe = $pythonCommand.Source }
    }
    if (-not $pythonExe) {
        Write-Host '未找到 Python。请安装 Python 3.10+，并确保 NumPy、SciPy、Pandas、Matplotlib、OpenPyXL 可用。' -ForegroundColor Red
        exit 1
    }

    $stdoutLog = Join-Path $outputDir 'server.log'
    $stderrLog = Join-Path $outputDir 'server_error.log'
    $proc = Start-Process -FilePath $pythonExe -ArgumentList @('"' + $serverScript + '"') -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if (Test-LocalEegService) { $ready = $true; break }
        if ($proc.HasExited) { break }
    }
    if (-not $ready) {
        if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
        Write-Host '本机网页服务未能在 30 秒内启动。' -ForegroundColor Red
        if (Test-Path -LiteralPath $stderrLog) {
            Write-Host '最近的错误日志：'
            Get-Content -LiteralPath $stderrLog -Tail 20
        }
        Write-Host "日志文件：$stderrLog"
        exit 1
    }
    Set-Content -LiteralPath $pidFile -Value $proc.Id -Encoding ASCII
}

if (-not (Test-LocalEegService)) {
    Write-Host '服务刚启动后无法响应，请重新启动并查看错误日志。' -ForegroundColor Red
    exit 1
}
Write-Host "脑电特征网页服务正常：$url"
Start-Process $url
exit 0
