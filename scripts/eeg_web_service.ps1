param([switch]$Stop, [switch]$Watch, [switch]$InstallAutostart,
      [switch]$NoBrowser, [switch]$Status)
# 网站和巡检分开运行。进程意外退出后恢复，正在执行的分析不会被强制中断。
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $root 'results\eeg_web_output'
$serverScript = Join-Path $root 'backend\eeg_web_server.py'
$serviceScript = $MyInvocation.MyCommand.Path
$pidFile = Join-Path $outputDir 'web_server.pid'
$watchPidFile = Join-Path $outputDir 'watchdog.pid'
$stopFile = Join-Path $outputDir 'service.stopped'
$taskName = 'LocalEEGWeb8765'
$url = 'http://127.0.0.1:8765/'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

function Write-ServiceLog($message) {
    $log = Join-Path $outputDir 'supervisor.log'
    if ((Test-Path -LiteralPath $log) -and (Get-Item -LiteralPath $log).Length -gt 5MB) {
        Move-Item -LiteralPath $log -Destination ($log + '.previous') -Force
    }
    Add-Content -LiteralPath $log -Value ((Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $message) -Encoding UTF8
}
function Test-LocalEegService {
    try {
        $request = [Net.WebRequest]::Create($url + 'health')
        $request.Proxy = $null
        $request.Timeout = 3000
        $response = $request.GetResponse()
        try {
            $reader = New-Object IO.StreamReader($response.GetResponseStream())
            $body = $reader.ReadToEnd() | ConvertFrom-Json
            return ($body.ok -eq $true -and $body.service -eq 'local-eeg')
        } finally { $response.Close() }
    } catch { return $false }
}
function Find-Listener {
    # .NET无需查询进程所有者权限，也能判断端口被占用。
    [Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
        Where-Object Port -eq 8765 | Select-Object -First 1
}
function Find-OwnedWebProcesses {
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine.IndexOf($serverScript, [StringComparison]::OrdinalIgnoreCase) -ge 0 }
}
function Stop-OwnedProcess($number, $expected) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$number" -ErrorAction SilentlyContinue
    if ($process -and $process.CommandLine -and $process.CommandLine.IndexOf($expected, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
        Stop-Process -Id $number -Force -ErrorAction SilentlyContinue
    }
}
function Find-Python {
    $candidates = @(
        (Join-Path $root '.venv-mne-review\Scripts\python.exe'),
        (Join-Path $root 'github\bci-egg-online-analysis\.venv-mne-review\Scripts\python.exe'),
        (Join-Path $root '.venv\Scripts\python.exe'),
        (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
    )
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }
    $oldPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = Join-Path $root 'backend'
        foreach ($candidate in $candidates) {
            if (-not (Test-Path -LiteralPath $candidate)) { continue }
            & $candidate -c 'import numpy,scipy,pandas,matplotlib,openpyxl;from eeg_mne_runtime import load_mne;load_mne()' *> (Join-Path $outputDir 'dependency_probe.log')
            if ($LASTEXITCODE -eq 0) { return $candidate }
        }
    } finally { $env:PYTHONPATH = $oldPath }
    throw '找不到依赖完整的Python环境，请检查dependency_probe.log。'
}
function Start-WebProcess($python) {
    if (Test-LocalEegService) { return }
    # 端口还在监听时不杀进程，避免慢分析被误判为崩溃，也不结束其他软件。
    if (Find-Listener) { Write-ServiceLog 'Port is listening but health is unavailable; leaving process untouched.'; return }
    foreach ($name in @('server.log', 'server_error.log')) {
        $path = Join-Path $outputDir $name
        if (Test-Path -LiteralPath $path) { Move-Item -LiteralPath $path -Destination ($path + '.previous') -Force }
    }
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUNBUFFERED = '1'
    $process = Start-Process -FilePath $python -ArgumentList @('-u', ('"{0}"' -f $serverScript)) -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputDir 'server.log') -RedirectStandardError (Join-Path $outputDir 'server_error.log')
    Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ASCII
    Write-ServiceLog ('Started Python PID ' + $process.Id)
}
if ($Status) {
    Write-Output ('Healthy: ' + (Test-LocalEegService))
    Write-Output ('Manually stopped: ' + (Test-Path -LiteralPath $stopFile))
    Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue | Select-Object TaskName,State
    exit 0
}
if ($Stop) {
    Set-Content -LiteralPath $stopFile -Value 'Stopped by user' -Encoding ASCII
    if (Test-Path -LiteralPath $watchPidFile) { Stop-OwnedProcess ([int](Get-Content -LiteralPath $watchPidFile -Raw)) $serviceScript }
    # Windows虚拟环境的Python可能有父/子两个进程，按完整脚本路径一并核对。
    foreach ($owned in @(Find-OwnedWebProcesses)) { Stop-OwnedProcess $owned.ProcessId $serverScript }
    if (Test-Path -LiteralPath $pidFile) { Stop-OwnedProcess ([int](Get-Content -LiteralPath $pidFile -Raw)) $serverScript }
    Write-ServiceLog 'Manually stopped; start shortcut resumes supervision.'
    Write-Host '网页和巡检已停止；双击启动脚本可以恢复。'
    exit 0
}
if ($Watch) {
    $mutex = New-Object Threading.Mutex($false, 'Local\LocalEEGWeb8765Watchdog')
    $locked = $false
    try {
        try { $locked = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $locked = $true }
        if (-not $locked -or (Test-Path -LiteralPath $stopFile)) { exit 0 }
        Set-Content -LiteralPath $watchPidFile -Value $PID -Encoding ASCII
        $python = Find-Python
        Write-ServiceLog ('Watchdog running PID ' + $PID)
        while (-not (Test-Path -LiteralPath $stopFile)) {
            try { Start-WebProcess $python } catch { Write-ServiceLog ('Recovery error: ' + $_.Exception.Message) }
            Start-Sleep -Seconds 15
        }
    } catch { Write-ServiceLog ('Watchdog failed: ' + $_.Exception.Message); exit 1 }
    finally { if ($locked) { $mutex.ReleaseMutex() }; $mutex.Dispose() }
    exit 0
}
if (Test-Path -LiteralPath $stopFile) { Remove-Item -LiteralPath $stopFile }
if ($InstallAutostart) {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $arguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $serviceScript + '" -Watch'
    try {
        $action = New-ScheduledTaskAction -Execute $powershell -Argument $arguments -WorkingDirectory $root
        $logon = New-ScheduledTaskTrigger -AtLogOn -User $identity
        $periodic = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1)) -RepetitionInterval (New-TimeSpan -Minutes 1)
        $principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger @($logon,$periodic) -Principal $principal -Settings $settings -Description 'Local EEG website logon start and crash recovery; loopback only.' -Force | Out-Null
        Write-ServiceLog 'Installed current-user scheduled task.'
        Write-Host '已配置当前用户登录自启和后台恢复。'
    } catch {
        Write-ServiceLog ('Task unavailable; using Startup shortcut: ' + $_.Exception.Message)
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Startup')) 'Local EEG Web.lnk'))
        $shortcut.TargetPath = $powershell
        $shortcut.Arguments = $arguments
        $shortcut.WorkingDirectory = $root
        $shortcut.WindowStyle = 7
        $shortcut.Save()
        Write-Host '已配置当前用户启动文件夹自启。'
    }
}
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task -and $task.State -ne 'Disabled') { Start-ScheduledTask -TaskName $taskName }
else {
    $powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    Start-Process -FilePath $powershell -ArgumentList ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $serviceScript + '" -Watch') -WorkingDirectory $root -WindowStyle Hidden
}
$ready = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    if (Test-LocalEegService) { $ready = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $ready) { Write-Host ('启动未就绪，请查看 ' + (Join-Path $outputDir 'supervisor.log')) -ForegroundColor Red; exit 1 }
Write-Host ('脑电网页正常：' + $url)
if (-not $NoBrowser) { Start-Process $url }
