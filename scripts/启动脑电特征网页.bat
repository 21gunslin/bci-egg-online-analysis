@echo off
chcp 65001 >nul
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0eeg_web_service.ps1"
if errorlevel 1 (
    echo.
    echo 网页服务未能启动。请查看上方提示和 results\eeg_web_output\server_error.log。
    pause
)
endlocal
