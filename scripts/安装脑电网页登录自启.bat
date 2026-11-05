@echo off
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0eeg_web_service.ps1" -InstallAutostart
if errorlevel 1 echo 安装或启动失败，请查看 results\eeg_web_output\supervisor.log。
pause
