@echo off
cd /d "%~dp0\frontend"
if not exist node_modules (
  echo Please run install_windows.bat first.
  pause
  exit /b 1
)
npm run dev
