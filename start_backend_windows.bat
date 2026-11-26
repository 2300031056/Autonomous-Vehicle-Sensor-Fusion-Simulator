@echo off
cd /d "%~dp0\backend"
if not exist .venv\Scripts\activate.bat (
  echo Please run install_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
