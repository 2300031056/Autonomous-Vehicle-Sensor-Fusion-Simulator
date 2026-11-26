@echo off
setlocal
cd /d "%~dp0"
echo [1/2] Installing backend dependencies...
cd backend
if not exist .venv py -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
cd ..\frontend
echo [2/2] Installing frontend dependencies...
call npm install
cd ..
echo.
echo Installation complete.
echo Start backend: run start_backend_windows.bat
echo Start frontend: run start_frontend_windows.bat
pause
