@echo off
REM Astra — Windows one-click launcher
REM Starts Astra in offline (single-machine) mode with the web UI.

setlocal

set REPO_ROOT=%~dp0..
set PYTHON_VENV=%REPO_ROOT%\.venv\Scripts\python.exe
set API_PORT=8080

if not exist "%PYTHON_VENV%" (
    echo Astra is not installed yet. Please run installer\install.bat first.
    pause
    exit /b 1
)

echo Starting Astra (offline mode) on http://localhost:%API_PORT% ...
echo Press Ctrl-C to stop.
echo.

start "" "http://localhost:%API_PORT%"

"%PYTHON_VENV%" "%REPO_ROOT%\scripts\run_node.py" ^
    --mode offline ^
    --node-id my-node ^
    --port 50051 ^
    --api-port %API_PORT%

pause
