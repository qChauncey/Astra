@echo off
REM Astra — Windows installer (cmd.exe)
REM Usage: Double-click or run from Command Prompt as:
REM   installer\install.bat

setlocal enabledelayedexpansion

set REPO_ROOT=%~dp0..
set VENV=%REPO_ROOT%\.venv
set PYTHON=python

echo ============================================================
echo   Astra Installer (Windows)
echo   Target: %REPO_ROOT%
echo ============================================================

REM ── Python version check ──────────────────────────────────────
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11 from:
    echo        https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('%PYTHON% --version 2^>^&1') do set PYVER=%%v
echo Python %PYVER% found.

REM ── Create virtual environment ────────────────────────────────
if not exist "%VENV%\" (
    echo Creating virtual environment ...
    %PYTHON% -m venv "%VENV%"
)

set PIP=%VENV%\Scripts\pip.exe
set PYTHON_VENV=%VENV%\Scripts\python.exe

REM ── Install dependencies ──────────────────────────────────────
echo Installing core dependencies ...
"%PIP%" install --upgrade pip --quiet
"%PIP%" install -r "%REPO_ROOT%\requirements.txt" --quiet

echo Installing Astra package ...
"%PIP%" install -e "%REPO_ROOT%" --quiet

REM Optional: uvicorn for API gateway / UI
"%PIP%" install uvicorn --quiet 2>nul

echo.
echo ============================================================
echo   Installation complete!
echo.
echo   Quick start:
echo     Double-click installer\start.bat
echo     OR run:
echo       .venv\Scripts\python scripts\run_node.py --mode offline --api-port 8080
echo     Then open: http://localhost:8080
echo.
echo   Run environment check:
echo     .venv\Scripts\python scripts\check_env.py
echo ============================================================

pause
