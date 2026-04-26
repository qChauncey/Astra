# Astra — Windows installer (PowerShell)
# Usage: Right-click > "Run with PowerShell"  OR  from a PS terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\installer\install.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Venv     = Join-Path $RepoRoot ".venv"
$Python   = "python"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Astra Installer (Windows / PowerShell)" -ForegroundColor Cyan
Write-Host "  Target: $RepoRoot" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ── Python version check ──────────────────────────────────────────────────
try {
    $pyver = & $Python --version 2>&1
    Write-Host "Found: $pyver"
} catch {
    Write-Host "ERROR: Python not found. Install Python 3.11 from:" -ForegroundColor Red
    Write-Host "       https://www.python.org/downloads/" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# ── Create virtual environment ────────────────────────────────────────────
if (-not (Test-Path $Venv)) {
    Write-Host "Creating virtual environment ..."
    & $Python -m venv $Venv
}

$Pip       = Join-Path $Venv "Scripts\pip.exe"
$PythonVenv = Join-Path $Venv "Scripts\python.exe"

# ── Install dependencies ──────────────────────────────────────────────────
Write-Host "Installing core dependencies ..."
& $Pip install --upgrade pip --quiet
& $Pip install -r (Join-Path $RepoRoot "requirements.txt") --quiet

Write-Host "Installing Astra package ..."
& $Pip install -e $RepoRoot --quiet

# Optional: uvicorn for API gateway / UI
& $Pip install uvicorn --quiet 2>$null

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Quick start:"
Write-Host "    Double-click installer\start.bat"
Write-Host "    OR run in PowerShell:"
Write-Host "      .venv\Scripts\python scripts\run_node.py --mode offline --api-port 8080"
Write-Host "    Then open: http://localhost:8080"
Write-Host ""
Write-Host "  Environment check:"
Write-Host "    .venv\Scripts\python scripts\check_env.py"
Write-Host "============================================================" -ForegroundColor Green

Read-Host "Press Enter to exit"
