#!/usr/bin/env bash
# Astra — Linux/macOS installer
# Usage: bash installer/install.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.venv"
PYTHON="${PYTHON:-python3}"

echo "============================================================"
echo "  Astra Installer"
echo "  Target: $REPO_ROOT"
echo "============================================================"

# ── Python version check ──────────────────────────────────────
PYVER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
PYMAJ=$(echo "$PYVER" | cut -d. -f1)
PYMIN=$(echo "$PYVER" | cut -d. -f2)
if [ "$PYMAJ" -lt 3 ] || { [ "$PYMAJ" -eq 3 ] && [ "$PYMIN" -lt 10 ]; }; then
  echo "ERROR: Python 3.10+ required (found $PYVER)."
  echo "       Install Python 3.11: https://www.python.org/downloads/"
  exit 1
fi
echo "Python $PYVER ... OK"

# ── Create virtual environment ────────────────────────────────
if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment at $VENV ..."
  $PYTHON -m venv "$VENV"
fi

PIP="$VENV/bin/pip"
PYTHON_VENV="$VENV/bin/python"

# ── Install dependencies ──────────────────────────────────────
echo "Installing core dependencies ..."
"$PIP" install --upgrade pip --quiet
"$PIP" install -r "$REPO_ROOT/requirements.txt" --quiet

echo "Installing Astra package ..."
"$PIP" install -e "$REPO_ROOT" --quiet

# ── Optional: uvicorn for API gateway / UI ────────────────────
"$PIP" install uvicorn --quiet 2>/dev/null || true

echo ""
echo "============================================================"
echo "  Installation complete!"
echo ""
echo "  Quick start:"
echo "    # Offline mode (single machine, all layers local)"
echo "    $VENV/bin/python scripts/run_node.py --mode offline --api-port 8080"
echo "    # Then open: http://localhost:8080"
echo ""
echo "  Run environment check:"
echo "    $VENV/bin/python scripts/check_env.py"
echo "============================================================"
