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

# ── KTransformers CUDA kernels (GPU inference) ───────────────────
echo ""
echo "------------------------------------------------------------"
echo "  GPU inference: KTransformers CUDA kernels"
echo "------------------------------------------------------------"
echo ""

HAS_NVCC=false
command -v nvcc &>/dev/null && HAS_NVCC=true

HAS_GPU=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_GPU=true
fi

HAS_TORCH_CUDA=false
if "$PYTHON_VENV" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    HAS_TORCH_CUDA=true
fi

if $HAS_GPU && $HAS_TORCH_CUDA; then
    if $HAS_NVCC; then
        echo "CUDA GPU detected with compiler — building KTransformers kernels ..."
        echo "  (This may take 5–15 minutes, depending on your system)"
        echo ""

        KT_BUILD_SCRIPT="$REPO_ROOT/scripts/build_ktransformers.sh"
        if [[ -f "$KT_BUILD_SCRIPT" ]]; then
            if bash "$KT_BUILD_SCRIPT"; then
                echo ""
                echo "KTransformers CUDA kernels built successfully!"
            else
                echo ""
                echo "WARNING: KTransformers build failed."
                echo "  GPU inference will fall back to pure PyTorch (slower)."
                echo "  Re-run later with: bash scripts/build_ktransformers.sh"
            fi
        else
            echo "WARNING: Build script not found at $KT_BUILD_SCRIPT"
            echo "  Skipping KTransformers build. GPU inference will use PyTorch fallback."
        fi
    else
        echo "GPU detected but nvcc (CUDA compiler) not found."
        echo "  Install CUDA Toolkit for full KTransformers performance:"
        echo "    sudo apt-get install -y cuda-toolkit-12-6"
        echo "  Then run: bash scripts/build_ktransformers.sh"
        echo "  For now, GPU inference will use pure PyTorch (5–20× slower)."
    fi
elif $HAS_GPU && ! $HAS_TORCH_CUDA; then
    echo "GPU detected but PyTorch CUDA not available."
    echo "  Install PyTorch with CUDA: https://pytorch.org/get-started/locally/"
    echo "  Then run: bash scripts/build_ktransformers.sh"
else
    echo "No GPU detected — skipping KTransformers build."
    echo "  CPU-only inference (slower) or API gateway role."
fi

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
