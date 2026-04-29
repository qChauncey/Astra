#!/usr/bin/env bash
# Astra — KTransformers CUDA kernel build script
# Usage: bash scripts/build_ktransformers.sh [--clone-dir /path/to/ktransformers]
#
# Detects GPU compute capability, patches CMake arch list if needed,
# and builds kt-kernel with CUDA support.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_NAME="$(basename "$0")"

# ── Colors ──────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $*"; }

# ── CLI args ────────────────────────────────────────────────────
CLONE_DIR="${KT_CLONE_DIR:-/tmp/ktransformers}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clone-dir) CLONE_DIR="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $SCRIPT_NAME [--clone-dir /path/to/ktransformers]"
            echo ""
            echo "Builds kt-kernel (KTransformers CUDA kernels) for your GPU."
            echo ""
            echo "Environment variables:"
            echo "  KT_CLONE_DIR         clone destination (default /tmp/ktransformers)"
            echo "  CPUINFER_PARALLEL     parallel build jobs (auto-detect)"
            echo "  CPUINFER_BUILD_TYPE   Debug | Release (default Release)"
            echo "  CMAKE_CUDA_ARCHITECTURES   override auto-detected GPU arch list"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Prerequisite checks ─────────────────────────────────────────
log_step "Checking prerequisites ..."

# 1. nvcc (CUDA compiler)
if ! command -v nvcc &>/dev/null; then
    log_error "nvcc not found — CUDA Toolkit must be installed."
    echo ""
    echo "  Install CUDA Toolkit 12.x:"
    echo "    Ubuntu 22.04:"
    echo "      sudo apt-get install -y cuda-toolkit-12-6"
    echo "    Other distros: https://developer.nvidia.com/cuda-downloads"
    echo ""
    echo "  Or, if PyTorch ships its own CUDA compiler, set:"
    echo "    export PATH=\$PATH:\$(python -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent.parent / \"nvidia\" / \"cuda_runtime\" / \"bin\")')"
    echo ""
    exit 1
fi
NVCC_VER=$(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')
log_info "nvcc version: $NVCC_VER"

# 2. cmake
if ! command -v cmake &>/dev/null; then
    log_error "cmake not found — install with: sudo apt-get install -y cmake"
    exit 1
fi
CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
log_info "cmake version: $CMAKE_VER"

# 3. git
if ! command -v git &>/dev/null; then
    log_error "git not found — install with: sudo apt-get install -y git"
    exit 1
fi

# 4. pybind11 (vendored in ktransformers, but needs a system install for CMake)
#    kt-kernel vendors pybind11; this check is informational only.
if python -c "import pybind11" 2>/dev/null; then
    log_info "pybind11: found (system)"
else
    log_info "pybind11: using vendored copy from ktransformers"
fi

echo ""

# ── Detect GPU compute capability ───────────────────────────────
log_step "Detecting GPU compute capability ..."

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    log_info "GPU: $GPU_NAME"
else
    log_warn "nvidia-smi not found — cannot auto-detect GPU"
    GPU_NAME="unknown"
fi

# Try PyTorch for precise compute capability
DETECTED_ARCH=""
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CC=$(python -c "
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f'{major}{minor}')
" 2>/dev/null)
    if [[ -n "$CC" ]]; then
        DETECTED_ARCH="$CC"
        log_info "Detected compute capability: sm_$CC (from PyTorch)"
    fi
fi

# Generate arch list
if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
    ARCH_LIST="$CMAKE_CUDA_ARCHITECTURES"
    log_info "Using user-specified CMAKE_CUDA_ARCHITECTURES=$ARCH_LIST"
elif [[ -n "$DETECTED_ARCH" ]]; then
    # Generate: your GPU's arch + common older ones for max compatibility
    # Blackwell = 120, Ada = 89, Ampere = 80/86, Hopper = 90
    KNOWN_ARCHS=("$DETECTED_ARCH" "90" "89" "86" "80" "75")
    # Deduplicate while preserving order
    declare -A seen
    ARCH_LIST=""
    for a in "${KNOWN_ARCHS[@]}"; do
        if [[ -z "${seen[$a]:-}" ]]; then
            seen[$a]=1
            if [[ -n "$ARCH_LIST" ]]; then
                ARCH_LIST="$ARCH_LIST;$a"
            else
                ARCH_LIST="$a"
            fi
        fi
    done
    log_info "Generated arch list: $ARCH_LIST"
else
    log_warn "Cannot detect GPU architecture — using default arch list"
    ARCH_LIST="80;86;89;90"
fi

echo ""

# ── Clone ktransformers (if not already present) ─────────────────
log_step "Cloning ktransformers ..."

if [[ -d "$CLONE_DIR/.git" ]]; then
    log_info "Existing clone found at $CLONE_DIR — pulling latest ..."
    git -C "$CLONE_DIR" pull --ff-only 2>/dev/null || log_warn "Could not pull (may be on a detached HEAD or have local changes)"
else
    log_info "Cloning https://github.com/kvcache-ai/ktransformers into $CLONE_DIR ..."
    git clone --depth 1 https://github.com/kvcache-ai/ktransformers.git "$CLONE_DIR"
fi

KT_KERNEL_DIR="$CLONE_DIR/kt-kernel"
if [[ ! -d "$KT_KERNEL_DIR" ]]; then
    log_error "kt-kernel directory not found at $KT_KERNEL_DIR — clone may be incomplete"
    log_error "Try: rm -rf $CLONE_DIR && $0"
    exit 1
fi

log_info "KTransformers source: $CLONE_DIR"
echo ""

# ── Patch CMake arch list if needed ─────────────────────────────
log_step "Checking CUDA arch support ..."

CMAKE_FILE="$KT_KERNEL_DIR/CMakeLists.txt"
if [[ ! -f "$CMAKE_FILE" ]]; then
    log_error "CMakeLists.txt not found at $CMAKE_FILE"
    exit 1
fi

# Check if the default arch list includes our GPU
DEFAULT_ARCH_LINE=$(grep -n 'CMAKE_CUDA_ARCHITECTURES' "$CMAKE_FILE" | head -1)
log_info "Found arch config: $DEFAULT_ARCH_LINE"

# Expand arch list: check each arch individually
NEEDS_PATCH=false
if [[ -n "$DETECTED_ARCH" ]]; then
    CURRENT_DEFAULTS=$(grep 'set(CMAKE_CUDA_ARCHITECTURES' "$CMAKE_FILE" | head -1 | grep -oP '"[^"]*"')
    if [[ "$CURRENT_DEFAULTS" != *"$DETECTED_ARCH"* ]]; then
        NEEDS_PATCH=true
        log_warn "CMake defaults ($CURRENT_DEFAULTS) do not include sm_$DETECTED_ARCH"
    else
        log_info "CMake defaults already include sm_$DETECTED_ARCH — no patch needed"
    fi
else
    log_info "Skipping arch patch (using CMAKE_CUDA_ARCHITECTURES=$ARCH_LIST)"
fi

if [[ "$NEEDS_PATCH" == "true" ]]; then
    log_step "Patching CMakeLists.txt to add sm_$DETECTED_ARCH ..."
    # Create a backup
    cp "$CMAKE_FILE" "$CMAKE_FILE.bak"
    # Replace the default arch list line
    sed -i "s/set(CMAKE_CUDA_ARCHITECTURES \".*\")/set(CMAKE_CUDA_ARCHITECTURES \"$ARCH_LIST\")/" "$CMAKE_FILE"
    log_info "Patched. Backup saved to $CMAKE_FILE.bak"
fi

echo ""

# ── Build kt-kernel with CUDA ────────────────────────────────────
log_step "Building kt-kernel with CUDA support ..."

cd "$KT_KERNEL_DIR"

export CPUINFER_USE_CUDA=1
export CPUINFER_BUILD_TYPE="${CPUINFER_BUILD_TYPE:-Release}"
export CPUINFER_PARALLEL="${CPUINFER_PARALLEL:-}"
export CMAKE_CUDA_ARCHITECTURES="$ARCH_LIST"

# The kt-kernel setup.py reads CPUINFER_USE_CUDA and forwards it to CMake
# but nvcc needs to be on PATH
log_info "Build env:"
log_info "  CPUINFER_USE_CUDA=$CPUINFER_USE_CUDA"
log_info "  CPUINFER_BUILD_TYPE=$CPUINFER_BUILD_TYPE"
log_info "  CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES"
log_info "  CPUINFER_PARALLEL=${CPUINFER_PARALLEL:-auto}"
echo ""

# Run pip install
PIP_CMD=("pip" "install" "-e" "." "--no-build-isolation" "--verbose")
log_info "Running: ${PIP_CMD[*]}"
echo ""

if "${PIP_CMD[@]}"; then
    log_info "Build succeeded!"
else
    log_error "Build failed. See output above for errors."
    echo ""
    echo "  Troubleshooting:"
    echo "  1. Verify CUDA toolkit version matches nvcc: nvcc --version"
    echo "  2. Verify GPU driver: nvidia-smi"
    echo "  3. Check CMake can find CUDA: cmake --find-package -DNAME=CUDA"
    echo "  4. Try building with debug output: CPUINFER_BUILD_TYPE=Debug $0"
    echo "  5. Check disk space for build artifacts"
    exit 1
fi

echo ""

# ── Verify installation ─────────────────────────────────────────
log_step "Verifying ktransformers CUDA installation ..."

if python -c "
from astra.inference.ktransformers_adapter import detect_ktransformers
result = detect_ktransformers()
print(f'Available: {result[\"available\"]}')
print(f'Backend: {result.get(\"backend\", \"none\")}')
assert result['available'], 'KTransformers not available'
assert result.get('backend') in ('ktransformers_cpp', 'kt_kernel'), f'Unexpected backend: {result.get(\"backend\")}'
" 2>&1; then
    BACKEND=$(python -c "
from astra.inference.ktransformers_adapter import detect_ktransformers
r = detect_ktransformers()
print(r.get('backend', 'unknown'))
")
    log_info "Verification PASSED — backend: $BACKEND"
else
    log_warn "Verification returned warnings — KTransformers may still be functional"
    log_warn "Run 'python scripts/smoke_kt_adapter.py' for a detailed check"
fi

echo ""
echo "============================================================"
echo -e "  ${GREEN}KTransformers CUDA build complete!${NC}"
echo ""
echo "  Verify with:"
echo "    python scripts/check_env.py"
echo "    python scripts/smoke_kt_adapter.py"
echo ""
echo "  Start inference:"
echo "    python scripts/run_node.py --mode offline --gpu --api-port 8080"
echo "============================================================"