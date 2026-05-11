#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda-12.6/bin:$PATH"
echo '=== nvcc check ==='
nvcc --version
echo '=== GPU check ==='
nvidia-smi --query-gpu=name --format=csv,noheader
echo '=== Starting build ==='
cd ~/ktransformers/kt-kernel
export CPUINFER_USE_CUDA=1
export CPUINFER_BUILD_TYPE=Release
export CPUINFER_CUDA_ARCHS="80;86;89;90"
export CPUINFER_CPU_INSTRUCT=FANCY
export CPUINFER_PARALLEL=8
pip install -e . --no-build-isolation --verbose 2>&1
echo '=== Build finished ==='