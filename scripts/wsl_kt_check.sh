#!/bin/bash
# KTransformers + DeepSeek-V2 一键检查脚本
set -e

echo "=== 1. Python & pip ==="
python3 --version
pip --version 2>/dev/null || pip3 --version

echo ""
echo "=== 2. pip list (kt/sglang) ==="
pip list 2>/dev/null | grep -iE "kt-kernel|ktransformer|sglang" || echo "(none installed)"

echo ""
echo "=== 3. nvcc ==="
which nvcc 2>/dev/null && nvcc --version || echo "nvcc not found"

echo ""
echo "=== 4. nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"

echo ""
echo "=== 5. KTransformers repo ==="
echo "KT repo: ~/ktransformers"
ls ~/ktransformers/kt-kernel/kt_kernel/ 2>/dev/null || echo "kt_kernel dir not found"

echo ""
echo "=== 6. Check kt-kernel model registry ==="
if [ -f ~/ktransformers/kt-kernel/kt_kernel/model_config.py ]; then
    echo "--- model_config.py ---"
    grep -n "deepseek\|v2\|v3\|chat" ~/ktransformers/kt-kernel/kt_kernel/model_config.py 2>/dev/null | head -20 || echo "no deepseek entries"
fi

echo ""
echo "=== 7. Check install.sh ==="
if [ -f ~/ktransformers/install.sh ]; then
    echo "--- install.sh preview ---"
    head -40 ~/ktransformers/install.sh
fi

echo ""
echo "=== 8. Disk usage of model ==="
du -sh /home/chauncey/deepseek-v4/ 2>/dev/null || echo "Model dir not found at ~/deepseek-v4"

echo ""
echo "=== 9. Model files (first 20) ==="
ls /home/chauncey/deepseek-v4/ 2>/dev/null | head -20 || echo "Cannot list"

echo ""
echo "=== 10. python3-venv package ==="
dpkg -l python3.10-venv 2>/dev/null | tail -3

echo ""
echo "=== DONE ==="