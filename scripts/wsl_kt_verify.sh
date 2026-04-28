#!/bin/bash
# 验证 kt-kernel 导入 + 源码目录结构

echo "=== kt_kernel import test ==="
python3 -c 'import kt_kernel; print(dir(kt_kernel))' 2>&1

echo ""
echo "=== kt_kernel source dir ==="
ls ~/ktransformers/kt-kernel/kt_kernel/ 2>&1

echo ""
echo "=== Check sglang installed ==="
pip list 2>/dev/null | grep -i sglang || echo "sglang NOT installed yet"

echo ""
echo "=== Check kt-kernel version ==="
pip show kt-kernel 2>&1 | head -5

echo ""
echo "=== Check ktransformers top-level ==="
pip show ktransformers 2>&1 | head -5 || echo "ktransformers top-level NOT installed"

echo ""
echo "=== Check MODEL_CONFIG in kt_kernel ==="
if [ -f ~/ktransformers/kt-kernel/kt_kernel/model_config.py ]; then
    grep -n "class\|ModelConfig\|v2\|V2" ~/ktransformers/kt-kernel/kt_kernel/model_config.py 2>/dev/null | head -20
else
    echo "model_config.py not found, searching..."
    find ~/ktransformers/kt-kernel -name "*.py" -exec grep -l "ModelConfig\|model_config\|DeepseekV" {} \; 2>/dev/null | head -10
fi

echo ""
echo "=== DONE ==="