#!/bin/bash
# Quick status check for WSL DeepSeek environment
echo "=== Python ==="
python3 --version
echo ""
echo "=== Models ==="
ls -lh ~/models/ 2>/dev/null || echo "No ~/models/ directory"
echo ""
echo "=== PyTorch ==="
python3 -c '
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} VRAM={p.total_memory/1e9:.1f}GB")
'
echo ""
echo "=== ktransformers ==="
python3 -c '
from kt_kernel import kt_kernel_ext
import kt_kernel
print("kt-kernel:", kt_kernel.__version__, "| C++ extension: OK")
'
echo ""
echo "=== flashinfer ==="
python3 -c 'import flashinfer; print("flashinfer:", flashinfer.__version__)' 2>&1
echo ""
echo "=== transformers ==="
python3 -c 'import transformers; print("transformers:", transformers.__version__)' 2>&1
echo ""
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found"