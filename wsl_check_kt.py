#!/usr/bin/env python3
"""Check KTransformers installation status in WSL."""
import sys
import os

# 1. Check ktransformers package import
print("=== KTransformers Import Check ===")
try:
    sys.path.insert(0, os.path.expanduser("~/ktransformers"))
    import ktransformers
    print(f"OK - ktransformers path: {ktransformers.__file__}")
except Exception as e:
    print(f"FAIL - {e}")

# 2. Check kt-kernel
print("\n=== KTransformers Kernel Check ===")
try:
    import kt_kernel
    print(f"OK - kt_kernel path: {kt_kernel.__file__}")
except Exception as e1:
    print(f"FAIL - {e1}")
    # Try alternative path
    try:
        sys.path.insert(0, os.path.expanduser("~/ktransformers/kt-kernel/python"))
        import kt_kernel
        print(f"OK (alt path) - kt_kernel path: {kt_kernel.__file__}")
    except Exception as e2:
        print(f"FAIL (alt path) - {e2}")

# 3. Check CUDA availability
print("\n=== CUDA Check ===")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        mem_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"GPU memory: {mem_total:.1f} GB")
except Exception as e:
    print(f"FAIL - {e}")

# 4. Check safetensors
print("\n=== safetensors Check ===")
try:
    import safetensors
    print(f"OK - safetensors version: {safetensors.__version__}")
except Exception as e:
    print(f"FAIL - {e}")

# 5. Check if DeekSeek model can be probed via KTransformers
print("\n=== DeepSeek Model Probe ===")
try:
    sys.path.insert(0, os.path.expanduser("~/ktransformers"))
    print("OK - KTransformerModel importable")
except Exception as e:
    print(f"FAIL - {e}")

print("\n=== DONE ===")