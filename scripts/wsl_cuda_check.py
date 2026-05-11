#!/usr/bin/env python3
"""Check CUDA availability in WSL."""
import subprocess

print("=== nvcc ===")
try:
    r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
    if r.returncode == 0:
        print(r.stdout)
    else:
        print(f"nvcc error: {r.stderr}")
        # Search common paths
        import os
        for candidate in [
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-12.6/bin/nvcc",
            "/usr/local/cuda-12.5/bin/nvcc",
            "/usr/local/cuda-12.4/bin/nvcc",
            "/usr/local/cuda-12.3/bin/nvcc",
            "/usr/local/cuda-12.2/bin/nvcc",
            "/usr/local/cuda-12.1/bin/nvcc",
            "/usr/local/cuda-12.0/bin/nvcc",
            "/usr/local/cuda-11.8/bin/nvcc",
            "/usr/lib/cuda/bin/nvcc",
            "/opt/cuda/bin/nvcc",
        ]:
            if os.path.isfile(candidate):
                print(f"Found: {candidate}")
except FileNotFoundError:
    print("nvcc not found on PATH")

print()
print("=== nvidia-smi ===")
try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=10)
    if r.returncode == 0:
        print(f"Compute capability: {r.stdout.strip()}")
    else:
        print(f"nvidia-smi error: {r.stderr}")
except FileNotFoundError:
    print("nvidia-smi not found")

print()
print("=== PyTorch CUDA ===")
try:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute capability: {cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
except ImportError:
    print("torch not installed")
