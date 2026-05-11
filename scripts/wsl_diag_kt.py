#!/usr/bin/env python3
"""Diagnose ktransformers / kt_kernel installation and CUDA capability."""
import sys
import os
import subprocess
import site
import importlib


def main():
    print("=== Python:", sys.version)
    print("=== Python exe:", sys.executable)
    print()

    # 0. Check site-packages
    print("=== site-packages:")
    for sp in site.getsitepackages():
        print(" ", sp)
    print()

    # 0.1 Check sys.path
    print("=== sys.path (first 10):")
    for p in sys.path[:10]:
        print(" ", p)
    print()

    # 0.2 pip show
    for pkg in ("kt-kernel", "ktransformers"):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True, text=True,
        )
        print(f"pip show {pkg}:")
        print(result.stdout.strip() or "NOT FOUND")
        print()

    # 0.3 /tmp/ktransformers
    kt_dir = "/tmp/ktransformers"
    if os.path.isdir(kt_dir):
        print("=== /tmp/ktransformers contents:")
        for item in sorted(os.listdir(kt_dir)):
            print(" ", item)
    else:
        print("=== /tmp/ktransformers DOES NOT EXIST")
    print()

    # 1. kt_kernel
    print("=== kt_kernel import check ===")
    spec = importlib.util.find_spec("kt_kernel")
    print("find_spec:", spec)
    if spec and spec.origin:
        print("origin:", spec.origin)

    try:
        import kt_kernel  # noqa: F811
        print("kt_kernel imported OK")
        print("kt_kernel.__file__:", kt_kernel.__file__)
        attrs = [a for a in dir(kt_kernel) if not a.startswith("_") or a.startswith("__")]
        print("kt_kernel public attrs:", attrs[:30])
    except ImportError as e:
        print("kt_kernel import FAILED:", e)
        # Check if dir exists in site-packages
        for sp in site.getsitepackages():
            so_dir = os.path.join(sp, "kt_kernel")
            if os.path.isdir(so_dir):
                print("  kt_kernel dir:", so_dir)
                for f in sorted(os.listdir(so_dir))[:20]:
                    print("   ", f)
    print()

    # 2. ktransformers
    print("=== ktransformers import check ===")
    spec2 = importlib.util.find_spec("ktransformers")
    print("find_spec:", spec2)
    if spec2 and spec2.origin:
        print("origin:", spec2.origin)

    try:
        import ktransformers
        print("ktransformers imported OK")
        print("ktransformers.__file__:", ktransformers.__file__)
        ver = getattr(ktransformers, "__version__", "unknown")
        print("ktransformers version:", ver)
        ops = getattr(ktransformers, "ops", None)
        if ops:
            print("ktransformers.ops attrs:", [a for a in dir(ops) if not a.startswith("_") or a.startswith("__")])
        else:
            print("ktransformers.ops: None")
    except ImportError as e:
        print("ktransformers import FAILED:", e)
    print()

    # 3. torch CUDA
    print("=== torch CUDA ===")
    import torch
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    try:
        cap = torch.cuda.get_device_capability(0)
        print("Device 0 capability:", cap)
        sm = cap[0] * 10 + cap[1]
        print(f"SM version: sm_{sm}")
    except Exception as e:
        print("get_device_capability error:", e)
    print("PyTorch CUDA:", torch.version.cuda)
    print()

    # 4. nvidia-smi
    print("=== nvidia-smi ===")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
