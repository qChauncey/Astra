"""Probe kt_kernel and detect available GPU backends."""
import sys

# 1. kt_kernel probe
try:
    import kt_kernel
    print("kt_kernel FOUND")
    attrs = [x for x in dir(kt_kernel) if not x.startswith("_")]
    print(f"kt_kernel exports ({len(attrs)}): {', '.join(sorted(attrs))}")
except ImportError as e:
    print(f"kt_kernel NOT FOUND: {e}")

# 2. flashinfer probe
try:
    import flashinfer
    ver = getattr(flashinfer, "__version__", "unknown")
    print(f"flashinfer FOUND version={ver}")
    fi_attrs = [x for x in dir(flashinfer) if not x.startswith("_")]
    print(f"flashinfer exports ({len(fi_attrs)}): {', '.join(sorted(fi_attrs)[:40])}...")
except ImportError as e:
    print(f"flashinfer NOT FOUND: {e}")

# 3. ktransformers probe
try:
    import ktransformers
    print("ktransformers FOUND")
    kt_attrs = [x for x in dir(ktransformers) if not x.startswith("_")]
    print(f"ktransformers exports ({len(kt_attrs)}): {', '.join(sorted(kt_attrs)[:40])}")
except ImportError as e:
    print(f"ktransformers NOT FOUND: {e}")

# 4. torch CUDA probe
try:
    import torch
    print(f"torch FOUND version={torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA mem: {torch.cuda.get_device_properties(0).total_mem // 1024**3} GB")
except ImportError as e:
    print(f"torch NOT FOUND: {e}")

# 5. Run detect_ktransformers from our adapter
sys.path.insert(0, "/mnt/c/Users/Qchau/Documents/GitHub/Astra")
try:
    from astra.inference.ktransformers_adapter import detect_ktransformers
    result = detect_ktransformers()
    print("\n=== detect_ktransformers() result ===")
    for k, v in sorted(result.items()):
        if callable(v):
            print(f"  {k}: <callable>")
        else:
            print(f"  {k}: {v}")
except Exception as e:
    print(f"detect_ktransformers FAILED: {e}")
