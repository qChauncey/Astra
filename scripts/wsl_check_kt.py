#!/usr/bin/env python3
"""Diagnose kt_kernel / ktransformers import state."""

print("=== kt_kernel ===")
try:
    import kt_kernel
    print(f"version: {getattr(kt_kernel, '__version__', 'N/A')}")
    pub = [x for x in dir(kt_kernel) if not x.startswith('_')]
    print(f"public attrs: {pub}")
    # Try loading the native extension
    try:
        from kt_kernel import kt_kernel_ext  # noqa: F401
        print("kt_kernel_ext: OK")
    except ImportError as e:
        print(f"kt_kernel_ext: FAIL - {e}")
except ImportError as e:
    print(f"IMPORT FAIL: {e}")

print()
print("=== ktransformers ===")
try:
    import ktransformers
    print(f"version: {getattr(ktransformers, '__version__', 'N/A')}")
    ops = getattr(ktransformers, "ops", None)
    if ops:
        ops_attrs = [x for x in dir(ops) if not x.startswith('_')]
        print(f"ops attrs: {ops_attrs}")
    else:
        print("No 'ops' module found in ktransformers")
except ImportError as e:
    print(f"IMPORT FAIL: {e}")

print()
print("=== torch custom ops ===")
try:
    import torch
    if hasattr(torch.ops, "ktransformers"):
        kt_ops = torch.ops.ktransformers
        print(f"torch.ops.ktransformers attrs: {[x for x in dir(kt_ops) if not x.startswith('_')]}")
    else:
        print("torch.ops.ktransformers: NOT FOUND")
except ImportError as e:
    print(f"torch IMPORT FAIL: {e}")

print()
print("=== kt_kernel CUDA kernels ===")
try:
    import kt_kernel
    for attr in ['mla_forward', 'rms_norm', 'rope', 'rope_embedding', 'mxfp4_routed_moe', 'mxfp8_quantize', 'nsa_sparse_mla']:
        val = getattr(kt_kernel, attr, None)
        print(f"  kt_kernel.{attr}: {'YES' if val else 'None'}")
except ImportError:
    print("kt_kernel not available")
