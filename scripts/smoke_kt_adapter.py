#!/usr/bin/env python
"""Smoke-test the KTransformersAdapter end-to-end on a real GPU (WSL2)."""
import os
import sys
import numpy as np

# Ensure we import from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("ASTRA_USE_KTRANSFORMERS", "1")

from astra.inference.ktransformers_adapter import detect_ktransformers, KTransformersAdapter

info = detect_ktransformers()
print("available:", info["available"], "backend:", info["backend"])

if not info["available"]:
    print("Error:", info.get("error"))
    sys.exit(1)

adapter = KTransformersAdapter(probe=info)

# --- MLA attention ---
q = np.random.randn(2, 4, 256).astype(np.float16)
k = np.random.randn(2, 4, 256).astype(np.float16)
v = np.random.randn(2, 4, 256).astype(np.float16)
out = adapter.multi_latent_attention(q, k, v)
print(f"MLA output shape: {out.shape}, dtype: {out.dtype}, no NaN: {not np.any(np.isnan(out))}")

# --- RMSNorm ---
rms_out = adapter.rms_layer_norm(
    np.random.randn(4, 512).astype(np.float32),
    np.ones(512, dtype=np.float32),
)
print(f"RMSNorm output shape: {rms_out.shape}, dtype: {rms_out.dtype}")

# --- RoPE ---
x_rope = np.random.randn(8, 64).astype(np.float16)
pos = np.arange(8, dtype=np.int64)
rope_out = adapter.rope_embedding(x_rope, pos)
print(f"RoPE output shape: {rope_out.shape}, dtype: {rope_out.dtype}")

# --- matmul ---
a = np.random.randn(3, 128).astype(np.float32)
b = np.random.randn(128, 64).astype(np.float32)
mm_out = adapter.matrix_multiply(a, b)
print(f"matmul output shape: {mm_out.shape}")

print("summary:", adapter.summary())
print("ALL SMOKE TESTS PASSED")
