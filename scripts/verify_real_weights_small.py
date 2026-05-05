#!/usr/bin/env python3
"""Lightweight real-weight verification: MiniMax-M2.5 single shard, single layer."""
import pathlib
import sys
import time
import numpy as np

_project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from astra.inference.weight_loader import MmapWeightStore, ModelIndex, detect_attention_format, AttentionFormat  # noqa: E402

MODEL_DIR = pathlib.Path("/home/chauncey/minimax-m2.5")

def check(msg, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        sys.exit(1)

# Stage 1: Index + format
print("=== Stage 1: ModelIndex + format detection ===")
index = ModelIndex(MODEL_DIR)
print(f"  Tensors: {len(index._tensor_to_shard)}, Shards: {len(index._shards)}")
fmt = detect_attention_format(MODEL_DIR)
print(f"  Format: {fmt}")
check("Format is GQA", fmt == AttentionFormat.GQA)
check("At least 90k tensors", len(index._tensor_to_shard) >= 90000)

# Stage 2: MmapWeightStore (zero-copy)
print("=== Stage 2: MmapWeightStore ===")
t0 = time.perf_counter()
store = MmapWeightStore(MODEL_DIR, max_open_shards=2, index=index)
dt = time.perf_counter() - t0
print(f"  Opened in {dt:.3f}s")
check("Mmap open < 5s", dt < 5.0)

embed = store.get_tensor("model.embed_tokens.weight")
print(f"  embed_tokens: shape={embed.shape}, dtype={embed.dtype}")
check("embed shape (200064, 3072)", embed.shape == (200064, 3072))

# Stage 3: Read GQA attention tensors via MmapWeightStore
# Shapes are (out_features, in_features) = (output_dim, hidden_dim)
# q_proj: 48 query heads * 128 head_dim = 6144 → (6144, 3072)
# k_proj: 8 KV heads * 128 head_dim = 1024 → (1024, 3072)
# v_proj: 8 KV heads * 128 head_dim = 1024 → (1024, 3072)
# o_proj: 3072 output, 48 query heads * 128 = 6144 input → (3072, 6144)
print("=== Stage 3: GQA attention tensors (layer 0) ===")
q_proj = store.get_tensor("model.layers.0.self_attn.q_proj.weight")
k_proj = store.get_tensor("model.layers.0.self_attn.k_proj.weight")
v_proj = store.get_tensor("model.layers.0.self_attn.v_proj.weight")
o_proj = store.get_tensor("model.layers.0.self_attn.o_proj.weight")
print(f"  q_proj.shape={q_proj.shape}, k_proj.shape={k_proj.shape}")
print(f"  v_proj.shape={v_proj.shape}, o_proj.shape={o_proj.shape}")
print(f"  dtypes: q={q_proj.dtype}, k={k_proj.dtype}, v={v_proj.dtype}, o={o_proj.dtype}")
# Check input dim = hidden_dim (3072) on the in_features axis
check("q_proj in_dim=3072 (shape[1])", q_proj.shape[1] == 3072)
check("q_proj out_dim=6144 (shape[0], 48Q heads)", q_proj.shape[0] == 6144)
check("k_proj in_dim=3072", k_proj.shape[1] == 3072)
check("k_proj out_dim=1024 (8 KV heads)", k_proj.shape[0] == 1024)
check("v_proj out_dim=1024 (8 KV heads)", v_proj.shape[0] == 1024)
check("o_proj in_dim=6144 (48Q heads)", o_proj.shape[1] == 6144)
check("o_proj out_dim=3072", o_proj.shape[0] == 3072)
# Weights stored as FP8 uint8 on disk
check("q_proj dtype uint8 (FP8 raw)", q_proj.dtype == np.uint8)
check("k_proj dtype uint8 (FP8 raw)", k_proj.dtype == np.uint8)

# Stage 4: Read MoE expert weight tensors via MmapWeightStore
print("=== Stage 4: MoE expert weight tensors (layer 0, expert 0) ===")
gate_raw = store.get_tensor("model.layers.0.block_sparse_moe.experts.0.w1.weight")
up_raw = store.get_tensor("model.layers.0.block_sparse_moe.experts.0.w3.weight")
down_raw = store.get_tensor("model.layers.0.block_sparse_moe.experts.0.w2.weight")
gate_scale = store.get_tensor("model.layers.0.block_sparse_moe.experts.0.w1.weight_scale_inv")
up_scale = store.get_tensor("model.layers.0.block_sparse_moe.experts.0.w3.weight_scale_inv")
down_scale = store.get_tensor("model.layers.0.block_sparse_moe.experts.0.w2.weight_scale_inv")
print(f"  gate_raw: shape={gate_raw.shape}, dtype={gate_raw.dtype}")
print(f"  up_raw:   shape={up_raw.shape},   dtype={up_raw.dtype}")
print(f"  down_raw: shape={down_raw.shape}, dtype={down_raw.dtype}")
print(f"  gate_scale: shape={gate_scale.shape}, dtype={gate_scale.dtype}")
print(f"  up_scale:   shape={up_scale.shape},   dtype={up_scale.dtype}")
print(f"  down_scale: shape={down_scale.shape}, dtype={down_scale.dtype}")
check("gate_raw dtype uint8 (FP8)", gate_raw.dtype == np.uint8)
check("up_raw dtype uint8 (FP8)", up_raw.dtype == np.uint8)
check("down_raw dtype uint8 (FP8)", down_raw.dtype == np.uint8)
check("gate_scale dtype float32", gate_scale.dtype == np.float32)
check("up_scale dtype float32", up_scale.dtype == np.float32)
check("down_scale dtype float32", down_scale.dtype == np.float32)

# Block-wise FP8 dequant (matching WeightLoader._dequant_minimax)
# scale shape: (r_blocks, c_blocks) where each block is 128×128
def _dequant_blockwise(w: np.ndarray, scale: np.ndarray) -> np.ndarray:
    w32 = w.astype(np.float32)
    s32 = scale.astype(np.float32)
    if s32.ndim <= 1:
        return w32 * s32
    r, c = w32.shape
    sr, sc = s32.shape
    br, bc = r // sr, c // sc
    w_blocks = w32.reshape(sr, br, sc, bc)
    w_blocks = w_blocks.transpose(0, 2, 1, 3)
    w_blocks = w_blocks * s32[:, :, np.newaxis, np.newaxis]
    return w_blocks.transpose(0, 2, 1, 3).reshape(r, c)

gate_dequant = _dequant_blockwise(gate_raw, gate_scale)
up_dequant = _dequant_blockwise(up_raw, up_scale)
down_dequant = _dequant_blockwise(down_raw, down_scale)
gate_rms = float(np.sqrt(np.mean(gate_dequant.astype(np.float64)**2)))
up_rms = float(np.sqrt(np.mean(up_dequant.astype(np.float64)**2)))
down_rms = float(np.sqrt(np.mean(down_dequant.astype(np.float64)**2)))
print(f"  dequant RMS: gate={gate_rms:.6f}, up={up_rms:.6f}, down={down_rms:.6f}")
check("gate RMS > 0.0001 after dequant", gate_rms > 0.0001)
check("up RMS > 0.0001 after dequant", up_rms > 0.0001)
check("down RMS > 0.0001 after dequant", down_rms > 0.0001)

store.close()
print()
print("ALL CHECKS PASSED - MiniMax-M2.5 real weights verified")
