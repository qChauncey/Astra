#!/usr/bin/env python3
"""Probe MiniMax-M2.5 FP8 scale_inv shapes."""
import json
import pathlib
import struct

MODEL_DIR = pathlib.Path("/home/chauncey/minimax-m2.5")

# Load index
with open(MODEL_DIR / "model.safetensors.index.json") as f:
    index = json.load(f)
wm = index["weight_map"]

# Attention tensors
for suffix in ["q_proj.weight", "q_proj.weight_scale_inv",
               "k_proj.weight", "k_proj.weight_scale_inv",
               "v_proj.weight", "v_proj.weight_scale_inv",
               "o_proj.weight", "o_proj.weight_scale_inv"]:
    name = f"model.layers.0.self_attn.{suffix}"
    shard = wm.get(name)
    # Read header from shard
    if shard:
        with open(MODEL_DIR / shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
        meta = header[name]
        print(f"{suffix}: shape={meta['shape']}, dtype={meta['dtype']}")

print("---")

# Expert tensors
for suffix in ["w1.weight", "w1.weight_scale_inv",
               "w2.weight", "w2.weight_scale_inv",
               "w3.weight", "w3.weight_scale_inv"]:
    name = f"model.layers.0.block_sparse_moe.experts.0.{suffix}"
    shard = wm.get(name)
    if shard:
        with open(MODEL_DIR / shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
        meta = header[name]
        print(f"{suffix}: shape={meta['shape']}, dtype={meta['dtype']}")
