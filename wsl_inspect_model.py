#!/usr/bin/env python3
"""Inspect DeepSeek-V2 model tensor naming format."""
import json

index_path = "/home/chauncey/deepseek-v4/model.safetensors.index.json"
with open(index_path) as f:
    data = json.load(f)

weight_map = data.get("weight_map", {})
print(f"Total tensors: {len(weight_map)}")

# Check expert keys for layer 0
layer0_mlp = [k for k in sorted(weight_map) if "layers.0.mlp" in k]
print(f"\nLayer 0 MLP keys ({len(layer0_mlp)}):")
for k in layer0_mlp[:20]:
    print(f"  {k} -> {weight_map[k]}")
if len(layer0_mlp) > 20:
    print(f"  ... and {len(layer0_mlp)-20} more")

# Check shared experts
shared_mlp = [k for k in sorted(weight_map) if "shared" in k.lower() and "mlp" in k]
print(f"\nShared MLP keys ({len(shared_mlp)}):")
for k in shared_mlp:
    print(f"  {k}")

# Check last layer to see if all layers are present
for layer_id in [0, 30, 59]:
    mlp_keys = [k for k in sorted(weight_map) if f"layers.{layer_id}.mlp" in k]
    print(f"\nLayer {layer_id} MLP keys: {len(mlp_keys)}")