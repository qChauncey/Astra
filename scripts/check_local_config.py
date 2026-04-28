#!/usr/bin/env python3
"""Read config.json from local model directory to identify which model."""
import json
import sys

path = "/home/chauncey/deepseek-v4/config.json"
try:
    with open(path) as f:
        d = json.load(f)
except FileNotFoundError:
    print(f"ERROR: {path} not found")
    sys.exit(1)

print("=== Model Identification ===")
print(f"architectures: {d.get('architectures')}")
print(f"model_type: {d.get('model_type')}")
print(f"num_hidden_layers: {d.get('num_hidden_layers')}")

# MoE specific
print("\n=== MoE Params ===")
for k in sorted(d.keys()):
    if any(term in k.lower() for term in ['expert', 'moe', 'n_routed', 'n_shared', 'num_expert']):
        print(f"  {k}: {d[k]}")

# Key dims
print("\n=== Key Dims ===")
for k in ['hidden_size', 'intermediate_size', 'num_attention_heads',
           'num_key_value_heads', 'vocab_size', 'max_position_embeddings',
           'torch_dtype', 'quantization_config']:
    if k in d:
        print(f"  {k}: {d[k]}")

# First layer config (if nested)
first_layer = d.get('first_k_dense_replace', None)
if first_layer is not None:
    print(f"\n  first_k_dense_replace: {first_layer}")

moe_layer_freq = d.get('moe_layer_freq', None)
if moe_layer_freq is not None:
    print(f"  moe_layer_freq: {moe_layer_freq}")