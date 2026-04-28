#!/usr/bin/env python3
"""Verify MiniMax-M2.5 model shard integrity via WeightLoader.ModelIndex."""
import pathlib
import sys

# Add astra project root to path (run from WSL with cwd = project root)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from astra.inference.weight_loader import ModelIndex

MODEL_DIR = pathlib.Path("/home/chauncey/minimax-m2.5")

def main():
    print(f"Model dir: {MODEL_DIR}")
    idx = ModelIndex(MODEL_DIR)
    print(f"Total tensors in index: {len(idx._tensor_to_shard)}")
    
    shard_set = set(idx._tensor_to_shard.values())
    print(f"Unique shards: {len(shard_set)}")

    # Spot-check layer 0 MLP keys
    layer0_mlp = [k for k in sorted(idx._tensor_to_shard) if "layers.0.mlp" in k]
    print(f"Layer 0 MLP tensors: {len(layer0_mlp)}")
    for k in layer0_mlp[:5]:
        print(f"  {k} -> {idx._tensor_to_shard[k]}")

    # Spot-check last layer
    last_layer = 61
    last_mlp = [k for k in sorted(idx._tensor_to_shard) if f"layers.{last_layer}.mlp" in k]
    print(f"Layer {last_layer} MLP tensors: {len(last_mlp)}")

    # 4. Verify all shard files exist on disk
    missing = [s for s in sorted(shard_set) if not (MODEL_DIR / s).is_file()]
    print(f"Shards total: {len(shard_set)}, missing: {len(missing)}")
    if missing:
        print(f"MISSING: {missing[:10]}...")
        sys.exit(1)
    else:
        print("All shard files present!")

    # 5. Key naming analysis
    keys = sorted(idx._tensor_to_shard.keys())
    print("\nKey prefixes (first segment):")
    prefixes = {}
    for k in keys:
        p = k.split(".")[0]
        prefixes[p] = prefixes.get(p, 0) + 1
    for p, n in sorted(prefixes.items()):
        print(f"  {p}: {n}")

    # 6. Sample full keys
    print("\nSample tensor keys (first 20):")
    for k in keys[:20]:
        print(f"  {k}")

    # 7. Expert weight naming pattern
    expert_keys = [k for k in keys if "expert" in k.lower()]
    print(f"\nKeys containing 'expert': {len(expert_keys)}")
    if expert_keys:
        print(f"  Sample: {expert_keys[0]}")
        print(f"  Sample: {expert_keys[-1]}")

    # 8. Layer-like pattern (any key with a dot-digit-dot or digits)
    import re
    layer_pattern = re.compile(r"\.\d+\.")
    layer_keys = [k for k in keys if layer_pattern.search(k)]
    print(f"\nKeys matching layer pattern '.N.': {len(layer_keys)}")
    if layer_keys:
        print(f"  Sample: {layer_keys[0]}")
        print(f"  Sample: {layer_keys[-1]}")

    print("\nModel verification passed.")

if __name__ == "__main__":
    main()