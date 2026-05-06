#!/usr/bin/env python3
"""Verify DeepSeek-V4-Flash model files and configuration in WSL."""
import os
import json
import sys
import subprocess
from pathlib import Path

MODEL_DIR = os.path.expanduser("~/models/DeepSeek-V4-Flash")
EXPECTED_SHARDS = 46
EXPECTED_ARCH = "DeepseekV4ForCausalLM"
EXPECTED_MODEL_TYPE = "deepseek_v4"

def check(msg: str, ok: bool, detail: str = "") -> bool:
    status = "OK" if ok else "FAIL"
    line = f"  [{status}] {msg}"
    if detail and not ok:
        line += f" — {detail}"
    print(line)
    return ok

def main():
    print("=" * 60)
    print("DeepSeek-V4-Flash WSL Verification")
    print("=" * 60)

    all_ok = True

    # 1. Directory exists
    print("\n--- 1. Model Directory ---")
    if not os.path.isdir(MODEL_DIR):
        print(f"  [FAIL] Directory not found: {MODEL_DIR}")
        print(f"  Expected: {MODEL_DIR}")
        print("  Run: mkdir -p ~/models/DeepSeek-V4-Flash")
        sys.exit(1)
    all_ok &= check(f"Directory exists: {MODEL_DIR}", True)
    print(f"  Path resolved: {os.path.realpath(MODEL_DIR)}")

    # 2. config.json
    print("\n--- 2. config.json ---")
    config_path = os.path.join(MODEL_DIR, "config.json")
    if not os.path.isfile(config_path):
        all_ok &= check("config.json present", False, f"Missing: {config_path}")
    else:
        all_ok &= check("config.json present", True)
        with open(config_path) as f:
            cfg = json.load(f)
        arch = cfg.get("architectures", [])
        all_ok &= check(f"architectures = {arch}", arch and EXPECTED_ARCH in arch)

        # Key dimensional checks
        dims = {
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "intermediate_size": 2048,
            "vocab_size": 129280,
            "max_position_embeddings": 1048576,
            "rope_theta": 10000.0,
        }
        for key, expected in dims.items():
            actual = cfg.get(key)
            match = actual == expected
            all_ok &= check(f"{key} = {actual}", match, f"Expected {expected}")

        # MoE checks
        moe_checks = {
            "num_experts": 256,
            "num_shared_experts": 1,
        }
        for key, expected in moe_checks.items():
            actual = cfg.get(key, cfg.get(f"n_{key}", None))
            if actual is not None:
                match = actual == expected
                all_ok &= check(f"{key} = {actual}", match, f"Expected {expected}")

    # 3. Safetensors index
    print("\n--- 3. safetensors index ---")
    index_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        all_ok &= check("model.safetensors.index.json present", False, f"Missing: {index_path}")
    else:
        all_ok &= check("model.safetensors.index.json present", True)
        with open(index_path) as f:
            idx = json.load(f)
        wmap = idx.get("weight_map", {})
        print(f"  Weight map entries: {len(wmap)}")
        total_params_missing = idx.get("metadata", {}).get("total_size", 0)
        if total_params_missing:
            print(f"  Metadata total_size: {total_params_missing}")

        # Count shards
        shards = set(wmap.values())
        print(f"  Unique safetensor shards: {len(shards)}")
        all_ok &= check(
            f"Expected shards: {EXPECTED_SHARDS}",
            len(shards) == EXPECTED_SHARDS,
            f"Got {len(shards)}"
        )

        # Check for expert key patterns
        sample_expert_keys = [k for k in list(wmap)[:50] if "expert" in k.lower() or "mlp" in k.lower()]
        print(f"  Sample expert/MLP keys in first 50 entries: {len(sample_expert_keys)}")
        if sample_expert_keys:
            for k in sample_expert_keys[:5]:
                print(f"    {k} -> {wmap[k]}")

    # 4. Safetensor files
    print("\n--- 4. Safetensor files ---")
    safetensor_files = sorted(Path(MODEL_DIR).glob("*.safetensors"))
    print(f"  Found: {len(safetensor_files)} .safetensors files")
    expected_shards_list = list(range(1, EXPECTED_SHARDS + 1))
    found_shard_nums = []
    for sf in safetensor_files:
        name = sf.name
        size_gb = sf.stat().st_size / (1024**3)
        # Parse shard number
        try:
            num = int(name.replace("model-", "").replace(".safetensors", ""))
            found_shard_nums.append(num)
        except ValueError:
            pass
        print(f"    {name}: {size_gb:.2f} GB")

    missing_shards = set(expected_shards_list) - set(found_shard_nums)
    if missing_shards:
        all_ok &= check(
            "All shards present",
            False,
            f"Missing shards: {sorted(missing_shards)[:10]}..."
        )
    else:
        all_ok &= check(f"All {EXPECTED_SHARDS} shards present", True)

    # Total size
    total_sf_gb = sum(sf.stat().st_size for sf in safetensor_files) / (1024**3)
    print(f"  Total safetensors size: {total_sf_gb:.1f} GB")
    all_ok &= check(
        "Total size ~78 GB (MXFP4 packed)",
        70 < total_sf_gb < 90,
        f"Got {total_sf_gb:.1f} GB"
    )

    # 5. Tokenizer
    print("\n--- 5. Tokenizer ---")
    for fname in ["tokenizer.json", "tokenizer_config.json"]:
        path = os.path.join(MODEL_DIR, fname)
        all_ok &= check(f"{fname} present", os.path.isfile(path))

    # 6. Disk usage summary
    print("\n--- 6. Disk Usage ---")
    result = subprocess.run(
        ["du", "-sh", MODEL_DIR],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  du -sh: {result.stdout.strip()}")

    # 7. KTransformers compatibility
    print("\n--- 7. KTransformers Compatibility ---")
    kt_check = os.path.expanduser("~/ktransformers")
    if os.path.isdir(kt_check):
        all_ok &= check("KTransformers installed", True)
    else:
        all_ok &= check("KTransformers installed", False, f"Expected at ~/ktransformers")

    # Final summary
    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: ALL CHECKS PASSED ✅")
        print("DeepSeek-V4-Flash is ready for Astra inference.")
    else:
        print("RESULT: SOME CHECKS FAILED ❌")
        print("See FAIL items above. Common fixes:")
        print("  - Missing files: re-download from HuggingFace")
        print("  - Wrong path: ensure model is at ~/models/DeepSeek-V4-Flash")
        print("  - Corrupted shards: run md5sum check against HF index")
    print("=" * 60)
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())