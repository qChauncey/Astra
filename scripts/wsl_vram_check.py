#!/usr/bin/env python3
"""Check model feasibility for RTX 5070 12GB VRAM + 16GB system RAM with KTransformers."""

import os
import sys
sys.path.insert(0, os.path.expanduser("~/ktransformers/kt-kernel/python"))

vram = float(os.environ.get("KT_GPU_MEM_GB", "12.0"))
tp = 1       # single GPU

# ---- Select model profile (auto-detect or CLI argument) ----
model_name = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ASTRA_MODEL", "v4-flash")).lower()

# DeepSeek-V4-Flash: 284B total, 13B activated, 256 routed experts, 43 layers, hd=4096
# DeepSeek-V3.2:     671B total, 37B activated, 256 routed experts, 61 layers, hd=7168
if model_name in ("v4-flash", "v4flash", "deepseek-v4-flash"):
    total_params_b = 284
    total_experts = 256
    num_layers = 43
    hd = 4096
    expert_intermediate = 2048
    model_label = "DeepSeek-V4-Flash (284B)"
elif model_name in ("v4-pro", "v4pro", "deepseek-v4-pro"):
    total_params_b = 1600
    total_experts = 384
    num_layers = 61
    hd = 7168
    expert_intermediate = 3072
    model_label = "DeepSeek-V4-Pro (1.6T)"
elif model_name in ("v3", "v32", "v3.2", "deepseek-v3"):
    total_params_b = 671
    total_experts = 256
    num_layers = 61
    hd = 7168
    expert_intermediate = 2048
    model_label = "DeepSeek-V3.2 (671B)"
else:
    print(f"Unknown model: {model_name}. Defaulting to V4-Flash")
    total_params_b = 284
    total_experts = 256
    num_layers = 43
    hd = 4096
    expert_intermediate = 2048
    model_label = "DeepSeek-V4-Flash (284B)"

# KTransformers GPU expert count (try V3 compute util, fallback to formula)
gpu_experts = None
try:
    from cli.utils.model_registry import compute_deepseek_v3_gpu_experts
    gpu_experts = compute_deepseek_v3_gpu_experts(tp, vram)
except ImportError:
    # Fallback formula for V4 series: ~(vram - non_expert_gb) / expert_gb_per_layer
    pass
if gpu_experts is None or gpu_experts <= 0:
    gpu_experts = max(1, int(vram // 6))

# Expert params: 3 weight matrices * hidden * intermediate (gate + up + down)
expert_params_per = 3 * hd * expert_intermediate
expert_params_per_layer = total_experts * expert_params_per
total_expert_params = expert_params_per_layer * num_layers

# Non-expert: embeddings + attention (MLA) + norms + shared experts + LM head
non_expert_params = total_params_b * 1e9 - total_expert_params
non_expert_gb = non_expert_params * 2 / 1e9  # BF16 = 2 bytes

print(f"\n--- {model_label} Architecture ---")
print(f"Total params:       {total_params_b}B")
print(f"Expert params:      {total_expert_params/1e9:.1f}B")
print(f"Non-expert params:  {non_expert_params/1e9:.1f}B")
print(f"Non-expert BF16 GB: {non_expert_gb:.1f} GB")

# KTransformers CPU offload: all experts stay on CPU, non-experts in RAM
# Use MXFP4 download sizes for V4 series (2.7 GB per shard for Flash, ~10 GB per for Pro)
if model_name in ("v4-flash", "v4flash", "deepseek-v4-flash"):
    disk_gb = total_params_b * 2.0 / 8.0 * 1.1  # MXFP4 ≈ 2 bits avg → 0.25× BF16 + overhead
elif model_name in ("v4-pro", "v4pro", "deepseek-v4-pro"):
    disk_gb = total_params_b * 2.0 / 8.0 * 1.1
else:
    disk_gb = 340  # FP8 download for V3.2
disk_free = 295  # available (user-reported)
ram_free = 14    # available (user-reported)

print("\n--- System Resources ---")
print(f"Disk free:          {disk_free} GB")
model_short = model_name.upper() if model_name else "MODEL"
print(f"Disk needed ({model_short}): {disk_gb:.0f} GB  => {'OK' if disk_free > disk_gb else 'FAIL: not enough space'}")
print(f"RAM free:           {ram_free} GB")
print(f"RAM needed to load non-expert weights (BF16): {non_expert_gb:.1f} GB  => {'OK' if ram_free > non_expert_gb else 'FAIL: not enough RAM'}")
print(f"VRAM free:          {vram} GB")

# KTransformers default: kt-method=FP8, attention-backend=flashinfer
# With CPU offload, GPU only runs attention kernel + small expert cache
# Default kt-num-gpu-experts=1 means 1 expert layer stays in VRAM (rest CPU)
print("\n--- KTransformers Default Configuration ---")
if model_name in ("v4-flash", "v4flash", "deepseek-v4-flash", "v4-pro", "v4pro", "deepseek-v4-pro"):
    print("kt-method:          MXFP4 (CPU offloaded experts)")
else:
    print("kt-method:          FP8 (CPU quantized)")
print("attention-backend:  flashinfer (GPU)")
print(f"kt-num-gpu-experts: {gpu_experts} (auto-computed)")
print("kt-gpu-prefill-token-threshold: 4096")

# Expert per-layer VRAM: gate+up+down = 3 * hd * expert_intermediate * 2 bytes
expert_layer_gb = 3 * hd * expert_intermediate * 2 / 1e9
vram_for_kv = 2  # KV cache
vram_for_attention = 1  # attention buffers
vram_needed = expert_layer_gb + vram_for_kv + vram_for_attention
print("\n--- VRAM Breakdown (per layer, BF16) ---")
print(f"Expert weights:     {expert_layer_gb:.2f} GB")
print(f"KV cache buffer:    ~{vram_for_kv} GB")
print(f"Attention kernels:  ~{vram_for_attention} GB")
print(f"Total needed:       ~{vram_needed:.1f} GB")
print(f"VRAM available:     {vram} GB")
print(f"VRAM status:        {'OK' if vram > vram_needed else 'TIGHT - may OOM'}")

print("\n===== FINAL VERDICT =====")
if disk_free <= disk_gb:
    print(f"BLOCKED: Not enough disk space for {model_label} (need ~{disk_gb:.0f}GB, have {disk_free}GB)")
    print("RECOMMEND: Use MiniMax-M2.5 (78GB, already downloaded) instead")
elif ram_free <= non_expert_gb:
    print(f"BLOCKED: Not enough CPU RAM (need {non_expert_gb:.1f}GB, have {ram_free}GB)")
    print("RECOMMEND: Use MiniMax-M2.5 (much smaller non-expert footprint)")
elif vram <= vram_needed:
    print(f"TIGHT: VRAM may be borderline (need {vram_needed:.1f}GB, have {vram}GB)")
    print("CAN TRY but may OOM. MiniMax-M2.5 is safer.")
else:
    print(f"FEASIBLE: {model_label} should run on this system")
