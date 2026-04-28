#!/usr/bin/env python3
"""Check V3.2 feasibility for RTX 5070 12GB VRAM + 16GB system RAM with KTransformers."""

import sys
sys.path.insert(0, '/home/chauncey/ktransformers/kt-kernel/python')

from cli.utils.model_registry import compute_deepseek_v3_gpu_experts

vram = 12.0  # GB
tp = 1       # single GPU

gpu_experts = compute_deepseek_v3_gpu_experts(tp, vram)
print(f"kt-num-gpu-experts for {vram}GB VRAM (TP={tp}): {gpu_experts}")

# DeepSeek-V3 architecture numbers (publicly known)
total_params_b = 671        # total parameters, billions
total_experts = 256         # routed experts per layer
num_layers = 61
hd = 7168                   # hidden_dim
expert_intermediate = 2048  # per-expert intermediate

# Expert params: 3 weight matrices * hidden * intermediate (gate + up + down)
expert_params_per = 3 * hd * expert_intermediate
expert_params_per_layer = total_experts * expert_params_per
total_expert_params = expert_params_per_layer * num_layers

# Non-expert: embeddings + attention (MLA) + norms + shared experts + LM head
non_expert_params = total_params_b * 1e9 - total_expert_params
non_expert_gb = non_expert_params * 2 / 1e9  # BF16 = 2 bytes

print("\n--- DeepSeek-V3.2 Architecture ---")
print(f"Total params:       {total_params_b}B")
print(f"Expert params:      {total_expert_params/1e9:.1f}B")
print(f"Non-expert params:  {non_expert_params/1e9:.1f}B")
print(f"Non-expert BF16 GB: {non_expert_gb:.1f} GB")

# KTransformers CPU offload: all experts stay on CPU, non-experts in RAM
disk_gb = 340  # FP8 download
disk_free = 295  # available
ram_free = 14   # available

print("\n--- System Resources ---")
print(f"Disk free:          {disk_free} GB")
print(f"Disk needed (V3.2): {disk_gb} GB  => {'OK' if disk_free > disk_gb else 'FAIL: not enough space'}")
print(f"RAM free:           {ram_free} GB")
print(f"RAM needed to load non-expert weights (BF16): {non_expert_gb:.1f} GB  => {'OK' if ram_free > non_expert_gb else 'FAIL: not enough RAM'}")
print(f"VRAM free:          {vram} GB")

# KTransformers default: kt-method=FP8, attention-backend=flashinfer
# With CPU offload, GPU only runs attention kernel + small expert cache
# Default kt-num-gpu-experts=1 means 1 expert layer stays in VRAM (rest CPU)
print("\n--- KTransformers V3.2 Default Configuration ---")
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
    print("BLOCKED: Not enough disk space for V3.2 (need ~340GB, have {disk_free}GB)")
    print("RECOMMEND: Use MiniMax-M2.5 (78GB, already downloaded) instead")
elif ram_free <= non_expert_gb:
    print(f"BLOCKED: Not enough CPU RAM (need {non_expert_gb:.1f}GB, have {ram_free}GB)")
    print("RECOMMEND: Use MiniMax-M2.5 (much smaller non-expert footprint)")
elif vram <= vram_needed:
    print(f"TIGHT: VRAM may be borderline (need {vram_needed:.1f}GB, have {vram}GB)")
    print("CAN TRY but may OOM. MiniMax-M2.5 is safer.")
else:
    print("FEASIBLE: DeepSeek-V3.2 should run on this system")