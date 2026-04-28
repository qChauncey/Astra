#!/usr/bin/env python3
"""Inspect the DeepSeek model at /home/chauncey/deepseek-v4/config.json"""
import json

with open('/home/chauncey/deepseek-v4/config.json') as f:
    c = json.load(f)

print("=" * 60)
print("DEEPSEEK MODEL CONFIG (full)")
print("=" * 60)
for k, v in sorted(c.items()):
    print(f"  {k}: {v}")

# Identify the model version
arch = c.get('architectures', ['UNKNOWN'])[0]
model_type = c.get('model_type', 'UNKNOWN')
hidden_size = c.get('hidden_size', '?')
num_layers = c.get('num_hidden_layers', '?')
num_experts = c.get('num_experts', c.get('n_routed_experts', '?'))
num_kv_heads = c.get('num_key_value_heads', '?')
vocab_size = c.get('vocab_size', '?')

# Model fingerprints:
# DeepSeek-V2: arch=DeepseekV2ForCausalLM, hidden=5120, 160 experts, 60 layers
# DeepSeek-V3: arch=DeepseekV3ForCausalLM, hidden=7168, 256 experts, 61 layers  
# DeepSeek-R1: arch=DeepseekV3ForCausalLM (same base), hidden=7168, 256 experts, 61 layers
# DeepSeek-V4: arch would be DeepseekV4ForCausalLM (or V3), hidden unknown

print()
print("=" * 60)
print("MODEL IDENTIFICATION")
print("=" * 60)

# Check for V4 fingerprints
has_v4_field = any('v4' in str(k).lower() or 'v4' in str(v).lower() if isinstance(v, str) else False for k, v in c.items())

if arch == 'DeepseekV2ForCausalLM':
    if hidden_size == 5120 and str(num_layers) == '60' and str(num_experts) == '160':
        print("IDENTIFIED: DeepSeek-V2 (236B params, 160 experts, 60 layers, hd=5120)")
    elif hidden_size == 5120:
        print("IDENTIFIED: DeepSeek-V2 variant (hd=5120)")
    else:
        print(f"IDENTIFIED: DeepSeek-V2 architecture (hd={hidden_size}, layers={num_layers}, experts={num_experts})")
elif arch == 'DeepseekV3ForCausalLM':
    print("IDENTIFIED: DeepSeek-V3 / V3-0324 / R1-0528 (671B family)")
elif 'V4' in arch or 'v4' in str(arch).lower():
    print(f"IDENTIFIED: DeepSeek-V4 family ({arch})")
else:
    print(f"UNKNOWN architecture: {arch}")
    print(f"Model type: {model_type}")
    
# Check if this model is in KTransformers registry
print()
print("=" * 60)
print("KTRANSFORMERS COMPATIBILITY")
print("=" * 60)

# KTransformers supports 6 models:
# DeepSeek-V3-0324 (V3 architecture)
# DeepSeek-V3.2 (V3 architecture) 
# DeepSeek-R1-0528 (V3 architecture)
# Kimi-K2-Thinking (different arch)
# MiniMax-M2 (different arch)
# MiniMax-M2.1 (different arch)

if arch == 'DeepseekV2ForCausalLM':
    print("STATUS: NOT in KTransformers built-in registry")
    print("V2 architecture (DeepseekV2ForCausalLM) differs from V3 (DeepseekV3ForCausalLM)")
    print("The MLA kernel in KTransformers targets V3 tensor shapes")
    print("V2 uses different weight_layout/config keys; most V3 kernels will NOT work out-of-box")
elif arch == 'DeepseekV3ForCausalLM':
    print("STATUS: DEEPSEEKV3 ARCHITECTURE - compatible with these KTransformers models:")
    print("  - DeepSeek-V3-0324 (deepseek-ai/DeepSeek-V3-0324)")
    print("  - DeepSeek-V3.2    (deepseek-ai/DeepSeek-V3.2)")
    print("  - DeepSeek-R1-0528 (deepseek-ai/DeepSeek-R1-0528)")
    print("ACTION: Compare your model's tensor names against V3.2 to confirm compatibility")
else:
    print(f"STATUS: Unsupported architecture '{arch}'")

# Check for config.json special fields that indicate model version
print()
print("=" * 60)
print("VERSION CLUES (look for model-specific fields)")
print("=" * 60)
version_fields = [
    'model_version', 'version', 'deepseek_version',
    'num_nextn_predict_layers',  # V3 has 1 (MTP), V2 has 0
    'first_k_dense_replace',     # V3 specific
    'moe_intermediate_size',     # V3 specific
    'kv_lora_rank',              # MLA spec
    'q_lora_rank',               # MLA spec
    'qk_rope_head_dim',          # MLA spec
    'v_head_dim',                # MLA spec
    'scoring_func',              # Expert routing
    'topk_group',                # Expert routing
    'n_group',                   # Expert routing
    'topk_method',               # Expert routing
    'routed_scaling_factor',     # V3 specific
    'quantization_config',       # Any quant settings
]
for field in version_fields:
    if field in c:
        print(f"  {field}: {c[field]}")