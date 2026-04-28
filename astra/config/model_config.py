# Copyright 2025 Project Astra Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Centralised model configuration profiles for Astra.

Each profile describes the architecture constants of a supported MoE LLM.
All other modules import `get_model_config()` instead of hard-coding constants.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Optional


class AttentionType(enum.Enum):
    """Supported attention variants."""
    MLA = "mla"          # Multi-head Latent Attention (DeepSeek-V3/V4)
    GQA = "gqa"          # Grouped Query Attention (MiniMax-M2, Qwen2, Llama-3)


class QuantizationType(enum.Enum):
    """Supported weight quantization formats."""
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"


@dataclass(frozen=True)
class ModelConfig:
    """
    Architecture constants for a single MoE LLM.

    All numeric fields are immutable; downstream code should treat instances
    as read-only configuration atoms.
    """

    # ---- Identity ----
    model_id: str                    # HuggingFace repo (e.g. "MiniMaxAI/MiniMax-M2.5")
    display_name: str                # Human-readable (e.g. "MiniMax-M2.5")
    arch_type: str                   # HuggingFace architectures[0] value
    model_type: str                  # HuggingFace model_type field

    # ---- Transformer dimensions ----
    hidden_dim: int                  # d_model
    num_layers: int                  # number of transformer blocks
    head_dim: int                    # per-head dimension
    num_attention_heads: int         # total Q heads
    num_key_value_heads: int         # GQA KV heads (MLA: equals num_attention_heads)
    intermediate_size: int           # FFN intermediate (per expert)
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float                # RoPE base frequency
    rotary_dim: int                  # RoPE rotation dimension (MLA partial)
    rms_norm_eps: float

    # ---- Attention variant ----
    attention_type: AttentionType    # MLA or GQA

    # ---- MoE topology ----
    num_local_experts: int           # total routed experts
    num_experts_per_tok: int         # top-k experts activated per token
    num_shared_experts: int          # dense shared experts (0 for pure MoE)
    scoring_func: str                # "sigmoid" or "softmax"

    # ---- Quantization (required, must precede fields with defaults) ----
    native_quant: QuantizationType   # format shipped on HuggingFace

    # ---- Flags & optional fields (all with defaults below) ----
    use_qk_norm: bool = False
    use_routing_bias: bool = False

    # ---- MTP / speculative decoding ----
    use_mtp: bool = False
    num_mtp_modules: int = 0
    mtp_transformer_layers: int = 0

    # ---- KTransformers ----
    ktransformers_supported: bool = True
    ktransformers_arch_name: str = ""

    # ---- Model file layout ----
    num_safetensors_shards: int = 0
    total_size_gb: float = 0.0

    # ---- HuggingFace auto_map (for trust_remote_code models) ----
    auto_map: Dict[str, str] = field(default_factory=dict)

    @property
    def num_shared_experts_per_gpu(self) -> int:
        """Number of shared expert replicas needed in a micro-cluster."""
        return max(1, self.num_shared_experts)

    @property
    def total_experts(self) -> int:
        """Total experts including shared (for affinity routing)."""
        return self.num_local_experts + self.num_shared_experts

    @property
    def gqa_groups(self) -> int:
        """Number of GQA groups (KV head replication factor)."""
        if self.attention_type == AttentionType.MLA:
            return 1  # MLA is conceptually 1:1
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def per_layer_bytes_bf16(self) -> int:
        """Approximate bytes per transformer layer (BF16/FP16 weights only)."""
        # Rough estimate: 4 * hidden_dim^2 + MoE routing + attention projections
        base = 4 * self.hidden_dim * self.hidden_dim * 2  # bytes per param
        attn = 4 * self.hidden_dim * self.hidden_dim * 2
        ffn = 3 * self.hidden_dim * self.intermediate_size * 2
        return int(base + attn + ffn)


# =============================================================================
# Built-in model profiles
# =============================================================================

DEEPSEEK_V4_FLASH = ModelConfig(
    model_id="deepseek-ai/DeepSeek-V4-Flash",
    display_name="DeepSeek-V4-Flash",
    arch_type="DeepseekV3ForCausalLM",
    model_type="deepseek_v3",
    hidden_dim=7168,
    num_layers=61,
    head_dim=128,
    num_attention_heads=128,
    num_key_value_heads=128,          # MLA: full heads for latent compression
    intermediate_size=18432,
    vocab_size=129280,
    max_position_embeddings=163840,
    rope_theta=500000.0,
    rotary_dim=64,
    rms_norm_eps=1e-6,
    attention_type=AttentionType.MLA,
    use_qk_norm=True,
    num_local_experts=256,
    num_experts_per_tok=8,
    num_shared_experts=2,
    scoring_func="sigmoid",
    use_routing_bias=True,
    use_mtp=True,
    num_mtp_modules=3,
    mtp_transformer_layers=1,
    native_quant=QuantizationType.BF16,
    ktransformers_supported=True,
    ktransformers_arch_name="deepseek_v3",
    num_safetensors_shards=163,
    total_size_gb=671.0,
    auto_map={
        "AutoConfig": "configuration_deepseek.DeepseekV3Config",
        "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM",
    },
)

MINIMAX_M2_5 = ModelConfig(
    model_id="MiniMaxAI/MiniMax-M2.5",
    display_name="MiniMax-M2.5",
    arch_type="MiniMaxM2ForCausalLM",
    model_type="minimax_m2",
    hidden_dim=3072,
    num_layers=62,
    head_dim=128,
    num_attention_heads=48,
    num_key_value_heads=8,           # GQA: 48Q / 8KV = 6 groups
    intermediate_size=1536,
    vocab_size=200064,
    max_position_embeddings=196608,
    rope_theta=5000000.0,
    rotary_dim=64,
    rms_norm_eps=1e-6,
    attention_type=AttentionType.GQA,
    use_qk_norm=True,
    num_local_experts=256,
    num_experts_per_tok=8,
    num_shared_experts=0,            # Pure MoE — no shared experts
    scoring_func="sigmoid",
    use_routing_bias=True,
    use_mtp=True,
    num_mtp_modules=3,
    mtp_transformer_layers=1,
    native_quant=QuantizationType.FP8,
    ktransformers_supported=False,   # Not yet supported by KTransformers
    ktransformers_arch_name="minimax_m2",
    num_safetensors_shards=126,
    total_size_gb=126.0,
    auto_map={
        "AutoConfig": "configuration_minimax_m2.MiniMaxM2Config",
        "AutoModelForCausalLM": "modeling_minimax_m2.MiniMaxM2ForCausalLM",
    },
)


# =============================================================================
# Registry
# =============================================================================

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "deepseek-ai/DeepSeek-V4-Flash": DEEPSEEK_V4_FLASH,
    "deepseek-v4-flash": DEEPSEEK_V4_FLASH,
    "deepseekv4": DEEPSEEK_V4_FLASH,
    "MiniMaxAI/MiniMax-M2.5": MINIMAX_M2_5,
    "minimax-m2.5": MINIMAX_M2_5,
    "minimax-m2-5": MINIMAX_M2_5,
}
"""Canonical model configuration registry.  Keys are normalised model IDs."""

DEFAULT_MODEL: str = "minimax-m2.5"
"""Model to use when no model is explicitly specified."""


def register_model_config(config: ModelConfig) -> None:
    """Register a custom model profile at runtime."""
    MODEL_CONFIGS[config.model_id] = config
    MODEL_CONFIGS[config.display_name.lower()] = config


def get_model_config(model_id: Optional[str] = None) -> ModelConfig:
    """
    Resolve a model identifier to its ModelConfig.

    Parameters
    ----------
    model_id : str or None
        HuggingFace repo, short alias, or None for the default.

    Returns
    -------
    ModelConfig

    Raises
    ------
    KeyError
        If the provided identifier is not registered.
    """
    if model_id is None:
        model_id = DEFAULT_MODEL
    key = model_id.lower().replace("_", "-")
    if key in MODEL_CONFIGS:
        return MODEL_CONFIGS[key]
    # Try exact match (case-sensitive) as fallback
    for k, v in MODEL_CONFIGS.items():
        if k == model_id:
            return v
    raise KeyError(
        f"Unknown model '{model_id}'. "
        f"Registered: {sorted(set(MODEL_CONFIGS.keys()))}"
    )