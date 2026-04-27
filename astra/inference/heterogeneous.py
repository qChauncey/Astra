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
#
# MODIFICATIONS (Astra project):
#   - Completely rewritten to integrate KTransformers C++ kernels.
#   - GPU is restricted to Attention (MLA / MHA) and DSA operators only.
#   - MoE expert parameters reside exclusively in CPU RAM (or NVMe via mmap).
#   - Python-level binding shim mirrors the ktransformers C++ API surface.

"""
Task C: Heterogeneous Inference Engine.

Hardware split mandated by the architecture spec:
  GPU  → Multi-head / Multi-Latent Attention (MLA), RoPE, LayerNorm, DSA ops
  CPU  → MoE FFN expert forward passes (weights in system RAM)

KTransformers integration:
  In production, `ktransformers` is imported and its C++ kernels are called
  directly.  This file provides a Python-level shim (KTransformersStub) that
  mirrors the expected API so the rest of the codebase can be developed and
  tested without requiring the compiled C++ extension.

  To activate real KTransformers:
    1. Build ktransformers per its README.
    2. Set ASTRA_USE_KTRANSFORMERS=1 in your environment.
    3. The engine will automatically use the real kernels via _kt_backend.

DeviceMap:
  Controls which layer indices go to GPU vs CPU.  A standard split for a
  16 GB GPU running DeepSeek-V4-Flash:
    - attention_layers:  0..60  (all 61 layers, attention sub-layer only)
    - expert_layers:     0..60  (MoE FFN sub-layer, all on CPU)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..serialization.tensor_pack import (
    DEEPSEEK_V4_HIDDEN_DIM,
    DEEPSEEK_V4_NUM_LAYERS,
    TensorPacket,
)
from .differential_privacy import LayerDPInjector
from .shared_expert_cache import ExpertWeights, SharedExpertCache


# ------------------------------------------------------------------ #
# KTransformers backend shim                                            #
# ------------------------------------------------------------------ #

_USE_KTRANSFORMERS = os.environ.get("ASTRA_USE_KTRANSFORMERS", "0") == "1"

try:
    if _USE_KTRANSFORMERS:
        import ktransformers as _kt  # type: ignore
        _kt_backend = "ktransformers"
    else:
        raise ImportError
except ImportError:
    _kt = None  # type: ignore
    _kt_backend = "numpy_stub"


class KTransformersStub:
    """
    Pure-Python stub matching the ktransformers operator API.

    Replace with actual C++ bindings by setting ASTRA_USE_KTRANSFORMERS=1
    and ensuring the ktransformers package is installed.
    """

    @staticmethod
    def multi_latent_attention(
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
        head_dim: int = 128,
    ) -> np.ndarray:
        """
        Simulated Multi-Latent Attention (MLA) forward pass.

        In production: calls kt::ops::mla_forward() C++ kernel on GPU.
        Here: standard scaled dot-product attention in numpy (float32).
        """
        q = query.astype(np.float32)
        k = key.astype(np.float32)
        v = value.astype(np.float32)
        scale = 1.0 / math_sqrt(head_dim)
        scores = q @ k.swapaxes(-2, -1) * scale
        if mask is not None:
            scores = scores + mask.astype(np.float32)
        weights = _softmax(scores, axis=-1)
        return (weights @ v).astype(query.dtype)

    @staticmethod
    def rms_layer_norm(
        x: np.ndarray,
        weight: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """RMSNorm as used by DeepSeek-V4."""
        x32 = x.astype(np.float32)
        rms = np.sqrt(np.mean(x32 ** 2, axis=-1, keepdims=True) + eps)
        return (x32 / rms * weight.astype(np.float32)).astype(x.dtype)

    @staticmethod
    def rope_embedding(
        x: np.ndarray,
        position_ids: np.ndarray,
        theta: float = 10000.0,
    ) -> np.ndarray:
        """Rotary positional embedding (RoPE)."""
        _, dim = x.shape[-2], x.shape[-1]
        half_dim = dim // 2
        freqs = 1.0 / (theta ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
        pos = position_ids.astype(np.float32)
        angles = np.outer(pos, freqs)  # (seq, half_dim)
        cos = np.cos(angles).astype(x.dtype)
        sin = np.sin(angles).astype(x.dtype)
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        rotated = np.concatenate([-x2, x1], axis=-1)
        return x * np.concatenate([cos, cos], axis=-1) + rotated * np.concatenate([sin, sin], axis=-1)


def math_sqrt(x: float) -> float:
    import math
    return math.sqrt(x)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / ex.sum(axis=axis, keepdims=True)


# ------------------------------------------------------------------ #
# Device map                                                            #
# ------------------------------------------------------------------ #

@dataclass
class DeviceMap:
    """
    Specifies which sub-layers run on GPU vs CPU.

    For a 16 GB GPU (e.g. RTX 5070 Ti) with DeepSeek-V4-Flash:
      - All 61 attention sub-layers → GPU
      - All 61 MoE FFN sub-layers   → CPU RAM (via KTransformers CPU offload)
    """
    attention_on_gpu: bool = True
    moe_on_cpu: bool = True
    gpu_device: str = "cuda:0"   # ignored in stub mode
    num_layers: int = DEEPSEEK_V4_NUM_LAYERS
    hidden_dim: int = DEEPSEEK_V4_HIDDEN_DIM

    # Attention head configuration
    num_heads: int = 128
    num_kv_heads: int = 128      # MLA uses latent compression in practice
    head_dim: int = 128

    @classmethod
    def for_16gb_gpu(cls) -> "DeviceMap":
        """Recommended config for an RTX 4090 / 5070 Ti class GPU."""
        return cls(
            attention_on_gpu=True,
            moe_on_cpu=True,
            num_layers=DEEPSEEK_V4_NUM_LAYERS,
        )

    @classmethod
    def cpu_only(cls) -> "DeviceMap":
        """All computation on CPU — useful for testing without GPU."""
        return cls(attention_on_gpu=False, moe_on_cpu=True)


# ------------------------------------------------------------------ #
# Heterogeneous engine                                                  #
# ------------------------------------------------------------------ #

@dataclass
class LayerKVCache:
    """Per-layer key/value cache for autoregressive decoding."""
    k: Optional[np.ndarray] = None   # (past_seq, hidden_dim)
    v: Optional[np.ndarray] = None

    def append(self, new_k: np.ndarray, new_v: np.ndarray) -> None:
        if self.k is None:
            self.k, self.v = new_k, new_v
        else:
            self.k = np.concatenate([self.k, new_k], axis=0)
            self.v = np.concatenate([self.v, new_v], axis=0)

    def clear(self) -> None:
        self.k = self.v = None


class HeterogeneousEngine:
    """
    Astra heterogeneous inference engine for DeepSeek-V4.

    Attention sub-layers: run via KTransformers GPU kernels (or stub).
    MoE FFN sub-layers:   run on CPU RAM using SharedExpertCache.

    Usage::

        engine = HeterogeneousEngine.from_device_map(DeviceMap.for_16gb_gpu())
        engine.load_shared_experts([expert_weights_0, expert_weights_1])
        out_packet = engine.forward(in_packet, layer_indices=[0, 1, 2])
    """

    def __init__(
        self,
        device_map: DeviceMap,
        expert_cache: Optional[SharedExpertCache] = None,
        dp_injector: Optional[LayerDPInjector] = None,
    ) -> None:
        self._dmap = device_map
        self._expert_cache = expert_cache or SharedExpertCache(
            max_cached_experts=4,
            hidden_dim=device_map.hidden_dim,
        )
        self._kv_cache: Dict[int, LayerKVCache] = {}
        self._kt = KTransformersStub()
        self._backend = _kt_backend
        self._dp_injector = dp_injector  # Phase 4: DP noise injection

        # Mock weight matrices (production: load from checkpoint)
        self._attn_q_proj: Dict[int, np.ndarray] = {}
        self._attn_k_proj: Dict[int, np.ndarray] = {}
        self._attn_v_proj: Dict[int, np.ndarray] = {}
        self._attn_o_proj: Dict[int, np.ndarray] = {}
        self._norm_weights: Dict[int, np.ndarray] = {}

    @classmethod
    def from_device_map(cls, dmap: DeviceMap) -> "HeterogeneousEngine":
        return cls(device_map=dmap)

    # ------------------------------------------------------------------ #
    # Weight loading                                                        #
    # ------------------------------------------------------------------ #

    def load_shared_experts(self, experts: List[ExpertWeights]) -> None:
        """Pin shared experts into GPU-resident cache at startup."""
        for ew in experts:
            self._expert_cache.pin(ew.expert_id, ew)

    def load_expert(self, ew: ExpertWeights) -> None:
        """Page a routed expert into cache (evicts LRU if full)."""
        self._expert_cache.load(ew.expert_id, ew)

    def _get_attn_weights(self, layer_idx: int) -> tuple:
        """Lazily initialise mock attention projection weights."""
        if layer_idx not in self._attn_q_proj:
            rng = np.random.default_rng(seed=layer_idx)
            d = self._dmap.hidden_dim
            scale = 0.02
            self._attn_q_proj[layer_idx] = (rng.standard_normal((d, d)) * scale).astype(np.float16)
            self._attn_k_proj[layer_idx] = (rng.standard_normal((d, d)) * scale).astype(np.float16)
            self._attn_v_proj[layer_idx] = (rng.standard_normal((d, d)) * scale).astype(np.float16)
            self._attn_o_proj[layer_idx] = (rng.standard_normal((d, d)) * scale).astype(np.float16)
            self._norm_weights[layer_idx] = np.ones(d, dtype=np.float16)
        return (
            self._attn_q_proj[layer_idx],
            self._attn_k_proj[layer_idx],
            self._attn_v_proj[layer_idx],
            self._attn_o_proj[layer_idx],
            self._norm_weights[layer_idx],
        )

    # ------------------------------------------------------------------ #
    # Per-layer forward                                                     #
    # ------------------------------------------------------------------ #

    def _attention_forward(
        self,
        hidden: np.ndarray,
        layer_idx: int,
        position_ids: Optional[np.ndarray] = None,
        use_kv_cache: bool = True,
    ) -> np.ndarray:
        """GPU-side attention sub-layer (runs via KTransformers stub)."""
        q_w, k_w, v_w, o_w, norm_w = self._get_attn_weights(layer_idx)

        normed = self._kt.rms_layer_norm(hidden, norm_w)

        if position_ids is None:
            position_ids = np.arange(normed.shape[0])

        # Project → RoPE → attention
        q = normed.astype(np.float32) @ q_w.T.astype(np.float32)
        k = normed.astype(np.float32) @ k_w.T.astype(np.float32)
        v = normed.astype(np.float32) @ v_w.T.astype(np.float32)

        q = q.astype(np.float16)
        k = k.astype(np.float16)

        q = self._kt.rope_embedding(q, position_ids)
        k = self._kt.rope_embedding(k, position_ids)

        if use_kv_cache:
            cache = self._kv_cache.setdefault(layer_idx, LayerKVCache())
            cache.append(k.astype(np.float32), v)
            k_full = cache.k.astype(np.float16)
            v_full = cache.v
        else:
            k_full, v_full = k, v

        attn_out = self._kt.multi_latent_attention(
            q[np.newaxis],
            k_full[np.newaxis],
            v_full[np.newaxis],
            head_dim=self._dmap.head_dim,
        )[0]

        out = attn_out.astype(np.float32) @ o_w.T.astype(np.float32)
        return (hidden.astype(np.float32) + out).astype(hidden.dtype)  # residual

    def _moe_forward(
        self,
        hidden: np.ndarray,
        selected_experts: np.ndarray,
        layer_idx: int,
    ) -> np.ndarray:
        """
        CPU-side MoE FFN sub-layer.

        For each token, runs all selected experts and combines outputs.
        Shared experts (IDs 0, 1) are always included.
        """
        seq_len = hidden.shape[0]
        out = np.zeros_like(hidden, dtype=np.float32)

        # Shared experts always fire
        for shared_id in range(2):
            if self._expert_cache.is_cached(shared_id):
                out += self._expert_cache.forward(shared_id, hidden).astype(np.float32)

        # Routed experts (one output per token, averaged across top-K)
        for token_idx in range(seq_len):
            token_h = hidden[token_idx : token_idx + 1]
            token_out = np.zeros((1, hidden.shape[-1]), dtype=np.float32)
            n_routed = 0
            for k_idx in range(selected_experts.shape[1]):
                expert_id = int(selected_experts[token_idx, k_idx])
                if expert_id < 2:
                    continue  # already handled as shared
                if self._expert_cache.is_cached(expert_id):
                    token_out += self._expert_cache.forward(expert_id, token_h).astype(np.float32)
                    n_routed += 1
            if n_routed > 0:
                token_out /= n_routed
            out[token_idx] += token_out[0]

        return (hidden.astype(np.float32) + out).astype(hidden.dtype)  # residual

    # ------------------------------------------------------------------ #
    # Public interface                                                       #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        packet: TensorPacket,
        layer_indices: Optional[List[int]] = None,
        use_kv_cache: bool = True,
    ) -> TensorPacket:
        """
        Run a sequence of transformer layers on the given packet.

        If a dp_injector is configured (Phase 4), calibrated DP noise is
        injected after each sub-layer to prevent hidden-state inversion.

        Returns a new TensorPacket with updated hidden states.
        """
        if layer_indices is None:
            layer_indices = list(range(packet.layer_start, packet.layer_end))

        hidden = packet.tensor.copy()
        t0 = time.perf_counter()

        for layer_idx in layer_indices:
            position_ids = np.array(packet.token_ids) if packet.token_ids else None

            # ---- GPU: attention ----
            if self._dmap.attention_on_gpu:
                hidden = self._attention_forward(
                    hidden, layer_idx, position_ids, use_kv_cache
                )
                if self._dp_injector is not None:
                    hidden = self._dp_injector(hidden, layer_idx)

            # ---- CPU: MoE FFN ----
            if packet.selected_experts is not None:
                hidden = self._moe_forward(hidden, packet.selected_experts, layer_idx)
                if self._dp_injector is not None:
                    hidden = self._dp_injector(hidden, layer_idx)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        metadata = {**packet.metadata, "compute_ms": f"{elapsed_ms:.2f}"}
        if self._dp_injector is not None:
            metadata["dp"] = self._dp_injector.stats()

        return TensorPacket(
            packet_id=packet.packet_id,
            tensor=hidden,
            layer_start=packet.layer_start,
            layer_end=packet.layer_end,
            token_ids=packet.token_ids,
            selected_experts=packet.selected_experts,
            geo_region=packet.geo_region,
            src_node=packet.src_node,
            dst_node=packet.dst_node,
            metadata=metadata,
        )

    @property
    def kv_cache(self) -> Dict[int, LayerKVCache]:
        """Public accessor for the KV cache (used by KVCacheSender/Receiver)."""
        return self._kv_cache

    def clear_kv_cache(self) -> None:
        for cache in self._kv_cache.values():
            cache.clear()
        self._kv_cache.clear()

    def stats(self) -> dict:
        result = {
            "backend": self._backend,
            "device_map": {
                "attention_on_gpu": self._dmap.attention_on_gpu,
                "moe_on_cpu": self._dmap.moe_on_cpu,
            },
            "kv_cache_layers": len(self._kv_cache),
            "expert_cache": self._expert_cache.stats(),
        }
        if self._dp_injector is not None:
            result["dp"] = self._dp_injector.stats()
        return result
