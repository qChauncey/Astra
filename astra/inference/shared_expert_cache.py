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
#   - Implements "Shared Expert Pinning" strategy for DeepSeek-V4's
#     2 shared experts that fire on every token, keeping them resident in
#     GPU VRAM to eliminate repeated PCIe transfers.

"""
Shared Expert Pinning Cache.

DeepSeek-V4 has 2 shared experts activated for every token on top of the
top-K routed experts.  Keeping these always on GPU avoids the dominant I/O
bottleneck that occurs when MoE weights are paged from CPU RAM.

Usage::

    cache = SharedExpertCache(max_gpu_experts=4, hidden_dim=7168)
    cache.pin(expert_id=0, weights=expert0_weights)
    cache.pin(expert_id=1, weights=expert1_weights)
    out = cache.forward(expert_id=0, hidden_states=x)
"""

from __future__ import annotations

import threading
from typing import Dict

import numpy as np


class ExpertWeights:
    """Minimal representation of one MoE expert's weight matrices."""

    def __init__(
        self,
        expert_id: int,
        gate_proj: np.ndarray,   # (intermediate_dim, hidden_dim)
        up_proj: np.ndarray,     # (intermediate_dim, hidden_dim)
        down_proj: np.ndarray,   # (hidden_dim, intermediate_dim)
    ) -> None:
        self.expert_id = expert_id
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    @property
    def nbytes(self) -> int:
        return self.gate_proj.nbytes + self.up_proj.nbytes + self.down_proj.nbytes

    @classmethod
    def mock(
        cls,
        expert_id: int,
        hidden_dim: int = 7168,
        intermediate_dim: int = 2048,
        dtype: np.dtype = np.float16,
    ) -> "ExpertWeights":
        """Create zero-weight expert for simulation."""
        rng = np.random.default_rng(seed=expert_id)
        scale = 0.02
        return cls(
            expert_id=expert_id,
            gate_proj=rng.standard_normal((intermediate_dim, hidden_dim)).astype(dtype) * scale,
            up_proj=rng.standard_normal((intermediate_dim, hidden_dim)).astype(dtype) * scale,
            down_proj=rng.standard_normal((hidden_dim, intermediate_dim)).astype(dtype) * scale,
        )


class SharedExpertCache:
    """
    LRU-style pinned cache for MoE expert weights on the accelerator device.

    In production the 'device' is a CUDA tensor store; here we use numpy
    arrays on CPU RAM as a drop-in simulation layer.

    The two DeepSeek-V4 shared experts (IDs 0 and 1) should be pinned at
    startup and never evicted.
    """

    def __init__(
        self,
        max_cached_experts: int = 4,
        hidden_dim: int = 7168,
        intermediate_dim: int = 2048,
    ) -> None:
        self._max = max_cached_experts
        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._cache: Dict[int, ExpertWeights] = {}
        self._pinned: set[int] = set()
        self._access_order: list[int] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Cache management                                                      #
    # ------------------------------------------------------------------ #

    def pin(self, expert_id: int, weights: ExpertWeights) -> None:
        """Pin an expert permanently in cache (never evicted)."""
        with self._lock:
            self._cache[expert_id] = weights
            self._pinned.add(expert_id)
            if expert_id not in self._access_order:
                self._access_order.append(expert_id)

    def load(self, expert_id: int, weights: ExpertWeights) -> None:
        """Load an expert into cache, evicting LRU non-pinned entry if full."""
        with self._lock:
            if expert_id in self._cache:
                self._touch(expert_id)
                return
            if len(self._cache) >= self._max:
                self._evict_lru()
            self._cache[expert_id] = weights
            self._access_order.append(expert_id)

    def _touch(self, expert_id: int) -> None:
        if expert_id in self._access_order:
            self._access_order.remove(expert_id)
        self._access_order.append(expert_id)

    def _evict_lru(self) -> None:
        for eid in self._access_order:
            if eid not in self._pinned:
                del self._cache[eid]
                self._access_order.remove(eid)
                return
        raise RuntimeError("All cached experts are pinned; cannot evict")

    def is_cached(self, expert_id: int) -> bool:
        return expert_id in self._cache

    def cache_size(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        with self._lock:
            return {
                "cached_experts": list(self._cache.keys()),
                "pinned_experts": list(self._pinned),
                "cache_utilization": f"{len(self._cache)}/{self._max}",
                "total_bytes": sum(w.nbytes for w in self._cache.values()),
            }

    # ------------------------------------------------------------------ #
    # Forward pass                                                          #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        expert_id: int,
        hidden_states: np.ndarray,
    ) -> np.ndarray:
        """
        Run one expert's FFN forward pass.

        hidden_states: (num_tokens, hidden_dim) float16
        returns:       (num_tokens, hidden_dim) float16

        Uses SiLU-gated MLP matching DeepSeek-V4's expert architecture:
            out = down_proj( silu(gate_proj(x)) * up_proj(x) )
        """
        with self._lock:
            if expert_id not in self._cache:
                raise KeyError(f"Expert {expert_id} not in cache; call load() first")
            w = self._cache[expert_id]
            self._touch(expert_id)

        x = hidden_states.astype(np.float32)   # compute in fp32 for numerical stability
        gate = x @ w.gate_proj.T.astype(np.float32)
        up   = x @ w.up_proj.T.astype(np.float32)
        activated = self._silu(gate) * up
        out = activated @ w.down_proj.T.astype(np.float32)
        return out.astype(hidden_states.dtype)

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))
