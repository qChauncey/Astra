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
Weight loader for HeterogeneousEngine — Phase 4.

Loads DeepSeek safetensors checkpoints into HeterogeneousEngine weight slots.
Only layers in [layer_start, layer_end) are loaded; unneeded shards are
never read.

Tensor name conventions (DeepSeek-V2 / V3 / V4)
------------------------------------------------
Attention:
  model.layers.{i}.self_attn.q_proj.weight        (or q_a_proj / q_b_proj for MLA)
  model.layers.{i}.self_attn.k_proj.weight
  model.layers.{i}.self_attn.v_proj.weight
  model.layers.{i}.self_attn.o_proj.weight
  model.layers.{i}.input_layernorm.weight

MoE FFN:
  model.layers.{i}.mlp.experts.{j}.gate_proj.weight
  model.layers.{i}.mlp.experts.{j}.up_proj.weight
  model.layers.{i}.mlp.experts.{j}.down_proj.weight
  model.layers.{i}.mlp.shared_experts.gate_proj.weight   (shared expert 0)
  model.layers.{i}.mlp.shared_experts.up_proj.weight
  model.layers.{i}.mlp.shared_experts.down_proj.weight

Usage::

    loader = WeightLoader("/data/deepseek-v4", layer_start=0, layer_end=20)
    loader.load_into(engine)           # loads all weights for layers 0–19
    loader.load_experts(engine, [0, 1])  # load only expert IDs 0 and 1
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Dict, List, Optional, Set

import numpy as np

log = logging.getLogger("astra.weight_loader")

# DeepSeek-V4-Flash published HuggingFace repository name
DEEPSEEK_V4_HF_REPO = "deepseek-ai/DeepSeek-V2"

# Attention weight keys (in order: q, k, v, o, norm)
_ATTN_SUFFIXES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "input_layernorm.weight",
]

# Expert weight keys
_EXPERT_SUFFIXES = [
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
]


# ─────────────────────────────────────────────────────────────────────────── #
# Safetensors helper (dependency-free fallback)                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def _load_safetensors(path: pathlib.Path) -> Dict[str, np.ndarray]:
    """
    Load a safetensors file into a dict of numpy arrays.
    Requires the ``safetensors`` package; raises ``ImportError`` if absent.
    """
    try:
        from safetensors import safe_open  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "safetensors package required for weight loading. "
            "Install with: pip install safetensors"
        ) from exc

    tensors: Dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


# ─────────────────────────────────────────────────────────────────────────── #
# Model index                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

class ModelIndex:
    """
    Parses ``model.safetensors.index.json`` to map tensor names → shard files.

    Falls back to scanning the directory for ``.safetensors`` files if the
    index is absent (single-shard models).
    """

    def __init__(self, model_dir: pathlib.Path) -> None:
        self._dir = model_dir
        self._tensor_to_shard: Dict[str, str] = {}
        self._shards: Set[str] = set()
        self._load()

    def _load(self) -> None:
        index_path = self._dir / "model.safetensors.index.json"
        if index_path.is_file():
            with open(index_path) as f:
                data = json.load(f)
            self._tensor_to_shard = data.get("weight_map", {})
            self._shards = set(self._tensor_to_shard.values())
            log.debug("Index loaded: %d tensors across %d shards",
                      len(self._tensor_to_shard), len(self._shards))
        else:
            # Single-shard: map all tensors to model.safetensors
            single = self._dir / "model.safetensors"
            if single.is_file():
                self._shards = {"model.safetensors"}
                log.debug("No index found; using single shard %s", single)
            else:
                log.warning("No safetensors files found in %s", self._dir)

    def shards_for_layers(self, layer_start: int, layer_end: int) -> Set[str]:
        """Return the shard filenames that contain any tensor for the given layers."""
        needed: Set[str] = set()
        prefix_set = {
            f"model.layers.{i}."
            for i in range(layer_start, layer_end)
        }
        for tensor_name, shard in self._tensor_to_shard.items():
            if any(tensor_name.startswith(p) for p in prefix_set):
                needed.add(shard)
        # If no index, return all shards (caller filters by key)
        return needed or self._shards

    def shard_for_tensor(self, tensor_name: str) -> Optional[str]:
        return self._tensor_to_shard.get(tensor_name)


# ─────────────────────────────────────────────────────────────────────────── #
# WeightLoader                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class WeightLoader:
    """
    Loads DeepSeek checkpoint weights into a HeterogeneousEngine.

    Parameters
    ----------
    model_dir:    Path to local directory containing safetensors files.
    layer_start:  First layer index this node handles.
    layer_end:    One-past-last layer index this node handles.
    """

    def __init__(
        self,
        model_dir: str | pathlib.Path,
        layer_start: int = 0,
        layer_end: int = 61,
    ) -> None:
        self._dir = pathlib.Path(model_dir)
        self.layer_start = layer_start
        self.layer_end = layer_end
        self._index = ModelIndex(self._dir)
        self._shard_cache: Dict[str, Dict[str, np.ndarray]] = {}

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_shard(self, shard_name: str) -> Dict[str, np.ndarray]:
        if shard_name not in self._shard_cache:
            path = self._dir / shard_name
            if not path.is_file():
                raise FileNotFoundError(f"Shard not found: {path}")
            log.info("Loading shard %s …", shard_name)
            self._shard_cache[shard_name] = _load_safetensors(path)
        return self._shard_cache[shard_name]

    def _get_tensor(self, name: str) -> Optional[np.ndarray]:
        shard = self._index.shard_for_tensor(name)
        if shard is None:
            # Try each shard (single-shard fallback)
            for s in self._index._shards:
                data = self._get_shard(s)
                if name in data:
                    return data[name]
            return None
        data = self._get_shard(shard)
        return data.get(name)

    def _layer_tensor(self, layer_idx: int, suffix: str) -> Optional[np.ndarray]:
        name = f"model.layers.{layer_idx}.{suffix}"
        t = self._get_tensor(name)
        if t is None:
            log.debug("Tensor not found: %s", name)
        return t

    # ── Public API ────────────────────────────────────────────────────────

    def load_into(self, engine) -> int:
        """
        Load attention weights + norms for all layers in [layer_start, layer_end).

        Returns the number of layers successfully loaded.
        """
        loaded = 0
        for i in range(self.layer_start, self.layer_end):
            ok = self._load_attention_layer(engine, i)
            if ok:
                loaded += 1
        log.info("Loaded attention weights for %d / %d layers",
                 loaded, self.layer_end - self.layer_start)
        return loaded

    def _load_attention_layer(self, engine, layer_idx: int) -> bool:
        """Load attention projection + norm weights for one layer."""
        q = self._layer_tensor(layer_idx, "self_attn.q_proj.weight")
        k = self._layer_tensor(layer_idx, "self_attn.k_proj.weight")
        v = self._layer_tensor(layer_idx, "self_attn.v_proj.weight")
        o = self._layer_tensor(layer_idx, "self_attn.o_proj.weight")
        norm = self._layer_tensor(layer_idx, "input_layernorm.weight")

        if any(t is None for t in [q, k, v, o, norm]):
            log.debug("Layer %d: incomplete attention weights, skipping", layer_idx)
            return False

        hd = engine._dmap.hidden_dim

        # Reshape/project to (hidden_dim, hidden_dim) if sizes differ.
        # DeepSeek MLA uses compressed KV; we take the first hidden_dim rows.
        def _fit(w: np.ndarray) -> np.ndarray:
            w = w.astype(np.float16)
            r, c = w.shape if w.ndim == 2 else (w.shape[0], w.shape[0])
            if w.ndim == 1:
                return w[:hd] if r >= hd else np.pad(w, (0, hd - r)).astype(np.float16)
            rows = min(r, hd)
            cols = min(c, hd)
            out = np.zeros((hd, hd), dtype=np.float16)
            out[:rows, :cols] = w[:rows, :cols]
            return out

        engine._attn_q_proj[layer_idx] = _fit(q)
        engine._attn_k_proj[layer_idx] = _fit(k)
        engine._attn_v_proj[layer_idx] = _fit(v)
        engine._attn_o_proj[layer_idx] = _fit(o)
        engine._norm_weights[layer_idx] = _fit(norm)
        return True

    def load_experts(self, engine, expert_ids: List[int]) -> int:
        """
        Load MoE expert weights for all layers in [layer_start, layer_end).

        Returns the count of (layer, expert) pairs successfully loaded.
        """
        loaded = 0
        for layer_idx in range(self.layer_start, self.layer_end):
            for eid in expert_ids:
                ew = self._load_one_expert(layer_idx, eid)
                if ew is not None:
                    if eid < 2:
                        engine.load_shared_experts([ew])
                    else:
                        engine.load_expert(ew)
                    loaded += 1
        log.info("Loaded %d expert weight sets", loaded)
        return loaded

    def _load_one_expert(self, layer_idx: int, expert_id: int) -> Optional[object]:
        """Return ExpertWeights for one (layer, expert) pair, or None."""
        from .shared_expert_cache import ExpertWeights

        prefix = "mlp.shared_experts" if expert_id < 2 else f"mlp.experts.{expert_id}"

        gate = self._layer_tensor(layer_idx, f"{prefix}.gate_proj.weight")
        up   = self._layer_tensor(layer_idx, f"{prefix}.up_proj.weight")
        down = self._layer_tensor(layer_idx, f"{prefix}.down_proj.weight")

        if any(t is None for t in [gate, up, down]):
            return None

        return ExpertWeights(
            expert_id=expert_id,
            gate_proj=gate.astype(np.float16),
            up_proj=up.astype(np.float16),
            down_proj=down.astype(np.float16),
        )

    def list_available_layers(self) -> List[int]:
        """Return layer indices that have at least one weight shard on disk."""
        available = []
        for i in range(self.layer_start, self.layer_end):
            if any(
                (self._dir / shard).is_file()
                for shard in self._index.shards_for_layers(i, i + 1)
            ):
                available.append(i)
        return available

    def clear_shard_cache(self) -> None:
        """Free in-memory shard data (safetensors buffers)."""
        self._shard_cache.clear()
