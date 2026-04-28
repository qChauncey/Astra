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

Loads safetensors checkpoints into HeterogeneousEngine weight slots.
Only layers in [layer_start, layer_end) are loaded; unneeded shards are
never read.

Supports three attention formats via auto-detection:

  Standard MHA (legacy, DeepSeek-V2):
    model.layers.{i}.self_attn.q_proj.weight
    model.layers.{i}.self_attn.k_proj.weight
    model.layers.{i}.self_attn.v_proj.weight
    model.layers.{i}.self_attn.o_proj.weight
    model.layers.{i}.input_layernorm.weight

  Multi-head Latent Attention (MLA, DeepSeek-V3 / V4 / R1):
    model.layers.{i}.self_attn.q_a_proj.weight
    model.layers.{i}.self_attn.q_b_proj.weight
    model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight
    model.layers.{i}.self_attn.kv_b_proj.weight
    model.layers.{i}.self_attn.o_proj.weight
    model.layers.{i}.self_attn.q_a_layernorm.weight
    model.layers.{i}.self_attn.kv_a_layernorm.weight
    model.layers.{i}.input_layernorm.weight

  Grouped Query Attention (GQA, MiniMax-M2.x / Qwen2 / Llama-3):
    model.layers.{i}.self_attn.q_proj.weight
    model.layers.{i}.self_attn.k_proj.weight
    model.layers.{i}.self_attn.v_proj.weight
    model.layers.{i}.self_attn.o_proj.weight
    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.pre_attention_layernorm.weight   (MiniMax-M2.x)
    model.layers.{i}.post_attention_layernorm.weight  (MiniMax-M2.x)
    model.layers.{i}.qk_norm.weight                   (MiniMax-M2.x, if use_qk_norm)

Detection priority:
  1. Check safetensors index / shard for ``q_a_proj`` → MLA mode.
  2. Check safetensors index for ``pre_attention_layernorm`` → GQA mode
     (MiniMax-M2.x style dual-norm attention).
  3. Otherwise legacy MHA mode (works for both legacy DeepSeek-V2 MHA
     and standard GQA models that only use 5 core projections).

MoE FFN (shared experts and routed experts) tensor names are shared
across all formats:

  model.layers.{i}.mlp.experts.{j}.gate_proj.weight
  model.layers.{i}.mlp.experts.{j}.up_proj.weight
  model.layers.{i}.mlp.experts.{j}.down_proj.weight
  model.layers.{i}.mlp.shared_experts.gate_proj.weight
  model.layers.{i}.mlp.shared_experts.up_proj.weight
  model.layers.{i}.mlp.shared_experts.down_proj.weight

For GQA models (MiniMax-M2.x), attention loading respects the model's
native hidden_dim (3072) rather than using DeepSeek-style shape fitting.
Tensors are loaded as-is and stored in the engine's GQA slot.

Memory-mapped weight access (mmap):
  MmapWeightStore maps safetensors shard files directly into virtual
  memory using numpy's memmap-like interface.  This allows large
  checkpoints (580 GB+ DeepSeek-V4 or 126 GB MiniMax-M2.5) to sit on
  NVMe without being fully read into RAM.  The store maintains an LRU
  cache of open shard file handles.

Usage::

    # MLA model (DeepSeek-V4)
    loader = WeightLoader("/data/deepseek-v4", layer_start=0, layer_end=20)
    loader.load_into(engine)

    # GQA model (MiniMax-M2.5)
    loader = WeightLoader("/data/minimax-m2.5", layer_start=0, layer_end=30)
    loader.load_into(engine)

    # mmap mode - zero-copy tensor access for NVMe-hosted checkpoints
    store = MmapWeightStore("/data/minimax-m2.5")
    w = store.get_tensor("model.layers.0.self_attn.q_proj.weight")
    w_read_once = store.get_tensor_read_once(shard_path, name)
"""

from __future__ import annotations

import io
import json
import logging
import mmap
import os
import pathlib
import struct
from collections import OrderedDict
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger("astra.weight_loader")

# ------------------------------------------------------------------ #
# Attention format constants                                          #
# ------------------------------------------------------------------ #

class AttentionFormat:
    """Enumeration of supported attention formats."""
    LEGACY = "legacy"   # Standard MHA (DeepSeek-V2) or basic GQA
    MLA = "mla"          # Multi-head Latent Attention (DeepSeek-V3/V4)
    GQA = "gqa"          # Grouped Query Attention (MiniMax-M2, Qwen2, Llama-3)


# ------------------------------------------------------------------ #
# Legacy MHA / GQA attention weight keys                               #
# ------------------------------------------------------------------ #
_GQA_ATTN_SUFFIXES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "input_layernorm.weight",
    "pre_attention_layernorm.weight",
    "post_attention_layernorm.weight",
    "qk_norm.weight",
]

# Aliases for backward compatibility
_ATTN_SUFFIXES = _GQA_ATTN_SUFFIXES
_LEGACY_ATTN_SUFFIXES = _GQA_ATTN_SUFFIXES

# ------------------------------------------------------------------ #
# MLA (Multi-head Latent Attention) tensor name mapping                #
# ------------------------------------------------------------------ #
# Each entry:  safetensors suffix -> MLAWeights dataclass field name.
_MLA_ATTN_TENSORS: Dict[str, str] = {
    "self_attn.q_a_proj.weight":               "q_a_proj",
    "self_attn.q_b_proj.weight":               "q_b_proj",
    "self_attn.kv_a_proj_with_mqa.weight":     "kv_a_proj",
    "self_attn.kv_b_proj.weight":              "kv_b_proj",
    "self_attn.o_proj.weight":                 "o_proj",
    "self_attn.q_a_layernorm.weight":          "q_norm",
    "self_attn.kv_a_layernorm.weight":         "kv_norm",
    "input_layernorm.weight":                  "attn_norm",
}

# Public alias for external consumers (tests, engine)
MLA_TENSOR_MAP = _MLA_ATTN_TENSORS

# ------------------------------------------------------------------ #
# GQA (Grouped Query Attention) tensor name mapping                    #
# ------------------------------------------------------------------ #
# Each entry:  safetensors suffix -> GQAWeights dataclass field name.
_GQA_ATTN_TENSORS: Dict[str, str] = {
    "self_attn.q_proj.weight":               "q_proj",
    "self_attn.k_proj.weight":               "k_proj",
    "self_attn.v_proj.weight":               "v_proj",
    "self_attn.o_proj.weight":               "o_proj",
    "input_layernorm.weight":                "attn_norm",
    "pre_attention_layernorm.weight":        "pre_attn_norm",
    "post_attention_layernorm.weight":       "post_attn_norm",
    "qk_norm.weight":                        "qk_norm",
}

# Public alias
GQA_TENSOR_MAP = _GQA_ATTN_TENSORS

# Expert weight keys (standard format: mlp.experts.{id}.{proj}.weight)
_EXPERT_SUFFIXES = [
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
]

# Expert weight keys for MiniMax-M2.x block_sparse_moe format.
# Mapping:  w1 → gate_proj, w2 → up_proj, w3 → down_proj
_MINIMAX_MOE_EXPERT_SUFFIXES: Dict[str, str] = {
    "w1.weight": "gate_proj",
    "w2.weight": "up_proj",
    "w3.weight": "down_proj",
}
_MINIMAX_MOE_EXPERT_TENSORS: List[str] = list(_MINIMAX_MOE_EXPERT_SUFFIXES.keys())

# Safetensors dtype byte -> numpy dtype
_DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.dtype(np.uint16),  # raw bytes - consumer casts to float32
    "F8_E4M3": np.dtype(np.uint8),  # raw bytes - FP8 (1 byte/elem, MiniMax-M2.x)
    "F8_E5M2": np.dtype(np.uint8),  # raw bytes - FP8 alt format
    "I32": np.int32,
    "I64": np.int64,
    "F64": np.float64,
    "BOOL": np.bool_,
}


def detect_attention_format(model_dir: str | pathlib.Path) -> str:
    """Detect the attention format of a checkpoint directory.

    Probes ``model.safetensors.index.json`` (or first safetensors shard)
    for distinctive tensor name patterns.

    Returns
    -------
    str
        One of ``"mla"``, ``"gqa"``, or ``"legacy"``.
    """
    model_dir = pathlib.Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"

    # Also check config.json for model_type hint
    config_path = model_dir / "config.json"
    if config_path.is_file():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type", "")
            if model_type in ("minimax_m2",):
                return AttentionFormat.GQA
        except (json.JSONDecodeError, OSError):
            pass

    if index_path.is_file():
        try:
            with open(index_path) as f:
                data = json.load(f)
            weight_map = data.get("weight_map", {})
            for name in weight_map:
                if "q_a_proj" in name:
                    return AttentionFormat.MLA
                if "pre_attention_layernorm" in name:
                    return AttentionFormat.GQA
        except (json.JSONDecodeError, OSError):
            pass

    # No index - peek at the first shard header
    shard = model_dir / "model.safetensors"
    if not shard.is_file():
        candidates = list(model_dir.glob("*.safetensors"))
        if not candidates:
            return AttentionFormat.LEGACY
        shard = candidates[0]

    try:
        with open(shard, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return AttentionFormat.LEGACY
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
            for name in header:
                if name != "__metadata__":
                    if "q_a_proj" in name:
                        return AttentionFormat.MLA
                    if "pre_attention_layernorm" in name:
                        return AttentionFormat.GQA
    except (OSError, struct.error, json.JSONDecodeError):
        pass

    return AttentionFormat.LEGACY


def detect_mla_format(model_dir: str | pathlib.Path) -> bool:
    """Return True if the model directory contains MLA-formatted tensors.

    Kept for backward compatibility.  Prefer ``detect_attention_format()``
    for new code.
    """
    return detect_attention_format(model_dir) == AttentionFormat.MLA


# ------------------------------------------------------------------ #
# Safetensors helper (dependency-free fallback)                       #
# ------------------------------------------------------------------ #

def _load_safetensors(path: pathlib.Path) -> Dict[str, np.ndarray]:
    """Load a safetensors file into a dict of numpy arrays.

    Reads tensor data directly from the file using header offsets, bypassing
    the safetensors ``safe_open`` API entirely.  This avoids NumPy dtype
    compatibility issues with bfloat16 (used by MiniMax-M2.5 and other
    FP8-era checkpoints).  BF16 tensors are loaded as uint16 raw bytes;
    downstream consumers cast to float32 as needed.
    """
    # We need the _original_ raw header bytes to compute the correct data
    # offset — re-serialising the JSON may produce a different number of
    # bytes than the on-disk representation.
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        raw_header_bytes = f.read(header_len)
        data_offset = 8 + len(raw_header_bytes)
        header = json.loads(raw_header_bytes.decode("utf-8"))

        tensors: Dict[str, np.ndarray] = {}
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            dtype_str = meta["dtype"]
            shape = meta["shape"]
            start, end = meta["data_offsets"]
            byte_len = end - start
            f.seek(data_offset + start)
            raw = f.read(byte_len)
            np_dtype = _DTYPE_MAP.get(dtype_str, np.float16)
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
            tensors[key] = arr
        return tensors


def _parse_safetensors_header(path: pathlib.Path) -> Dict:
    """Parse the JSON header of a safetensors file without loading tensors."""
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header


# ------------------------------------------------------------------ #
# Model index                                                        #
# ------------------------------------------------------------------ #

class ModelIndex:
    """Parse ``model.safetensors.index.json`` to map tensor names to shard files."""

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
            single = self._dir / "model.safetensors"
            if single.is_file():
                self._shards = {"model.safetensors"}
                log.debug("No index found; using single shard %s", single)
            else:
                log.warning("No safetensors files found in %s", self._dir)

    def shards_for_layers(self, layer_start: int, layer_end: int) -> Set[str]:
        """Return shard filenames that contain tensors for the given layer range."""
        needed: Set[str] = set()
        for i in range(layer_start, layer_end):
            prefix = f"model.layers.{i}."
            for tensor_name, shard in self._tensor_to_shard.items():
                if tensor_name.startswith(prefix):
                    needed.add(shard)
        return needed or self._shards

    def shard_for_tensor(self, tensor_name: str) -> Optional[str]:
        return self._tensor_to_shard.get(tensor_name)

    @property
    def tensor_map(self) -> Dict[str, str]:
        return dict(self._tensor_to_shard)


# ------------------------------------------------------------------ #
# WeightLoader                                                        #
# ------------------------------------------------------------------ #

class WeightLoader:
    """Load safetensors checkpoint weights into a HeterogeneousEngine.

    Auto-detects attention format (MLA, GQA, or legacy MHA) and dispatches
    weight loading accordingly.

    Parameters
    ----------
    model_dir:    Path to local directory containing safetensors files.
    layer_start:  First layer index this node handles.
    layer_end:    One-past-last layer index this node handles.
    verify_integrity:  If True, validate shards against manifest.
    """

    def __init__(
        self,
        model_dir: str | pathlib.Path,
        layer_start: int = 0,
        layer_end: int = 61,
        verify_integrity: bool = True,
    ) -> None:
        self._dir = pathlib.Path(model_dir)
        self.layer_start = layer_start
        self.layer_end = layer_end
        self._index = ModelIndex(self._dir)
        self._shard_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._verify_integrity = verify_integrity
        self._manifest = self._load_manifest() if verify_integrity else None
        self._verified_shards: set[str] = set()
        self._attn_format = detect_attention_format(self._dir)
        log.info("Detected attention format: %s for %s", self._attn_format, self._dir)

    @property
    def attention_format(self) -> str:
        """The detected attention format: ``"mla"``, ``"gqa"``, or ``"legacy"``."""
        return self._attn_format

    @property
    def is_mla(self) -> bool:
        """True if the checkpoint uses Multi-head Latent Attention."""
        return self._attn_format == AttentionFormat.MLA

    @property
    def is_gqa(self) -> bool:
        """True if the checkpoint uses Grouped Query Attention."""
        return self._attn_format == AttentionFormat.GQA

    def _load_manifest(self):
        """Load weight manifest if present. Returns None if not found."""
        from .weight_manifest import WeightManifest, find_manifest
        path = find_manifest(self._dir)
        if path is None:
            log.warning(
                "No astra_manifest.json in %s - weight integrity NOT verified.",
                self._dir,
            )
            return None
        try:
            m = WeightManifest.load(path)
            log.info("Loaded manifest: %d shards, algorithm=%s", len(m), m.algorithm)
            return m
        except Exception as exc:
            log.error("Failed to load manifest %s: %s", path, exc)
            return None

    # -- Internal helpers --------------------------------------------------

    def _get_shard(self, shard_name: str) -> Dict[str, np.ndarray]:
        if shard_name not in self._shard_cache:
            path = self._dir / shard_name
            if not path.is_file():
                raise FileNotFoundError(f"Shard not found: {path}")
            self._verify_shard(shard_name, path)
            log.info("Loading shard %s ...", shard_name)
            self._shard_cache[shard_name] = _load_safetensors(path)
        return self._shard_cache[shard_name]

    def _verify_shard(self, shard_name: str, path: pathlib.Path) -> None:
        if self._manifest is None:
            return
        if shard_name in self._verified_shards:
            return
        if shard_name not in self._manifest:
            raise RuntimeError(
                f"Shard {shard_name!r} is not in the manifest. "
                "Possible tampering or unauthorised file."
            )
        if not self._manifest.verify_file(path):
            raise RuntimeError(
                f"Shard {shard_name!r} hash mismatch - refusing to load."
            )
        self._verified_shards.add(shard_name)
        log.debug("Verified shard %s OK", shard_name)

    def _get_tensor(self, name: str) -> Optional[np.ndarray]:
        shard = self._index.shard_for_tensor(name)
        if shard is None:
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

    # -- Public API --------------------------------------------------------

    def load_into(self, engine) -> int:
        """Load attention weights + norms for layers in [layer_start, layer_end).

        Dispatches to the correct loader based on detected attention format.

        Returns the number of layers successfully loaded.
        """
        if self._attn_format == AttentionFormat.MLA:
            return self._load_mla_attention(engine)
        elif self._attn_format == AttentionFormat.GQA:
            return self._load_gqa_attention(engine)
        else:
            return self._load_legacy_attention(engine)

    # -- Legacy attention loading ------------------------------------------

    def _load_legacy_attention(self, engine) -> int:
        """Load standard MHA attention weights (q, k, v, o, norm)."""
        loaded = 0
        for i in range(self.layer_start, self.layer_end):
            ok = self._load_legacy_attention_layer(engine, i)
            if ok:
                loaded += 1
        log.info("Loaded legacy attention weights for %d / %d layers",
                 loaded, self.layer_end - self.layer_start)
        return loaded

    def _load_legacy_attention_layer(self, engine, layer_idx: int) -> bool:
        q = self._layer_tensor(layer_idx, "self_attn.q_proj.weight")
        k = self._layer_tensor(layer_idx, "self_attn.k_proj.weight")
        v = self._layer_tensor(layer_idx, "self_attn.v_proj.weight")
        o = self._layer_tensor(layer_idx, "self_attn.o_proj.weight")
        norm = self._layer_tensor(layer_idx, "input_layernorm.weight")

        if any(t is None for t in [q, k, v, o, norm]):
            log.debug("Layer %d: incomplete attention weights, skipping", layer_idx)
            return False

        hd = engine._dmap.hidden_dim

        def _fit(w: np.ndarray) -> np.ndarray:
            w = w.astype(np.float16)
            r = w.shape[0] if w.ndim == 2 else w.shape[0]
            c = w.shape[1] if w.ndim == 2 else r
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

    # -- MLA attention loading ---------------------------------------------

    def _load_mla_attention(self, engine) -> int:
        """Load MLA attention weights (q_a_proj, kv_a_proj, etc.)."""
        from .heterogeneous import MLAWeights

        loaded = 0
        for i in range(self.layer_start, self.layer_end):
            mla_weights = self._load_mla_attention_layer(i)
            if mla_weights is not None:
                engine._mla_weights[i] = mla_weights
                loaded += 1
            else:
                log.debug("Layer %d: incomplete MLA weights, skipping", i)
        log.info("Loaded MLA attention weights for %d / %d layers",
                 loaded, self.layer_end - self.layer_start)
        engine.enable_mla_mode()
        return loaded

    def _load_mla_attention_layer(self, layer_idx: int) -> Optional[object]:
        """Build MLAWeights for one layer from checkpoint tensors.

        Returns MLAWeights on success, or None if any required tensor is missing.
        """
        from .heterogeneous import MLAWeights

        tensors: Dict[str, Optional[np.ndarray]] = {}
        for suffix in _MLA_ATTN_TENSORS:
            t = self._layer_tensor(layer_idx, suffix)
            if t is None:
                log.debug("Layer %d: MLA tensor %s not found", layer_idx, suffix)
                return None
            tensors[suffix] = t.astype(np.float16)

        return MLAWeights(
            layer_idx=layer_idx,
            q_a_proj=tensors["self_attn.q_a_proj.weight"],
            q_b_proj=tensors["self_attn.q_b_proj.weight"],
            kv_a_proj=tensors["self_attn.kv_a_proj_with_mqa.weight"],
            kv_b_proj=tensors["self_attn.kv_b_proj.weight"],
            o_proj=tensors["self_attn.o_proj.weight"],
            q_norm=tensors["self_attn.q_a_layernorm.weight"],
            kv_norm=tensors["self_attn.kv_a_layernorm.weight"],
            attn_norm=tensors["input_layernorm.weight"],
        )

    # -- GQA attention loading ---------------------------------------------

    def _load_gqa_attention(self, engine) -> int:
        """Load GQA attention weights for MiniMax-M2.x / Qwen2 / Llama-3 style.

        Uses the engine's native hidden_dim from the model config (e.g. 3072
        for MiniMax-M2.5) rather than legacy DeepSeek-style fitting.
        """
        from .heterogeneous import GQAWeights

        loaded = 0
        for i in range(self.layer_start, self.layer_end):
            gqa_weights = self._load_gqa_attention_layer(i)
            if gqa_weights is not None:
                if not hasattr(engine, '_gqa_weights'):
                    engine._gqa_weights = {}
                engine._gqa_weights[i] = gqa_weights
                loaded += 1
            else:
                log.debug("Layer %d: incomplete GQA weights, skipping", i)
        log.info("Loaded GQA attention weights for %d / %d layers",
                 loaded, self.layer_end - self.layer_start)
        if loaded > 0:
            engine.enable_gqa_mode()
        return loaded

    def _load_gqa_attention_layer(self, layer_idx: int) -> Optional[object]:
        """Build GQAWeights for one layer from checkpoint tensors.

        Tries to load all known GQA tensor suffixes.  Required tensors are
        q_proj, k_proj, v_proj, o_proj, and input_layernorm.  Optional
        tensors (pre/post_attention_layernorm, qk_norm) are loaded if present.

        For MiniMax-M2.x FP8 checkpoints, companion ``weight_scale_inv``
        tensors are applied to dequantise Q/K/V/O projection weights.
        """
        from .heterogeneous import GQAWeights

        # Required tensors (raw FP8 for MiniMax-M2.x)
        q = self._layer_tensor(layer_idx, "self_attn.q_proj.weight")
        k = self._layer_tensor(layer_idx, "self_attn.k_proj.weight")
        v = self._layer_tensor(layer_idx, "self_attn.v_proj.weight")
        o = self._layer_tensor(layer_idx, "self_attn.o_proj.weight")
        attn_norm = self._layer_tensor(layer_idx, "input_layernorm.weight")

        if any(t is None for t in [q, k, v, o, attn_norm]):
            log.debug("Layer %d: missing required GQA tensors", layer_idx)
            return None

        # FP8 dequant for Q/K/V/O projections (MiniMax-M2.x stores all
        # weights in FP8 with per-tensor scale factors)
        q = self._dequant_minimax(q, self._layer_tensor(
            layer_idx, "self_attn.q_proj.weight_scale_inv"))
        k = self._dequant_minimax(k, self._layer_tensor(
            layer_idx, "self_attn.k_proj.weight_scale_inv"))
        v = self._dequant_minimax(v, self._layer_tensor(
            layer_idx, "self_attn.v_proj.weight_scale_inv"))
        o = self._dequant_minimax(o, self._layer_tensor(
            layer_idx, "self_attn.o_proj.weight_scale_inv"))
        # RMS norm weights stay in float16 (no scale factor)

        # Optional tensors (may not exist in all GQA models)
        pre_attn_norm = self._layer_tensor(layer_idx, "pre_attention_layernorm.weight")
        post_attn_norm = self._layer_tensor(layer_idx, "post_attention_layernorm.weight")
        qk_norm = self._layer_tensor(layer_idx, "qk_norm.weight")

        return GQAWeights(
            layer_idx=layer_idx,
            q_proj=q.astype(np.float16),
            k_proj=k.astype(np.float16),
            v_proj=v.astype(np.float16),
            o_proj=o.astype(np.float16),
            attn_norm=attn_norm.astype(np.float16),
            pre_attn_norm=pre_attn_norm.astype(np.float16) if pre_attn_norm is not None else None,
            post_attn_norm=post_attn_norm.astype(np.float16) if post_attn_norm is not None else None,
            qk_norm=qk_norm.astype(np.float16) if qk_norm is not None else None,
        )

    # -- Expert loading ----------------------------------------------------

    def load_experts(self, engine, expert_ids: List[int]) -> int:
        """Load MoE expert weights for all layers in [layer_start, layer_end).

        Automatically detects between standard ``mlp.experts`` naming and
        MiniMax-style ``block_sparse_moe.experts`` naming.  The first tensor
        fetch attempt per layer determines which prefix is used for the
        remainder of that layer.

        For models with ``num_shared_experts == 0`` (e.g. MiniMax-M2.5),
        shared expert IDs (0, 1) are treated as routed experts.

        Returns the count of (layer, expert) pairs successfully loaded.
        """
        loaded = 0
        shared_count = engine._dmap.num_shared_experts

        for layer_idx in range(self.layer_start, self.layer_end):
            for eid in expert_ids:
                ew = self._load_one_expert(layer_idx, eid)
                if ew is not None:
                    if shared_count > 0 and eid < shared_count:
                        engine.load_shared_experts([ew])
                    else:
                        engine.load_expert(ew)
                    loaded += 1
        log.info("Loaded %d expert weight sets", loaded)
        return loaded

    def _load_one_expert(self, layer_idx: int, expert_id: int) -> Optional[object]:
        """Load one MoE expert's weights from safetensors.

        Probes two naming conventions in order:
          1. Standard:  ``mlp.experts.{id}.{gate_proj,up_proj,down_proj}.weight``
          2. MiniMax:   ``block_sparse_moe.experts.{id}.{w1,w2,w3}.weight``
             (with optional ``.weight_scale_inv`` suffix for FP8 dequant).

        Returns ``ExpertWeights`` on success, or None if any required tensor
        is missing.
        """
        from .shared_expert_cache import ExpertWeights

        # ---- Probe strategy ----
        # First try MiniMax naming (block_sparse_moe.*.w1.weight).  If
        # that tensor is present for this layer, use it.  Otherwise fall
        # back to standard naming.
        _minimax_probe = self._layer_tensor(
            layer_idx, f"block_sparse_moe.experts.{expert_id}.w1.weight"
        )
        if _minimax_probe is not None:
            return self._load_one_expert_minimax(layer_idx, expert_id)

        # Standard naming
        gate = self._layer_tensor(
            layer_idx, f"mlp.experts.{expert_id}.gate_proj.weight"
        )
        up = self._layer_tensor(
            layer_idx, f"mlp.experts.{expert_id}.up_proj.weight"
        )
        down = self._layer_tensor(
            layer_idx, f"mlp.experts.{expert_id}.down_proj.weight"
        )

        if any(t is None for t in [gate, up, down]):
            return None

        return ExpertWeights(
            expert_id=expert_id,
            gate_proj=gate.astype(np.float16),
            up_proj=up.astype(np.float16),
            down_proj=down.astype(np.float16),
        )

    def _load_one_expert_minimax(
        self, layer_idx: int, expert_id: int
    ) -> Optional[object]:
        """Load one MoE expert from MiniMax-M2.x ``block_sparse_moe`` format.

        Tensor mapping::

            block_sparse_moe.experts.{j}.w1.weight → gate_proj
            block_sparse_moe.experts.{j}.w2.weight → up_proj
            block_sparse_moe.experts.{j}.w3.weight → down_proj

        FP8 dequant is applied when a companion ``weight_scale_inv`` tensor
        is present for each weight matrix.
        """
        from .shared_expert_cache import ExpertWeights

        gate = self._layer_tensor(
            layer_idx, f"block_sparse_moe.experts.{expert_id}.w1.weight"
        )
        up = self._layer_tensor(
            layer_idx, f"block_sparse_moe.experts.{expert_id}.w2.weight"
        )
        down = self._layer_tensor(
            layer_idx, f"block_sparse_moe.experts.{expert_id}.w3.weight"
        )

        if any(t is None for t in [gate, up, down]):
            return None

        # FP8 dequant: if weight_scale_inv exists, multiply through
        gate = self._dequant_minimax(gate, self._layer_tensor(
            layer_idx, f"block_sparse_moe.experts.{expert_id}.w1.weight_scale_inv"
        ))
        up = self._dequant_minimax(up, self._layer_tensor(
            layer_idx, f"block_sparse_moe.experts.{expert_id}.w2.weight_scale_inv"
        ))
        down = self._dequant_minimax(down, self._layer_tensor(
            layer_idx, f"block_sparse_moe.experts.{expert_id}.w3.weight_scale_inv"
        ))

        return ExpertWeights(
            expert_id=expert_id,
            gate_proj=gate.astype(np.float16),
            up_proj=up.astype(np.float16),
            down_proj=down.astype(np.float16),
        )

    @staticmethod
    def _dequant_minimax(weight: np.ndarray, scale_inv: Optional[np.ndarray]) -> np.ndarray:
        """Apply per-tensor or block-wise FP8 dequantisation.

        MiniMax-M2.x stores all projection weights in FP8 (``F8_E4M3``)
        with companion ``weight_scale_inv`` tensors.  Two modes exist:

        * **Per-tensor**: ``scale_inv`` is a scalar or 1-d tensor —
          ``weight * scale_inv`` element-wise.
        * **Block-wise** (MiniMax-M2.5): ``scale_inv`` is a 2-d grid of
          float32 scales, one per (128×128) block.  Each block is
          multiplied by its corresponding scale value.

        If *scale_inv* is None (no quantization factor present), returns
        *weight* cast to float16 unchanged.
        """
        if scale_inv is None:
            return weight.astype(np.float16)

        w32 = weight.astype(np.float32)
        s32 = scale_inv.astype(np.float32)

        # Per-tensor: element-wise multiply
        if s32.ndim <= 1:
            return (w32 * s32).astype(np.float16)

        # Block-wise: 2-d scale grid (r_blocks, c_blocks) for block size B×B
        r, c = w32.shape
        scale_r, scale_c = s32.shape
        block_r = r // scale_r
        block_c = c // scale_c

        # Reshape into blocks:
        #   (scale_r, block_r, scale_c, block_c)
        #   -> transpose -> (scale_r, scale_c, block_r, block_c)
        #   -> multiply scales -> transpose back -> reshape to (r, c)
        w_blocks = w32.reshape(scale_r, block_r, scale_c, block_c)
        w_blocks = w_blocks.transpose(0, 2, 1, 3)
        w_blocks = w_blocks * s32[:, :, np.newaxis, np.newaxis]
        w_blocks = w_blocks.transpose(0, 2, 1, 3).reshape(r, c)

        return w_blocks.astype(np.float16)

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


# ------------------------------------------------------------------ #
# SafetensorsMmapReader - mmap-backed tensor access                   #
# ------------------------------------------------------------------ #

class SafetensorsMmapReader:
    """Low-level mmap reader for safetensors shard files.

    Opens shard files with memory-mapped I/O and constructs numpy arrays
    that reference the mmap buffer directly.  Tensor data is never copied
    into a Python buffer unless explicitly requested.  Uses an LRU cache
    to bound the number of concurrently open file handles.

    Parameters
    ----------
    max_open_shards:  Maximum number of file handles to keep open (LRU).
    """

    def __init__(self, max_open_shards: int = 8) -> None:
        self._max_open = max_open_shards
        self._handles: OrderedDict[str, Tuple[mmap.mmap, Dict]] = OrderedDict()
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tensor(
        self, model_dir: str, shard_name: str, tensor_name: str
    ) -> np.ndarray:
        """Return a numpy view of *tensor_name* backed by mmap.

        The returned array shares memory with the mmap'd file; do not
        mutate it unless you intend to alter the on-disk shard (generally
        undesirable).
        """
        with self._lock:
            handle, header = self._get_or_open(model_dir, shard_name)
        return self._build_array(handle, header, tensor_name)

    def close(self) -> None:
        """Close all open mmap handles.

        On Windows, mmap.close() may raise BufferError if numpy buffer
        views created via :meth:`get_tensor` still reference the mmap
        data.  Callers should release all returned arrays before calling
        this method.  If views remain, the handle is left to be cleaned
        up by the interpreter at process exit.
        """
        with self._lock:
            for key in list(self._handles.keys()):
                try:
                    h, _ = self._handles[key]
                    h.close()
                except BufferError:
                    # Outstanding numpy buffer views — caller still
                    # holds a reference.  The handle will be garbage-
                    # collected when the last view is released.
                    pass
            self._handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_open(
        self, model_dir: str, shard_name: str
    ) -> Tuple[mmap.mmap, Dict]:
        key = f"{model_dir}/{shard_name}"
        if key in self._handles:
            self._handles.move_to_end(key)
            return self._handles[key]

        path = pathlib.Path(model_dir) / shard_name
        if not path.is_file():
            raise FileNotFoundError(f"Shard not found: {path}")

        # Parse header
        header = _parse_safetensors_header(path)
        # Calculate data offset: 8 bytes header_len + header bytes
        header_bytes = json.dumps(header).encode("utf-8")
        data_offset = 8 + len(header_bytes)

        # Open + mmap
        fd = os.open(str(path), os.O_RDONLY)
        try:
            file_size = os.fstat(fd).st_size
            mapping = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)  # mmap holds reference, safe to close fd

        self._handles[key] = (mapping, header)
        if len(self._handles) > self._max_open:
            # Evict LRU (oldest)
            evict_key, (evict_mmap, _) = self._handles.popitem(last=False)
            evict_mmap.close()

        return self._handles[key]

    def _build_array(
        self, mapping: mmap.mmap, header: Dict, tensor_name: str
    ) -> np.ndarray:
        meta = header[tensor_name]
        dtype_str = meta["dtype"]
        shape = meta["shape"]
        start, end = meta["data_offsets"]
        np_dtype = _DTYPE_MAP.get(dtype_str, np.float16)

        # Zero-copy: create numpy array from the mmap buffer slice
        arr = np.frombuffer(mapping, dtype=np_dtype, count=int(np.prod(shape)),
                            offset=start)
        return arr.reshape(shape)


# ------------------------------------------------------------------ #
# MmapWeightStore - high-level mmap weight store                       #
# ------------------------------------------------------------------ #

class MmapWeightStore:
    """Memory-mapped weight store for large model checkpoints.

    Maps safetensors shard files into virtual memory so that large
    checkpoints (580 GB+ DeepSeek-V4 or 126 GB MiniMax-M2.5) can reside
    on NVMe without being fully loaded into RAM.  Maintains an LRU cache
    of open shard handles via an underlying ``SafetensorsMmapReader``.

    Parameters
    ----------
    model_dir:   Path to the safetensors checkpoint directory.
    max_open_shards:  Max file handles; defaults to 8.
    index:       Optional pre-loaded ModelIndex; created from model_dir
                 if None.
    """

    def __init__(
        self,
        model_dir: str | pathlib.Path,
        max_open_shards: int = 8,
        index: Optional[ModelIndex] = None,
    ) -> None:
        self._dir = pathlib.Path(model_dir)
        self._index = index or ModelIndex(self._dir)
        self._reader = SafetensorsMmapReader(max_open_shards=max_open_shards)

    @property
    def num_shards(self) -> int:
        return len(self._index._shards)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tensor(self, tensor_name: str) -> np.ndarray:
        """Return a mmap-backed numpy view of *tensor_name*.

        Uses the LRU-cached mmap reader so frequently-accessed shards
        stay open.  Raises ``KeyError`` if *tensor_name* is not found
        in the index.
        """
        shard = self._index.shard_for_tensor(tensor_name)
        if shard is None:
            raise KeyError(
                f"Tensor {tensor_name!r} not found in shard index "
                f"for {self._dir}"
            )
        return self._reader.get_tensor(str(self._dir), shard, tensor_name)

    def get_tensor_read_once(
        self, shard_path: pathlib.Path, tensor_name: str
    ) -> np.ndarray:
        """Read one tensor from a shard without LRU caching.

        Opens the shard, reads the tensor, closes the shard.  Suitable
        for one-shot access where you do not want to pollute the cache.
        """
        header = _parse_safetensors_header(shard_path)
        header_bytes = json.dumps(header).encode("utf-8")
        data_offset = 8 + len(header_bytes)

        fd = os.open(str(shard_path), os.O_RDONLY)
        try:
            file_size = os.fstat(fd).st_size
            mapping = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)

        try:
            meta = header[tensor_name]
            dtype_str = meta["dtype"]
            shape = meta["shape"]
            start, end = meta["data_offsets"]
            np_dtype = _DTYPE_MAP.get(dtype_str, np.float16)
            arr = np.frombuffer(mapping, dtype=np_dtype,
                                count=int(np.prod(shape)), offset=start)
            arr = arr.reshape(shape).copy()
            return arr
        finally:
            mapping.close()

    def close(self) -> None:
        """Release all mmap handles."""
        self._reader.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass