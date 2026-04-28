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
Unit tests for MLA weight mapping, mmap weight store, and WeightLoader.

These tests exercise the tensor-name discovery logic without requiring
real safetensors files on disk.  Each test creates a minimal in-memory
mock shard directory via tmp_path.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from astra.inference.heterogeneous import HeterogeneousEngine, MLAWeights, DeviceMap
from astra.inference.weight_loader import (
    MLA_TENSOR_MAP,
    MmapWeightStore,
    SafetensorsMmapReader,
    WeightLoader,
    _MLA_ATTN_TENSORS,
    _LEGACY_ATTN_SUFFIXES,
    detect_mla_format,
)


# ────────────────────────────────────────────────────────────────── #
# Helpers: create minimal safetensors files in tmp_path               #
# ────────────────────────────────────────────────────────────────── #

DTYPE_MAP = {
    np.float32: "F32",
    np.float16: "F16",
}


def _make_safetensors_metadata(tensors: Dict[str, np.ndarray]) -> dict:
    """Build a safetensors-format metadata header."""
    offset = 0
    tensor_meta = {}
    for name, arr in tensors.items():
        dtype_str = DTYPE_MAP.get(arr.dtype.type, "F32")
        size = arr.nbytes
        tensor_meta[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    return tensor_meta


def _write_safetensors(path: Path, tensors: Dict[str, np.ndarray]) -> None:
    """Write a valid minimal safetensors file to disk."""
    header = _make_safetensors_metadata(tensors)
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_bytes))

    with open(path, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        for arr in tensors.values():
            f.write(arr.tobytes())


# ────────────────────────────────────────────────────────────────── #
# Fixtures                                                            #
# ────────────────────────────────────────────────────────────────── #

HDIM = 64   # small hidden dim for test speed
SEQ = 4
N_HEADS = 4
HEAD_DIM = 16


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    """Create a directory that will hold safetensors mock files."""
    d = tmp_path / "deepseek-v4-mock"
    d.mkdir()
    return d


@pytest.fixture
def mla_tensors() -> Dict[str, np.ndarray]:
    """Mock MLA tensors for one layer (layer 0)."""
    rng = np.random.default_rng(42)
    q_lora_rank = 32
    kv_lora_rank = 16
    qk_rope_head_dim = 8
    n_heads = N_HEADS
    hd = HDIM

    return {
        "model.layers.0.self_attn.q_a_proj.weight": rng.normal(0, 0.02, (q_lora_rank, hd)).astype(np.float16),
        "model.layers.0.self_attn.q_b_proj.weight": rng.normal(0, 0.02, (n_heads * HEAD_DIM, q_lora_rank)).astype(np.float16),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": rng.normal(0, 0.02, (kv_lora_rank + qk_rope_head_dim, hd)).astype(np.float16),
        "model.layers.0.self_attn.kv_b_proj.weight": rng.normal(0, 0.02, (n_heads * (HEAD_DIM // 2 + HEAD_DIM), kv_lora_rank)).astype(np.float16),
        "model.layers.0.self_attn.o_proj.weight": rng.normal(0, 0.02, (hd, n_heads * HEAD_DIM)).astype(np.float16),
        "model.layers.0.self_attn.q_a_layernorm.weight": rng.normal(0, 0.02, (q_lora_rank,)).astype(np.float16),
        "model.layers.0.self_attn.kv_a_layernorm.weight": rng.normal(0, 0.02, (kv_lora_rank,)).astype(np.float16),
        "model.layers.0.input_layernorm.weight": rng.normal(0, 0.02, (hd,)).astype(np.float16),
    }


@pytest.fixture
def mla_checkpoint(tmp_model_dir: Path, mla_tensors: Dict[str, np.ndarray]) -> Path:
    """Write a single-shard safetensors checkpoint with MLA tensors."""
    tensors = dict(mla_tensors)
    _write_safetensors(tmp_model_dir / "model-00001-of-00001.safetensors", tensors)

    # Write index
    index = {
        "metadata": {"total_size": sum(arr.nbytes for arr in tensors.values())},
        "weight_map": {
            name: "model-00001-of-00001.safetensors"
            for name in tensors
        },
    }
    with open(tmp_model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    return tmp_model_dir


@pytest.fixture
def legacy_mha_tensors() -> Dict[str, np.ndarray]:
    """Mock standard MHA tensors for one layer (layer 0)."""
    rng = np.random.default_rng(99)
    hd = HDIM
    return {
        "model.layers.0.self_attn.q_proj.weight": rng.normal(0, 0.02, (hd, hd)).astype(np.float16),
        "model.layers.0.self_attn.k_proj.weight": rng.normal(0, 0.02, (hd, hd)).astype(np.float16),
        "model.layers.0.self_attn.v_proj.weight": rng.normal(0, 0.02, (hd, hd)).astype(np.float16),
        "model.layers.0.self_attn.o_proj.weight": rng.normal(0, 0.02, (hd, hd)).astype(np.float16),
        "model.layers.0.input_layernorm.weight": rng.normal(0, 0.02, (hd,)).astype(np.float16),
    }


@pytest.fixture
def legacy_checkpoint(tmp_model_dir: Path, legacy_mha_tensors: Dict[str, np.ndarray]) -> Path:
    """Write a single-shard safetensors checkpoint with legacy MHA tensors."""
    tensors = dict(legacy_mha_tensors)
    _write_safetensors(tmp_model_dir / "model.safetensors", tensors)
    # No index file → single-shard fallback

    return tmp_model_dir


# ────────────────────────────────────────────────────────────────── #
# detect_mla_format                                                   #
# ────────────────────────────────────────────────────────────────── #

class TestDetectMLAFormat:
    def test_detects_mla_format(self, mla_checkpoint: Path):
        assert detect_mla_format(mla_checkpoint) is True

    def test_detects_legacy_format(self, legacy_checkpoint: Path):
        assert detect_mla_format(legacy_checkpoint) is False

    def test_returns_false_for_empty_dir(self, tmp_path: Path):
        d = tmp_path / "empty"
        d.mkdir()
        assert detect_mla_format(d) is False


# ────────────────────────────────────────────────────────────────── #
# WeightLoader — MLA attention                                          #
# ────────────────────────────────────────────────────────────────── #

class TestWeightLoaderMLA:
    def test_loads_mla_weights_into_engine(self, mla_checkpoint: Path):
        engine = HeterogeneousEngine(DeviceMap(model_id="test-mla-load", attention_on_gpu=False, moe_on_cpu=True))
        loader = WeightLoader(mla_checkpoint, layer_start=0, layer_end=1, verify_integrity=False)
        loader.load_into(engine)

        assert 0 in engine._mla_weights
        mw = engine._mla_weights[0]
        assert isinstance(mw, MLAWeights)
        assert mw.q_a_proj.shape[1] == HDIM   # hidden_dim on axis-1
        assert mw.kv_a_proj.shape[1] == HDIM

    def test_mla_layer_count(self, mla_checkpoint: Path):
        engine = HeterogeneousEngine(DeviceMap(model_id="test-mla-count", attention_on_gpu=False, moe_on_cpu=True))
        loader = WeightLoader(mla_checkpoint, layer_start=0, layer_end=1, verify_integrity=False)
        loaded = loader.load_into(engine)
        assert loaded == 1


# ────────────────────────────────────────────────────────────────── #
# WeightLoader — legacy MHA fallback                                    #
# ────────────────────────────────────────────────────────────────── #

class TestWeightLoaderLegacy:
    def test_legacy_loads_q_k_v_o_norm(self, legacy_checkpoint: Path):
        engine = HeterogeneousEngine(DeviceMap(model_id="test-legacy", attention_on_gpu=False, moe_on_cpu=True))
        loader = WeightLoader(legacy_checkpoint, layer_start=0, layer_end=1, verify_integrity=False)
        loaded = loader.load_into(engine)
        assert loaded == 1
        assert 0 in engine._attn_q_proj
        assert 0 in engine._attn_k_proj


# ────────────────────────────────────────────────────────────────── #
# MLA_TENSOR_MAP                                                       #
# ────────────────────────────────────────────────────────────────── #

class TestMLATensorMap:
    def test_map_has_8_keys(self):
        assert len(_MLA_ATTN_TENSORS) == 8

    def test_legacy_map_has_8_keys(self):
        assert len(_LEGACY_ATTN_SUFFIXES) == 8

    def test_mapping_includes_q_layernorm(self):
        assert _MLA_ATTN_TENSORS["self_attn.q_a_layernorm.weight"] == "q_norm"

    def test_mapping_includes_kv_layernorm(self):
        assert _MLA_ATTN_TENSORS["self_attn.kv_a_layernorm.weight"] == "kv_norm"

    def test_mla_tensor_map_public(self):
        assert isinstance(MLA_TENSOR_MAP, dict)
        # Each entry maps suffix → dataclass field name
        assert MLA_TENSOR_MAP["self_attn.q_a_proj.weight"] == "q_a_proj"
        assert MLA_TENSOR_MAP["self_attn.kv_a_proj_with_mqa.weight"] == "kv_a_proj"


# ────────────────────────────────────────────────────────────────── #
# MLAWeights dataclass                                                  #
# ────────────────────────────────────────────────────────────────── #

class TestMLAWeights:
    def test_construction_from_arrays(self, mla_tensors: Dict[str, np.ndarray]):
        mw = MLAWeights(
            layer_idx=0,
            q_a_proj=mla_tensors["model.layers.0.self_attn.q_a_proj.weight"],
            q_b_proj=mla_tensors["model.layers.0.self_attn.q_b_proj.weight"],
            kv_a_proj=mla_tensors["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"],
            kv_b_proj=mla_tensors["model.layers.0.self_attn.kv_b_proj.weight"],
            o_proj=mla_tensors["model.layers.0.self_attn.o_proj.weight"],
            q_norm=mla_tensors["model.layers.0.self_attn.q_a_layernorm.weight"],
            kv_norm=mla_tensors["model.layers.0.self_attn.kv_a_layernorm.weight"],
            attn_norm=mla_tensors["model.layers.0.input_layernorm.weight"],
        )
        assert mw.q_lora_rank == mw.q_a_proj.shape[0]
        assert mw.kv_lora_rank == mw.kv_norm.shape[0]
        assert mw.layer_idx == 0

    def test_derived_dimensions(self, mla_tensors: Dict[str, np.ndarray]):
        mw = MLAWeights(
            layer_idx=0,
            q_a_proj=mla_tensors["model.layers.0.self_attn.q_a_proj.weight"],
            q_b_proj=mla_tensors["model.layers.0.self_attn.q_b_proj.weight"],
            kv_a_proj=mla_tensors["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"],
            kv_b_proj=mla_tensors["model.layers.0.self_attn.kv_b_proj.weight"],
            o_proj=mla_tensors["model.layers.0.self_attn.o_proj.weight"],
            q_norm=mla_tensors["model.layers.0.self_attn.q_a_layernorm.weight"],
            kv_norm=mla_tensors["model.layers.0.self_attn.kv_a_layernorm.weight"],
            attn_norm=mla_tensors["model.layers.0.input_layernorm.weight"],
        )
        assert mw.q_lora_rank == 32
        assert mw.kv_lora_rank == 16


# ────────────────────────────────────────────────────────────────── #
# HeterogeneousEngine — MLA forward path                                #
# ────────────────────────────────────────────────────────────────── #

class TestHeterogeneousEngineMLAForward:
    @pytest.fixture
    def engine_with_mla(self, mla_tensors: Dict[str, np.ndarray]) -> HeterogeneousEngine:
        engine = HeterogeneousEngine(DeviceMap(
            model_id="test-mla-forward",
            attention_on_gpu=False,
            moe_on_cpu=True,
        ))
        mw = MLAWeights(
            layer_idx=0,
            q_a_proj=mla_tensors["model.layers.0.self_attn.q_a_proj.weight"],
            q_b_proj=mla_tensors["model.layers.0.self_attn.q_b_proj.weight"],
            kv_a_proj=mla_tensors["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"],
            kv_b_proj=mla_tensors["model.layers.0.self_attn.kv_b_proj.weight"],
            o_proj=mla_tensors["model.layers.0.self_attn.o_proj.weight"],
            q_norm=mla_tensors["model.layers.0.self_attn.q_a_layernorm.weight"],
            kv_norm=mla_tensors["model.layers.0.self_attn.kv_a_layernorm.weight"],
            attn_norm=mla_tensors["model.layers.0.input_layernorm.weight"],
        )
        engine.load_mla_weights([mw])
        return engine

    def test_mla_forward_preserves_shape(self, engine_with_mla: HeterogeneousEngine):
        from astra.serialization.tensor_pack import TensorPacket

        hidden = np.random.default_rng(1).standard_normal((SEQ, HDIM)).astype(np.float16)
        packet = TensorPacket(
            packet_id="mla-test",
            tensor=hidden,
            layer_start=0,
            layer_end=1,
            token_ids=list(range(SEQ)),
            geo_region="local",
            src_node="test",
            dst_node="test",
            metadata={},
        )
        out = engine_with_mla.forward(packet, layer_indices=[0])
        assert out.tensor.shape == (SEQ, HDIM)
        assert out.tensor.dtype == np.float16

    def test_mla_forward_no_nan(self, engine_with_mla: HeterogeneousEngine):
        from astra.serialization.tensor_pack import TensorPacket

        hidden = np.random.default_rng(2).standard_normal((SEQ, HDIM)).astype(np.float16)
        packet = TensorPacket(
            packet_id="mla-test-2",
            tensor=hidden,
            layer_start=0,
            layer_end=1,
            token_ids=list(range(SEQ)),
            geo_region="local",
            src_node="test",
            dst_node="test",
            metadata={},
        )
        out = engine_with_mla.forward(packet, layer_indices=[0])
        assert not np.any(np.isnan(out.tensor))
        assert not np.any(np.isinf(out.tensor))


# ────────────────────────────────────────────────────────────────── #
# MmapWeightStore                                                       #
# ────────────────────────────────────────────────────────────────── #

class TestMmapWeightStore:
    def test_creation(self, mla_checkpoint: Path):
        store = MmapWeightStore(mla_checkpoint)
        assert store.num_shards > 0

    def test_get_tensor_returns_numpy_array(self, mla_checkpoint: Path):
        store = MmapWeightStore(mla_checkpoint)
        t = store.get_tensor("model.layers.0.self_attn.q_a_proj.weight")
        assert isinstance(t, np.ndarray)
        assert t.shape[1] == HDIM

    def test_get_tensor_raises_for_missing(self, mla_checkpoint: Path):
        store = MmapWeightStore(mla_checkpoint)
        with pytest.raises(KeyError):
            store.get_tensor("nonexistent.tensor.weight")

    def test_lru_eviction_does_not_crash(self, mla_checkpoint: Path):
        """Getting many different tensors exercises the LRU eviction path."""
        store = MmapWeightStore(mla_checkpoint, max_open_shards=1)
        names = list(MLA_TENSOR_MAP.keys())
        for i in range(3):
            for name in names:
                t = store.get_tensor(f"model.layers.0.{name}")
                assert t is not None

    def test_get_tensor_read_once_no_cache(self, mla_checkpoint: Path):
        """Verify get_tensor_read_once does not go through the LRU cache."""
        store = MmapWeightStore(mla_checkpoint)
        t = store.get_tensor_read_once(
            mla_checkpoint / "model-00001-of-00001.safetensors",
            "model.layers.0.self_attn.q_a_proj.weight"
        )
        assert isinstance(t, np.ndarray)

    def test_close_releases_handles(self, mla_checkpoint: Path):
        store = MmapWeightStore(mla_checkpoint, max_open_shards=2)
        store.get_tensor("model.layers.0.self_attn.q_a_proj.weight")
        store.close()  # should not raise


# ────────────────────────────────────────────────────────────────── #
# SafetensorsMmapReader                                                  #
# ────────────────────────────────────────────────────────────────── #

class TestSafetensorsMmapReader:
    def test_open_and_read(self, mla_checkpoint: Path):
        reader = SafetensorsMmapReader(max_open_shards=2)
        t = reader.get_tensor(
            str(mla_checkpoint),
            "model-00001-of-00001.safetensors",
            "model.layers.0.self_attn.q_a_proj.weight",
        )
        assert isinstance(t, np.ndarray)
        reader.close()

    def test_close(self, mla_checkpoint: Path):
        reader = SafetensorsMmapReader(max_open_shards=2)
        reader.close()  # should not raise if no handles open