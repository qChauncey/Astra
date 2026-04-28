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

"""Tests for astra.inference.weight_loader — shard loading and integrity checks."""

from __future__ import annotations

import json
import pathlib
import struct

import numpy as np
import pytest

from astra.inference.weight_loader import WeightLoader, ModelIndex


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_safetensors(path: pathlib.Path, tensors: dict) -> None:
    """Write a minimal safetensors file with numpy arrays."""
    # Safetensors format: 8-byte header length (LE uint64) + JSON header + data
    metadata = {}
    data_parts = []
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        dtype_map = {np.float16: "F16", np.float32: "F32", np.int32: "I32"}
        dtype_str = dtype_map.get(arr.dtype.type, "F32")
        nbytes = arr.nbytes
        metadata[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        data_parts.append(arr.tobytes())
        offset += nbytes

    header_json = json.dumps(metadata).encode("utf-8")
    # Pad to 8-byte alignment
    pad = (8 - len(header_json) % 8) % 8
    header_json += b" " * pad
    header_len = struct.pack("<Q", len(header_json))

    with open(path, "wb") as f:
        f.write(header_len)
        f.write(header_json)
        for part in data_parts:
            f.write(part)


def _make_model_dir(tmp_path: pathlib.Path, layers: range, hidden: int = 16) -> pathlib.Path:
    """Create a minimal model directory with one safetensors shard."""
    tensors = {}
    for i in layers:
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "input_layernorm.weight",
        ]:
            key = f"model.layers.{i}.{suffix}"
            if suffix.endswith("layernorm.weight"):
                tensors[key] = np.ones(hidden, dtype=np.float16)
            else:
                tensors[key] = np.random.randn(hidden, hidden).astype(np.float16)
    _write_safetensors(tmp_path / "model.safetensors", tensors)
    return tmp_path


# ── ModelIndex ────────────────────────────────────────────────────────────────

class TestModelIndex:
    def test_single_shard_no_index(self, tmp_path):
        """Single model.safetensors with no index file — ModelIndex falls back."""
        _write_safetensors(tmp_path / "model.safetensors", {"key": np.zeros(4, np.float16)})
        idx = ModelIndex(tmp_path)
        assert "model.safetensors" in idx._shards

    def test_index_json_parsed(self, tmp_path):
        """model.safetensors.index.json is parsed into tensor→shard mapping."""
        index_data = {
            "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
                "model.layers.1.self_attn.q_proj.weight": "model-00002.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index_data))
        idx = ModelIndex(tmp_path)
        assert idx.shard_for_tensor("model.layers.0.self_attn.q_proj.weight") == "model-00001.safetensors"
        assert "model-00001.safetensors" in idx._shards
        assert "model-00002.safetensors" in idx._shards

    def test_shards_for_layers(self, tmp_path):
        """shards_for_layers returns shards containing tensors for the given range."""
        index_data = {
            "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "shard-0.safetensors",
                "model.layers.5.self_attn.q_proj.weight": "shard-1.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index_data))
        idx = ModelIndex(tmp_path)
        shards = idx.shards_for_layers(0, 1)
        assert "shard-0.safetensors" in shards
        assert "shard-1.safetensors" not in shards

    def test_empty_dir_no_crash(self, tmp_path):
        """ModelIndex on an empty directory does not raise."""
        idx = ModelIndex(tmp_path)
        assert idx._shards == set()


# ── WeightLoader (no-manifest mode) ──────────────────────────────────────────

class TestWeightLoaderNoManifest:
    def test_init_no_manifest_warns(self, tmp_path, caplog):
        """WeightLoader without manifest logs a warning."""
        import logging
        _make_model_dir(tmp_path, range(2))
        with caplog.at_level(logging.WARNING, logger="astra.weight_loader"):
            loader = WeightLoader(tmp_path, layer_start=0, layer_end=2, verify_integrity=True)
        assert "integrity" in caplog.text.lower() or "manifest" in caplog.text.lower()
        assert loader._manifest is None

    def test_init_integrity_disabled(self, tmp_path):
        """WeightLoader with verify_integrity=False never loads manifest."""
        _make_model_dir(tmp_path, range(2))
        loader = WeightLoader(tmp_path, layer_start=0, layer_end=2, verify_integrity=False)
        assert loader._manifest is None

    def test_load_into_returns_layer_count(self, tmp_path):
        """load_into returns number of successfully loaded layers."""
        pytest.importorskip("safetensors")
        _make_model_dir(tmp_path, range(3), hidden=16)
        loader = WeightLoader(tmp_path, layer_start=0, layer_end=3, verify_integrity=False)
        from astra.inference.heterogeneous import HeterogeneousEngine, DeviceMap
        engine = HeterogeneousEngine(
            device_map=DeviceMap(attention_on_gpu=False, moe_on_cpu=True, _hidden_dim_override=16),
        )
        n = loader.load_into(engine)
        assert n == 3

    def test_load_experts_returns_count(self, tmp_path):
        """load_experts returns number of experts loaded (0 if no expert tensors)."""
        pytest.importorskip("safetensors")
        _make_model_dir(tmp_path, range(2))
        loader = WeightLoader(tmp_path, layer_start=0, layer_end=2, verify_integrity=False)
        from astra.inference.heterogeneous import HeterogeneousEngine, DeviceMap
        engine = HeterogeneousEngine(
            device_map=DeviceMap(attention_on_gpu=False, moe_on_cpu=True, _hidden_dim_override=16),
        )
        # No expert tensors in our fixture — returns 0, no crash
        n = loader.load_experts(engine, [0, 1])
        assert n == 0

    def test_missing_shard_raises(self, tmp_path):
        """_get_shard raises FileNotFoundError for non-existent shard."""
        loader = WeightLoader(tmp_path, layer_start=0, layer_end=1, verify_integrity=False)
        with pytest.raises(FileNotFoundError):
            loader._get_shard("nonexistent.safetensors")


# ── WeightLoader + manifest integrity ────────────────────────────────────────

class TestWeightLoaderIntegrity:
    def _make_with_manifest(self, tmp_path):
        """Build a model dir + valid manifest."""
        _make_model_dir(tmp_path, range(2), hidden=8)
        from astra.inference.weight_manifest import WeightManifest
        manifest = WeightManifest.create_from_dir(tmp_path)
        manifest.save(tmp_path / "astra_manifest.json")
        return tmp_path

    def test_valid_manifest_accepted(self, tmp_path):
        """WeightLoader with a valid manifest loads without error."""
        self._make_with_manifest(tmp_path)
        loader = WeightLoader(tmp_path, layer_start=0, layer_end=2, verify_integrity=True)
        assert loader._manifest is not None

    def test_tampered_shard_raises_runtime_error(self, tmp_path):
        """_verify_shard raises RuntimeError when shard content is tampered."""
        self._make_with_manifest(tmp_path)
        shard_path = tmp_path / "model.safetensors"
        # Corrupt the file
        data = shard_path.read_bytes()
        shard_path.write_bytes(data[:-16] + bytes(16))

        loader = WeightLoader(tmp_path, layer_start=0, layer_end=2, verify_integrity=True)
        with pytest.raises(RuntimeError, match="hash mismatch|tamper"):
            loader._verify_shard("model.safetensors", shard_path)

    def test_unknown_shard_raises_runtime_error(self, tmp_path):
        """_verify_shard raises RuntimeError for a shard not in the manifest."""
        self._make_with_manifest(tmp_path)
        extra = tmp_path / "extra.safetensors"
        extra.write_bytes(b"junk")
        loader = WeightLoader(tmp_path, layer_start=0, layer_end=2, verify_integrity=True)
        with pytest.raises(RuntimeError, match="not in the manifest"):
            loader._verify_shard("extra.safetensors", extra)

    def test_verified_shards_cached(self, tmp_path):
        """A shard verified once is not re-hashed on subsequent access."""
        self._make_with_manifest(tmp_path)
        loader = WeightLoader(tmp_path, layer_start=0, layer_end=2, verify_integrity=True)
        shard_path = tmp_path / "model.safetensors"
        loader._verify_shard("model.safetensors", shard_path)
        assert "model.safetensors" in loader._verified_shards
        # Second call should not raise (cached)
        loader._verify_shard("model.safetensors", shard_path)
