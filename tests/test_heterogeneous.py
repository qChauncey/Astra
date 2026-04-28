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

"""Unit tests for HeterogeneousEngine and KTransformersStub."""

import numpy as np
import pytest

from astra.inference.heterogeneous import (
    DeviceMap,
    HeterogeneousEngine,
    KTransformersStub,
    LayerKVCache,
    GQAWeights,
)
from astra.inference.shared_expert_cache import ExpertWeights
from astra.serialization.tensor_pack import TensorPacket
from astra.config.model_config import register_model_config, ModelConfig, AttentionType


# ────────────────────────────────────────────────────────────────── #
# Register a small test model config                                     #
# ────────────────────────────────────────────────────────────────── #

from astra.config.model_config import QuantizationType

register_model_config(
    ModelConfig(
        model_id="test-small",
        display_name="Test Small",
        arch_type="TestForCausalLM",
        model_type="test",
        hidden_dim=64,
        num_layers=4,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256,
        vocab_size=1024,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rotary_dim=16,
        rms_norm_eps=1e-6,
        attention_type=AttentionType.GQA,
        num_local_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=2,
        scoring_func="softmax",
        native_quant=QuantizationType.FP16,
    )
)


# ────────────────────────────────────────────────────────────────── #
# Fixtures                                                            #
# ────────────────────────────────────────────────────────────────── #

HIDDEN = 64
SEQ = 4


@pytest.fixture
def dmap():
    return DeviceMap(
        attention_on_gpu=False,  # CPU-only for CI
        moe_on_cpu=True,
        model_id="test-small",
    )


@pytest.fixture
def engine(dmap):
    return HeterogeneousEngine(device_map=dmap)


@pytest.fixture
def packet():
    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((SEQ, HIDDEN)).astype(np.float16)
    return TensorPacket(
        packet_id="test-packet",
        tensor=tensor,
        layer_start=0,
        layer_end=2,
        token_ids=list(range(SEQ)),
        selected_experts=None,
        geo_region="local",
        src_node="test",
        dst_node="test",
        metadata={},
    )


# ────────────────────────────────────────────────────────────────── #
# KTransformersStub                                                   #
# ────────────────────────────────────────────────────────────────── #

class TestKTransformersStub:
    def test_rms_layer_norm_output_shape(self):
        x = np.ones((SEQ, HIDDEN), dtype=np.float16)
        w = np.ones(HIDDEN, dtype=np.float16)
        out = KTransformersStub.rms_layer_norm(x, w)
        assert out.shape == x.shape

    def test_rms_layer_norm_dtype_preserved(self):
        x = np.ones((SEQ, HIDDEN), dtype=np.float16)
        w = np.ones(HIDDEN, dtype=np.float16)
        out = KTransformersStub.rms_layer_norm(x, w)
        assert out.dtype == np.float16

    def test_rms_layer_norm_ones(self):
        x = np.ones((2, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        out = KTransformersStub.rms_layer_norm(x, w)
        np.testing.assert_allclose(out, np.ones((2, 4)), atol=1e-5)

    def test_rope_embedding_shape(self):
        x = np.ones((SEQ, HIDDEN), dtype=np.float16)
        pos = np.arange(SEQ)
        out = KTransformersStub.rope_embedding(x, pos)
        assert out.shape == x.shape

    def test_rope_embedding_dtype_preserved(self):
        x = np.ones((SEQ, HIDDEN), dtype=np.float16)
        pos = np.arange(SEQ)
        out = KTransformersStub.rope_embedding(x, pos)
        assert out.dtype == np.float16

    def test_mla_output_shape(self):
        q = np.ones((1, SEQ, HIDDEN), dtype=np.float16)
        k = np.ones((1, SEQ, HIDDEN), dtype=np.float16)
        v = np.ones((1, SEQ, HIDDEN), dtype=np.float16)
        out = KTransformersStub.multi_latent_attention(q, k, v, head_dim=HIDDEN)
        assert out.shape == (1, SEQ, HIDDEN)


# ────────────────────────────────────────────────────────────────── #
# DeviceMap                                                           #
# ────────────────────────────────────────────────────────────────── #

class TestDeviceMap:
    def test_for_16gb_gpu(self):
        dm = DeviceMap.for_16gb_gpu()
        assert dm.attention_on_gpu is True
        assert dm.moe_on_cpu is True

    def test_cpu_only(self):
        dm = DeviceMap.cpu_only()
        assert dm.attention_on_gpu is False
        assert dm.moe_on_cpu is True


# ────────────────────────────────────────────────────────────────── #
# LayerKVCache                                                        #
# ────────────────────────────────────────────────────────────────── #

class TestLayerKVCache:
    def test_append_first(self):
        cache = LayerKVCache()
        k = np.ones((2, 8), dtype=np.float32)
        v = np.ones((2, 8), dtype=np.float32)
        cache.append(k, v)
        assert cache.k.shape == (2, 8)
        assert cache.v.shape == (2, 8)

    def test_append_grows(self):
        cache = LayerKVCache()
        k = np.ones((2, 8), dtype=np.float32)
        v = np.ones((2, 8), dtype=np.float32)
        cache.append(k, v)
        cache.append(k, v)
        assert cache.k.shape == (4, 8)

    def test_clear(self):
        cache = LayerKVCache()
        k = np.ones((2, 8), dtype=np.float32)
        v = np.ones((2, 8), dtype=np.float32)
        cache.append(k, v)
        cache.clear()
        assert cache.k is None
        assert cache.v is None


# ────────────────────────────────────────────────────────────────── #
# HeterogeneousEngine                                                 #
# ────────────────────────────────────────────────────────────────── #

class TestHeterogeneousEngine:
    def test_forward_preserves_shape(self, engine, packet):
        out = engine.forward(packet, layer_indices=[0])
        assert out.tensor.shape == packet.tensor.shape

    def test_forward_preserves_dtype(self, engine, packet):
        out = engine.forward(packet, layer_indices=[0])
        assert out.tensor.dtype == packet.tensor.dtype

    def test_forward_metadata_updated(self, engine, packet):
        out = engine.forward(packet, layer_indices=[0])
        assert "compute_ms" in out.metadata

    def test_forward_no_kv_cache(self, engine, packet):
        out = engine.forward(packet, layer_indices=[0], use_kv_cache=False)
        assert out.tensor.shape == packet.tensor.shape
        assert len(engine.kv_cache) == 0

    def test_kv_cache_grows_with_attention_enabled(self, packet):
        dmap_with_attn = DeviceMap(
            attention_on_gpu=True,
            moe_on_cpu=True,
            model_id="test-small",
        )
        eng = HeterogeneousEngine(device_map=dmap_with_attn)
        eng.forward(packet, layer_indices=[0], use_kv_cache=True)
        assert 0 in eng.kv_cache

    def test_clear_kv_cache(self, engine, packet):
        engine.forward(packet, layer_indices=[0], use_kv_cache=True)
        engine.clear_kv_cache()
        assert len(engine.kv_cache) == 0

    def test_forward_with_moe(self, engine):
        rng = np.random.default_rng(1)
        tensor = rng.standard_normal((SEQ, HIDDEN)).astype(np.float16)
        sel_experts = rng.integers(0, 8, size=(SEQ, 2)).astype(np.int32)
        packet = TensorPacket(
            packet_id="moe-test",
            tensor=tensor,
            layer_start=0,
            layer_end=1,
            token_ids=list(range(SEQ)),
            selected_experts=sel_experts,
            geo_region="local",
            src_node="a",
            dst_node="b",
            metadata={},
        )
        out = engine.forward(packet, layer_indices=[0])
        assert out.tensor.shape == (SEQ, HIDDEN)

    def test_load_shared_experts(self, engine):
        ew = ExpertWeights(
            expert_id=0,
            gate_proj=np.ones((16, HIDDEN), dtype=np.float16),
            up_proj=np.ones((16, HIDDEN), dtype=np.float16),
            down_proj=np.ones((HIDDEN, 16), dtype=np.float16),
        )
        engine.load_shared_experts([ew])
        assert engine._expert_cache.is_cached(0)

    def test_stats_keys(self, engine):
        s = engine.stats()
        assert "backend" in s
        assert "kv_cache_layers" in s
        assert "expert_cache" in s

    def test_public_kv_cache_property(self, engine, packet):
        engine.forward(packet, layer_indices=[0], use_kv_cache=True)
        assert isinstance(engine.kv_cache, dict)

    def test_from_device_map(self, dmap):
        e = HeterogeneousEngine.from_device_map(dmap)
        assert isinstance(e, HeterogeneousEngine)

    def test_multiple_layers(self, engine, packet):
        out = engine.forward(packet, layer_indices=[0, 1, 2])
        assert out.tensor.shape == packet.tensor.shape
