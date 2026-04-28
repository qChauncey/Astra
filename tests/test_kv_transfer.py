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
Unit tests for KV-cache chunked streaming (KVCacheSender / KVCacheReceiver).

These tests exercise chunking / reassembly logic entirely in-process,
without starting a real gRPC server.  The gRPC round-trip is tested in
test_pipeline_grpc.py.
"""

import numpy as np

from astra.inference.heterogeneous import DeviceMap, HeterogeneousEngine, LayerKVCache
from astra.rpc.kv_transfer import (
    MAX_CHUNK_BYTES,
    KVCacheReceiver,
    _decode_chunk,
    _encode_array,
    _split_array,
)


# ────────────────────────────────────────────────────────────────── #
# Helpers / fixtures                                                  #
# ────────────────────────────────────────────────────────────────── #

HIDDEN = 64


def _make_engine(hidden: int = HIDDEN) -> HeterogeneousEngine:
    dmap = DeviceMap(attention_on_gpu=False, moe_on_cpu=True, model_id="test-small")
    return HeterogeneousEngine(device_map=dmap)


def _make_cache(seq: int, hidden: int = HIDDEN) -> LayerKVCache:
    rng = np.random.default_rng(7)
    cache = LayerKVCache()
    cache.k = rng.standard_normal((seq, hidden)).astype(np.float32)
    cache.v = rng.standard_normal((seq, hidden)).astype(np.float32)
    return cache


# Minimal stub for pb2.KVCacheChunk without importing grpc in test
class _FakeChunk:
    def __init__(self, *, request_id, layer_idx, k_bytes, k_shape, v_bytes, v_shape, dtype):
        self.request_id = request_id
        self.layer_idx = layer_idx
        self.k_bytes = k_bytes
        self.k_shape = k_shape
        self.v_bytes = v_bytes
        self.v_shape = v_shape
        self.dtype = dtype


# ────────────────────────────────────────────────────────────────── #
# _encode_array / _decode_chunk                                       #
# ────────────────────────────────────────────────────────────────── #

class TestEncodeDecodeArray:
    def test_round_trip_float32(self):
        arr = np.arange(24, dtype=np.float32).reshape(4, 6)
        raw, shape, dtype = _encode_array(arr)
        out = _decode_chunk(raw, shape, dtype)
        np.testing.assert_array_equal(arr, out)

    def test_round_trip_float16(self):
        arr = np.ones((8, 16), dtype=np.float16)
        raw, shape, dtype = _encode_array(arr)
        out = _decode_chunk(raw, shape, dtype)
        assert out.dtype == np.float16
        np.testing.assert_array_equal(arr, out)

    def test_shape_preserved(self):
        arr = np.zeros((5, 12), dtype=np.float32)
        _, shape, _ = _encode_array(arr)
        assert shape == [5, 12]

    def test_dtype_string(self):
        arr = np.zeros((2, 4), dtype=np.float16)
        _, _, dtype = _encode_array(arr)
        assert dtype == "float16"

    def test_contiguous(self):
        arr = np.ones((6, 8), dtype=np.float32)
        sliced = arr[::2]  # non-contiguous
        raw, shape, dtype = _encode_array(sliced)
        out = _decode_chunk(raw, shape, dtype)
        np.testing.assert_array_equal(sliced, out)


# ────────────────────────────────────────────────────────────────── #
# _split_array                                                        #
# ────────────────────────────────────────────────────────────────── #

class TestSplitArray:
    def test_small_array_single_chunk(self):
        arr = np.zeros((4, 64), dtype=np.float32)
        chunks = list(_split_array(arr))
        assert len(chunks) == 1
        chunk_arr, bounds = chunks[0]
        np.testing.assert_array_equal(chunk_arr, arr)

    def test_large_array_multiple_chunks(self):
        # Force multiple chunks: each row = 4 MB → 2 rows → 2 chunks
        row_bytes = MAX_CHUNK_BYTES + 1
        ncols = row_bytes // 4  # float32 = 4 bytes
        arr = np.zeros((3, ncols), dtype=np.float32)
        chunks = list(_split_array(arr))
        assert len(chunks) >= 3  # each row forced into its own chunk

    def test_chunks_reassemble(self):
        arr = np.arange(120, dtype=np.float32).reshape(10, 12)
        chunks = list(_split_array(arr))
        reconstructed = np.concatenate([c for c, _ in chunks], axis=0)
        np.testing.assert_array_equal(arr, reconstructed)

    def test_bounds_contiguous(self):
        arr = np.zeros((6, 8), dtype=np.float32)
        bounds_list = [b for _, b in _split_array(arr)]
        # bounds should cover [0, total) without gaps
        assert bounds_list[0][0] == 0
        assert bounds_list[-1][1] == 6
        for i in range(1, len(bounds_list)):
            assert bounds_list[i][0] == bounds_list[i - 1][1]


# ────────────────────────────────────────────────────────────────── #
# KVCacheReceiver — in-process reassembly                             #
# ────────────────────────────────────────────────────────────────── #

class TestKVCacheReceiver:
    def _make_chunks(self, layer_idx: int, k: np.ndarray, v: np.ndarray, req_id: str):
        chunks = []
        raw_k, shape_k, dtype = _encode_array(k)
        chunks.append(_FakeChunk(
            request_id=req_id,
            layer_idx=layer_idx,
            k_bytes=raw_k,
            k_shape=shape_k,
            v_bytes=b"",
            v_shape=[],
            dtype=dtype,
        ))
        raw_v, shape_v, dtype = _encode_array(v)
        chunks.append(_FakeChunk(
            request_id=req_id,
            layer_idx=layer_idx,
            k_bytes=b"",
            k_shape=[],
            v_bytes=raw_v,
            v_shape=shape_v,
            dtype=dtype,
        ))
        return chunks

    def test_single_layer_round_trip(self):
        engine = _make_engine()
        seq = 8
        k = np.arange(seq * HIDDEN, dtype=np.float32).reshape(seq, HIDDEN)
        v = k * 2

        chunks = self._make_chunks(0, k, v, "req-abc")
        rid, applied = KVCacheReceiver.receive_and_apply(engine, iter(chunks))

        assert rid == "req-abc"
        assert applied == 1
        assert 0 in engine.kv_cache
        np.testing.assert_array_equal(engine.kv_cache[0].k, k)
        np.testing.assert_array_equal(engine.kv_cache[0].v, v)

    def test_multiple_layers(self):
        engine = _make_engine()
        all_chunks = []
        for li in range(3):
            k = np.ones((4, HIDDEN), dtype=np.float32) * li
            v = np.ones((4, HIDDEN), dtype=np.float32) * (li + 10)
            all_chunks.extend(self._make_chunks(li, k, v, "req-xyz"))

        _, applied = KVCacheReceiver.receive_and_apply(engine, iter(all_chunks))
        assert applied == 3

    def test_empty_stream(self):
        engine = _make_engine()
        rid, applied = KVCacheReceiver.receive_and_apply(engine, iter([]))
        assert applied == 0

    def test_k_only_chunk(self):
        engine = _make_engine()
        k = np.ones((2, HIDDEN), dtype=np.float32)
        raw_k, shape_k, dtype = _encode_array(k)
        chunk = _FakeChunk(
            request_id="k-only",
            layer_idx=5,
            k_bytes=raw_k,
            k_shape=shape_k,
            v_bytes=b"",
            v_shape=[],
            dtype=dtype,
        )
        rid, applied = KVCacheReceiver.receive_and_apply(engine, iter([chunk]))
        assert applied == 1
        assert engine.kv_cache[5].k is not None
        assert engine.kv_cache[5].v is None

    def test_v_only_chunk(self):
        engine = _make_engine()
        v = np.ones((2, HIDDEN), dtype=np.float32)
        raw_v, shape_v, dtype = _encode_array(v)
        chunk = _FakeChunk(
            request_id="v-only",
            layer_idx=7,
            k_bytes=b"",
            k_shape=[],
            v_bytes=raw_v,
            v_shape=shape_v,
            dtype=dtype,
        )
        KVCacheReceiver.receive_and_apply(engine, iter([chunk]))
        assert engine.kv_cache[7].v is not None
        assert engine.kv_cache[7].k is None

    def test_request_id_returned(self):
        engine = _make_engine()
        k = np.ones((2, HIDDEN), dtype=np.float32)
        v = np.ones((2, HIDDEN), dtype=np.float32)
        chunks = self._make_chunks(0, k, v, "my-req-id")
        rid, _ = KVCacheReceiver.receive_and_apply(engine, iter(chunks))
        assert rid == "my-req-id"
