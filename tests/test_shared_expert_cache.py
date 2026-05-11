# Copyright 2025 Project Astra Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for astra.inference.shared_expert_cache."""

import numpy as np
import pytest

from astra.inference.shared_expert_cache import ExpertWeights, SharedExpertCache


HIDDEN = 32
INTER = 16
MAX_TOKENS = 8


def _make_cache(max_cached: int = 4) -> SharedExpertCache:
    return SharedExpertCache(max_cached_experts=max_cached, hidden_dim=HIDDEN, intermediate_dim=INTER)


def _mock_ew(eid: int) -> ExpertWeights:
    return ExpertWeights.mock(eid, hidden_dim=HIDDEN, intermediate_dim=INTER)


class TestExpertWeights:
    def test_mock_shapes(self):
        ew = _mock_ew(0)
        assert ew.gate_proj.shape == (INTER, HIDDEN)
        assert ew.up_proj.shape   == (INTER, HIDDEN)
        assert ew.down_proj.shape == (HIDDEN, INTER)

    def test_nbytes(self):
        ew = _mock_ew(0)
        expected = INTER * HIDDEN * 2 * 3   # float16 × 3 matrices
        assert ew.nbytes == expected


class TestSharedExpertCachePinning:
    def test_pin_never_evicted(self):
        cache = _make_cache(max_cached=2)
        cache.pin(0, _mock_ew(0))
        cache.pin(1, _mock_ew(1))
        # Loading a third should try to evict non-pinned; all are pinned → error
        with pytest.raises(RuntimeError, match="pinned"):
            cache.load(2, _mock_ew(2))

    def test_pinned_expert_cached(self):
        cache = _make_cache()
        cache.pin(0, _mock_ew(0))
        assert cache.is_cached(0)

    def test_pin_multiple(self):
        cache = _make_cache(max_cached=4)
        for i in range(4):
            cache.pin(i, _mock_ew(i))
        assert cache.cache_size() == 4


class TestSharedExpertCacheLRU:
    def test_lru_eviction(self):
        cache = _make_cache(max_cached=2)
        cache.pin(0, _mock_ew(0))    # pinned, never evicted
        cache.load(1, _mock_ew(1))   # LRU candidate
        cache.load(2, _mock_ew(2))   # evicts 1
        assert cache.is_cached(0)
        assert not cache.is_cached(1)
        assert cache.is_cached(2)

    def test_access_updates_lru_order(self):
        cache = _make_cache(max_cached=3)
        for i in range(3):
            cache.load(i, _mock_ew(i))
        # Access 0 to make it recently used
        cache.forward(0, np.zeros((1, HIDDEN), dtype=np.float16))
        # Load 3 — should evict 1 (now LRU), not 0
        cache.load(3, _mock_ew(3))
        assert cache.is_cached(0)
        assert not cache.is_cached(1)

    def test_load_existing_hits_cache(self):
        cache = _make_cache()
        ew = _mock_ew(5)
        cache.load(5, ew)
        size_before = cache.cache_size()
        cache.load(5, ew)   # second load of same ID — no-op
        assert cache.cache_size() == size_before


class TestSharedExpertCacheForward:
    def test_forward_shape(self):
        cache = _make_cache()
        cache.pin(0, _mock_ew(0))
        x = np.random.default_rng(0).standard_normal((4, HIDDEN)).astype(np.float16)
        out = cache.forward(0, x)
        assert out.shape == (4, HIDDEN)
        assert out.dtype == np.float16

    def test_forward_missing_expert_raises(self):
        cache = _make_cache()
        x = np.zeros((1, HIDDEN), dtype=np.float16)
        with pytest.raises(KeyError):
            cache.forward(99, x)

    def test_stats_keys(self):
        cache = _make_cache()
        cache.pin(0, _mock_ew(0))
        s = cache.stats()
        assert "cached_experts" in s
        assert "pinned_experts" in s
        assert "cache_utilization" in s
        assert "total_bytes" in s


class TestCrossLayerExpertCache:
    """Phase 8 – Gated cross-layer expert intermediate cache."""

    def test_store_lookup_roundtrip(self):
        cache = _make_cache()
        hid = np.random.default_rng(42).standard_normal((MAX_TOKENS, HIDDEN)).astype(np.float32)
        cache.cross_layer_store("layer_0:expert_1", hid)
        result = cache.cross_layer_lookup("layer_0:expert_1")
        assert result is not None
        np.testing.assert_array_equal(result, hid)

    def test_lookup_miss_returns_none(self):
        cache = _make_cache()
        assert cache.cross_layer_lookup("no_such_key") is None

    def test_lru_eviction_when_full(self):
        cache = _make_cache()
        # Bypass private _cross_max
        cache._cross_max = 3
        for i in range(5):
            hid = np.full((1, HIDDEN), float(i), dtype=np.float32)
            cache.cross_layer_store(f"layer_{i}:expert_0", hid)
        # Oldest entries (0, 1) should be evicted — only 3 slots remain
        assert cache.cross_layer_lookup("layer_0:expert_0") is None
        assert cache.cross_layer_lookup("layer_1:expert_0") is None
        # Newer entries still present
        for i in range(2, 5):
            assert cache.cross_layer_lookup(f"layer_{i}:expert_0") is not None

    def test_lookup_refreshes_lru_order(self):
        cache = _make_cache()
        cache._cross_max = 2
        cache.cross_layer_store("a", np.zeros((1, HIDDEN), dtype=np.float32))
        cache.cross_layer_store("b", np.ones((1, HIDDEN), dtype=np.float32))
        # Access "a" so it becomes MRU; "b" becomes LRU
        cache.cross_layer_lookup("a")
        # Insert "c" — should evict "b", not "a"
        cache.cross_layer_store("c", np.full((1, HIDDEN), 2.0, dtype=np.float32))
        assert cache.cross_layer_lookup("a") is not None
        assert cache.cross_layer_lookup("b") is None
        assert cache.cross_layer_lookup("c") is not None

    def test_warmup_populates_cache(self):
        cache = _make_cache()
        entries = {
            f"L{i}": (np.arange(HIDDEN, dtype=np.float32) + i).reshape(1, HIDDEN)
            for i in range(4)
        }
        cache.cross_layer_warmup(entries)
        for k in entries:
            assert cache.cross_layer_lookup(k) is not None

    def test_hit_miss_stats(self):
        cache = _make_cache()
        hid = np.ones((1, HIDDEN), dtype=np.float32)
        cache.cross_layer_store("hit_me", hid)
        cache.cross_layer_lookup("hit_me")          # hit
        cache.cross_layer_lookup("hit_me")          # hit
        cache.cross_layer_lookup("miss_me")         # miss
        cache.cross_layer_lookup("miss_again")      # miss
        assert cache._cross_hit_count == 2
        assert cache._cross_miss_count == 2

    def test_weights_cache_namespace_isolation(self):
        """Cross-layer and weight caches do not share storage."""
        cache = _make_cache()
        # Use an integer key that collides with expert IDs
        hid = np.zeros((1, HIDDEN), dtype=np.float32)
        cache.cross_layer_store("0", hid)   # string "0"
        cache.pin(0, _mock_ew(0))           # integer 0
        # Cross-layer lookup finds the hidden state
        out = cache.cross_layer_lookup("0")
        assert out is not None
        np.testing.assert_array_equal(out, hid)
        # Weight cache forward still works independently
        cache.forward(0, np.zeros((1, HIDDEN), dtype=np.float16))

    def test_evict_on_empty_cache(self):
        cache = _make_cache()
        cache._cross_max = 1
        hid = np.zeros((1, HIDDEN), dtype=np.float32)
        cache.cross_layer_store("only", hid)
        # Now insert another → evict "only", which is the LRU
        cache.cross_layer_store("second", hid)
        assert cache.cross_layer_lookup("only") is None
        assert cache.cross_layer_lookup("second") is not None
