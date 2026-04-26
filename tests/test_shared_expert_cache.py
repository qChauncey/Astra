# Copyright 2025 Project Astra Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for astra.inference.shared_expert_cache."""

import numpy as np
import pytest

from astra.inference.shared_expert_cache import ExpertWeights, SharedExpertCache


HIDDEN = 32
INTER = 16


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
