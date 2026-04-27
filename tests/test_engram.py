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

"""Tests for astra.network.engram — storage-only DHT peers."""

from __future__ import annotations

import pathlib

import pytest

from astra.network.dht import AstraDHT
from astra.network.engram import (
    DiskEngramStore,
    EngramCapability,
    EngramNode,
    InMemoryEngramStore,
    ROLE_ENGRAM,
    discover_engrams,
    find_blob_holder,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_dht():
    """Each test gets an isolated DHT view."""
    return AstraDHT(node_id="test-dht")


# ── InMemoryEngramStore ──────────────────────────────────────────────────────

class TestInMemoryStore:
    def test_put_get_roundtrip(self):
        s = InMemoryEngramStore()
        s.put_blob("k1", b"hello")
        assert s.get_blob("k1") == b"hello"

    def test_missing_key_returns_none(self):
        s = InMemoryEngramStore()
        assert s.get_blob("never") is None

    def test_delete(self):
        s = InMemoryEngramStore()
        s.put_blob("k1", b"x")
        assert s.delete_blob("k1") is True
        assert s.get_blob("k1") is None
        assert s.delete_blob("k1") is False

    def test_list_keys_sorted(self):
        s = InMemoryEngramStore()
        s.put_blob("zeta", b"z")
        s.put_blob("alpha", b"a")
        assert s.list_keys() == ["alpha", "zeta"]

    def test_total_bytes(self):
        s = InMemoryEngramStore()
        s.put_blob("a", b"12345")
        s.put_blob("b", b"678")
        assert s.total_bytes() == 8


# ── DiskEngramStore ──────────────────────────────────────────────────────────

class TestDiskStore:
    def test_put_get_roundtrip(self, tmp_path: pathlib.Path):
        s = DiskEngramStore(tmp_path)
        s.put_blob("kv-cache-42", b"binary-blob")
        assert s.get_blob("kv-cache-42") == b"binary-blob"

    def test_persists_across_instances(self, tmp_path: pathlib.Path):
        s1 = DiskEngramStore(tmp_path)
        s1.put_blob("survives", b"yes")
        s2 = DiskEngramStore(tmp_path)
        assert s2.get_blob("survives") == b"yes"

    def test_path_traversal_sanitised(self, tmp_path: pathlib.Path):
        s = DiskEngramStore(tmp_path)
        s.put_blob("../escape", b"contained")
        # Should not have written outside tmp_path
        assert not (tmp_path.parent / "escape").exists()
        assert s.get_blob("../escape") == b"contained"

    def test_delete(self, tmp_path: pathlib.Path):
        s = DiskEngramStore(tmp_path)
        s.put_blob("k", b"v")
        assert s.delete_blob("k") is True
        assert s.get_blob("k") is None
        assert s.delete_blob("k") is False

    def test_total_bytes(self, tmp_path: pathlib.Path):
        s = DiskEngramStore(tmp_path)
        s.put_blob("a", b"1234")
        s.put_blob("b", b"56")
        assert s.total_bytes() == 6


# ── EngramNode ───────────────────────────────────────────────────────────────

class TestEngramNode:
    def test_basic_storage(self):
        e = EngramNode("engram-A", "127.0.0.1:60001")
        e.put_blob("k", b"data")
        assert e.has_blob("k") is True
        assert e.get_blob("k") == b"data"

    def test_capacity_enforced(self):
        e = EngramNode("engram-A", "127.0.0.1:60001", capacity_bytes=10)
        e.put_blob("k1", b"123456")    # 6 bytes — fine
        with pytest.raises(MemoryError):
            e.put_blob("k2", b"7890123")    # would total 13 > 10

    def test_zero_capacity_means_unlimited(self):
        e = EngramNode("engram-A", "127.0.0.1:60001", capacity_bytes=0)
        e.put_blob("big", b"x" * 10_000)
        assert e.get_blob("big") == b"x" * 10_000

    def test_capability_reports_usage(self):
        e = EngramNode("engram-A", "127.0.0.1:60001", capacity_bytes=1024)
        e.put_blob("k", b"hello")
        cap = e.capability()
        assert cap.role == ROLE_ENGRAM
        assert cap.used_bytes == 5
        assert cap.capacity_bytes == 1024
        assert "k" in cap.blob_keys

    def test_capability_serialisation_roundtrip(self):
        cap = EngramCapability(
            node_id="x", address="a:1", capacity_bytes=100, used_bytes=50,
            blob_keys=["foo", "bar"], geo_region="us-west",
        )
        d = cap.to_dict()
        cap2 = EngramCapability.from_dict(d)
        assert cap2.node_id == cap.node_id
        assert cap2.role == ROLE_ENGRAM
        assert cap2.blob_keys == cap.blob_keys


# ── DHT integration ──────────────────────────────────────────────────────────

class TestDHTIntegration:
    def test_announce_makes_engram_discoverable(self, fresh_dht):
        e = EngramNode("engram-A", "127.0.0.1:60001")
        e.put_blob("kv-1", b"data")
        e.announce(fresh_dht)
        found = discover_engrams(fresh_dht)
        ids = [c.node_id for c in found]
        assert "engram-A" in ids

    def test_revoke_removes_engram(self, fresh_dht):
        e = EngramNode("engram-A", "127.0.0.1:60001")
        e.announce(fresh_dht)
        e.revoke(fresh_dht)
        assert "engram-A" not in [c.node_id for c in discover_engrams(fresh_dht)]

    def test_find_blob_holder(self, fresh_dht):
        a = EngramNode("engram-A", "127.0.0.1:60001")
        b = EngramNode("engram-B", "127.0.0.1:60002")
        a.put_blob("kv-X", b"x")
        b.put_blob("kv-Y", b"y")
        a.announce(fresh_dht)
        b.announce(fresh_dht)

        holder = find_blob_holder(fresh_dht, "kv-X")
        assert holder is not None
        assert holder.node_id == "engram-A"

        holder2 = find_blob_holder(fresh_dht, "kv-Y")
        assert holder2.node_id == "engram-B"

        assert find_blob_holder(fresh_dht, "nonexistent") is None
