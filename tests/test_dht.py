# Copyright 2025 Project Astra Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for astra.network.dht."""

import time


from astra.network.dht import AstraDHT, DHTNodeRecord, _GlobalStore


def _record(node_id: str, ls: int = 0, le: int = 10, region: str = "local") -> DHTNodeRecord:
    return DHTNodeRecord(
        node_id=node_id,
        address=f"localhost:{50000 + ls}",
        layer_start=ls,
        layer_end=le,
        expert_shards=list(range(4)),
        geo_region=region,
    )


class TestGlobalStore:
    def test_put_get(self):
        store = _GlobalStore()
        store.put("k1", "hello", ttl=60, owner="a")
        assert store.get("k1") == "hello"

    def test_ttl_expiry(self):
        store = _GlobalStore()
        store.put("k2", "val", ttl=0.05, owner="a")
        time.sleep(0.1)
        assert store.get("k2") is None

    def test_no_expiry_when_ttl_zero(self):
        store = _GlobalStore()
        store.put("k3", "persist", ttl=0, owner="a")
        time.sleep(0.05)
        assert store.get("k3") == "persist"

    def test_delete_by_owner(self):
        store = _GlobalStore()
        store.put("k4", "v", ttl=60, owner="owner1")
        store.delete("k4", owner="owner1")
        assert store.get("k4") is None

    def test_delete_wrong_owner(self):
        store = _GlobalStore()
        store.put("k5", "v", ttl=60, owner="owner1")
        result = store.delete("k5", owner="owner2")
        assert not result
        assert store.get("k5") == "v"

    def test_keys_by_prefix(self):
        store = _GlobalStore()
        store.put("ns/a", 1, ttl=60, owner="x")
        store.put("ns/b", 2, ttl=60, owner="x")
        store.put("other/c", 3, ttl=60, owner="x")
        keys = store.keys_by_prefix("ns/")
        assert "ns/a" in keys
        assert "ns/b" in keys
        assert "other/c" not in keys

    def test_subscribe_callback(self):
        store = _GlobalStore()
        received = []
        store.subscribe("watch/", lambda k, v: received.append((k, v)))
        store.put("watch/x", "data", ttl=60, owner="a")
        time.sleep(0.05)
        assert any(k == "watch/x" for k, _ in received)


class TestAstraDHT:
    def test_announce_and_get_peer(self):
        dht = AstraDHT(node_id="test-announce")
        rec = _record("test-announce")
        dht.announce(rec, ttl=60)
        found = dht.get_peer("test-announce")
        assert found is not None
        assert found.node_id == "test-announce"
        dht.revoke()

    def test_get_all_peers(self):
        store = _GlobalStore()
        dht_a = AstraDHT(node_id="pa", store=store)
        dht_b = AstraDHT(node_id="pb", store=store)
        dht_a.announce(_record("pa"), ttl=60)
        dht_b.announce(_record("pb"), ttl=60)
        peers = dht_a.get_all_peers()
        ids = {p.node_id for p in peers}
        assert "pa" in ids
        assert "pb" in ids
        dht_a.revoke()
        dht_b.revoke()

    def test_revoke_removes_peer(self):
        store = _GlobalStore()
        dht = AstraDHT(node_id="rm-me", store=store)
        dht.announce(_record("rm-me"), ttl=60)
        dht.revoke()
        assert dht.get_peer("rm-me") is None

    def test_peers_for_layer(self):
        store = _GlobalStore()
        dht_a = AstraDHT(node_id="la", store=store)
        dht_b = AstraDHT(node_id="lb", store=store)
        dht_a.announce(_record("la", ls=0, le=20), ttl=60)
        dht_b.announce(_record("lb", ls=20, le=40), ttl=60)
        peers0 = dht_a.peers_for_layer(5)
        peers30 = dht_a.peers_for_layer(30)
        assert any(p.node_id == "la" for p in peers0)
        assert any(p.node_id == "lb" for p in peers30)
        dht_a.revoke()
        dht_b.revoke()

    def test_peers_for_expert(self):
        store = _GlobalStore()
        dht = AstraDHT(node_id="ex", store=store)
        rec = DHTNodeRecord(
            node_id="ex", address="localhost:9000",
            layer_start=0, layer_end=10,
            expert_shards=[10, 20, 30], geo_region="local"
        )
        dht.announce(rec, ttl=60)
        assert any(p.node_id == "ex" for p in dht.peers_for_expert(20))
        assert not any(p.node_id == "ex" for p in dht.peers_for_expert(99))
        dht.revoke()

    def test_generic_kv_store(self):
        dht = AstraDHT(node_id="kv-test")
        dht.store("my-key", {"data": 42}, ttl=60)
        assert dht.fetch("my-key") == {"data": 42}

    def test_subscribe_peers_callback(self):
        store = _GlobalStore()
        dht_watcher = AstraDHT(node_id="watcher", store=store)
        arrivals = []
        dht_watcher.subscribe_peers(lambda nid, rec: arrivals.append(nid))

        dht_joiner = AstraDHT(node_id="joiner", store=store)
        dht_joiner.announce(_record("joiner"), ttl=60)
        time.sleep(0.05)
        assert "joiner" in arrivals
        dht_joiner.revoke()
