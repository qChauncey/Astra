# Copyright 2025 Project Astra Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for astra.network.hivemind_bridge — DHT factory and backend detection."""

from unittest import mock

import pytest

from astra.network.dht import AstraDHT, DHTNodeRecord, _GlobalStore
from astra.network.hivemind_bridge import (
    HivemindDHT,
    _HAS_HIVEMIND,
    _HIVEMIND_IMPORT_ERROR,
    create_dht,
    is_hivemind_available,
)


class TestBackendDetection:
    def test_is_hivemind_available_returns_bool(self):
        result = is_hivemind_available()
        assert isinstance(result, bool)

    def test_hivemind_available_matches_module_constant(self):
        assert is_hivemind_available() == _HAS_HIVEMIND

    def test_import_error_is_none_when_available(self):
        if _HAS_HIVEMIND:
            assert _HIVEMIND_IMPORT_ERROR is None


class TestCreateDhtDefault:
    """When use_hivemind=False (default), AstraDHT is always used."""

    def test_default_returns_astra_dht(self):
        dht = create_dht(node_id="test-node")
        assert isinstance(dht, AstraDHT)
        assert dht.node_id == "test-node"

    def test_explicit_astra_dht(self):
        dht = create_dht(node_id="n1", use_hivemind=False)
        assert isinstance(dht, AstraDHT)

    def test_unique_node_ids_when_not_provided(self):
        d1 = create_dht()
        d2 = create_dht()
        assert d1.node_id != d2.node_id


class TestCreateDhtHivemindMode:
    """When use_hivemind=True but hivemind is not installed, falls back to AstraDHT."""

    def test_fallback_when_hivemind_not_installed(self):
        if _HAS_HIVEMIND:
            pytest.skip("hivemind is installed — cannot test fallback path")
        dht = create_dht(node_id="fallback-1", use_hivemind=True)
        assert isinstance(dht, AstraDHT)
        assert dht.node_id == "fallback-1"

    def test_hivemind_not_installed_fallback_logs_info(self, caplog):
        if _HAS_HIVEMIND:
            pytest.skip("hivemind is installed — cannot test fallback path")
        with caplog.at_level("INFO"):
            create_dht(node_id="fb2", use_hivemind=True)
        matching = [r for r in caplog.records if "not installed" in r.message]
        assert len(matching) >= 1


class TestHivemindDHTClass:
    """Tests for HivemindDHT class definition (statically available only when imported)."""

    def test_class_defined_when_hivemind_available(self):
        if _HAS_HIVEMIND:
            assert HivemindDHT is not None
        else:
            assert HivemindDHT is None

    @pytest.mark.skipif(not _HAS_HIVEMIND, reason="hivemind not installed")
    def test_hivemind_dht_can_be_instantiated(self):
        """Smoke test: HivemindDHT instantiates without error (starts real DHT)."""
        import hivemind
        # Use a unique port to avoid conflicts
        dht = HivemindDHT(node_id="smoke-test", port=15555)
        try:
            assert dht.node_id == "smoke-test"
            assert dht.get_all_peers() == []
        finally:
            dht.revoke()


class TestDHTAnnounceGetRoundTrip:
    """End-to-end test: announce → get_all_peers round-trip with AstraDHT backend."""

    def test_announce_and_discover_single_peer(self):
        store = _GlobalStore()
        dht = AstraDHT(node_id="publisher", store=store)
        record = DHTNodeRecord(
            node_id="publisher",
            address="localhost:5555",
            layer_start=0,
            layer_end=30,
            expert_shards=[1, 2, 3],
            geo_region="us-west",
        )
        dht.announce(record, ttl=300)
        peers = dht.get_all_peers()
        assert any(p.node_id == "publisher" for p in peers)

    def test_announce_and_discover_multiple_peers(self):
        store = _GlobalStore()
        for i in range(3):
            dht = AstraDHT(node_id=f"peer-{i}", store=store)
            record = DHTNodeRecord(
                node_id=f"peer-{i}",
                address=f"localhost:600{i}",
                layer_start=i * 10,
                layer_end=(i + 1) * 10,
                expert_shards=list(range(i * 4, i * 4 + 4)),
                geo_region="local",
            )
            dht.announce(record, ttl=300)
        consumer = AstraDHT(node_id="consumer", store=store)
        peers = consumer.get_all_peers()
        assert len(peers) >= 3

    def test_get_peer_by_id(self):
        store = _GlobalStore()
        dht = AstraDHT(node_id="lookup-src", store=store)
        record = DHTNodeRecord(
            node_id="lookup-target",
            address="localhost:7000",
            layer_start=0,
            layer_end=10,
            expert_shards=[0],
            geo_region="eu-west",
        )
        dht.announce(record, ttl=300)
        consumer = AstraDHT(node_id="consumer", store=store)
        found = consumer.get_peer("lookup-target")
        assert found is not None
        assert found.node_id == "lookup-target"
        assert found.geo_region == "eu-west"

    def test_revoke_removes_peer(self):
        store = _GlobalStore()
        dht = AstraDHT(node_id="revoke-me", store=store)
        record = DHTNodeRecord(
            node_id="revoke-me",
            address="localhost:7001",
            layer_start=0,
            layer_end=5,
            expert_shards=[0],
            geo_region="local",
        )
        dht.announce(record, ttl=300)
        assert any(p.node_id == "revoke-me" for p in dht.get_all_peers())
        dht.revoke()
        peers = dht.get_all_peers()
        assert not any(p.node_id == "revoke-me" for p in peers)

    def test_subscribe_peers_callback(self):
        store = _GlobalStore()
        consumer = AstraDHT(node_id="subscriber", store=store)
        received = []

        def on_peer(node_id, record):
            received.append((node_id, record))

        consumer.subscribe_peers(on_peer)
        dht = AstraDHT(node_id="new-peer", store=store)
        record = DHTNodeRecord(
            node_id="new-peer",
            address="localhost:8000",
            layer_start=0,
            layer_end=10,
            expert_shards=[0, 1],
            geo_region="asia-east",
        )
        dht.announce(record, ttl=300)
        assert len(received) >= 1
        assert received[0][0] == "new-peer"


class TestCreateDhtWithStore:
    """Test that create_dht produces a DHT usable with shared stores."""

    def test_shared_store_between_two_dhts(self):
        dht_a = create_dht(node_id="shared-a")
        dht_b = create_dht(node_id="shared-b")
        # They use the global store by default, so an announcement from A
        # should be visible to B.
        record = DHTNodeRecord(
            node_id="shared-a",
            address="localhost:9000",
            layer_start=0,
            layer_end=10,
            expert_shards=[0, 1],
            geo_region="local",
        )
        dht_a.announce(record, ttl=300)
        peers = dht_b.get_all_peers()
        assert any(p.node_id == "shared-a" for p in peers)