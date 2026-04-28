# Copyright 2025 Project Astra Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for PipelineOrchestrator + AstraDHT integration."""

import threading
import time

import pytest

from astra.inference.heterogeneous import DeviceMap
from astra.network.dht import AstraDHT, DHTNodeRecord, _GlobalStore
from astra.network.orchestrator import PipelineConfig, PipelineOrchestrator
from astra.rpc.server import InferenceServer

HIDDEN = 128
PORT_BASE = 50300


def _start_server(node_id: str, port: int, ls: int, le: int) -> InferenceServer:
    dmap = DeviceMap.cpu_only(model_id="test-orch")
    s = InferenceServer(node_id=node_id, layer_start=ls, layer_end=le,
                        port=port, geo_region="local", device_map=dmap)
    threading.Thread(target=s.start, daemon=True).start()
    return s


def _dht_record(node_id: str, port: int, ls: int, le: int) -> DHTNodeRecord:
    return DHTNodeRecord(
        node_id=node_id,
        address=f"localhost:{port}",
        layer_start=ls,
        layer_end=le,
        expert_shards=list(range(8)),
        geo_region="local",
    )


@pytest.fixture(scope="module")
def two_node_cluster():
    s1 = _start_server("orch-A", PORT_BASE, 0, 10)
    s2 = _start_server("orch-B", PORT_BASE + 1, 10, 20)
    time.sleep(0.5)

    store = _GlobalStore()
    dht = AstraDHT(node_id="orchestrator", store=store)
    dht_a = AstraDHT(node_id="orch-A", store=store)
    dht_b = AstraDHT(node_id="orch-B", store=store)
    dht_a.announce(_dht_record("orch-A", PORT_BASE, 0, 10), ttl=300)
    dht_b.announce(_dht_record("orch-B", PORT_BASE + 1, 10, 20), ttl=300)

    cfg = PipelineConfig(num_layers=20, hidden_dim=HIDDEN,
                         num_experts=8, top_k=2, num_shared_experts=2)
    orch = PipelineOrchestrator(dht=dht, config=cfg)

    yield orch, dht

    s1.stop(grace=1.0)
    s2.stop(grace=1.0)
    dht_a.revoke()
    dht_b.revoke()


class TestOrchestrator:
    def test_topology_shows_two_nodes(self, two_node_cluster):
        orch, _ = two_node_cluster
        topo = orch.topology()
        assert topo["num_peers"] == 2

    def test_run_produces_correct_shape(self, two_node_cluster):
        orch, _ = two_node_cluster
        result = orch.run(token_ids=list(range(8)))
        assert result.output.tensor.shape[0] == 8
        assert result.output.tensor.shape[1] == HIDDEN

    def test_run_records_two_hops(self, two_node_cluster):
        orch, _ = two_node_cluster
        result = orch.run(token_ids=[1, 2, 3, 4])
        assert result.num_hops == 2
        assert all(h.success for h in result.hops)

    def test_run_total_ms_positive(self, two_node_cluster):
        orch, _ = two_node_cluster
        result = orch.run(token_ids=[0])
        assert result.total_ms > 0

    def test_gap_detection_raises(self):
        store = _GlobalStore()
        dht = AstraDHT(node_id="gap-test", store=store)
        # Only announce layers 0-10, leaving 10-20 uncovered
        r = _dht_record("gap-node", PORT_BASE + 10, 0, 10)
        dht2 = AstraDHT(node_id="gap-node", store=store)
        dht2.announce(r, ttl=60)
        cfg = PipelineConfig(num_layers=20, hidden_dim=HIDDEN)
        orch = PipelineOrchestrator(dht=dht, config=cfg)
        with pytest.raises(RuntimeError, match="uncovered|gap"):
            orch.run([1, 2])
        dht2.revoke()

    def test_no_peers_raises(self):
        store = _GlobalStore()
        dht = AstraDHT(node_id="empty", store=store)
        orch = PipelineOrchestrator(dht=dht)
        with pytest.raises(RuntimeError, match="No peers"):
            orch.run([1])
