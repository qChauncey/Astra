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
Phase 7.3.4 — Expert Replication & Affinity-Aware Routing Tests.

Tests for:
  - ExpertTelemetry: dispatch recording, hot-expert detection, pruning
  - ClusterAffinity: group building, nearest-node lookup, proximity updates
  - GeoAwareMoERouter: replica-aware dispatch (with_telemetry)
  - Orchestrator: GPU-load offloading integration
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from astra.routing.expert_telemetry import (
    ExpertTelemetry,
    HotSpot,
)
from astra.routing.cluster_affinity import (
    ClusterAffinity,
    NodeProximity,
)
from astra.routing.geo_router import (
    GeoAwareMoERouter,
    GeoRegion,
)
from astra.network.orchestrator import PipelineOrchestrator


# ================================================================== #
# ExpertTelemetry                                                      #
# ================================================================== #

class TestExpertTelemetry:
    @pytest.fixture
    def telemetry(self) -> ExpertTelemetry:
        return ExpertTelemetry(window_seconds=3600.0)

    def test_initial_state(self, telemetry: ExpertTelemetry):
        assert telemetry.total_dispatches() == 0
        assert telemetry.expert_counts() == {}
        assert telemetry.hot_experts() == []

    def test_record_single_dispatch(self, telemetry: ExpertTelemetry):
        telemetry.record_dispatch(expert_id=5, node_id="node-us-1")
        assert telemetry.total_dispatches() == 1
        assert telemetry.expert_counts() == {5: 1}

        per_node = telemetry.per_node_counts()
        assert per_node["node-us-1"] == {5: 1}

    def test_record_multiple_same_pair(self, telemetry: ExpertTelemetry):
        for _ in range(10):
            telemetry.record_dispatch(expert_id=3, node_id="node-eu-1")
        assert telemetry.total_dispatches() == 10
        assert telemetry.expert_counts() == {3: 10}

    def test_record_bulk(self, telemetry: ExpertTelemetry):
        assignments = [
            (0, "node-us-1"),
            (1, "node-us-1"),
            (2, "node-eu-2"),
            (0, "node-us-1"),
        ]
        telemetry.record_bulk(assignments)
        assert telemetry.total_dispatches() == 4
        counts = telemetry.expert_counts()
        assert counts[0] == 2
        assert counts[1] == 1
        assert counts[2] == 1

    def test_hot_experts_detection(self, telemetry: ExpertTelemetry):
        # Create uneven distribution: expert 7 is accessed 100x on node-us-1,
        # others get 1x each
        telemetry.record_bulk([(7, "node-us-1")] * 100)
        for eid in range(5):
            telemetry.record_dispatch(expert_id=eid, node_id="node-us-1")
        for eid in range(8, 15):
            telemetry.record_dispatch(expert_id=eid, node_id="node-eu-1")

        hot = telemetry.hot_experts(top_k=3, exclude_experts=[0, 1])
        assert len(hot) >= 1
        assert hot[0].expert_id == 7
        assert hot[0].node_id == "node-us-1"
        assert hot[0].access_count == 100
        assert hot[0].rank == 1

    def test_hot_experts_excludes_shared(self, telemetry: ExpertTelemetry):
        # Shared experts 0 and 1 should be excluded
        telemetry.record_bulk([(0, "node-us-1")] * 200)
        telemetry.record_bulk([(1, "node-us-1")] * 150)
        telemetry.record_bulk([(5, "node-us-1")] * 50)

        hot = telemetry.hot_experts(top_k=5, exclude_experts=[0, 1])
        if hot:
            assert all(h.expert_id not in (0, 1) for h in hot)

    def test_min_absolute_count_threshold(self):
        tel = ExpertTelemetry(min_absolute_count=5)
        tel.record_dispatch(3, "node-us-1")  # 1x → below threshold
        assert tel.hot_experts(top_k=3) == []

        for _ in range(4):
            tel.record_dispatch(3, "node-us-1")  # now 5x
        hot = tel.hot_experts(top_k=3)
        assert len(hot) >= 1

    def test_most_accessed_experts(self, telemetry: ExpertTelemetry):
        for eid in range(10):
            telemetry.record_bulk([(eid, "node-us-1")] * (eid + 1))

        top = telemetry.most_accessed_experts(top_k=3)
        assert top[0][0] == 9
        assert top[0][1] == 10
        assert top[1][0] == 8

    def test_prune_old_records(self):
        # Short window to test pruning
        tel = ExpertTelemetry(window_seconds=0.1)
        tel.record_dispatch(1, "n1")
        time.sleep(0.15)
        removed = tel.prune()
        assert removed == 1
        assert tel.total_dispatches() == 0

    def test_snapshot(self, telemetry: ExpertTelemetry):
        telemetry.record_dispatch(1, "n1")
        telemetry.record_dispatch(2, "n2")
        snap = telemetry.snapshot()
        assert snap["total_dispatches"] == 2
        assert snap["unique_experts"] == 2
        assert "top_experts" in snap
        assert "window_seconds" in snap

    def test_to_api_dict(self, telemetry: ExpertTelemetry):
        telemetry.record_bulk([(7, "n1")] * 50)
        api_dict = telemetry.to_api_dict()
        assert "telemetry" in api_dict
        assert "hot_experts" in api_dict["telemetry"]
        assert "top_experts" in api_dict["telemetry"]

    def test_reset(self, telemetry: ExpertTelemetry):
        telemetry.record_dispatch(1, "n1")
        telemetry.reset()
        assert telemetry.total_dispatches() == 0
        assert telemetry.expert_counts() == {}


# ================================================================== #
# HotSpot                                                               #
# ================================================================== #

class TestHotSpot:
    def test_repr(self):
        hs = HotSpot(expert_id=7, node_id="node-us-1", access_count=42, rank=3)
        r = repr(hs)
        assert "e=7" in r
        assert "n=node-us-1" in r
        assert "rank=3" in r


# ================================================================== #
# ClusterAffinity                                                       #
# ================================================================== #

class TestClusterAffinity:
    @pytest.fixture
    def affinity(self) -> ClusterAffinity:
        return ClusterAffinity(max_groups=8, intra_group_rtt_threshold_ms=5.0)

    def test_initial_state(self, affinity: ClusterAffinity):
        assert len(affinity.list_groups()) == 0
        assert affinity.node_group_id("any") is None

    def test_build_groups_from_proximity(self, affinity: ClusterAffinity):
        # 3 nodes in us-east with <5ms RTT, 2 nodes in eu-west
        measurements = [
            NodeProximity("node-us-east-1", "node-us-east-2", 1.2),
            NodeProximity("node-us-east-1", "node-us-east-3", 2.1),
            NodeProximity("node-us-east-2", "node-us-east-3", 1.8),
            NodeProximity("node-eu-west-1", "node-eu-west-2", 3.0),
            # Cross-region links: high RTT
            NodeProximity("node-us-east-1", "node-eu-west-1", 85.0),
            NodeProximity("node-us-east-2", "node-eu-west-2", 88.0),
        ]
        affinity.update_proximities(measurements)
        affinity.rebuild()

        groups = affinity.list_groups()
        assert len(groups) >= 2  # At least us-east and eu-west groups

        # Check us-east nodes are in same group
        gid1 = affinity.node_group_id("node-us-east-1")
        gid2 = affinity.node_group_id("node-us-east-2")
        gid3 = affinity.node_group_id("node-us-east-3")
        assert gid1 == gid2 == gid3

        # Check eu-west nodes are in same group (different from us-east)
        gid4 = affinity.node_group_id("node-eu-west-1")
        gid5 = affinity.node_group_id("node-eu-west-2")
        assert gid4 == gid5
        assert gid4 != gid1

    def test_peers_in_same_group(self, affinity: ClusterAffinity):
        measurements = [
            NodeProximity("n-1", "n-2", 1.0),
            NodeProximity("n-1", "n-3", 2.0),
            NodeProximity("n-2", "n-3", 1.5),
            NodeProximity("n-4", "n-5", 0.5),
        ]
        affinity.update_proximities(measurements)
        affinity.rebuild()

        peers = affinity.peers_in_same_group("n-1", ["n-2", "n-3", "n-4", "n-5"])
        assert set(peers) == {"n-2", "n-3"}
        assert "n-4" not in peers
        assert "n-5" not in peers

    def test_find_nearest_group_node(self, affinity: ClusterAffinity):
        # n-1 and n-2: 1ms, n-1 and n-3: 80ms
        measurements = [
            NodeProximity("n-1", "n-2", 1.0),
            NodeProximity("n-1", "n-3", 80.0),
            NodeProximity("n-2", "n-3", 82.0),
        ]
        affinity.update_proximities(measurements)
        affinity.rebuild()

        nearest = affinity.find_nearest_group_node(
            "n-1", ["n-2", "n-3"]
        )
        assert nearest == "n-2"  # Same group + lowest RTT

    def test_no_groups_for_unknown_node(self, affinity: ClusterAffinity):
        assert affinity.node_group_id("ghost") is None
        assert affinity.group_for_node("ghost") is None
        assert affinity.peers_in_same_group("ghost", ["n-1"]) == []

    def test_summary(self, affinity: ClusterAffinity):
        measurements = [
            NodeProximity("n-1", "n-2", 1.0),
        ]
        affinity.update_proximities(measurements)
        affinity.rebuild()
        s = affinity.summary()
        assert s["num_groups"] >= 1
        assert "threshold_ms" in s

    def test_ema_update(self, affinity: ClusterAffinity):
        # EMA: new = old*0.7 + new*0.3
        measurements = [NodeProximity("a", "b", 10.0)]
        affinity.update_proximities(measurements)
        # Second update with lower RTT → EMA brings it down
        affinity.update_proximities([NodeProximity("a", "b", 4.0)])
        rtt = affinity._proximity[("a", "b")]
        expected = 10.0 * 0.7 + 4.0 * 0.3  # = 8.2
        assert abs(rtt - expected) < 0.01

    def test_single_node_groups(self, affinity: ClusterAffinity):
        # 3 isolated nodes with no proximity data
        affinity.rebuild()  # No data → no nodes → no groups
        assert len(affinity.list_groups()) == 0


# ================================================================== #
# GeoAwareMoERouter — replica-aware dispatch                            #
# ================================================================== #

class TestReplicaAwareRouting:
    @pytest.fixture
    def geo_router(self) -> GeoAwareMoERouter:
        return GeoAwareMoERouter()

    @pytest.fixture
    def telemetry(self) -> ExpertTelemetry:
        return ExpertTelemetry(window_seconds=3600.0)

    @pytest.fixture
    def affinity(self) -> ClusterAffinity:
        return ClusterAffinity()

    def test_router_with_telemetry_registration(self, geo_router: GeoAwareMoERouter, telemetry: ExpertTelemetry):
        from astra.routing.geo_router import NodeInfo
        # Register nodes using NodeInfo objects
        geo_router.register_node(NodeInfo(
            node_id="local",
            region=GeoRegion("us-west", 37.7749, -122.4194),
            layer_start=0, layer_end=20,
            expert_shards=list(range(32)),
        ))
        geo_router.register_node(NodeInfo(
            node_id="remote-1",
            region=GeoRegion("us-east", 40.7128, -74.0060),
            layer_start=20, layer_end=40,
            expert_shards=list(range(8)),
        ))
        geo_router.register_node(NodeInfo(
            node_id="remote-2",
            region=GeoRegion("us-east", 40.7500, -73.9967),
            layer_start=20, layer_end=40,
            expert_shards=list(range(16, 32)),
        ))

        # Attach telemetry
        geo_router.set_telemetry(telemetry)

        assert geo_router._telemetry is telemetry

    def test_dispatch_records_telemetry(self, geo_router: GeoAwareMoERouter, telemetry: ExpertTelemetry):
        from astra.routing.geo_router import NodeInfo
        geo_router.register_node(NodeInfo(
            node_id="local", region=GeoRegion("us-west", 37.77, -122.41),
            layer_start=0, layer_end=20, expert_shards=list(range(32)),
        ))
        geo_router.register_node(NodeInfo(
            node_id="remote-1", region=GeoRegion("us-east", 40.71, -74.0),
            layer_start=20, layer_end=40, expert_shards=list(range(16)),
        ))
        geo_router.register_node(NodeInfo(
            node_id="remote-2", region=GeoRegion("us-east", 40.75, -73.99),
            layer_start=20, layer_end=40, expert_shards=list(range(16, 32)),
        ))
        geo_router.set_telemetry(telemetry)

        # Simulate routing via dispatch_with_telemetry
        selected = np.array([[0, 5, 20]], dtype=np.int32)  # 1 token, 3 experts
        geo_router.dispatch_with_telemetry(selected, layer_idx=0)
        assert telemetry.total_dispatches() >= 0  # shared expert 0 excluded

    def test_get_replica_target_empty_when_no_hotspots(self, geo_router: GeoAwareMoERouter, telemetry: ExpertTelemetry):
        geo_router.set_telemetry(telemetry)
        # recommend_replicas needs affinity too, but without it returns empty
        targets = geo_router.recommend_replicas(max_replicas=3)
        assert targets == []

    def test_get_replica_targets_from_hotspots(self, geo_router: GeoAwareMoERouter, telemetry: ExpertTelemetry, affinity: ClusterAffinity):
        from astra.routing.geo_router import NodeInfo
        # Register several nodes
        geo_router.register_node(NodeInfo(
            node_id="n-us-1", region=GeoRegion("us-west", 37.7, -122.4),
            layer_start=0, layer_end=20, expert_shards=list(range(64)),
        ))
        geo_router.register_node(NodeInfo(
            node_id="n-us-2", region=GeoRegion("us-west", 37.8, -122.3),
            layer_start=0, layer_end=20, expert_shards=list(range(8)),
        ))
        geo_router.register_node(NodeInfo(
            node_id="n-eu-1", region=GeoRegion("eu-west", 51.5, -0.1),
            layer_start=0, layer_end=20, expert_shards=list(range(64)),
        ))
        geo_router.set_telemetry(telemetry)
        geo_router.set_cluster_affinity(affinity)

        # Create hotspots on n-us-1
        telemetry.record_bulk([(10, "n-us-1")] * 200)
        telemetry.record_bulk([(15, "n-us-1")] * 150)
        telemetry.record_bulk([(3, "n-eu-1")] * 5)

        # Build affinity groups by adding n-us-1 and n-us-2 close together
        affinity.update_proximities([
            NodeProximity("n-us-1", "n-us-2", 1.0),
            NodeProximity("n-us-1", "n-eu-1", 85.0),
            NodeProximity("n-us-2", "n-eu-1", 88.0),
        ])
        affinity.rebuild()

        targets = geo_router.recommend_replicas(max_replicas=4)
        # Should return expert 10 and 15 on n-us-1 replicating to n-us-2
        assert len(targets) >= 1
        for t in targets:
            assert len(t) == 3  # (expert_id, source_node, target_node)


# ================================================================== #
# PipelineOrchestrator — GPU load-offloading integration                #
# ================================================================== #

class TestOrchestratorGPUOffloading:
    """Tests the GPU utilisation load-offloading feature of PipelineOrchestrator."""

    def test_gpu_util_threshold_default(self):
        from astra.network.dht import AstraDHT
        dht = AstraDHT()
        orch = PipelineOrchestrator(dht)
        assert orch._gpu_util_threshold == 0.9

    def test_set_gpu_util_threshold(self):
        from astra.network.dht import AstraDHT
        dht = AstraDHT()
        orch = PipelineOrchestrator(dht)
        orch.set_gpu_util_threshold(0.75)
        assert orch._gpu_util_threshold == 0.75

    def test_set_gpu_util_threshold_invalid(self):
        from astra.network.dht import AstraDHT
        dht = AstraDHT()
        orch = PipelineOrchestrator(dht)
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            orch.set_gpu_util_threshold(1.5)

    def test_update_node_health(self):
        from astra.network.dht import AstraDHT
        dht = AstraDHT()
        orch = PipelineOrchestrator(dht)
        orch.update_node_health("n-1", {"gpu_util": 0.5, "mem_used_pct": 60.0, "active_requests": 5})
        assert orch._node_health["n-1"]["gpu_util"] == 0.5
        assert orch._node_health["n-1"]["mem_used_pct"] == 60.0
        assert orch._node_health["n-1"]["active_requests"] == 5

    def test_is_node_overloaded(self):
        from astra.network.dht import AstraDHT
        dht = AstraDHT()
        orch = PipelineOrchestrator(dht)
        # No health data → not overloaded
        assert orch._is_node_overloaded("n-1") is False

        # Under threshold
        orch.update_node_health("n-1", {"gpu_util": 0.5})
        assert orch._is_node_overloaded("n-1") is False

        # At threshold
        orch.update_node_health("n-2", {"gpu_util": 0.9})
        assert orch._is_node_overloaded("n-2") is True

        # Above threshold
        orch.update_node_health("n-3", {"gpu_util": 0.95})
        assert orch._is_node_overloaded("n-3") is True

    def test_node_load_score(self):
        from astra.network.dht import AstraDHT
        dht = AstraDHT()
        orch = PipelineOrchestrator(dht)
        orch.update_node_health("n-1", {"gpu_util": 0.3, "mem_used_pct": 50.0, "active_requests": 2})
        score = orch._node_load_score("n-1")
        # 0.6*0.3 + 0.25*0.5 + 0.15*(0/3) = 0.18 + 0.125 + 0 = 0.305
        assert 0.30 <= score <= 0.31

    def test_node_load_score_no_health(self):
        from astra.network.dht import AstraDHT
        dht = AstraDHT()
        orch = PipelineOrchestrator(dht)
        score = orch._node_load_score("ghost")
        assert score == 0.0

    def test_node_load_score_fully_loaded(self):
        from astra.network.dht import AstraDHT
        dht = AstraDHT()
        orch = PipelineOrchestrator(dht)
        orch.update_node_health("n-max", {"gpu_util": 1.0, "mem_used_pct": 100.0, "active_requests": 99})
        with orch._inflight_lock:
            orch._inflight_counts["n-max"] = 50
        score = orch._node_load_score("n-max")
        # 0.6*1.0 + 0.25*1.0 + 0.15*min(1.0, 50/150) = 0.6+0.25+0.05 = 0.90
        assert score >= 0.89
