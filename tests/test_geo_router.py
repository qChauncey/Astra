# Copyright 2025 Project Astra Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for astra.routing.geo_router."""

import numpy as np
import pytest

from astra.routing.geo_router import (
    DispatchPlan,
    GeoAwareMoERouter,
    GeoRegion,
    NodeInfo,
    REGIONS,
)
from astra.serialization.tensor_pack import TensorPacket


HIDDEN = 64
NUM_EXPERTS = 8
TOP_K = 2
NUM_SHARED = 2


def _make_router(local: str = "local") -> GeoAwareMoERouter:
    return GeoAwareMoERouter(
        local_region=local,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        num_shared=NUM_SHARED,
    )


def _make_node(node_id: str, region: str, experts: list) -> NodeInfo:
    return NodeInfo(
        node_id=node_id,
        region=REGIONS.get(region, GeoRegion(region, 0, 0)),
        layer_start=0,
        layer_end=10,
        expert_shards=experts,
    )


class TestGeoRegion:
    def test_same_region_zero_distance(self):
        r = GeoRegion("a", 40.0, -74.0)
        assert r.distance_km(r) == pytest.approx(0.0, abs=1.0)

    def test_cross_atlantic_distance(self):
        us = REGIONS["us-east"]
        eu = REGIONS["eu-central"]
        d = us.distance_km(eu)
        assert 5000 < d < 8000   # roughly 6000 km

    def test_rtt_positive(self):
        a, b = REGIONS["us-west"], REGIONS["ap-east"]
        assert a.rtt_ms(b) > 0

    def test_rtt_local_low(self):
        local = REGIONS["local"]
        assert local.rtt_ms(local) < 10.0


class TestGeoAwareMoERouterGate:
    def test_gate_output_shape(self):
        router = _make_router()
        hidden = np.zeros((6, HIDDEN), dtype=np.float16)
        selected = router.gate(hidden, layer_idx=0)
        assert selected.shape == (6, TOP_K)
        assert selected.dtype == np.int32

    def test_gate_expert_ids_in_range(self):
        router = _make_router()
        hidden = np.random.default_rng(0).standard_normal((10, HIDDEN)).astype(np.float16)
        selected = router.gate(hidden, layer_idx=0)
        assert (selected >= 0).all()
        assert (selected < NUM_EXPERTS).all()

    def test_gate_deterministic(self):
        router = _make_router()
        hidden = np.ones((4, HIDDEN), dtype=np.float16)
        s1 = router.gate(hidden, layer_idx=5)
        s2 = router.gate(hidden, layer_idx=5)
        assert np.array_equal(s1, s2)


class TestGeoAwareMoERouterDispatch:
    def test_shared_tokens_in_plan(self):
        router = _make_router()
        # Force all tokens to select shared expert 0 and 1
        selected = np.zeros((4, TOP_K), dtype=np.int32)   # experts 0,0 for all
        plan = router.dispatch(selected, layer_idx=0)
        assert len(plan.shared_tokens) > 0

    def test_routed_tokens_assigned_to_node(self):
        router = _make_router()
        node = _make_node("n1", "us-west", list(range(2, NUM_EXPERTS)))
        router.register_node(node)
        # Token 0 routes to expert 3 (routed, not shared)
        selected = np.array([[3, 4], [3, 4]], dtype=np.int32)
        plan = router.dispatch(selected, layer_idx=0)
        assert "n1" in plan.node_assignments

    def test_deregister_removes_node(self):
        router = _make_router()
        node = _make_node("n2", "local", [5, 6])
        router.register_node(node)
        router.deregister_node("n2")
        assert router.summary()["num_nodes"] == 0

    def test_no_nodes_falls_back_to_local(self):
        router = _make_router()
        selected = np.array([[3, 4]], dtype=np.int32)
        plan = router.dispatch(selected, layer_idx=0)
        # Should not raise; fallback to "local"
        assert isinstance(plan, DispatchPlan)

    def test_nearest_node_preferred(self):
        router = _make_router(local="us-west")
        near = _make_node("near", "us-west", [5])
        far  = _make_node("far",  "ap-east", [5])
        router.register_node(near)
        router.register_node(far)
        selected = np.array([[5, 6]], dtype=np.int32)
        plan = router.dispatch(selected, layer_idx=0)
        # "near" should be preferred over "far"
        assert "near" in plan.node_assignments or not plan.node_assignments


class TestRoute:
    def test_route_updates_packet(self):
        router = _make_router()
        pkt = TensorPacket.make_input([1, 2, 3], hidden_dim=HIDDEN)
        updated, plan = router.route(pkt, layer_idx=2)
        assert updated.selected_experts is not None
        assert updated.selected_experts.shape == (3, TOP_K)
        assert updated.layer_start == 2

    def test_summary_keys(self):
        router = _make_router()
        s = router.summary()
        assert "num_nodes" in s
        assert "local_region" in s
