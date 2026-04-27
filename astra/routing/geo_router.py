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
#
# MODIFICATIONS (Astra project):
#   - Completely rewritten from scratch (not derived from Petals source).
#   - Implements token-level geographic-aware MoE dispatch for DeepSeek-V4.
#   - Uses haversine distance + RTT estimates to build micro-cluster affinity.

"""
Token-level Geographic-Aware MoE Router.

For each token's top-K expert assignments, this router selects the node in
the P2P cluster that (a) hosts the required expert shard and (b) has the
lowest estimated round-trip latency from the requesting node.

In the mock/simulation mode, RTT is estimated from geographic coordinates
using the haversine formula plus a fixed base latency per hop.

Architecture:
  GeoRegion      — named geographic cluster (e.g. "us-west", "eu-central")
  NodeInfo       — per-peer metadata: region, hosted layers, expert shards
  GeoAwareMoERouter
      .register_node()    — add a peer to the routing table
      .gate()             — run MoE gate and assign experts to tokens
      .dispatch()         — produce per-node dispatch plan
      .route()            — full gate + dispatch in one call
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .expert_telemetry import ExpertTelemetry
    from .cluster_affinity import ClusterAffinity

from ..serialization.tensor_pack import (
    DEEPSEEK_V4_TOP_K_EXPERTS,
    DEEPSEEK_V4_SHARED_EXPERTS,
    TensorPacket,
)

log = logging.getLogger(__name__)


@dataclass
class GeoRegion:
    name: str
    lat: float   # degrees
    lon: float   # degrees

    def distance_km(self, other: "GeoRegion") -> float:
        """Haversine great-circle distance in km."""
        R = 6371.0
        phi1, phi2 = math.radians(self.lat), math.radians(other.lat)
        dphi = math.radians(other.lat - self.lat)
        dlam = math.radians(other.lon - self.lon)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    def rtt_ms(self, other: "GeoRegion", base_ms: float = 5.0) -> float:
        """Estimated RTT (ms) = 2 × propagation + base overhead."""
        speed_of_light_fiber = 200_000.0  # km/s in fiber (~2/3 c)
        prop_ms = (self.distance_km(other) / speed_of_light_fiber) * 1000.0
        return 2 * prop_ms + base_ms


# Pre-defined regions for simulation
REGIONS: Dict[str, GeoRegion] = {
    "us-west":    GeoRegion("us-west",    37.77,  -122.42),
    "us-east":    GeoRegion("us-east",    40.71,   -74.01),
    "eu-central": GeoRegion("eu-central", 50.11,     8.68),
    "ap-east":    GeoRegion("ap-east",    35.69,   139.69),
    "local":      GeoRegion("local",       0.00,     0.00),
}


@dataclass
class NodeInfo:
    node_id: str
    region: GeoRegion
    layer_start: int          # first transformer layer this node hosts
    layer_end: int            # last transformer layer (exclusive)
    expert_shards: List[int]  # expert IDs this node can compute
    available: bool = True
    last_seen: float = field(default_factory=time.time)
    address: Optional[str] = None  # "host:port" — used for active RTT probes

    @property
    def num_experts(self) -> int:
        return len(self.expert_shards)

    def hosts_expert(self, expert_id: int) -> bool:
        return expert_id in self.expert_shards

    def hosts_layer(self, layer_idx: int) -> bool:
        return self.layer_start <= layer_idx < self.layer_end


@dataclass
class DispatchPlan:
    """
    Describes how tokens should be dispatched to nodes for one MoE layer.

    node_assignments: {node_id → list of (token_idx, expert_id)}
    shared_tokens:    token indices that hit shared experts (always local)
    """
    node_assignments: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    shared_tokens: List[int] = field(default_factory=list)
    layer_idx: int = 0

    def total_tokens_routed(self) -> int:
        return sum(len(v) for v in self.node_assignments.values())

    def num_nodes_used(self) -> int:
        return len(self.node_assignments)


class GeoAwareMoERouter:
    """
    Token-level, geography-aware MoE dispatch router for DeepSeek-V4.

    The router runs a mock gating network (softmax over expert scores) then
    maps each (token, expert) pair to the best available node using RTT
    estimates derived from geographic coordinates.
    """

    def __init__(
        self,
        local_region: str = "local",
        num_experts: int = 256,
        top_k: int = DEEPSEEK_V4_TOP_K_EXPERTS,
        num_shared: int = DEEPSEEK_V4_SHARED_EXPERTS,
        rtt_monitor: Optional[object] = None,
    ) -> None:
        self._local_region = REGIONS.get(local_region, GeoRegion(local_region, 0, 0))
        self._num_experts = num_experts
        self._top_k = top_k
        self._num_shared = num_shared
        self._nodes: Dict[str, NodeInfo] = {}
        # expert_id → list of node_ids that can serve it
        self._expert_index: Dict[int, List[str]] = {}
        # cached gate weight matrices keyed by layer_idx
        self._router_weights: Dict[int, np.ndarray] = {}
        # Optional RTTMonitor — when set, real measurements override haversine.
        self._rtt_monitor = rtt_monitor

    def _effective_rtt_ms(self, node: NodeInfo) -> float:
        """
        Return the best RTT estimate to *node*: real measurement if available
        and the peer is healthy; otherwise haversine fallback.
        """
        if self._rtt_monitor is not None and getattr(node, "address", None):
            measured = self._rtt_monitor.get_rtt(node.address)
            healthy = self._rtt_monitor.is_healthy(node.address)
            if measured is not None and healthy:
                return measured
        return self._local_region.rtt_ms(node.region)

    # ------------------------------------------------------------------ #
    # Registry                                                              #
    # ------------------------------------------------------------------ #

    def register_node(self, node: NodeInfo) -> None:
        self._nodes[node.node_id] = node
        for eid in node.expert_shards:
            self._expert_index.setdefault(eid, []).append(node.node_id)

    def deregister_node(self, node_id: str) -> None:
        node = self._nodes.pop(node_id, None)
        if node:
            for eid in node.expert_shards:
                lst = self._expert_index.get(eid, [])
                if node_id in lst:
                    lst.remove(node_id)

    # ------------------------------------------------------------------ #
    # Gating                                                                #
    # ------------------------------------------------------------------ #

    def gate(
        self,
        hidden_states: np.ndarray,
        layer_idx: int,
        router_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute top-K expert assignments for each token.

        hidden_states: (seq_len, hidden_dim) float16/32
        router_weights: (num_experts, hidden_dim) — mock gate weight matrix.
                        If None, a deterministic random matrix is used.
        Returns: selected_experts (seq_len, top_k) int32
        """
        seq_len, hidden_dim = hidden_states.shape
        if router_weights is None:
            if layer_idx not in self._router_weights:
                rng = np.random.default_rng(seed=layer_idx)
                self._router_weights[layer_idx] = rng.standard_normal(
                    (self._num_experts, hidden_dim)
                ).astype(np.float32)
            router_weights = self._router_weights[layer_idx]

        logits = hidden_states.astype(np.float32) @ router_weights.T  # (seq, num_experts)
        # DeepSeek-V4 uses sigmoid + top-k rather than softmax
        scores = 1.0 / (1.0 + np.exp(-logits))
        selected = np.argsort(-scores, axis=-1)[:, : self._top_k].astype(np.int32)
        return selected  # (seq_len, top_k)

    # ------------------------------------------------------------------ #
    # Dispatch                                                              #
    # ------------------------------------------------------------------ #

    def _best_node_for_expert(self, expert_id: int) -> Optional[str]:
        """Return the available node with lowest RTT that hosts this expert."""
        candidates = [
            nid for nid in self._expert_index.get(expert_id, [])
            if self._nodes[nid].available
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda nid: self._effective_rtt_ms(self._nodes[nid]),
        )

    def dispatch(
        self,
        selected_experts: np.ndarray,
        layer_idx: int,
    ) -> DispatchPlan:
        """
        Build a DispatchPlan mapping each (token, expert) to a node.

        Shared experts (IDs 0..num_shared-1) are handled locally and not
        dispatched to remote nodes.
        """
        plan = DispatchPlan(layer_idx=layer_idx)
        seq_len = selected_experts.shape[0]
        actual_k = selected_experts.shape[1] if selected_experts.ndim > 1 else 1

        for token_idx in range(seq_len):
            for k_idx in range(min(self._top_k, actual_k)):
                expert_id = int(selected_experts[token_idx, k_idx])
                if expert_id < self._num_shared:
                    # Shared expert: always local
                    plan.shared_tokens.append(token_idx)
                    continue
                node_id = self._best_node_for_expert(expert_id)
                if node_id is None:
                    # Fallback: assign to local node (first available)
                    node_id = next(iter(self._nodes), "local")
                plan.node_assignments.setdefault(node_id, []).append(
                    (token_idx, expert_id)
                )

        return plan

    def route(
        self,
        packet: TensorPacket,
        layer_idx: int,
    ) -> Tuple[TensorPacket, DispatchPlan]:
        """Gate + dispatch in one call; returns updated packet and plan."""
        selected = self.gate(packet.tensor, layer_idx)
        packet.selected_experts = selected
        packet.layer_start = layer_idx
        packet.layer_end = layer_idx + 1
        plan = self.dispatch(selected, layer_idx)
        return packet, plan

    def summary(self) -> dict:
        return {
            "num_nodes": len(self._nodes),
            "local_region": self._local_region.name,
            "expert_coverage": len(self._expert_index),
            "nodes": {
                nid: {
                    "region": n.region.name,
                    "layers": f"{n.layer_start}:{n.layer_end}",
                    "experts": len(n.expert_shards),
                    "available": n.available,
                }
                for nid, n in self._nodes.items()
            },
        }

    # ------------------------------------------------------------------ #
    # Phase 7.3.4 — Telemetry & Expert Replication                        #
    # ------------------------------------------------------------------ #

    def set_telemetry(self, telemetry: "ExpertTelemetry") -> None:
        """Attach an ExpertTelemetry collector for access-frequency tracking."""
        self._telemetry: Optional[ExpertTelemetry] = telemetry

    def set_cluster_affinity(self, affinity: "ClusterAffinity") -> None:
        """Attach a ClusterAffinity instance for replica placement."""
        self._affinity: Optional[ClusterAffinity] = affinity

    def dispatch_with_telemetry(
        self,
        selected_experts: np.ndarray,
        layer_idx: int,
    ) -> DispatchPlan:
        """
        Build dispatch plan *and* record every (expert, node) assignment
        in the telemetry collector for hot-expert detection.

        Falls back to plain dispatch() if no telemetry is attached.
        """
        plan = self.dispatch(selected_experts, layer_idx)
        if getattr(self, "_telemetry", None) is not None:
            assignments: List[Tuple[int, str]] = []
            for nid, pairs in plan.node_assignments.items():
                for token_idx, eid in pairs:
                    if eid >= self._num_shared:
                        assignments.append((eid, nid))
            self._telemetry.record_bulk(assignments)
        return plan

    def route_with_telemetry(
        self,
        packet: TensorPacket,
        layer_idx: int,
    ) -> Tuple[TensorPacket, DispatchPlan]:
        """Gate + telemetry dispatch in one call."""
        selected = self.gate(packet.tensor, layer_idx)
        packet.selected_experts = selected
        packet.layer_start = layer_idx
        packet.layer_end = layer_idx + 1
        plan = self.dispatch_with_telemetry(selected, layer_idx)
        return packet, plan

    def identify_hotspot_nodes(
        self,
        top_k: int = 8,
    ) -> List[Tuple[str, int, float]]:
        """
        Identify nodes that host the most hot experts.

        Returns list of (node_id, hotspot_count, avg_access_count) sorted
        by hotspot_count descending.  Empty if no telemetry is attached.
        """
        if not hasattr(self, "_telemetry") or self._telemetry is None:
            return []
        from .expert_telemetry import HotSpot
        hot: List[HotSpot] = self._telemetry.hot_experts(
            top_k=top_k, exclude_experts=list(range(self._num_shared))
        )
        # Aggregate by node
        node_count: Dict[str, int] = {}
        node_total: Dict[str, int] = {}
        for hs in hot:
            node_count[hs.node_id] = node_count.get(hs.node_id, 0) + 1
            node_total[hs.node_id] = node_total.get(hs.node_id, 0) + hs.access_count
        result = [
            (nid, node_count[nid], node_total[nid] / max(1, node_count[nid]))
            for nid in node_count
        ]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def recommend_replicas(
        self,
        max_replicas: int = 4,
        exclude_shared: bool = True,
    ) -> List[Tuple[int, str, str]]:
        """
        Recommend replica placement: (expert_id, source_node, target_node).

        For each hot expert, finds the best nearby node (within the same
        affinity group) that does NOT currently host the expert and has
        available capacity.
        """
        if (
            not hasattr(self, "_telemetry") or self._telemetry is None
            or not hasattr(self, "_affinity") or self._affinity is None
        ):
            return []

        from .expert_telemetry import HotSpot

        exclude_ids = list(range(self._num_shared)) if exclude_shared else []
        hot: List[HotSpot] = self._telemetry.hot_experts(
            top_k=max_replicas * 2, exclude_experts=exclude_ids
        )

        recommendations: List[Tuple[int, str, str]] = []
        affinity: "ClusterAffinity" = self._affinity

        for hs in hot:
            if len(recommendations) >= max_replicas:
                break
            current_hosts = set(self._expert_index.get(hs.expert_id, []))
            group = affinity.group_for_node(hs.node_id)
            if group is None:
                continue
            # Find nodes in the same group that don't host this expert
            candidates = [
                nid for nid in group.node_ids
                if nid not in current_hosts
                and self._nodes.get(nid) is not None
                and self._nodes[nid].available
            ]
            if not candidates:
                continue
            target = affinity.find_nearest_group_node(hs.node_id, candidates)
            if target:
                recommendations.append((hs.expert_id, hs.node_id, target))

        return recommendations

    def apply_replica_placements(
        self, placements: List[Tuple[int, str, str]]
    ) -> int:
        """
        Apply replica placements: register the expert on the target node.

        Returns number of replicas actually created.
        """
        count = 0
        for expert_id, source_node, target_node in placements:
            if target_node in self._nodes and target_node not in self._expert_index.get(expert_id, []):
                self._nodes[target_node].expert_shards.append(expert_id)
                self._expert_index.setdefault(expert_id, []).append(target_node)
                count += 1
                log.info(
                    "Replica created: expert=%d from=%s to=%s",
                    expert_id, source_node, target_node,
                )
        return count

    def get_telemetry_report(self) -> Optional[dict]:
        """Return a telemetry summary dict or None if not configured."""
        if getattr(self, "_telemetry", None) is None:
            return None
        tel: ExpertTelemetry = self._telemetry
        return {
            **tel.to_api_dict(),
            "hotspot_nodes": [
                {"node_id": nid, "hotspot_count": cnt, "avg_access": round(avg, 1)}
                for nid, cnt, avg in self.identify_hotspot_nodes(top_k=8)
            ],
            "replica_recommendations": [
                {"expert_id": eid, "source_node": src, "target_node": tgt}
                for eid, src, tgt in self.recommend_replicas(max_replicas=4)
            ],
        }
