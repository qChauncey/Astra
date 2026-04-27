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
Phase 7.3.4 — Cluster Affinity Groups.

Groups nodes into low-latency affinity clusters based on inter-node RTT
measurements.  The GeoAwareMoERouter uses this information to place expert
replicas within the same affinity group as the hot node, minimizing replica
fetch latency while avoiding cross-region data movement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple



@dataclass
class NodeProximity:
    """Pairwise RTT between two nodes."""
    node_a: str
    node_b: str
    rtt_ms: float


@dataclass
class AffinityGroup:
    """A set of nodes that share low inter-node RTT."""
    group_id: int
    label: str  # descriptive label like "us-east-1" or "cluster-3"
    node_ids: Set[str] = field(default_factory=set)
    region: str = "local"
    avg_rtt_ms: float = 0.0
    num_nodes: int = 0

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "label": self.label,
            "region": self.region,
            "num_nodes": self.num_nodes,
            "avg_rtt_ms": round(self.avg_rtt_ms, 1),
            "node_ids": sorted(self.node_ids),
        }


class ClusterAffinity:
    """
    Builds and maintains node affinity groups from RTT measurements.

    Nodes within the same affinity group typically share a physical data
    center or metro region, giving them <3ms inter-node RTT.  The router
    prefers placing replicas within the same group to minimize overhead.

    Parameters
    ----------
    max_groups : int
        Maximum number of affinity groups to maintain.
    intra_group_rtt_threshold_ms : float
        Nodes with mutual RTT below this threshold are grouped together.
    """

    def __init__(
        self,
        max_groups: int = 32,
        intra_group_rtt_threshold_ms: float = 5.0,
    ) -> None:
        self._max_groups = max_groups
        self._rtt_threshold_ms = intra_group_rtt_threshold_ms
        self._groups: Dict[int, AffinityGroup] = {}
        self._node_group: Dict[str, int] = {}  # node_id → group_id
        self._proximity: Dict[Tuple[str, str], float] = {}  # (a,b)→rtt_ms, a<b
        self._next_group_id = 0

    # ------------------------------------------------------------------ #
    # Build / update                                                         #
    # ------------------------------------------------------------------ #

    def update_proximities(self, measurements: List[NodeProximity]) -> None:
        """Merge new pairwise RTT measurements into the affinity graph."""
        for m in measurements:
            key = self._sort_pair(m.node_a, m.node_b)
            current = self._proximity.get(key)
            if current is None:
                self._proximity[key] = m.rtt_ms
            else:
                # Exponential moving average
                self._proximity[key] = current * 0.7 + m.rtt_ms * 0.3

    def rebuild(self) -> None:
        """
        Reconstruct affinity groups from current proximity data.

        Uses a greedy clustering algorithm:
          1. Pick the unassigned node with the most low-RTT neighbours.
          2. Form a group of all unassigned nodes within RTT threshold.
          3. Repeat until all nodes are assigned or max_groups reached.
        """
        all_nodes: Set[str] = set()
        for (a, b), _ in self._proximity.items():
            all_nodes.add(a)
            all_nodes.add(b)

        self._groups.clear()
        self._node_group.clear()
        self._next_group_id = 0

        if not all_nodes:
            return

        assigned: Set[str] = set()

        while assigned != all_nodes and len(self._groups) < self._max_groups:
            # Find the unassigned node with the most unassigned low-RTT neighbours
            best_node = self._find_best_seed(all_nodes, assigned)

            if best_node is None:
                # All remaining nodes are isolated; group them individually
                break

            # Build group around the seed
            group_ids: Set[str] = {best_node}
            for candidate in all_nodes - assigned - {best_node}:
                key = self._sort_pair(best_node, candidate)
                rtt = self._proximity.get(key, float("inf"))
                if rtt <= self._rtt_threshold_ms:
                    group_ids.add(candidate)

            gid = self._next_group_id
            self._next_group_id += 1

            region = self._infer_region(group_ids)
            avg_rtt = self._calc_avg_rtt(group_ids)

            self._groups[gid] = AffinityGroup(
                group_id=gid,
                label=f"affinity-{gid}",
                node_ids=group_ids,
                region=region,
                avg_rtt_ms=avg_rtt,
                num_nodes=len(group_ids),
            )
            for nid in group_ids:
                self._node_group[nid] = gid
            assigned |= group_ids

        # Any remaining nodes get their own solo group
        for nid in all_nodes - assigned:
            gid = self._next_group_id
            self._next_group_id += 1
            region = self._infer_region({nid})
            self._groups[gid] = AffinityGroup(
                group_id=gid,
                label=f"affinity-{gid}",
                node_ids={nid},
                region=region,
                avg_rtt_ms=0.0,
                num_nodes=1,
            )
            self._node_group[nid] = gid

    def _find_best_seed(
        self, all_nodes: Set[str], assigned: Set[str]
    ) -> Optional[str]:
        """Pick the unassigned node with the most low-RTT unassigned neighbours."""
        best_node = None
        best_score = -1
        unassigned = all_nodes - assigned
        for n in unassigned:
            score = 0
            for other in unassigned - {n}:
                rtt = self._proximity.get(self._sort_pair(n, other), float("inf"))
                if rtt <= self._rtt_threshold_ms:
                    score += 1
            if score > best_score:
                best_score = score
                best_node = n
        return best_node

    def _infer_region(self, node_ids: Set[str]) -> str:
        """Infer the dominant region from node IDs (by convention node-<region>-*)."""
        region_counts: Dict[str, int] = {}
        for nid in node_ids:
            # Simple heuristic: extract region from node ID
            parts = nid.split("-")
            if len(parts) >= 2:
                region = parts[1] if len(parts) <= 3 else parts[1]
            else:
                region = "local"
            region_counts[region] = region_counts.get(region, 0) + 1
        return max(region_counts, key=region_counts.get) if region_counts else "local"

    def _calc_avg_rtt(self, node_ids: Set[str]) -> float:
        """Average pairwise RTT within the group."""
        nlist = list(node_ids)
        if len(nlist) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(nlist)):
            for j in range(i + 1, len(nlist)):
                rtt = self._proximity.get(self._sort_pair(nlist[i], nlist[j]))
                if rtt is not None:
                    total += rtt
                    count += 1
        return total / count if count > 0 else 0.0

    @staticmethod
    def _sort_pair(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a < b else (b, a)

    # ------------------------------------------------------------------ #
    # Query                                                                  #
    # ------------------------------------------------------------------ #

    def node_group_id(self, node_id: str) -> Optional[int]:
        """Return the affinity group ID for *node_id*, or None."""
        return self._node_group.get(node_id)

    def group_for_node(self, node_id: str) -> Optional[AffinityGroup]:
        """Return the affinity group containing *node_id*."""
        gid = self._node_group.get(node_id)
        if gid is not None:
            return self._groups.get(gid)
        return None

    def peers_in_same_group(
        self, node_id: str, candidate_ids: List[str]
    ) -> List[str]:
        """Filter *candidate_ids* to only those in the same group as *node_id*."""
        gid = self._node_group.get(node_id)
        if gid is None:
            return []
        group = self._groups.get(gid)
        if group is None:
            return []
        return [c for c in candidate_ids if c in group.node_ids]

    def find_nearest_group_node(
        self,
        target_node_id: str,
        candidate_node_ids: List[str],
    ) -> Optional[str]:
        """
        Find the candidate node with the lowest RTT to the target.

        Prefer candidates in the same affinity group; fall back to global minimum.
        """
        gid = self._node_group.get(target_node_id)
        group = self._groups.get(gid) if gid is not None else None

        # Phase 1: same-group candidates
        if group:
            same_group = [c for c in candidate_node_ids if c in group.node_ids]
            if same_group:
                return self._lowest_rtt(target_node_id, same_group)

        # Phase 2: any candidate
        return self._lowest_rtt(target_node_id, candidate_node_ids)

    def _lowest_rtt(
        self, target: str, candidates: List[str]
    ) -> Optional[str]:
        best = None
        best_rtt = float("inf")
        for c in candidates:
            rtt = self._proximity.get(self._sort_pair(target, c), float("inf"))
            if rtt < best_rtt:
                best_rtt = rtt
                best = c
        return best

    def list_groups(self) -> List[AffinityGroup]:
        return sorted(self._groups.values(), key=lambda g: g.group_id)

    def summary(self) -> dict:
        return {
            "num_groups": len(self._groups),
            "num_nodes": len(self._node_group),
            "groups": [g.to_dict() for g in self.list_groups()],
            "threshold_ms": self._rtt_threshold_ms,
        }
