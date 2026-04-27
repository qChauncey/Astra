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
Phase 7.3.4 — Expert Access Frequency Telemetry.

Tracks (expert_id, node_id) dispatch events so the GeoAwareMoERouter can
identify hot experts and trigger replica placement on nearby nodes.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ExpertAccessRecord:
    """Per-expert, per-node access statistics."""
    expert_id: int
    node_id: str
    count: int = 0
    last_access: float = 0.0

    def touch(self) -> None:
        self.count += 1
        self.last_access = time.time()


@dataclass(frozen=True)
class HotSpot:
    """A hot expert identified by telemetry analysis."""
    expert_id: int
    node_id: str
    access_count: int
    rank: int  # 1 = hottest

    def __repr__(self) -> str:
        return f"HotSpot(e={self.expert_id}, n={self.node_id}, count={self.access_count}, rank={self.rank})"


class ExpertTelemetry:
    """
    Thread-safe collector for expert access frequency data.

    Records every ``(expert_id, node_id)`` dispatch event so the router can
    identify hot experts that benefit from replica placement.

    Parameters
    ----------
    window_seconds : float
        Sliding window duration.  Records older than this are pruned on read.
    hot_threshold_percentile : float
        Top percentile considered "hot" (e.g. 90.0 = top 10% by access count).
    min_absolute_count : int
        Minimum accesses before an expert can be considered hot, preventing
        false positives on cold-start.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        hot_threshold_percentile: float = 90.0,
        min_absolute_count: int = 10,
    ) -> None:
        self._window_seconds = window_seconds
        self._hot_threshold_pct = hot_threshold_percentile
        self._min_absolute_count = min_absolute_count
        self._records: Dict[Tuple[int, str], ExpertAccessRecord] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Record                                                                 #
    # ------------------------------------------------------------------ #

    def record_dispatch(self, expert_id: int, node_id: str) -> None:
        """Increment the access counter for (expert_id, node_id)."""
        key = (expert_id, node_id)
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                rec = ExpertAccessRecord(expert_id=expert_id, node_id=node_id)
                self._records[key] = rec
            rec.touch()

    def record_bulk(self, assignments: List[Tuple[int, str]]) -> None:
        """Batch-record many (expert_id, node_id) dispatches at once."""
        if not assignments:
            return
        with self._lock:
            for expert_id, node_id in assignments:
                key = (expert_id, node_id)
                rec = self._records.get(key)
                if rec is None:
                    rec = ExpertAccessRecord(expert_id=expert_id, node_id=node_id)
                    self._records[key] = rec
                rec.touch()

    # ------------------------------------------------------------------ #
    # Query                                                                  #
    # ------------------------------------------------------------------ #

    def total_dispatches(self) -> int:
        """Total dispatch events in the current window."""
        with self._lock:
            return sum(r.count for r in self._records.values())

    def expert_counts(self) -> Dict[int, int]:
        """Return {expert_id: total_access_count} in current window."""
        result: Dict[int, int] = defaultdict(int)
        with self._lock:
            for (eid, _nid), rec in self._records.items():
                result[eid] += rec.count
        return dict(result)

    def per_node_counts(self) -> Dict[str, Dict[int, int]]:
        """Return {node_id: {expert_id: count}}."""
        result: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        with self._lock:
            for (eid, nid), rec in self._records.items():
                result[nid][eid] += rec.count
        return {k: dict(v) for k, v in result.items()}

    def hot_experts(
        self,
        top_k: int = 8,
        exclude_experts: Optional[List[int]] = None,
    ) -> List[HotSpot]:
        """
        Return the top-K hottest (expert_id, node_id) pairs.

        Sorted by access count descending.  Only pairs where the expert
        access count exceeds both the absolute minimum and the configured
        percentile threshold are included.

        Parameters
        ----------
        top_k : int
            Maximum number of hotspots to return.
        exclude_experts : list[int] | None
            Expert IDs to exclude (e.g. shared experts 0 and 1).
        """
        exclude_set = set(exclude_experts or [])
        with self._lock:
            candidates = [
                ((eid, nid), rec.count)
                for (eid, nid), rec in self._records.items()
                if rec.count >= self._min_absolute_count
                and eid not in exclude_set
            ]
            if not candidates:
                return []

            # Sort by count descending
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Compute percentile threshold
            counts = sorted(c[1] for c in candidates)
            idx = max(0, int(len(counts) * (self._hot_threshold_pct / 100.0)) - 1)
            threshold = counts[idx] if counts else 0

            # Filter + rank
            hot: List[HotSpot] = []
            for (eid, nid), cnt in candidates:
                if cnt >= threshold and len(hot) < top_k:
                    hot.append(HotSpot(
                        expert_id=eid,
                        node_id=nid,
                        access_count=cnt,
                        rank=len(hot) + 1,
                    ))
            return hot

    def most_accessed_experts(self, top_k: int = 8) -> List[Tuple[int, int]]:
        """Return the top-K expert IDs by total access count across all nodes."""
        ec = self.expert_counts()
        sorted_experts = sorted(ec.items(), key=lambda x: x[1], reverse=True)
        return sorted_experts[:top_k]

    # ------------------------------------------------------------------ #
    # Maintenance                                                            #
    # ------------------------------------------------------------------ #

    def prune(self, window_seconds: Optional[float] = None) -> int:
        """Remove records older than the sliding window. Returns count removed."""
        cutoff = time.time() - (window_seconds or self._window_seconds)
        removed = 0
        with self._lock:
            stale = [
                k for k, r in self._records.items()
                if r.last_access < cutoff
            ]
            for k in stale:
                del self._records[k]
            removed = len(stale)
        return removed

    def reset(self) -> None:
        """Clear all telemetry data."""
        with self._lock:
            self._records.clear()

    def snapshot(self) -> dict:
        """Return a JSON-safe summary of current telemetry state."""
        with self._lock:
            total = sum(r.count for r in self._records.values())
            per_expert: Dict[int, int] = defaultdict(int)
            per_node: Dict[str, int] = defaultdict(int)
            for (eid, nid), rec in self._records.items():
                per_expert[eid] += rec.count
                per_node[nid] += rec.count
            return {
                "total_dispatches": total,
                "unique_experts": len(per_expert),
                "unique_nodes": len(per_node),
                "top_experts": sorted(per_expert.items(), key=lambda x: x[1], reverse=True)[:10],
                "top_nodes": sorted(per_node.items(), key=lambda x: x[1], reverse=True)[:10],
                "window_seconds": self._window_seconds,
            }

    # ------------------------------------------------------------------ #
    # Telemetry endpoint (exposable via /api/monitor)                       #
    # ------------------------------------------------------------------ #

    def to_api_dict(self) -> dict:
        """Return a dict suitable for the /api/monitor endpoint."""
        return {
            "telemetry": {
                "total_dispatches": self.total_dispatches(),
                "hot_experts": [
                    {"expert_id": h.expert_id, "node_id": h.node_id,
                     "count": h.access_count, "rank": h.rank}
                    for h in self.hot_experts(top_k=8, exclude_experts=[0, 1])
                ],
                "top_experts": [
                    {"expert_id": eid, "total_count": cnt}
                    for eid, cnt in self.most_accessed_experts(top_k=8)
                ],
            }
        }