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
#   - Written from scratch; no derivation from Petals source.
#   - Integrates AstraDHT for dynamic node discovery rather than a static list.
#   - Supports N-node pipeline chains built from DHT peer advertisements.

"""
Pipeline Orchestrator — dynamic N-node chaining via DHT.

The orchestrator:
  1. Queries the DHT to discover all live peers.
  2. Builds a layer-coverage map and sorts peers into a pipeline order.
  3. Chains gRPC calls across the pipeline to process a full sequence.
  4. Retries failed hops with an exponential-backoff fallback to the next
     peer that covers the same layer range.

Usage::

    config = PipelineConfig(num_layers=61, max_retries=2)
    orch = PipelineOrchestrator(dht=my_dht, config=config)
    result = orch.run(token_ids=[1, 2, 3, 4])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..routing.geo_router import GeoAwareMoERouter, GeoRegion, REGIONS
from ..rpc.client import InferenceClient
from ..serialization.tensor_pack import (
    DEEPSEEK_V4_HIDDEN_DIM,
    DEEPSEEK_V4_NUM_LAYERS,
    TensorPacket,
)
from .dht import AstraDHT, DHTNodeRecord

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    num_layers: int = DEEPSEEK_V4_NUM_LAYERS
    hidden_dim: int = DEEPSEEK_V4_HIDDEN_DIM
    local_region: str = "local"
    rpc_timeout: float = 30.0
    max_retries: int = 2
    retry_base_delay: float = 1.0   # seconds; doubles each retry
    num_experts: int = 256
    top_k: int = 8
    num_shared_experts: int = 2


@dataclass
class HopResult:
    node_id: str
    address: str
    layer_start: int
    layer_end: int
    rtt_ms: float
    compute_ms: float
    success: bool
    error: str = ""


@dataclass
class PipelineRunResult:
    packet_id: str
    output: TensorPacket
    hops: List[HopResult] = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def num_hops(self) -> int:
        return len(self.hops)

    def summary(self) -> dict:
        return {
            "packet_id": self.packet_id[:8] + "…",
            "num_hops": self.num_hops,
            "total_ms": round(self.total_ms, 1),
            "hops": [
                {
                    "node": h.node_id,
                    "layers": f"{h.layer_start}:{h.layer_end}",
                    "rtt_ms": round(h.rtt_ms, 1),
                    "compute_ms": round(h.compute_ms, 1),
                    "ok": h.success,
                }
                for h in self.hops
            ],
        }


class PipelineOrchestrator:
    """
    Queries DHT → builds layer pipeline → chains gRPC hops.

    Pipeline construction:
      - Collect all live peers from DHT.
      - Sort by layer_start ascending.
      - Verify full layer coverage (0 → num_layers); raise if gaps exist.
      - For each layer segment, pick the geo-nearest peer.

    Retry logic per hop:
      - On failure, backoff and try the next peer covering the same layers.
      - If all peers for a segment fail, abort with a clear error.
    """

    def __init__(
        self,
        dht: AstraDHT,
        config: Optional[PipelineConfig] = None,
        local_region: Optional[str] = None,
    ) -> None:
        self._dht = dht
        self._cfg = config or PipelineConfig()
        self._local_region = REGIONS.get(
            local_region or self._cfg.local_region,
            GeoRegion(self._cfg.local_region, 0, 0),
        )
        self._router = GeoAwareMoERouter(
            local_region=self._cfg.local_region,
            num_experts=self._cfg.num_experts,
            top_k=self._cfg.top_k,
            num_shared=self._cfg.num_shared_experts,
        )

    # ------------------------------------------------------------------ #
    # Pipeline planning                                                     #
    # ------------------------------------------------------------------ #

    def _build_pipeline(self) -> List[List[DHTNodeRecord]]:
        """
        Returns an ordered list of peer groups.
        Each group covers a contiguous layer segment; peers are sorted
        nearest-first by RTT.

        Raises RuntimeError if coverage has gaps.
        """
        peers = self._dht.get_all_peers()
        if not peers:
            raise RuntimeError("No peers found in DHT — cannot build pipeline")

        # Group peers by their (layer_start, layer_end) segment
        segments: Dict[Tuple[int, int], List[DHTNodeRecord]] = {}
        for p in peers:
            key = (p.layer_start, p.layer_end)
            segments.setdefault(key, []).append(p)

        # Sort segments by layer_start
        ordered_segments = sorted(segments.items(), key=lambda x: x[0][0])

        # Verify contiguous coverage and deduplicate overlapping segments.
        # A segment is skipped if it is fully covered by an earlier segment.
        # A partially-overlapping segment is included but logged as a warning.
        cursor = 0
        for (ls, le), _ in ordered_segments:
            if ls > cursor:
                raise RuntimeError(
                    f"Pipeline coverage gap: layers {cursor}–{ls} have no peer"
                )
            cursor = max(cursor, le)
        if cursor < self._cfg.num_layers:
            raise RuntimeError(
                f"Pipeline coverage incomplete: layers {cursor}–{self._cfg.num_layers} uncovered"
            )

        # For each segment sort peers nearest-first; skip fully-covered segments
        pipeline: List[List[DHTNodeRecord]] = []
        coverage = 0
        for (ls, le), group in ordered_segments:
            if le <= coverage:
                log.debug(
                    "Skipping fully-overlapping segment %d:%d (already covered to %d)",
                    ls, le, coverage,
                )
                continue
            if ls < coverage:
                log.warning(
                    "Partially-overlapping segment %d:%d (covered to %d) — layers %d:%d will be computed twice",
                    ls, le, coverage, ls, coverage,
                )
            sorted_group = sorted(
                group,
                key=lambda p: self._local_region.rtt_ms(
                    GeoRegion(p.geo_region, *_region_coords(p.geo_region))
                ),
            )
            pipeline.append(sorted_group)
            coverage = max(coverage, le)
        return pipeline

    # ------------------------------------------------------------------ #
    # Inference                                                             #
    # ------------------------------------------------------------------ #

    def run(
        self,
        token_ids: List[int],
        use_kv_cache: bool = True,
    ) -> PipelineRunResult:
        """
        Full pipeline run: gate tokens → chain gRPC hops → return output.
        """
        t_start = time.perf_counter()
        pipeline = self._build_pipeline()

        # Initial packet
        packet = TensorPacket.make_input(
            token_ids,
            hidden_dim=self._cfg.hidden_dim,
            geo_region=self._cfg.local_region,
            src_node="orchestrator",
        )
        # MoE gate
        packet, _ = self._router.route(packet, layer_idx=0)
        result_id = packet.packet_id
        hops: List[HopResult] = []

        for peer_group in pipeline:
            segment_start = peer_group[0].layer_start
            segment_end = peer_group[0].layer_end
            packet = self._run_hop(
                packet, peer_group, segment_start, segment_end,
                use_kv_cache, hops
            )

        total_ms = (time.perf_counter() - t_start) * 1000.0
        return PipelineRunResult(
            packet_id=result_id,
            output=packet,
            hops=hops,
            total_ms=total_ms,
        )

    def _run_hop(
        self,
        packet: TensorPacket,
        peer_group: List[DHTNodeRecord],
        layer_start: int,
        layer_end: int,
        use_kv_cache: bool,
        hops: List[HopResult],
    ) -> TensorPacket:
        last_error = ""
        delay = self._cfg.retry_base_delay

        for attempt, peer in enumerate(peer_group[: self._cfg.max_retries + 1]):
            if attempt > 0:
                log.warning(
                    "Hop retry %d/%d for layers %d:%d via %s (after: %s)",
                    attempt, self._cfg.max_retries,
                    layer_start, layer_end, peer.node_id, last_error,
                )
                time.sleep(delay)
                delay *= 2

            t0 = time.perf_counter()
            try:
                with InferenceClient(
                    peer.address,
                    node_id="orchestrator",
                    timeout=self._cfg.rpc_timeout,
                ) as client:
                    out = client.run_layer(
                        packet,
                        layer_start=layer_start,
                        layer_end=layer_end,
                        use_kv_cache=use_kv_cache,
                    )
                rtt_ms = (time.perf_counter() - t0) * 1000.0
                compute_ms = float(out.metadata.get("compute_ms", 0))
                hops.append(HopResult(
                    node_id=peer.node_id,
                    address=peer.address,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    rtt_ms=rtt_ms,
                    compute_ms=compute_ms,
                    success=True,
                ))
                log.info(
                    "Hop OK: %s layers=%d:%d rtt=%.1fms compute=%.1fms",
                    peer.node_id, layer_start, layer_end, rtt_ms, compute_ms,
                )
                return out
            except Exception as exc:
                last_error = str(exc)
                rtt_ms = (time.perf_counter() - t0) * 1000.0
                hops.append(HopResult(
                    node_id=peer.node_id,
                    address=peer.address,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    rtt_ms=rtt_ms,
                    compute_ms=0.0,
                    success=False,
                    error=last_error,
                ))
                log.warning("Hop FAIL: %s — %s", peer.node_id, last_error)

        raise RuntimeError(
            f"All {len(peer_group)} peer(s) failed for layers "
            f"{layer_start}:{layer_end}. Last error: {last_error}"
        )

    def topology(self) -> dict:
        """Return a summary of the current pipeline topology from DHT."""
        peers = self._dht.get_all_peers()
        return {
            "num_peers": len(peers),
            "peers": [
                {
                    "node_id": p.node_id,
                    "address": p.address,
                    "layers": f"{p.layer_start}:{p.layer_end}",
                    "region": p.geo_region,
                    "experts": len(p.expert_shards),
                    "backend": p.backend,
                }
                for p in sorted(peers, key=lambda x: x.layer_start)
            ],
        }


# ─────────────────────────────────────────────────────────────────────────── #
# Helper                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _region_coords(region_name: str) -> Tuple[float, float]:
    r = REGIONS.get(region_name)
    if r:
        return r.lat, r.lon
    return 0.0, 0.0
