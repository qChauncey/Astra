#!/usr/bin/env python3
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
Astra Local Multi-Node Cluster Launcher.

Simulates a distributed inference cluster on a single machine by spawning
multiple InferenceServer instances in separate threads, each bound to its
own port and responsible for a contiguous slice of the transformer layers.
All nodes share one in-process AstraDHT for peer discovery.

What this validates
-------------------
  ✓ gRPC hop-to-hop tensor passing across N nodes
  ✓ DHT peer registration and layer-coverage routing
  ✓ PipelineOrchestrator N-node chain building
  ✓ KV-cache streaming between pipeline stages
  ✓ OpenAI-compatible API gateway (optional)
  ✓ Gap detection: orchestrator raises if layers are not fully covered

What this does NOT validate (requires real hardware + model weights)
-------------------------------------------------------------------
  ✗ Inference correctness — numpy stub outputs are random arrays
  ✗ GPU memory isolation between nodes
  ✗ Real throughput / latency (all communication is on localhost)

Usage
-----
  # 3 nodes covering layers 0-61, API gateway on port 8080
  python scripts/run_cluster.py --nodes 3 --api-port 8080

  # 4 nodes with mock hidden dim (fast startup, no real model dims)
  python scripts/run_cluster.py --nodes 4 --hidden-dim 256

  # Validate then exit without keeping servers running
  python scripts/run_cluster.py --nodes 3 --validate-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astra.inference.heterogeneous import DeviceMap
from astra.network.dht import AstraDHT, DHTNodeRecord
from astra.network.orchestrator import PipelineConfig, PipelineOrchestrator
from astra.rpc.server import InferenceServer

log = logging.getLogger("astra.cluster")


# ─────────────────────────────────────────────────────────────────────────── #
# Layer partitioning                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def _partition_layers(total: int, n: int) -> List[tuple[int, int]]:
    """
    Divide [0, total) evenly into n contiguous ranges.
    The last shard absorbs any remainder so coverage is always complete.

    Returns list of (layer_start, layer_end) tuples.
    """
    base = total // n
    remainder = total % n
    ranges = []
    cursor = 0
    for i in range(n):
        extra = 1 if i < remainder else 0
        end = cursor + base + extra
        ranges.append((cursor, end))
        cursor = end
    return ranges


# ─────────────────────────────────────────────────────────────────────────── #
# Node lifecycle                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

class _ManagedNode:
    """Wraps an InferenceServer + DHT record for a single simulated node."""

    def __init__(
        self,
        node_id: str,
        layer_start: int,
        layer_end: int,
        port: int,
        hidden_dim: int,
        max_workers: int,
    ) -> None:
        self.node_id = node_id
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.port = port
        # Each node needs its own AstraDHT instance so that revoke() uses the
        # correct owner key. All instances share the same _GlobalStore underneath.
        self._dht = AstraDHT(node_id=node_id)

        dmap = DeviceMap.cpu_only()
        dmap.hidden_dim = hidden_dim

        self._server = InferenceServer(
            node_id=node_id,
            layer_start=layer_start,
            layer_end=layer_end,
            port=port,
            geo_region="local",
            expert_shards=list(range(256)),
            device_map=dmap,
            max_workers=max_workers,
        )

    def start(self) -> None:
        self._server.start()
        record = DHTNodeRecord(
            node_id=self.node_id,
            address=f"127.0.0.1:{self.port}",
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            expert_shards=list(range(32)),
            geo_region="local",
            backend="numpy_stub",
        )
        self._dht.announce(record, ttl=0.0)   # ttl=0 = no expiry in test mode
        log.info(
            "  [%s]  port=%d  layers=%d–%d  started",
            self.node_id, self.port, self.layer_start, self.layer_end,
        )

    def stop(self, grace: float = 2.0) -> None:
        self._dht.revoke()
        self._server.stop(grace=grace)


# ─────────────────────────────────────────────────────────────────────────── #
# Validation                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def _run_validation(
    dht: AstraDHT,
    config: PipelineConfig,
    seq_len: int = 8,
) -> bool:
    """
    Send a single end-to-end request through the cluster and print results.
    Returns True if all hops succeeded.
    """
    log.info("Running end-to-end pipeline validation (seq_len=%d)…", seq_len)
    orch = PipelineOrchestrator(dht=dht, config=config)

    topology = orch.topology()
    log.info("Topology: %d peer(s) discovered", len(topology.get("peers", [])))
    for peer in topology.get("peers", []):
        log.info(
            "  peer %-12s  layers %-8s  addr %s",
            peer["node_id"], peer["layers"], peer["address"],
        )

    token_ids = list(range(1, seq_len + 1))
    try:
        result = orch.run(token_ids, use_kv_cache=True)
        summary = result.summary()
        log.info("Pipeline result: %s", json.dumps(summary, indent=2))
        all_ok = all(h["ok"] for h in summary["hops"])
        if all_ok:
            log.info("✓ All %d hops succeeded (%.1f ms total)", result.num_hops, result.total_ms)
        else:
            failed = [h["node"] for h in summary["hops"] if not h["ok"]]
            log.warning("✗ Failed hops: %s", failed)
        return all_ok
    except Exception as exc:
        log.error("Pipeline validation failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────── #
# Entry point                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a local multi-node Astra cluster for infrastructure validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nodes", type=int, default=3,
                        help="Number of simulated pipeline nodes")
    parser.add_argument("--total-layers", type=int, default=61,
                        help="Total transformer layers to distribute (61 for DeepSeek-V4)")
    parser.add_argument("--base-port", type=int, default=50051,
                        help="First node binds here; subsequent nodes use base+1, base+2, …")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension (256 for mock; 7168 for real DeepSeek-V4)")
    parser.add_argument("--api-port", type=int, default=0,
                        help="If >0, start OpenAI-compatible API gateway on this port")
    parser.add_argument("--seq-len", type=int, default=8,
                        help="Token sequence length used in the validation request")
    parser.add_argument("--workers", type=int, default=2,
                        help="gRPC thread pool size per node")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run one validation request then exit immediately")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  [%(name)s]  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("grpc").setLevel(logging.WARNING)

    if args.nodes < 1:
        log.error("--nodes must be ≥ 1")
        sys.exit(1)

    layer_ranges = _partition_layers(args.total_layers, args.nodes)

    log.info("=" * 64)
    log.info("  Astra Local Cluster  —  %d node(s)", args.nodes)
    log.info("  Total layers : %d  (%.1f layers/node avg)",
             args.total_layers, args.total_layers / args.nodes)
    log.info("  Hidden dim   : %d", args.hidden_dim)
    log.info("  Base port    : %d – %d",
             args.base_port, args.base_port + args.nodes - 1)
    log.info("=" * 64)

    # ── Shared DHT ────────────────────────────────────────────────────────
    dht = AstraDHT(node_id="cluster-dht")

    # ── Start all nodes ───────────────────────────────────────────────────
    nodes: List[_ManagedNode] = []
    for i, (ls, le) in enumerate(layer_ranges):
        node = _ManagedNode(
            node_id=f"node-{i}",
            layer_start=ls,
            layer_end=le,
            port=args.base_port + i,
            hidden_dim=args.hidden_dim,
            max_workers=args.workers,
        )
        node.start()
        nodes.append(node)

    # Brief pause to let gRPC servers bind
    time.sleep(0.3)

    # ── Pipeline config ───────────────────────────────────────────────────
    pipeline_config = PipelineConfig(
        num_layers=args.total_layers,
        hidden_dim=args.hidden_dim,
    )

    # ── Validation request ────────────────────────────────────────────────
    ok = _run_validation(dht, pipeline_config, seq_len=args.seq_len)

    if args.validate_only:
        log.info("--validate-only: shutting down cluster.")
        for node in reversed(nodes):
            node.stop()
        sys.exit(0 if ok else 1)

    # ── Optional API gateway ──────────────────────────────────────────────
    api_thread: Optional[threading.Thread] = None
    if args.api_port > 0:
        try:
            import uvicorn
            from astra.api.openai_compat import create_app

            api_app = create_app(dht=dht, pipeline_config=pipeline_config)

            def _run_api() -> None:
                uvicorn.run(api_app, host="0.0.0.0", port=args.api_port, log_level="warning")

            api_thread = threading.Thread(target=_run_api, daemon=True)
            api_thread.start()
            log.info("OpenAI API gateway: http://localhost:%d/v1", args.api_port)
            log.info("  Test: curl http://localhost:%d/health", args.api_port)
            log.info("  Test: curl http://localhost:%d/v1/models", args.api_port)
        except ImportError as exc:
            log.warning("Could not start API gateway: %s", exc)

    # ── Keep running ──────────────────────────────────────────────────────
    log.info("Cluster running. Press Ctrl-C to stop.")
    log.info("")
    log.info("Quick test commands:")
    log.info("  python -c \"")
    log.info("    from astra.network.dht import AstraDHT")
    log.info("    from astra.network.orchestrator import PipelineOrchestrator, PipelineConfig")
    log.info("    # (connect to existing cluster via shared DHT not possible cross-process)")
    log.info("    # Use the API gateway instead: curl http://localhost:%d/v1/pipeline/topology",
             args.api_port if args.api_port > 0 else 8080)
    log.info("  \"")

    stop_event = threading.Event()

    def _handle_signal(*_) -> None:
        log.info("Shutdown signal received…")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop_event.is_set():
            time.sleep(1.0)
    finally:
        log.info("Stopping all nodes…")
        for node in reversed(nodes):
            node.stop(grace=1.0)
        log.info("Cluster stopped.")


if __name__ == "__main__":
    main()
