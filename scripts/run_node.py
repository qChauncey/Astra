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
Astra Node Launch CLI.

Starts a single Astra inference node:
  - Binds a gRPC InferenceServer on the specified port.
  - Announces itself to the DHT (in-process store for local runs; replace
    with real hivemind for multi-machine deployments).
  - Optionally starts the OpenAI-compatible API gateway on a separate port.

Usage::

    # Start a node that serves layers 0-30
    python scripts/run_node.py \\
        --node-id node-A \\
        --port 50051 \\
        --layer-start 0 \\
        --layer-end 30 \\
        --region us-west \\
        --experts 0-127

    # Start with API gateway
    python scripts/run_node.py \\
        --node-id gateway \\
        --port 50051 \\
        --layer-start 0 \\
        --layer-end 61 \\
        --api-port 8080

Multi-node example (two terminals on the same machine):
    Terminal 1: python scripts/run_node.py --node-id A --port 50051 --layer-start 0  --layer-end 30
    Terminal 2: python scripts/run_node.py --node-id B --port 50052 --layer-start 30 --layer-end 61
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from typing import List, Optional

# Ensure project root is on path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astra.inference.heterogeneous import DeviceMap
from astra.inference.shared_expert_cache import ExpertWeights
from astra.network.dht import AstraDHT, DHTNodeRecord
from astra.rpc.server import InferenceServer

log = logging.getLogger("astra.run_node")


def _parse_expert_range(spec: str) -> List[int]:
    """
    Parse expert specification:
      "0-127"   → [0, 1, ..., 127]
      "0,1,2"   → [0, 1, 2]
      "all"     → [0, 1, ..., 255]
    """
    if spec.lower() == "all":
        return list(range(256))
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def _build_device_map(args: argparse.Namespace) -> DeviceMap:
    if args.gpu:
        return DeviceMap.for_16gb_gpu()
    return DeviceMap.cpu_only()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch an Astra pipeline node",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--node-id", default="node-0", help="Unique peer identifier")
    parser.add_argument("--port", type=int, default=50051, help="gRPC listen port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (for advertise)")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=30)
    parser.add_argument(
        "--region",
        default="local",
        choices=["local", "us-west", "us-east", "eu-central", "ap-east"],
        help="Geographic region tag",
    )
    parser.add_argument(
        "--experts",
        default="all",
        help='Expert shards: "all", "0-127", or "0,1,2"',
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU DeviceMap (requires CUDA + torch)",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=0,
        help="If >0, also start the OpenAI-compatible API gateway on this port",
    )
    parser.add_argument("--workers", type=int, default=4, help="gRPC thread pool size")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  [%(name)s]  %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        logging.getLogger("grpc").setLevel(logging.WARNING)

    expert_shards = _parse_expert_range(args.experts)
    dmap = _build_device_map(args)
    dmap.hidden_dim = 256  # use small dim for mock; 7168 for real model

    log.info("=" * 60)
    log.info("  Astra Node: %s", args.node_id)
    log.info("  Port:       %d", args.port)
    log.info("  Layers:     %d – %d", args.layer_start, args.layer_end)
    log.info("  Region:     %s", args.region)
    log.info("  Experts:    %d shards", len(expert_shards))
    log.info("  Backend:    %s", "GPU (KTransformers)" if args.gpu else "CPU (numpy stub)")
    log.info("=" * 60)

    # ── Start gRPC server ─────────────────────────────────────────────
    server = InferenceServer(
        node_id=args.node_id,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        port=args.port,
        geo_region=args.region,
        expert_shards=expert_shards,
        device_map=dmap,
        max_workers=args.workers,
    )
    server.start()

    # ── Announce to DHT ───────────────────────────────────────────────
    dht = AstraDHT(node_id=args.node_id)
    advertise_addr = f"{args.host}:{args.port}".replace("0.0.0.0", "127.0.0.1")
    record = DHTNodeRecord(
        node_id=args.node_id,
        address=advertise_addr,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        expert_shards=expert_shards[:32],  # truncate for DHT store
        geo_region=args.region,
        backend="numpy_stub",
    )
    dht.announce(record, ttl=120.0)
    log.info("Announced to DHT: %s → %s", args.node_id, advertise_addr)

    # ── Optional API gateway ──────────────────────────────────────────
    api_thread: Optional[threading.Thread] = None
    if args.api_port > 0:
        try:
            import uvicorn
            from astra.api.openai_compat import create_app
            from astra.network.orchestrator import PipelineConfig

            api_app = create_app(
                dht=dht,
                pipeline_config=PipelineConfig(
                    num_layers=args.layer_end - args.layer_start,
                    hidden_dim=dmap.hidden_dim,
                ),
            )

            def _run_api() -> None:
                uvicorn.run(api_app, host="0.0.0.0", port=args.api_port, log_level="warning")

            api_thread = threading.Thread(target=_run_api, daemon=True)
            api_thread.start()
            log.info("OpenAI API gateway: http://0.0.0.0:%d/v1", args.api_port)
        except ImportError as exc:
            log.warning("Could not start API gateway: %s", exc)

    # ── Graceful shutdown ─────────────────────────────────────────────
    stop_event = threading.Event()

    def _handle_signal(*_) -> None:
        log.info("Shutdown signal received…")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("Node running. Press Ctrl-C to stop.")
    try:
        while not stop_event.is_set():
            time.sleep(1.0)
    finally:
        dht.revoke()
        server.stop(grace=2.0)
        log.info("Node %s stopped cleanly.", args.node_id)


if __name__ == "__main__":
    main()
