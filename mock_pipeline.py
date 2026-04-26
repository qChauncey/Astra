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
#
# MODIFICATIONS (Astra project):
#   - Written from scratch as the Phase 1 & 2 local simulation harness.
#   - Demonstrates "打包-传输-运算" (pack-transmit-compute) closed loop.

"""
Astra Mock Pipeline — Phase 1 & 2 Local Simulation
===================================================

Demonstrates the full pack → transmit → compute loop across two simulated
pipeline nodes running in separate threads on a single machine.

Topology:
  [Client]  ──gRPC──►  [Node A: layers 0-29]  ──gRPC──►  [Node B: layers 30-60]
                                                                    │
                                          ◄────────────────────────┘
                                         final hidden states returned

Each node runs a HeterogeneousEngine (CPU-only in this mock; swap
DeviceMap.for_16gb_gpu() when a CUDA device is available).

Usage:
    python mock_pipeline.py [--seq-len N] [--hidden-dim D] [--verbose]

Example:
    python mock_pipeline.py --seq-len 32 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time

import numpy as np

# ── Astra imports ──────────────────────────────────────────────────────────
from astra.inference.heterogeneous import DeviceMap, HeterogeneousEngine
from astra.inference.shared_expert_cache import ExpertWeights, SharedExpertCache
from astra.routing.geo_router import GeoAwareMoERouter, GeoRegion, NodeInfo, REGIONS
from astra.rpc.client import InferenceClient
from astra.rpc.server import InferenceServer
from astra.serialization.tensor_pack import (
    DEEPSEEK_V4_HIDDEN_DIM,
    DEEPSEEK_V4_NUM_LAYERS,
    TensorPacket,
    TensorSerializer,
)

log = logging.getLogger("astra.mock_pipeline")


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def _banner(msg: str) -> None:
    width = 72
    log.info("=" * width)
    log.info(msg.center(width))
    log.info("=" * width)


def _make_server(
    node_id: str,
    layer_start: int,
    layer_end: int,
    port: int,
    geo_region: str,
    hidden_dim: int,
) -> InferenceServer:
    """Construct an InferenceServer with shared-expert pinning."""
    dmap = DeviceMap.cpu_only()
    dmap.hidden_dim = hidden_dim

    server = InferenceServer(
        node_id=node_id,
        layer_start=layer_start,
        layer_end=layer_end,
        port=port,
        geo_region=geo_region,
        device_map=dmap,
        max_workers=4,
    )
    return server


def _start_server_thread(server: InferenceServer) -> threading.Thread:
    t = threading.Thread(target=server.start, daemon=True)
    t.start()
    time.sleep(0.3)   # allow gRPC to bind
    return t


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 1: Single-node heterogeneous inference (local, no network)             #
# ─────────────────────────────────────────────────────────────────────────── #

def run_phase1(seq_len: int, hidden_dim: int) -> None:
    _banner("Phase 1 — Local Heterogeneous Single-Node Inference")

    # ── Build engine ─────────────────────────────────────────────────────
    dmap = DeviceMap.cpu_only()
    dmap.hidden_dim = hidden_dim
    engine = HeterogeneousEngine.from_device_map(dmap)

    # Pin shared experts 0 & 1
    for sid in range(2):
        ew = ExpertWeights.mock(sid, hidden_dim=hidden_dim, intermediate_dim=hidden_dim // 4)
        engine.load_shared_experts([ew])

    # Load a few routed experts into cache
    for eid in range(2, 6):
        ew = ExpertWeights.mock(eid, hidden_dim=hidden_dim, intermediate_dim=hidden_dim // 4)
        engine.load_expert(ew)

    # ── Build router ─────────────────────────────────────────────────────
    router = GeoAwareMoERouter(local_region="local", num_experts=8, top_k=2, num_shared=2)

    # ── Create input packet ──────────────────────────────────────────────
    token_ids = list(range(seq_len))
    packet = TensorPacket.make_input(token_ids, hidden_dim=hidden_dim, geo_region="local")

    log.info("[Phase 1] Input packet: %s", packet)
    log.info("[Phase 1] Tensor shape=%s  dtype=%s  size=%.1f KB",
             packet.tensor.shape, packet.tensor.dtype, packet.byte_size() / 1024)

    # ── Step A: Serialize / deserialize (pack-unpack round-trip) ─────────
    log.info("\n── Step A: Tensor Pack / Unpack ──")
    t0 = time.perf_counter()
    wire_bytes = TensorSerializer.serialize(packet)
    t_ser = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    recovered = TensorSerializer.deserialize(wire_bytes)
    t_deser = (time.perf_counter() - t0) * 1000.0

    assert recovered.packet_id == packet.packet_id, "packet_id mismatch"
    assert recovered.tensor.shape == packet.tensor.shape, "shape mismatch"
    assert np.allclose(recovered.tensor.astype(np.float32),
                       packet.tensor.astype(np.float32)), "tensor data mismatch"

    log.info("  Wire size:     %.1f KB", len(wire_bytes) / 1024)
    log.info("  Serialize:     %.2f ms", t_ser)
    log.info("  Deserialize:   %.2f ms", t_deser)
    log.info("  Round-trip OK: ✓")

    # ── Step B: MoE routing (gate + dispatch plan) ────────────────────────
    log.info("\n── Step B: MoE Gate + Geo-Aware Dispatch ──")
    packet, plan = router.route(packet, layer_idx=0)
    log.info("  Selected experts shape: %s", packet.selected_experts.shape)
    log.info("  Dispatch plan: %d node(s), %d shared tokens",
             plan.num_nodes_used(), len(plan.shared_tokens))

    # ── Step C: Heterogeneous forward (attention GPU stub + MoE CPU) ──────
    log.info("\n── Step C: Heterogeneous Inference (layers 0–2) ──")
    packet.layer_end = 3
    t0 = time.perf_counter()
    out_packet = engine.forward(packet, layer_indices=[0, 1, 2])
    t_fwd = (time.perf_counter() - t0) * 1000.0

    log.info("  Output shape:    %s", out_packet.tensor.shape)
    log.info("  Compute time:    %.2f ms (3 layers)", t_fwd)
    log.info("  Engine stats:    %s",
             json.dumps(engine.stats(), indent=2))

    log.info("\n[Phase 1] COMPLETE ✓\n")
    return out_packet


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 2: Two-node LAN pipeline (local gRPC relay)                           #
# ─────────────────────────────────────────────────────────────────────────── #

def run_phase2(seq_len: int, hidden_dim: int) -> None:
    _banner("Phase 2 — Dual-Node gRPC Pipeline (LAN Simulation)")

    PORT_A = 50051
    PORT_B = 50052
    MID_LAYER = DEEPSEEK_V4_NUM_LAYERS // 2   # 30

    log.info("Topology:  client → Node-A (layers 0-29) → Node-B (layers 30-60)")

    # ── Start Node A ─────────────────────────────────────────────────────
    log.info("\n[Phase 2] Starting Node-A (port %d, layers 0-%d)…", PORT_A, MID_LAYER - 1)
    node_a = _make_server("node-A", 0, MID_LAYER, PORT_A, "us-west", hidden_dim)
    _start_server_thread(node_a)

    # ── Start Node B ─────────────────────────────────────────────────────
    log.info("[Phase 2] Starting Node-B (port %d, layers %d-60)…", PORT_B, MID_LAYER)
    node_b = _make_server("node-B", MID_LAYER, DEEPSEEK_V4_NUM_LAYERS, PORT_B, "us-east", hidden_dim)
    _start_server_thread(node_b)

    # ── Create input ─────────────────────────────────────────────────────
    token_ids = list(range(seq_len))
    packet = TensorPacket.make_input(token_ids, hidden_dim=hidden_dim, geo_region="local",
                                     src_node="client-0")

    # Run MoE gate so selected_experts is populated
    router = GeoAwareMoERouter(local_region="local", num_experts=8, top_k=2, num_shared=2)
    packet, _ = router.route(packet, layer_idx=0)

    log.info("\n── Step 1: Ping nodes ──")
    with InferenceClient(f"localhost:{PORT_A}", node_id="client-0") as ca:
        ping_a = ca.ping()
        log.info("  Node-A ping: %s", ping_a)

    with InferenceClient(f"localhost:{PORT_B}", node_id="client-0") as cb:
        ping_b = cb.ping()
        log.info("  Node-B ping: %s", ping_b)

    log.info("\n── Step 2: Relay  client → Node-A ──")
    t_total = time.perf_counter()
    packet.dst_node = "node-A"

    with InferenceClient(f"localhost:{PORT_A}", node_id="client-0") as client_a:
        t0 = time.perf_counter()
        mid_packet = client_a.run_layer(packet, layer_start=0, layer_end=MID_LAYER)
        rtt_a = (time.perf_counter() - t0) * 1000.0

    log.info("  Received from Node-A: %s  (RTT %.1f ms)", mid_packet, rtt_a)

    log.info("\n── Step 3: Relay  Node-A → Node-B ──")
    mid_packet.dst_node = "node-B"

    with InferenceClient(f"localhost:{PORT_B}", node_id="client-0") as client_b:
        t0 = time.perf_counter()
        final_packet = client_b.run_layer(mid_packet, layer_start=MID_LAYER,
                                          layer_end=DEEPSEEK_V4_NUM_LAYERS)
        rtt_b = (time.perf_counter() - t0) * 1000.0

    t_total_ms = (time.perf_counter() - t_total) * 1000.0
    log.info("  Received from Node-B: %s  (RTT %.1f ms)", final_packet, rtt_b)

    log.info("\n── Summary ──")
    log.info("  Input  shape:  %s", packet.tensor.shape)
    log.info("  Output shape:  %s", final_packet.tensor.shape)
    log.info("  Node-A RTT:    %.1f ms", rtt_a)
    log.info("  Node-B RTT:    %.1f ms", rtt_b)
    log.info("  Total wall:    %.1f ms", t_total_ms)
    log.info("  Pipeline OK:   ✓")

    node_a.stop(grace=1.0)
    node_b.stop(grace=1.0)
    log.info("\n[Phase 2] COMPLETE ✓\n")


# ─────────────────────────────────────────────────────────────────────────── #
# Entry point                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Astra mock pipeline — Phase 1 & 2 local simulation"
    )
    parser.add_argument(
        "--seq-len", type=int, default=16,
        help="Number of input tokens (default: 16)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="Hidden dimension size; use 7168 for real DeepSeek-V4 (default: 256 for speed)"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 12], default=12,
        help="Which phases to run: 1, 2, or 12 (both, default)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy gRPC internals unless verbose
    if not args.verbose:
        logging.getLogger("grpc").setLevel(logging.WARNING)

    log.info("Astra Mock Pipeline")
    log.info("  seq_len    = %d", args.seq_len)
    log.info("  hidden_dim = %d", args.hidden_dim)
    log.info("  phases     = %s", args.phase)

    if args.phase in (1, 12):
        run_phase1(args.seq_len, args.hidden_dim)

    if args.phase in (2, 12):
        run_phase2(args.seq_len, args.hidden_dim)

    _banner("All requested phases completed successfully")


if __name__ == "__main__":
    main()
