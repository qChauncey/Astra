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
#   - Phase 5: added --tls (mTLS) and --hivemind (real P2P DHT) flags.

"""
Astra Mock Pipeline — Phase 1, 2, 4, 5 local & multi-node simulation
=====================================================================

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

Multi-machine (Phase 5):
    # Bootstrap node
    python mock_pipeline.py --node-id node-1 --layer-start 0 --layer-end 30 \
        --port 50051 --hivemind --dht-port 1337 --tls

    # Worker node (connects to bootstrap via DHT)
    python mock_pipeline.py --node-id node-2 --layer-start 30 --layer-end 61 \
        --port 50052 --hivemind --dht-port 1338 \
        --peers /ip4/192.168.1.10/tcp/1337/p2p/<BOOTSTRAP-PEER-ID> --tls

Example:
    python mock_pipeline.py --seq-len 32 --verbose
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from typing import List, Optional

import numpy as np

# ── Astra imports ──────────────────────────────────────────────────────────
from astra.inference.differential_privacy import LayerDPInjector
from astra.inference.heterogeneous import DeviceMap, HeterogeneousEngine
from astra.inference.shared_expert_cache import ExpertWeights, SharedExpertCache
from astra.network.hivemind_bridge import create_dht
from astra.routing.geo_router import GeoAwareMoERouter, GeoRegion, NodeInfo, REGIONS
from astra.rpc.client import InferenceClient
from astra.rpc.server import InferenceServer
from astra.rpc.tls import (
    TLSConfig,
    generate_self_signed_cert_bundle,
)
from astra.serialization.tensor_pack import (
    DEEPSEEK_V4_HIDDEN_DIM,
    DEEPSEEK_V4_NUM_LAYERS,
    TensorPacket,
    TensorSerializer,
)
from astra.tee import get_tee_backend, list_available_backends

log = logging.getLogger("astra.mock_pipeline")

_TEMP_CERT_DIR: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def _banner(msg: str) -> None:
    width = 72
    log.info("=" * width)
    log.info(msg.center(width))
    log.info("=" * width)


def _cleanup_temp_certs() -> None:
    global _TEMP_CERT_DIR
    if _TEMP_CERT_DIR and os.path.isdir(_TEMP_CERT_DIR):
        shutil.rmtree(_TEMP_CERT_DIR, ignore_errors=True)
        _TEMP_CERT_DIR = None


def _make_tls_config(node_id: str) -> TLSConfig:
    """Generate temporary self-signed certs for *node_id*, return TLSConfig."""
    global _TEMP_CERT_DIR
    if not _TEMP_CERT_DIR:
        _TEMP_CERT_DIR = tempfile.mkdtemp(prefix="astra-tls-")
        atexit.register(_cleanup_temp_certs)
        log.info("TLS cert dir: %s", _TEMP_CERT_DIR)

    # Generate node cert
    bundle = generate_self_signed_cert_bundle(node_id=node_id, days_valid=7)
    cert_path, key_path = bundle.write(_TEMP_CERT_DIR)

    # Generate shared CA cert (once)
    ca_cert_path = os.path.join(_TEMP_CERT_DIR, "ca.cert.pem")
    if not os.path.exists(ca_cert_path):
        ca = generate_self_signed_cert_bundle(node_id="astra-mock-ca", days_valid=7)
        ca.write(_TEMP_CERT_DIR)
        # rename ca cert to ca.cert.pem
        ca_orig = os.path.join(_TEMP_CERT_DIR, "astra-mock-ca.cert.pem")
        if os.path.exists(ca_orig):
            os.rename(ca_orig, ca_cert_path)

    return TLSConfig(
        enabled=True,
        cert_path=cert_path,
        key_path=key_path,
        ca_cert_path=ca_cert_path,
    )


def _make_server(
    node_id: str,
    layer_start: int,
    layer_end: int,
    port: int,
    geo_region: str,
    hidden_dim: int,
    tls_config: Optional[TLSConfig] = None,
) -> InferenceServer:
    """Construct an InferenceServer with shared-expert pinning."""
    dmap = DeviceMap.cpu_only()

    server = InferenceServer(
        node_id=node_id,
        layer_start=layer_start,
        layer_end=layer_end,
        port=port,
        geo_region=geo_region,
        device_map=dmap,
        max_workers=4,
        tls_config=tls_config,
    )
    return server


def _start_server_thread(server: InferenceServer) -> threading.Thread:
    t = threading.Thread(target=server.start, daemon=True)
    t.start()
    time.sleep(0.3)  # allow gRPC to bind
    return t


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 1: Single-node heterogeneous inference (local, no network)             #
# ─────────────────────────────────────────────────────────────────────────── #

def run_phase1(seq_len: int, hidden_dim: int) -> None:
    _banner("Phase 1 — Local Heterogeneous Single-Node Inference")

    # ── Build engine ─────────────────────────────────────────────────────
    dmap = DeviceMap.cpu_only()
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
    log.info("  Round-trip OK: [PASS]")

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

    log.info("\n[Phase 1] COMPLETE [PASS]\n")
    return out_packet


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 2: Two-node LAN pipeline (local gRPC relay)                           #
# ─────────────────────────────────────────────────────────────────────────── #

def run_phase2(seq_len: int, hidden_dim: int, use_tls: bool = False) -> None:
    banner = "Phase 2 — Dual-Node gRPC Pipeline (LAN Simulation)"
    if use_tls:
        banner += " [mTLS enabled]"
    _banner(banner)

    PORT_A = 50051
    PORT_B = 50052
    MID_LAYER = DEEPSEEK_V4_NUM_LAYERS // 2  # 30

    tls_a = _make_tls_config("node-A") if use_tls else None
    tls_b = _make_tls_config("node-B") if use_tls else None

    log.info("Topology:  client → Node-A (layers 0-29) → Node-B (layers 30-60)")

    # ── Start Node A ─────────────────────────────────────────────────────
    log.info("\n[Phase 2] Starting Node-A (port %d, layers 0-%d)…", PORT_A, MID_LAYER - 1)
    node_a = _make_server("node-A", 0, MID_LAYER, PORT_A, "us-west", hidden_dim,
                          tls_config=tls_a)
    _start_server_thread(node_a)

    # ── Start Node B ─────────────────────────────────────────────────────
    log.info("[Phase 2] Starting Node-B (port %d, layers %d-60)…", PORT_B, MID_LAYER)
    node_b = _make_server("node-B", MID_LAYER, DEEPSEEK_V4_NUM_LAYERS, PORT_B, "us-east", hidden_dim,
                          tls_config=tls_b)
    _start_server_thread(node_b)

    # ── Create input ─────────────────────────────────────────────────────
    token_ids = list(range(seq_len))
    packet = TensorPacket.make_input(token_ids, hidden_dim=hidden_dim, geo_region="local",
                                     src_node="client-0")

    # Run MoE gate so selected_experts is populated
    router = GeoAwareMoERouter(local_region="local", num_experts=8, top_k=2, num_shared=2)
    packet, _ = router.route(packet, layer_idx=0)

    log.info("\n── Step 1: Ping nodes ──")
    with InferenceClient(f"localhost:{PORT_A}", node_id="client-0",
                         tls_config=tls_a) as ca:
        ping_a = ca.ping()
        log.info("  Node-A ping: %s", ping_a)

    with InferenceClient(f"localhost:{PORT_B}", node_id="client-0",
                         tls_config=tls_b) as cb:
        ping_b = cb.ping()
        log.info("  Node-B ping: %s", ping_b)

    log.info("\n── Step 2: Relay  client → Node-A ──")
    t_total = time.perf_counter()
    packet.dst_node = "node-A"

    with InferenceClient(f"localhost:{PORT_A}", node_id="client-0",
                         tls_config=tls_a) as client_a:
        t0 = time.perf_counter()
        mid_packet = client_a.run_layer(packet, layer_start=0, layer_end=MID_LAYER)
        rtt_a = (time.perf_counter() - t0) * 1000.0

    log.info("  Received from Node-A: %s  (RTT %.1f ms)", mid_packet, rtt_a)

    log.info("\n── Step 3: Relay  Node-A → Node-B ──")
    mid_packet.dst_node = "node-B"

    with InferenceClient(f"localhost:{PORT_B}", node_id="client-0",
                         tls_config=tls_b) as client_b:
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
    log.info("  Pipeline OK:   [PASS]")

    node_a.stop(grace=1.0)
    node_b.stop(grace=1.0)
    log.info("\n[Phase 2] COMPLETE [PASS]\n")


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 4: Differential Privacy + TEE smoke test                               #
# ─────────────────────────────────────────────────────────────────────────── #

def run_phase4(seq_len: int, hidden_dim: int) -> None:
    _banner("Phase 4 — Differential Privacy & TEE (Security Hardening)")

    # ── 4.1 Differential Privacy smoke test ──────────────────────────────
    log.info("\n── 4.1 Differential Privacy (Hidden-State Noise Injection) ──")

    injector = LayerDPInjector(epsilon=1.0, num_layers=61)
    log.info("  DP injector created: ε_target=%.2f  layers=%d  mechanism=%s",
             injector.controller._epsilon, injector._num_layers, injector.controller._mechanism)

    # Simulate a forward pass through 2 layers with dummy hidden states
    hidden = np.random.randn(seq_len, hidden_dim).astype(np.float32)
    for layer_idx in range(2):
        noisy_hidden = injector(hidden, layer_idx)
        before_norm = float(np.linalg.norm(hidden))
        after_norm = float(np.linalg.norm(noisy_hidden))
        utility = before_norm / max(after_norm, 1e-8)
        log.info("  Layer %d: noise injected  ε_per_layer=%.6f  utility_signal_ratio=%.4f",
                 layer_idx, injector.eps_per_layer, utility)

    stats = injector.stats()
    log.info("  DP stats: %s", json.dumps(stats, indent=2))
    assert stats["epsilon_consumed"] > 0, "no epsilon consumed — DP not applied"
    assert stats["steps"] == 2, "expected 2 DP steps"
    log.info("  DP smoke test: PASS")

    # ── 4.2 TEE backends ─────────────────────────────────────────────────
    log.info("\n── 4.2 TEE (Trusted Execution Environment) ──")

    available = list_available_backends()
    log.info("  Available TEE backends: %s", available)

    # On non-TEE hardware, get_tee_backend() returns None
    backend = get_tee_backend()
    if backend is None:
        log.info("  (No TEE hardware detected -- running in software simulation mode)")
    else:
        status = backend.status()
        log.info("  Active backend: %s  status=%s", backend.name, status.name)

        # Attestation
        report = backend.attest(b"astra-phase4-smoke-test")
        log.info("  Attestation report: id=%s  is_debug=%s  measurement=%s",
                 report.report_id[:12] + "..." if len(report.report_id) > 12 else report.report_id,
                 report.is_debug,
                 report.measurement[:20] + "..." if report.measurement and len(report.measurement) > 20 else report.measurement)

        # Seal / unseal round-trip
        plaintext = b"Phase-4 TEE smoke test payload"
        sealed = backend.seal(plaintext)
        unsealed = backend.unseal(sealed)
        assert unsealed == plaintext, "TEE seal/unseal round-trip failed"
        log.info("  Seal/unseal round-trip: %d bytes [PASS]", len(plaintext))

    log.info("\n[Phase 4] COMPLETE\n")


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 5: mTLS + Hivemind DHT multi-node pipeline                             #
# ─────────────────────────────────────────────────────────────────────────── #

def run_phase5(
    seq_len: int,
    hidden_dim: int,
    node_id: str,
    layer_start: int,
    layer_end: int,
    port: int,
    use_hivemind: bool,
    dht_port: int,
    initial_peers: List[str],
    use_tls: bool,
) -> None:
    _banner(f"Phase 5 — Multi-Node Pipeline (node={node_id}, layers={layer_start}-{layer_end})")

    tls_config = _make_tls_config(node_id) if use_tls else None

    # ── 5.1 DHT peer discovery ───────────────────────────────────────────
    log.info("\n── 5.1 DHT Peer Discovery ──")
    if use_hivemind:
        log.info("  Creating HivemindDHT: node=%s port=%d peers=%s",
                 node_id, dht_port, initial_peers)
    else:
        log.info("  Creating AstraDHT (in-memory, local-only)")

    dht = create_dht(
        node_id=node_id,
        use_hivemind=use_hivemind,
        host_addr="0.0.0.0",
        port=dht_port,
        initial_peers=initial_peers,
    )
    log.info("  DHT created: node_id=%s backend=%s", node_id, type(dht).__name__)

    # Announce self to DHT
    from astra.network.dht import DHTNodeRecord
    record = DHTNodeRecord(
        node_id=node_id,
        address=f"localhost:{port}",
        layer_start=layer_start,
        layer_end=layer_end,
        expert_shards=list(range(8)),
        geo_region="local",
    )
    dht.announce(record, ttl=120)
    log.info("  Self-announced to DHT: %s", node_id)

    # Discover peers
    peers = dht.get_all_peers()
    log.info("  Discovered %d peer(s): %s", len(peers),
             [p.node_id for p in peers if p.node_id != node_id])

    # ── 5.2 Start local inference server ─────────────────────────────────
    log.info("\n── 5.2 Inference Server ──")
    server = _make_server(node_id, layer_start, layer_end, port, "local", hidden_dim,
                          tls_config=tls_config)
    _start_server_thread(server)
    log.info("  Server listening on port %d (TLS=%s)", port, use_tls)

    # ── 5.3 Run local inference ──────────────────────────────────────────
    if layer_start == 0:
        log.info("\n── 5.3 Local Inference (entry node) ──")
        token_ids = list(range(seq_len))
        packet = TensorPacket.make_input(token_ids, hidden_dim=hidden_dim, geo_region="local",
                                         src_node=node_id)
        packet.layer_end = layer_end

        # Run MoE gate
        router = GeoAwareMoERouter(local_region="local", num_experts=8, top_k=2, num_shared=2)
        packet, _ = router.route(packet, layer_idx=0)

        # Forward through local layers
        with InferenceClient(f"localhost:{port}", node_id=node_id,
                             tls_config=tls_config) as client:
            ping = client.ping()
            log.info("  Self-ping: %s", ping)

            t0 = time.perf_counter()
            result = client.run_layer(packet, layer_start=layer_start, layer_end=layer_end)
            rtt = (time.perf_counter() - t0) * 1000.0
            log.info("  Local inference: output shape=%s  RTT=%.1f ms", result.tensor.shape, rtt)

    log.info("\n[Phase 5] Node '%s' running (layers %d-%d). Press Ctrl+C to stop.\n",
             node_id, layer_start, layer_end)

    # Keep running for multi-machine scenarios
    try:
        while True:
            time.sleep(10)
            # Re-announce heartbeat
            dht.announce(record, ttl=120)
    except KeyboardInterrupt:
        log.info("Shutting down node %s…", node_id)

    dht.revoke(node_id)
    server.stop(grace=1.0)
    log.info("[Phase 5] Node '%s' shut down.\n", node_id)


# ─────────────────────────────────────────────────────────────────────────── #
# Entry point                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Astra mock pipeline — Phase 1, 2, 4, 5 local & multi-node simulation"
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
        "--phase", type=int, choices=[1, 2, 4, 5, 12, 124, 1245], default=1245,
        help="Which phases to run: 1, 2, 4, 5, 12 (1+2), 124 (1+2+4), or 1245 (all, default)"
    )

    # Phase 5 multi-node options
    parser.add_argument(
        "--node-id", type=str, default="node-0",
        help="This node's unique identifier (Phase 5 multi-machine)"
    )
    parser.add_argument(
        "--layer-start", type=int, default=0,
        help="First transformer layer this node serves (default: 0)"
    )
    parser.add_argument(
        "--layer-end", type=int, default=61,
        help="First layer of next node (exclusive, default: 61 = all layers)"
    )
    parser.add_argument(
        "--port", type=int, default=50051,
        help="gRPC port for this node (default: 50051)"
    )

    # Phase 5 networking flags
    parser.add_argument(
        "--tls", action="store_true",
        help="Enable mTLS with auto-generated self-signed certificates"
    )
    parser.add_argument(
        "--hivemind", action="store_true",
        help="Use Hivemind real P2P DHT for peer discovery (requires pip install hivemind)"
    )
    parser.add_argument(
        "--dht-port", type=int, default=1337,
        help="TCP port for Hivemind DHT listener (default: 1337)"
    )
    parser.add_argument(
        "--peers", nargs="*", default=[],
        help="Initial peer multiaddrs for Hivemind DHT bootstrap (e.g. /ip4/1.2.3.4/tcp/1337/p2p/<id>)"
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

    if args.phase in (1, 12, 124, 1245):
        run_phase1(args.seq_len, args.hidden_dim)

    if args.phase in (2, 12, 124, 1245):
        run_phase2(args.seq_len, args.hidden_dim, use_tls=args.tls)

    if args.phase in (4, 124, 1245):
        run_phase4(args.seq_len, args.hidden_dim)

    if args.phase in (5, 1245):
        run_phase5(
            seq_len=args.seq_len,
            hidden_dim=args.hidden_dim,
            node_id=args.node_id,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            port=args.port,
            use_hivemind=args.hivemind,
            dht_port=args.dht_port,
            initial_peers=args.peers,
            use_tls=args.tls,
        )
        # Phase 5 blocks (server loop), so we never return
        return

    _banner("All requested phases completed successfully")


if __name__ == "__main__":
    main()