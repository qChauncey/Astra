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

Modes
-----
  --mode p2p      (default) Contribute layers to a shared P2P cluster.
                  Other peers can route requests through this node.
  --mode offline  Single-machine mode.  The node covers all layers
                  (0..total-layers) locally.  No P2P discovery needed.
                  Best for solo testing or when you have no GPU peers.

Usage::

    # P2P: contribute layers 0-30 to a cluster
    python scripts/run_node.py \\
        --node-id node-A --port 50051 \\
        --layer-start 0 --layer-end 30 --region us-west

    # P2P: gateway node with UI
    python scripts/run_node.py \\
        --node-id gateway --port 50051 \\
        --layer-start 0 --layer-end 61 --api-port 8080

    # Offline / single-machine (all layers local, UI on port 8080)
    python scripts/run_node.py --mode offline --api-port 8080

Multi-node P2P example (two terminals):
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
from astra.network.dht import AstraDHT, DHTNodeRecord
from astra.rpc.server import InferenceServer
from astra.config.model_config import get_model_config

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
    parser.add_argument(
        "--mode",
        default="p2p",
        choices=["p2p", "offline"],
        help=(
            "p2p: contribute layers to a distributed cluster; "
            "offline: run all layers locally on a single machine"
        ),
    )
    parser.add_argument("--node-id", default="node-0", help="Unique peer identifier")
    parser.add_argument("--port", type=int, default=50051, help="gRPC listen port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (for advertise)")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=30)
    parser.add_argument("--total-layers", type=int, default=61,
                        help="Total transformer layers (used in offline mode)")
    parser.add_argument(
        "--region",
        default="local",
        choices=["local", "us-west", "us-east", "eu-central", "ap-east"],
        help="Geographic region tag",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID (e.g. deepseek-ai/DeepSeek-V4-Flash). Uses model config defaults if set.",
    )
    parser.add_argument(
        "--experts",
        default="all",
        help='Expert shards: "all", "0-127", or "0,1,2"',
    )
    # ── KT-Kernel / SGLang-style parameters (DeepSeek-V4-Flash MXFP4) ──
    parser.add_argument(
        "--kt-method",
        default=None,
        help="KT kernel method: MXFP4 (V4-Flash), FP8, or empty for legacy",
    )
    parser.add_argument(
        "--kt-weight-path",
        default=None,
        help="Path to model weight files for KT-Kernel CPU inference",
    )
    parser.add_argument(
        "--kt-num-gpu-experts",
        type=int,
        default=None,
        help="Number of routed experts kept on GPU (default from model config)",
    )
    parser.add_argument(
        "--kt-cpuinfer",
        type=int,
        default=None,
        help="CPU inference workers for offline experts (default from model config)",
    )
    parser.add_argument(
        "--kt-threadpool-count",
        type=int,
        default=None,
        help="Threadpool count for CPU inference (default from model config)",
    )
    parser.add_argument(
        "--kt-gpu-prefill-token-threshold",
        type=int,
        default=4096,
        help="Token threshold below which prefill runs on GPU only",
    )
    parser.add_argument(
        "--kt-enable-dynamic-expert-update",
        action="store_true",
        help="Enable dynamic expert offload update during inference",
    )
    parser.add_argument(
        "--attention-backend",
        default="flashinfer",
        choices=["flashinfer", "triton", "torch"],
        help="Attention backend for NSA sparse MLA or flash attention",
    )
    parser.add_argument(
        "--disable-shared-experts-fusion",
        action="store_true",
        help="Disable shared expert fusion (recommended for MXFP4)",
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
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=0,
        help="Hidden dimension override (0 = use model default 7168; use 256 for mock runs)",
    )
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
    if args.hidden_dim > 0:
        dmap._hidden_dim_override = args.hidden_dim

    # Resolve model config for defaults
    mc = get_model_config(args.model) if args.model else None

    # Offline mode: this node owns all layers
    if args.mode == "offline":
        args.layer_start = 0
        args.layer_end = args.total_layers

    log.info("=" * 60)
    log.info("  Astra Node: %s  [%s mode]", args.node_id, args.mode.upper())
    log.info("  Port:       %d", args.port)
    log.info("  Layers:     %d – %d", args.layer_start, args.layer_end)
    log.info("  Region:     %s", args.region)
    log.info("  Experts:    %d shards", len(expert_shards))
    log.info("  Backend:    %s", "GPU (KTransformers)" if args.gpu else "CPU (numpy stub)")
    if args.gpu and mc:
        kt_method = args.kt_method or mc.kt_method
        kt_gpu = args.kt_num_gpu_experts or mc.kt_num_gpu_experts
        kt_cpu = args.kt_cpuinfer or mc.kt_cpuinfer
        kt_threads = args.kt_threadpool_count or mc.kt_threadpool_count
        log.info("  KT-Method:  %s", kt_method)
        log.info("  KT-GPU Exp: %d", kt_gpu)
        log.info("  KT-CPU Inf: %d", kt_cpu)
        log.info("  KT-Threads: %d", kt_threads)
        log.info("  Attention:  %s", args.attention_backend)
    log.info("=" * 60)

    # ── Start gRPC server ─────────────────────────────────────────────
    server_kwargs: dict = {
        "node_id": args.node_id,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "port": args.port,
        "geo_region": args.region,
        "expert_shards": expert_shards,
        "device_map": dmap,
        "max_workers": args.workers,
    }
    # Forward KT-Kernel parameters to the inference server
    if args.gpu and mc:
        server_kwargs["kt_method"] = args.kt_method or mc.kt_method
        server_kwargs["kt_weight_path"] = args.kt_weight_path
        server_kwargs["kt_num_gpu_experts"] = args.kt_num_gpu_experts or mc.kt_num_gpu_experts
        server_kwargs["kt_cpuinfer"] = args.kt_cpuinfer or mc.kt_cpuinfer
        server_kwargs["kt_threadpool_count"] = args.kt_threadpool_count or mc.kt_threadpool_count
        server_kwargs["kt_gpu_prefill_token_threshold"] = args.kt_gpu_prefill_token_threshold
        server_kwargs["kt_enable_dynamic_expert_update"] = args.kt_enable_dynamic_expert_update
        server_kwargs["attention_backend"] = args.attention_backend
        server_kwargs["disable_shared_experts_fusion"] = args.disable_shared_experts_fusion
        server_kwargs["model_config"] = mc
    server = InferenceServer(**server_kwargs)
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
                    num_layers=args.total_layers,
                    hidden_dim=dmap.hidden_dim,
                ),
                node_id=args.node_id,
                layer_start=args.layer_start,
                layer_end=args.layer_end,
                mode=args.mode,
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
    if hasattr(signal, "SIGTERM"):  # SIGTERM is not available on Windows
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
