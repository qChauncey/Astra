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
Astra Benchmark Harness — token throughput, P50/P99 latency, gRPC profiling.

Works with the numpy stub (no GPU required).  Replace ASTRA_USE_KTRANSFORMERS=1
to measure real C++ kernel performance once hardware is available.

Usage examples:

  # Single-node stub throughput
  python scripts/benchmark.py --mode single --num-runs 100 --seq-len 64

  # Multi-node gRPC pipeline latency (start 3 nodes first)
  python scripts/benchmark.py --mode grpc \\
      --nodes 127.0.0.1:50051 127.0.0.1:50052 \\
      --num-runs 200 --output results.json

  # API endpoint throughput (server must be running)
  python scripts/benchmark.py --mode api \\
      --api-url http://localhost:8080 --num-runs 50

  # Print a summary to stdout only (no JSON output)
  python scripts/benchmark.py --mode single --num-runs 50 --quiet
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    latency_ms: float
    tokens_generated: int
    error: Optional[str] = None

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms <= 0 or self.tokens_generated <= 0:
            return 0.0
        return self.tokens_generated / (self.latency_ms / 1000.0)


@dataclass
class BenchmarkReport:
    mode: str
    seq_len: int
    hidden_dim: int
    num_warmup: int
    num_runs: int
    results: List[RunResult] = field(default_factory=list)

    # Computed after collection
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_tokens_per_sec: float = 0.0
    error_rate: float = 0.0

    def compute_stats(self) -> None:
        latencies = [r.latency_ms for r in self.results if r.error is None]
        tps_list = [r.tokens_per_second for r in self.results if r.error is None]
        errors = [r for r in self.results if r.error is not None]

        if not latencies:
            return

        latencies.sort()
        self.p50_ms = statistics.median(latencies)
        self.p95_ms = _percentile(latencies, 95)
        self.p99_ms = _percentile(latencies, 99)
        self.mean_ms = statistics.mean(latencies)
        self.min_ms = min(latencies)
        self.max_ms = max(latencies)
        self.mean_tokens_per_sec = statistics.mean(tps_list) if tps_list else 0.0
        self.error_rate = len(errors) / len(self.results) if self.results else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("results")  # raw results excluded from JSON by default
        return d

    def print_summary(self) -> None:
        ok = len([r for r in self.results if r.error is None])
        err = len(self.results) - ok
        print(f"\n{'─' * 60}")
        print(f"  Astra Benchmark  |  mode={self.mode}")
        print(f"{'─' * 60}")
        print(f"  seq_len       = {self.seq_len}")
        print(f"  hidden_dim    = {self.hidden_dim}")
        print(f"  warmup runs   = {self.num_warmup}")
        print(f"  timed runs    = {self.num_runs}  (ok={ok}, err={err})")
        print(f"{'─' * 60}")
        print("  Latency (ms)")
        print(f"    P50  = {self.p50_ms:>8.2f}")
        print(f"    P95  = {self.p95_ms:>8.2f}")
        print(f"    P99  = {self.p99_ms:>8.2f}")
        print(f"    mean = {self.mean_ms:>8.2f}")
        print(f"    min  = {self.min_ms:>8.2f}")
        print(f"    max  = {self.max_ms:>8.2f}")
        print("  Throughput")
        print(f"    mean tokens/s = {self.mean_tokens_per_sec:>8.1f}")
        print(f"  Error rate      = {self.error_rate * 100:.1f}%")
        print(f"{'─' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark modes
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_node(
    seq_len: int,
    hidden_dim: int,
    num_layers: int = 4,
) -> RunResult:
    """Benchmark the HeterogeneousEngine stub directly (no gRPC)."""
    from astra.inference.heterogeneous import (
        DeviceMap,
        HeterogeneousEngine,
    )
    from astra.serialization.tensor_pack import TensorPacket

    device_map = DeviceMap.cpu_only()
    engine = HeterogeneousEngine(device_map=device_map)
    hidden = np.random.randn(seq_len, hidden_dim).astype(np.float16)
    token_ids = np.random.randint(0, 32000, size=(seq_len,)).tolist()

    packet = TensorPacket(
        tensor=hidden,
        layer_start=0,
        layer_end=num_layers,
        token_ids=token_ids,
    )

    t0 = time.perf_counter()
    output = engine.forward(packet)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    tokens_out = output.tensor.shape[0] if hasattr(output, "tensor") else seq_len
    return RunResult(latency_ms=elapsed_ms, tokens_generated=tokens_out)


def _run_grpc(
    nodes: List[str],
    seq_len: int,
    hidden_dim: int,
) -> RunResult:
    """Benchmark gRPC round-trip through a running node chain."""
    from astra.rpc.client import InferenceClient
    from astra.serialization.tensor_pack import TensorPacket

    hidden = np.random.randn(seq_len, hidden_dim).astype(np.float16)
    packet = TensorPacket(
        tensor=hidden,
        token_ids=list(range(seq_len)),
        layer_start=0,
        layer_end=1,
    )

    t0 = time.perf_counter()
    client = InferenceClient(target=nodes[0])
    result = client.infer(packet)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    tokens_out = result.tensor.shape[0] if result else seq_len
    return RunResult(latency_ms=elapsed_ms, tokens_generated=tokens_out)


def _run_api(api_url: str, seq_len: int) -> RunResult:
    """Benchmark the OpenAI-compatible API endpoint."""
    try:
        import httpx
    except ImportError:
        return RunResult(
            latency_ms=0.0,
            tokens_generated=0,
            error="httpx not installed — pip install httpx",
        )

    payload = {
        "model": "astra-deepseek-v4",
        "messages": [{"role": "user", "content": "benchmark " * seq_len}],
        "max_tokens": 16,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{api_url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return RunResult(latency_ms=0.0, tokens_generated=0, error=str(exc))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    tokens_out = data.get("usage", {}).get("completion_tokens", 1)
    return RunResult(latency_ms=elapsed_ms, tokens_generated=tokens_out)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _percentile(sorted_data: List[float], p: int) -> float:
    if not sorted_data:
        return 0.0
    idx = max(0, int(len(sorted_data) * p / 100) - 1)
    return sorted_data[idx]


def _progress(current: int, total: int, label: str = "") -> None:
    pct = int(current / total * 40)
    bar = "█" * pct + "░" * (40 - pct)
    print(f"\r  {label}[{bar}] {current}/{total}", end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Astra benchmark harness — throughput and latency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["single", "grpc", "api"],
        default="single",
        help="Benchmark mode: single-node stub, gRPC pipeline, or API endpoint",
    )
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument(
        "--nodes",
        nargs="+",
        default=["127.0.0.1:50051"],
        metavar="HOST:PORT",
        help="gRPC node addresses (--mode grpc only)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8080",
        help="API gateway URL (--mode api only)",
    )
    parser.add_argument("--num-warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--num-runs", type=int, default=100, help="Timed iterations")
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write JSON report to file (omit to print only)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress bar"
    )
    args = parser.parse_args()

    dispatch = {
        "single": lambda: _run_single_node(args.seq_len, args.hidden_dim),
        "grpc": lambda: _run_grpc(args.nodes, args.seq_len, args.hidden_dim),
        "api": lambda: _run_api(args.api_url, args.seq_len),
    }
    run_fn = dispatch[args.mode]

    report = BenchmarkReport(
        mode=args.mode,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )

    # Warmup
    if not args.quiet:
        print(f"\nWarming up ({args.num_warmup} runs)…")
    for _ in range(args.num_warmup):
        run_fn()

    # Timed runs
    if not args.quiet:
        print(f"Benchmarking ({args.num_runs} runs)…")
    for i in range(args.num_runs):
        result = run_fn()
        report.results.append(result)
        if not args.quiet:
            _progress(i + 1, args.num_runs)

    if not args.quiet:
        print()  # newline after progress bar

    report.compute_stats()
    report.print_summary()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"  Results written to: {args.output}")

    sys.exit(0 if report.error_rate < 0.05 else 1)


if __name__ == "__main__":
    main()
