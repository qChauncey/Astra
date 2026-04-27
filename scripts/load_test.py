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
Astra Load Test — concurrent inference stress test.

Sends N concurrent requests to the OpenAI-compatible API and reports
throughput, P50/P95/P99 latency, and error rate.  Uses asyncio + httpx
(already a project dependency) — no locust required.

Usage:

  # Start the server first:
  python scripts/run_node.py --mode offline --api-port 8080

  # 100 concurrent users, 60-second ramp, report every 10 s:
  python scripts/load_test.py --url http://localhost:8080 \\
      --concurrency 100 --duration 60 --report-interval 10

  # Quick smoke test (10 users, 15 s):
  python scripts/load_test.py --concurrency 10 --duration 15

  # Write JSON report:
  python scripts/load_test.py --concurrency 50 --duration 30 \\
      --output load_test_results.json

Exit codes:
  0  — all success-rate and latency thresholds passed
  1  — error_rate > --max-error-rate  OR  p99_ms > --max-p99-ms
  2  — server unreachable
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    start_time: float
    latency_ms: float
    status_code: int
    tokens_out: int
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status_code < 300


@dataclass
class LoadTestReport:
    url: str
    concurrency: int
    duration_s: float
    max_tokens: int
    total_requests: int = 0
    ok_requests: int = 0
    error_requests: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    requests_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    error_rate: float = 0.0
    errors: List[str] = field(default_factory=list)

    def compute(self, results: List[RequestResult], elapsed: float) -> None:
        ok = [r for r in results if r.ok]
        err = [r for r in results if not r.ok]
        self.total_requests = len(results)
        self.ok_requests = len(ok)
        self.error_requests = len(err)
        self.error_rate = len(err) / len(results) if results else 0.0
        self.errors = list({r.error for r in err if r.error})[:10]

        lat = sorted(r.latency_ms for r in ok)
        if lat:
            self.p50_ms = _pct(lat, 50)
            self.p95_ms = _pct(lat, 95)
            self.p99_ms = _pct(lat, 99)
            self.mean_ms = statistics.mean(lat)
            self.min_ms = min(lat)
            self.max_ms = max(lat)

        self.requests_per_sec = len(results) / elapsed if elapsed > 0 else 0.0
        total_tokens = sum(r.tokens_out for r in ok)
        self.tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

    def print_summary(self) -> None:
        print(f"\n{'─' * 60}")
        print(f"  Astra Load Test  |  {self.url}")
        print(f"{'─' * 60}")
        print(f"  concurrency     = {self.concurrency}")
        print(f"  duration        = {self.duration_s:.1f} s")
        print(f"  total requests  = {self.total_requests}")
        print(f"  ok / error      = {self.ok_requests} / {self.error_requests}")
        print(f"  error rate      = {self.error_rate * 100:.1f}%")
        print(f"{'─' * 60}")
        print("  Throughput")
        print(f"    requests/s    = {self.requests_per_sec:>8.1f}")
        print(f"    tokens/s      = {self.tokens_per_sec:>8.1f}")
        print("  Latency (ms)")
        print(f"    P50           = {self.p50_ms:>8.2f}")
        print(f"    P95           = {self.p95_ms:>8.2f}")
        print(f"    P99           = {self.p99_ms:>8.2f}")
        print(f"    mean          = {self.mean_ms:>8.2f}")
        print(f"    min           = {self.min_ms:>8.2f}")
        print(f"    max           = {self.max_ms:>8.2f}")
        if self.errors:
            print("  Top errors:")
            for e in self.errors[:5]:
                print(f"    • {e}")
        print(f"{'─' * 60}\n")

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

_MESSAGES = [
    "Explain how mixture-of-experts models work.",
    "What is the difference between attention and FFN layers?",
    "Describe the pipeline parallelism strategy used in Petals.",
    "How does KV-cache improve autoregressive decoding?",
    "What is speculative decoding and when does it help?",
    "Explain the haversine formula and its use in P2P routing.",
    "What are Ed25519 keys and how are they used for peer identity?",
    "Describe the safetensors format for model weight storage.",
]


async def _worker(
    client,
    url: str,
    max_tokens: int,
    results: List[RequestResult],
    stop_event: asyncio.Event,
    worker_id: int,
) -> None:
    idx = worker_id % len(_MESSAGES)
    while not stop_event.is_set():
        payload = {
            "model": "astra-deepseek-v4",
            "messages": [{"role": "user", "content": _MESSAGES[idx]}],
            "max_tokens": max_tokens,
            "stream": False,
        }
        idx = (idx + 1) % len(_MESSAGES)
        start = time.perf_counter()
        try:
            resp = await client.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=30.0,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if resp.status_code == 200:
                data = resp.json()
                tokens_out = data.get("usage", {}).get("completion_tokens", 1)
                results.append(RequestResult(
                    start_time=start,
                    latency_ms=elapsed_ms,
                    status_code=resp.status_code,
                    tokens_out=tokens_out,
                ))
            else:
                results.append(RequestResult(
                    start_time=start,
                    latency_ms=elapsed_ms,
                    status_code=resp.status_code,
                    tokens_out=0,
                    error=f"HTTP {resp.status_code}",
                ))
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            results.append(RequestResult(
                start_time=start,
                latency_ms=elapsed_ms,
                status_code=0,
                tokens_out=0,
                error=str(exc)[:120],
            ))


async def _progress_reporter(
    results: List[RequestResult],
    stop_event: asyncio.Event,
    interval: float,
    duration: float,
    start_wall: float,
) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(interval)
        elapsed = time.perf_counter() - start_wall
        n = len(results)
        ok = sum(1 for r in results if r.ok)
        rps = n / elapsed if elapsed > 0 else 0
        pct = min(100, int(elapsed / duration * 100))
        print(
            f"\r  [{elapsed:5.1f}s / {duration:.0f}s  {pct:3d}%]"
            f"  requests={n}  ok={ok}  err={n-ok}  rps={rps:.1f}",
            end="",
            flush=True,
        )


async def run_load_test(
    url: str,
    concurrency: int,
    duration: float,
    max_tokens: int,
    report_interval: float,
) -> tuple[LoadTestReport, List[RequestResult]]:
    try:
        import httpx
    except ImportError:
        print("ERROR: httpx is required — pip install httpx", file=sys.stderr)
        sys.exit(2)

    # Quick connectivity check
    async with httpx.AsyncClient() as probe:
        try:
            await probe.get(f"{url}/health", timeout=5.0)
        except Exception as exc:
            print(f"ERROR: Cannot reach {url} — {exc}", file=sys.stderr)
            sys.exit(2)

    results: List[RequestResult] = []
    stop_event = asyncio.Event()
    start_wall = time.perf_counter()

    async with httpx.AsyncClient() as client:
        workers = [
            asyncio.create_task(
                _worker(client, url, max_tokens, results, stop_event, i)
            )
            for i in range(concurrency)
        ]
        reporter = asyncio.create_task(
            _progress_reporter(results, stop_event, report_interval, duration, start_wall)
        )

        await asyncio.sleep(duration)
        stop_event.set()
        await asyncio.gather(*workers, return_exceptions=True)
        reporter.cancel()

    print()  # newline after progress bar
    elapsed = time.perf_counter() - start_wall
    report = LoadTestReport(
        url=url,
        concurrency=concurrency,
        duration_s=elapsed,
        max_tokens=max_tokens,
    )
    report.compute(results, elapsed)
    return report, results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct(sorted_data: List[float], p: int) -> float:
    if not sorted_data:
        return 0.0
    idx = max(0, int(len(sorted_data) * p / 100) - 1)
    return sorted_data[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Astra load test — concurrent inference stress test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url", default="http://localhost:8080", help="API gateway URL"
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Number of concurrent virtual users"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Test duration in seconds"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32,
        help="max_tokens per request"
    )
    parser.add_argument(
        "--report-interval", type=float, default=5.0,
        help="Live stats refresh interval (seconds)"
    )
    parser.add_argument(
        "--output", metavar="FILE",
        help="Write JSON report to file"
    )
    parser.add_argument(
        "--max-error-rate", type=float, default=0.05,
        help="Fail if error rate exceeds this threshold (0–1)"
    )
    parser.add_argument(
        "--max-p99-ms", type=float, default=10_000.0,
        help="Fail if P99 latency exceeds this threshold (ms)"
    )
    args = parser.parse_args()

    print("\n  Astra Load Test")
    print(f"  url={args.url}  concurrency={args.concurrency}  duration={args.duration}s")

    report, _results = asyncio.run(run_load_test(
        url=args.url,
        concurrency=args.concurrency,
        duration=args.duration,
        max_tokens=args.max_tokens,
        report_interval=args.report_interval,
    ))

    report.print_summary()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"  Report written to: {args.output}")

    # SLA check
    passed = True
    if report.error_rate > args.max_error_rate:
        print(
            f"  FAIL: error_rate={report.error_rate:.1%} > threshold={args.max_error_rate:.1%}",
            file=sys.stderr,
        )
        passed = False
    if report.p99_ms > args.max_p99_ms:
        print(
            f"  FAIL: p99={report.p99_ms:.1f}ms > threshold={args.max_p99_ms:.1f}ms",
            file=sys.stderr,
        )
        passed = False
    if passed:
        print("  PASS: all SLA thresholds met.")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
