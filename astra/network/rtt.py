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
Active RTT measurement (Phase 3.3).

Replaces the haversine-distance estimate in ``GeoAwareMoERouter`` with real
round-trip times measured via gRPC ``Ping`` calls.

Design
------
- Each peer is probed periodically.  Latencies are smoothed with an
  exponentially-weighted moving average (EWMA) so a single packet jitter
  does not destabilise the routing decision.
- Failed probes mark the peer ``unhealthy`` until a subsequent probe succeeds.
- The geographic estimate from ``GeoRegion.rtt_ms`` remains the **fallback**
  when a peer has not yet been measured.

Usage::

    monitor = RTTMonitor(probe_interval_s=5.0)
    monitor.start()                                  # background thread
    monitor.update_peers(["10.0.0.2:50051", ...])    # peers to probe
    rtt = monitor.get_rtt("10.0.0.2:50051")          # in ms, or None
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

log = logging.getLogger("astra.rtt")

# EWMA smoothing factor: new_value = alpha * sample + (1 - alpha) * prev
_DEFAULT_ALPHA = 0.3
# How many consecutive failed probes before a peer is marked unhealthy
_DEFAULT_FAIL_THRESHOLD = 3
# Default interval between probes
_DEFAULT_INTERVAL_S = 5.0
# Probe timeout in seconds
_DEFAULT_TIMEOUT_S = 2.0


# ─────────────────────────────────────────────────────────────────────────── #
# Per-peer measurement                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class PeerRTT:
    """Smoothed RTT statistics for one peer."""
    address: str
    last_rtt_ms: Optional[float] = None
    ewma_rtt_ms: Optional[float] = None
    samples: int = 0
    last_probe_at: float = 0.0
    consecutive_failures: int = 0
    healthy: bool = True

    def record_success(self, rtt_ms: float, alpha: float = _DEFAULT_ALPHA) -> None:
        self.last_rtt_ms = rtt_ms
        self.last_probe_at = time.time()
        self.consecutive_failures = 0
        self.healthy = True
        if self.ewma_rtt_ms is None:
            self.ewma_rtt_ms = rtt_ms
        else:
            self.ewma_rtt_ms = alpha * rtt_ms + (1.0 - alpha) * self.ewma_rtt_ms
        self.samples += 1

    def record_failure(self, fail_threshold: int = _DEFAULT_FAIL_THRESHOLD) -> None:
        self.last_probe_at = time.time()
        self.consecutive_failures += 1
        if self.consecutive_failures >= fail_threshold:
            self.healthy = False


# ─────────────────────────────────────────────────────────────────────────── #
# Probe function (defaults to TCP-connect probe; can be replaced)               #
# ─────────────────────────────────────────────────────────────────────────── #

def tcp_probe(address: str, timeout: float = _DEFAULT_TIMEOUT_S) -> float:
    """
    Probe *address* (``host:port``) via TCP connect; return RTT in ms.
    Raises ``OSError`` on failure.

    A successful TCP connect is a reasonable lower bound for gRPC RTT and
    requires no protobuf compilation, so it works for the in-memory test
    fixtures used in our test suite.  Production deployments should use
    ``grpc_ping_probe`` instead.
    """
    import socket
    host, _, port_s = address.partition(":")
    port = int(port_s) if port_s else 50051
    t0 = time.perf_counter()
    with socket.create_connection((host, port), timeout=timeout):
        pass
    return (time.perf_counter() - t0) * 1000.0


def grpc_ping_probe(address: str, timeout: float = _DEFAULT_TIMEOUT_S) -> float:
    """
    Probe *address* via the gRPC ``Ping`` RPC; return RTT in ms.
    Lazy-imports ``InferenceClient`` to avoid pulling gRPC into modules that
    only need timing.
    """
    from ..rpc.client import InferenceClient
    client = InferenceClient(target=address, timeout_s=timeout)
    t0 = time.perf_counter()
    try:
        client.ping()
    finally:
        client.close()
    return (time.perf_counter() - t0) * 1000.0


# ─────────────────────────────────────────────────────────────────────────── #
# Monitor                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class RTTMonitor:
    """
    Background RTT-measurement service.

    Thread-safe.  ``start()`` and ``stop()`` are idempotent.
    """

    def __init__(
        self,
        probe_interval_s: float = _DEFAULT_INTERVAL_S,
        probe_timeout_s: float = _DEFAULT_TIMEOUT_S,
        ewma_alpha: float = _DEFAULT_ALPHA,
        fail_threshold: int = _DEFAULT_FAIL_THRESHOLD,
        probe_fn: Callable[[str, float], float] = tcp_probe,
    ) -> None:
        self._interval = probe_interval_s
        self._timeout = probe_timeout_s
        self._alpha = ewma_alpha
        self._fail_threshold = fail_threshold
        self._probe_fn = probe_fn

        self._peers: Dict[str, PeerRTT] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────

    def update_peers(self, addresses: List[str]) -> None:
        """Add new peers; absent peers are removed."""
        with self._lock:
            new_set = set(addresses)
            for addr in list(self._peers):
                if addr not in new_set:
                    del self._peers[addr]
            for addr in addresses:
                if addr not in self._peers:
                    self._peers[addr] = PeerRTT(address=addr)

    def get_rtt(self, address: str) -> Optional[float]:
        """Smoothed RTT in ms, or None if no successful probe yet."""
        with self._lock:
            peer = self._peers.get(address)
            return peer.ewma_rtt_ms if peer else None

    def is_healthy(self, address: str) -> bool:
        """True if the peer is reachable; False after fail_threshold failures."""
        with self._lock:
            peer = self._peers.get(address)
            return bool(peer and peer.healthy)

    def all_stats(self) -> Dict[str, PeerRTT]:
        with self._lock:
            return {a: p for a, p in self._peers.items()}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def probe_once(self) -> None:
        """Probe every known peer once.  Used both by the loop and by tests."""
        with self._lock:
            peers = list(self._peers.values())
        for peer in peers:
            try:
                rtt_ms = self._probe_fn(peer.address, self._timeout)
                with self._lock:
                    peer.record_success(rtt_ms, self._alpha)
            except Exception as exc:
                log.debug("Probe failed for %s: %s", peer.address, exc)
                with self._lock:
                    peer.record_failure(self._fail_threshold)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="astra-rtt-monitor", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.probe_once()
            self._stop.wait(self._interval)
