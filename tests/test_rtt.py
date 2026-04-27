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

"""Tests for astra.network.rtt — RTT measurement and EWMA smoothing."""

from __future__ import annotations

import socket
import threading
import time

import pytest

from astra.network.rtt import (
    PeerRTT,
    RTTMonitor,
    tcp_probe,
)


# ── PeerRTT ──────────────────────────────────────────────────────────────────

class TestPeerRTT:
    def test_first_sample_sets_ewma(self):
        p = PeerRTT(address="x:1")
        p.record_success(50.0, alpha=0.3)
        assert p.ewma_rtt_ms == 50.0
        assert p.last_rtt_ms == 50.0
        assert p.samples == 1
        assert p.healthy is True

    def test_ewma_smoothing(self):
        p = PeerRTT(address="x:1")
        p.record_success(50.0, alpha=0.5)
        p.record_success(100.0, alpha=0.5)
        # ewma = 0.5 * 100 + 0.5 * 50 = 75
        assert p.ewma_rtt_ms == 75.0

    def test_failure_increments_counter(self):
        p = PeerRTT(address="x:1")
        p.record_failure(fail_threshold=3)
        assert p.consecutive_failures == 1
        assert p.healthy is True   # below threshold

    def test_failure_threshold_marks_unhealthy(self):
        p = PeerRTT(address="x:1")
        for _ in range(3):
            p.record_failure(fail_threshold=3)
        assert p.healthy is False

    def test_success_resets_failure_count(self):
        p = PeerRTT(address="x:1")
        p.record_failure()
        p.record_failure()
        p.record_success(10.0)
        assert p.consecutive_failures == 0
        assert p.healthy is True


# ── tcp_probe ────────────────────────────────────────────────────────────────

class TestTcpProbe:
    def test_probe_localhost(self):
        # Bind ephemeral port and probe it
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]

        # Accept loop in background so connect succeeds
        accepted = threading.Event()

        def _accept_once():
            try:
                conn, _ = srv.accept()
                conn.close()
            finally:
                accepted.set()

        threading.Thread(target=_accept_once, daemon=True).start()
        rtt = tcp_probe(f"127.0.0.1:{port}", timeout=1.0)
        accepted.wait(timeout=1.0)
        srv.close()
        assert rtt >= 0.0
        assert rtt < 1000.0   # localhost should be sub-second

    def test_probe_unreachable_raises(self):
        # Use a port unlikely to be open
        with pytest.raises(OSError):
            tcp_probe("127.0.0.1:1", timeout=0.2)


# ── RTTMonitor ───────────────────────────────────────────────────────────────

class TestRTTMonitor:
    def _fake_probe(self, latencies: dict):
        """Returns a probe_fn that returns hardcoded RTTs for each address."""
        def probe(address: str, timeout: float) -> float:
            if address in latencies:
                return float(latencies[address])
            raise OSError("unreachable")
        return probe

    def test_update_peers_adds_and_removes(self):
        m = RTTMonitor()
        m.update_peers(["a:1", "b:2"])
        assert set(m.all_stats().keys()) == {"a:1", "b:2"}
        m.update_peers(["b:2", "c:3"])
        assert set(m.all_stats().keys()) == {"b:2", "c:3"}

    def test_probe_once_records_success(self):
        m = RTTMonitor(probe_fn=self._fake_probe({"a:1": 25.0}))
        m.update_peers(["a:1"])
        m.probe_once()
        assert m.get_rtt("a:1") == 25.0
        assert m.is_healthy("a:1") is True

    def test_probe_once_records_failure(self):
        m = RTTMonitor(probe_fn=self._fake_probe({}), fail_threshold=2)
        m.update_peers(["bad:1"])
        m.probe_once()
        m.probe_once()
        assert m.is_healthy("bad:1") is False
        assert m.get_rtt("bad:1") is None

    def test_get_rtt_unknown_peer(self):
        m = RTTMonitor()
        assert m.get_rtt("never:seen") is None

    def test_smoothing_across_probes(self):
        m = RTTMonitor(
            probe_fn=self._fake_probe({"a:1": 100.0}),
            ewma_alpha=0.5,
        )
        m.update_peers(["a:1"])
        m.probe_once()
        assert m.get_rtt("a:1") == 100.0
        # Switch the probe to return 50 ms
        m._probe_fn = self._fake_probe({"a:1": 50.0})
        m.probe_once()
        assert m.get_rtt("a:1") == 75.0   # 0.5*50 + 0.5*100

    def test_start_stop_idempotent(self):
        m = RTTMonitor(probe_interval_s=10.0)
        m.start()
        m.start()   # second call should be no-op
        m.stop()
        m.stop()    # idempotent

    def test_background_loop_actually_probes(self):
        calls = []

        def probe(address, timeout):
            calls.append(address)
            return 1.0

        m = RTTMonitor(probe_interval_s=0.05, probe_fn=probe)
        m.update_peers(["x:1"])
        m.start()
        time.sleep(0.15)
        m.stop()
        assert len(calls) >= 1
