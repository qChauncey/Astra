# Copyright 2025 Project Astra Contributors
# Licensed under the Apache License, Version 2.0

"""
Integration tests for the gRPC inference pipeline.

Starts real InferenceServer instances in background threads and exercises
the full pack → gRPC → compute → deserialize loop.
"""

import threading
import time
from typing import List

import numpy as np
import pytest

from astra.inference.heterogeneous import DeviceMap
from astra.rpc.client import InferenceClient
from astra.rpc.server import InferenceServer
from astra.serialization.tensor_pack import TensorPacket


HIDDEN = 64
PORT_BASE = 50200   # offset to avoid conflicts with mock_pipeline.py


def _make_server(node_id: str, port: int, ls: int, le: int) -> InferenceServer:
    dmap = DeviceMap.cpu_only()
    dmap.hidden_dim = HIDDEN
    return InferenceServer(
        node_id=node_id,
        layer_start=ls,
        layer_end=le,
        port=port,
        geo_region="local",
        device_map=dmap,
        max_workers=2,
    )


def _start(server: InferenceServer) -> None:
    server.start()


@pytest.fixture(scope="module")
def single_server():
    s = _make_server("s0", PORT_BASE, 0, 10)
    t = threading.Thread(target=_start, args=(s,), daemon=True)
    t.start()
    time.sleep(0.4)
    yield s
    s.stop(grace=1.0)


@pytest.fixture(scope="module")
def two_servers():
    s1 = _make_server("s1", PORT_BASE + 1, 0, 10)
    s2 = _make_server("s2", PORT_BASE + 2, 10, 20)
    for s in (s1, s2):
        t = threading.Thread(target=_start, args=(s,), daemon=True)
        t.start()
    time.sleep(0.5)
    yield s1, s2
    s1.stop(grace=1.0)
    s2.stop(grace=1.0)


def _make_packet(seq_len: int = 4) -> TensorPacket:
    pkt = TensorPacket.make_input(list(range(seq_len)), hidden_dim=HIDDEN)
    pkt.selected_experts = np.zeros((seq_len, 2), dtype=np.int32)
    return pkt


# ─────────────────────────────────────────────────────────────────────────── #
# Ping                                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

class TestPing:
    def test_ping_ready(self, single_server):
        with InferenceClient(f"localhost:{PORT_BASE}", node_id="test") as c:
            info = c.ping()
        assert info["ready"] is True
        assert info["node_id"] == "s0"
        assert info["layer_start"] == 0
        assert info["layer_end"] == 10

    def test_ping_unreachable_returns_error(self):
        with InferenceClient("localhost:19999", node_id="test") as c:
            info = c.ping()
        assert not info.get("ready", False)
        assert "error" in info


# ─────────────────────────────────────────────────────────────────────────── #
# Single-hop inference                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

class TestSingleHop:
    def test_output_shape_preserved(self, single_server):
        pkt = _make_packet(seq_len=6)
        with InferenceClient(f"localhost:{PORT_BASE}", node_id="test") as c:
            out = c.run_layer(pkt, layer_start=0, layer_end=10)
        assert out.tensor.shape == (6, HIDDEN)

    def test_packet_id_preserved(self, single_server):
        pkt = _make_packet()
        with InferenceClient(f"localhost:{PORT_BASE}", node_id="test") as c:
            out = c.run_layer(pkt, layer_start=0, layer_end=10)
        assert out.packet_id == pkt.packet_id

    def test_output_dtype_float16(self, single_server):
        pkt = _make_packet()
        with InferenceClient(f"localhost:{PORT_BASE}", node_id="test") as c:
            out = c.run_layer(pkt, layer_start=0, layer_end=10)
        assert out.tensor.dtype == np.float16

    def test_multiple_sequential_calls(self, single_server):
        with InferenceClient(f"localhost:{PORT_BASE}", node_id="test") as c:
            for _ in range(5):
                pkt = _make_packet(seq_len=2)
                out = c.run_layer(pkt, layer_start=0, layer_end=10)
                assert out.tensor.shape == (2, HIDDEN)

    def test_client_stats_increment(self, single_server):
        with InferenceClient(f"localhost:{PORT_BASE}", node_id="test") as c:
            for _ in range(3):
                c.run_layer(_make_packet(), layer_start=0, layer_end=10)
            s = c.stats()
        assert s["total_calls"] == 3
        assert s["total_bytes_sent"] > 0


# ─────────────────────────────────────────────────────────────────────────── #
# Two-hop pipeline                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class TestTwoHop:
    def test_two_hop_output_shape(self, two_servers):
        pkt = _make_packet(seq_len=8)
        with InferenceClient(f"localhost:{PORT_BASE + 1}", node_id="test") as c1:
            mid = c1.run_layer(pkt, layer_start=0, layer_end=10)
        with InferenceClient(f"localhost:{PORT_BASE + 2}", node_id="test") as c2:
            out = c2.run_layer(mid, layer_start=10, layer_end=20)
        assert out.tensor.shape == (8, HIDDEN)

    def test_packet_id_consistent_across_hops(self, two_servers):
        pkt = _make_packet()
        original_id = pkt.packet_id
        with InferenceClient(f"localhost:{PORT_BASE + 1}", node_id="test") as c1:
            mid = c1.run_layer(pkt, layer_start=0, layer_end=10)
        with InferenceClient(f"localhost:{PORT_BASE + 2}", node_id="test") as c2:
            out = c2.run_layer(mid, layer_start=10, layer_end=20)
        assert out.packet_id == original_id


# ─────────────────────────────────────────────────────────────────────────── #
# Streaming                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

class TestStreaming:
    def test_stream_batch(self, single_server):
        packets = [_make_packet(seq_len=2) for _ in range(3)]
        with InferenceClient(f"localhost:{PORT_BASE}", node_id="test") as c:
            results = c.run_layer_stream(packets, layer_start=0, layer_end=10)
        assert len(results) == 3
        for r in results:
            assert r.tensor.shape == (2, HIDDEN)
