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
#   - Written from scratch for Astra's P2P pipeline relay protocol.
#   - Task B: implements "打包-传输-接收" (pack-transmit-receive) client side.

"""
Astra gRPC Inference Client.

The client serializes a TensorPacket, ships it to a remote InferenceServer,
and deserializes the result.  Designed for chained use across multiple
pipeline stages (P2P relay).

Usage::

    client = InferenceClient("localhost:50051", node_id="client-0")
    result = client.run_layer(packet, layer_start=0, layer_end=10)
"""

from __future__ import annotations

import logging
import time
import zlib
from typing import Dict, Iterator, List

import grpc

from ..serialization.tensor_pack import TensorPacket, TensorSerializer
from .generated import inference_pb2 as pb2
from .generated import inference_pb2_grpc as pb2_grpc

log = logging.getLogger(__name__)

_DEFAULT_OPTIONS = [
    ("grpc.max_send_message_length", 512 * 1024 * 1024),    # 512 MB
    ("grpc.max_receive_message_length", 512 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 10_000),
    ("grpc.keepalive_timeout_ms", 5_000),
]


class InferenceClient:
    """
    gRPC client for one Astra inference node.

    Parameters
    ----------
    address:   "host:port" of the target InferenceServer.
    node_id:   This client's peer ID (used as src_node in requests).
    timeout:   Per-RPC deadline in seconds.
    """

    def __init__(
        self,
        address: str,
        node_id: str = "client",
        timeout: float = 30.0,
    ) -> None:
        self._address = address
        self._node_id = node_id
        self._timeout = timeout
        self._channel = grpc.insecure_channel(address, options=_DEFAULT_OPTIONS)
        self._stub = pb2_grpc.InferenceServiceStub(self._channel)
        self._total_calls = 0
        self._total_bytes_sent = 0

    # ------------------------------------------------------------------ #
    # Core RPC                                                              #
    # ------------------------------------------------------------------ #

    def run_layer(
        self,
        packet: TensorPacket,
        layer_start: int,
        layer_end: int,
        use_kv_cache: bool = True,
    ) -> TensorPacket:
        """
        Serialize packet, RPC to server, deserialize response.

        This is the "打包-传输-接收" (pack-transmit-receive) loop closure.
        """
        payload = TensorSerializer.serialize(packet)
        self._total_bytes_sent += len(payload)
        self._total_calls += 1

        request = pb2.InferenceRequest(
            request_id=packet.packet_id,
            hidden_states=pb2.TensorFrame(
                payload=payload,
                byte_len=len(payload),
                crc32=zlib.crc32(payload) & 0xFFFFFFFF,
            ),
            layer_start=layer_start,
            layer_end=layer_end,
            use_kv_cache=use_kv_cache,
            src_node=self._node_id,
            dst_node=packet.dst_node,
        )

        t0 = time.perf_counter()
        try:
            response = self._stub.RunLayer(request, timeout=self._timeout)
        except grpc.RpcError as err:
            log.error("RPC to %s failed: %s", self._address, err)
            raise

        rtt_ms = (time.perf_counter() - t0) * 1000.0
        log.debug(
            "RunLayer %s→%s layers=%d:%d rtt=%.1fms compute=%.1fms",
            self._node_id,
            self._address,
            layer_start,
            layer_end,
            rtt_ms,
            response.compute_time_ms,
        )

        if not response.success:
            raise RuntimeError(
                f"Remote inference failed on {self._address}: {response.error_message}"
            )

        out_bytes = response.output_states.payload
        received_crc = zlib.crc32(out_bytes) & 0xFFFFFFFF
        if received_crc != response.output_states.crc32:
            raise ValueError(
                f"CRC32 mismatch on response: got {received_crc:#x}, "
                f"expected {response.output_states.crc32:#x}"
            )

        return TensorSerializer.deserialize(out_bytes)

    # ------------------------------------------------------------------ #
    # Streaming variant                                                     #
    # ------------------------------------------------------------------ #

    def run_layer_stream(
        self,
        packets: List[TensorPacket],
        layer_start: int,
        layer_end: int,
    ) -> List[TensorPacket]:
        """Send a batch of packets as a bidirectional stream."""

        def _request_iter() -> Iterator[pb2.InferenceRequest]:
            for pkt in packets:
                payload = TensorSerializer.serialize(pkt)
                yield pb2.InferenceRequest(
                    request_id=pkt.packet_id,
                    hidden_states=pb2.TensorFrame(payload=payload, byte_len=len(payload)),
                    layer_start=layer_start,
                    layer_end=layer_end,
                    src_node=self._node_id,
                )

        results = []
        for resp in self._stub.RunLayerStream(_request_iter(), timeout=self._timeout):
            if resp.success:
                results.append(TensorSerializer.deserialize(resp.output_states.payload))
        return results

    # ------------------------------------------------------------------ #
    # Health / discovery                                                    #
    # ------------------------------------------------------------------ #

    def ping(self) -> Dict:
        """Return server capability info as a plain dict."""
        try:
            resp = self._stub.Ping(
                pb2.PingRequest(
                    node_id=self._node_id,
                    timestamp=int(time.time()),
                ),
                timeout=5.0,
            )
            return {
                "node_id": resp.node_id,
                "ready": resp.ready,
                "geo_region": resp.geo_region,
                "layer_start": resp.layer_start,
                "layer_end": resp.layer_end,
                "expert_shards": list(resp.expert_shards[:10]),  # truncate for display
                "backend": resp.backend,
            }
        except grpc.RpcError as err:
            return {"error": str(err), "ready": False}

    # ------------------------------------------------------------------ #
    # Lifecycle                                                             #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "InferenceClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def stats(self) -> dict:
        return {
            "address": self._address,
            "total_calls": self._total_calls,
            "total_bytes_sent": self._total_bytes_sent,
        }
