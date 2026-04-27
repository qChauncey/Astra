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
#   - Written from scratch; implements the Astra gRPC InferenceService.
#   - Integrates HeterogeneousEngine for split GPU/CPU computation.
#   - Task B: simulates intra-cluster node relay for P2P pipeline testing.

"""
Astra gRPC Inference Server.

Each server instance represents one node in the P2P pipeline.  It receives
TensorPackets from upstream nodes, runs heterogeneous inference (GPU attention
+ CPU MoE), and forwards results to the next downstream node.

Quick start::

    server = InferenceServer(
        node_id="node-A",
        layer_start=0,
        layer_end=20,
        port=50051,
    )
    server.serve()   # blocks
"""

from __future__ import annotations

import logging
import threading
import time
import zlib
from concurrent import futures
from typing import Iterator, List, Optional

import grpc

import numpy as np

from ..inference.batch_scheduler import BatchGroup, BatchRequest, ContinuousBatchScheduler
from ..inference.batch_utils import BatchInfo, pad_sequences, unpad_output
from ..inference.heterogeneous import DeviceMap, HeterogeneousEngine
from ..inference.shared_expert_cache import ExpertWeights
from ..serialization.tensor_pack import TensorSerializer
from .generated import inference_pb2 as pb2
from .generated import inference_pb2_grpc as pb2_grpc
from .tls import TLSConfig, load_server_credentials

log = logging.getLogger(__name__)


class _InferenceServicer(pb2_grpc.InferenceServiceServicer):
    """gRPC servicer implementation wired to HeterogeneousEngine."""

    def __init__(
        self,
        node_id: str,
        engine: HeterogeneousEngine,
        layer_start: int,
        layer_end: int,
        geo_region: str,
        expert_shards: List[int],
    ) -> None:
        self._node_id = node_id
        self._engine = engine
        self._layer_start = layer_start
        self._layer_end = layer_end
        self._geo_region = geo_region
        self._expert_shards = expert_shards
        self._request_count = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # RunLayer                                                              #
    # ------------------------------------------------------------------ #

    def RunLayer(
        self,
        request: pb2.InferenceRequest,
        context: grpc.ServicerContext,
    ) -> pb2.InferenceResponse:
        t0 = time.perf_counter()
        try:
            packet = TensorSerializer.deserialize(request.hidden_states.payload)
            layer_indices = list(range(request.layer_start, request.layer_end))

            out_packet = self._engine.forward(packet, layer_indices=layer_indices)
            out_packet.src_node = self._node_id
            out_packet.dst_node = request.src_node

            out_bytes = TensorSerializer.serialize(out_packet)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            with self._lock:
                self._request_count += 1

            return pb2.InferenceResponse(
                request_id=request.request_id,
                output_states=pb2.TensorFrame(
                    payload=out_bytes,
                    byte_len=len(out_bytes),
                    crc32=zlib.crc32(out_bytes) & 0xFFFFFFFF,
                ),
                node_id=self._node_id,
                compute_time_ms=elapsed_ms,
                success=True,
            )
        except Exception as exc:
            log.exception("RunLayer failed for request %s", request.request_id)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return pb2.InferenceResponse(
                request_id=request.request_id,
                node_id=self._node_id,
                success=False,
                error_message=str(exc),
            )

    # ------------------------------------------------------------------ #
    # RunLayerStream                                                         #
    # ------------------------------------------------------------------ #

    def RunLayerStream(
        self,
        request_iterator: Iterator[pb2.InferenceRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[pb2.InferenceResponse]:
        for request in request_iterator:
            yield self.RunLayer(request, context)

    # ------------------------------------------------------------------ #
    # Ping                                                                   #
    # ------------------------------------------------------------------ #

    def Ping(
        self,
        request: pb2.PingRequest,
        context: grpc.ServicerContext,
    ) -> pb2.PingResponse:
        stats = self._engine.stats()
        with self._lock:
            _request_count = self._request_count
        return pb2.PingResponse(
            node_id=self._node_id,
            ready=True,
            geo_region=self._geo_region,
            layer_start=self._layer_start,
            layer_end=self._layer_end,
            expert_shards=self._expert_shards,
            backend=stats.get("backend", "unknown"),
            gpu_util=stats.get("gpu_util", 0.0),
            cpu_util=stats.get("cpu_util", 0.0),
        )

    # ------------------------------------------------------------------ #
    # KV cache transfer (Phase 3)                                           #
    # ------------------------------------------------------------------ #

    def TransferKVCache(
        self,
        request_iterator: Iterator[pb2.KVCacheChunk],
        context: grpc.ServicerContext,
    ) -> pb2.PingResponse:
        from .kv_transfer import KVCacheReceiver
        request_id, applied = KVCacheReceiver.receive_and_apply(
            self._engine, request_iterator
        )
        log.info(
            "TransferKVCache complete: request_id=%s layers_applied=%d",
            request_id[:8] if request_id else "?", applied,
        )
        return pb2.PingResponse(node_id=self._node_id, ready=True)


class InferenceServer:
    """
    High-level wrapper that owns the gRPC server lifecycle.

    Parameters
    ----------
    node_id:       Unique string identifier for this peer.
    layer_start:   First transformer layer this node serves.
    layer_end:     First transformer layer of the *next* node (exclusive).
    port:          TCP port to bind on.
    geo_region:    Geographic region tag (e.g. "us-west").
    expert_shards: Expert IDs this node can compute (default: all).
    device_map:    GPU/CPU split configuration.
    max_workers:   Thread pool size for the gRPC server.
    """

    def __init__(
        self,
        node_id: str = "node-0",
        layer_start: int = 0,
        layer_end: int = 10,
        port: int = 50051,
        geo_region: str = "local",
        expert_shards: Optional[List[int]] = None,
        device_map: Optional[DeviceMap] = None,
        max_workers: int = 4,
        tls_config: Optional[TLSConfig] = None,
    ) -> None:
        self.node_id = node_id
        self.port = port
        self._expert_shards = expert_shards or list(range(256))
        self._tls_config = tls_config

        dmap = device_map or DeviceMap.cpu_only()
        self._engine = HeterogeneousEngine.from_device_map(dmap)

        # Pre-pin shared experts 0 and 1
        for sid in range(2):
            self._engine.load_shared_experts([ExpertWeights.mock(sid, hidden_dim=dmap.hidden_dim)])

        self._servicer = _InferenceServicer(
            node_id=node_id,
            engine=self._engine,
            layer_start=layer_start,
            layer_end=layer_end,
            geo_region=geo_region,
            expert_shards=self._expert_shards,
        )

        self._grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers)
        )
        pb2_grpc.add_InferenceServiceServicer_to_server(
            self._servicer, self._grpc_server
        )

        if self._tls_config and self._tls_config.enabled and self._tls_config.is_ready():
            creds = load_server_credentials(
                self._tls_config.cert_path,
                self._tls_config.key_path,
                self._tls_config.ca_cert_path or None,
            )
            self._grpc_server.add_secure_port(f"[::]:{port}", creds)
            log.info("TLS enabled on port %d (mTLS=%s)", port, bool(self._tls_config.ca_cert_path))
        else:
            self._grpc_server.add_insecure_port(f"[::]:{port}")

        self._max_workers = max_workers

    def start(self) -> None:
        self._grpc_server.start()
        log.info("InferenceServer %s listening on port %d", self.node_id, self.port)

    def stop(self, grace: float = 2.0) -> None:
        self._grpc_server.stop(grace)
        log.info("InferenceServer %s stopped", self.node_id)

    def serve(self) -> None:
        """Start and block until interrupted."""
        self.start()
        try:
            self._grpc_server.wait_for_termination()
        except KeyboardInterrupt:
            self.stop()

    def engine_stats(self) -> dict:
        return self._engine.stats()

    # ------------------------------------------------------------------ #
    # Phase 7.3.2 — Continuous Batching Support                            #
    # ------------------------------------------------------------------ #

    def run_batch(
        self,
        batch_group: BatchGroup,
        layer_indices: Optional[List[int]] = None,
    ) -> BatchGroup:
        """
        Execute a pre-formed batch group through the heterogeneous engine.

        Takes a BatchGroup with padded_tensor and batch_info, runs the engine's
        forward pass, unpads the output, and returns the batch_group updated
        with per-request results.

        Parameters
        ----------
        batch_group: BatchGroup from ContinuousBatchScheduler.form_batches()
        layer_indices: optional override for layer range (defaults to
                       self._servicer._layer_start.._layer_end)

        Returns
        -------
        The same BatchGroup with each BatchRequest.result populated.
        """
        if batch_group.padded_tensor is None or batch_group.batch_info is None:
            raise ValueError("BatchGroup must have padded_tensor and batch_info set")

        if layer_indices is None:
            layer_indices = list(range(
                self._servicer._layer_start,
                self._servicer._layer_end,
            ))

        t0 = time.perf_counter()
        engine_output: np.ndarray = self._engine.forward_batch(
            batch_group.padded_tensor,
            layer_indices=layer_indices,
            attention_mask=batch_group.batch_info.attention_mask,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        unpadded = unpad_output(engine_output, batch_group.batch_info)
        for i, req in enumerate(batch_group.requests):
            if i < len(unpadded):
                req.result = unpadded[i]
                req.metadata["compute_ms"] = elapsed_ms
            else:
                req.error = f"No output slice for request index {i}"
        return batch_group

    def run_batches_from_scheduler(
        self,
        scheduler: ContinuousBatchScheduler,
        layer_indices: Optional[List[int]] = None,
    ) -> int:
        """
        Drain all ready batches from a scheduler and execute them.

        Returns number of batches executed.
        """
        batches = scheduler.form_batches()
        for bg in batches:
            try:
                self.run_batch(bg, layer_indices=layer_indices)
                # run_batch has already set per-request results; we just
                # complete with a dummy output so scheduler tracks completion.
                scheduler.complete_batch(
                    bg.batch_id,
                    bg.padded_tensor if bg.padded_tensor is not None else
                    np.zeros((1, 1, 1), dtype=np.float32),
                )
            except Exception as exc:
                log.exception("Batch %s failed: %s", bg.batch_id, exc)
                scheduler.complete_batch(
                    bg.batch_id,
                    np.zeros((1, 1, 1), dtype=np.float32),
                    error=str(exc),
                )
        return len(batches)
