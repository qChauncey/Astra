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
#   - Written from scratch; implements the TransferKVCache streaming RPC.
#   - Chunks KV tensors into ≤4 MB gRPC messages to stay within limits.
#   - Integrates with HeterogeneousEngine.LayerKVCache.

"""
KV-Cache Streaming Transfer.

Allows one pipeline node to push its accumulated key/value cache to a
peer node so that the peer can continue decoding without recomputing past
tokens.  This is the primary mechanism for cross-node KV reuse in Phase 3.

Protocol:
  Sender calls KVCacheTransferClient.push(engine, request_id, target_addr).
  Receiver's InferenceServer applies received chunks via
  KVCacheTransferClient.apply_chunks(engine, chunks).

Chunking:
  Large KV tensors are split into ≤ MAX_CHUNK_BYTES slices so the payload
  fits in gRPC's default 4 MB message limit.

Usage::

    # On the upstream node after processing layers 0-20:
    sender = KVCacheSender("localhost:50052", src_node="node-A")
    sender.push(engine, request_id="abc123", layer_indices=[0,1,...,20])

    # The downstream node's InferenceServer automatically receives and
    # applies the cache via the TransferKVCache RPC handler.
"""

from __future__ import annotations

import logging
import struct
from typing import Dict, Iterator, List, Optional, Tuple

import grpc
import numpy as np

from ..inference.heterogeneous import HeterogeneousEngine, LayerKVCache
from .generated import inference_pb2 as pb2
from .generated import inference_pb2_grpc as pb2_grpc

log = logging.getLogger(__name__)

MAX_CHUNK_BYTES = 3 * 1024 * 1024   # 3 MB — safely under gRPC 4 MB limit


def _split_array(arr: np.ndarray) -> Iterator[Tuple[np.ndarray, List[int]]]:
    """
    Yield (chunk_array, [row_start, row_end]) slices of `arr` along axis 0
    such that each chunk's raw bytes fit within MAX_CHUNK_BYTES.
    """
    row_bytes = arr[0].nbytes if arr.ndim > 1 else arr.nbytes
    rows_per_chunk = max(1, MAX_CHUNK_BYTES // row_bytes)
    total = arr.shape[0]
    for start in range(0, total, rows_per_chunk):
        end = min(start + rows_per_chunk, total)
        yield arr[start:end], [start, end]


def _encode_array(arr: np.ndarray) -> Tuple[bytes, List[int], str]:
    """Serialize numpy array → (raw_bytes, shape, dtype_str)."""
    contiguous = np.ascontiguousarray(arr)
    return contiguous.tobytes(), list(contiguous.shape), str(contiguous.dtype)


def _decode_chunk(
    raw: bytes,
    shape: List[int],
    dtype_str: str,
) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape)


# ─────────────────────────────────────────────────────────────────────────── #
# Sender                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class KVCacheSender:
    """
    Pushes a local engine's KV cache to a remote InferenceServer.

    Each layer's (K, V) tensors are streamed as KVCacheChunk messages.
    The receiver reassembles and inserts them into its own LayerKVCache.
    """

    def __init__(
        self,
        target_address: str,
        src_node: str = "sender",
        timeout: float = 60.0,
    ) -> None:
        self._addr = target_address
        self._src_node = src_node
        self._timeout = timeout
        options = [
            ("grpc.max_send_message_length", 8 * 1024 * 1024),
            ("grpc.max_receive_message_length", 8 * 1024 * 1024),
        ]
        self._channel = grpc.insecure_channel(target_address, options=options)
        self._stub = pb2_grpc.InferenceServiceStub(self._channel)

    def push(
        self,
        engine: HeterogeneousEngine,
        request_id: str,
        layer_indices: Optional[List[int]] = None,
    ) -> bool:
        """
        Stream all (or selected) layers' KV caches to the target node.

        Returns True if the transfer completed successfully.
        """
        if layer_indices is None:
            layer_indices = list(engine.kv_cache.keys())

        if not layer_indices:
            log.debug("KVCacheSender.push: no layers to transfer")
            return True

        def _chunk_iter() -> Iterator[pb2.KVCacheChunk]:
            for layer_idx in layer_indices:
                cache: LayerKVCache = engine.kv_cache.get(layer_idx)  # type: ignore[attr-defined]
                if cache is None or cache.k is None:
                    continue

                k = cache.k
                v = cache.v

                # Stream K tensor in chunks
                for chunk_arr, [rs, re] in _split_array(k):
                    raw, shape, dtype = _encode_array(chunk_arr)
                    yield pb2.KVCacheChunk(
                        request_id=request_id,
                        layer_idx=layer_idx,
                        k_bytes=raw,
                        k_shape=shape,
                        dtype=dtype,
                        v_bytes=b"",   # K-only chunk
                        v_shape=[],
                    )

                # Stream V tensor in chunks
                for chunk_arr, [rs, re] in _split_array(v):
                    raw, shape, dtype = _encode_array(chunk_arr)
                    yield pb2.KVCacheChunk(
                        request_id=request_id,
                        layer_idx=layer_idx,
                        k_bytes=b"",   # V-only chunk
                        k_shape=[],
                        v_bytes=raw,
                        v_shape=shape,
                        dtype=dtype,
                    )

        try:
            resp = self._stub.TransferKVCache(_chunk_iter(), timeout=self._timeout)
            log.info(
                "KV-cache transfer → %s request_id=%s layers=%s ready=%s",
                self._addr, request_id[:8], layer_indices, resp.ready,
            )
            return resp.ready
        except grpc.RpcError as err:
            log.error("KV-cache transfer failed: %s", err)
            return False

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "KVCacheSender":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ─────────────────────────────────────────────────────────────────────────── #
# Receiver-side reassembler (called inside InferenceServer's servicer)          #
# ─────────────────────────────────────────────────────────────────────────── #

class KVCacheReceiver:
    """
    Reassembles streamed KVCacheChunk messages and injects them into an engine.

    The InferenceServer passes the chunk iterator to `receive_and_apply`.
    """

    @staticmethod
    def receive_and_apply(
        engine: HeterogeneousEngine,
        chunk_iterator: Iterator[pb2.KVCacheChunk],
    ) -> Tuple[str, int]:
        """
        Consume the chunk stream, reassemble K and V, insert into engine cache.

        Returns (request_id, num_layers_applied).
        """
        # Accumulate raw K and V bytes per layer
        k_bufs: Dict[int, List[bytes]] = {}
        v_bufs: Dict[int, List[bytes]] = {}
        k_shapes: Dict[int, List[int]] = {}
        v_shapes: Dict[int, List[int]] = {}
        dtypes: Dict[int, str] = {}
        request_id = ""
        chunk_count = 0

        for chunk in chunk_iterator:
            request_id = chunk.request_id
            li = chunk.layer_idx
            dtypes[li] = chunk.dtype or "float32"
            chunk_count += 1

            if chunk.k_bytes:
                k_bufs.setdefault(li, []).append(chunk.k_bytes)
                if chunk.k_shape:
                    k_shapes[li] = list(chunk.k_shape)

            if chunk.v_bytes:
                v_bufs.setdefault(li, []).append(chunk.v_bytes)
                if chunk.v_shape:
                    v_shapes[li] = list(chunk.v_shape)

        # Reassemble and inject
        applied = 0
        all_layers = set(k_bufs) | set(v_bufs)
        for li in all_layers:
            try:
                dtype = np.dtype(dtypes.get(li, "float32"))
                k_arr = v_arr = None

                if li in k_bufs:
                    raw_k = b"".join(k_bufs[li])
                    k_arr = np.frombuffer(raw_k, dtype=dtype)
                    if li in k_shapes:
                        k_arr = k_arr.reshape(k_shapes[li])

                if li in v_bufs:
                    raw_v = b"".join(v_bufs[li])
                    v_arr = np.frombuffer(raw_v, dtype=dtype)
                    if li in v_shapes:
                        v_arr = v_arr.reshape(v_shapes[li])

                if k_arr is not None or v_arr is not None:
                    cache = engine.kv_cache.setdefault(li, LayerKVCache())
                    if k_arr is not None:
                        cache.k = k_arr
                    if v_arr is not None:
                        cache.v = v_arr
                    applied += 1
            except Exception as exc:
                log.warning("KV reassembly failed for layer %d: %s", li, exc)

        log.info(
            "KV-cache received: request_id=%s chunks=%d layers_applied=%d",
            request_id[:8] if request_id else "?", chunk_count, applied,
        )
        return request_id, applied
