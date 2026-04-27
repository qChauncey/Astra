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
Phase 7.3.2 — Continuous Batching Scheduler.

Dynamically groups pending inference requests into batched GPU invocations,
using length-based binning to minimise padding waste.

Key features:
  - Length binning: groups sequences of similar lengths together
  - Configurable max batch size (sequences) and max tokens per batch
  - FIFO queuing with optional priority inversion for short sequences
  - Metrics tracking: batch utilisation, padding overhead, queue depth, latency
  - Supports per-layer batching for pipeline-parallel decode
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .batch_utils import BatchInfo, pad_sequences, unpad_output, compute_batch_metrics

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Types & Configuration                                                #
# ------------------------------------------------------------------ #

class RequestStatus(Enum):
    PENDING = auto()
    BATCHED = auto()
    COMPUTING = auto()
    DONE = auto()
    ERROR = auto()


@dataclass
class BatchRequest:
    """A single inference request waiting to be batched."""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    token_ids: List[int] = field(default_factory=list)
    hidden_states: Optional[np.ndarray] = None  # (seq_len, hidden_dim) pre-embedded
    status: RequestStatus = RequestStatus.PENDING
    arrival_time: float = field(default_factory=time.time)
    priority: int = 0  # higher = more urgent; short seq gets slight boost
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[np.ndarray] = None
    error: Optional[str] = None

    @property
    def seq_len(self) -> int:
        if self.hidden_states is not None:
            return self.hidden_states.shape[0]
        return len(self.token_ids)

    @property
    def age_ms(self) -> float:
        return (time.time() - self.arrival_time) * 1000.0


@dataclass
class BatchGroup:
    """A formed batch ready for GPU invocation."""
    batch_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    requests: List[BatchRequest] = field(default_factory=list)
    padded_tensor: Optional[np.ndarray] = None
    batch_info: Optional[BatchInfo] = None
    formed_at: float = field(default_factory=time.time)
    layer_idx: Optional[int] = None

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def total_tokens(self) -> int:
        return sum(r.seq_len for r in self.requests)


@dataclass
class BatchingConfig:
    max_batch_size: int = 16           # max sequences per batch
    max_tokens_per_batch: int = 2048   # max total real tokens (not padded) per batch
    max_wait_ms: float = 10.0          # max time to wait for queue accumulation
    length_bin_ratio: float = 2.0      # sequences with length difference ≤ ratio can bin together
    min_batch_size: int = 1            # minimum sequences to form a batch (set to 1 for low traffic)
    pad_to_multiple_of: int = 8        # pad max_seq_len to nearest multiple for GPU efficiency
    enable_length_buckets: bool = True # use explicit length buckets for grouping


# Pre-defined length buckets for efficient grouping
_DEFAULT_LENGTH_BUCKETS: List[Tuple[int, int]] = [
    (1, 128),
    (129, 256),
    (257, 512),
    (513, 1024),
    (1025, 2048),
    (2049, 4096),
]


class ContinuousBatchScheduler:
    """
    Continuous batching scheduler for LLM inference.

    Maintains a pending queue → forms length-binned batch groups → dispatches
    to GPU → unpad results → returns per-request outputs.

    Usage::

        scheduler = ContinuousBatchScheduler(BatchingConfig(max_batch_size=8))

        # Enqueue requests
        scheduler.enqueue(token_ids=[1,2,3], hidden_states=emb)
        scheduler.enqueue(token_ids=[4,5], hidden_states=emb2)

        # Form and execute batches (delegated to caller or internal forward_fn)
        for batch in scheduler.form_batches():
            result = forward_fn(batch.padded_tensor, batch.batch_info)
            scheduler.complete_batch(batch.batch_id, result)

        # Retrieve per-request results
        results = scheduler.drain_completed()
    """

    def __init__(
        self,
        config: Optional[BatchingConfig] = None,
        length_buckets: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self._config = config or BatchingConfig()
        self._length_buckets = length_buckets or _DEFAULT_LENGTH_BUCKETS

        # Internal state
        self._queue: Deque[BatchRequest] = deque()
        self._pending_batches: Dict[str, BatchGroup] = {}
        self._completed: Dict[str, BatchGroup] = {}
        self._lock = threading.Lock()
        self._next_batch_id = 0

        # Metrics
        self._metrics: Dict[str, Any] = {
            "total_enqueued": 0,
            "total_completed": 0,
            "total_batches_formed": 0,
            "total_padding_overhead_pct": 0.0,
            "peak_queue_depth": 0,
            "last_batch_time_ms": 0.0,
        }

    # -------------------------------------------------------------- #
    # Public API                                                      #
    # -------------------------------------------------------------- #

    def enqueue(self, **kwargs) -> str:
        """
        Enqueue a single inference request.

        Accepts keyword args to BatchRequest fields:
          - token_ids: List[int]
          - hidden_states: np.ndarray (seq_len, hidden_dim)
          - priority: int
          - metadata: dict

        Returns the request_id.
        """
        req = BatchRequest(**kwargs)
        with self._lock:
            self._queue.append(req)
            self._metrics["total_enqueued"] += 1
            self._metrics["peak_queue_depth"] = max(
                self._metrics["peak_queue_depth"], len(self._queue)
            )
        return req.request_id

    def enqueue_bulk(self, requests: List[BatchRequest]) -> None:
        """Enqueue multiple requests at once."""
        with self._lock:
            for req in requests:
                self._queue.append(req)
                self._metrics["total_enqueued"] += 1
            self._metrics["peak_queue_depth"] = max(
                self._metrics["peak_queue_depth"], len(self._queue)
            )

    def form_batches(self, max_batches: int = -1) -> List[BatchGroup]:
        """
        Form length-binned batch groups from the pending queue.

        Strategy:
          1. Sort pending requests by sequence length.
          2. Assign each request to the best-fitting length bucket.
          3. Within each bucket, greedily fill batches up to max_batch_size
             and max_tokens_per_batch.
          4. Skip batches that don't meet min_batch_size (unless max_wait_ms
             has been exceeded for the oldest request in the queue — then
             flush as-is).

        Returns list of ready-to-execute BatchGroup objects.
        """
        t0 = time.perf_counter()
        with self._lock:
            if not self._queue:
                return []

            # Drain queue into a local list
            pending: List[BatchRequest] = list(self._queue)
            self._queue.clear()

        # Sort by sequence length (ascending); tie-break by age (oldest first)
        pending.sort(key=lambda r: (r.seq_len, r.arrival_time))

        # Assign to length buckets
        buckets: Dict[int, List[BatchRequest]] = {i: [] for i in range(len(self._length_buckets))}
        unbucketed: List[BatchRequest] = []

        for req in pending:
            assigned = False
            for bi, (lo, hi) in enumerate(self._length_buckets):
                if lo <= req.seq_len <= hi:
                    buckets[bi].append(req)
                    assigned = True
                    break
            if not assigned:
                unbucketed.append(req)

        # Form batches per bucket
        batches: List[BatchGroup] = []
        for bi in range(len(self._length_buckets)):
            batches.extend(
                self._form_batches_from_list(
                    buckets[bi], f"bucket_{bi}"
                )
            )
        batches.extend(self._form_batches_from_list(unbucketed, "unbucketed"))

        # Enforce min_batch_size with aging exception
        final_batches: List[BatchGroup] = []
        age_threshold_ms = self._config.max_wait_ms
        now = time.time()

        for bg in batches:
            if bg.size >= self._config.min_batch_size:
                final_batches.append(bg)
            else:
                # Check if oldest request has exceeded max wait
                oldest_age = max((now - r.arrival_time) * 1000.0 for r in bg.requests)
                if oldest_age > age_threshold_ms:
                    final_batches.append(bg)  # flush anyway to avoid starvation
                else:
                    # Re-queue for next attempt
                    with self._lock:
                        # Reset status to PENDING
                        for r in bg.requests:
                            r.status = RequestStatus.PENDING
                        self._queue.extend(bg.requests)

        # Pad each batch
        for bg in final_batches:
            self._pad_batch(bg)

        # Register batches
        with self._lock:
            for bg in final_batches:
                self._pending_batches[bg.batch_id] = bg
                self._metrics["total_batches_formed"] += 1

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._metrics["last_batch_time_ms"] = dt_ms

        if max_batches > 0:
            return final_batches[:max_batches]
        return final_batches

    def complete_batch(
        self, batch_id: str, padded_output: np.ndarray, error: Optional[str] = None
    ) -> None:
        """
        Called after GPU forward pass completes. Unpads and distributes
        results back to individual requests.
        """
        with self._lock:
            bg = self._pending_batches.pop(batch_id, None)
        if bg is None:
            log.warning("Batch %s not found for completion", batch_id)
            return

        if error:
            for req in bg.requests:
                req.status = RequestStatus.ERROR
                req.error = error
            self._completed[batch_id] = bg
            return

        if bg.batch_info is None:
            raise ValueError(f"Batch {batch_id} has no BatchInfo; was pad_batch called?")

        unpadded = unpad_output(padded_output, bg.batch_info)
        for i, req in enumerate(bg.requests):
            if i < len(unpadded):
                req.result = unpadded[i]
                req.status = RequestStatus.DONE
            else:
                req.status = RequestStatus.ERROR
                req.error = f"No output slice for request index {i}"

        self._completed[batch_id] = bg
        self._metrics["total_completed"] += bg.size

        # Track padding overhead
        bm = compute_batch_metrics(bg.batch_info)
        n = self._metrics["total_batches_formed"]
        current_avg = self._metrics["total_padding_overhead_pct"]
        self._metrics["total_padding_overhead_pct"] = (
            (current_avg * (n - 1) + bm["padding_overhead_pct"]) / max(1, n)
        )

    def drain_completed(self) -> Dict[str, BatchRequest]:
        """Pop all completed requests and return by request_id."""
        result: Dict[str, BatchRequest] = {}
        with self._lock:
            done_ids = list(self._completed.keys())
            for bid in done_ids:
                bg = self._completed.pop(bid)
                for req in bg.requests:
                    result[req.request_id] = req
        return result

    def get_request(self, request_id: str) -> Optional[BatchRequest]:
        """Look up a request by ID across all states."""
        with self._lock:
            # Check queue
            for req in self._queue:
                if req.request_id == request_id:
                    return req
            # Check pending batches
            for bg in self._pending_batches.values():
                for req in bg.requests:
                    if req.request_id == request_id:
                        return req
            # Check completed
            for bg in self._completed.values():
                for req in bg.requests:
                    if req.request_id == request_id:
                        return req
        return None

    # -------------------------------------------------------------- #
    # Properties & Metrics                                             #
    # -------------------------------------------------------------- #

    @property
    def queue_depth(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def pending_batch_count(self) -> int:
        with self._lock:
            return len(self._pending_batches)

    def metrics(self) -> dict:
        """Return a snapshot of current scheduler metrics."""
        with self._lock:
            m = dict(self._metrics)
            m["queue_depth"] = len(self._queue)
            m["pending_batches"] = len(self._pending_batches)
            m["completed_batches"] = len(self._completed)
            return m

    def reset_metrics(self) -> None:
        """Reset all accumulated metrics to zero."""
        with self._lock:
            self._metrics = {
                "total_enqueued": 0,
                "total_completed": 0,
                "total_batches_formed": 0,
                "total_padding_overhead_pct": 0.0,
                "peak_queue_depth": 0,
                "last_batch_time_ms": 0.0,
            }

    # -------------------------------------------------------------- #
    # Internal helpers                                                  #
    # -------------------------------------------------------------- #

    def _form_batches_from_list(
        self, requests: List[BatchRequest], source_label: str
    ) -> List[BatchGroup]:
        """Greedy batch formation from a list of similarly-length requests."""
        batches: List[BatchGroup] = []
        i = 0
        while i < len(requests):
            batch_group = BatchGroup(layer_idx=None)
            total_tokens = 0
            max_len = requests[i].seq_len

            while (
                i < len(requests)
                and batch_group.size < self._config.max_batch_size
                and total_tokens + requests[i].seq_len <= self._config.max_tokens_per_batch
            ):
                req = requests[i]
                # Check length compatibility ratio
                if batch_group.size > 0:
                    ratio = max(req.seq_len, max_len) / max(1, min(req.seq_len, max_len))
                    if ratio > self._config.length_bin_ratio:
                        break

                batch_group.requests.append(req)
                total_tokens += req.seq_len
                max_len = max(max_len, req.seq_len)
                req.status = RequestStatus.BATCHED
                i += 1

            if batch_group.requests:
                batches.append(batch_group)
            else:
                i += 1  # Shouldn't happen, but guard against infinite loop

        return batches

    def _pad_batch(self, bg: BatchGroup) -> None:
        """Pad a formed batch group for GPU invocation."""
        sequences: List[np.ndarray] = []
        for req in bg.requests:
            if req.hidden_states is not None:
                sequences.append(req.hidden_states)
            else:
                # Create placeholder zeros; caller should have embedded tokens
                hidden_dim = 4096  # default
                if sequences:
                    hidden_dim = sequences[-1].shape[1]
                sequences.append(np.zeros((req.seq_len, hidden_dim), dtype=np.float32))

        # Pad to multiple of configured value for GPU efficiency
        raw_max_len = max(s.shape[0] for s in sequences)
        pad_mult = self._config.pad_to_multiple_of
        max_len = ((raw_max_len + pad_mult - 1) // pad_mult) * pad_mult if pad_mult > 0 else raw_max_len

        padded, info = pad_sequences(
            sequences,
            max_seq_len=max_len,
            generate_attention_mask=True,
        )
        bg.padded_tensor = padded
        bg.batch_info = info
        bg.batch_info.sequence_ids = [r.request_id for r in bg.requests]

    def _age_exceeded(self, req: BatchRequest) -> bool:
        """Check if a request has waited longer than max_wait_ms."""
        return req.age_ms > self._config.max_wait_ms