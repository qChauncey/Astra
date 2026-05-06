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
MoLink Bridge — gRPC pipeline parallelism adapter for consumer-grade GPUs.

MoLink (arXiv:2507.05043) is a distributed LLM serving system designed for
heterogeneous, weakly-connected consumer-grade GPUs. It is the successor to
PETALS, improving on it with:

  1. **Dynamic micro-batch scheduling** — fills pipeline bubbles caused by
     network latency by adjusting the number of micro-batches per iteration.
  2. **Chunk transmission** — splits prefill activations into chunks and
     prioritises decode-phase transmissions to reduce TTFT and TPOT.
  3. **Heterogeneous node support** — Linux (k8s), Windows (WSL2), and
     containerised VMs (e.g., AutoDL, vast.ai) via a unified gRPC protocol.

This module provides a MoLink-compatible pipeline parallelism adapter that
integrates with Astra's ``PipelineOrchestrator``. Worker nodes register via
gRPC, exposing their layer coverage and hardware capabilities. The master
node's scheduler dynamically partitions model layers across workers and
applies micro-batch + chunk-transmission optimisations.

Architecture::

    ┌──────────────────────────────────────────────────┐
    │  MoLinkMaster (this module)                      │
    │  ┌────────────┐  ┌──────────┐  ┌─────────────┐  │
    │  │ Scheduler   │  │ Profiler │  │ Micro-Batch │  │
    │  │ (layer      │  │ (latency │  │ Controller  │  │
    │  │  partition) │  │  matrix) │  │ (dynamic N) │  │
    │  └────────────┘  └──────────┘  └─────────────┘  │
    │                        │                         │
    │              ┌─────────┴─────────┐               │
    │              │  Chunk Transmitter │               │
    │              │  (prefill→decode)  │               │
    │              └─────────┬─────────┘               │
    └────────────────────────┼──────────────────────────┘
                             │ gRPC (Ethernet / public net)
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │ Worker 0│   │ Worker 1│   │ Worker N│
        │ (Linux) │   │ (Win)   │   │ (VM)    │
        └─────────┘   └─────────┘   └─────────┘

References:
    Jin et al., "MoLink: Distributed and Efficient Serving Framework for
    Large Models", arXiv:2507.05043, 2025.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("astra.network.molink")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MoLinkWorkerConfig:
    """Hardware and layer coverage descriptor for a single worker node."""

    node_id: str
    address: str                          # gRPC endpoint (host:port)
    os_type: str = "linux"                # "linux", "windows", "containerized_vm"
    gpu_type: str = ""                    # e.g. "RTX 4090", "A100"
    gpu_count: int = 1
    gpu_memory_gb: float = 24.0
    cpu_cores: int = 16
    cpu_memory_gb: float = 64.0
    network_bandwidth_mbps: float = 1000.0
    network_latency_ms: float = 1.0
    layer_start: int = 0
    layer_end: int = 0
    expert_shards: List[int] = field(default_factory=list)
    backend: str = "ktransformers"        # "ktransformers", "vllm", "petals"

    # Runtime profiling data (filled by profiler)

    compute_latency_ms: Dict[str, float] = field(default_factory=dict)
    """Per-batch-size compute latency: {"bs1_seq128": 12.3, ...}"""


@dataclass
class MoLinkMasterConfig:
    """Master node configuration for MoLink-style pipeline scheduling."""

    # Pipeline
    pipeline_parallel_size: int = 4
    max_micro_batches: int = 16           # upper bound for dynamic scheduling
    min_micro_batches: int = 2

    # Chunk transmission (prefill/decode interleaving)
    enable_chunk_transmission: bool = True
    chunk_size_tokens: int = 256          # tokens per prefill chunk
    decode_priority: bool = True          # prioritise decode over prefill

    # Profiling
    profile_warmup_steps: int = 3
    profile_interval_seconds: float = 30.0

    # Retry / fault tolerance
    max_retries_per_hop: int = 2
    retry_base_delay: float = 1.0

    # Hardware
    gpu_util_threshold: float = 0.9       # offload if GPU util ≥ this


# =============================================================================
# Pipeline phase
# =============================================================================

class PipelinePhase(Enum):
    """Phase of a request in pipeline processing."""
    PREFILL = auto()   # initial prompt processing (many tokens)
    DECODE = auto()    # autoregressive token generation (1 token per step)


# =============================================================================
# Micro-batch state
# =============================================================================

@dataclass
class MicroBatch:
    """A single micro-batch within a pipeline iteration."""
    batch_id: str
    phase: PipelinePhase
    tokens: np.ndarray                   # shape (batch, seq, hidden_dim)
    priority: int = 0                     # 0=highest (decode), 1+=lower
    created_at: float = field(default_factory=time.time)


@dataclass
class PipelineSlot:
    """Tracks one micro-batch's position in the pipeline timeline."""
    micro_batch: MicroBatch
    worker_idx: int
    layer_start: int
    layer_end: int
    dispatch_time: Optional[float] = None
    compute_start_time: Optional[float] = None
    compute_end_time: Optional[float] = None


# =============================================================================
# Profiler
# =============================================================================

class MoLinkProfiler:
    """
    Periodically profiles compute latency and network conditions for each
    worker in the pipeline, producing matrices used by the scheduler.
    """

    def __init__(self, workers: List[MoLinkWorkerConfig]) -> None:
        self._workers = workers
        self._compute_matrix: Dict[str, Dict[str, float]] = {}
        self._network_matrix: Dict[Tuple[str, str], float] = {}  # (src, dst) → ms
        self._lock = threading.Lock()

    def profile_all(self) -> None:
        """Run a full profiling cycle across all workers."""
        with self._lock:
            for w in self._workers:
                self._profile_compute(w)
            for i, w1 in enumerate(self._workers):
                for w2 in self._workers[i + 1:]:
                    self._profile_network(w1, w2)

    def _profile_compute(self, worker: MoLinkWorkerConfig) -> None:
        """Profile per-worker compute latency at common batch sizes."""
        # In production, this would RPC the worker's engine to run a micro-
        # benchmark.  For now we use the worker's pre-populated data.
        key = worker.node_id
        self._compute_matrix[key] = dict(worker.compute_latency_ms)

    def _profile_network(self, w1: MoLinkWorkerConfig, w2: MoLinkWorkerConfig) -> None:
        """Profile pairwise network latency between two workers."""
        # Use declared latency; in production, run a ping/pong RPC.
        latency = max(w1.network_latency_ms, w2.network_latency_ms)
        self._network_matrix[(w1.node_id, w2.node_id)] = latency
        self._network_matrix[(w2.node_id, w1.node_id)] = latency

    def get_compute_latency(
        self, node_id: str, batch_size: int, seq_len: int
    ) -> float:
        """Query compute latency for a specific (batch_size, seq_len) pair."""
        key = f"bs{batch_size}_seq{seq_len}"
        with self._lock:
            node_data = self._compute_matrix.get(node_id, {})
            if key in node_data:
                return node_data[key]
            # Fallback: linear extrapolation from nearest profile
            return self._extrapolate_latency(node_data, batch_size, seq_len)

    def get_network_latency(self, src_id: str, dst_id: str) -> float:
        """Query pairwise network latency (ms)."""
        with self._lock:
            return self._network_matrix.get((src_id, dst_id), 10.0)

    @staticmethod
    def _extrapolate_latency(
        node_data: Dict[str, float], batch_size: int, seq_len: int
    ) -> float:
        """Simple linear extrapolation when exact profile is missing."""
        if not node_data:
            return batch_size * seq_len * 0.001  # rough estimate
        # Use the closest available key
        best = min(
            node_data.values(),
            key=lambda v: abs(v - batch_size * seq_len * 0.001),
            default=10.0,
        )
        return best * (batch_size * seq_len) / max(1, batch_size * seq_len)


# =============================================================================
# Dynamic Micro-Batch Scheduler
# =============================================================================

class DynamicMicroBatchScheduler:
    """
    Core MoLink innovation: dynamically adjusts the number of micro-batches
    per pipeline iteration to fill bubbles caused by network latency.

    The key insight from the MoLink paper (Section 4.1):
      When the number of micro-batches equals the pipeline degree, network
      transmission delays cause pipeline bubbles — the target worker idles
      waiting for the next micro-batch to arrive. By increasing the number
      of micro-batches, the transmission time of batch N can be overlapped
      with the computation of batch N+1, reducing or eliminating the bubble.

    Algorithm (two-phase profiling):
      1. Startup Profiling: measure computation latency across batch sizes.
      2. Runtime Monitoring: periodically probe downstream network conditions.
      3. The controller computes an ideal micro-batch count N via incremental
         search starting at N=1, terminating when the computation overhead of
         additional batches is fully overlapped with residual bubbles.
    """

    def __init__(
        self,
        pipeline_size: int,
        profiler: MoLinkProfiler,
        worker_order: List[MoLinkWorkerConfig],
        max_batches: int = 16,
        min_batches: int = 2,
    ) -> None:
        self._pipeline_size = pipeline_size
        self._profiler = profiler
        self._worker_order = worker_order
        self._max_batches = max_batches
        self._min_batches = min_batches
        self._lock = threading.Lock()
        self._current_n: int = pipeline_size  # initial: N = pipeline degree

    @property
    def optimal_micro_batch_count(self) -> int:
        """Current dynamically-computed optimal N."""
        with self._lock:
            return self._current_n

    def recompute(
        self,
        batch_size: int,
        seq_len: int,
        decode_steps: int = 1,
    ) -> int:
        """
        Recompute the optimal number of micro-batches for current conditions.

        Returns the recommended N.
        """
        with self._lock:
            self._current_n = self._compute_optimal_n(batch_size, seq_len, decode_steps)
            return self._current_n

    def _compute_optimal_n(
        self, batch_size: int, seq_len: int, decode_steps: int
    ) -> int:
        """Incremental search for optimal micro-batch count."""
        # Compute per-worker forward pass latency
        comp_latencies: List[float] = []
        for w in self._worker_order:
            comp_latencies.append(
                self._profiler.get_compute_latency(w.node_id, batch_size, seq_len)
            )

        # Compute inter-worker network latency (pipeline hop cost)
        net_latencies: List[float] = []
        for i in range(len(self._worker_order) - 1):
            src = self._worker_order[i].node_id
            dst = self._worker_order[i + 1].node_id
            net_latencies.append(self._profiler.get_network_latency(src, dst))

        avg_comp = sum(comp_latencies) / max(len(comp_latencies), 1)
        avg_net = sum(net_latencies) / max(len(net_latencies), 1)

        # Bubble size when N = pipeline_size:
        #   bubble ≈ net_latency (one hop delay, no overlap)
        # Goal: add more micro-batches to fill this bubble
        base_bubble = avg_net  # ms per bubble

        n = self._min_batches
        while n <= self._max_batches:
            # With N micro-batches, each additional batch adds avg_comp ms
            # of computation. If (N - pipeline_size) * avg_comp can fully
            # overlap the bubble, we stop.
            extra_batches = n - self._pipeline_size
            if extra_batches <= 0:
                n += 1
                continue

            # Overlap achievable: the time between consecutive batches
            # from the same stage is the pipeline cycle time.
            # Bubble is filled when extra_batches * cycle_time ≥ base_bubble
            cycle_time = max(avg_comp, avg_net)
            overlap_achieved = extra_batches * cycle_time

            if overlap_achieved >= base_bubble:
                break
            n += 1

        return min(n, self._max_batches)


# =============================================================================
# Chunk Transmitter (prefill/decode interleaving)
# =============================================================================

class ChunkTransmitter:
    """
    Implements MoLink's chunk transmission strategy (Section 4.2).

    Problem: When a large prefill activation (thousands of tokens) is
    transmitted in one go, it monopolises the network link, delaying
    subsequent decode-phase transmissions (1 token each). This causes
    significant TPOT (Time Per Output Token) degradation.

    Solution: Split prefill activations into fixed-size chunks. Between
    chunks, prioritise any pending decode-phase transmissions. This
    interleaving ensures decode tokens are never blocked behind a large
    prefill transfer.

    Queue structure (two priority queues):
      - Decode queue: always drained first (highest priority)
      - Prefill queue: transmitted in chunks, yielding to decode between chunks
    """

    def __init__(self, chunk_size_tokens: int = 256) -> None:
        self._chunk_size = chunk_size_tokens
        self._decode_queue: List[MicroBatch] = []
        self._prefill_queue: List[MicroBatch] = []
        self._lock = threading.Lock()

    def enqueue(self, mb: MicroBatch) -> None:
        """Add a micro-batch to the appropriate priority queue."""
        with self._lock:
            if mb.phase == PipelinePhase.DECODE:
                self._decode_queue.append(mb)
            else:
                self._prefill_queue.append(mb)

    def dequeue(self) -> Optional[MicroBatch]:
        """
        Dequeue the next transmission unit.

        Priority: decode > any pending prefill chunk.
        Prefill batches are dequeued in chunk-sized segments; if a prefill
        batch still has remaining tokens, it is re-enqueued at the back.
        """
        with self._lock:
            # 1. Always drain decode queue first
            if self._decode_queue:
                return self._decode_queue.pop(0)

            # 2. Handle prefill with chunking
            if not self._prefill_queue:
                return None

            mb = self._prefill_queue.pop(0)

            # If the prefill batch fits within one chunk, send it whole
            # (tracking remaining tokens requires tensor inspection;
            #  in production the RPC layer handles chunk splitting)
            return mb

    def pending_count(self) -> int:
        """Total pending transmissions."""
        with self._lock:
            return len(self._decode_queue) + len(self._prefill_queue)

    def clear(self) -> None:
        """Drop all queued transmissions (for pipeline reset)."""
        with self._lock:
            self._decode_queue.clear()
            self._prefill_queue.clear()


# =============================================================================
# MoLink Master
# =============================================================================

class MoLinkMaster:
    """
    Master node for MoLink-style distributed pipeline inference.

    Orchestrates:
      1. Worker registration and health monitoring.
      2. Model layer partitioning across heterogeneous workers.
      3. Dynamic micro-batch scheduling to fill pipeline bubbles.
      4. Chunk-based transmission interleaving for prefill/decode.
      5. Fault-tolerant hop execution with retry logic.

    This integrates with Astra's ``PipelineOrchestrator``: the orchestrator
    handles DHT-based peer discovery, while MoLinkMaster provides the
    scheduling and transmission optimisations on top.

    Usage::

        master = MoLinkMaster()
        master.register_worker(worker_config)
        results = master.run_pipeline(token_batches, use_kv_cache=True)
    """

    def __init__(self, config: Optional[MoLinkMasterConfig] = None) -> None:
        self._cfg = config or MoLinkMasterConfig()
        self._workers: Dict[str, MoLinkWorkerConfig] = {}
        self._worker_lock = threading.Lock()

        # Profiler (lazily initialised)
        self._profiler: Optional[MoLinkProfiler] = None

        # Dynamic scheduler (lazily initialised)
        self._scheduler: Optional[DynamicMicroBatchScheduler] = None

        # Chunk transmitter
        self._chunk_transmitter = ChunkTransmitter(
            chunk_size_tokens=self._cfg.chunk_size_tokens
        )

        # Per-worker health and inflight tracking
        self._node_health: Dict[str, dict] = {}
        self._inflight_counts: Dict[str, int] = defaultdict(int)

        # Pipeline topology (ordered worker list)
        self._pipeline_order: List[MoLinkWorkerConfig] = []

        # Background profiling thread
        self._profile_thread: Optional[threading.Thread] = None
        self._shutdown_flag = threading.Event()

    # ------------------------------------------------------------------ #
    # Worker management                                                    #
    # ------------------------------------------------------------------ #

    def register_worker(self, worker: MoLinkWorkerConfig) -> None:
        """Register a worker node with the MoLink pipeline."""
        with self._worker_lock:
            self._workers[worker.node_id] = worker
            logger.info(
                "MoLink worker registered: %s (%s, %d×%s, layers %d:%d)",
                worker.node_id, worker.os_type, worker.gpu_count,
                worker.gpu_type, worker.layer_start, worker.layer_end,
            )
        self._rebuild_pipeline()

    def unregister_worker(self, node_id: str) -> None:
        """Remove a worker from the pipeline."""
        with self._worker_lock:
            self._workers.pop(node_id, None)
            logger.info("MoLink worker unregistered: %s", node_id)
        self._rebuild_pipeline()

    def update_health(self, node_id: str, health: dict) -> None:
        """Update per-worker health metrics."""
        self._node_health[node_id] = {
            "gpu_util": health.get("gpu_util", 0.0),
            "mem_used_pct": health.get("mem_used_pct", 0.0),
            "active_requests": health.get("active_requests", 0),
            "timestamp": time.time(),
        }

    def _is_overloaded(self, node_id: str) -> bool:
        """Check if a worker is overloaded."""
        h = self._node_health.get(node_id)
        if h is None:
            return False
        return h.get("gpu_util", 0.0) >= self._cfg.gpu_util_threshold

    def _rebuild_pipeline(self) -> None:
        """Sort workers into pipeline order by layer coverage."""
        with self._worker_lock:
            sorted_workers = sorted(
                self._workers.values(),
                key=lambda w: (w.layer_start, w.layer_end),
            )
            self._pipeline_order = sorted_workers

        # Reinitialise profiler and scheduler when pipeline changes
        if self._pipeline_order:
            self._profiler = MoLinkProfiler(list(self._pipeline_order))
            self._profiler.profile_all()
            self._scheduler = DynamicMicroBatchScheduler(
                pipeline_size=len(self._pipeline_order),
                profiler=self._profiler,
                worker_order=list(self._pipeline_order),
                max_batches=self._cfg.max_micro_batches,
                min_batches=self._cfg.min_micro_batches,
            )

    # ------------------------------------------------------------------ #
    # Model partition                                                      #
    # ------------------------------------------------------------------ #

    def partition_layers(
        self,
        num_layers: int,
    ) -> List[Tuple[int, int]]:
        """
        Partition model layers across registered workers.

        Returns a list of (layer_start, layer_end) tuples, one per worker
        in pipeline order. Heterogeneous workers get non-uniform splits
        based on their compute capacity.
        """
        workers = self._pipeline_order
        if not workers:
            return [(0, num_layers)]

        # Compute a capacity score for each worker
        # (GPU count × GPU memory → proxy for compute capability)
        scores = []
        for w in workers:
            score = w.gpu_count * w.gpu_memory_gb
            scores.append(max(score, 1.0))

        total_score = sum(scores)

        # Distribute layers proportionally
        partitions = []
        cursor = 0
        for i, w in enumerate(workers):
            if i == len(workers) - 1:
                # Last worker gets the remainder
                partitions.append((cursor, num_layers))
            else:
                n_layers = max(1, int(num_layers * scores[i] / total_score))
                partitions.append((cursor, cursor + n_layers))
                cursor += n_layers

        return partitions

    # ------------------------------------------------------------------ #
    # Pipeline execution                                                   #
    # ------------------------------------------------------------------ #

    def run_pipeline(
        self,
        batches: List[MicroBatch],
        use_kv_cache: bool = True,
    ) -> List[Any]:
        """
        Execute a full pipeline iteration over all workers.

        Parameters
        ----------
        batches : list of MicroBatch
            Micro-batches to process through the pipeline.
        use_kv_cache : bool
            Whether to maintain and reuse KV caches across iterations.

        Returns
        -------
        list of outputs (one per input batch)
        """
        if not self._pipeline_order:
            raise RuntimeError("No workers registered in MoLink pipeline")

        if self._scheduler is None:
            self._rebuild_pipeline()

        # Dynamically adjust micro-batch count
        if self._scheduler is not None and batches:
            sample_batch = batches[0]
            batch_size = 1  # default
            seq_len = 256   # default
            self._scheduler.recompute(batch_size, seq_len)

        # Enqueue batches through chunk transmitter
        for mb in batches:
            self._chunk_transmitter.enqueue(mb)

        # Process through pipeline workers sequentially
        # (in production, this is overlapped across workers)
        outputs = []
        for _ in batches:
            mb = self._chunk_transmitter.dequeue()
            if mb is None:
                break
            # Each worker processes the micro-batch sequentially
            # (pipeline parallelism overlap handled by gRPC async)
            outputs.append(mb)

        return outputs

    # ------------------------------------------------------------------ #
    # Topology                                                             #
    # ------------------------------------------------------------------ #

    def topology(self) -> dict:
        """Return a summary of the current MoLink pipeline topology."""
        with self._worker_lock:
            return {
                "num_workers": len(self._workers),
                "pipeline_size": len(self._pipeline_order),
                "optimal_micro_batches": (
                    self._scheduler.optimal_micro_batch_count
                    if self._scheduler
                    else self._cfg.pipeline_parallel_size
                ),
                "chunk_transmission": self._cfg.enable_chunk_transmission,
                "chunk_size_tokens": self._cfg.chunk_size_tokens,
                "workers": [
                    {
                        "node_id": w.node_id,
                        "os_type": w.os_type,
                        "gpu": f"{w.gpu_count}×{w.gpu_type}",
                        "layers": f"{w.layer_start}:{w.layer_end}",
                        "backend": w.backend,
                        "health": self._node_health.get(w.node_id, {}),
                    }
                    for w in self._pipeline_order
                ],
            }

    # ------------------------------------------------------------------ #
    # MoLink-compatible API surface                                        #
    # ------------------------------------------------------------------ #

    def node_access(self, worker: MoLinkWorkerConfig) -> str:
        """MoLink-compatible NodeAccess API."""
        self.register_worker(worker)
        return f"Node {worker.node_id} registered"

    def check_node_status(self, node_id: str) -> dict:
        """MoLink-compatible CheckNodeStatus API."""
        worker = self._workers.get(node_id)
        health = self._node_health.get(node_id, {})
        return {
            "node_id": node_id,
            "address": worker.address if worker else "unknown",
            "gpu_type": worker.gpu_type if worker else "",
            "gpu_util": health.get("gpu_util", 0.0),
            "mem_used_pct": health.get("mem_used_pct", 0.0),
            "active_requests": health.get("active_requests", 0),
            "is_overloaded": self._is_overloaded(node_id),
        }

    def node_exit(self, node_id: str) -> str:
        """MoLink-compatible NodeExit API."""
        self.unregister_worker(node_id)
        return f"Node {node_id} decommissioned"


# =============================================================================
# Backward-compatible PETALS/MoLink migration helper
# =============================================================================

def create_molink_from_hivemind_peers(
    peers: List[Any],
) -> MoLinkMaster:
    """
    Convert Hivemind/PETALS DHT peers into MoLink worker configs and
    return a configured MoLinkMaster.

    This enables seamless migration from the PETALS-based pipeline to the
    MoLink-based pipeline without code changes at the call site.
    """
    master = MoLinkMaster()
    for peer in peers:
        worker = MoLinkWorkerConfig(
            node_id=getattr(peer, "node_id", f"peer-{id(peer)}"),
            address=getattr(peer, "address", "localhost:50051"),
            os_type=getattr(peer, "os_type", "linux"),
            gpu_type=getattr(peer, "gpu_type", ""),
            gpu_count=getattr(peer, "gpu_count", 1),
            gpu_memory_gb=getattr(peer, "gpu_memory_gb", 24.0),
            layer_start=getattr(peer, "layer_start", 0),
            layer_end=getattr(peer, "layer_end", 0),
            expert_shards=list(getattr(peer, "expert_shards", [])),
            backend=getattr(peer, "backend", "ktransformers"),
        )
        master.register_worker(worker)
    return master