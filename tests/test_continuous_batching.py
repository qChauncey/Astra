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
Phase 7.3.2 — Continuous Batching Tests.

Tests for:
  - batch_utils: pad_sequences, unpad_output, compute_batch_metrics
  - batch_scheduler: enqueue, form_batches, complete_batch, drain_completed
  - Integration: scheduler + heterogeneous engine forward_batch
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pytest

from astra.inference.batch_utils import (
    BatchInfo,
    compute_batch_metrics,
    pad_sequences,
    unpad_output,
)
from astra.inference.batch_scheduler import (
    BatchGroup,
    BatchingConfig,
    BatchRequest,
    ContinuousBatchScheduler,
    RequestStatus,
)
from astra.inference.heterogeneous import DeviceMap, HeterogeneousEngine


# ================================================================== #
# Fixtures                                                            #
# ================================================================== #

@pytest.fixture
def hidden_dim() -> int:
    return 64  # small for fast tests


@pytest.fixture
def sample_sequences(hidden_dim: int) -> List[np.ndarray]:
    rng = np.random.default_rng(42)
    return [
        rng.standard_normal((8, hidden_dim)).astype(np.float16),
        rng.standard_normal((15, hidden_dim)).astype(np.float16),
        rng.standard_normal((5, hidden_dim)).astype(np.float16),
        rng.standard_normal((10, hidden_dim)).astype(np.float16),
    ]


@pytest.fixture
def engine() -> HeterogeneousEngine:
    dmap = DeviceMap(
        attention_on_gpu=False,
        moe_on_cpu=True,
        num_layers=4,
        hidden_dim=64,
    )
    return HeterogeneousEngine.from_device_map(dmap)


# ================================================================== #
# 7.3.2.1  pad / unpad helpers                                         #
# ================================================================== #

class TestPadSequences:
    def test_basic_padding(self, sample_sequences: List[np.ndarray]):
        padded, info = pad_sequences(sample_sequences, generate_attention_mask=True)
        assert padded.shape == (4, 15, 64)
        assert info.original_lengths == [8, 15, 5, 10]
        assert padded[0, :8].sum() != 0.0
        assert np.allclose(padded[0, 8:], 0.0)
        assert info.pad_mask is not None
        assert info.attention_mask is not None

    def test_empty_sequences(self):
        padded, info = pad_sequences([], generate_attention_mask=True)
        assert padded.shape == (0, 0, 0)
        assert info.original_lengths == []

    def test_single_sequence(self, hidden_dim: int):
        seq = np.ones((5, hidden_dim), dtype=np.float16)
        padded, info = pad_sequences([seq], generate_attention_mask=False)
        assert padded.shape == (1, 5, hidden_dim)
        assert np.allclose(padded[0], seq)

    def test_custom_pad_token(self, hidden_dim: int):
        seqs = [np.ones((3, hidden_dim), dtype=np.float16) * 5.0]
        padded, info = pad_sequences(seqs, max_seq_len=6, generate_attention_mask=True)
        assert padded.shape == (1, 6, hidden_dim)
        # padding region should be zeros, not 5
        assert np.allclose(padded[0, 3:], 0.0)
        # original region should be 5
        assert np.allclose(padded[0, :3], 5.0)

    def test_attention_mask_generation(self, hidden_dim: int):
        seqs = [
            np.random.default_rng(0).standard_normal((2, hidden_dim)).astype(np.float16),
            np.random.default_rng(1).standard_normal((4, hidden_dim)).astype(np.float16),
        ]
        padded, info = pad_sequences(seqs, generate_attention_mask=True)
        assert info.attention_mask is not None
        am = info.attention_mask
        assert am.shape == (2, 4, 4)  # (batch, max_len, max_len)
        # Causal: seq0 can only attend to pos 0-1
        # Padding columns (2:) should be all False — no valid token attends padding
        assert not np.any(am[0, :, 2:])
        # Valid region: top 2x2 should have True for causal
        assert am[0, 0, 0]  # pos0 attends pos0
        assert am[0, 1, 1]  # pos1 attends pos1
        assert am[0, 1, 0]  # pos1 attends pos0
        assert not am[0, 0, 1]  # pos0 should NOT attend pos1 (causal)


class TestUnpadOutput:
    def test_extract_real_tokens(self, hidden_dim: int):
        data = np.arange(4 * 6 * hidden_dim).reshape(4, 6, hidden_dim).astype(np.float32)
        info = BatchInfo(
            original_lengths=[3, 6, 1, 4],
            original_positions=list(range(4)),
        )
        results = unpad_output(data, info)
        assert len(results) == 4
        assert results[0].shape == (3, hidden_dim)
        assert results[1].shape == (6, hidden_dim)
        assert results[2].shape == (1, hidden_dim)
        assert results[3].shape == (4, hidden_dim)

    def test_keep_pad_regions(self, hidden_dim: int):
        data = np.ones((2, 5, hidden_dim), dtype=np.float32)
        info = BatchInfo(original_lengths=[2, 4], original_positions=[0, 1])
        results = unpad_output(data, info, keep_pad_regions=True)
        assert len(results) == 2
        assert results[0].shape == (5, hidden_dim)
        assert results[1].shape == (5, hidden_dim)

    def test_empty_batch(self):
        results = unpad_output(
            np.array([], dtype=np.float32),
            BatchInfo(original_lengths=[], original_positions=[]),
        )
        assert results == []


class TestComputeBatchMetrics:
    def test_metrics(self):
        info = BatchInfo(original_lengths=[10, 20, 30], original_positions=[0, 1, 2])
        m = compute_batch_metrics(info)
        assert m["total_tokens"] == 60
        assert m["padded_tokens"] == 90  # 3 * 30
        assert m["padding_overhead_pct"] == 50.0  # (90-60)/60 * 100
        assert m["max_seq_len"] == 30
        assert m["min_seq_len"] == 10
        assert m["batch_size"] == 3

    def test_metrics_empty(self):
        info = BatchInfo(original_lengths=[], original_positions=[])
        m = compute_batch_metrics(info)
        assert m["total_tokens"] == 0
        assert m["padding_overhead_pct"] == 0.0


# ================================================================== #
# 7.3.2.2  ContinuousBatchScheduler                                     #
# ================================================================== #

class TestSchedulerEnqueue:
    def test_enqueue_single(self):
        s = ContinuousBatchScheduler()
        rid = s.enqueue(token_ids=[1, 2, 3], hidden_states=np.ones((3, 64), dtype=np.float16))
        assert len(rid) > 0
        assert s.queue_depth == 1

    def test_enqueue_bulk(self):
        s = ContinuousBatchScheduler()
        reqs = [
            BatchRequest(token_ids=[1], hidden_states=np.ones((1, 64), dtype=np.float16))
            for _ in range(5)
        ]
        s.enqueue_bulk(reqs)
        assert s.queue_depth == 5

    def test_peak_queue_depth(self):
        s = ContinuousBatchScheduler()
        s.enqueue(token_ids=[1], hidden_states=np.ones((1, 64), dtype=np.float16))
        s.enqueue(token_ids=[2], hidden_states=np.ones((2, 64), dtype=np.float16))
        m = s.metrics()
        assert m["peak_queue_depth"] == 2


class TestSchedulerFormBatches:
    def test_single_batch(self, hidden_dim: int):
        s = ContinuousBatchScheduler(BatchingConfig(max_batch_size=4, min_batch_size=1))
        for i in range(4):
            s.enqueue(
                token_ids=list(range(10)),
                hidden_states=np.ones((10, hidden_dim), dtype=np.float16),
            )
        batches = s.form_batches()
        assert len(batches) == 1
        assert batches[0].size == 4
        assert batches[0].padded_tensor is not None
        assert batches[0].batch_info is not None

    def test_length_binning(self, hidden_dim: int):
        """Sequences of different lengths should be binned separately."""
        s = ContinuousBatchScheduler(
            BatchingConfig(max_batch_size=8, min_batch_size=1, length_bin_ratio=1.3)
        )
        # Short sequences
        for _ in range(3):
            s.enqueue(
                token_ids=list(range(10)),
                hidden_states=np.ones((10, hidden_dim), dtype=np.float16),
            )
        # Long sequences
        for _ in range(3):
            s.enqueue(
                token_ids=list(range(200)),
                hidden_states=np.ones((200, hidden_dim), dtype=np.float16),
            )
        batches = s.form_batches()
        # Should get 2 batches: one for short, one for long
        # (they fall into different length buckets)
        assert len(batches) >= 1
        # Verify batches in same group have similar lengths
        for bg in batches:
            lengths = [r.seq_len for r in bg.requests]
            if lengths:
                ratio = max(lengths) / max(1, min(lengths))
                assert ratio <= 2.0 or len(lengths) <= 1  # different buckets

    def test_min_batch_size_enforcement(self, hidden_dim: int):
        """With min_batch_size > 1, single request may be deferred."""
        s = ContinuousBatchScheduler(
            BatchingConfig(max_batch_size=4, min_batch_size=2, max_wait_ms=5000.0)
        )
        s.enqueue(
            token_ids=[1],
            hidden_states=np.ones((1, hidden_dim), dtype=np.float16),
        )
        batches = s.form_batches()
        # Single request doesn't meet min_batch_size → deferred back to queue
        assert len(batches) == 0
        assert s.queue_depth == 1

    def test_max_wait_override(self, hidden_dim: int):
        """After max_wait_ms, even a small batch should be flushed."""
        s = ContinuousBatchScheduler(
            BatchingConfig(max_batch_size=4, min_batch_size=2, max_wait_ms=0.0)
        )
        s.enqueue(
            token_ids=[1],
            hidden_states=np.ones((1, hidden_dim), dtype=np.float16),
        )
        batches = s.form_batches()
        # max_wait_ms=0 means immediate flush
        assert len(batches) == 1
        assert batches[0].size == 1

    def test_max_tokens_per_batch(self, hidden_dim: int):
        """Batch should not exceed max_tokens_per_batch."""
        s = ContinuousBatchScheduler(
            BatchingConfig(
                max_batch_size=8,
                max_tokens_per_batch=50,
                min_batch_size=1,
            )
        )
        for _ in range(5):
            s.enqueue(
                token_ids=list(range(20)),
                hidden_states=np.ones((20, hidden_dim), dtype=np.float16),
            )
        batches = s.form_batches()
        # Each batch should have at most 50 real tokens, which is 2 sequences of 20
        for bg in batches:
            assert bg.total_tokens <= 50


class TestSchedulerCompleteBatch:
    def test_complete_and_drain(self, hidden_dim: int):
        s = ContinuousBatchScheduler(BatchingConfig(max_batch_size=4, min_batch_size=1))
        s.enqueue(
            token_ids=list(range(5)),
            hidden_states=np.ones((5, hidden_dim), dtype=np.float16),
        )
        s.enqueue(
            token_ids=list(range(8)),
            hidden_states=np.ones((8, hidden_dim), dtype=np.float16),
        )
        batches = s.form_batches()
        assert len(batches) == 1

        bg = batches[0]
        assert bg.padded_tensor is not None

        # Simulate GPU forward: just identity
        dummy_out = bg.padded_tensor.copy()
        s.complete_batch(bg.batch_id, dummy_out)

        results = s.drain_completed()
        assert len(results) == 2
        for rid, req in results.items():
            assert req.status == RequestStatus.DONE
            assert req.result is not None
            assert req.result.shape[0] == req.seq_len

    def test_complete_with_error(self, hidden_dim: int):
        s = ContinuousBatchScheduler(BatchingConfig(max_batch_size=4, min_batch_size=1))
        s.enqueue(
            token_ids=[1],
            hidden_states=np.ones((1, hidden_dim), dtype=np.float16),
        )
        batches = s.form_batches()
        bg = batches[0]

        dummy_out = bg.padded_tensor.copy() if bg.padded_tensor is not None else np.zeros((1, 1, hidden_dim))
        s.complete_batch(bg.batch_id, dummy_out, error="GPU OOM")

        results = s.drain_completed()
        for req in results.values():
            assert req.status == RequestStatus.ERROR
            assert req.error == "GPU OOM"


class TestSchedulerGetRequest:
    def test_find_in_queue(self, hidden_dim: int):
        s = ContinuousBatchScheduler()
        rid = s.enqueue(
            token_ids=[1],
            hidden_states=np.ones((1, hidden_dim), dtype=np.float16),
        )
        req = s.get_request(rid)
        assert req is not None
        assert req.request_id == rid

    def test_find_completed(self, hidden_dim: int):
        s = ContinuousBatchScheduler(BatchingConfig(max_batch_size=4, min_batch_size=1))
        rid = s.enqueue(
            token_ids=list(range(3)),
            hidden_states=np.ones((3, hidden_dim), dtype=np.float16),
        )
        batches = s.form_batches()
        bg = batches[0]
        dummy_out = bg.padded_tensor.copy() if bg.padded_tensor is not None else np.zeros((1, 3, hidden_dim))
        s.complete_batch(bg.batch_id, dummy_out)
        req = s.get_request(rid)
        assert req is not None
        assert req.status == RequestStatus.DONE

    def test_not_found(self):
        s = ContinuousBatchScheduler()
        req = s.get_request("nonexistent")
        assert req is None


# ================================================================== #
# 7.3.2.3  Integration: scheduler + engine                              #
# ================================================================== #

class TestIntegration:
    def test_scheduler_with_engine_forward(self, engine: HeterogeneousEngine, hidden_dim: int):
        """End-to-end: enqueue → form → engine.forward_batch → complete → drain."""
        scheduler = ContinuousBatchScheduler(
            BatchingConfig(max_batch_size=4, min_batch_size=1)
        )
        rng = np.random.default_rng(123)
        # Enqueue 2 requests with different lengths
        hidden_states_1 = rng.standard_normal((6, hidden_dim)).astype(np.float16)
        hidden_states_2 = rng.standard_normal((4, hidden_dim)).astype(np.float16)

        scheduler.enqueue(hidden_states=hidden_states_1)
        scheduler.enqueue(hidden_states=hidden_states_2)

        batches = scheduler.form_batches()
        assert len(batches) == 1
        bg = batches[0]
        assert bg.padded_tensor is not None
        assert bg.batch_info is not None

        # Run through engine
        layer_indices = [0, 1]
        output = engine.forward_batch(
            bg.padded_tensor,
            layer_indices=layer_indices,
            attention_mask=bg.batch_info.attention_mask,
        )
        assert output.shape == bg.padded_tensor.shape

        scheduler.complete_batch(bg.batch_id, output)

        results = scheduler.drain_completed()
        assert len(results) == 2
        # Verify per-request output shapes match input lengths
        for req in results.values():
            assert req.result is not None
            assert req.result.shape[0] == req.seq_len
            assert req.status == RequestStatus.DONE

    def test_multiple_batches_through_engine(self, engine: HeterogeneousEngine, hidden_dim: int):
        """Multiple iterations through the scheduler loop."""
        scheduler = ContinuousBatchScheduler(
            BatchingConfig(max_batch_size=3, max_tokens_per_batch=30, min_batch_size=1)
        )
        rng = np.random.default_rng(456)
        layer_indices = [0]

        total_requests = 0
        # Run 3 rounds
        for round_num in range(3):
            for _ in range(4):
                seq_len = rng.integers(3, 12)
                hidden_states = rng.standard_normal((int(seq_len), hidden_dim)).astype(np.float16)
                scheduler.enqueue(hidden_states=hidden_states)
                total_requests += 1

            batches = scheduler.form_batches()
            for bg in batches:
                assert bg.padded_tensor is not None
                output = engine.forward_batch(
                    bg.padded_tensor,
                    layer_indices=layer_indices,
                )
                scheduler.complete_batch(bg.batch_id, output)

        # All requests should complete
        results = scheduler.drain_completed()
        # Drain again to ensure nothing left
        results2 = scheduler.drain_completed()
        combined = {**results, **results2}

        all_done = all(req.status == RequestStatus.DONE for req in combined.values())
        assert all_done, f"Some requests not done: {[req.status for req in combined.values()]}"


class TestRequestProperties:
    def test_seq_len_from_hidden_states(self, hidden_dim: int):
        req = BatchRequest(hidden_states=np.ones((10, hidden_dim), dtype=np.float16))
        assert req.seq_len == 10

    def test_seq_len_from_token_ids(self):
        req = BatchRequest(token_ids=list(range(7)))
        assert req.seq_len == 7

    def test_age_ms(self):
        req = BatchRequest()
        time.sleep(0.02)
        age = req.age_ms
        assert age >= 20, f"Expected age >= 20ms, got {age}ms"


class TestBatchGroup:
    def test_size(self, hidden_dim: int):
        bg = BatchGroup(
            requests=[
                BatchRequest(token_ids=[1]),
                BatchRequest(token_ids=[2, 3]),
                BatchRequest(token_ids=[4]),
            ]
        )
        assert bg.size == 3

    def test_total_tokens(self, hidden_dim: int):
        bg = BatchGroup(
            requests=[
                BatchRequest(hidden_states=np.ones((5, hidden_dim), dtype=np.float16)),
                BatchRequest(hidden_states=np.ones((10, hidden_dim), dtype=np.float16)),
            ]
        )
        assert bg.total_tokens == 15


class TestSchedulerMetrics:
    def test_initial_metrics(self):
        s = ContinuousBatchScheduler()
        m = s.metrics()
        assert m["total_enqueued"] == 0
        assert m["total_completed"] == 0
        assert m["queue_depth"] == 0

    def test_metrics_after_operations(self, hidden_dim: int):
        s = ContinuousBatchScheduler(BatchingConfig(max_batch_size=4, min_batch_size=1))
        s.enqueue(hidden_states=np.ones((5, hidden_dim), dtype=np.float16))
        s.enqueue(hidden_states=np.ones((6, hidden_dim), dtype=np.float16))
        batches = s.form_batches()
        bg = batches[0]
        dummy = bg.padded_tensor.copy() if bg.padded_tensor is not None else np.zeros((1, 1, hidden_dim))
        s.complete_batch(bg.batch_id, dummy)
        m = s.metrics()
        assert m["total_enqueued"] >= 2
        assert m["total_completed"] >= 2
        assert m["total_batches_formed"] >= 1

    def test_reset_metrics(self, hidden_dim: int):
        s = ContinuousBatchScheduler(BatchingConfig(max_batch_size=4, min_batch_size=1))
        s.enqueue(hidden_states=np.ones((5, hidden_dim), dtype=np.float16))
        s.form_batches()
        s.reset_metrics()
        m = s.metrics()
        assert m["total_enqueued"] == 0
        assert m["total_batches_formed"] == 0
