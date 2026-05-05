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

"""Unit tests for astra.inference.batch_utils."""

import numpy as np
import pytest
from astra.inference.batch_utils import (
    BatchInfo,
    compute_batch_metrics,
    pad_sequences,
    unpad_output,
)


# ---------------------------------------------------------------------------
# BatchInfo
# ---------------------------------------------------------------------------

class TestBatchInfo:
    def test_default_construction(self):
        info = BatchInfo(original_lengths=[], original_positions=[])
        assert info.original_lengths == []
        assert info.original_positions == []
        assert info.pad_mask is None
        assert info.attention_mask is None
        assert info.sequence_ids == []

    def test_full_construction(self):
        pad_mask = np.ones((2, 3), dtype=bool)
        attn_mask = np.ones((2, 3, 3), dtype=bool)
        info = BatchInfo(
            original_lengths=[2, 3],
            original_positions=[0, 1],
            pad_mask=pad_mask,
            attention_mask=attn_mask,
            sequence_ids=["req_0", "req_1"],
        )
        assert info.original_lengths == [2, 3]
        assert info.original_positions == [0, 1]
        np.testing.assert_array_equal(info.pad_mask, pad_mask)
        np.testing.assert_array_equal(info.attention_mask, attn_mask)
        assert info.sequence_ids == ["req_0", "req_1"]

    def test_sequence_ids_default_factory(self):
        info1 = BatchInfo(original_lengths=[], original_positions=[])
        info2 = BatchInfo(original_lengths=[], original_positions=[])
        assert info1.sequence_ids is not info2.sequence_ids  # independent lists


# ---------------------------------------------------------------------------
# pad_sequences
# ---------------------------------------------------------------------------

class TestPadSequences:
    def test_empty_sequences(self):
        padded, info = pad_sequences([])
        assert padded.shape == (0, 0, 0)
        assert info.original_lengths == []
        assert info.original_positions == []

    def test_single_sequence_no_mask(self):
        seq = np.random.randn(5, 64).astype(np.float32)
        padded, info = pad_sequences([seq], generate_attention_mask=False)
        assert padded.shape == (1, 5, 64)
        np.testing.assert_array_almost_equal(padded[0], seq)
        assert info.original_lengths == [5]
        assert info.pad_mask is None
        assert info.attention_mask is None

    def test_single_sequence_with_mask(self):
        seq = np.random.randn(5, 64).astype(np.float32)
        padded, info = pad_sequences([seq], generate_attention_mask=True)
        assert padded.shape == (1, 5, 64)
        assert info.pad_mask is not None
        assert info.pad_mask.shape == (1, 5)
        # No padding tokens in a single sequence => all False (valid)
        np.testing.assert_array_equal(info.pad_mask, np.zeros((1, 5), dtype=bool))
        assert info.attention_mask is not None
        assert info.attention_mask.shape == (1, 5, 5)

    def test_multiple_sequences_same_length(self):
        seqs = [np.random.randn(4, 32).astype(np.float32) for _ in range(3)]
        padded, info = pad_sequences(seqs, generate_attention_mask=False)
        assert padded.shape == (3, 4, 32)
        for i, seq in enumerate(seqs):
            np.testing.assert_array_almost_equal(padded[i], seq)
        assert info.original_lengths == [4, 4, 4]

    def test_multiple_sequences_variable_lengths(self):
        seqs = [
            np.random.randn(3, 16).astype(np.float32),
            np.random.randn(7, 16).astype(np.float32),
            np.random.randn(1, 16).astype(np.float32),
        ]
        padded, info = pad_sequences(seqs, generate_attention_mask=False)
        assert padded.shape == (3, 7, 16)
        # Check original content preserved
        np.testing.assert_array_almost_equal(padded[0, :3, :], seqs[0])
        np.testing.assert_array_almost_equal(padded[1, :7, :], seqs[1])
        np.testing.assert_array_almost_equal(padded[2, :1, :], seqs[2])
        # Padding region should be zeros
        assert np.all(padded[0, 3:, :] == 0)
        assert np.all(padded[2, 1:, :] == 0)
        assert info.original_lengths == [3, 7, 1]

    def test_pad_mask_generation(self):
        seqs = [
            np.random.randn(2, 8).astype(np.float32),
            np.random.randn(4, 8).astype(np.float32),
        ]
        padded, info = pad_sequences(seqs, generate_attention_mask=True)
        # pad_mask: False = valid token, True = padding
        expected_pad = np.array([[False, False, True, True],
                                  [False, False, False, False]])
        np.testing.assert_array_equal(info.pad_mask, expected_pad)
        # attention_mask shape
        assert info.attention_mask.shape == (2, 4, 4)
        # attention_mask[i, j, k] = causal[j,k] & ~pad_mask[i,j] & ~pad_mask[i,k]
        # That is: causal AND query-j is valid AND key-k is valid
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    causal = k <= j
                    query_valid = not (i == 0 and j >= 2)  # seq 0 pad at pos 2,3
                    key_valid = not (i == 0 and k >= 2)    # same for key side
                    expected = causal and query_valid and key_valid
                    assert info.attention_mask[i, j, k] == expected, \
                        f"attn[{i},{j},{k}] expected {expected}"

    def test_max_seq_len_override(self):
        seqs = [np.random.randn(3, 10).astype(np.float32)]
        padded, info = pad_sequences(seqs, max_seq_len=10, generate_attention_mask=False)
        assert padded.shape == (1, 10, 10)  # forced to 10 even though seq is 3
        np.testing.assert_array_almost_equal(padded[0, :3, :], seqs[0])
        assert np.all(padded[0, 3:, :] == 0)

    def test_preserves_dtype(self):
        seq = np.random.randn(3, 8).astype(np.float64)
        padded, _ = pad_sequences([seq], generate_attention_mask=False)
        assert padded.dtype == np.float64

    def test_nonuniform_hidden_dim_raises(self):
        seq1 = np.random.randn(3, 16).astype(np.float32)
        seq2 = np.random.randn(3, 32).astype(np.float32)  # different hidden_dim
        with pytest.raises(ValueError):
            pad_sequences([seq1, seq2])


# ---------------------------------------------------------------------------
# unpad_output
# ---------------------------------------------------------------------------

class TestUnpadOutput:
    def test_empty(self):
        result = unpad_output(np.zeros((0, 0, 0), dtype=np.float32),
                              BatchInfo(original_lengths=[], original_positions=[]))
        assert result == []

    def test_single_sequence(self):
        original = np.random.randn(3, 16).astype(np.float32)
        padded = np.tile(original[np.newaxis, ...], (1, 1, 1))
        info = BatchInfo(original_lengths=[3], original_positions=[0])
        result = unpad_output(padded, info)
        assert len(result) == 1
        np.testing.assert_array_almost_equal(result[0], original)

    def test_unpad_mixed_lengths(self):
        padded = np.random.randn(2, 6, 8).astype(np.float32)
        info = BatchInfo(original_lengths=[4, 6], original_positions=[0, 1])
        result = unpad_output(padded, info)
        assert len(result) == 2
        assert result[0].shape == (4, 8)
        assert result[1].shape == (6, 8)
        np.testing.assert_array_almost_equal(result[0], padded[0, :4, :])
        np.testing.assert_array_almost_equal(result[1], padded[1, :6, :])

    def test_keep_pad_regions(self):
        padded = np.random.randn(2, 5, 4).astype(np.float32)
        info = BatchInfo(original_lengths=[3, 2], original_positions=[0, 1])
        result = unpad_output(padded, info, keep_pad_regions=True)
        assert len(result) == 2
        assert result[0].shape == (5, 4)
        assert result[1].shape == (5, 4)
        # Full slices including padding
        np.testing.assert_array_equal(result[0], padded[0])
        np.testing.assert_array_equal(result[1], padded[1])

    def test_does_not_mutate_input(self):
        padded = np.random.randn(1, 3, 4).astype(np.float32)
        original = padded.copy()
        info = BatchInfo(original_lengths=[3], original_positions=[0])
        unpad_output(padded, info)
        np.testing.assert_array_equal(padded, original)


# ---------------------------------------------------------------------------
# compute_batch_metrics
# ---------------------------------------------------------------------------

class TestComputeBatchMetrics:
    def test_empty(self):
        info = BatchInfo(original_lengths=[], original_positions=[])
        metrics = compute_batch_metrics(info)
        assert metrics == {
            "total_tokens": 0,
            "padded_tokens": 0,
            "padding_overhead_pct": 0.0,
            "max_seq_len": 0,
            "min_seq_len": 0,
            "batch_size": 0,
        }

    def test_uniform_sequences_no_overhead(self):
        info = BatchInfo(original_lengths=[5, 5, 5], original_positions=[0, 1, 2])
        metrics = compute_batch_metrics(info)
        assert metrics["total_tokens"] == 15
        assert metrics["padded_tokens"] == 15  # 3 * 5
        assert metrics["padding_overhead_pct"] == 0.0
        assert metrics["max_seq_len"] == 5
        assert metrics["min_seq_len"] == 5
        assert metrics["batch_size"] == 3

    def test_variable_sequences_with_overhead(self):
        info = BatchInfo(original_lengths=[1, 3, 5], original_positions=[0, 1, 2])
        metrics = compute_batch_metrics(info)
        assert metrics["total_tokens"] == 9
        assert metrics["padded_tokens"] == 15  # 3 * 5
        # overhead = (15 - 9) / 9 * 100 = 66.7
        assert metrics["padding_overhead_pct"] == 66.7
        assert metrics["max_seq_len"] == 5
        assert metrics["min_seq_len"] == 1
        assert metrics["batch_size"] == 3

    def test_single_sequence(self):
        info = BatchInfo(original_lengths=[10], original_positions=[0])
        metrics = compute_batch_metrics(info)
        assert metrics["total_tokens"] == 10
        assert metrics["padded_tokens"] == 10
        assert metrics["padding_overhead_pct"] == 0.0
        assert metrics["batch_size"] == 1


# ---------------------------------------------------------------------------
# Integration: pad → unpad roundtrip
# ---------------------------------------------------------------------------

class TestPadUnpadRoundtrip:
    def test_roundtrip_preserves_content(self):
        seqs = [
            np.random.randn(2, 32).astype(np.float32),
            np.random.randn(5, 32).astype(np.float32),
            np.random.randn(3, 32).astype(np.float32),
        ]
        padded, info = pad_sequences(seqs, generate_attention_mask=False)
        recovered = unpad_output(padded, info)
        assert len(recovered) == len(seqs)
        for orig, rec in zip(seqs, recovered):
            np.testing.assert_array_almost_equal(orig, rec)

    def test_roundtrip_single_sequence(self):
        seq = np.random.randn(100, 64).astype(np.float32)
        padded, info = pad_sequences([seq], generate_attention_mask=True)
        # Verify attention mask is valid causal for a single sequence
        assert info.attention_mask.shape == (1, 100, 100)
        causal = np.tri(100, 100, k=0, dtype=bool)
        np.testing.assert_array_equal(info.attention_mask[0], causal)
        recovered = unpad_output(padded, info)
        np.testing.assert_array_almost_equal(recovered[0], seq)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_token_sequences(self):
        seqs = [np.random.randn(1, 64).astype(np.float32),
                np.random.randn(1, 64).astype(np.float32)]
        padded, info = pad_sequences(seqs, generate_attention_mask=True)
        assert padded.shape == (2, 1, 64)
        assert info.attention_mask.shape == (2, 1, 1)
        # Single token => causal mask is just True
        np.testing.assert_array_equal(info.attention_mask, np.ones((2, 1, 1), dtype=bool))

    def test_large_batch(self):
        batch_size = 100
        seqs = [np.random.randn(3, 4).astype(np.float32) for _ in range(batch_size)]
        padded, info = pad_sequences(seqs, generate_attention_mask=False)
        assert padded.shape == (batch_size, 3, 4)
        assert info.original_lengths == [3] * batch_size

    def test_zero_length_sequence_in_batch_raises(self):
        """A zero-length sequence should be handled gracefully."""
        seqs = [
            np.random.randn(3, 8).astype(np.float32),
            np.zeros((0, 8), dtype=np.float32),  # empty seq
        ]
        # Should not raise — works because max length is 3
        padded, info = pad_sequences(seqs, generate_attention_mask=False)
        assert padded.shape == (2, 3, 8)
        assert info.original_lengths == [3, 0]
        # Zero-length sequence region is all zeros
        assert np.all(padded[1] == 0)