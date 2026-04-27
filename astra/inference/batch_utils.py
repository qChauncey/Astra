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
Phase 7.3.2 — Batch Utilities: pad / unpad helpers for dynamic batching.

Supports variable-length sequences in a single GPU invocation by padding
all sequences to the batch maximum, tracking original lengths, and slicing
results back to the original per-sequence dimensions after the forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BatchInfo:
    """Metadata produced by padding; consumed by unpad to restore original shapes."""
    original_lengths: List[int]       # per-sequence token counts
    original_positions: List[int]     # per-sequence start position in the padded seq
    pad_mask: Optional[np.ndarray] = None  # (batch, max_seq_len) bool: True = padding
    attention_mask: Optional[np.ndarray] = None  # combined causal+pad mask
    sequence_ids: List[str] = field(default_factory=list)  # optional request IDs


def pad_sequences(
    sequences: List[Dict[int, np.ndarray]],
    max_seq_len: int = 0,
    pad_token_id: int = 0,
    generate_attention_mask: bool = True,
) -> Tuple[np.ndarray, BatchInfo]:
    """
    Pad a list of per-layer token embeddings to a uniform batch.

    Each entry in *sequences* is a dict `{layer_idx: (seq_len, hidden_dim)}`.
    All layers across all sequences must share the same hidden_dim.

    Returns:
      padded: (total_tokens, max_seq_len, hidden_dim) where total_tokens = sum(len(s)) but
              the output shape is (num_layers, batch_size, max_seq_len, hidden_dim) — NO,
              the contract is simpler:

      Actually we just need: for each sequence's token_embedding (seq_len, hidden_dim),
      produce a (batch, max_len, hidden) tensor and a pad_mask.

      But since different sequences may be at different layers in continuous batching,
      we treat each sequence independently and the scheduler handles layer alignment.
      So this function is per-layer: input is a list of (seq_len_i, hidden_dim) numpy arrays
      for a single layer.

    Simplified API:
      sequences: list of np.ndarray, each shape (seq_len_i, hidden_dim)
    Returns:
      padded_batch: (batch, max_len, hidden_dim)
      batch_info: BatchInfo with original lengths
    """
    if not sequences:
        return np.zeros((0, 0, 0), dtype=np.float32), BatchInfo(
            original_lengths=[], original_positions=[]
        )

    hidden_dim = sequences[0].shape[1]
    lengths = [s.shape[0] for s in sequences]
    batch_len = max(lengths) if max_seq_len <= 0 else max(max(lengths), max_seq_len)

    batch_size = len(sequences)
    padded = np.zeros((batch_size, batch_len, hidden_dim), dtype=sequences[0].dtype)

    for i, seq in enumerate(sequences):
        l = seq.shape[0]
        padded[i, :l, :] = seq

    pad_mask = None
    attn_mask = None
    if generate_attention_mask:
        pad_mask = np.ones((batch_size, batch_len), dtype=bool)
        for i, l in enumerate(lengths):
            pad_mask[i, :l] = False  # False = valid token
        # Causal + pad: True = attend (not masked)
        causal = np.tri(batch_len, batch_len, k=0, dtype=bool)  # (max_len, max_len)
        attn_mask = causal[np.newaxis, :, :] & ~pad_mask[:, np.newaxis, :]  # (batch, max, max)

    return padded, BatchInfo(
        original_lengths=lengths,
        original_positions=list(range(batch_size)),
        pad_mask=pad_mask,
        attention_mask=attn_mask,
    )


def unpad_output(
    padded_output: np.ndarray,
    batch_info: BatchInfo,
    keep_pad_regions: bool = False,
) -> List[np.ndarray]:
    """
    Extract per-sequence outputs from a padded batch result.

    padded_output: (batch, max_len, hidden_dim)
    batch_info: from pad_sequences

    Returns list of np.ndarray, each shape (original_len_i, hidden_dim).
    If keep_pad_regions is True, returns the full (max_len, hidden_dim) slices
    (useful for profiling padding overhead).
    """
    if padded_output.size == 0:
        return []

    results: List[np.ndarray] = []
    for i, orig_len in enumerate(batch_info.original_lengths):
        if keep_pad_regions:
            results.append(padded_output[i].copy())
        else:
            results.append(padded_output[i, :orig_len, :].copy())
    return results


def compute_batch_metrics(batch_info: BatchInfo) -> dict:
    """
    Compute efficiency metrics for a given batch padding.

    Returns dict with:
      - total_tokens: sum of real token counts
      - padded_tokens: batch_size * max_len
      - padding_overhead_pct: (padded - real) / real * 100
      - max_seq_len
      - min_seq_len
      - batch_size
    """
    lengths = batch_info.original_lengths
    if not lengths:
        return {
            "total_tokens": 0,
            "padded_tokens": 0,
            "padding_overhead_pct": 0.0,
            "max_seq_len": 0,
            "min_seq_len": 0,
            "batch_size": 0,
        }
    total = sum(lengths)
    max_l = max(lengths)
    padded_tokens = len(lengths) * max_l
    overhead = (padded_tokens - total) / max(1, total) * 100.0
    return {
        "total_tokens": total,
        "padded_tokens": padded_tokens,
        "padding_overhead_pct": round(overhead, 1),
        "max_seq_len": max_l,
        "min_seq_len": min(lengths),
        "batch_size": len(lengths),
    }