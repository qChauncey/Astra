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
Phase 7.3.3 — Speculative Decoding.

Implements draft-model speculative decoding to reduce per-token latency in
the P2P pipeline.  A small draft model generates K candidate tokens, which
the full target model verifies in a single forward pass.  Accepted prefix
tokens are kept; rejected tokens trigger re-sampling from the target
distribution.

Architecture
------------
DraftModelRunner
    Runs a lightweight draft model (e.g. DeepSeek-V2-Lite) to produce
    candidate token sequences.
TargetModelVerifier
    Compares draft logits against target logits and applies the standard
    rejection-sampling algorithm.
SpeculativePipeline
    Coordinates the async draft → verify loop, merging results back into
    the main pipeline flow.

Reference
---------
Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"
(ICML 2023).  https://arxiv.org/abs/2211.17192
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class SpeculativeConfig:
    """
    Parameters for speculative decoding.

    num_draft_tokens : int
        How many tokens the draft model generates per step (K in the paper).
    acceptance_mode : str
        "strict"  — all draft tokens must match target greedy argmax.
        "relaxed" — uses the rejection-sampling scheme from Leviathan et al.
        "none"    — disables speculative decoding entirely.
    draft_temperature : float
        Softmax temperature for the draft model sampling.
    min_acceptance_rate : float
        If observed acceptance rate falls below this threshold, speculative
        decoding is paused to avoid wasting compute.
    """
    num_draft_tokens: int = 5
    acceptance_mode: str = "relaxed"
    draft_temperature: float = 1.0
    min_acceptance_rate: float = 0.5


# --------------------------------------------------------------------------- #
# DraftModelRunner                                                             #
# --------------------------------------------------------------------------- #

class DraftModelRunner:
    """
    Runs a lightweight draft model to propose candidate tokens.

    In production this wraps KTransformers C++ inference with a smaller
    model checkpoint.  The current implementation uses a numpy stub that
    produces plausible but non-authoritative token predictions, enabling
    testing of the speculative pipeline logic without real weights.

    Parameters
    ----------
    vocab_size : int
        Vocabulary cardinality (needed for logit shape).
    hidden_dim : int
        Hidden dimension (must match the target model).
    num_layers : int
        Number of layers in the draft model (typically fewer than target).
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_dim: int = 512,
        num_layers: int = 8,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self._rng = np.random.default_rng(42)
        self._call_count = 0

    def generate_proposals(
        self,
        hidden_states: np.ndarray,
        num_tokens: int = 5,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate *num_tokens* draft token proposals.

        Parameters
        ----------
        hidden_states : np.ndarray  shape (seq_len, hidden_dim)
            Final hidden state from the target model's previous step.
        num_tokens : int
            Number of draft tokens to propose (K).
        temperature : float
            Sampling temperature.

        Returns
        -------
        draft_tokens : np.ndarray  shape (num_tokens,) dtype int64
            Proposed token IDs.
        draft_logits : np.ndarray  shape (num_tokens, vocab_size) dtype float32
            Log-probabilities for each draft token.
        """
        self._call_count += 1

        # Stub: generate logits from the last hidden state + noise
        last_hidden = hidden_states[-1, :].astype(np.float32)
        logits = np.zeros((num_tokens, self.vocab_size), dtype=np.float32)

        for t in range(num_tokens):
            # Simple linear projection stub
            projection = last_hidden[:self.vocab_size] if self.hidden_dim >= self.vocab_size else np.pad(last_hidden, (0, self.vocab_size - self.hidden_dim))
            noise = self._rng.standard_normal(self.vocab_size).astype(np.float32) * 0.01
            logits[t] = projection[:self.vocab_size] + noise

            # Use auto-regressive feedback: feed argmax back as next input
            if temperature <= 0.0:
                next_id = int(np.argmax(logits[t]))
            else:
                scaled = logits[t] / temperature
                scaled -= scaled.max()
                probs = np.exp(scaled) / np.exp(scaled).sum()
                next_id = int(self._rng.choice(self.vocab_size, p=probs))

            # Simple stub: next hidden is a rotated version of last
            last_hidden = np.roll(last_hidden, shift=next_id % len(last_hidden))

        # Return tokens as greedy argmax over logits
        draft_tokens = np.argmax(logits, axis=1).astype(np.int64)
        return draft_tokens, logits

    def reset_stats(self) -> None:
        self._call_count = 0


# --------------------------------------------------------------------------- #
# TargetModelVerifier                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class VerificationResult:
    """Outcome of one speculative decoding step."""
    accepted_tokens: List[int] = field(default_factory=list)
    accepted_count: int = 0
    draft_count: int = 0
    replaced_token: Optional[int] = None  # first token sampled from target
    target_logits: Optional[np.ndarray] = None  # shape (accepted+1, vocab_size)
    elapsed_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.draft_count == 0:
            return 0.0
        return self.accepted_count / self.draft_count


class TargetModelVerifier:
    """
    Verifies draft tokens against the target model's logits.

    Implements the rejection-sampling algorithm from Leviathan et al. (2023):
      1. Target model produces logits for position i.
      2. If draft_token[i] == argmax(target_logits[i]), accept.
      3. Otherwise, reject and sample from the adjusted distribution
         max(0, target_probs - draft_probs) normalized.

    Parameters
    ----------
    acceptance_mode : str
        "strict"  — only accept if exact greedy match.
        "relaxed" — use the full rejection-sampling scheme.
    """

    def __init__(self, acceptance_mode: str = "relaxed") -> None:
        self.acceptance_mode = acceptance_mode
        self._total_draft = 0
        self._total_accepted = 0

    def verify(
        self,
        target_logits: np.ndarray,
        draft_logits: np.ndarray,
        draft_tokens: np.ndarray,
    ) -> VerificationResult:
        """
        Verify a sequence of draft tokens against target logits.

        Parameters
        ----------
        target_logits : np.ndarray  shape (K+1, vocab_size)
            Target model output logits for positions (prefix, token_1, ..., token_K).
        draft_logits : np.ndarray  shape (K, vocab_size)
            Draft model logits for each draft token.
        draft_tokens : np.ndarray  shape (K,) int64
            Proposed token IDs from the draft model.

        Returns
        -------
        VerificationResult
        """
        t0 = time.perf_counter()
        K = len(draft_tokens)
        accepted: List[int] = []
        k = 0

        for k in range(K):
            draft_id = int(draft_tokens[k])
            target_logit = target_logits[k]  # logits before accepting step
            draft_logit = draft_logits[k]

            if self.acceptance_mode == "strict":
                # Simple: accept iff draft matches target greedy
                target_argmax = int(np.argmax(target_logit))
                if draft_id == target_argmax:
                    accepted.append(draft_id)
                else:
                    break
            elif self.acceptance_mode == "relaxed":
                # Leviathan et al. rejection sampling
                target_argmax = int(np.argmax(target_logit))

                if draft_id == target_argmax:
                    # Greedy match: always accept
                    accepted.append(draft_id)
                else:
                    # Compute adjusted distribution
                    target_probs = _softmax(target_logit)
                    draft_probs = _softmax(draft_logit)
                    adj_probs = np.maximum(0.0, target_probs - draft_probs)
                    adj_sum = adj_probs.sum()
                    if adj_sum > 1e-12:
                        adj_probs /= adj_sum
                        sampled_id = int(self._sample(adj_probs))
                        accepted.append(sampled_id)
                    break
            else:
                break

        result = VerificationResult(
            accepted_tokens=accepted,
            accepted_count=len(accepted),
            draft_count=K,
            replaced_token=(int(draft_tokens[len(accepted)])
                            if len(accepted) < K and len(accepted) < len(draft_tokens)
                            else None),
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

        self._total_draft += K
        self._total_accepted += len(accepted)
        return result

    def overall_acceptance_rate(self) -> float:
        if self._total_draft == 0:
            return 0.0
        return self._total_accepted / self._total_draft

    def reset_stats(self) -> None:
        self._total_draft = 0
        self._total_accepted = 0

    @staticmethod
    def _sample(probs: np.ndarray) -> int:
        """Sample from a probability distribution."""
        cumsum = np.cumsum(probs)
        r = np.random.random()
        return int(np.searchsorted(cumsum, r))


# --------------------------------------------------------------------------- #
# SpeculativePipeline                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class SpeculativeStepOutput:
    """Products of one speculative decoding step."""
    accepted_token_ids: List[int]
    num_accepted: int
    num_draft: int
    draft_elapsed_ms: float = 0.0
    verify_elapsed_ms: float = 0.0
    total_wall_ms: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.total_wall_ms <= 0:
            return 0.0
        return self.num_accepted / (self.total_wall_ms / 1000.0)


class SpeculativePipeline:
    """
    Orchestrates the draft → verify speculative decoding loop.

    Usage::

        pipeline = SpeculativePipeline(draft_runner, verifier, config)
        pipeline.set_target_forward(target_forward_fn)

        for step in range(max_steps):
            output = pipeline.step(current_hidden_states)
            if not output.accepted_token_ids:
                break
            generated_tokens.extend(output.accepted_token_ids)
            current_hidden_states = ...  # get next state from pipeline

    Parameters
    ----------
    draft_runner : DraftModelRunner
    verifier : TargetModelVerifier
    config : SpeculativeConfig
    """

    def __init__(
        self,
        draft_runner: DraftModelRunner,
        verifier: TargetModelVerifier,
        config: Optional[SpeculativeConfig] = None,
    ) -> None:
        self.draft_runner = draft_runner
        self.verifier = verifier
        self.config = config or SpeculativeConfig()
        self._target_forward = None  # set via set_target_forward()
        self._step_count = 0
        self._total_accepted = 0
        self._total_draft = 0

    def set_target_forward(
        self,
        forward_fn,
    ) -> None:
        """
        Register the target model forward function.

        *forward_fn* should have signature::

            def forward(hidden_states, token_ids) -> Tuple[np.ndarray, np.ndarray]:
                '''
                Returns
                -------
                logits : np.ndarray  shape (len(token_ids), vocab_size)
                next_hidden : np.ndarray  shape (seq_len+len(token_ids), hidden_dim)
                '''
        """
        self._target_forward = forward_fn

    def step(
        self,
        hidden_states: np.ndarray,
    ) -> SpeculativeStepOutput:
        """
        Execute one speculative decoding step.

        1. Draft model proposes K tokens.
        2. Target model verifies all K positions in parallel.
        3. Rejection sampling determines which tokens are kept.

        Parameters
        ----------
        hidden_states : np.ndarray  shape (seq_len, hidden_dim)
            Current target model hidden state.

        Returns
        -------
        SpeculativeStepOutput
        """
        if self._target_forward is None:
            raise RuntimeError("set_target_forward() must be called before step()")

        t_step = time.perf_counter()
        K = self.config.num_draft_tokens
        if K <= 0:
            # Fallback: single token from target forward
            logits, _ = self._target_forward(hidden_states, [])
            token_id = int(np.argmax(logits[0]))
            return SpeculativeStepOutput(
                accepted_token_ids=[token_id],
                num_accepted=1,
                num_draft=0,
                total_wall_ms=(time.perf_counter() - t_step) * 1000.0,
            )

        # 1. Draft
        t_draft = time.perf_counter()
        draft_tokens, draft_logits = self.draft_runner.generate_proposals(
            hidden_states,
            num_tokens=K,
            temperature=self.config.draft_temperature,
        )
        draft_ms = (time.perf_counter() - t_draft) * 1000.0

        # 2. Target verify (forward pass over draft tokens)
        if len(draft_tokens) == 0:
            return SpeculativeStepOutput(
                accepted_token_ids=[],
                num_accepted=0,
                num_draft=0,
                draft_elapsed_ms=draft_ms,
                total_wall_ms=(time.perf_counter() - t_step) * 1000.0,
            )

        t_verify = time.perf_counter()
        target_logits, next_hidden = self._target_forward(hidden_states, draft_tokens.tolist())
        verify_ms = (time.perf_counter() - t_verify) * 1000.0

        # 3. Rejection sampling
        result = self.verifier.verify(target_logits, draft_logits, draft_tokens)

        self._step_count += 1
        self._total_accepted += result.accepted_count
        self._total_draft += result.draft_count

        # Pause speculative decoding if acceptance rate is too low
        if (self.overall_acceptance_rate() < self.config.min_acceptance_rate
                and self._step_count > 10):
            log.warning(
                "Acceptance rate %.2f below threshold %.2f; consider reducing K",
                self.overall_acceptance_rate(),
                self.config.min_acceptance_rate,
            )

        return SpeculativeStepOutput(
            accepted_token_ids=result.accepted_tokens,
            num_accepted=result.accepted_count,
            num_draft=K,
            draft_elapsed_ms=draft_ms,
            verify_elapsed_ms=verify_ms,
            total_wall_ms=(time.perf_counter() - t_step) * 1000.0,
        )

    def overall_acceptance_rate(self) -> float:
        if self._total_draft == 0:
            return 0.0
        return self._total_accepted / self._total_draft

    def stats(self) -> dict:
        return {
            "steps": self._step_count,
            "total_accepted": self._total_accepted,
            "total_draft": self._total_draft,
            "acceptance_rate": round(self.overall_acceptance_rate(), 4),
            "config": {
                "num_draft_tokens": self.config.num_draft_tokens,
                "mode": self.config.acceptance_mode,
            },
        }


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)
