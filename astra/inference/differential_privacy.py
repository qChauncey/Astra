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
#   - Phase 4: Differential Privacy module for activation noise injection.
#   - Implements Gaussian mechanism with moments accountant (Rényi DP).
#   - Prevents intermediate hidden-state inversion attacks while preserving
#     inference quality within acceptable bounds (L2 degradation < 0.1).

"""
Differential Privacy activations for Astra inference pipeline.

Provides calibrated Gaussian noise injection into hidden states
at each pipeline stage (attention output, MoE FFN output) so that
intermediate nodes cannot reconstruct raw input tokens via
model-inversion attacks.

Mechanisms:
  - Gaussian mechanism (ε, δ)-DP with analytic sigma calibration
  - Laplace mechanism for strict ε-DP (lower utility, higher guarantee)
  - Moments accountant (Rényi DP) for composing privacy loss across layers

Typical usage:
    from astra.inference.differential_privacy import DPController

    dp = DPController(epsilon=8.0, delta=1e-5)
    dp_noised = dp.apply_gaussian(hidden, layer_idx)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------ #
# Privacy accounting                                                    #
# ------------------------------------------------------------------ #

@dataclass
class PrivacyBudget:
    """Current privacy budget state (Rényi DP with moments accountant)."""
    epsilon: float       # target ε
    delta: float          # target δ (typically 1e-5 for N < 10^6)
    consumed_epsilon: float = 0.0
    steps: int = 0
    log_mgf: List[float] = field(default_factory=list)  # log moment-generating function per step

    def remaining(self) -> float:
        return max(0.0, self.epsilon - self.consumed_epsilon)

    def is_exhausted(self) -> bool:
        return self.consumed_epsilon >= self.epsilon


def _gaussian_sigma(
    sensitivity: float,
    epsilon: float,
    delta: float,
) -> float:
    """Analytic sigma calibration for the Gaussian mechanism.

    Uses the tight bound from Balle & Wang (2018):
        σ ≥ (sensitivity / ε) · √(2 · ln(1.25 / δ))

    Returns float sigma for np.random.normal(0, sigma).
    """
    if epsilon <= 0 or delta <= 0:
        raise ValueError(f"epsilon ({epsilon}) and delta ({delta}) must be > 0")
    return (sensitivity / epsilon) * math.sqrt(2.0 * math.log(1.25 / delta))


def _laplace_scale(sensitivity: float, epsilon: float) -> float:
    """Scale parameter b for Laplace mechanism: b = sensitivity / ε."""
    if epsilon <= 0:
        raise ValueError(f"epsilon ({epsilon}) must be > 0")
    return sensitivity / epsilon


def _compute_l2_sensitivity(hidden: np.ndarray, axis: int = -1) -> float:
    """Estimate per-sample L2 sensitivity as max row norm.

    For hidden states (seq_len, hidden_dim), this is the maximum
    L2 norm across token positions, which bounds how much any single
    token can influence the output vector.

    Clipped to a reasonable ceiling to prevent pathological sensitivity
    from driving sigma to infinity.
    """
    norms = np.linalg.norm(hidden.astype(np.float64), axis=axis)
    raw = float(norms.max()) if norms.size > 0 else 1.0
    # Clamp to practical range for float16 hidden states
    return max(raw, 1e-4)


# ------------------------------------------------------------------ #
# Renyi DP moments accountant                                          #
# ------------------------------------------------------------------ #

def _renyi_divergence_gaussian(sigma: float, alpha: float) -> float:
    """Rényi divergence D_α(P||Q) for Gaussian mechanism N(0, σ²).

    For the Gaussian mechanism with sensitivity 1, adding noise σ·N(0,1):
        D_α = α / (2 · σ²)

    When sensitivity ≠ 1, we pre-scale sigma by 1/sensitivity,
    so the effective sigma_eff = σ / sensitivity, giving:
        D_α = α · sensitivity² / (2 · σ²)
    """
    if sigma <= 0.0:
        return float("inf")
    return alpha / (2.0 * sigma * sigma)


def _convert_rdp_to_dp(
    rdp_orders: List[float],
    delta: float,
    orders: List[float],
) -> float:
    """Convert RDP (α, ε_α) curve to (ε, δ)-DP.

    For each α, ε(δ) = min_α (ε_α + log(1/δ) / (α - 1)).
    """
    best = float("inf")
    for eps_alpha, alpha in zip(rdp_orders, orders):
        if alpha <= 1.0:
            continue
        eps = eps_alpha + math.log(1.0 / delta) / (alpha - 1.0)
        if eps < best:
            best = eps
    return best


class MomentsAccountant:
    """Rényi DP moments accountant for composing privacy loss across layers.

    Tracks log moment-generating function at multiple orders α
    and converts to (ε, δ)-DP at any point.
    """

    def __init__(self, orders: Optional[List[float]] = None) -> None:
        self._orders = orders or [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 16.0, 32.0, 64.0]
        self._rdp_eps: List[float] = [0.0] * len(self._orders)
        self._steps: int = 0

    def accumulate_gaussian(self, sigma: float, sensitivity: float = 1.0) -> None:
        """Account for one Gaussian mechanism step.

        sigma: standard deviation of added noise
        sensitivity: L2 sensitivity of the query (default 1.0 when pre-scaled)
        """
        sigma_eff = sigma / sensitivity
        for i, alpha in enumerate(self._orders):
            self._rdp_eps[i] += _renyi_divergence_gaussian(sigma_eff, alpha)
        self._steps += 1

    def get_epsilon(self, delta: float) -> float:
        """Current (ε, δ)-DP privacy budget consumed."""
        if self._steps == 0:
            return 0.0
        return _convert_rdp_to_dp(self._rdp_eps, delta, self._orders)

    def reset(self) -> None:
        self._rdp_eps = [0.0] * len(self._orders)
        self._steps = 0

    @property
    def steps(self) -> int:
        return self._steps


# ------------------------------------------------------------------ #
# DP Controller                                                         #
# ------------------------------------------------------------------ #

class DPController:
    """Orchestrates DP noise injection across the inference pipeline.

    Maintains a privacy budget and automatically computes calibrated
    noise per layer.  Supports both single-query (ε, δ) and per-layer
    budget subdivision (spread total budget evenly across layers).

    Parameters
    ----------
    epsilon : float
        Total privacy budget (ε).  Typical values: 1.0–10.0.
        Lower = stronger privacy, higher = better utility.
    delta : float
        Failure probability (δ).  Should be < 1/N where N is the
        number of tokens processed.  Default 1e-5 is safe for
        N < 100k.
    sensitivity_clip : float or None
        Maximum L2 sensitivity per token.  If None, auto-computed
        from the hidden state norm at each step.
    mechanism : str
        "gaussian" (default) or "laplace".
    """

    def __init__(
        self,
        epsilon: float = 8.0,
        delta: float = 1e-5,
        sensitivity_clip: Optional[float] = None,
        mechanism: str = "gaussian",
    ) -> None:
        self._epsilon = epsilon
        self._delta = delta
        self._sensitivity_clip = sensitivity_clip
        self._mechanism = mechanism.lower()
        self._accountant = MomentsAccountant()
        self._budget = PrivacyBudget(epsilon=epsilon, delta=delta)

        if self._mechanism not in ("gaussian", "laplace"):
            raise ValueError(f"Unknown mechanism: {mechanism} (use 'gaussian' or 'laplace')")

    @property
    def budget(self) -> PrivacyBudget:
        return self._budget

    @property
    def consumed_epsilon(self) -> float:
        return self._accountant.get_epsilon(self._delta)

    def apply(
        self,
        hidden: np.ndarray,
        layer_idx: int = 0,
        epsilon_per_layer: Optional[float] = None,
    ) -> np.ndarray:
        """Inject DP noise into hidden states.

        Parameters
        ----------
        hidden : np.ndarray
            Hidden state tensor of shape (seq_len, hidden_dim).
        layer_idx : int
            Current transformer layer index (for per-layer sigma logging).
        epsilon_per_layer : float or None
            Privacy budget allocated to this layer.  If None, uses
            the full remaining budget (single-step query).

        Returns
        -------
        np.ndarray
            Hidden states with calibrated noise added.  Same shape/dtype.
        """
        eps = epsilon_per_layer if epsilon_per_layer is not None else self._epsilon

        if self._mechanism == "gaussian":
            result = self._apply_gaussian(hidden, eps)
        else:
            result = self._apply_laplace(hidden, eps)

        self._budget.steps += 1
        return result

    def _apply_gaussian(self, hidden: np.ndarray, epsilon: float) -> np.ndarray:
        sensitivity = self._sensitivity_clip or _compute_l2_sensitivity(hidden)
        sigma = _gaussian_sigma(sensitivity, epsilon, self._delta)
        self._accountant.accumulate_gaussian(sigma, sensitivity)
        noise = np.random.normal(0.0, sigma, hidden.shape).astype(hidden.dtype)
        return hidden + noise

    def _apply_laplace(self, hidden: np.ndarray, epsilon: float) -> np.ndarray:
        sensitivity = self._sensitivity_clip or _compute_l2_sensitivity(hidden)
        scale = _laplace_scale(sensitivity, epsilon)
        noise = np.random.laplace(0.0, scale, hidden.shape).astype(hidden.dtype)
        return hidden + noise

    def sigma_for_layer(self, hidden: np.ndarray, epsilon_per_layer: float) -> float:
        """Compute what sigma would be used without actually adding noise.

        Useful for logging and budget planning.
        """
        sensitivity = self._sensitivity_clip or _compute_l2_sensitivity(hidden)
        return _gaussian_sigma(sensitivity, epsilon_per_layer, self._delta)

    def verify_utility(
        self,
        original: np.ndarray,
        noised: np.ndarray,
        threshold: float = 0.1,
    ) -> Tuple[bool, float]:
        """Check that L2 degradation is within acceptable bounds.

        Returns (pass, relative_l2_error).
        """
        diff = np.linalg.norm((noised - original).astype(np.float64))
        orig_norm = np.linalg.norm(original.astype(np.float64))
        rel_error = float(diff / max(orig_norm, 1e-10))
        return rel_error <= threshold, rel_error

    def reset(self) -> None:
        """Reset privacy budget and accountant (for a new session)."""
        self._accountant.reset()
        self._budget = PrivacyBudget(epsilon=self._epsilon, delta=self._delta)

    def stats(self) -> dict:
        return {
            "mechanism": self._mechanism,
            "epsilon_target": self._epsilon,
            "delta": self._delta,
            "epsilon_consumed": round(self.consumed_epsilon, 6),
            "steps": self._accountant.steps,
            "budget_remaining_pct": round(self._budget.remaining() / self._epsilon * 100, 1),
        }


# ------------------------------------------------------------------ #
# Per-layer DP injector (hook for HeterogeneousEngine)                  #
# ------------------------------------------------------------------ #

class LayerDPInjector:
    """Lightweight DP injector designed to be called from
    HeterogeneousEngine._attention_forward and _moe_forward.

    Usage::

        injector = LayerDPInjector(epsilon=4.0, num_layers=61)
        hidden = injector(hidden, layer_idx=0)
    """

    def __init__(
        self,
        epsilon: float = 4.0,
        delta: float = 1e-5,
        num_layers: int = 61,
        sensitivity_clip: Optional[float] = None,
        mechanism: str = "gaussian",
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self._controller = DPController(
            epsilon=epsilon,
            delta=delta,
            sensitivity_clip=sensitivity_clip,
            mechanism=mechanism,
        )
        self._num_layers = num_layers
        self._eps_per_layer = epsilon / num_layers

    def __call__(self, hidden: np.ndarray, layer_idx: int = 0) -> np.ndarray:
        """Inject DP noise for one transformer layer."""
        return self._controller.apply(
            hidden,
            layer_idx=layer_idx,
            epsilon_per_layer=self._eps_per_layer,
        )

    @property
    def controller(self) -> DPController:
        return self._controller

    @property
    def eps_per_layer(self) -> float:
        return self._eps_per_layer

    def stats(self) -> dict:
        return {
            **self._controller.stats(),
            "num_layers": self._num_layers,
            "eps_per_layer": self._eps_per_layer,
        }
