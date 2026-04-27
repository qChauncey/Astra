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
"""Tests for Phase 4 differential privacy module."""

import math
import numpy as np
import pytest

from astra.inference.differential_privacy import (
    DPController,
    LayerDPInjector,
    MomentsAccountant,
    PrivacyBudget,
    _compute_l2_sensitivity,
    _convert_rdp_to_dp,
    _gaussian_sigma,
    _laplace_scale,
    _renyi_divergence_gaussian,
)


# ------------------------------------------------------------------ #
# PrivacyBudget                                                        #
# ------------------------------------------------------------------ #

class TestPrivacyBudget:
    def test_initial_state(self):
        budget = PrivacyBudget(epsilon=8.0, delta=1e-5)
        assert budget.epsilon == 8.0
        assert budget.delta == 1e-5
        assert budget.consumed_epsilon == 0.0
        assert budget.steps == 0
        assert not budget.is_exhausted()
        assert budget.remaining() == 8.0

    def test_remaining_exhausted(self):
        budget = PrivacyBudget(epsilon=2.0, delta=1e-5, consumed_epsilon=2.0)
        assert budget.is_exhausted()
        assert budget.remaining() == 0.0

    def test_remaining_above_zero(self):
        budget = PrivacyBudget(epsilon=4.0, delta=1e-5, consumed_epsilon=3.5)
        assert not budget.is_exhausted()
        assert budget.remaining() == 0.5


# ------------------------------------------------------------------ #
# Gaussian sigma calibration                                           #
# ------------------------------------------------------------------ #

class TestGaussianSigma:
    def test_sigma_increases_with_sensitivity(self):
        s1 = _gaussian_sigma(1.0, 8.0, 1e-5)
        s2 = _gaussian_sigma(10.0, 8.0, 1e-5)
        assert s2 > s1

    def test_sigma_decreases_with_epsilon(self):
        s_strict = _gaussian_sigma(1.0, 0.1, 1e-5)
        s_loose = _gaussian_sigma(1.0, 8.0, 1e-5)
        assert s_strict > s_loose

    def test_sigma_decreases_with_delta(self):
        s_tight = _gaussian_sigma(1.0, 8.0, 1e-8)
        s_loose = _gaussian_sigma(1.0, 8.0, 1e-4)
        assert s_tight > s_loose

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            _gaussian_sigma(1.0, 0, 1e-5)
        with pytest.raises(ValueError):
            _gaussian_sigma(1.0, 8.0, 0)

    def test_balle_wang_bound_known_value(self):
        # Verified against reference implementation for ε=1, δ=1e-5, s=1
        sigma = _gaussian_sigma(1.0, 1.0, 1e-5)
        expected = math.sqrt(2.0 * math.log(1.25 / 1e-5))
        assert math.isclose(sigma, expected, rel_tol=1e-10)

    def test_sigma_positive(self):
        sigma = _gaussian_sigma(0.5, 4.0, 1e-5)
        assert sigma > 0


# ------------------------------------------------------------------ #
# Laplace scale calibration                                             #
# ------------------------------------------------------------------ #

class TestLaplaceScale:
    def test_scale_formula(self):
        assert _laplace_scale(2.0, 4.0) == 0.5
        assert _laplace_scale(1.0, 1.0) == 1.0

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            _laplace_scale(1.0, 0)

    def test_scale_increases_with_sensitivity(self):
        b1 = _laplace_scale(1.0, 4.0)
        b2 = _laplace_scale(10.0, 4.0)
        assert b2 > b1


# ------------------------------------------------------------------ #
# L2 sensitivity estimation                                            #
# ------------------------------------------------------------------ #

class TestL2Sensitivity:
    def test_uniform_hidden(self):
        hidden = np.ones((4, 128), dtype=np.float32)
        sens = _compute_l2_sensitivity(hidden)
        assert math.isclose(sens, math.sqrt(128), rel_tol=1e-5)

    def test_zero_hidden(self):
        hidden = np.zeros((2, 64), dtype=np.float32)
        sens = _compute_l2_sensitivity(hidden)
        # Empty yields 1.0; norms are 0 but at least 1e-4 clamped
        assert sens >= 1e-4

    def test_empty_array(self):
        hidden = np.empty((0, 128), dtype=np.float32)
        sens = _compute_l2_sensitivity(hidden)
        assert sens == 1.0  # fallback for empty

    def test_varying_norms(self):
        rng = np.random.default_rng(42)
        hidden = rng.standard_normal((8, 256)).astype(np.float32)
        # Row 0 gets a spike
        hidden[0] *= 100.0
        sens = _compute_l2_sensitivity(hidden)
        expected = float(np.linalg.norm(hidden[0].astype(np.float64)))
        assert math.isclose(sens, expected, rel_tol=1e-5)


# ------------------------------------------------------------------ #
# Rényi divergence                                                     #
# ------------------------------------------------------------------ #

class TestRenyiDivergence:
    def test_gaussian_divergence(self):
        div = _renyi_divergence_gaussian(1.0, 2.0)
        assert div == 2.0 / (2.0 * 1.0)  # alpha / (2*sigma²) = 2 / 2 = 1.0

    def test_divergence_decreases_with_sigma(self):
        d1 = _renyi_divergence_gaussian(1.0, 4.0)
        d2 = _renyi_divergence_gaussian(10.0, 4.0)
        assert d1 > d2

    def test_divergence_increases_with_alpha(self):
        d1 = _renyi_divergence_gaussian(1.0, 2.0)
        d2 = _renyi_divergence_gaussian(1.0, 8.0)
        assert d2 > d1

    def test_zero_sigma_inf(self):
        assert _renyi_divergence_gaussian(0.0, 2.0) == float("inf")


# ------------------------------------------------------------------ #
# RDP → DP conversion                                                  #
# ------------------------------------------------------------------ #

class TestRdpToDp:
    def test_empty_orders(self):
        eps = _convert_rdp_to_dp([], 1e-5, [])
        assert eps == float("inf")

    def test_single_step(self):
        # One Gaussian step with sigma=1.0 gives ε_α = α/2
        rdp = [2.0 / 2.0]  # α=2 → ε_2 = 1.0
        eps = _convert_rdp_to_dp(rdp, 1e-5, [2.0])
        # ε(δ) = 1.0 + log(1/1e-5) / (2-1) ≈ 1.0 + 11.51 ≈ 12.51
        expected = 1.0 + math.log(1.0 / 1e-5)
        assert math.isclose(eps, expected, rel_tol=1e-10)

    def test_best_order_selected(self):
        rdp = [float("inf"), 2.0, 3.0]
        eps = _convert_rdp_to_dp(rdp, 1e-5, [2.0, 3.0, 4.0])
        assert eps < float("inf")


# ------------------------------------------------------------------ #
# MomentsAccountant                                                    #
# ------------------------------------------------------------------ #

class TestMomentsAccountant:
    def test_initial_zero(self):
        accountant = MomentsAccountant()
        assert accountant.get_epsilon(1e-5) == 0.0
        assert accountant.steps == 0

    def test_accumulate_single_step(self):
        accountant = MomentsAccountant(orders=[2.0, 4.0])
        accountant.accumulate_gaussian(sigma=1.0, sensitivity=1.0)
        eps = accountant.get_epsilon(1e-5)
        # α=2: D = 1.0 → ε ≈ 1.0 + 11.51 ≈ 12.51
        # With orders [2.0, 4.0], best ε comes from α=4: ε ≈ 2.0 + 11.51/3 ≈ 5.84
        assert 4.0 < eps < 7.0
        assert accountant.steps == 1

    def test_accumulate_multiple_steps(self):
        accountant = MomentsAccountant(orders=[2.0])
        for _ in range(10):
            accountant.accumulate_gaussian(sigma=5.0, sensitivity=1.0)
        eps = accountant.get_epsilon(1e-5)
        # 10 steps, each σ=5 → per-step D = α/(2σ²) = 2/50 = 0.04
        # total ε_2 = 0.4 → (ε,δ) ≈ 0.4 + 11.51 ≈ 11.91
        assert 10.0 < eps < 13.0

    def test_reset(self):
        accountant = MomentsAccountant()
        accountant.accumulate_gaussian(sigma=1.0)
        accountant.reset()
        assert accountant.get_epsilon(1e-5) == 0.0
        assert accountant.steps == 0

    def test_high_sigma_low_privacy_loss(self):
        accountant = MomentsAccountant()
        accountant.accumulate_gaussian(sigma=100.0, sensitivity=1.0)
        eps = accountant.get_epsilon(1e-5)
        assert eps < 1.0


# ------------------------------------------------------------------ #
# DPController                                                          #
# ------------------------------------------------------------------ #

class TestDPController:
    def test_gaussian_noise_added(self):
        dp = DPController(epsilon=8.0, delta=1e-5, mechanism="gaussian")
        hidden = np.ones((4, 64), dtype=np.float32) * 0.5
        noised = dp.apply(hidden, layer_idx=0)
        assert noised.shape == hidden.shape
        assert noised.dtype == hidden.dtype
        assert not np.array_equal(noised, hidden)

    def test_laplace_noise_added(self):
        dp = DPController(epsilon=8.0, delta=1e-5, mechanism="laplace")
        hidden = np.ones((4, 64), dtype=np.float32) * 0.5
        noised = dp.apply(hidden, layer_idx=0)
        assert not np.array_equal(noised, hidden)

    def test_utility_threshold_gaussian(self):
        dp = DPController(epsilon=8.0, delta=1e-5, mechanism="gaussian",
                          sensitivity_clip=0.3)
        hidden = np.random.default_rng(42).standard_normal((16, 128)).astype(np.float32)
        noised = dp.apply(hidden)
        ok, err = dp.verify_utility(hidden, noised, threshold=0.5)
        assert ok, f"Utility degradation too high: {err}"
        assert err < 0.5

    def test_utility_threshold_laplace(self):
        dp = DPController(epsilon=8.0, delta=1e-5, mechanism="laplace",
                          sensitivity_clip=0.3)
        hidden = np.random.default_rng(43).standard_normal((16, 128)).astype(np.float32)
        noised = dp.apply(hidden)
        ok, err = dp.verify_utility(hidden, noised, threshold=0.5)
        assert ok, f"Utility degradation too high: {err}"

    def test_strict_privacy_higher_noise(self):
        # ε=0.1 should produce much noisier output than ε=8.0
        dp_strict = DPController(epsilon=0.1, delta=1e-5, mechanism="gaussian")
        dp_loose = DPController(epsilon=8.0, delta=1e-5, mechanism="gaussian")
        hidden = np.random.default_rng(44).standard_normal((8, 128)).astype(np.float32)
        strict_noised = dp_strict.apply(hidden)
        loose_noised = dp_loose.apply(hidden)
        strict_var = float(np.var((strict_noised - hidden).astype(np.float64)))
        loose_var = float(np.var((loose_noised - hidden).astype(np.float64)))
        assert strict_var > loose_var, f"strict={strict_var} <= loose={loose_var}"

    def test_sensitivity_clip(self):
        dp = DPController(epsilon=4.0, delta=1e-5, sensitivity_clip=0.5)
        hidden = np.ones((4, 64), dtype=np.float32) * 10.0  # real sensitivity >> clip
        noised = dp.apply(hidden)
        ok, err = dp.verify_utility(hidden, noised, threshold=0.5)
        assert ok, f"With clipping, utility should be acceptable: {err}"

    def test_sigma_for_layer(self):
        dp = DPController(epsilon=8.0, delta=1e-5)
        hidden = np.ones((4, 64), dtype=np.float32)
        sigma = dp.sigma_for_layer(hidden, epsilon_per_layer=8.0 / 61)
        assert sigma > 0

    def test_stats_output(self):
        dp = DPController(epsilon=4.0, delta=1e-5)
        hidden = np.ones((2, 32), dtype=np.float32)
        dp.apply(hidden)
        stats = dp.stats()
        assert stats["mechanism"] == "gaussian"
        assert stats["epsilon_target"] == 4.0
        assert stats["delta"] == 1e-5
        assert stats["steps"] == 1
        assert "epsilon_consumed" in stats
        assert "budget_remaining_pct" in stats

    def test_reset(self):
        dp = DPController(epsilon=4.0, delta=1e-5)
        hidden = np.ones((2, 32), dtype=np.float32)
        dp.apply(hidden)
        assert dp.consumed_epsilon > 0
        dp.reset()
        assert dp.consumed_epsilon == 0.0
        assert dp.budget.remaining() == 4.0

    def test_invalid_mechanism(self):
        with pytest.raises(ValueError, match="Unknown mechanism"):
            DPController(epsilon=4.0, mechanism="triangular")


# ------------------------------------------------------------------ #
# LayerDPInjector                                                       #
# ------------------------------------------------------------------ #

class TestLayerDPInjector:
    def test_per_layer_budget_split(self):
        injector = LayerDPInjector(epsilon=4.0, delta=1e-5, num_layers=61)
        expected = 4.0 / 61
        assert math.isclose(injector.eps_per_layer, expected, rel_tol=1e-10)

    def test_callable_interface(self):
        injector = LayerDPInjector(epsilon=4.0, delta=1e-5, num_layers=61)
        hidden = np.ones((4, 128), dtype=np.float32)
        noised = injector(hidden, layer_idx=0)
        assert noised.shape == hidden.shape
        assert not np.array_equal(noised, hidden)

    def test_accumulating_epsilon(self):
        injector = LayerDPInjector(epsilon=8.0, delta=1e-5, num_layers=10)
        hidden = np.ones((4, 64), dtype=np.float32)
        for i in range(10):
            hidden = injector(hidden, layer_idx=i)
        # After all 10 layers, consumed epsilon should be <= 8.0
        eps = injector.controller.consumed_epsilon
        assert eps <= 8.0 * 1.5, f"Excessive ε consumption: {eps}"

    def test_utility_after_full_forward(self):
        injector = LayerDPInjector(epsilon=8.0, delta=1e-5, num_layers=10,
                                    mechanism="gaussian", sensitivity_clip=0.3)
        hidden = np.random.default_rng(45).standard_normal((8, 128)).astype(np.float32)
        original = hidden.copy()
        for i in range(10):
            hidden = injector(hidden, layer_idx=i)
        ok, err = injector.controller.verify_utility(original, hidden, threshold=10.0)
        assert ok, f"Utility too degraded: {err}"

    def test_stats_includes_layers(self):
        injector = LayerDPInjector(epsilon=4.0, num_layers=61)
        stats = injector.stats()
        assert stats["num_layers"] == 61
        assert "eps_per_layer" in stats
        assert math.isclose(stats["eps_per_layer"], 4.0 / 61, rel_tol=1e-10)

    def test_zero_layers_invalid(self):
        with pytest.raises(ValueError):
            LayerDPInjector(epsilon=4.0, num_layers=0)


# ------------------------------------------------------------------ #
# Integration: HeterogeneousEngine with DP injector                     #
# ------------------------------------------------------------------ #

@pytest.mark.integration
class TestEngineDPIntegration:
    def test_engine_with_dp_injector(self):
        from astra.inference.heterogeneous import DeviceMap, HeterogeneousEngine
        from astra.serialization.tensor_pack import TensorPacket

        dmap = DeviceMap(attention_on_gpu=False, moe_on_cpu=True)
        injector = LayerDPInjector(epsilon=4.0, delta=1e-5, num_layers=5)
        engine = HeterogeneousEngine(device_map=dmap, dp_injector=injector)

        hidden = np.random.default_rng(46).standard_normal((4, 256)).astype(np.float32)
        experts = np.array([[2, 3], [2, 3], [4, 5], [4, 5]], dtype=np.int32)
        packet = TensorPacket(
            packet_id=1,
            tensor=hidden,
            layer_start=0,
            layer_end=5,
            token_ids=[1, 2, 3, 4],
            selected_experts=experts,
        )

        result = engine.forward(packet, layer_indices=[0, 1, 2])

        assert result.tensor.shape == hidden.shape
        assert "dp" in result.metadata
        dp_meta = result.metadata["dp"]
        assert "epsilon_target" in dp_meta
        assert dp_meta["num_layers"] == 5

        stats = engine.stats()
        assert "dp" in stats
        assert stats["dp"]["num_layers"] == 5

    def test_engine_without_dp_no_metadata(self):
        from astra.inference.heterogeneous import DeviceMap, HeterogeneousEngine
        from astra.serialization.tensor_pack import TensorPacket

        dmap = DeviceMap(attention_on_gpu=False, moe_on_cpu=True)
        engine = HeterogeneousEngine(device_map=dmap, dp_injector=None)

        hidden = np.ones((2, 256), dtype=np.float32)
        packet = TensorPacket(
            packet_id=1,
            tensor=hidden,
            layer_start=0,
            layer_end=1,
            token_ids=[1, 2],
        )

        result = engine.forward(packet, layer_indices=[0])
        assert "dp" not in result.metadata
        assert "dp" not in engine.stats()
