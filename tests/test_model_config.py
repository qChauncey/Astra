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

"""Unit tests for astra.config.model_config."""

import copy

import pytest

from astra.config.model_config import (
    MODEL_CONFIGS,
    DEFAULT_MODEL,
    AttentionType,
    ModelConfig,
    QuantizationType,
    get_model_config,
    register_model_config,
)


# ---------------------------------------------------------------------------
# AttentionType enum
# ---------------------------------------------------------------------------

class TestAttentionType:
    def test_values(self):
        assert AttentionType.MLA.value == "mla"
        assert AttentionType.GQA.value == "gqa"

    def test_from_string(self):
        assert AttentionType("mla") == AttentionType.MLA
        assert AttentionType("gqa") == AttentionType.GQA

    def test_invalid_attention_type(self):
        with pytest.raises(ValueError):
            AttentionType("invalid")


# ---------------------------------------------------------------------------
# QuantizationType enum
# ---------------------------------------------------------------------------

class TestQuantizationType:
    def test_values(self):
        assert QuantizationType.BF16.value == "bf16"
        assert QuantizationType.FP16.value == "fp16"
        assert QuantizationType.FP8.value == "fp8"
        assert QuantizationType.INT8.value == "int8"
        assert QuantizationType.INT4.value == "int4"
        assert QuantizationType.MXFP4.value == "mxfp4"

    def test_from_string(self):
        assert QuantizationType("bf16") == QuantizationType.BF16
        assert QuantizationType("mxfp4") == QuantizationType.MXFP4

    def test_invalid_quantization_type(self):
        with pytest.raises(ValueError):
            QuantizationType("fp32")


# ---------------------------------------------------------------------------
# ModelConfig dataclass — basic construction
# ---------------------------------------------------------------------------

class TestModelConfigConstruction:
    def test_minimal_construction(self):
        """ModelConfig requires identity, dims, attention, MoE, and quant fields."""
        cfg = ModelConfig(
            model_id="test/test-model",
            display_name="TestModel",
            arch_type="TestForCausalLM",
            model_type="test",
            hidden_dim=1024,
            num_layers=12,
            head_dim=64,
            num_attention_heads=16,
            num_key_value_heads=4,
            intermediate_size=4096,
            vocab_size=32000,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            rotary_dim=32,
            rms_norm_eps=1e-5,
            attention_type=AttentionType.GQA,
            num_local_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=1,
            scoring_func="softmax",
            native_quant=QuantizationType.BF16,
        )
        assert cfg.model_id == "test/test-model"
        assert cfg.display_name == "TestModel"
        assert cfg.hidden_dim == 1024
        assert cfg.num_layers == 12

    def test_default_values(self):
        cfg = ModelConfig(
            model_id="test/t2",
            display_name="T2",
            arch_type="T2ForCausalLM",
            model_type="t2",
            hidden_dim=512,
            num_layers=4,
            head_dim=32,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=2048,
            vocab_size=16000,
            max_position_embeddings=2048,
            rope_theta=50000.0,
            rotary_dim=16,
            rms_norm_eps=1e-6,
            attention_type=AttentionType.MLA,
            num_local_experts=4,
            num_experts_per_tok=1,
            num_shared_experts=0,
            scoring_func="sigmoid",
            native_quant=QuantizationType.FP8,
        )
        # Optional fields should have defaults
        assert cfg.use_qk_norm is False
        assert cfg.use_routing_bias is False
        assert cfg.use_mtp is False
        assert cfg.num_mtp_modules == 0
        assert cfg.mtp_transformer_layers == 0
        assert cfg.ktransformers_supported is True
        assert cfg.ktransformers_arch_name == ""
        assert cfg.kt_method == ""
        assert cfg.attention_variant == "standard"
        assert cfg.kt_num_gpu_experts == 144
        assert cfg.kt_cpuinfer == 8
        assert cfg.kt_threadpool_count == 2
        assert cfg.num_safetensors_shards == 0
        assert cfg.total_size_gb == 0.0
        assert cfg.auto_map == {}

    def test_frozen_dataclass(self):
        cfg = ModelConfig(
            model_id="test/frozen",
            display_name="Frozen",
            arch_type="Test",
            model_type="test",
            hidden_dim=256,
            num_layers=2,
            head_dim=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=1024,
            vocab_size=10000,
            max_position_embeddings=1024,
            rope_theta=10000.0,
            rotary_dim=16,
            rms_norm_eps=1e-6,
            attention_type=AttentionType.MLA,
            num_local_experts=2,
            num_experts_per_tok=1,
            num_shared_experts=0,
            scoring_func="sigmoid",
            native_quant=QuantizationType.FP16,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.hidden_dim = 512  # type: ignore[misc]

    def test_auto_map_default_factory_independence(self):
        cfg1 = ModelConfig(
            model_id="test/am1",
            display_name="AM1",
            arch_type="Test",
            model_type="test",
            hidden_dim=256,
            num_layers=2,
            head_dim=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=1024,
            vocab_size=10000,
            max_position_embeddings=1024,
            rope_theta=10000.0,
            rotary_dim=16,
            rms_norm_eps=1e-6,
            attention_type=AttentionType.MLA,
            num_local_experts=2,
            num_experts_per_tok=1,
            num_shared_experts=0,
            scoring_func="sigmoid",
            native_quant=QuantizationType.FP16,
            auto_map={"AutoConfig": "config_test.TestConfig"},
        )
        cfg2 = ModelConfig(
            model_id="test/am2",
            display_name="AM2",
            arch_type="Test",
            model_type="test",
            hidden_dim=256,
            num_layers=2,
            head_dim=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=1024,
            vocab_size=10000,
            max_position_embeddings=1024,
            rope_theta=10000.0,
            rotary_dim=16,
            rms_norm_eps=1e-6,
            attention_type=AttentionType.MLA,
            num_local_experts=2,
            num_experts_per_tok=1,
            num_shared_experts=0,
            scoring_func="sigmoid",
            native_quant=QuantizationType.FP16,
        )
        assert cfg2.auto_map == {}
        assert cfg1.auto_map["AutoConfig"] == "config_test.TestConfig"


# ---------------------------------------------------------------------------
# ModelConfig properties
# ---------------------------------------------------------------------------

class TestModelConfigProperties:
    @pytest.fixture
    def gqa_config(self):
        return ModelConfig(
            model_id="test/gqa",
            display_name="GQAModel",
            arch_type="Test",
            model_type="test",
            hidden_dim=1024,
            num_layers=12,
            head_dim=64,
            num_attention_heads=16,
            num_key_value_heads=4,
            intermediate_size=4096,
            vocab_size=32000,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            rotary_dim=32,
            rms_norm_eps=1e-5,
            attention_type=AttentionType.GQA,
            num_local_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=1,
            scoring_func="softmax",
            native_quant=QuantizationType.BF16,
        )

    @pytest.fixture
    def mla_config(self):
        return ModelConfig(
            model_id="test/mla",
            display_name="MLAModel",
            arch_type="Test",
            model_type="test",
            hidden_dim=7168,
            num_layers=61,
            head_dim=128,
            num_attention_heads=128,
            num_key_value_heads=128,
            intermediate_size=18432,
            vocab_size=129280,
            max_position_embeddings=163840,
            rope_theta=500000.0,
            rotary_dim=64,
            rms_norm_eps=1e-6,
            attention_type=AttentionType.MLA,
            num_local_experts=256,
            num_experts_per_tok=8,
            num_shared_experts=2,
            scoring_func="sigmoid",
            native_quant=QuantizationType.MXFP4,
        )

    def test_total_experts(self, gqa_config, mla_config):
        assert gqa_config.total_experts == 9   # 8 local + 1 shared
        assert mla_config.total_experts == 258  # 256 local + 2 shared

    def test_num_shared_experts_per_gpu_nonzero(self, gqa_config):
        assert gqa_config.num_shared_experts_per_gpu == 1

    def test_num_shared_experts_per_gpu_zero_returns_one(self):
        cfg = ModelConfig(
            model_id="test/no-shared",
            display_name="NoShared",
            arch_type="Test",
            model_type="test",
            hidden_dim=512,
            num_layers=4,
            head_dim=32,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=2048,
            vocab_size=10000,
            max_position_embeddings=1024,
            rope_theta=10000.0,
            rotary_dim=16,
            rms_norm_eps=1e-6,
            attention_type=AttentionType.MLA,
            num_local_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=0,
            scoring_func="softmax",
            native_quant=QuantizationType.FP16,
        )
        assert cfg.num_shared_experts_per_gpu == 1

    def test_gqa_groups(self, gqa_config):
        # GQA: 16 Q heads / 4 KV heads = 4 groups
        assert gqa_config.gqa_groups == 4

    def test_gqa_groups_mla(self, mla_config):
        # MLA always returns 1
        assert mla_config.gqa_groups == 1

    def test_per_layer_bytes_bf16_is_positive(self, gqa_config):
        assert gqa_config.per_layer_bytes_bf16 > 0

    def test_per_layer_bytes_bf16_is_int(self, gqa_config):
        assert isinstance(gqa_config.per_layer_bytes_bf16, int)


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

class TestBuiltinProfiles:
    def test_deepseek_v4_flash(self):
        from astra.config.model_config import DEEPSEEK_V4_FLASH
        cfg = DEEPSEEK_V4_FLASH
        assert cfg.model_id == "deepseek-ai/DeepSeek-V4-Flash"
        # V4-Flash uses Hybrid CSA (Compressed Sparse Attention) + HCA (Hash Compressed Attention)
        assert cfg.attention_type == AttentionType.HYBRID_CSA_HCA
        assert cfg.native_quant == QuantizationType.MXFP4
        assert cfg.num_local_experts == 256
        assert cfg.num_experts_per_tok == 6
        assert cfg.num_shared_experts == 1
        assert cfg.hidden_dim == 4096
        assert cfg.num_layers == 43
        assert cfg.total_experts == 257
        assert cfg.gqa_groups == 64  # 64 Q heads / 1 KV head = 64 groups (MLA-like)

    def test_minimax_m2_5(self):
        from astra.config.model_config import MINIMAX_M2_5
        cfg = MINIMAX_M2_5
        assert cfg.model_id == "MiniMaxAI/MiniMax-M2.5"
        assert cfg.attention_type == AttentionType.GQA
        assert cfg.native_quant == QuantizationType.FP8
        assert cfg.num_local_experts == 256
        assert cfg.num_experts_per_tok == 8
        assert cfg.num_shared_experts == 0
        assert cfg.hidden_dim == 3072
        assert cfg.num_layers == 62
        assert cfg.total_experts == 256
        assert cfg.gqa_groups == 6  # 48/8

    def test_deepseek_v4_ktransformers_fields(self):
        from astra.config.model_config import DEEPSEEK_V4_FLASH
        cfg = DEEPSEEK_V4_FLASH
        assert cfg.ktransformers_supported is True
        assert cfg.ktransformers_arch_name == "deepseek_v4"
        assert cfg.kt_method == "MXFP4"
        assert cfg.attention_variant == "hybrid_csa_hca"
        assert cfg.kt_num_gpu_experts == 144

    def test_minimax_m2_5_ktransformers_fields(self):
        from astra.config.model_config import MINIMAX_M2_5
        cfg = MINIMAX_M2_5
        assert cfg.ktransformers_supported is False
        assert cfg.ktransformers_arch_name == "minimax_m2"


# ---------------------------------------------------------------------------
# MODEL_CONFIGS registry
# ---------------------------------------------------------------------------

class TestModelConfigsRegistry:
    def test_registry_contains_canonical_keys(self):
        assert "deepseek-ai/DeepSeek-V4-Flash" in MODEL_CONFIGS
        assert "MiniMaxAI/MiniMax-M2.5" in MODEL_CONFIGS

    def test_registry_contains_short_alias_keys(self):
        assert "deepseek-v4-flash" in MODEL_CONFIGS
        assert "deepseekv4" in MODEL_CONFIGS
        assert "minimax-m2.5" in MODEL_CONFIGS
        assert "minimax-m2-5" in MODEL_CONFIGS

    def test_registry_entries_are_model_configs(self):
        for key, val in MODEL_CONFIGS.items():
            assert isinstance(val, ModelConfig), \
                f"Key '{key}' is not a ModelConfig instance"

    def test_default_model_is_in_registry(self):
        assert DEFAULT_MODEL in MODEL_CONFIGS
        cfg = MODEL_CONFIGS[DEFAULT_MODEL]
        assert isinstance(cfg, ModelConfig)


# ---------------------------------------------------------------------------
# get_model_config
# ---------------------------------------------------------------------------

class TestGetModelConfig:
    def test_default_returns_default_model(self):
        cfg = get_model_config()
        assert cfg.model_id == "deepseek-ai/DeepSeek-V4-Flash"

    def test_none_returns_default_model(self):
        cfg = get_model_config(None)
        assert cfg.model_id == "deepseek-ai/DeepSeek-V4-Flash"

    def test_canonical_model_id(self):
        cfg = get_model_config("deepseek-ai/DeepSeek-V4-Flash")
        assert cfg.display_name == "DeepSeek-V4-Flash"

    def test_short_alias(self):
        # "deepseekv4" alias maps to Pro (strongest model in family)
        cfg = get_model_config("deepseekv4")
        assert cfg.display_name == "DeepSeek-V4-Pro"

    def test_lowercase_alias(self):
        cfg = get_model_config("minimax-m2-5")
        assert cfg.display_name == "MiniMax-M2.5"

    def test_case_insensitive(self):
        cfg = get_model_config("MINIMAX-M2.5")
        assert cfg.display_name == "MiniMax-M2.5"

    def test_underscore_normalized_to_hyphen(self):
        cfg = get_model_config("minimax_m2_5")
        assert cfg.display_name == "MiniMax-M2.5"

    def test_unknown_model_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model_config("nonexistent/model")

    def test_unknown_model_error_includes_registered_keys(self):
        with pytest.raises(KeyError) as excinfo:
            get_model_config("bogus-model")
        assert "Unknown model" in str(excinfo.value)
        assert "Registered:" in str(excinfo.value)

    def test_exact_match_fallback(self):
        """Test that exact string match works even if key isn't lowercased."""
        # Register a model with mixed-case key
        from astra.config.model_config import MODEL_CONFIGS as reg
        cfg = copy.deepcopy(MODEL_CONFIGS["minimax-m2.5"])
        # Temporarily add a mixed-case key
        reg["MixedCase-Model"] = cfg
        try:
            result = get_model_config("MixedCase-Model")
            assert result.display_name == "MiniMax-M2.5"
        finally:
            del reg["MixedCase-Model"]


# ---------------------------------------------------------------------------
# register_model_config
# ---------------------------------------------------------------------------

class TestRegisterModelConfig:
    def test_register_new_model(self):
        new_cfg = ModelConfig(
            model_id="testorg/NewModel",
            display_name="NewModel",
            arch_type="NewForCausalLM",
            model_type="new",
            hidden_dim=512,
            num_layers=3,
            head_dim=32,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=1024,
            vocab_size=10000,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            rotary_dim=16,
            rms_norm_eps=1e-5,
            attention_type=AttentionType.MLA,
            num_local_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=0,
            scoring_func="softmax",
            native_quant=QuantizationType.BF16,
        )
        try:
            register_model_config(new_cfg)
            assert "testorg/NewModel" in MODEL_CONFIGS
            assert "newmodel" in MODEL_CONFIGS  # display_name.lower()
            retrieved = get_model_config("testorg/NewModel")
            assert retrieved is new_cfg
        finally:
            # Cleanup
            MODEL_CONFIGS.pop("testorg/NewModel", None)
            MODEL_CONFIGS.pop("newmodel", None)

    def test_register_overwrites_existing(self):
        original = MODEL_CONFIGS["minimax-m2.5"]
        new_cfg = ModelConfig(
            model_id="MiniMaxAI/MiniMax-M2.5",
            display_name="MiniMax-M2.5",
            arch_type="Override",
            model_type="test",
            hidden_dim=256,
            num_layers=1,
            head_dim=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            vocab_size=100,
            max_position_embeddings=128,
            rope_theta=1.0,
            rotary_dim=16,
            rms_norm_eps=1e-6,
            attention_type=AttentionType.MLA,
            num_local_experts=1,
            num_experts_per_tok=1,
            num_shared_experts=0,
            scoring_func="softmax",
            native_quant=QuantizationType.BF16,
        )
        try:
            register_model_config(new_cfg)
            assert MODEL_CONFIGS["MiniMaxAI/MiniMax-M2.5"] is new_cfg
            assert get_model_config("minimax-m2.5") is new_cfg
        finally:
            # Restore
            MODEL_CONFIGS["MiniMaxAI/MiniMax-M2.5"] = original
            MODEL_CONFIGS["minimax-m2.5"] = original
            MODEL_CONFIGS["minimax-m2-5"] = original