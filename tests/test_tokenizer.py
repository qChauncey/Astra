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

"""Tests for astra.inference.tokenizer."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from astra.inference.tokenizer import (
    AstraTokenizer,
    _StubBackend,
    _TransformersBackend,
    load_tokenizer,
    get_tokenizer,
    reset_tokenizer,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_global():
    """Ensure global tokenizer state is clean before/after every test."""
    reset_tokenizer(None)
    yield
    reset_tokenizer(None)


# ── _StubBackend ──────────────────────────────────────────────────────────────

class TestStubBackend:
    def test_encode_returns_list_of_ints(self):
        b = _StubBackend()
        ids = b.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_non_empty(self):
        b = _StubBackend()
        assert len(b.encode("foo bar baz")) == 3

    def test_encode_empty_string_returns_fallback(self):
        b = _StubBackend()
        ids = b.encode("")
        assert ids == [1]

    def test_encode_deterministic(self):
        b = _StubBackend()
        assert b.encode("hello") == b.encode("hello")

    def test_encode_different_words_differ(self):
        b = _StubBackend()
        assert b.encode("hello") != b.encode("world")

    def test_ids_within_vocab(self):
        b = _StubBackend()
        from astra.inference.tokenizer import _STUB_VOCAB_SIZE
        for idx in b.encode("the quick brown fox"):
            assert 0 <= idx < _STUB_VOCAB_SIZE

    def test_decode_returns_stub_string(self):
        b = _StubBackend()
        result = b.decode([1, 2, 3])
        assert "stub" in result or "3" in result


# ── _TransformersBackend ──────────────────────────────────────────────────────

class TestTransformersBackend:
    def _mock_hf_tok(self, enc_return, dec_return="decoded text"):
        hf = MagicMock()
        hf.encode.return_value = enc_return
        hf.decode.return_value = dec_return
        return hf

    def test_encode_delegates_to_hf(self):
        hf = self._mock_hf_tok([10, 20, 30])
        b = _TransformersBackend(hf)
        assert b.encode("hello") == [10, 20, 30]
        hf.encode.assert_called_once_with("hello", add_special_tokens=False)

    def test_decode_delegates_to_hf(self):
        hf = self._mock_hf_tok([], dec_return="hello world")
        b = _TransformersBackend(hf)
        assert b.decode([1, 2]) == "hello world"
        hf.decode.assert_called_once_with([1, 2], skip_special_tokens=True)


# ── load_tokenizer ────────────────────────────────────────────────────────────

class TestLoadTokenizer:
    def test_offline_flag_returns_stub(self):
        tok = load_tokenizer(offline=True)
        assert tok.is_stub is True

    def test_no_transformers_returns_stub(self):
        with patch.dict("sys.modules", {"transformers": None}):
            tok = load_tokenizer()
        assert tok.is_stub is True

    def test_transformers_import_error_returns_stub(self):
        with patch("astra.inference.tokenizer.log"):
            with patch.dict("sys.modules", {"transformers": None}):
                tok = load_tokenizer()
        assert tok.is_stub is True

    def test_transformers_exception_returns_stub(self):
        mock_transformers = types.ModuleType("transformers")
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = OSError("model not found")
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            tok = load_tokenizer()
        assert tok.is_stub is True

    def test_transformers_success_returns_real_tokenizer(self):
        hf_tok = MagicMock()
        hf_tok.vocab_size = 102400
        hf_tok.encode.return_value = [1, 2, 3]
        hf_tok.decode.return_value = "hello"

        mock_transformers = types.ModuleType("transformers")
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = hf_tok

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            tok = load_tokenizer("some/model")

        assert tok.is_stub is False
        assert tok.vocab_size == 102400
        assert tok.name == "some/model"

    def test_real_tokenizer_encode_decode(self):
        hf_tok = MagicMock()
        hf_tok.vocab_size = 102400
        hf_tok.encode.return_value = [5, 6, 7]
        hf_tok.decode.return_value = "world"

        mock_transformers = types.ModuleType("transformers")
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = hf_tok

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            tok = load_tokenizer("some/model")

        assert tok.encode("anything") == [5, 6, 7]
        assert tok.decode([5, 6, 7]) == "world"

    def test_env_var_path_used(self, monkeypatch):
        monkeypatch.setenv("ASTRA_TOKENIZER_PATH", "/fake/path")
        mock_transformers = types.ModuleType("transformers")
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = OSError("not found")
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            tok = load_tokenizer()
        assert tok.is_stub is True  # load failed, fell back


# ── AstraTokenizer properties ─────────────────────────────────────────────────

class TestAstraTokenizer:
    def test_repr_stub(self):
        tok = load_tokenizer(offline=True)
        assert "stub" in repr(tok)

    def test_repr_real(self):
        hf_tok = MagicMock()
        hf_tok.vocab_size = 102400
        mock_transformers = types.ModuleType("transformers")
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = hf_tok

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            tok = load_tokenizer("some/model")

        assert "real" in repr(tok)

    def test_vocab_size_stub(self):
        tok = load_tokenizer(offline=True)
        assert tok.vocab_size == 102400

    def test_name_stub(self):
        tok = load_tokenizer(offline=True)
        assert tok.name == "stub"


# ── get_tokenizer / reset_tokenizer (global singleton) ───────────────────────

class TestGlobalSingleton:
    def test_get_tokenizer_returns_instance(self):
        with patch("astra.inference.tokenizer._GLOBAL", None):
            with patch("astra.inference.tokenizer.load_tokenizer",
                       return_value=load_tokenizer(offline=True)) as mock_load:
                tok = get_tokenizer()
        assert isinstance(tok, AstraTokenizer)

    def test_get_tokenizer_cached(self):
        sentinel = load_tokenizer(offline=True)
        reset_tokenizer(sentinel)
        tok1 = get_tokenizer()
        tok2 = get_tokenizer()
        assert tok1 is tok2 is sentinel

    def test_reset_clears_global(self):
        sentinel = load_tokenizer(offline=True)
        reset_tokenizer(sentinel)
        assert get_tokenizer() is sentinel
        reset_tokenizer(None)
        # After reset, get_tokenizer() will re-initialise (returns a new instance)
        tok = get_tokenizer()
        assert tok is not sentinel
