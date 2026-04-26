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
Tokenizer interface for Astra.

Load priority
-------------
1. Local path from ``ASTRA_TOKENIZER_PATH`` environment variable
2. Path/name passed explicitly to :func:`load_tokenizer`
3. HuggingFace Hub: ``deepseek-ai/DeepSeek-V2`` (requires ``transformers``
   and internet; ~50 MB tokenizer config, no model weights needed)
4. Stub fallback: whitespace + CRC32 — always works, produces non-meaningful
   token IDs only useful for infrastructure testing

Phase status
------------
Phase 3 (current): stub always used — transformers/weights not required.
Phase 4: set ``ASTRA_TOKENIZER_PATH`` to a local tokenizer directory, or
         leave it unset for automatic HuggingFace download.

Usage::

    from astra.inference.tokenizer import load_tokenizer
    tok = load_tokenizer()
    ids = tok.encode("Hello, world!")
    text = tok.decode(ids)
    print(tok.is_stub)  # True until real tokenizer is available
"""

from __future__ import annotations

import logging
import os
import zlib
from typing import List, Optional

log = logging.getLogger("astra.tokenizer")

# HuggingFace model name used when no local path is specified.
_DEFAULT_HF_NAME = "deepseek-ai/DeepSeek-V2"
# Vocabulary size assumed by the stub (DeepSeek-V2 actual: 102400).
_STUB_VOCAB_SIZE = 102400


# ─────────────────────────────────────────────────────────────────────────── #
# Public interface                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class AstraTokenizer:
    """Unified tokenizer wrapper used throughout the Astra codebase."""

    def __init__(self, backend, *, is_stub: bool, vocab_size: int, name: str) -> None:
        self._backend = backend
        self._is_stub = is_stub
        self._vocab_size = vocab_size
        self._name = name

    # ── Core API ──────────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """Convert text to a list of integer token IDs."""
        return self._backend.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs back to text."""
        return self._backend.decode(token_ids)

    # ── Metadata ─────────────────────────────────────────────────────────

    @property
    def is_stub(self) -> bool:
        """True when using the whitespace/CRC32 fallback (no real tokenizer)."""
        return self._is_stub

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        kind = "stub" if self._is_stub else "real"
        return f"AstraTokenizer({kind}, vocab={self._vocab_size}, name={self._name!r})"


# ─────────────────────────────────────────────────────────────────────────── #
# Backends                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class _StubBackend:
    """
    Whitespace tokenizer using CRC32 hashes as token IDs.
    Deterministic and dependency-free; unsuitable for real inference.
    """

    def encode(self, text: str) -> List[int]:
        words = text.split()
        return [zlib.crc32(w.encode()) % _STUB_VOCAB_SIZE for w in words] or [1]

    def decode(self, token_ids: List[int]) -> str:
        return f"[stub — {len(token_ids)} token(s)]"


class _TransformersBackend:
    """Wraps a HuggingFace ``PreTrainedTokenizer``."""

    def __init__(self, hf_tokenizer) -> None:
        self._tok = hf_tokenizer

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────── #
# Factory                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def load_tokenizer(
    path_or_name: Optional[str] = None,
    *,
    offline: bool = False,
) -> AstraTokenizer:
    """
    Load and return an :class:`AstraTokenizer`.

    Parameters
    ----------
    path_or_name:
        Local directory containing ``tokenizer.json`` / ``tokenizer_config.json``,
        or a HuggingFace model name (e.g. ``"deepseek-ai/DeepSeek-V2"``).
        If ``None``, checks ``ASTRA_TOKENIZER_PATH`` then attempts HF Hub download.
    offline:
        If ``True``, skip the HF Hub download attempt and use the stub.
        Useful for air-gapped machines.
    """
    # Resolve path/name
    source = path_or_name or os.environ.get("ASTRA_TOKENIZER_PATH")

    # Try to load via HuggingFace transformers
    if not offline:
        try:
            from transformers import AutoTokenizer  # type: ignore[import]

            name = source or _DEFAULT_HF_NAME
            log.info("Loading tokenizer from %r …", name)
            hf_tok = AutoTokenizer.from_pretrained(
                name,
                trust_remote_code=True,
                local_files_only=(source is not None and os.path.isdir(source)),
            )
            vocab_size = hf_tok.vocab_size or _STUB_VOCAB_SIZE
            log.info("Tokenizer loaded: vocab_size=%d", vocab_size)
            return AstraTokenizer(
                _TransformersBackend(hf_tok),
                is_stub=False,
                vocab_size=vocab_size,
                name=name,
            )
        except ImportError:
            log.debug("transformers not installed — using stub tokenizer")
        except Exception as exc:
            log.warning("Could not load tokenizer (%s) — using stub", exc)

    log.info("Using stub tokenizer (whitespace + CRC32)")
    return AstraTokenizer(
        _StubBackend(),
        is_stub=True,
        vocab_size=_STUB_VOCAB_SIZE,
        name="stub",
    )


# Module-level singleton — lazily initialised on first access.
_GLOBAL: Optional[AstraTokenizer] = None


def get_tokenizer() -> AstraTokenizer:
    """
    Return the module-level shared tokenizer instance.
    Initialised on first call; subsequent calls return the cached instance.
    """
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = load_tokenizer()
    return _GLOBAL


def reset_tokenizer(tok: Optional[AstraTokenizer] = None) -> None:
    """Replace (or clear) the global tokenizer. Primarily for testing."""
    global _GLOBAL
    _GLOBAL = tok
