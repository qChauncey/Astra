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

"""Tests for astra.inference.weight_manifest."""

from __future__ import annotations

import json
import pathlib

import pytest

from astra.inference.weight_manifest import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    WeightManifest,
    find_manifest,
    hash_file,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def model_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build a small fake checkpoint with two safetensors and a config."""
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"shard one bytes")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"shard two bytes")
    (tmp_path / "config.json").write_text(json.dumps({"hidden_dim": 7168}))
    (tmp_path / "tokenizer.model").write_bytes(b"fake tokenizer")
    # Junk files that should be excluded
    (tmp_path / "README.md").write_text("ignore me")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x01")
    return tmp_path


# ── hash_file ─────────────────────────────────────────────────────────────────

class TestHashFile:
    def test_returns_hex_digest(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello")
        digest = hash_file(p)
        assert isinstance(digest, str)
        assert len(digest) == 64  # SHA-256 hex length

    def test_deterministic(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"abc")
        assert hash_file(p) == hash_file(p)

    def test_different_content_different_hash(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"one")
        b.write_bytes(b"two")
        assert hash_file(a) != hash_file(b)


# ── create_from_dir ───────────────────────────────────────────────────────────

class TestCreateFromDir:
    def test_lists_only_relevant_files(self, model_dir):
        m = WeightManifest.create_from_dir(model_dir, model_name="test")
        # safetensors + config.json + tokenizer.model = 4
        assert len(m) == 4
        assert "README.md" not in m
        assert ".DS_Store" not in m

    def test_records_correct_hashes(self, model_dir):
        m = WeightManifest.create_from_dir(model_dir, model_name="test")
        for name in m.shards:
            actual = hash_file(model_dir / name)
            assert m.shards[name] == actual

    def test_model_name_recorded(self, model_dir):
        m = WeightManifest.create_from_dir(model_dir, model_name="deepseek-v4")
        assert m.model == "deepseek-v4"

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            WeightManifest.create_from_dir(tmp_path / "nope")


# ── save / load round-trip ───────────────────────────────────────────────────

class TestSaveLoad:
    def test_roundtrip(self, model_dir):
        m1 = WeightManifest.create_from_dir(model_dir, model_name="x")
        path = model_dir / MANIFEST_FILENAME
        m1.save(path)
        m2 = WeightManifest.load(path)
        assert m2.shards == m1.shards
        assert m2.model == "x"
        assert m2.algorithm == m1.algorithm

    def test_saved_file_is_valid_json(self, model_dir):
        m = WeightManifest.create_from_dir(model_dir, model_name="x")
        path = model_dir / MANIFEST_FILENAME
        m.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["version"] == MANIFEST_VERSION
        assert "shards" in data

    def test_unsupported_version_rejected(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"version": 999, "shards": {}}))
        with pytest.raises(ValueError, match="version"):
            WeightManifest.load(path)


# ── verify ────────────────────────────────────────────────────────────────────

class TestVerify:
    def _build(self, model_dir):
        m = WeightManifest.create_from_dir(model_dir, model_name="x")
        return m

    def test_clean_dir_passes(self, model_dir):
        m = self._build(model_dir)
        ok, mismatched = m.verify_dir(model_dir)
        assert ok is True
        assert mismatched == []

    def test_tampered_shard_detected(self, model_dir):
        m = self._build(model_dir)
        (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"EVIL replacement bytes")
        ok, mismatched = m.verify_dir(model_dir)
        assert ok is False
        assert "model-00001-of-00002.safetensors" in mismatched

    def test_missing_shard_silent_unless_require_all(self, model_dir):
        m = self._build(model_dir)
        (model_dir / "config.json").unlink()
        ok, mismatched = m.verify_dir(model_dir)
        assert ok is True   # missing files are tolerated by default
        ok2, mismatched2 = m.verify_dir(model_dir, require_all=True)
        assert ok2 is False
        assert "config.json" in mismatched2

    def test_extra_shard_detected_when_require_all(self, model_dir):
        m = self._build(model_dir)
        (model_dir / "extra.safetensors").write_bytes(b"unauthorised file")
        ok, mismatched = m.verify_dir(model_dir, require_all=True)
        assert ok is False
        assert "extra.safetensors" in mismatched

    def test_verify_file(self, model_dir):
        m = self._build(model_dir)
        good = model_dir / "model-00001-of-00002.safetensors"
        assert m.verify_file(good) is True

    def test_verify_file_unknown_returns_false(self, model_dir):
        m = self._build(model_dir)
        unknown = model_dir / "model-00001-of-00002.safetensors"
        unknown.write_bytes(b"DIFFERENT")
        assert m.verify_file(unknown) is False


# ── find_manifest ─────────────────────────────────────────────────────────────

class TestFindManifest:
    def test_finds_when_present(self, model_dir):
        (model_dir / MANIFEST_FILENAME).write_text(json.dumps({"version": 1, "shards": {}}))
        assert find_manifest(model_dir) is not None

    def test_returns_none_when_absent(self, model_dir):
        assert find_manifest(model_dir) is None
