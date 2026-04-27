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
Weight shard integrity manifest (Phase 3.4 security).

Goal
----
Prevent malicious peers in a P2P network from serving tampered model weights.
A trusted manifest is published once (by the model author) and every node
verifies its local shards match the manifest before loading them.

Manifest format
---------------
JSON file at ``model_dir/astra_manifest.json``::

    {
      "version": 1,
      "model": "deepseek-v4-flash",
      "algorithm": "sha256",
      "created_at": "2026-04-27T00:00:00Z",
      "shards": {
        "model-00001-of-00163.safetensors": "a3f5...e21",
        "model-00002-of-00163.safetensors": "b7c2...91d",
        ...
        "tokenizer.json": "...",
        "config.json": "..."
      }
    }

Trust model
-----------
The manifest itself is **out of scope** for hashing — it is expected to be
distributed via a separate trusted channel (HTTPS download, signed git tag,
etc.). What the manifest guarantees is that after you have a valid manifest,
no peer can swap a shard underneath you.

Usage::

    # Manifest author side
    manifest = WeightManifest.create_from_dir("/data/deepseek-v4")
    manifest.save("/data/deepseek-v4/astra_manifest.json")

    # Node side (during weight loading)
    manifest = WeightManifest.load("/data/deepseek-v4/astra_manifest.json")
    ok, mismatched = manifest.verify_dir("/data/deepseek-v4")
    if not ok:
        raise RuntimeError(f"Tampered shards detected: {mismatched}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("astra.weight_manifest")

MANIFEST_FILENAME = "astra_manifest.json"
MANIFEST_VERSION = 1
DEFAULT_ALGORITHM = "sha256"

# Files relevant to a model checkpoint (extension-based; case-sensitive)
_DEFAULT_INCLUDE_EXTENSIONS = {".safetensors", ".bin", ".json", ".model", ".txt"}
# Always exclude the manifest itself and OS junk
_DEFAULT_EXCLUDE_FILES = {MANIFEST_FILENAME, ".DS_Store", "Thumbs.db"}

# Read in chunks so we can hash files larger than memory.
_HASH_CHUNK_BYTES = 1 << 20   # 1 MiB


# ─────────────────────────────────────────────────────────────────────────── #
# Hashing helpers                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def hash_file(path: pathlib.Path, algorithm: str = DEFAULT_ALGORITHM) -> str:
    """Return the hex-encoded digest of *path* using *algorithm*."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _list_shards(model_dir: pathlib.Path) -> List[pathlib.Path]:
    """Return the sorted list of shard files relevant for a checkpoint."""
    files: List[pathlib.Path] = []
    for entry in sorted(model_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.name in _DEFAULT_EXCLUDE_FILES:
            continue
        if entry.suffix.lower() not in _DEFAULT_INCLUDE_EXTENSIONS:
            continue
        files.append(entry)
    return files


# ─────────────────────────────────────────────────────────────────────────── #
# Manifest                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class WeightManifest:
    """In-memory representation of a weight shard manifest."""

    model: str
    shards: Dict[str, str]
    algorithm: str = DEFAULT_ALGORITHM
    version: int = MANIFEST_VERSION
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def create_from_dir(
        cls,
        model_dir: str | pathlib.Path,
        model_name: str = "unknown",
        algorithm: str = DEFAULT_ALGORITHM,
    ) -> "WeightManifest":
        """Hash every shard in *model_dir* and return a fresh manifest."""
        model_dir = pathlib.Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Not a directory: {model_dir}")

        shards: Dict[str, str] = {}
        for path in _list_shards(model_dir):
            log.debug("Hashing %s …", path.name)
            shards[path.name] = hash_file(path, algorithm)
        log.info("Created manifest: %d shards in %s", len(shards), model_dir)
        return cls(model=model_name, shards=shards, algorithm=algorithm)

    @classmethod
    def load(cls, manifest_path: str | pathlib.Path) -> "WeightManifest":
        """Read a manifest JSON file from disk."""
        path = pathlib.Path(manifest_path)
        with open(path) as f:
            data = json.load(f)
        if data.get("version") != MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported manifest version {data.get('version')}; expected {MANIFEST_VERSION}"
            )
        return cls(
            model=data.get("model", "unknown"),
            shards=dict(data.get("shards", {})),
            algorithm=data.get("algorithm", DEFAULT_ALGORITHM),
            version=int(data["version"]),
            created_at=data.get("created_at", ""),
        )

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, manifest_path: str | pathlib.Path) -> None:
        """Serialize to JSON on disk."""
        path = pathlib.Path(manifest_path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        log.info("Wrote manifest: %s", path)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "model": self.model,
            "algorithm": self.algorithm,
            "created_at": self.created_at,
            "shards": dict(self.shards),
        }

    # ── Verification ─────────────────────────────────────────────────────

    def verify_file(self, file_path: str | pathlib.Path) -> bool:
        """Return True if *file_path* matches the recorded hash."""
        path = pathlib.Path(file_path)
        expected = self.shards.get(path.name)
        if expected is None:
            return False
        actual = hash_file(path, self.algorithm)
        return actual == expected

    def verify_dir(
        self,
        model_dir: str | pathlib.Path,
        *,
        require_all: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        Hash every file in *model_dir* and compare against the manifest.

        Returns
        -------
        (ok, mismatched):
            ``ok`` is True when every shard present in *model_dir* matches its
            recorded hash.  ``mismatched`` is the list of filenames that
            differed (or were missing if ``require_all=True``).
        """
        model_dir = pathlib.Path(model_dir)
        mismatched: List[str] = []
        present = {p.name for p in _list_shards(model_dir)}

        for name, expected in self.shards.items():
            path = model_dir / name
            if not path.is_file():
                if require_all:
                    mismatched.append(name)
                continue
            actual = hash_file(path, self.algorithm)
            if actual != expected:
                mismatched.append(name)

        if require_all:
            for name in present:
                if name not in self.shards:
                    mismatched.append(name)

        return (len(mismatched) == 0, sorted(set(mismatched)))

    # ── Convenience ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.shards)

    def __contains__(self, name: str) -> bool:
        return name in self.shards


# ─────────────────────────────────────────────────────────────────────────── #
# Public helper for WeightLoader                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def find_manifest(model_dir: str | pathlib.Path) -> Optional[pathlib.Path]:
    """Return the manifest path inside *model_dir* if one exists."""
    candidate = pathlib.Path(model_dir) / MANIFEST_FILENAME
    return candidate if candidate.is_file() else None
