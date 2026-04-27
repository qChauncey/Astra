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
Engram memory nodes (Phase 3.1).

An **Engram node** is a P2P peer whose role is **storage only** — it serves
KV-cache fragments and weight shards to compute peers but does not perform
any forward-pass computation itself.  This separates capacity scaling
(memory + bandwidth) from compute scaling (GPU).

Two storage backends are provided:

* ``InMemoryEngramStore`` — keeps everything in a process-local dict.
  Suitable for unit tests and small single-machine clusters.
* ``DiskEngramStore``     — writes blobs to a directory on disk; survives
  restarts.  Production deployments would wrap this with mmap to avoid
  re-reading large weight shards on every fetch.

The ``EngramNode`` glues a store to ``AstraDHT`` so other peers can locate
storage via the DHT advertisement (role=``"engram"``) and then issue
:meth:`get_blob` calls.

The current implementation uses simple Python method calls rather than gRPC
because the existing inference proto only models compute traffic.  Adding a
``StorageService`` to ``inference.proto`` is straightforward and is left as
a follow-up when the first multi-machine Engram is deployed.
"""

from __future__ import annotations

import logging
import pathlib
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

log = logging.getLogger("astra.engram")

ROLE_COMPUTE = "compute"
ROLE_ENGRAM = "engram"


# ─────────────────────────────────────────────────────────────────────────── #
# Storage backends                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

class EngramStore:
    """Abstract base — minimal blob KV interface used by EngramNode."""

    def put_blob(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def get_blob(self, key: str) -> Optional[bytes]:
        raise NotImplementedError

    def delete_blob(self, key: str) -> bool:
        raise NotImplementedError

    def list_keys(self) -> List[str]:
        raise NotImplementedError

    def total_bytes(self) -> int:
        raise NotImplementedError


class InMemoryEngramStore(EngramStore):
    """Process-local dict-backed store — convenient for tests."""

    def __init__(self) -> None:
        self._blobs: Dict[str, bytes] = {}
        self._lock = threading.RLock()

    def put_blob(self, key: str, data: bytes) -> None:
        with self._lock:
            self._blobs[key] = data

    def get_blob(self, key: str) -> Optional[bytes]:
        with self._lock:
            return self._blobs.get(key)

    def delete_blob(self, key: str) -> bool:
        with self._lock:
            return self._blobs.pop(key, None) is not None

    def list_keys(self) -> List[str]:
        with self._lock:
            return sorted(self._blobs.keys())

    def total_bytes(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._blobs.values())


class DiskEngramStore(EngramStore):
    """Persistent file-system store: each blob is one file under root_dir."""

    def __init__(self, root_dir: str | pathlib.Path) -> None:
        self._root = pathlib.Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path_for(self, key: str) -> pathlib.Path:
        # Sanitise: replace path separators so attackers can't traverse.
        safe = key.replace("/", "_").replace("\\", "_")
        return self._root / safe

    def put_blob(self, key: str, data: bytes) -> None:
        with self._lock:
            self._path_for(key).write_bytes(data)

    def get_blob(self, key: str) -> Optional[bytes]:
        path = self._path_for(key)
        with self._lock:
            return path.read_bytes() if path.is_file() else None

    def delete_blob(self, key: str) -> bool:
        path = self._path_for(key)
        with self._lock:
            if path.is_file():
                path.unlink()
                return True
            return False

    def list_keys(self) -> List[str]:
        with self._lock:
            return sorted(p.name for p in self._root.iterdir() if p.is_file())

    def total_bytes(self) -> int:
        with self._lock:
            return sum(p.stat().st_size for p in self._root.iterdir() if p.is_file())


# ─────────────────────────────────────────────────────────────────────────── #
# Capability advertisement                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class EngramCapability:
    """
    Describes what an Engram node holds. Published into the DHT so compute
    peers can find storage for their KV-cache fragments / weight shards.
    """
    node_id: str
    address: str
    role: str = ROLE_ENGRAM
    capacity_bytes: int = 0
    used_bytes: int = 0
    blob_keys: List[str] = field(default_factory=list)
    geo_region: str = "local"
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "address": self.address,
            "role": self.role,
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "blob_keys": list(self.blob_keys),
            "geo_region": self.geo_region,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EngramCapability":
        return cls(
            node_id=d["node_id"],
            address=d["address"],
            role=d.get("role", ROLE_ENGRAM),
            capacity_bytes=int(d.get("capacity_bytes", 0)),
            used_bytes=int(d.get("used_bytes", 0)),
            blob_keys=list(d.get("blob_keys", [])),
            geo_region=d.get("geo_region", "local"),
            last_seen=float(d.get("last_seen", time.time())),
        )


# ─────────────────────────────────────────────────────────────────────────── #
# EngramNode                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

class EngramNode:
    """
    Storage-only P2P peer.

    Wraps an :class:`EngramStore` plus a thin advertisement layer so other
    peers can discover us through the shared :class:`AstraDHT`.

    Parameters
    ----------
    node_id:        Unique peer identifier.
    address:        ``host:port`` advertised to compute peers.
    store:          Backing storage (defaults to in-memory).
    capacity_bytes: Soft cap on total stored bytes.  ``put_blob`` raises
                    :class:`MemoryError` once exceeded.
    geo_region:     Region label used by ``GeoAwareMoERouter`` to prefer
                    nearby Engrams.
    """

    def __init__(
        self,
        node_id: str,
        address: str,
        store: Optional[EngramStore] = None,
        capacity_bytes: int = 0,
        geo_region: str = "local",
    ) -> None:
        self.node_id = node_id
        self.address = address
        self.geo_region = geo_region
        self.capacity_bytes = capacity_bytes
        self._store = store or InMemoryEngramStore()
        self._lock = threading.RLock()

    # ── Storage API ──────────────────────────────────────────────────────

    def put_blob(self, key: str, data: bytes) -> None:
        """Store *data* under *key*. Raises if the soft capacity is exceeded."""
        with self._lock:
            new_used = self._store.total_bytes() + len(data)
            if 0 < self.capacity_bytes < new_used:
                raise MemoryError(
                    f"Engram {self.node_id!r} would exceed capacity "
                    f"({new_used} > {self.capacity_bytes} bytes)"
                )
            self._store.put_blob(key, data)
            log.debug("Engram %s stored %d bytes under %r", self.node_id, len(data), key)

    def get_blob(self, key: str) -> Optional[bytes]:
        return self._store.get_blob(key)

    def delete_blob(self, key: str) -> bool:
        return self._store.delete_blob(key)

    def has_blob(self, key: str) -> bool:
        return self._store.get_blob(key) is not None

    # ── Capability ───────────────────────────────────────────────────────

    def capability(self) -> EngramCapability:
        return EngramCapability(
            node_id=self.node_id,
            address=self.address,
            role=ROLE_ENGRAM,
            capacity_bytes=self.capacity_bytes,
            used_bytes=self._store.total_bytes(),
            blob_keys=self._store.list_keys(),
            geo_region=self.geo_region,
        )

    # ── DHT integration ──────────────────────────────────────────────────

    def announce(self, dht) -> None:
        """Publish our capability to *dht* so compute peers can find us."""
        cap = self.capability()
        dht.set(_engram_dht_key(self.node_id), cap.to_dict(), ttl=120.0)
        log.info(
            "Engram %s announced: %d bytes used / %d capacity",
            self.node_id, cap.used_bytes, cap.capacity_bytes,
        )

    def revoke(self, dht) -> None:
        dht.delete(_engram_dht_key(self.node_id))


# ─────────────────────────────────────────────────────────────────────────── #
# Discovery helpers                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

_ENGRAM_PREFIX = "astra/engrams/"


def _engram_dht_key(node_id: str) -> str:
    return f"{_ENGRAM_PREFIX}{node_id}"


def discover_engrams(dht) -> List[EngramCapability]:
    """Return all currently advertised Engram nodes from *dht*."""
    found: List[EngramCapability] = []
    seen: Set[str] = set()
    for key, value in dht.scan(_ENGRAM_PREFIX):
        node_id = key[len(_ENGRAM_PREFIX):]
        if node_id in seen:
            continue
        seen.add(node_id)
        try:
            found.append(EngramCapability.from_dict(value))
        except Exception as exc:
            log.warning("Skipping malformed engram record %r: %s", key, exc)
    return found


def find_blob_holder(dht, blob_key: str) -> Optional[EngramCapability]:
    """Find any Engram node currently advertising *blob_key*.  None if none."""
    for cap in discover_engrams(dht):
        if blob_key in cap.blob_keys:
            return cap
    return None
