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
#   - Written from scratch as a pure-Python DHT simulation layer.
#   - API surface mirrors hivemind.DHT so it can be swapped in for Phase 3.
#   - Thread-safe in-process store; multi-process mode via shared dict stub.

"""
Mock Distributed Hash Table (DHT) for Astra peer discovery.

Mimics the subset of hivemind.DHT that Astra needs:
  - store(key, value, ttl)
  - get(key) → value | None
  - node registration / deregistration
  - change subscribers (callbacks on key updates)

Drop-in replacement path:
    import hivemind
    dht = hivemind.DHT(initial_peers=[...], start=True)
    # ↑ same interface as AstraDHT below once Phase 3 integration is done.

In-process mode (default): all nodes in the same Python process share one
global `_GlobalStore`.  This is sufficient for mock_pipeline.py and tests.

Multi-process mode (future): replace `_GlobalStore` with a multiprocessing
Manager dict or a real hivemind DHT instance.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────── #
# Global in-process store (shared across all AstraDHT instances in one run)   #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class _Entry:
    value: Any
    expires_at: float         # Unix timestamp; 0 = never expires
    owner: str                # node_id that stored this entry


class _GlobalStore:
    """Thread-safe key-value store with TTL expiry."""

    def __init__(self) -> None:
        self._data: Dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._subscribers: Dict[str, List[Callable]] = {}
        self._start_reaper()

    def put(self, key: str, value: Any, ttl: float, owner: str) -> None:
        expires = time.time() + ttl if ttl > 0 else 0.0
        with self._lock:
            self._data[key] = _Entry(value, expires, owner)
        self._notify(key, value)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.expires_at and time.time() > entry.expires_at:
                del self._data[key]
                return None
            return entry.value

    def delete(self, key: str, owner: str) -> bool:
        with self._lock:
            entry = self._data.get(key)
            if entry and entry.owner == owner:
                del self._data[key]
                return True
            return False

    def keys_by_prefix(self, prefix: str) -> List[str]:
        now = time.time()
        with self._lock:
            return [
                k for k, e in self._data.items()
                if k.startswith(prefix)
                and (e.expires_at == 0 or e.expires_at > now)
            ]

    def subscribe(self, key_prefix: str, callback: Callable) -> None:
        with self._lock:
            self._subscribers.setdefault(key_prefix, []).append(callback)

    def _notify(self, key: str, value: Any) -> None:
        with self._lock:
            cbs = [
                cb for prefix, cbs in self._subscribers.items()
                for cb in cbs if key.startswith(prefix)
            ]
        for cb in cbs:
            try:
                cb(key, value)
            except Exception:
                pass

    def _reap(self) -> None:
        while True:
            time.sleep(5.0)
            now = time.time()
            with self._lock:
                expired = [k for k, e in self._data.items()
                           if e.expires_at and e.expires_at < now]
                for k in expired:
                    del self._data[k]

    def _start_reaper(self) -> None:
        t = threading.Thread(target=self._reap, daemon=True)
        t.start()

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            return {
                k: e.value for k, e in self._data.items()
                if e.expires_at == 0 or e.expires_at > now
            }


# One shared store per process
_GLOBAL_STORE = _GlobalStore()


# ─────────────────────────────────────────────────────────────────────────── #
# Node record stored in DHT                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class DHTNodeRecord:
    """Capability advertisement stored by each peer in the DHT."""
    node_id: str
    address: str               # "host:port"
    layer_start: int
    layer_end: int
    expert_shards: List[int]
    geo_region: str
    backend: str = "numpy_stub"
    gpu_util: float = 0.0
    cpu_util: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "address": self.address,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "expert_shards": self.expert_shards,
            "geo_region": self.geo_region,
            "backend": self.backend,
            "gpu_util": self.gpu_util,
            "cpu_util": self.cpu_util,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DHTNodeRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────────────────── #
# AstraDHT — the public API                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

_NODE_PREFIX = "astra/nodes/"
_DEFAULT_TTL = 60.0   # seconds; nodes must re-announce to stay visible


class AstraDHT:
    """
    Astra mock DHT.  Drop-in for hivemind.DHT for local and LAN simulation.

    Usage::

        dht = AstraDHT(node_id="node-A")
        dht.announce(record)          # publish capability
        peers = dht.get_all_peers()   # discover others
        dht.subscribe_peers(callback) # react to new peers
        dht.revoke()                  # leave the network
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        store: Optional[_GlobalStore] = None,
        heartbeat_interval: float = 20.0,
    ) -> None:
        self._node_id = node_id or uuid.uuid4().hex[:12]
        self._store = store or _GLOBAL_STORE
        self._heartbeat_interval = heartbeat_interval
        self._record: Optional[DHTNodeRecord] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()

    @property
    def node_id(self) -> str:
        return self._node_id

    # ------------------------------------------------------------------ #
    # Announce / revoke                                                     #
    # ------------------------------------------------------------------ #

    def announce(self, record: DHTNodeRecord, ttl: float = _DEFAULT_TTL) -> None:
        """Publish this node's capabilities to the DHT and start heartbeating."""
        self._record = record
        self._store.put(
            f"{_NODE_PREFIX}{record.node_id}",
            record.to_dict(),
            ttl=ttl,
            owner=self._node_id,
        )
        if self._heartbeat_thread is None or not self._heartbeat_thread.is_alive():
            self._stopped.clear()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, args=(ttl,), daemon=True
            )
            self._heartbeat_thread.start()

    def revoke(self) -> None:
        """Remove this node from the DHT."""
        self._stopped.set()
        if self._record:
            self._store.delete(
                f"{_NODE_PREFIX}{self._record.node_id}",
                owner=self._node_id,
            )
            self._record = None

    def _heartbeat_loop(self, ttl: float) -> None:
        while not self._stopped.wait(self._heartbeat_interval):
            if self._record:
                self._store.put(
                    f"{_NODE_PREFIX}{self._record.node_id}",
                    self._record.to_dict(),
                    ttl=ttl,
                    owner=self._node_id,
                )

    # ------------------------------------------------------------------ #
    # Discovery                                                             #
    # ------------------------------------------------------------------ #

    def get_all_peers(self) -> List[DHTNodeRecord]:
        keys = self._store.keys_by_prefix(_NODE_PREFIX)
        records = []
        for k in keys:
            v = self._store.get(k)
            if v:
                try:
                    records.append(DHTNodeRecord.from_dict(v))
                except Exception:
                    pass
        return records

    def get_peer(self, node_id: str) -> Optional[DHTNodeRecord]:
        v = self._store.get(f"{_NODE_PREFIX}{node_id}")
        if v:
            try:
                return DHTNodeRecord.from_dict(v)
            except Exception:
                return None
        return None

    def peers_for_layer(self, layer_idx: int) -> List[DHTNodeRecord]:
        return [
            r for r in self.get_all_peers()
            if r.layer_start <= layer_idx < r.layer_end
        ]

    def peers_for_expert(self, expert_id: int) -> List[DHTNodeRecord]:
        return [
            r for r in self.get_all_peers()
            if expert_id in r.expert_shards
        ]

    # ------------------------------------------------------------------ #
    # Generic key-value API (used by EngramNode and other helpers)         #
    # ------------------------------------------------------------------ #

    def set(self, key: str, value: Any, ttl: float = _DEFAULT_TTL) -> None:
        """Publish an arbitrary value under *key*; we are the owner."""
        self._store.put(key, value, ttl=ttl, owner=self._node_id)

    def get(self, key: str) -> Optional[Any]:
        """Fetch an arbitrary value from the DHT, or None if absent/expired."""
        return self._store.get(key)

    def delete(self, key: str) -> bool:
        """Remove a key we own. Returns True if deleted."""
        return self._store.delete(key, owner=self._node_id)

    def scan(self, prefix: str) -> List[tuple]:
        """Yield (key, value) pairs for every non-expired key starting with *prefix*."""
        out = []
        for k in self._store.keys_by_prefix(prefix):
            v = self._store.get(k)
            if v is not None:
                out.append((k, v))
        return out

    def subscribe_peers(self, callback: Callable[[str, DHTNodeRecord], None]) -> None:
        """Call `callback(node_id, record)` whenever a peer announces."""
        def _wrap(key: str, value: dict) -> None:
            node_id = key.removeprefix(_NODE_PREFIX)
            try:
                callback(node_id, DHTNodeRecord.from_dict(value))
            except Exception:
                pass
        self._store.subscribe(_NODE_PREFIX, _wrap)

    # ------------------------------------------------------------------ #
    # Generic KV store (for Engram memory nodes)                            #
    # ------------------------------------------------------------------ #

    def store(self, key: str, value: Any, ttl: float = 300.0) -> None:
        self._store.put(key, value, ttl=ttl, owner=self._node_id)

    def fetch(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    def snapshot(self) -> dict:
        return self._store.snapshot()

    def __repr__(self) -> str:
        peers = len(self.get_all_peers())
        return f"AstraDHT(node_id={self._node_id}, peers={peers})"
