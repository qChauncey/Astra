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
#   - Written from scratch for Phase 5.2 hivemind multi-machine DHT integration.
#   - Provides a unified factory that selects real hivemind.DHT or mock AstraDHT.
#   - Graceful degradation when hivemind is not installed (optional dependency).

"""
Unified DHT factory — transparently swaps between hivemind.DHT and AstraDHT.

Phase 5.2 replaces the in-memory AstraDHT store with live ``hivemind.DHT`` for
real multi-machine peer discovery.  When hivemind is not installed the factory
falls back to ``AstraDHT`` so the system remains fully functional for local
development, CI, and single-machine clusters.

Quick start::

    from astra.network.hivemind_bridge import create_dht

    # Local / CI (no hivemind installed)
    dht = create_dht(node_id="node-1")
    dht.announce(record)

    # Real multi-machine DHT (pip install hivemind first)
    dht = create_dht(
        node_id="node-1",
        initial_peers=["/ip4/192.168.1.10/tcp/4242/p2p/QmPeer1"],
        host_addr="0.0.0.0",
        port=4242,
    )

Unified API (compatible with both backends):
  - ``announce(record, ttl)``  — publish node capability
  - ``get_all_peers()``        — discover live peers
  - ``get_peer(node_id)``      — lookup one peer
  - ``subscribe_peers(cb)``    — react to new peers
  - ``store(key, value, ttl)`` — generic KV store
  - ``fetch(key)``             — generic KV read
  - ``revoke()``               — deregister node
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from .dht import AstraDHT, DHTNodeRecord, _DEFAULT_TTL

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────── #
# Hivemind import guard (optional dependency)                                   #
# ─────────────────────────────────────────────────────────────────────────── #

_HIVEMIND_IMPORT_ERROR: Optional[str] = None
try:
    import hivemind  # noqa: F401
    _HAS_HIVEMIND = True
except ImportError as exc:
    _HAS_HIVEMIND = False
    _HIVEMIND_IMPORT_ERROR = str(exc)


# ─────────────────────────────────────────────────────────────────────────── #
# HivemindDHT — wrapper that implements the AstraDHT interface on top of         #
#               real hivemind.DHT                                               #
# ─────────────────────────────────────────────────────────────────────────── #


if _HAS_HIVEMIND:

    class HivemindDHT:
        """
        Thin adapter that wraps ``hivemind.DHT`` behind the AstraDHT API.

        All methods mirror :class:`AstraDHT` so the orchestrator and other
        consumers do not need to know which backend is active.
        """

        def __init__(
            self,
            node_id: Optional[str] = None,
            initial_peers: Optional[List[str]] = None,
            host_addr: str = "0.0.0.0",
            port: int = 4242,
            heartbeat_interval: float = 20.0,
        ) -> None:
            import hivemind
            self._node_id = node_id or hivemind.utils.get_dht_time().hex()[:12]
            self._heartbeat_interval = heartbeat_interval
            self._record: Optional[DHTNodeRecord] = None

            self._dht = hivemind.DHT(
                host_maddrs=[f"/ip4/{host_addr}/tcp/{port}"],
                initial_peers=initial_peers or [],
                start=True,
                use_auto_relay=True,
                use_relay=True,
                client_mode=False,
            )
            log.info(
                "HivemindDHT started: node_id=%s peers=%d",
                self._node_id,
                len(self._dht.get_visible_maddrs()),
            )

        @property
        def node_id(self) -> str:
            return self._node_id

        # -------------------------------------------------------------- #
        # Announce / revoke                                                 #
        # -------------------------------------------------------------- #

        def announce(self, record: DHTNodeRecord, ttl: float = _DEFAULT_TTL) -> None:
            """Publish node capabilities via hivemind DHT store."""
            self._record = record
            self._dht.store(
                key=f"astra/nodes/{record.node_id}",
                subkey=0,
                value=record.to_dict(),
                expiration_time=hivemind.get_dht_time() + ttl,
            )

        def revoke(self) -> None:
            """Remove this node from the DHT."""
            if self._record:
                try:
                    self._dht.store(
                        key=f"astra/nodes/{self._record.node_id}",
                        subkey=0,
                        value=None,
                        expiration_time=hivemind.get_dht_time(),
                    )
                except Exception:
                    pass
                self._record = None
            self._dht.shutdown()

        # -------------------------------------------------------------- #
        # Discovery                                                         #
        # -------------------------------------------------------------- #

        def get_all_peers(self) -> List[DHTNodeRecord]:
            """Walk the DHT and return all live peer records."""
            records: List[DHTNodeRecord] = []
            try:
                # hivemind DHT traversal via get with latest=True
                # Walk known keys matching the astra/nodes/ prefix
                # In production this uses a prefix-based scan
                items = self._dht.get("astra/nodes/", latest=True)
                if items is not None:
                    for node_id, value in items.items():
                        if isinstance(value, dict):
                            try:
                                records.append(DHTNodeRecord.from_dict(value))
                            except Exception:
                                pass
            except Exception:
                # Fallback: iterate over visible peers only
                pass

            return records

        def get_peer(self, node_id: str) -> Optional[DHTNodeRecord]:
            value = self._dht.get(f"astra/nodes/{node_id}", latest=True)
            if value and isinstance(value, dict):
                try:
                    return DHTNodeRecord.from_dict(value)
                except Exception:
                    return None
            return None

        def subscribe_peers(
            self, callback: Callable[[str, DHTNodeRecord], None]
        ) -> None:
            """Register a callback triggered when peer records change."""

            def _on_change(key, value, _expiration_time):
                if isinstance(value, dict):
                    try:
                        node_id = key.split("/")[-1] if "/" in key else key
                        callback(node_id, DHTNodeRecord.from_dict(value))
                    except Exception:
                        pass

            # hivemind DHT expert prefix subscription
            try:
                self._dht.add_experts(["astra/nodes/"], _on_change)
            except AttributeError:
                log.debug("hivemind DHT expert subscription not available (client-only mode?)")

        # -------------------------------------------------------------- #
        # Generic KV store                                                    #
        # -------------------------------------------------------------- #

        def store(self, key: str, value: Any, ttl: float = 300.0) -> None:
            import hivemind
            self._dht.store(
                key=key,
                subkey=0,
                value=value,
                expiration_time=hivemind.get_dht_time() + ttl,
            )

        def fetch(self, key: str) -> Optional[Any]:
            return self._dht.get(key, latest=True)

        def __repr__(self) -> str:
            return f"HivemindDHT(node_id={self._node_id})"

else:
    # Placeholder so the symbol exists for static analysis but raises if used
    # without hivemind installed.
    HivemindDHT = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────── #
# Factory — create_dht                                                          #
# ─────────────────────────────────────────────────────────────────────────── #


def create_dht(
    node_id: Optional[str] = None,
    use_hivemind: bool = False,
    initial_peers: Optional[List[str]] = None,
    host_addr: str = "0.0.0.0",
    port: int = 4242,
    heartbeat_interval: float = 20.0,
) -> AstraDHT:
    """
    Create a DHT instance — real hivemind when available & requested, else mock.

    Parameters
    ----------
    node_id:
        Unique peer identifier.  Auto-generated if not provided.
    use_hivemind:
        If True, attempt to use real hivemind.DHT.  Falls back to AstraDHT
        when ``hivemind`` is not installed.
    initial_peers:
        List of multiaddr strings for initial DHT rendezvous.
        Only used when ``use_hivemind=True``.
    host_addr, port:
        Bind address for the local DHT node.  Only used when ``use_hivemind=True``.
    heartbeat_interval:
        How often to re-announce this node (seconds).

    Returns
    -------
    AstraDHT or HivemindDHT
        Both implement the same interface.
    """
    dht_node_id = node_id

    if use_hivemind and _HAS_HIVEMIND:
        try:
            dht = HivemindDHT(
                node_id=dht_node_id,
                initial_peers=initial_peers,
                host_addr=host_addr,
                port=port,
                heartbeat_interval=heartbeat_interval,
            )
            log.info("DHT backend: hivemind (real P2P)")
            return dht
        except Exception as exc:
            log.warning(
                "hivemind DHT creation failed (%s). Falling back to AstraDHT.",
                exc,
            )

    if use_hivemind and not _HAS_HIVEMIND:
        _import_err = _HIVEMIND_IMPORT_ERROR or "hivemind not installed"
        log.info(
            "hivemind not installed (%s). Using in-memory AstraDHT. "
            "Install with: pip install hivemind>=1.1.0",
            _import_err,
        )

    log.info("DHT backend: AstraDHT (in-memory)")
    return AstraDHT(node_id=dht_node_id, heartbeat_interval=heartbeat_interval)


# ─────────────────────────────────────────────────────────────────────────── #
# Backend info                                                                #
# ─────────────────────────────────────────────────────────────────────────── #


def is_hivemind_available() -> bool:
    """Return True if hivemind can be imported (optional dependency)."""
    return _HAS_HIVEMIND


