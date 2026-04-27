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
Peer identity (Phase 3.4 security).

Each Astra node owns an Ed25519 key pair.  The public key is the **node ID**
and is included in every DHT advertisement; the private key signs the
advertisement payload.  Receivers verify the signature against the embedded
public key — preventing peers from impersonating each other or tampering
with capability advertisements after they have been signed.

This is intentionally **TOFU (Trust-On-First-Use)**: there is no central CA.
Once a peer's public key has been seen, future advertisements from that node
must continue to verify against the same key (ratcheting trust).

Usage::

    # On each node, once at startup
    identity = PeerIdentity.load_or_create("/var/lib/astra/identity.key")
    record_payload = {"address": "10.0.0.2:50051", "layers": "0:30", ...}
    signed = identity.sign_payload(record_payload)

    # On the receiving side
    if not verify_signed_payload(signed):
        raise SecurityError("invalid signature on peer advertisement")
"""

from __future__ import annotations

import base64
import json
import logging
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

log = logging.getLogger("astra.identity")

# Public-key fingerprint length (truncated to keep node IDs human-readable)
_NODE_ID_PREFIX_LEN = 16


# ─────────────────────────────────────────────────────────────────────────── #
# Cryptography backend (lazy import — keeps the module usable in stub mode)     #
# ─────────────────────────────────────────────────────────────────────────── #

def _import_ed25519():
    """Lazy import; raises ImportError with a clear message on failure."""
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization
        return ed25519, serialization
    except ImportError as exc:
        raise ImportError(
            "PeerIdentity requires the `cryptography` package. "
            "Install with: pip install cryptography"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────── #
# Public types                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class SignedPayload:
    """A capability advertisement signed by its issuing peer."""

    payload: Dict[str, Any]
    public_key_b64: str
    signature_b64: str
    signed_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "payload": self.payload,
            "public_key": self.public_key_b64,
            "signature": self.signature_b64,
            "signed_at": self.signed_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SignedPayload":
        return cls(
            payload=dict(d["payload"]),
            public_key_b64=d["public_key"],
            signature_b64=d["signature"],
            signed_at=float(d.get("signed_at", 0.0)),
        )


# ─────────────────────────────────────────────────────────────────────────── #
# PeerIdentity                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class PeerIdentity:
    """Ed25519 key pair owned by a single peer."""

    def __init__(self, private_key) -> None:
        self._private_key = private_key
        self._public_key = private_key.public_key()

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def generate(cls) -> "PeerIdentity":
        """Create a fresh Ed25519 key pair in memory."""
        ed25519, _ = _import_ed25519()
        return cls(ed25519.Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(
        cls,
        path: str | pathlib.Path,
    ) -> "PeerIdentity":
        """
        Load an existing key from *path*, or create + persist one if missing.
        File permissions are tightened to 0600 on POSIX systems.
        """
        path = pathlib.Path(path)
        if path.is_file():
            return cls.load(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = cls.generate()
        identity.save(path)
        return identity

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "PeerIdentity":
        ed25519, serialization = _import_ed25519()
        path = pathlib.Path(path)
        with open(path, "rb") as f:
            data = f.read()
        priv = serialization.load_pem_private_key(data, password=None)
        if not isinstance(priv, ed25519.Ed25519PrivateKey):
            raise ValueError(f"Key file {path} is not an Ed25519 private key")
        return cls(priv)

    def save(self, path: str | pathlib.Path) -> None:
        _, serialization = _import_ed25519()
        path = pathlib.Path(path)
        pem = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        with open(path, "wb") as f:
            f.write(pem)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass   # non-POSIX or permission denied — best-effort
        log.info("Wrote identity key to %s", path)

    # ── Public-key accessors ─────────────────────────────────────────────

    def public_key_bytes(self) -> bytes:
        _, serialization = _import_ed25519()
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def public_key_b64(self) -> str:
        return base64.b64encode(self.public_key_bytes()).decode("ascii")

    def node_id(self) -> str:
        """Stable string identifier derived from the public key."""
        return self.public_key_b64()[:_NODE_ID_PREFIX_LEN]

    # ── Signing ──────────────────────────────────────────────────────────

    def sign_payload(self, payload: Dict[str, Any]) -> SignedPayload:
        """Return a SignedPayload wrapping *payload*."""
        signed_at = time.time()
        message = _canonical_message(payload, signed_at)
        sig = self._private_key.sign(message)
        return SignedPayload(
            payload=dict(payload),
            public_key_b64=self.public_key_b64(),
            signature_b64=base64.b64encode(sig).decode("ascii"),
            signed_at=signed_at,
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Verification                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def verify_signed_payload(signed: SignedPayload) -> bool:
    """
    Cryptographically verify a SignedPayload.  Returns ``False`` instead of
    raising so callers can drop bad records without surfacing internal errors.
    """
    try:
        ed25519, _ = _import_ed25519()
        pubkey_bytes = base64.b64decode(signed.public_key_b64)
        sig_bytes = base64.b64decode(signed.signature_b64)
        pubkey = ed25519.Ed25519PublicKey.from_public_bytes(pubkey_bytes)
        message = _canonical_message(signed.payload, signed.signed_at)
        pubkey.verify(sig_bytes, message)
        return True
    except Exception as exc:
        log.debug("Signature verification failed: %s", exc)
        return False


def _canonical_message(payload: Dict[str, Any], signed_at: float) -> bytes:
    """
    Deterministic byte representation of a payload for signing.
    JSON with sorted keys + the ``signed_at`` timestamp prevents replay of
    older advertisements with newer routing decisions.
    """
    canonical = {
        "payload": payload,
        "signed_at": signed_at,
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────── #
# TOFU trust ratchet                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class TrustRegistry:
    """
    Trust-On-First-Use registry: remembers the public key first seen for each
    node ID and rejects any subsequent advertisement that uses a different key.
    """

    def __init__(self) -> None:
        self._known: Dict[str, str] = {}   # node_id → public_key_b64

    def vouch(self, node_id: str, public_key_b64: str) -> bool:
        """
        Record that *node_id* is associated with *public_key_b64*.
        Returns False if a different key was previously registered.
        """
        existing = self._known.get(node_id)
        if existing is None:
            self._known[node_id] = public_key_b64
            return True
        return existing == public_key_b64

    def known_key(self, node_id: str) -> Optional[str]:
        return self._known.get(node_id)

    def forget(self, node_id: str) -> None:
        self._known.pop(node_id, None)

    def __len__(self) -> int:
        return len(self._known)
