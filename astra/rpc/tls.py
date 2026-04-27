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
#   - Written from scratch for Phase 5 gRPC mTLS.
#   - Self-signed X.509 certificate generation via cryptography library.
#   - TOFU (Trust-On-First-Use) trust store for P2P certificate pinning.
#   - Server and client gRPC credential helpers.

"""
Astra gRPC mTLS — certificate generation, credential loading, TOFU trust.

Phase 5 replaces insecure gRPC channels with mutual TLS (mTLS) using
self-signed X.509 certificates and a Trust-On-First-Use (TOFU) pinning model.

Quick start::

    # Generate per-node certificates
    bundle = generate_self_signed_cert_bundle(node_id="node-1")
    cert_path, key_path = bundle.write("/etc/astra/certs")

    # Server
    from grpc import ssl_server_credentials
    creds = load_server_credentials(cert_path, key_path, ca_cert_path)

    # Client
    from grpc import ssl_channel_credentials
    creds = load_client_credentials(cert_path, key_path, ca_cert_path)

    # TOFU trust store (for P2P without a central CA)
    store = TofuTrustStore()
    store.add("peer-node", peer_cert_pem)
    store.verify("peer-node", peer_cert_pem)  # → True
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import grpc

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────── #
# CertBundle — PEM certificate + private key for one node                       #
# ─────────────────────────────────────────────────────────────────────────── #


@dataclass
class CertBundle:
    """X.509 certificate and RSA private key in PEM format."""

    node_id: str
    cert_pem: str
    key_pem: str

    def write(self, directory: str) -> Tuple[str, str]:
        """Write cert and key to *directory*, returning (cert_path, key_path)."""
        os.makedirs(directory, exist_ok=True)
        cert_path = os.path.join(directory, f"{self.node_id}.cert.pem")
        key_path = os.path.join(directory, f"{self.node_id}.key.pem")
        with open(cert_path, "w") as f:
            f.write(self.cert_pem)
        with open(key_path, "w") as f:
            f.write(self.key_pem)
        return cert_path, key_path


def generate_self_signed_cert_bundle(
    node_id: str,
    common_name: Optional[str] = None,
    days_valid: int = 365,
    key_size: int = 2048,
) -> CertBundle:
    """
    Generate a self-signed X.509 certificate + RSA key pair for *node_id*.

    Parameters
    ----------
    node_id:      Unique peer identifier (embedded in SAN).
    common_name:  CN subject field.  Defaults to ``"astra.{node_id}"``.
    days_valid:   Certificate lifetime in days.
    key_size:     RSA key modulus size in bits.
    """
    cn = common_name or f"astra.{node_id}"
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Project Astra"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_dt.datetime.now(_dt.timezone.utc))
        .not_valid_after(
            _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(days=days_valid)
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(node_id)]),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .sign(private_key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode("ascii")
    key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    return CertBundle(node_id=node_id, cert_pem=cert_pem, key_pem=key_pem)


# ─────────────────────────────────────────────────────────────────────────── #
# TLSConfig — whether and how to enable mTLS                                     #
# ─────────────────────────────────────────────────────────────────────────── #


@dataclass
class TLSConfig:
    """Configuration for gRPC mTLS.

    Parameters
    ----------
    enabled:       Enable TLS (both server and client side).
    cert_path:     Path to this node's X.509 certificate (PEM).
    key_path:      Path to this node's private key (PEM).
    ca_cert_path:  Path to the CA certificate bundle for verifying peers.
                   Required for mTLS; optional for server-only TLS.
    """

    enabled: bool = False
    cert_path: str = ""
    key_path: str = ""
    ca_cert_path: str = ""

    def is_ready(self) -> bool:
        """Return True when all required paths are provided."""
        if not self.enabled:
            return False
        if not self.cert_path or not self.key_path:
            return False
        # CA cert is required for mTLS; server-only can skip it
        return bool(self.ca_cert_path)


# ─────────────────────────────────────────────────────────────────────────── #
# TofuTrustStore — Trust-On-First-Use certificate pinning                       #
# ─────────────────────────────────────────────────────────────────────────── #


class TofuTrustStore:
    """
    Trust-On-First-Use (TOFU) certificate pinning.

    The first time a peer's certificate is seen it is pinned.  Subsequent
    connections from the same node_id MUST present the identical certificate,
    or they are rejected.  This eliminates the need for a central CA in a
    P2P mesh.
    """

    def __init__(self) -> None:
        self._pins: Dict[str, bytes] = {}  # node_id → SHA-256 of cert_pem
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._pins)

    def add(self, node_id: str, cert_pem: str) -> None:
        """Pin *cert_pem* for *node_id* (TOFU: first seen wins)."""
        digest = _sha256_pem(cert_pem)
        with self._lock:
            if node_id not in self._pins:
                self._pins[node_id] = digest

    def verify(self, node_id: str, cert_pem: str) -> bool:
        """Return True if *cert_pem* matches the pinned certificate for *node_id*."""
        digest = _sha256_pem(cert_pem)
        with self._lock:
            pinned = self._pins.get(node_id)
            return pinned is not None and pinned == digest

    def remove(self, node_id: str) -> None:
        """Forget the pinned certificate for *node_id*."""
        with self._lock:
            self._pins.pop(node_id, None)

    def serialize(self) -> bytes:
        """Serialise the trust store to bytes for persistence."""
        with self._lock:
            return json.dumps({
                k: v.hex() for k, v in self._pins.items()
            }).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "TofuTrustStore":
        """Restore a TofuTrustStore from serialised bytes."""
        payload = json.loads(data.decode("utf-8"))
        store = cls()
        with store._lock:
            store._pins = {k: bytes.fromhex(v) for k, v in payload.items()}
        return store


def _sha256_pem(pem: str) -> bytes:
    digest = hashes.Hash(hashes.SHA256())
    digest.update(pem.encode("utf-8"))
    return digest.finalize()


# ─────────────────────────────────────────────────────────────────────────── #
# gRPC credential helpers                                                        #
# ─────────────────────────────────────────────────────────────────────────── #


def load_server_credentials(
    cert_path: str,
    key_path: str,
    ca_cert_path: Optional[str] = None,
) -> grpc.ServerCredentials:
    """
    Load gRPC server credentials for mTLS.

    Parameters
    ----------
    cert_path:     Path to the server's X.509 certificate (PEM).
    key_path:      Path to the server's private key (PEM).
    ca_cert_path:  Path to the CA cert for verifying client certificates.
                   If None, server-only TLS is used (no client cert check).
    """
    with open(key_path, "rb") as f:
        private_key = f.read()
    with open(cert_path, "rb") as f:
        certificate_chain = f.read()

    if ca_cert_path:
        with open(ca_cert_path, "rb") as f:
            root_certificates = f.read()
        return grpc.ssl_server_credentials(
            [(private_key, certificate_chain)],
            root_certificates=root_certificates,
            require_client_auth=True,
        )
    else:
        return grpc.ssl_server_credentials(
            [(private_key, certificate_chain)],
            root_certificates=None,
            require_client_auth=False,
        )


def load_client_credentials(
    cert_path: str,
    key_path: str,
    ca_cert_path: str,
) -> grpc.ChannelCredentials:
    """
    Load gRPC client credentials for mTLS.

    Parameters
    ----------
    cert_path:     Path to the client's X.509 certificate (PEM).
    key_path:      Path to the client's private key (PEM).
    ca_cert_path:  Path to the CA cert for verifying the server's certificate.
    """
    with open(key_path, "rb") as f:
        private_key = f.read()
    with open(cert_path, "rb") as f:
        certificate_chain = f.read()
    with open(ca_cert_path, "rb") as f:
        root_certificates = f.read()

    return grpc.ssl_channel_credentials(
        root_certificates=root_certificates,
        private_key=private_key,
        certificate_chain=certificate_chain,
    )