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
#   - Phase 4: Trusted Execution Environment abstraction layer.
#   - Provides hardware-backed isolation for inference computation
#     (Intel SGX via Gramine, AMD SEV-SNP).
#   - Abstract interface allows future plug-in of new TEE backends.

"""
Astra TEE (Trusted Execution Environment) abstraction layer.

This package defines the common interface for hardware-isolated inference:
  - `TEEBackend` — abstract base class for SGX / SEV / future backends
  - `gramine`    — Intel SGX support via the Gramine Library OS
  - `amd_sev`    — AMD SEV-SNP confidential-computing stub

Phase 4 delivers the interface skeleton and documentation.
Production deployment requires compatible hardware:
  - Intel SGX: 6th-gen Core or newer + SGX enabled in BIOS
  - AMD SEV:   EPYC 7002 "Rome" or newer server CPU

Usage::

    from astra.tee import get_tee_backend

    tee = get_tee_backend("sgx")
    if tee.is_available():
        attestation = tee.attest()
        sealed_model = tee.seal(model_bytes)

    from astra.tee.gramine import GramineBackend
    from astra.tee.amd_sev import SevBackend
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


# ------------------------------------------------------------------ #
# Types                                                                  #
# ------------------------------------------------------------------ #

class TEEStatus(Enum):
    """TEE availability status."""
    AVAILABLE = "available"         # Hardware present and configured
    UNAVAILABLE = "unavailable"     # No compatible hardware
    NOT_CONFIGURED = "not_configured"  # Hardware present but not enabled
    UNKNOWN = "unknown"             # Detection not yet run


@dataclass
class AttestationReport:
    """Result of a TEE attestation (remote verification)."""
    tee_type: str                   # "sgx" or "sev"
    is_valid: bool
    measurement: str                # MRENCLAVE (SGX) or MEASUREMENT (SEV)
    timestamp: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SealedBlob:
    """Sealed (encrypted-to-TEE) payload."""
    ciphertext: bytes
    tee_type: str
    api_version: str = "1.0"


# ------------------------------------------------------------------ #
# Abstract backend                                                        #
# ------------------------------------------------------------------ #

class TEEBackend(ABC):
    """Abstract interface for a Trusted Execution Environment.

    Each concrete backend (SGX, SEV, etc.) implements these primitives.
    The rest of Astra calls through this interface, so the inference
    engine need not know which TEE technology is in use.
    """

    @abstractmethod
    def status(self) -> TEEStatus:
        """Check if the TEE is available on this machine."""
        ...

    @abstractmethod
    def attest(self) -> AttestationReport:
        """Perform remote attestation and return a verifiable report.

        The report contains a cryptographic measurement that a remote
        party can validate against a known-good hash.  This proves the
        code running inside the TEE has not been tampered with.
        """
        ...

    @abstractmethod
    def seal(self, plaintext: bytes) -> SealedBlob:
        """Encrypt data so only *this* TEE instance can decrypt it.

        Sealing binds the ciphertext to the enclave identity (SGX) or
        the guest VM measurement (SEV), preventing offline decryption.
        """
        ...

    @abstractmethod
    def unseal(self, blob: SealedBlob) -> bytes:
        """Decrypt a previously sealed blob inside the same TEE."""
        ...

    @abstractmethod
    def get_quote(self) -> bytes:
        """Return a raw attestation quote for verification by a
        third-party service (e.g. Intel IAS, AMD KDS)."""
        ...


# ------------------------------------------------------------------ #
# Backend registry                                                        #
# ------------------------------------------------------------------ #

_BACKENDS: Dict[str, TEEBackend] = {}


def register_backend(name: str, backend: TEEBackend) -> None:
    """Register a TEE backend for lookup."""
    _BACKENDS[name] = backend


def get_tee_backend(name: str = "auto") -> Optional[TEEBackend]:
    """Return a TEE backend by name, or auto-detect if 'auto'.

    Currently supported: "sgx", "sev", "none".
    """
    if name == "none":
        return None
    if name == "auto":
        # Try SGX first, then SEV
        for candidate in ("sgx", "sev"):
            try:
                backend = _BACKENDS.get(candidate)
                if backend and backend.status() == TEEStatus.AVAILABLE:
                    return backend
            except Exception:
                continue
        return None
    return _BACKENDS.get(name)


def list_available_backends() -> Dict[str, TEEStatus]:
    """Return all registered backends and their availability."""
    result: Dict[str, TEEStatus] = {}
    for name, backend in _BACKENDS.items():
        try:
            result[name] = backend.status()
        except Exception:
            result[name] = TEEStatus.UNKNOWN
    return result


# ------------------------------------------------------------------ #
# Auto-registration on import                                            #
# ------------------------------------------------------------------ #

def _auto_register_backends() -> None:
    """Automatically register built-in TEE backends on import.

    This is called at module import time so that ``get_tee_backend()``
    works out of the box without manual registration.  Each backend's
    ``status()`` call is lazy — it won't touch hardware until queried.
    """
    try:
        from .gramine import GramineBackend
        if "sgx" not in _BACKENDS:
            register_backend("sgx", GramineBackend())
    except Exception:
        pass

    try:
        from .amd_sev import SevBackend
        if "sev" not in _BACKENDS:
            register_backend("sev", SevBackend())
    except Exception:
        pass


_auto_register_backends()
