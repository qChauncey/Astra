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
#   - Phase 4: AMD SEV-SNP confidential-computing TEE backend.
#   - Supports encrypted VM memory and SEV-SNP attestation.
#   - Provides the same TEEBackend interface as Gramine/SGX so
#     Astra inference code works on either platform.

"""
AMD SEV-SNP TEE backend for confidential computing.

AMD SEV (Secure Encrypted Virtualization) encrypts VM memory
with per-VM keys held in the AMD Secure Processor.  SEV-SNP
introduces Reverse Map Table (RMP) integrity protection and
an attestation protocol.

This backend monitors:
  - /dev/sev (SEV core device)
  - sev-guest driver for SEV-SNP attestation
  - KVM with SEV-SNP enabled (kernel ≥ 5.19)

Hardware requirements:
  - AMD EPYC 7002 "Rome" or newer (7003 "Milan"/9004 "Genoa")
  - SEV-SNP enabled in BIOS
  - Linux kernel with SEV-SNP host/guest support
  - QEMU/KVM with sev-snp-guest object

Phase 4 delivers the interface skeleton.  See docs/TEE.md
for full deployment instructions.

Usage::

    from astra.tee import register_backend
    from astra.tee.amd_sev import SevBackend

    sev = SevBackend()
    register_backend("sev", sev)

    if sev.status() == TEEStatus.AVAILABLE:
        report = sev.attest()
        sealed = sev.seal(model_weights)
"""

from __future__ import annotations

import hashlib
import os
import platform
import time
from dataclasses import dataclass
from typing import Optional
import logging

from . import (
    AttestationReport,
    SealedBlob,
    TEEBackend,
    TEEStatus,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# SEV detection                                                         #
# ------------------------------------------------------------------ #

@dataclass
class SevPlatformInfo:
    """SEV platform capabilities discovered at boot."""
    sev_supported: bool = False
    sev_es_supported: bool = False
    sev_snp_supported: bool = False
    max_guests: int = 0
    min_asid: int = 0
    cbit_position: int = 0


def _detect_sev_platform() -> SevPlatformInfo:
    """Read SEV capabilities from /sys/module/kvm_amd/parameters/sev
    and /dev/sev on Linux.  Returns empty struct on non-Linux.
    """
    info = SevPlatformInfo()

    if platform.system() != "Linux":
        return info

    if os.path.exists("/sys/module/kvm_amd/parameters/sev"):
        try:
            with open("/sys/module/kvm_amd/parameters/sev") as f:
                val = f.read().strip()
                info.sev_supported = val in ("1", "Y")
        except Exception:
            pass

    if os.path.exists("/sys/module/kvm_amd/parameters/sev_es"):
        try:
            with open("/sys/module/kvm_amd/parameters/sev_es") as f:
                val = f.read().strip()
                info.sev_es_supported = val in ("1", "Y")
        except Exception:
            pass

    if os.path.exists("/sys/module/kvm_amd/parameters/sev_snp"):
        try:
            with open("/sys/module/kvm_amd/parameters/sev_snp") as f:
                val = f.read().strip()
                info.sev_snp_supported = val in ("1", "Y")
        except Exception:
            pass

    if os.path.exists("/dev/sev"):
        try:
            with open("/dev/sev", "rb") as f:
                raw = f.read(64)
                if len(raw) >= 4:
                    # Parse SEV platform status packet (simplified)
                    pass
        except Exception:
            logger.debug("Could not read /dev/sev", exc_info=True)

    return info


# ------------------------------------------------------------------ #
# SevBackend                                                            #
# ------------------------------------------------------------------ #

class SevBackend(TEEBackend):
    """AMD SEV-SNP TEE backend.

    Implements the TEEBackend interface for AMD SEV-based
    confidential VMs.  When running on SEV-capable hardware,
    provides attestation and memory encryption.  On unsupported
    machines, reports TEEStatus.UNAVAILABLE.

    The SEV measurement (LAUNCH_MEASURE on SEV, REPORT_DATA on
    SEV-SNP) serves the same role as MRENCLAVE in SGX.
    """

    def __init__(self) -> None:
        self._status: Optional[TEEStatus] = None
        self._platform_info: Optional[SevPlatformInfo] = None
        self._measurement: str = ""
        self._is_guest = self._check_inside_sev_vm()

    # ------------------------------------------------------------------ #
    # Detection                                                              #
    # ------------------------------------------------------------------ #

    def _check_inside_sev_vm(self) -> bool:
        """Determine if we are running *inside* an SEV-protected guest VM.

        Checks for:
          - /sys/devices/system/cpu/caps/sev (SEV CPUID bit exposed)
          - DMI product name containing "SEV"
        """
        if os.path.exists("/sys/devices/system/cpu/caps/sev"):
            return True
        try:
            with open("/sys/class/dmi/id/product_name") as f:
                if "sev" in f.read().lower():
                    return True
        except Exception:
            pass
        return False

    def _detect_sev(self) -> TEEStatus:
        """Detect SEV availability.

        If we are inside a SEV VM → AVAILABLE.
        If we are on bare metal with SEV-SNP KVM support → NOT_CONFIGURED
          (need to launch a SEV-protected VM first).
        Otherwise → UNAVAILABLE.
        """
        self._platform_info = _detect_sev_platform()

        if self._is_guest:
            self._measurement = self._compute_dummy_measurement()
            return TEEStatus.AVAILABLE

        if self._platform_info.sev_snp_supported:
            # Bare-metal host with SEV-SNP KVM — needs VM launch
            self._measurement = self._compute_dummy_measurement()
            return TEEStatus.NOT_CONFIGURED

        if self._platform_info.sev_supported:
            self._measurement = self._compute_dummy_measurement()
            return TEEStatus.NOT_CONFIGURED

        self._measurement = self._compute_dummy_measurement()
        return TEEStatus.UNAVAILABLE

    def _compute_dummy_measurement(self) -> str:
        """Placeholder SEV measurement.

        In production, this comes from the AMD PSP-secured launch
        measurement (SEV-SNP REPORT).  For the Phase 4 skeleton,
        we hash a well-known string to provide a deterministic
        placeholder that downstream code can validate against.
        """
        data = b"astra-sev-snp-measurement-v1"
        return hashlib.sha384(data).hexdigest()[:96]

    # ------------------------------------------------------------------ #
    # TEEBackend interface                                                   #
    # ------------------------------------------------------------------ #

    def status(self) -> TEEStatus:
        if self._status is None:
            self._status = self._detect_sev()
        return self._status

    def attest(self) -> AttestationReport:
        """Generate an SEV-SNP attestation report.

        Production: calls the SEV-SNP guest driver
        (``/dev/sev-guest`` ioctl ``SNP_GET_REPORT``) to retrieve a
        VCEK-signed attestation report that includes the launch
        measurement, platform version, and ID block.

        Phase 4: returns a synthetic placeholder report.
        """
        status = self.status()
        is_valid = status == TEEStatus.AVAILABLE

        report = AttestationReport(
            tee_type="sev",
            is_valid=is_valid,
            measurement=self._measurement or "0000" * 24,
            timestamp=time.time(),
            details={
                "hardware": status.value,
                "sev_type": "SNP" if (
                    self._platform_info and self._platform_info.sev_snp_supported
                ) else "SEV",
                "is_guest": self._is_guest,
                "platform": {
                    "sev": self._platform_info.sev_supported if self._platform_info else False,
                    "sev_es": self._platform_info.sev_es_supported if self._platform_info else False,
                    "sev_snp": self._platform_info.sev_snp_supported if self._platform_info else False,
                },
            },
        )

        if is_valid:
            quote = self.get_quote()
            report.details["quote_size"] = len(quote)
            report.details["quote_hex"] = quote.hex()[:64] + "..."

        return report

    def seal(self, plaintext: bytes) -> SealedBlob:
        """Seal data to this SEV VM.

        SEV-NP: uses AES-GCM with a key derived from the VM's
        unique launch measurement, similar to SGX sealing.

        Phase 4: placeholder AES-GCM with deterministic mock key.
        """
        import secrets
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        key_material = hashlib.sha384(
            self._measurement.encode()
        ).digest()
        nonce = secrets.token_bytes(12)

        try:
            aesgcm = AESGCM(key_material[:32])
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        except ImportError:
            import base64
            ciphertext = base64.b64encode(plaintext)

        return SealedBlob(
            ciphertext=nonce + ciphertext,
            tee_type="sev",
            api_version="1.0",
        )

    def unseal(self, blob: SealedBlob) -> bytes:
        """Decrypt a sealed blob inside the SEV VM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = blob.ciphertext[:12]
        ct = blob.ciphertext[12:]

        key_material = hashlib.sha384(
            self._measurement.encode()
        ).digest()
        aesgcm = AESGCM(key_material[:32])
        return aesgcm.decrypt(nonce, ct, None)

    def get_quote(self) -> bytes:
        """Retrieve a raw SEV-SNP attestation report.

        Production: reads from ``/dev/sev-guest`` using the
        ``SNP_GET_REPORT`` ioctl.  The report is signed by the
        Versioned Chip Endorsement Key (VCEK).

        Phase 4: returns a zero-filled placeholder.
        """
        if not self._is_guest:
            logger.warning("Not inside a SEV VM — returning placeholder quote")
            return b"\\x00" * 1184  # SEV-SNP REPORT size

        try:
            with open("/dev/sev-guest", "rb") as f:
                # In production: issue SNP_GET_REPORT ioctl
                return f.read(1184)
        except (FileNotFoundError, PermissionError):
            logger.debug("/dev/sev-guest not accessible", exc_info=True)
            return b"\\x00" * 1184

    # ------------------------------------------------------------------ #
    # Helpers                                                                #
    # ------------------------------------------------------------------ #

    @property
    def is_guest(self) -> bool:
        return self._is_guest

    @property
    def platform_info(self) -> Optional[SevPlatformInfo]:
        if self._platform_info is None:
            self._detect_sev()
        return self._platform_info

    @property
    def measurement(self) -> str:
        """SEV launch measurement (LAUNCH_MEASURE or REPORT_DATA)."""
        if not self._measurement:
            self._detect_sev()
        return self._measurement
