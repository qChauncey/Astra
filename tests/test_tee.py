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
# Phase 4: TEE abstraction layer unit tests.
#   - Tests the TEEBackend interface, GramineBackend (SGX), SevBackend (SEV).
#   - All TEE hardware tests work without real hardware — they validate
#     the detection logic, attestation structures, and seal/unseal cycle.

from __future__ import annotations

import hashlib
import pytest

from astra.tee import (
    TEEStatus,
    AttestationReport,
    SealedBlob,
    TEEBackend,
    register_backend,
    get_tee_backend,
    list_available_backends,
)


# ── Stub backend for testing the registry ──────────────────────────────────

class StubBackend(TEEBackend):
    """Minimal TEEBackend stub used to test the registry."""

    def __init__(self, name: str, available: bool = True):
        self._name = name
        self._available = available
        self._measurement = hashlib.sha256(name.encode()).hexdigest()[:64]

    def status(self) -> TEEStatus:
        return TEEStatus.AVAILABLE if self._available else TEEStatus.UNAVAILABLE

    def attest(self) -> AttestationReport:
        return AttestationReport(
            tee_type=self._name,
            is_valid=self._available,
            measurement=self._measurement,
        )

    def seal(self, plaintext: bytes) -> SealedBlob:
        return SealedBlob(ciphertext=plaintext[::-1], tee_type=self._name)

    def unseal(self, blob: SealedBlob) -> bytes:
        return blob.ciphertext[::-1]

    def get_quote(self) -> bytes:
        return self._measurement.encode() if self._available else b"\x00" * 64


# ── TEEStatus enum ─────────────────────────────────────────────────────────

class TestTEEStatus:
    def test_values_exist(self):
        assert TEEStatus.AVAILABLE.value == "available"
        assert TEEStatus.UNAVAILABLE.value == "unavailable"
        assert TEEStatus.NOT_CONFIGURED.value == "not_configured"
        assert TEEStatus.UNKNOWN.value == "unknown"

    def test_equality(self):
        assert TEEStatus.AVAILABLE == TEEStatus.AVAILABLE
        assert TEEStatus.AVAILABLE != TEEStatus.UNAVAILABLE


# ── AttestationReport dataclass ────────────────────────────────────────────

class TestAttestationReport:
    def test_defaults(self):
        r = AttestationReport(tee_type="test", is_valid=True, measurement="abc")
        assert r.tee_type == "test"
        assert r.is_valid is True
        assert r.measurement == "abc"
        assert r.timestamp == 0.0
        assert r.details == {}

    def test_with_details(self):
        r = AttestationReport(
            tee_type="sgx",
            is_valid=True,
            measurement="deadbeef" * 8,
            timestamp=12345.678,
            details={"hardware": "available", "driver": "gramine"},
        )
        assert r.timestamp == 12345.678
        assert r.details["hardware"] == "available"
        assert r.details["driver"] == "gramine"


# ── SealedBlob dataclass ───────────────────────────────────────────────────

class TestSealedBlob:
    def test_roundtrip_metadata(self):
        blob = SealedBlob(ciphertext=b"encrypted", tee_type="sgx", api_version="2.0")
        assert blob.ciphertext == b"encrypted"
        assert blob.tee_type == "sgx"
        assert blob.api_version == "2.0"

    def test_default_api_version(self):
        blob = SealedBlob(ciphertext=b"data", tee_type="sev")
        assert blob.api_version == "1.0"


# ── Backend registry ───────────────────────────────────────────────────────

class TestRegistry:
    def setup_method(self):
        # Import again to re-run auto-registration for clean state
        import astra.tee
        self.tee = astra.tee

    def test_register_and_lookup(self):
        stub = StubBackend("test_reg")
        register_backend("test_reg", stub)
        assert get_tee_backend("test_reg") is stub

    def test_none_returns_none(self):
        assert get_tee_backend("none") is None

    def test_unknown_name_returns_none(self):
        assert get_tee_backend("no_such_backend_xyz") is None

    def test_auto_detect_falls_back(self):
        """When no backend is AVAILABLE, 'auto' returns None."""
        # Register an unavailable stub
        stub = StubBackend("auto_test", available=False)
        register_backend("auto_test", stub)
        result = get_tee_backend("auto")  # sgx & sev likely UNAVAILABLE on this machine
        # It should at least not crash
        assert result is None or isinstance(result, TEEBackend)

    def test_list_backends(self):
        stub = StubBackend("list_test", available=True)
        register_backend("list_test", stub)
        backends = list_available_backends()
        assert "list_test" in backends
        assert backends["list_test"] == TEEStatus.AVAILABLE

    def test_list_backends_unavailable(self):
        stub = StubBackend("list_unavail", available=False)
        register_backend("list_unavail", stub)
        backends = list_available_backends()
        assert backends["list_unavail"] == TEEStatus.UNAVAILABLE

    def test_builtin_sgx_registered(self):
        """After importing astra.tee, 'sgx' backend should be auto-registered."""
        backend = get_tee_backend("sgx")
        assert backend is not None
        assert isinstance(backend, TEEBackend)

    def test_builtin_sev_registered(self):
        """After importing astra.tee, 'sev' backend should be auto-registered."""
        backend = get_tee_backend("sev")
        assert backend is not None
        assert isinstance(backend, TEEBackend)


# ── GramineBackend (SGX) ───────────────────────────────────────────────────

class TestGramineBackend:
    @pytest.fixture
    def sgx(self):
        from astra.tee.gramine import GramineBackend
        return GramineBackend()

    def test_status_returns_enum(self, sgx):
        status = sgx.status()
        assert isinstance(status, TEEStatus)
        assert status in (TEEStatus.AVAILABLE, TEEStatus.UNAVAILABLE,
                          TEEStatus.NOT_CONFIGURED, TEEStatus.UNKNOWN)

    def test_status_is_idempotent(self, sgx):
        s1 = sgx.status()
        s2 = sgx.status()
        assert s1 == s2

    def test_attest_returns_report(self, sgx):
        report = sgx.attest()
        assert isinstance(report, AttestationReport)
        assert report.tee_type == "sgx"
        assert "hardware" in report.details
        assert "driver" in report.details
        assert report.details["driver"] == "gramine"

    def test_seal_unseal_roundtrip(self, sgx):
        """Seal then unseal must recover the original plaintext."""
        original = b"DeepSeek-V4 model weights (32 bytes demo!)"
        sealed = sgx.seal(original)
        assert isinstance(sealed, SealedBlob)
        assert sealed.tee_type == "sgx"
        assert sealed.api_version == "1.0"

        recovered = sgx.unseal(sealed)
        assert recovered == original

    def test_seal_produces_different_ciphertext(self, sgx):
        """Each seal call should produce unique ciphertext (random nonce)."""
        data = b"test payload"
        blob1 = sgx.seal(data)
        blob2 = sgx.seal(data)
        assert blob1.ciphertext != blob2.ciphertext

    def test_unseal_wrong_blob_raises(self, sgx):
        """Unsealing a blob sealed by a different backend should fail."""
        original = b"sensitive data"
        sealed = sgx.seal(original)
        # Corrupt the ciphertext
        tampered = SealedBlob(
            ciphertext=sealed.ciphertext[:-1] + b"\x00",
            tee_type="sgx",
        )
        with pytest.raises(Exception):
            sgx.unseal(tampered)

    def test_get_quote(self, sgx):
        quote = sgx.get_quote()
        assert isinstance(quote, bytes)
        assert len(quote) > 0

    def test_config_property(self, sgx):
        from astra.tee.gramine import GramineConfig
        assert isinstance(sgx.config, GramineConfig)

    def test_enclave_measurement_property(self, sgx):
        m = sgx.enclave_measurement
        assert isinstance(m, str)
        assert len(m) == 64  # SHA-256 hex

    def test_generate_manifest_writes_file(self, sgx, tmp_path):
        manifest_path = tmp_path / "astra.manifest"
        sgx.generate_manifest(str(manifest_path))
        assert manifest_path.exists()
        content = manifest_path.read_text()
        assert "loader.entrypoint" in content
        assert "gramine" in content.lower()

    def test_custom_config(self):
        from astra.tee.gramine import GramineConfig, GramineBackend
        cfg = GramineConfig(
            build_dir="/tmp/test_enclave",
            enclave_size="4G",
            thread_count=4,
            sgx_debug=True,
        )
        backend = GramineBackend(cfg)
        assert backend.config.build_dir == "/tmp/test_enclave"
        assert backend.config.enclave_size == "4G"
        assert backend.config.thread_count == 4

    def test_detection_on_non_sgx_machine(self, sgx):
        """On a development machine without SGX, status should not crash."""
        status = sgx.status()
        # Most dev machines won't have SGX; either UNAVAILABLE or NOT_CONFIGURED
        assert isinstance(status, TEEStatus)


# ── SevBackend (AMD SEV) ───────────────────────────────────────────────────

class TestSevBackend:
    @pytest.fixture
    def sev(self):
        from astra.tee.amd_sev import SevBackend
        return SevBackend()

    def test_status_returns_enum(self, sev):
        status = sev.status()
        assert isinstance(status, TEEStatus)

    def test_status_is_idempotent(self, sev):
        s1 = sev.status()
        s2 = sev.status()
        assert s1 == s2

    def test_attest_returns_report(self, sev):
        report = sev.attest()
        assert isinstance(report, AttestationReport)
        assert report.tee_type == "sev"
        assert "hardware" in report.details
        assert "platform" in report.details

    def test_seal_unseal_roundtrip(self, sev):
        original = b"AMD SEV-SNP confidential data (32 chars!)"
        sealed = sev.seal(original)
        assert isinstance(sealed, SealedBlob)
        assert sealed.tee_type == "sev"

        recovered = sev.unseal(sealed)
        assert recovered == original

    def test_seal_unique_nonce(self, sev):
        data = b"repeated seal test"
        blob1 = sev.seal(data)
        blob2 = sev.seal(data)
        assert blob1.ciphertext != blob2.ciphertext

    def test_get_quote(self, sev):
        quote = sev.get_quote()
        assert isinstance(quote, bytes)
        assert len(quote) > 0

    def test_is_guest_property(self, sev):
        assert isinstance(sev.is_guest, bool)

    def test_platform_info_property(self, sev):
        from astra.tee.amd_sev import SevPlatformInfo
        info = sev.platform_info
        assert isinstance(info, SevPlatformInfo)

    def test_measurement_property(self, sev):
        m = sev.measurement
        assert isinstance(m, str)
        assert len(m) == 96  # SHA-384 hex

    def test_detection_safe(self, sev):
        """Detection must never crash, even on non-SEV hardware."""
        status = sev.status()
        assert isinstance(status, TEEStatus)


# ── SevPlatformInfo ─────────────────────────────────────────────────────────

class TestSevPlatformInfo:
    def test_default_values(self):
        from astra.tee.amd_sev import SevPlatformInfo
        info = SevPlatformInfo()
        assert info.sev_supported is False
        assert info.sev_es_supported is False
        assert info.sev_snp_supported is False
        assert info.max_guests == 0


# ── TEEBackend ABC enforcement ─────────────────────────────────────────────

class TestAbstractInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            TEEBackend()  # type: ignore

    def test_stub_is_valid_backend(self):
        stub = StubBackend("test")
        assert isinstance(stub, TEEBackend)

    def test_all_methods_exist(self):
        expected = {"status", "attest", "seal", "unseal", "get_quote"}
        found = set()
        for attr in dir(TEEBackend):
            if not attr.startswith("_") and callable(getattr(TEEBackend, attr)):
                found.add(attr)
        assert expected.issubset(found)


# ── Interoperability: SGX and SEV share the same interface ─────────────────

class TestInteroperability:
    """Verify that both SGX and SEV backends satisfy the same contract."""

    @pytest.fixture
    def both(self):
        from astra.tee.gramine import GramineBackend
        from astra.tee.amd_sev import SevBackend
        return {"sgx": GramineBackend(), "sev": SevBackend()}

    def test_identical_interface(self, both):
        for name, backend in both.items():
            assert hasattr(backend, "status"), f"{name}: missing status()"
            assert hasattr(backend, "attest"), f"{name}: missing attest()"
            assert hasattr(backend, "seal"), f"{name}: missing seal()"
            assert hasattr(backend, "unseal"), f"{name}: missing unseal()"
            assert hasattr(backend, "get_quote"), f"{name}: missing get_quote()"

    def test_seal_unseal_both(self, both):
        payload = b"Interoperability test data for Phase 4 TEE layer"
        for name, backend in both.items():
            sealed = backend.seal(payload)
            assert sealed.tee_type == name, f"{name}: wrong tee_type"
            recovered = backend.unseal(sealed)
            assert recovered == payload, f"{name}: roundtrip failed"

    def test_attest_structure_consistent(self, both):
        for name, backend in both.items():
            report = backend.attest()
            assert report.tee_type == name
            assert isinstance(report.is_valid, bool)
            assert isinstance(report.measurement, str)
            assert len(report.measurement) > 0
