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

"""Tests for astra.network.identity — Ed25519 peer identity."""

from __future__ import annotations

import pathlib

import pytest

# Skip the whole module if cryptography isn't installed.
pytest.importorskip("cryptography")

from astra.network.identity import (
    PeerIdentity,
    SignedPayload,
    TrustRegistry,
    verify_signed_payload,
)


# ── PeerIdentity.generate ────────────────────────────────────────────────────

class TestGenerate:
    def test_returns_identity(self):
        pid = PeerIdentity.generate()
        assert isinstance(pid, PeerIdentity)

    def test_two_generates_differ(self):
        a = PeerIdentity.generate()
        b = PeerIdentity.generate()
        assert a.public_key_b64() != b.public_key_b64()

    def test_node_id_derived_from_pubkey(self):
        pid = PeerIdentity.generate()
        nid = pid.node_id()
        assert isinstance(nid, str)
        assert len(nid) == 16
        assert pid.public_key_b64().startswith(nid)


# ── load / save / load_or_create ─────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load(self, tmp_path: pathlib.Path):
        path = tmp_path / "id.pem"
        pid1 = PeerIdentity.generate()
        pid1.save(path)
        pid2 = PeerIdentity.load(path)
        assert pid1.public_key_b64() == pid2.public_key_b64()

    def test_load_or_create_creates_when_missing(self, tmp_path: pathlib.Path):
        path = tmp_path / "missing.pem"
        assert not path.exists()
        pid = PeerIdentity.load_or_create(path)
        assert path.exists()
        assert pid.public_key_b64()

    def test_load_or_create_reuses_existing(self, tmp_path: pathlib.Path):
        path = tmp_path / "id.pem"
        pid1 = PeerIdentity.load_or_create(path)
        pid2 = PeerIdentity.load_or_create(path)
        assert pid1.public_key_b64() == pid2.public_key_b64()

    def test_load_rejects_non_ed25519(self, tmp_path: pathlib.Path):
        path = tmp_path / "garbage.pem"
        path.write_bytes(b"-----BEGIN PRIVATE KEY-----\nnot a real key\n")
        with pytest.raises(Exception):
            PeerIdentity.load(path)


# ── sign_payload / verify_signed_payload ─────────────────────────────────────

class TestSignVerify:
    def test_sign_then_verify_succeeds(self):
        pid = PeerIdentity.generate()
        payload = {"address": "10.0.0.1:50051", "layers": "0:30"}
        signed = pid.sign_payload(payload)
        assert verify_signed_payload(signed) is True

    def test_tampered_payload_fails_verification(self):
        pid = PeerIdentity.generate()
        signed = pid.sign_payload({"address": "10.0.0.1:50051"})
        signed.payload["address"] = "10.0.0.99:50051"   # tamper
        assert verify_signed_payload(signed) is False

    def test_wrong_signature_fails_verification(self):
        pid = PeerIdentity.generate()
        signed = pid.sign_payload({"x": 1})
        # Replace signature with another peer's signature for a different payload
        other = PeerIdentity.generate()
        bad_sig = other.sign_payload({"y": 2}).signature_b64
        signed.signature_b64 = bad_sig
        assert verify_signed_payload(signed) is False

    def test_signed_at_field_set(self):
        pid = PeerIdentity.generate()
        signed = pid.sign_payload({"a": 1})
        assert signed.signed_at > 0


# ── SignedPayload to_dict / from_dict ────────────────────────────────────────

class TestSignedPayloadSerialization:
    def test_roundtrip(self):
        pid = PeerIdentity.generate()
        signed = pid.sign_payload({"hello": "world"})
        d = signed.to_dict()
        signed2 = SignedPayload.from_dict(d)
        assert signed2.public_key_b64 == signed.public_key_b64
        assert signed2.signature_b64 == signed.signature_b64
        assert signed2.payload == signed.payload
        assert verify_signed_payload(signed2) is True


# ── TrustRegistry ────────────────────────────────────────────────────────────

class TestTrustRegistry:
    def test_first_vouch_succeeds(self):
        r = TrustRegistry()
        assert r.vouch("node-A", "key123") is True

    def test_repeat_vouch_same_key_succeeds(self):
        r = TrustRegistry()
        r.vouch("node-A", "key123")
        assert r.vouch("node-A", "key123") is True

    def test_different_key_for_known_node_rejected(self):
        r = TrustRegistry()
        r.vouch("node-A", "key123")
        assert r.vouch("node-A", "evil456") is False

    def test_known_key(self):
        r = TrustRegistry()
        r.vouch("node-A", "k1")
        assert r.known_key("node-A") == "k1"
        assert r.known_key("node-B") is None

    def test_forget_clears_record(self):
        r = TrustRegistry()
        r.vouch("node-A", "k1")
        r.forget("node-A")
        # now a different key for node-A is fine
        assert r.vouch("node-A", "k2") is True
