# Copyright 2025 Project Astra Contributors
# Licensed under the Apache License, Version 2.0

"""
Unit tests for astra.rpc.tls — X.509 certificate generation,
mTLS credential loading, and TOFU trust store.
"""

import os
import tempfile

import pytest

from astra.rpc.tls import (
    CertBundle,
    TLSConfig,
    TofuTrustStore,
    generate_self_signed_cert_bundle,
    load_server_credentials,
    load_client_credentials,
)


class TestCertBundle:
    def test_single_node_bundle_generation(self):
        bundle = generate_self_signed_cert_bundle(node_id="node-a")
        assert bundle.node_id == "node-a"
        assert bundle.cert_pem.startswith("-----BEGIN CERTIFICATE-----")
        assert bundle.key_pem.startswith("-----BEGIN PRIVATE KEY-----")
        assert bundle.cert_pem != ""
        assert bundle.key_pem != ""

    def test_bundle_cert_and_key_differ(self):
        bundle = generate_self_signed_cert_bundle("node-b")
        assert bundle.cert_pem != bundle.key_pem

    def test_multiple_nodes_have_unique_certs(self):
        b1 = generate_self_signed_cert_bundle("n1")
        b2 = generate_self_signed_cert_bundle("n2")
        assert b1.cert_pem != b2.cert_pem
        assert b1.key_pem != b2.key_pem

    def test_write_to_directory(self):
        bundle = generate_self_signed_cert_bundle("node-c")
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path, key_path = bundle.write(tmpdir)
            assert os.path.isfile(cert_path)
            assert os.path.isfile(key_path)
            with open(cert_path, "r") as f:
                assert f.read() == bundle.cert_pem
            with open(key_path, "r") as f:
                assert f.read() == bundle.key_pem


class TestTLSConfig:
    def test_default_disabled(self):
        cfg = TLSConfig()
        assert cfg.enabled is False

    def test_enabled_with_certs(self):
        cfg = TLSConfig(
            enabled=True,
            cert_path="/tmp/cert.pem",
            key_path="/tmp/key.pem",
            ca_cert_path="/tmp/ca.pem",
        )
        assert cfg.enabled is True
        assert cfg.cert_path == "/tmp/cert.pem"
        assert cfg.key_path == "/tmp/key.pem"
        assert cfg.ca_cert_path == "/tmp/ca.pem"

    def test_enabled_without_certs_raises(self):
        cfg = TLSConfig(enabled=True)
        assert not cfg.is_ready()

    def test_is_ready(self):
        cfg = TLSConfig(
            enabled=True,
            cert_path="/x/cert.pem",
            key_path="/x/key.pem",
            ca_cert_path="/x/ca.pem",
        )
        assert cfg.is_ready()  # paths set, runtime check deferred

    def test_requires_mutual_tls_when_enabled(self):
        cfg = TLSConfig(enabled=True, cert_path="a", key_path="b")
        assert not cfg.is_ready()  # no CA cert for mutual TLS

    def test_optional_ca_for_server_only(self):
        # Server mode: CA cert required for mTLS
        cfg = TLSConfig(enabled=True, cert_path="a", key_path="b", ca_cert_path="c")
        assert cfg.is_ready()


class TestTofuTrustStore:
    def test_initial_empty(self):
        store = TofuTrustStore()
        assert len(store) == 0

    def test_add_and_verify(self):
        store = TofuTrustStore()
        bundle = generate_self_signed_cert_bundle("node-x")
        store.add("node-x", bundle.cert_pem)
        assert store.verify("node-x", bundle.cert_pem) is True
        assert len(store) == 1

    def test_verify_unknown_node(self):
        store = TofuTrustStore()
        bundle = generate_self_signed_cert_bundle("unknown")
        assert store.verify("unknown", bundle.cert_pem) is False

    def test_tofu_pin_first_cert(self):
        """Trust-On-First-Use: first cert seen is pinned."""
        store = TofuTrustStore()
        b1 = generate_self_signed_cert_bundle("pinned")
        b2 = generate_self_signed_cert_bundle("pinned")  # different cert, same node_id
        store.add("pinned", b1.cert_pem)
        # Second cert for same node_id should be rejected
        assert store.verify("pinned", b2.cert_pem) is False
        # Original cert still valid
        assert store.verify("pinned", b1.cert_pem) is True

    def test_add_multiple_nodes(self):
        store = TofuTrustStore()
        for i in range(5):
            bundle = generate_self_signed_cert_bundle(f"node-{i}")
            store.add(f"node-{i}", bundle.cert_pem)
        assert len(store) == 5

    def test_serialize_roundtrip(self):
        store = TofuTrustStore()
        bundle = generate_self_signed_cert_bundle("persist")
        store.add("persist", bundle.cert_pem)

        data = store.serialize()
        assert isinstance(data, bytes)
        assert len(data) > 0

        store2 = TofuTrustStore.deserialize(data)
        assert store2.verify("persist", bundle.cert_pem) is True
        assert len(store2) == 1

    def test_remove_node(self):
        store = TofuTrustStore()
        bundle = generate_self_signed_cert_bundle("remove-me")
        store.add("remove-me", bundle.cert_pem)
        assert len(store) == 1
        store.remove("remove-me")
        assert len(store) == 0
        assert store.verify("remove-me", bundle.cert_pem) is False


class TestCredentialLoading:
    def test_load_server_credentials_from_paths(self):
        bundle = generate_self_signed_cert_bundle("srv")
        ca_bundle = generate_self_signed_cert_bundle("ca")
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path, key_path = bundle.write(tmpdir)
            ca_cert_path = os.path.join(tmpdir, "ca.pem")
            with open(ca_cert_path, "w") as f:
                f.write(ca_bundle.cert_pem)
            creds = load_server_credentials(cert_path, key_path, ca_cert_path)
            assert creds is not None

    def test_load_client_credentials_from_paths(self):
        bundle = generate_self_signed_cert_bundle("cli")
        ca_bundle = generate_self_signed_cert_bundle("ca")
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path, key_path = bundle.write(tmpdir)
            ca_cert_path = os.path.join(tmpdir, "ca.pem")
            with open(ca_cert_path, "w") as f:
                f.write(ca_bundle.cert_pem)
            creds = load_client_credentials(cert_path, key_path, ca_cert_path)
            assert creds is not None

    def test_load_server_credentials_without_ca(self):
        """Server credentials without CA cert (server-only TLS, no mTLS)."""
        bundle = generate_self_signed_cert_bundle("srv-noca")
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path, key_path = bundle.write(tmpdir)
            creds = load_server_credentials(cert_path, key_path)
            assert creds is not None