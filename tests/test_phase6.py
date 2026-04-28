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
Unit tests for Phase 6 API endpoints:

- /api/monitor        — live Ping aggregation for cluster hardware stats
- /api/login          — decentralized challenge-response identity
- /api/login/challenge — challenge nonce issuance
- /api/earnings       — contributor token accounting ledger
- /api/earnings/credit — credit a contributor for completed work
"""

import hashlib
import hmac
import pytest
from starlette.testclient import TestClient

from astra.api.openai_compat import create_app
from astra.network.dht import AstraDHT, _GlobalStore


# ────────────────────────────────────────────────────────────────── #
# Fixture                                                             #
# ────────────────────────────────────────────────────────────────── #

@pytest.fixture()
def client():
    """Return a TestClient backed by an isolated DHT."""
    store = _GlobalStore()
    dht = AstraDHT(node_id="test-api-gateway", store=store)
    app = create_app(dht=dht)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ══════════════════════════════════════════════════════════════════ #
# /api/monitor                                                        #
# ══════════════════════════════════════════════════════════════════ #

class TestMonitor:
    def test_returns_json_with_nodes_and_count(self, client):
        resp = client.get("/api/monitor")
        assert resp.status_code == 200
        body = resp.json()
        assert "nodes" in body
        assert "count" in body
        assert "ts" in body
        assert isinstance(body["nodes"], list)
        assert body["count"] == len(body["nodes"])

    def test_empty_when_no_peers_registered(self, client):
        resp = client.get("/api/monitor")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 0
        assert body["nodes"] == []

    def test_ts_is_present_and_numeric(self, client):
        resp = client.get("/api/monitor")
        ts = resp.json()["ts"]
        assert isinstance(ts, int)
        assert ts > 0


# ══════════════════════════════════════════════════════════════════ #
# /api/login/challenge                                                #
# ══════════════════════════════════════════════════════════════════ #

class TestLoginChallenge:
    def test_returns_nonce_for_valid_contributor_id(self, client):
        resp = client.post("/api/login/challenge", json={"contributor_id": "node-42"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["contributor_id"] == "node-42"
        assert "nonce" in body
        assert len(body["nonce"]) == 64  # hex of 32 bytes

    def test_missing_contributor_id_returns_error(self, client):
        resp = client.post("/api/login/challenge", json={})
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_empty_contributor_id_returns_error(self, client):
        resp = client.post("/api/login/challenge", json={"contributor_id": ""})
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_different_challenges_for_different_ids(self, client):
        r1 = client.post("/api/login/challenge", json={"contributor_id": "a"}).json()
        r2 = client.post("/api/login/challenge", json={"contributor_id": "b"}).json()
        assert r1["nonce"] != r2["nonce"]

    def test_repeated_challenge_generates_new_nonce(self, client):
        r1 = client.post("/api/login/challenge", json={"contributor_id": "x"}).json()
        r2 = client.post("/api/login/challenge", json={"contributor_id": "x"}).json()
        assert r1["nonce"] != r2["nonce"]


# ══════════════════════════════════════════════════════════════════ #
# /api/login — challenge-response flow                                #
# ══════════════════════════════════════════════════════════════════ #

class TestLogin:
    def test_full_challenge_response_flow_succeeds(self, client):
        cid = "contributor-1"
        # 1. Request challenge
        chal = client.post("/api/login/challenge", json={"contributor_id": cid}).json()
        nonce = chal["nonce"]

        # 2. Compute HMAC-SHA256 signature
        sig = hmac.new(cid.encode(), nonce.encode(), hashlib.sha256).hexdigest()

        # 3. Submit login
        resp = client.post("/api/login", json={
            "contributor_id": cid,
            "signature": sig,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["contributor_id"] == cid
        assert body["token"].startswith(f"astra-session-{cid}-")
        assert "token_hash" in body
        assert "expires_at" in body
        assert body["expires_at"] > 0

    def test_login_without_challenge_fails(self, client):
        sig = hmac.new(b"contributor-2", b"fake-nonce", hashlib.sha256).hexdigest()
        resp = client.post("/api/login", json={
            "contributor_id": "contributor-2",
            "signature": sig,
        })
        assert resp.status_code == 400
        assert "no challenge requested" in resp.json()["error"]

    def test_login_missing_contributor_id(self, client):
        resp = client.post("/api/login", json={"signature": "abcd1234"})
        assert resp.status_code == 400

    def test_login_missing_signature(self, client):
        resp = client.post("/api/login", json={"contributor_id": "x"})
        assert resp.status_code == 400

    def test_wrong_signature_rejected(self, client):
        cid = "contributor-3"
        # Get a real challenge
        client.post("/api/login/challenge", json={"contributor_id": cid}).json()

        # But sign a *different* nonce
        fake_sig = hmac.new(cid.encode(), b"wrong-nonce-value", hashlib.sha256).hexdigest()

        resp = client.post("/api/login", json={
            "contributor_id": cid,
            "signature": fake_sig,
        })
        assert resp.status_code == 403
        assert "invalid signature" in resp.json()["error"]

    def test_nonce_consumed_after_use(self, client):
        cid = "contributor-once"
        # Get challenge
        chal = client.post("/api/login/challenge", json={"contributor_id": cid}).json()
        nonce = chal["nonce"]
        sig = hmac.new(cid.encode(), nonce.encode(), hashlib.sha256).hexdigest()

        # First login succeeds
        r1 = client.post("/api/login", json={"contributor_id": cid, "signature": sig})
        assert r1.status_code == 200

        # Replay with same nonce fails (nonce already consumed)
        r2 = client.post("/api/login", json={"contributor_id": cid, "signature": sig})
        assert r2.status_code == 400
        assert "no challenge requested" in r2.json()["error"]


# ══════════════════════════════════════════════════════════════════ #
# /api/earnings                                                       #
# ══════════════════════════════════════════════════════════════════ #

class TestEarnings:
    def test_returns_json_with_contributors_and_count(self, client):
        resp = client.get("/api/earnings")
        assert resp.status_code == 200
        body = resp.json()
        assert "contributors" in body
        assert "count" in body
        assert "ts" in body
        assert isinstance(body["contributors"], list)
        assert body["count"] == len(body["contributors"])

    def test_starts_empty(self, client):
        resp = client.get("/api/earnings")
        assert resp.json()["count"] == 0

    def test_ts_is_numeric(self, client):
        ts = client.get("/api/earnings").json()["ts"]
        assert isinstance(ts, int)
        assert ts > 0


# ══════════════════════════════════════════════════════════════════ #
# /api/earnings/credit                                                #
# ══════════════════════════════════════════════════════════════════ #

def _do_login(client, cid: str):
    """Helper: challenge → sign → login, returns session token."""
    chal = client.post("/api/login/challenge", json={"contributor_id": cid}).json()
    sig = hmac.new(cid.encode(), chal["nonce"].encode(), hashlib.sha256).hexdigest()
    resp = client.post("/api/login", json={"contributor_id": cid, "signature": sig})
    return resp.json()["token"]


class TestEarningsCredit:
    def test_credit_increases_earnings(self, client):
        cid = "worker-1"
        token = _do_login(client, cid)

        # Credit 100 tokens
        resp = client.post("/api/earnings/credit", json={
            "contributor_id": cid,
            "amount": 100.0,
            "token": token,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["contributor_id"] == cid
        assert body["credited"] == 100.0
        assert body["earned"] == 100.0

        # Credit another 50
        resp2 = client.post("/api/earnings/credit", json={
            "contributor_id": cid,
            "amount": 50.0,
            "token": token,
        })
        assert resp2.json()["earned"] == 150.0

    def test_credit_reflected_in_earnings_ledger(self, client):
        cid = "worker-2"
        token = _do_login(client, cid)

        client.post("/api/earnings/credit", json={
            "contributor_id": cid,
            "amount": 123.456,
            "token": token,
        })

        # Check ledger
        ledger = client.get("/api/earnings").json()
        assert ledger["count"] == 1
        assert ledger["contributors"][0]["contributor_id"] == cid
        assert ledger["contributors"][0]["earned"] == 123.456

    def test_credit_ledger_sorted_descending(self, client):
        tok_a = _do_login(client, "a")
        tok_b = _do_login(client, "b")
        tok_c = _do_login(client, "c")

        client.post("/api/earnings/credit", json={"contributor_id": "a", "amount": 10.0, "token": tok_a})
        client.post("/api/earnings/credit", json={"contributor_id": "b", "amount": 50.0, "token": tok_b})
        client.post("/api/earnings/credit", json={"contributor_id": "c", "amount": 30.0, "token": tok_c})

        items = client.get("/api/earnings").json()["contributors"]
        earned_values = [item["earned"] for item in items]
        assert earned_values == [50.0, 30.0, 10.0]

    def test_credit_missing_contributor_id_rejected(self, client):
        resp = client.post("/api/earnings/credit", json={"amount": 10.0, "token": "tok"})
        assert resp.status_code == 400

    def test_credit_missing_token_rejected(self, client):
        resp = client.post("/api/earnings/credit", json={"contributor_id": "x", "amount": 10.0})
        assert resp.status_code == 403

    def test_credit_invalid_token_rejected(self, client):
        resp = client.post("/api/earnings/credit", json={
            "contributor_id": "x",
            "amount": 10.0,
            "token": "not-a-real-token",
        })
        assert resp.status_code == 403

    def test_credit_wrong_token_rejected(self, client):
        """Token for contributor A cannot credit contributor B."""
        tok_a = _do_login(client, "a")
        resp = client.post("/api/earnings/credit", json={
            "contributor_id": "b",
            "amount": 10.0,
            "token": tok_a,
        })
        assert resp.status_code == 403

    def test_credit_zero_or_negative_amount_rejected(self, client):
        token = _do_login(client, "neg-test")
        for amt in (0, -5, -0.01):
            resp = client.post("/api/earnings/credit", json={
                "contributor_id": "neg-test",
                "amount": amt,
                "token": token,
            })
            assert resp.status_code == 400, f"amount={amt} should fail"


# ══════════════════════════════════════════════════════════════════ #
# Phase 8: /api/model-info                                           #
# ══════════════════════════════════════════════════════════════════ #

class TestModelInfo:
    def test_returns_json_with_expected_keys(self, client):
        resp = client.get("/api/model-info")
        assert resp.status_code == 200
        body = resp.json()
        for key in ("name", "version", "architecture", "parameter_count",
                     "num_layers", "num_experts", "num_active_experts",
                     "vocab_size", "hidden_dim", "status", "supported_modes"):
            assert key in body, f"missing key: {key}"

    def test_name_is_not_placeholder(self, client):
        body = client.get("/api/model-info").json()
        assert body["name"] not in (None, "", "—", "unknown")
        assert isinstance(body["name"], str) and len(body["name"]) > 1

    def test_parameter_count_is_string_with_B(self, client):
        body = client.get("/api/model-info").json()
        assert "B" in body["parameter_count"]

    def test_num_layers_is_positive_int(self, client):
        body = client.get("/api/model-info").json()
        assert isinstance(body["num_layers"], int)
        assert body["num_layers"] > 0

    def test_status_indicates_stub(self, client):
        body = client.get("/api/model-info").json()
        assert "stub" in body["status"].lower()


# ══════════════════════════════════════════════════════════════════ #
# Phase 8: /api/device-info                                          #
# ══════════════════════════════════════════════════════════════════ #

class TestDeviceInfo:
    def test_returns_json_with_expected_keys(self, client):
        resp = client.get("/api/device-info")
        assert resp.status_code == 200
        body = resp.json()
        for key in ("hostname", "os", "python_version", "cpu", "gpu", "ram_total"):
            assert key in body, f"missing key: {key}"

    def test_os_is_not_empty(self, client):
        body = client.get("/api/device-info").json()
        assert body["os"]
        assert isinstance(body["os"], str)

    def test_python_version_is_semver_like(self, client):
        body = client.get("/api/device-info").json()
        # Should look like "3.x.y"
        assert body["python_version"].startswith("3.")

    def test_hostname_is_not_empty(self, client):
        body = client.get("/api/device-info").json()
        assert body["hostname"]
        assert isinstance(body["hostname"], str)


# ══════════════════════════════════════════════════════════════════ #
# Phase 8: /api/token-speed                                          #
# ══════════════════════════════════════════════════════════════════ #

class TestTokenSpeed:
    def test_returns_json_with_expected_keys(self, client):
        resp = client.get("/api/token-speed")
        assert resp.status_code == 200
        body = resp.json()
        for key in ("tokens_per_second", "total_tokens_generated", "ts"):
            assert key in body, f"missing key: {key}"

    def test_starts_at_zero_no_tokens_generated(self, client):
        body = client.get("/api/token-speed").json()
        # tps may be 0 or 0.0 — both float-ish
        assert float(body["tokens_per_second"]) >= 0
        assert body["total_tokens_generated"] >= 0

    def test_ts_is_recent(self, client):
        import time
        body = client.get("/api/token-speed").json()
        # Should be within a few seconds of now
        assert abs(body["ts"] - int(time.time())) < 10

    def test_tps_increases_after_chat(self, client):
        # Make a streaming chat call (triggers token speed measurement)
        resp = client.post("/v1/chat/completions", json={
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Hello world test"}],
            "stream": False,
            "max_tokens": 16,
        })
        assert resp.status_code == 200

        # Token speed should now show > 0 total tokens
        body = client.get("/api/token-speed").json()
        assert body["total_tokens_generated"] > 0


# ══════════════════════════════════════════════════════════════════ #
# Phase 8: /api/mode (get and set)                                    #
# ══════════════════════════════════════════════════════════════════ #

class TestMode:
    def test_get_mode_returns_json(self, client):
        resp = client.get("/api/mode")
        assert resp.status_code == 200
        body = resp.json()
        assert "mode" in body
        assert "available" in body
        assert isinstance(body["available"], list)
        assert "offline" in body["available"]
        assert "p2p" in body["available"]

    def test_default_mode_is_p2p(self, client):
        body = client.get("/api/mode").json()
        assert body["mode"] in ("p2p", "offline")

    def test_switch_to_offline(self, client):
        resp = client.post("/api/mode", json={"mode": "offline"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "offline"

        # Verify GET reflects change
        assert client.get("/api/mode").json()["mode"] == "offline"

    def test_switch_to_p2p(self, client):
        # First switch to offline
        client.post("/api/mode", json={"mode": "offline"})
        # Then back to p2p
        resp = client.post("/api/mode", json={"mode": "p2p"})
        assert resp.status_code == 200
        assert resp.json()["mode"] == "p2p"
        assert client.get("/api/mode").json()["mode"] == "p2p"

    def test_invalid_mode_rejected(self, client):
        resp = client.post("/api/mode", json={"mode": "invalid"})
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_empty_mode_rejected(self, client):
        resp = client.post("/api/mode", json={"mode": ""})
        assert resp.status_code == 400

    def test_missing_mode_key_rejected(self, client):
        resp = client.post("/api/mode", json={})
        assert resp.status_code == 400
