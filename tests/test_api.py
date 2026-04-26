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
Unit tests for the OpenAI-compatible FastAPI endpoint.

Uses Starlette's TestClient (synchronous) so no event-loop management is
required and the tests run cleanly in any CI environment.
All tests use an isolated DHT store to avoid cross-test pollution.
"""

import json

import pytest
from starlette.testclient import TestClient

from astra.api.openai_compat import _tokenize, _detokenize, create_app
from astra.network.dht import AstraDHT, _GlobalStore


# ────────────────────────────────────────────────────────────────── #
# Tokenizer helpers                                                   #
# ────────────────────────────────────────────────────────────────── #

class TestTokenize:
    def test_empty_string_returns_sentinel(self):
        assert _tokenize("") == [1]

    def test_single_word_returns_one_id(self):
        ids = _tokenize("hello")
        assert len(ids) == 1
        assert isinstance(ids[0], int)

    def test_deterministic_across_calls(self):
        assert _tokenize("hello world") == _tokenize("hello world")

    def test_different_words_different_ids(self):
        assert _tokenize("apple") != _tokenize("banana")

    def test_ids_in_valid_range(self):
        for i in _tokenize("the quick brown fox jumps"):
            assert 0 <= i <= 0x7FFF

    def test_whitespace_only_returns_sentinel(self):
        assert _tokenize("   ") == [1]


class TestDetokenize:
    def test_returns_string(self):
        assert isinstance(_detokenize([1, 2, 3]), str)

    def test_mentions_token_count(self):
        result = _detokenize([0] * 7)
        assert "7" in result


# ────────────────────────────────────────────────────────────────── #
# Fixtures                                                            #
# ────────────────────────────────────────────────────────────────── #

@pytest.fixture()
def client():
    store = _GlobalStore()
    dht = AstraDHT(node_id="test-api-gateway", store=store)
    app = create_app(dht=dht)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ────────────────────────────────────────────────────────────────── #
# /v1/models                                                          #
# ────────────────────────────────────────────────────────────────── #

def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    assert data["data"][0]["id"] == "deepseek-v4-flash"


# ────────────────────────────────────────────────────────────────── #
# /health                                                             #
# ────────────────────────────────────────────────────────────────── #

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "peers" in body


def test_health_peer_count_zero_with_no_nodes(client):
    resp = client.get("/health")
    assert resp.json()["peers"] == 0


# ────────────────────────────────────────────────────────────────── #
# /v1/pipeline/topology                                               #
# ────────────────────────────────────────────────────────────────── #

def test_topology_empty(client):
    resp = client.get("/v1/pipeline/topology")
    assert resp.status_code == 200
    body = resp.json()
    assert "num_peers" in body
    assert body["num_peers"] == 0


# ────────────────────────────────────────────────────────────────── #
# /v1/chat/completions — non-streaming                                #
# ────────────────────────────────────────────────────────────────── #

def test_chat_non_stream_response_schema(client):
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert "id" in body
    assert "choices" in body
    assert "usage" in body
    assert body["choices"][0]["message"]["role"] == "assistant"


def test_chat_usage_counts_consistent(client):
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Count tokens please"}],
        "stream": False,
    }
    usage = client.post("/v1/chat/completions", json=payload).json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_pipeline_error_graceful(client):
    """With no DHT peers the pipeline raises, but the API returns 200 gracefully."""
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] is not None


def test_chat_model_field_echoed(client):
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    assert client.post("/v1/chat/completions", json=payload).json()["model"] == "deepseek-v4-flash"


def test_chat_request_ids_unique(client):
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    r1 = client.post("/v1/chat/completions", json=payload).json()["id"]
    r2 = client.post("/v1/chat/completions", json=payload).json()["id"]
    assert r1 != r2


# ────────────────────────────────────────────────────────────────── #
# /v1/chat/completions — streaming (SSE)                              #
# ────────────────────────────────────────────────────────────────── #

def test_chat_stream_content_type(client):
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


def test_chat_stream_contains_done_marker(client):
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert "data: [DONE]" in resp.text


def test_chat_stream_chunks_parseable(client):
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Hi there"}],
        "stream": True,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    data_lines = [
        l for l in resp.text.strip().split("\n")
        if l.startswith("data:") and "[DONE]" not in l
    ]
    assert len(data_lines) > 0
    for line in data_lines:
        chunk = json.loads(line[len("data:"):].strip())
        assert chunk["object"] == "chat.completion.chunk"
        assert "choices" in chunk


# ────────────────────────────────────────────────────────────────── #
# Input validation                                                    #
# ────────────────────────────────────────────────────────────────── #

def test_chat_missing_messages_rejected(client):
    resp = client.post("/v1/chat/completions", json={"model": "x"})
    assert resp.status_code == 422


def test_chat_invalid_temperature_rejected(client):
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 5.0,  # max is 2.0
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 422
