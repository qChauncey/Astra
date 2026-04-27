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
#   - Written from scratch; implements OpenAI Chat Completions API surface
#     as a FastAPI application backed by the Astra P2P pipeline.

"""
OpenAI-compatible Chat Completions API for Astra.

Exposes an HTTP endpoint that mirrors the OpenAI `/v1/chat/completions`
request / response schema so existing OpenAI client libraries work
without modification:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="astra")
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Tokenization:
  In production: use DeepSeek-V4's tokenizer (tiktoken / HuggingFace).
  Here: simple whitespace tokenizer returns dummy token IDs so the full
  request→pipeline→response round-trip can be exercised without the model.

Run::

    python -m astra.api.openai_compat
    # or
    uvicorn astra.api.openai_compat:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import pathlib
import secrets
import time
import uuid
import zlib
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..network.dht import AstraDHT
from ..network.orchestrator import PipelineConfig, PipelineOrchestrator

STATIC_DIR = pathlib.Path(__file__).parent / "static"


# ─────────────────────────────────────────────────────────────────────────── #
# OpenAI schema models                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-v4-flash"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=256, ge=1, le=8192)
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    n: Optional[int] = Field(default=1, ge=1, le=4)
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int
    delta: ChoiceDelta
    finish_reason: Optional[str] = None


class StreamChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


class CompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


# ─────────────────────────────────────────────────────────────────────────── #
# Minimal tokenizer stub                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _tokenize(text: str) -> List[int]:
    """
    Whitespace tokenizer stand-in.
    Production: replace with DeepSeek-V4 tokenizer (tiktoken / transformers).
    """
    words = text.split()
    return [zlib.crc32(w.encode()) & 0x7FFF for w in words] or [1]


def _detokenize(token_ids: List[int]) -> str:
    """
    Dummy detokenizer: returns a placeholder string.
    Production: replace with vocabulary lookup.
    """
    return f"[Astra response — {len(token_ids)} output tokens generated via P2P pipeline]"


# ─────────────────────────────────────────────────────────────────────────── #
# Application factory                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def create_app(
    dht: Optional[AstraDHT] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    node_id: str = "astra-node",
    layer_start: int = 0,
    layer_end: int = 61,
    mode: str = "p2p",
) -> FastAPI:
    """
    Create and return the FastAPI application.

    Parameters
    ----------
    dht:             AstraDHT instance.  If None, a local DHT is created.
    pipeline_config: PipelineConfig for the orchestrator.
    node_id:         Identifier for this node (shown in UI).
    layer_start:     First transformer layer this node handles.
    layer_end:       One-past-last transformer layer this node handles.
    mode:            "offline" (single-machine) or "p2p" (distributed).
    """
    app = FastAPI(
        title="Astra Inference API",
        description="OpenAI-compatible endpoint backed by Astra P2P pipeline",
        version="0.1.0-alpha",
    )

    # Runtime state attached to app
    app.state.dht = dht or AstraDHT(node_id="api-gateway")
    app.state.pipeline_config = pipeline_config or PipelineConfig()
    app.state.node_id = node_id
    app.state.layer_start = layer_start
    app.state.layer_end = layer_end
    app.state.mode = mode

    # Serve static files (dashboard UI)
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ------------------------------------------------------------------ #
    # Routes                                                                #
    # ------------------------------------------------------------------ #

    @app.get("/", include_in_schema=False)
    async def root():
        index = STATIC_DIR / "index.html"
        if index.is_file():
            return FileResponse(index)
        return {"message": "Astra API — no UI found, use /v1/chat/completions"}

    @app.get("/api/my-node")
    async def my_node(raw: Request):
        """Return this node's identity and configuration."""
        state = raw.app.state
        return {
            "node_id": state.node_id,
            "layer_start": state.layer_start,
            "layer_end": state.layer_end,
            "mode": state.mode,
            "hidden_dim": state.pipeline_config.hidden_dim,
            "total_layers": state.pipeline_config.num_layers,
        }

    @app.get("/api/peers")
    async def peers(raw: Request):
        """Return all DHT-registered peers with their metadata."""
        all_peers = raw.app.state.dht.get_all_peers()
        result = []
        for p in all_peers:
            result.append({
                "node_id": p.node_id,
                "address": p.address,
                "layer_start": p.layer_start,
                "layer_end": p.layer_end,
                "geo_region": getattr(p, "geo_region", "unknown"),
                "backend": getattr(p, "backend", "unknown"),
            })
        return {"peers": result, "count": len(result)}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "deepseek-v4-flash",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "astra",
                    "description": "DeepSeek-V4-Flash 284B via Astra P2P (numpy stub)",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, raw: Request):
        # Assemble prompt text
        prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
        token_ids = _tokenize(prompt)
        prompt_tokens = len(token_ids)

        # Cap to a fast subset for the stub (full forward pass is slow without GPU)
        max_input = min(len(token_ids), 32)
        token_ids = token_ids[:max_input]

        # Run pipeline (or fall back gracefully if no peers available)
        orchestrator = PipelineOrchestrator(
            dht=raw.app.state.dht,
            config=raw.app.state.pipeline_config,
        )

        generated_ids: List[int] = []
        pipeline_error: Optional[str] = None
        try:
            orchestrator.run(token_ids, use_kv_cache=True)
            # In a real system: sample from logits.  Here: produce dummy tokens.
            n_out = min(req.max_tokens or 32, 32)
            rng = np.random.default_rng(seed=int(time.time()))
            generated_ids = rng.integers(1, 32000, size=n_out).tolist()
        except Exception as exc:
            pipeline_error = str(exc)
            # Graceful degradation: still return a response with error note
            generated_ids = [1]

        completion_text = _detokenize(generated_ids)
        if pipeline_error:
            completion_text = f"[Pipeline error: {pipeline_error}] " + completion_text

        completion_tokens = len(generated_ids)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        if req.stream:
            return StreamingResponse(
                _stream_response(request_id, req.model, completion_text, created_ts),
                media_type="text/event-stream",
            )

        return ChatCompletionResponse(
            id=request_id,
            created=created_ts,
            model=req.model,
            choices=[
                CompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    @app.get("/v1/pipeline/topology")
    async def pipeline_topology(raw: Request):
        """Return current DHT-discovered pipeline topology."""
        orch = PipelineOrchestrator(dht=raw.app.state.dht)
        return orch.topology()

    @app.get("/health")
    async def health(raw: Request):
        peers = raw.app.state.dht.get_all_peers()
        return {"status": "ok", "peers": len(peers)}

    # ------------------------------------------------------------------ #
    # Phase 6: Monitor — live Ping aggregation                              #
    # ------------------------------------------------------------------ #

    @app.get("/api/monitor")
    async def monitor(raw: Request):
        """Ping every known peer and return aggregated hardware stats."""
        peers = raw.app.state.dht.get_all_peers()
        nodes: List[dict] = []
        for p in peers:
            # Attempt gRPC Ping via InferenceClient
            try:
                from ..rpc.client import InferenceClient, PingResult
                client = InferenceClient(target=p.address)
                result = await asyncio.to_thread(client.ping)
                nodes.append({
                    "node_id": result.node_id,
                    "address": p.address,
                    "layer_start": result.layer_start,
                    "layer_end": result.layer_end,
                    "geo_region": result.geo_region,
                    "expert_shards": result.expert_shards,
                    "backend": result.backend,
                    "gpu_util": result.gpu_util,
                    "cpu_util": result.cpu_util,
                    "ready": result.ready,
                    "reachable": True,
                })
            except Exception:
                # Peer unreachable — report stale metadata
                nodes.append({
                    "node_id": p.node_id,
                    "address": p.address,
                    "layer_start": p.layer_start,
                    "layer_end": p.layer_end,
                    "geo_region": getattr(p, "geo_region", "unknown"),
                    "expert_shards": getattr(p, "expert_shards", []),
                    "backend": getattr(p, "backend", "unknown"),
                    "gpu_util": 0.0,
                    "cpu_util": 0.0,
                    "ready": False,
                    "reachable": False,
                })

        return {"nodes": nodes, "count": len(nodes), "ts": int(time.time())}

    # ------------------------------------------------------------------ #
    # Phase 6: Login — decentralized identity                                #
    # ------------------------------------------------------------------ #

    # In-process identity store (production: replace with persistent DB)
    _identity_nonce: Dict[str, str] = {}

    @app.post("/api/login")
    async def login(raw: Request):
        """Decentralized challenge-response login. Returns a session token."""
        import json as _json
        body = await raw.body()
        payload = _json.loads(body)
        contributor_id = payload.get("contributor_id", "").strip()
        signature = payload.get("signature", "")

        if not contributor_id or not signature:
            return JSONResponse(
                content={"error": "contributor_id and signature required"},
                status_code=400,
            )

        # Look up the stored challenge nonce for this contributor
        nonce = _identity_nonce.pop(contributor_id, None)
        if nonce is None:
            return JSONResponse(
                content={"error": "no challenge requested — call /api/login/challenge first"},
                status_code=400,
            )

        # Verify HMAC-SHA256 signature of nonce using contributor_id as shared secret
        expected = hmac.new(
            contributor_id.encode(),
            nonce.encode(),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            return JSONResponse(
                content={"error": "invalid signature"},
                status_code=403,
            )

        # Issue session token (production: use JWT with expiry)
        token = f"astra-session-{contributor_id}-{secrets.token_hex(16)}"
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        return {
            "token": token,
            "token_hash": token_hash,
            "contributor_id": contributor_id,
            "expires_at": int(time.time()) + 86400,
        }

    @app.post("/api/login/challenge")
    async def login_challenge(raw: Request):
        """Return a challenge nonce for the given contributor_id."""
        import json as _json
        body = await raw.body()
        payload = _json.loads(body)
        contributor_id = payload.get("contributor_id", "").strip()
        if not contributor_id:
            return JSONResponse(
                content={"error": "contributor_id required"},
                status_code=400,
            )

        nonce = secrets.token_hex(32)
        _identity_nonce[contributor_id] = nonce
        return {"contributor_id": contributor_id, "nonce": nonce}

    # ------------------------------------------------------------------ #
    # Phase 6: Earnings — token accounting                                   #
    # ------------------------------------------------------------------ #

    # In-process ledger (production: on-chain or auditable DB)
    _earnings_ledger: Dict[str, float] = {}
    _earnings_log: List[dict] = []

    @app.get("/api/earnings")
    async def earnings(raw: Request):
        """Return earnings ledger for all contributors."""
        items = [
            {"contributor_id": cid, "earned": round(amt, 6)}
            for cid, amt in sorted(_earnings_ledger.items(), key=lambda x: -x[1])
        ]
        return {"contributors": items, "count": len(items), "ts": int(time.time())}

    @app.post("/api/earnings/credit")
    async def earnings_credit(raw: Request):
        """Credit a contributor for completed work (token‑based incentive)."""
        import json as _json
        body = await raw.body()
        payload = _json.loads(body)
        contributor_id = payload.get("contributor_id", "").strip()
        amount = payload.get("amount", 0.0)
        token = payload.get("token", "")

        if not contributor_id or amount <= 0:
            return JSONResponse(
                content={"error": "contributor_id and positive amount required"},
                status_code=400,
            )

        # Verify session token
        expected_prefix = f"astra-session-{contributor_id}-"
        if not token.startswith(expected_prefix):
            return JSONResponse(
                content={"error": "invalid or expired token"},
                status_code=403,
            )

        _earnings_ledger[contributor_id] = (
            _earnings_ledger.get(contributor_id, 0.0) + amount
        )
        _earnings_log.append({
            "contributor_id": contributor_id,
            "amount": amount,
            "ts": int(time.time()),
        })

        return {
            "contributor_id": contributor_id,
            "earned": round(_earnings_ledger[contributor_id], 6),
            "credited": amount,
        }

    return app


# ─────────────────────────────────────────────────────────────────────────── #
# Streaming helper                                                              #
# ─────────────────────────────────────────────────────────────────────────── #


async def _stream_response(
    request_id: str,
    model: str,
    text: str,
    created: int,
) -> AsyncGenerator[str, None]:
    words = text.split()
    for i, word in enumerate(words):
        chunk = StreamChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant" if i == 0 else None,
                        content=word + (" " if i < len(words) - 1 else ""),
                    ),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        await asyncio.sleep(0.02)

    # Final chunk
    done_chunk = StreamChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[StreamChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
    )
    yield f"data: {done_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────────────────── #
# Standalone entry point                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

# Module-level app instance for `uvicorn astra.api.openai_compat:app`
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "astra.api.openai_compat:app",
        host="0.0.0.0",
        port=int(os.environ.get("ASTRA_API_PORT", "8080")),
        reload=False,
    )
