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

import os
import time
import uuid
from typing import AsyncGenerator, List, Literal, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..network.dht import AstraDHT, DHTNodeRecord
from ..network.orchestrator import PipelineConfig, PipelineOrchestrator
from ..serialization.tensor_pack import DEEPSEEK_V4_HIDDEN_DIM, DEEPSEEK_V4_NUM_LAYERS


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
    return [hash(w) & 0x7FFF for w in words] or [1]


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
) -> FastAPI:
    """
    Create and return the FastAPI application.

    Parameters
    ----------
    dht:             AstraDHT instance.  If None, a local DHT is created
                     (useful for single-process testing).
    pipeline_config: PipelineConfig for the orchestrator.
    """
    app = FastAPI(
        title="Astra Inference API",
        description="OpenAI-compatible endpoint backed by Astra P2P pipeline",
        version="0.1.0-alpha",
    )

    # Runtime state attached to app
    app.state.dht = dht or AstraDHT(node_id="api-gateway")
    app.state.pipeline_config = pipeline_config or PipelineConfig()

    # ------------------------------------------------------------------ #
    # Routes                                                                #
    # ------------------------------------------------------------------ #

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
            result = orchestrator.run(token_ids, use_kv_cache=True)
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
    async def health():
        peers = raw_app_dht(app).get_all_peers()
        return {"status": "ok", "peers": len(peers)}

    return app


def raw_app_dht(app: FastAPI) -> AstraDHT:
    return app.state.dht


# ─────────────────────────────────────────────────────────────────────────── #
# Streaming helper                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

import asyncio
import json


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
