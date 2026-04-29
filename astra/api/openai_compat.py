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
import logging
import os
import pathlib
import secrets
import socket
import time
import uuid
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..inference.heterogeneous import DeviceMap
from ..inference.tokenizer import get_tokenizer
from ..network.dht import AstraDHT, DHTNodeRecord
from ..network.orchestrator import PipelineConfig, PipelineOrchestrator
from ..rpc.server import InferenceServer

STATIC_DIR = pathlib.Path(__file__).parent / "static"


# ─────────────────────────────────────────────────────────────────────────── #
# Device detection helpers (Phase 8)                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def _detect_cpu_brand() -> str:
    """Return the CPU brand string or 'unknown'."""
    import platform
    try:
        import subprocess, sys
        if sys.platform == "win32":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "name"], text=True, timeout=5
            )
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            return lines[-1] if len(lines) > 1 else platform.processor() or "unknown"
        elif sys.platform == "darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
            )
            return out.strip() or "unknown"
        else:  # Linux
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except Exception:
                pass
            return platform.processor() or "unknown"
    except Exception:
        return platform.processor() or "unknown"


def _detect_gpu_brand() -> str:
    """Return the GPU brand string or 'none / not detected'."""
    try:
        import subprocess, sys
        if sys.platform == "win32":
            out = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                text=True, timeout=5,
            )
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            return lines[-1] if len(lines) > 1 else "none / not detected"
        else:
            try:
                out = subprocess.check_output(
                    ["lspci"], text=True, timeout=5
                )
                for line in out.splitlines():
                    if "VGA" in line or "3D" in line or "Display" in line:
                        return line.strip()
            except Exception:
                pass
            return "none / not detected"
    except Exception:
        return "none / not detected"


def _detect_ram_total() -> str:
    """Return total RAM in human-readable format."""
    import platform
    try:
        import psutil
        total = psutil.virtual_memory().total
        if total >= 1 << 30:
            return f"{total / (1 << 30):.1f} GB"
        return f"{total / (1 << 20):.0f} MB"
    except ImportError:
        pass
    try:
        import subprocess, sys, re
        if sys.platform == "win32":
            out = subprocess.check_output(
                ["wmic", "computersystem", "get", "totalphysicalmemory"],
                text=True, timeout=5,
            )
            m = re.search(r"(\d+)", out)
            if m:
                gb = int(m.group(1)) / (1024 ** 3)
                return f"{gb:.1f} GB"
        elif sys.platform == "darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=5
            )
            gb = int(out.strip()) / (1024 ** 3)
            return f"{gb:.1f} GB"
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(re.search(r"(\d+)", line).group(1))
                        gb = kb / (1024 * 1024)
                        return f"{gb:.1f} GB"
    except Exception:
        pass
    return f"{platform.system()} default"


# ─────────────────────────────────────────────────────────────────────────── #
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
# Tokenizer helpers (delegated to astra.inference.tokenizer)                   #
# ─────────────────────────────────────────────────────────────────────────── #

def _tokenize(text: str) -> List[int]:
    return get_tokenizer().encode(text)


def _detokenize(token_ids: List[int]) -> str:
    tok = get_tokenizer()
    if tok.is_stub:
        return f"[Astra response — {len(token_ids)} output tokens (stub, no real model)]"
    return tok.decode(token_ids)


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

    _log = logging.getLogger("astra.api")

    # Runtime state attached to app
    app.state.dht = dht or AstraDHT(node_id="api-gateway")
    app.state.pipeline_config = pipeline_config or PipelineConfig()
    app.state.node_id = node_id
    app.state.layer_start = layer_start
    app.state.layer_end = layer_end
    app.state.mode = mode

    # Offline-local InferenceServer (started on-demand, registered in DHT)
    app.state._offline_server: Optional[InferenceServer] = None
    app.state._offline_port: int = 0

    # ── Offline mode: self-register as a local pipeline peer ───────────
    if mode == "offline":
        try:
            # Find an available port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                app.state._offline_port = s.getsockname()[1]

            pc = app.state.pipeline_config
            num_layers = getattr(pc, "num_layers", 61)
            hidden_dim = getattr(pc, "hidden_dim", 4096)

            dmap = DeviceMap.cpu_only()

            app.state._offline_server = InferenceServer(
                node_id=f"{node_id}-local",
                layer_start=0,
                layer_end=num_layers,
                port=app.state._offline_port,
                geo_region="local",
                device_map=dmap,
                max_workers=4,
            )
            app.state._offline_server.start()
            _log.info(
                "Offline InferenceServer started on 127.0.0.1:%d (layers 0-%d)",
                app.state._offline_port, num_layers,
            )

            # Register local server in DHT so orchestrator can discover it
            record = DHTNodeRecord(
                node_id=f"{node_id}-local",
                address=f"127.0.0.1:{app.state._offline_port}",
                layer_start=0,
                layer_end=num_layers,
                expert_shards=list(range(256)),
                geo_region="local",
                backend="numpy_stub",
            )
            app.state.dht.announce(record, ttl=3600)
            _log.info("Offline node self-registered in DHT: %s", record.node_id)

        except Exception as exc:
            _log.exception("Failed to start offline InferenceServer: %s", exc)
            app.state._offline_server = None

    # ── Shutdown hook ──────────────────────────────────────────────────
    @app.on_event("shutdown")
    async def _shutdown():
        srv: Optional[InferenceServer] = getattr(app.state, "_offline_server", None)
        if srv is not None:
            try:
                srv.stop(grace=2.0)
            except Exception:
                pass
            try:
                app.state.dht.revoke()
            except Exception:
                pass

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

        # Update token speed (Phase 8)
        now = time.time()
        if _token_speed_state["last_measurement_ts"] > 0:
            elapsed = now - _token_speed_state["last_measurement_ts"]
            if elapsed > 0.01:
                _token_speed_state["tokens_per_second"] = completion_tokens / elapsed
        else:
            _token_speed_state["tokens_per_second"] = float(completion_tokens)
        _token_speed_state["last_measurement_ts"] = now
        _token_speed_state["total_tokens_generated"] += completion_tokens

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
    # Phase 8: Model info, device info, token speed, mode switching         #
    # ------------------------------------------------------------------ #

    # Cached model metadata (production: read from model config / runtime)
    _model_meta = {
        "name": "MiniMax-M2.5",
        "version": "M2.5-Chat-0.1.0",
        "architecture": "MoE + MLA (GQA)",
        "parameter_count": "126B",
        "num_layers": 62,
        "num_experts": 256,
        "num_active_experts": 8,
        "vocab_size": 200000,
        "hidden_dim": 4096,
        "status": "stub (numpy random, no GPU)",
        "supported_modes": ["offline", "p2p"],
    }

    # Token speed measurement (updated by the chat endpoint)
    _token_speed_state: Dict[str, Union[float, int]] = {
        "tokens_per_second": 0.0,
        "last_measurement_ts": 0,
        "total_tokens_generated": 0,
    }

    @app.get("/api/model-info")
    async def model_info(_raw: Request):
        """Return model metadata (name, version, architecture, parameter count)."""
        return dict(_model_meta)

    @app.get("/api/device-info")
    async def device_info(_raw: Request):
        """Return current device / hardware information."""
        import platform
        info: Dict[str, str] = {
            "hostname": platform.node(),
            "os": f"{platform.system()} {platform.release()}",
            "python_version": platform.python_version(),
            "cpu": _detect_cpu_brand(),
            "gpu": _detect_gpu_brand(),
            "ram_total": _detect_ram_total(),
        }
        return info

    @app.get("/api/token-speed")
    async def token_speed(_raw: Request):
        """Return current token generation speed (tok/s)."""
        return {
            "tokens_per_second": round(_token_speed_state["tokens_per_second"], 2),
            "total_tokens_generated": _token_speed_state["total_tokens_generated"],
            "ts": int(time.time()),
        }

    @app.get("/api/mode")
    async def get_mode(raw: Request):
        """Return current operating mode."""
        return {"mode": raw.app.state.mode, "available": ["offline", "p2p"]}

    @app.post("/api/mode")
    async def set_mode(raw: Request):
        """Switch operating mode (offline or p2p)."""
        import json as _json
        body = await raw.body()
        payload = _json.loads(body)
        new_mode = payload.get("mode", "").strip()
        if new_mode not in ("offline", "p2p"):
            return JSONResponse(
                content={"error": "mode must be 'offline' or 'p2p'"},
                status_code=400,
            )

        old_mode = raw.app.state.mode
        if new_mode == old_mode:
            return {"mode": new_mode}

        # ── Switch to offline: start local InferenceServer + register in DHT ──
        if new_mode == "offline":
            try:
                import socket
                from ..inference.heterogeneous import DeviceMap
                from ..rpc.server import InferenceServer
                from ..network.dht import DHTNodeRecord

                # Find an available port
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", 0))
                    raw.app.state._offline_port = s.getsockname()[1]

                pc = raw.app.state.pipeline_config
                num_layers = getattr(pc, "num_layers", 61)
                node_id = raw.app.state.node_id

                dmap = DeviceMap.cpu_only()
                raw.app.state._offline_server = InferenceServer(
                    node_id=f"{node_id}-local",
                    layer_start=0,
                    layer_end=num_layers,
                    port=raw.app.state._offline_port,
                    geo_region="local",
                    device_map=dmap,
                    max_workers=4,
                )
                raw.app.state._offline_server.start()
                _log.info(
                    "Offline InferenceServer started on 127.0.0.1:%d (layers 0-%d)",
                    raw.app.state._offline_port, num_layers,
                )

                # Register local server in DHT so orchestrator can discover it
                record = DHTNodeRecord(
                    node_id=f"{node_id}-local",
                    address=f"127.0.0.1:{raw.app.state._offline_port}",
                    layer_start=0,
                    layer_end=num_layers,
                    expert_shards=list(range(256)),
                    geo_region="local",
                    backend="numpy_stub",
                )
                raw.app.state.dht.announce(record, ttl=3600)
                _log.info("Offline node self-registered in DHT: %s", record.node_id)

            except Exception as exc:
                _log.exception("Failed to start offline InferenceServer: %s", exc)
                raw.app.state._offline_server = None
                return JSONResponse(
                    content={"error": f"Failed to start offline server: {exc}"},
                    status_code=500,
                )

        # ── Switch to p2p: stop local InferenceServer + revoke from DHT ──
        elif new_mode == "p2p":
            srv = getattr(raw.app.state, "_offline_server", None)
            if srv is not None:
                try:
                    srv.stop(grace=2.0)
                except Exception:
                    pass
                raw.app.state._offline_server = None
            try:
                raw.app.state.dht.revoke()
            except Exception:
                pass

        raw.app.state.mode = new_mode
        return {"mode": new_mode}

    # ------------------------------------------------------------------ #
    # Phase 6: Monitor — live Ping aggregation                              #
    # ------------------------------------------------------------------ #

    @app.get("/api/monitor")
    async def monitor(raw: Request):
        """Ping every known peer and return aggregated hardware stats."""
        peers = raw.app.state.dht.get_all_peers()
        nodes: List[dict] = []
        for p in peers:
            # Attempt gRPC Ping via InferenceClient — bounded to 5 s so one
            # unreachable peer cannot stall the entire monitor response.
            try:
                from ..rpc.client import InferenceClient
                client = InferenceClient(target=p.address)
                result = await asyncio.wait_for(
                    asyncio.to_thread(client.ping), timeout=5.0
                )
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

    # In-process identity store (production: replace with persistent DB).
    # Capped at 1 000 entries to prevent memory exhaustion from unredeemed
    # challenges; oldest entry evicted when the limit is reached.
    _identity_nonce: Dict[str, str] = {}
    _NONCE_MAX = 1_000

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

        if len(_identity_nonce) >= _NONCE_MAX:
            # Evict the oldest entry to keep the dict bounded.
            _identity_nonce.pop(next(iter(_identity_nonce)))
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