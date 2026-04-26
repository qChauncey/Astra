# Astra — Distributed P2P Inference for DeepSeek-V4

<div align="right">
  <a href="README.md"><b>English</b></a> ·
  <a href="README_zh.md">中文</a>
</div>

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-130%20passed-brightgreen)]()
[![Status](https://img.shields.io/badge/status-Phase%203%20in%20progress-yellow)]()

**Astra** is an open-source P2P distributed inference framework that runs **DeepSeek-V4-Flash (284B)** across a cluster of commodity PCs (e.g., RTX 5070 Ti, 16 GB VRAM each) by combining:

- **[Petals](https://github.com/bigscience-workshop/petals)**-style decentralized pipeline parallelism
- **[KTransformers](https://github.com/kvcache-ai/ktransformers)**-style heterogeneous GPU/CPU compute split
- **[hivemind](https://github.com/learning-at-home/hivemind)** DHT for peer discovery and key-value storage

> **Alpha.** Phase 1 & 2 (local + dual-node gRPC pipeline) are complete and tested. Phase 3 (full P2P network + API gateway) is in progress.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/qchauncey/astra.git && cd astra
pip install -e ".[proto]"

# 2. Check your environment
python scripts/check_env.py

# 3. Run the mock pipeline (no GPU required)
python mock_pipeline.py --seq-len 32 --hidden-dim 256

# 4. Start a node (with OpenAI-compatible API on port 8080)
python scripts/run_node.py --node-id node-A --port 50051 \
    --layer-start 0 --layer-end 30 --api-port 8080
```

---

## Architecture

### Network Topology

```mermaid
flowchart TB
    USER(["👤 User / OpenAI SDK"])

    subgraph GW["API Gateway"]
        direction LR
        CHAT["/v1/chat/completions"]
        TOPO["/v1/pipeline/topology"]
    end

    subgraph NET["P2P Network Layer"]
        direction LR
        DHT[["AstraDHT\nPeer Discovery & KV Store"]]
        ROUTER(["GeoAware MoE Router\nToken-level Expert Dispatch\nHaversine RTT"])
    end

    subgraph NA["Node A · RTX 5070 Ti · Layers 0 – 20"]
        direction LR
        NA1["🖥 GPU\nMLA Attention\nRoPE · LayerNorm"]
        NA2["💾 CPU RAM\nMoE FFN · 256 Experts\n📌 Shared Experts Pinned"]
    end

    subgraph NB["Node B · RTX 5070 Ti · Layers 21 – 40"]
        direction LR
        NB1["🖥 GPU · MLA Attention"]
        NB2["💾 CPU RAM · MoE FFN"]
    end

    subgraph NC["Node C · RTX 4090 · Layers 41 – 60"]
        direction LR
        NC1["🖥 GPU · MLA Attention"]
        NC2["💾 CPU RAM · MoE FFN"]
    end

    USER -- "HTTP / SSE" --> CHAT
    CHAT --> ROUTER
    ROUTER -- "TensorPacket\ngRPC" --> NA
    NA -- "gRPC" --> NB
    NB -- "gRPC" --> NC
    NC -- "output states" --> CHAT
    CHAT -- "tokens / stream" --> USER

    DHT -. "peer discovery" .-> ROUTER
    DHT -. "heartbeat announce" .-> NA
    DHT -. "heartbeat announce" .-> NB
    DHT -. "heartbeat announce" .-> NC
    NA -. "KV-cache stream" .-> NB
    NB -. "KV-cache stream" .-> NC
```

### Per-Node Compute Split (KTransformers Model)

```
┌─────────────────────────────────────────────────────────┐
│                  Single Astra Node                       │
│                                                          │
│  ┌─────────────── GPU (16 GB VRAM) ──────────────────┐  │
│  │  Multi-Latent Attention (MLA)                      │  │
│  │  ├─ Q / K / V projections   (fused CUDA kernel)   │  │
│  │  ├─ RoPE positional encoding                       │  │
│  │  ├─ Scaled dot-product attention                   │  │
│  │  └─ Output projection + residual                   │  │
│  └────────────────────────────────────────────────────┘  │
│                         │ hidden states (float16)         │
│                         ▼                                 │
│  ┌─────────────── CPU RAM (≥ 64 GB) ─────────────────┐  │
│  │  MoE FFN  (KTransformers CPU offload)              │  │
│  │  ├─ Shared Experts 0 & 1  ← PINNED, never evicted │  │
│  │  ├─ Routed Experts 2–255  ← LRU-paged from NVMe   │  │
│  │  └─ SiLU-gated MLP: down( silu(gate(x)) * up(x) ) │  │
│  └────────────────────────────────────────────────────┘  │
│                         │ TensorPacket (gRPC)             │
│                         ▼  next node                      │
└─────────────────────────────────────────────────────────┘
```

### Hardware Requirements per Node

| Sub-layer | Device | Memory |
|-----------|--------|--------|
| MLA Attention + RoPE + LayerNorm | GPU VRAM | ~16 GB |
| Shared experts 0 & 1 (fire every token) | Pinned GPU / fast RAM | ~2 GB |
| 254 routed MoE experts (top-8 per token) | CPU RAM / NVMe mmap | ~530 GB across cluster |
| KV cache (per request) | CPU RAM | ~8 GB @ 8k ctx |

---

## Project Layout

```
astra/
├── serialization/
│   └── tensor_pack.py          # TensorPacket wire format v1
├── inference/
│   ├── heterogeneous.py        # HeterogeneousEngine (GPU attn + CPU MoE)
│   └── shared_expert_cache.py  # LRU expert cache with permanent pinning
├── routing/
│   └── geo_router.py           # GeoAwareMoERouter (token-level dispatch)
├── rpc/
│   ├── proto/inference.proto   # gRPC service definition
│   ├── generated/              # auto-generated pb2 stubs
│   ├── server.py               # InferenceServer
│   ├── client.py               # InferenceClient (pack → transmit → receive)
│   └── kv_transfer.py          # KV-cache chunked streaming
├── network/
│   ├── dht.py                  # AstraDHT — hivemind drop-in peer discovery
│   └── orchestrator.py         # PipelineOrchestrator — N-node DHT chaining
└── api/
    └── openai_compat.py        # OpenAI-compatible FastAPI endpoint

mock_pipeline.py                # Phase 1 & 2 local simulation harness
scripts/
├── run_node.py                 # Production node launch CLI
└── check_env.py                # Environment readiness checker
tests/                          # 70 pytest tests (all passing)
docs/
├── ARCHITECTURE.md             # Detailed design & wire format spec
└── ROADMAP.md                  # Phase-by-phase implementation plan
```

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `astra.serialization.TensorPacket` | Binary wire format: hidden states + routing metadata, float16 |
| `astra.inference.HeterogeneousEngine` | Attention on GPU stub · MoE FFN on CPU RAM |
| `astra.inference.SharedExpertCache` | LRU cache; experts 0 & 1 pinned, never evicted |
| `astra.routing.GeoAwareMoERouter` | Token-level `(token, expert_id) → best_node` via haversine RTT |
| `astra.rpc.InferenceServer/Client` | gRPC pack → CRC32 verify → compute → deserialize loop |
| `astra.rpc.KVCacheSender/Receiver` | Chunked KV tensor streaming between pipeline stages |
| `astra.network.AstraDHT` | Peer discovery; drop-in for `hivemind.DHT` |
| `astra.network.PipelineOrchestrator` | DHT → layer coverage → retry-safe N-hop chaining |
| `astra.api.openai_compat` | OpenAI `/v1/chat/completions` + SSE streaming |

---

## Documentation

| 文档 | 内容 |
|-----|-----|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 系统设计、数据流、传输格式规范 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 分阶段实施计划（Phase 1 ✓ · Phase 2 ✓ · Phase 3 进行中） |
| [docs/TESTING.md](docs/TESTING.md) | 测试方案：已覆盖 70 项 + 待完成测试清单 |
| [docs/SECURITY.md](docs/SECURITY.md) | 加密方案、mTLS、差分隐私、输出防篡改 |
| [docs/FEASIBILITY.md](docs/FEASIBILITY.md) | 算力门槛、地理微集群划分、带宽需求分析 |
| [docs/COMPLIANCE.md](docs/COMPLIANCE.md) | 许可证合规、DeepSeek 模型条款、专利分析 |

---

## Licensing

Licensed under **Apache License 2.0**. See [LICENSE](LICENSE).

Incorporates ideas from [Petals](https://github.com/bigscience-workshop/petals) and [KTransformers](https://github.com/kvcache-ai/ktransformers) (both Apache 2.0). All modifications are described in [NOTICE](NOTICE) and per-file headers.

---

## Contributing

PRs welcome. Include Apache 2.0 headers in new files and describe modifications per the [NOTICE](NOTICE) pattern.
