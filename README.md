# Astra — Distributed P2P Inference for DeepSeek-V4

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![Status](https://img.shields.io/badge/status-Phase%201%20%26%202%20complete-green)]()

**Astra** is an open-source P2P distributed inference framework that runs **DeepSeek-V4-Flash (284B)** across a cluster of commodity PCs (e.g., RTX 5070 Ti, 16 GB VRAM each) by combining:

- **[Petals](https://github.com/bigscience-workshop/petals)**-style decentralized pipeline parallelism
- **[KTransformers](https://github.com/kvcache-ai/ktransformers)**-style heterogeneous GPU/CPU compute split
- **[hivemind](https://github.com/learning-at-home/hivemind)** DHT for peer discovery and key-value storage

> This project is in **alpha**. Phase 1 (local single-node) and Phase 2 (two-node gRPC relay) are implemented and tested. Phase 3 (full P2P network + frontend) is in progress.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/qchauncey/astra.git && cd astra
pip install -e ".[proto]"

# 2. Check your environment
python scripts/check_env.py

# 3. Run the mock pipeline (local simulation, no GPU required)
python mock_pipeline.py --seq-len 32 --hidden-dim 256

# 4. Run with real DeepSeek-V4 hidden dimensions (needs ~64 GB RAM for simulation)
python mock_pipeline.py --seq-len 8 --hidden-dim 7168
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Astra P2P Network                       │
│                                                            │
│  ┌──────────┐   gRPC    ┌──────────┐   gRPC   ┌────────┐  │
│  │  Node A  │──────────►│  Node B  │──────────►│ Node C │  │
│  │ L0 – L20 │           │ L21– L40 │           │L41–L60 │  │
│  │          │           │          │           │        │  │
│  │ GPU: MLA │           │ GPU: MLA │           │GPU:MLA │  │
│  │ CPU: MoE │           │ CPU: MoE │           │CPU:MoE │  │
│  └──────────┘           └──────────┘           └────────┘  │
│                                                            │
│  ◄──────────── hivemind DHT (peer discovery) ──────────►   │
│  ◄──────── Geo-Aware MoE Router (token dispatch) ───────►  │
└────────────────────────────────────────────────────────────┘
```

**Hardware split per node (KTransformers model):**

| Sub-layer | Device | Why |
|-----------|--------|-----|
| Multi-Latent Attention (MLA) | GPU VRAM | High compute density; fits in 16 GB |
| RoPE, LayerNorm, DSA ops | GPU | Fused kernel efficiency |
| MoE FFN expert weights (254 routed + 2 shared) | CPU RAM / NVMe | 284B × bf16 ≈ 568 GB total; paged on demand |
| Shared experts 0 & 1 | Pinned in GPU (or fast RAM) | Fire on every token; eliminate PCIe round-trips |

---

## Project Layout

```
astra/
├── serialization/
│   └── tensor_pack.py       # TensorPacket — wire format for P2P token relay
├── rpc/
│   ├── proto/inference.proto # gRPC service definition
│   ├── generated/           # auto-generated pb2 stubs
│   ├── server.py            # InferenceServer (one per pipeline node)
│   └── client.py            # InferenceClient (pack → transmit → receive)
├── inference/
│   ├── heterogeneous.py     # HeterogeneousEngine (GPU attn + CPU MoE)
│   └── shared_expert_cache.py # SharedExpertCache (pinned expert weights)
└── routing/
    └── geo_router.py        # GeoAwareMoERouter (token-level dispatch)

mock_pipeline.py             # Phase 1 & 2 local simulation harness
scripts/check_env.py         # Environment readiness checker
docs/
├── ARCHITECTURE.md          # Detailed system design
└── ROADMAP.md               # Implementation roadmap
```

---

## Module Overview

### `astra.serialization.TensorPacket`
Binary wire format for hidden states between pipeline nodes. Carries hidden state tensors (float16/bfloat16), layer range, token positions, selected expert indices, and geographic routing metadata.

### `astra.inference.HeterogeneousEngine`
Runs transformer layers with the KTransformers split:
- `_attention_forward()` → GPU (KTransformers C++ stub / numpy fallback)
- `_moe_forward()` → CPU RAM via `SharedExpertCache`

### `astra.inference.SharedExpertCache`
LRU cache for MoE expert weights. Shared experts (IDs 0–1) are pinned and never evicted, eliminating PCIe round-trips for the highest-frequency computation.

### `astra.routing.GeoAwareMoERouter`
Token-level dispatch router. Assigns each `(token, expert_id)` pair to the nearest available node using haversine RTT estimates. Shared experts always stay local.

### `astra.rpc.{InferenceServer, InferenceClient}`
gRPC layer implementing the pack → transmit → receive loop with CRC32 integrity verification, streaming support, and capability advertisement via Ping.

---

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — detailed system design and data flow
- [docs/ROADMAP.md](docs/ROADMAP.md) — phase-by-phase implementation plan

---

## Licensing

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE).

This project incorporates ideas from [Petals](https://github.com/bigscience-workshop/petals) (Apache 2.0) and [KTransformers](https://github.com/kvcache-ai/ktransformers) (Apache 2.0). All modifications are described in [NOTICE](NOTICE) and in per-file headers.

---

## Contributing

PRs welcome. Please include Apache 2.0 headers in new files and describe modifications per the [NOTICE](NOTICE) pattern.
