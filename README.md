# Astra — Distributed P2P Inference for DeepSeek-V4

<div align="right">
  <a href="README.md"><b>English</b></a> ·
  <a href="README_zh.md">中文</a>
</div>

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-389%20passed-brightgreen)]()
[![CI](https://github.com/qchauncey/astra/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Status](https://img.shields.io/badge/status-Phase%201--4%2C%206%20complete%2C%20Phase%205%20in%20progress%2C%20Phase%207%20hardware--blocked-blue)]()

**Astra** is an open-source P2P distributed inference framework that runs **DeepSeek-V4-Flash (284B)** across a cluster of commodity PCs (e.g., RTX 5070 Ti, 16 GB VRAM each) by combining:

- **[Petals](https://github.com/bigscience-workshop/petals)**-style decentralized pipeline parallelism
- **[KTransformers](https://github.com/kvcache-ai/ktransformers)**-style heterogeneous GPU/CPU compute split
- **[hivemind](https://github.com/learning-at-home/hivemind)** DHT for peer discovery and key-value storage

> **Alpha.** Phase 1–4 and Phase 6 are complete and tested (local + dual-node gRPC pipeline, full P2P infrastructure with peer identity, weight manifest verification, Engram storage nodes, real RTT measurement, DP/TEE security hardening, frontend portal). Phase 5 (mTLS + hivemind multi-machine DHT) is in implementation. Phase 7 (inference performance tuning — continuous batching, speculative decoding, KTransformers C++ binding, expert replication) is hardware-blocked: it requires a GPU cluster and DeepSeek-V4 weights to design and validate.

---

## Platform Support

| Component | Linux | macOS | Windows |
|-----------|:-----:|:-----:|:-------:|
| Development & testing | ✅ | ✅ | ✅ |
| API Gateway node | ✅ | ✅ | ✅ |
| DHT discovery node | ✅ | ✅ | ✅ |
| **Inference node** (real compute) | ✅ CUDA | ❌ | ⚠️ WSL2 + CUDA |
| KTransformers C++ kernel | ✅ Native | ❌ | ⚠️ WSL2 |

### What each hardware config can do

| Config | Inference compute | API Gateway | DHT node | Dev / test |
|--------|:-----------------:|:-----------:|:--------:|:----------:|
| Linux + NVIDIA GPU | ✅ | ✅ | ✅ | ✅ |
| Linux CPU-only | ❌ | ✅ | ✅ | ✅ |
| macOS (any) | ❌ | ✅ | ✅ | ✅ |
| Windows + GPU (WSL2) | ✅ | ✅ | ✅ | ✅ |
| Windows no GPU (native) | ❌ | ✅ | ✅ | ✅ |

> **No GPU = no inference contribution.** The numpy stub produces random outputs — it is only for development and testing, not for joining a real inference cluster.  
> To contribute actual compute on Windows, use [WSL2 + CUDA](#windows--gpu-inference-via-wsl2).  
> Without a GPU you can still run an **API Gateway** (handles user HTTP requests, routes to GPU peers) or a **DHT node** (peer discovery only).

---

## One-Click Install (Windows)

For Windows users who just want to try Astra without using the command line:

1. Clone or download this repository
2. Double-click **`installer\install.bat`** (or run `installer\install.ps1` in PowerShell)
3. Once installed, double-click **`installer\start.bat`** to launch

The launcher starts Astra in **offline mode** (single-machine, all layers local) and opens the web UI at `http://localhost:8080` automatically.

> **Note:** The current version uses a numpy inference stub — responses are placeholder text. Real AI output requires GPU + model weights (Phase 4).

---

## Quick Start

Jump to your platform:
- [Linux](#linux)
- [macOS](#macos)
- [Windows — no GPU (native)](#windows--no-gpu-native)
- [Windows — GPU inference via WSL2](#windows--gpu-inference-via-wsl2)

---

### Linux

```bash
# 1. Clone and install
git clone https://github.com/qchauncey/astra.git && cd astra
pip install -e ".[proto]"
pip install uvicorn   # needed for the web UI

# 2. Check your environment
python scripts/check_env.py

# 3. Offline mode — single machine, all layers local, web UI on port 8080
#    Opens a Claude-like chat interface at http://localhost:8080
python scripts/run_node.py --mode offline --api-port 8080

# 4. P2P mode — contribute a layer slice to a shared cluster
python scripts/run_node.py --node-id node-A --port 50051 \
    --layer-start 0 --layer-end 30 --hidden-dim 256 --api-port 8080

# 5. GPU mode (requires CUDA + KTransformers)
python scripts/run_node.py --node-id node-A --port 50051 \
    --layer-start 0 --layer-end 30 --gpu --api-port 8080

# 6. Single-machine multi-node cluster (validates full P2P pipeline without real GPU)
python scripts/run_cluster.py --nodes 3 --hidden-dim 256 --validate-only
python scripts/run_cluster.py --nodes 3 --hidden-dim 256 --api-port 8080
```

> **Stub disclaimer:** All inference output is currently random numpy arrays. The web UI, streaming, peer discovery, and gRPC pipeline are fully functional — only the model weights are absent (Phase 4).

---

### macOS

KTransformers C++ kernels require CUDA and are unavailable on macOS. Mac nodes **cannot contribute inference compute** to the cluster. Valid roles: API Gateway, DHT discovery node, development.

```bash
# Install Homebrew if needed (https://brew.sh)
brew install python@3.11 git

git clone https://github.com/qchauncey/astra.git && cd astra
pip3 install -e ".[proto]"

# Environment check and local testing
python scripts/check_env.py
python mock_pipeline.py --seq-len 32 --hidden-dim 256

# Role 1: API Gateway — receives user requests, routes to GPU peers
python scripts/run_node.py --node-id gateway --port 50051 --api-port 8080

# Role 2: DHT discovery node only (no API, no inference)
python scripts/run_node.py --node-id dht-node --port 50051
```

> **Apple Silicon note:** MPS (Metal Performance Shaders) backend is not yet integrated. Full GPU inference on Mac requires a future MPS adapter. Contributions welcome.

---

### Windows — No GPU (native)

Without a GPU, a Windows node **cannot contribute inference compute**. Valid roles: API Gateway, DHT discovery node, development and testing.  
To contribute real compute, use [WSL2 + CUDA](#windows--gpu-inference-via-wsl2).

**Option A — One-click installer (recommended)**

1. Clone this repo and double-click `installer\install.bat`
2. Double-click `installer\start.bat` — launches offline mode with the web UI

**Option B — Manual**

```powershell
# Install Python 3.10+ from https://python.org (check "Add to PATH")
# Install Git from https://git-scm.com

git clone https://github.com/qchauncey/astra.git
cd astra
pip install -e ".[proto]"
pip install uvicorn

# Environment check
python scripts/check_env.py

# Offline mode — single machine, all layers local, web UI on port 8080
python scripts/run_node.py --mode offline --api-port 8080

# Role 1: API Gateway — receives user HTTP requests, routes to GPU peers in the cluster
python scripts/run_node.py --node-id gateway --port 50051 --api-port 8080

# Role 2: DHT discovery node only (peer discovery, no inference)
python scripts/run_node.py --node-id dht-node --port 50051
```

---

### Windows — GPU Inference via WSL2

KTransformers requires Linux + CUDA. On Windows, WSL2 provides a full Linux kernel with GPU passthrough — Astra runs identically to native Linux inside it.

**Prerequisites**
- Windows 10 version 21H2 or later / Windows 11
- NVIDIA GPU with driver ≥ 535 (verify in PowerShell: `nvidia-smi`)

**Step 1 — Enable WSL2** *(PowerShell as Administrator)*

```powershell
wsl --install -d Ubuntu-22.04
# Restart Windows when prompted, then open "Ubuntu 22.04" from the Start menu
```

**Step 2 — Install the NVIDIA WSL2 CUDA driver** *(on Windows host, not inside WSL)*

1. Download the WSL2-compatible display driver from: https://developer.nvidia.com/cuda/wsl  
2. Install it on **Windows** as a normal driver update.  
3. Do **not** install the CUDA Toolkit on Windows — it lives inside WSL2 only.

**Step 3 — Install CUDA Toolkit inside WSL2 Ubuntu**

```bash
# Run these inside the WSL2 Ubuntu terminal
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4 build-essential python3-pip git

# Verify the GPU is visible
nvidia-smi
```

Expected output: your GPU name, driver version, and CUDA version.

**Step 4 — Clone and run Astra inside WSL2**

```bash
git clone https://github.com/qchauncey/astra.git && cd astra
pip3 install -e ".[proto]"

# Environment check
python scripts/check_env.py

# Mock pipeline (CPU-only, sanity check)
python mock_pipeline.py --seq-len 32 --hidden-dim 256

# Start a node with GPU enabled
python scripts/run_node.py --node-id node-A --port 50051 \
    --layer-start 0 --layer-end 30 --gpu --api-port 8080
```

**WSL2 tips**

| Topic | Detail |
|-------|--------|
| Accessing Windows files | Available at `/mnt/c/`, `/mnt/d/`, etc. inside WSL2 |
| Network | WSL2 ports are accessible from Windows at `localhost:<port>` — no extra config |
| GPU driver | Shared from Windows host; do **not** install a GPU driver inside WSL2 |
| Multi-machine | Each Windows machine runs its own WSL2; gRPC pipeline works identically to Linux |
| Performance | ~3–5% overhead vs bare-metal Linux; negligible for memory-bound MoE workloads |

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
│   ├── shared_expert_cache.py  # LRU expert cache with permanent pinning
│   ├── tokenizer.py            # HuggingFace AutoTokenizer + stub fallback
│   ├── weight_loader.py        # safetensors shard loader (verifies SHA-256)
│   ├── weight_manifest.py      # SHA-256 manifest — prevents tampered weights
│   └── differential_privacy.py # Differential privacy noise injection (ε/δ budget)
├── tee/
│   ├── __init__.py             # TEEBackend abstract interface
│   ├── gramine.py              # Intel SGX via Gramine Library OS
│   └── amd_sev.py              # AMD SEV-SNP confidential computing
├── routing/
│   └── geo_router.py           # GeoAwareMoERouter (token-level dispatch)
├── rpc/
│   ├── proto/inference.proto   # gRPC service definition
│   ├── generated/              # auto-generated pb2 stubs
│   ├── server.py               # InferenceServer
│   ├── client.py               # InferenceClient (pack → transmit → receive)
│   ├── tls.py                   # gRPC TLS secure channel (certificate management + mutual mTLS)
│   └── kv_transfer.py          # KV-cache chunked streaming
├── network/
│   ├── dht.py                  # AstraDHT — hivemind drop-in peer discovery + generic KV API
│   ├── engram.py               # EngramNode — storage-only DHT peers (KV-cache / weight shards)
│   ├── identity.py             # PeerIdentity & TrustRegistry — Ed25519 signing + TOFU
│   ├── orchestrator.py         # PipelineOrchestrator — N-node DHT chaining
│   └── rtt.py                  # RTTMonitor — TCP/gRPC latency probes with EWMA smoothing
└── api/
    ├── openai_compat.py        # OpenAI-compatible FastAPI endpoint + web UI serving
    └── static/
        └── index.html          # Phase 6 SPA dashboard (Chat, Monitor, Login, Earnings)

mock_pipeline.py                # Phase 1 & 2 local simulation harness
scripts/
├── run_node.py                 # Node launch CLI (--mode offline|p2p)
├── run_cluster.py              # Single-machine multi-node cluster launcher (Phase 3 validation)
└── check_env.py                # Environment readiness checker (prints node role eligibility)
installer/
├── install.sh                  # Linux/macOS one-command installer
├── install.bat                 # Windows CMD installer (double-click)
├── install.ps1                 # Windows PowerShell installer
└── start.bat                   # Windows one-click launcher (offline mode + browser)
tests/                          # 389 pytest tests (all passing)
.github/workflows/ci.yml        # CI: Python 3.10/3.11/3.12 matrix + lint
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
| `astra.inference.DPController` | Differential privacy: per-layer Gaussian/Laplace noise injection with ε/δ budget tracking |
| `astra.tee.GramineBackend` | Intel SGX TEE: attestation, model sealing, secure execution via Gramine Library OS |
| `astra.tee.SevBackend` | AMD SEV-SNP confidential computing: attestation, secure model loading |
| `astra.routing.GeoAwareMoERouter` | Token-level `(token, expert_id) → best_node` via haversine RTT |
| `astra.rpc.InferenceServer/Client` | gRPC pack → CRC32 verify → compute → deserialize loop |
| `astra.rpc.KVCacheSender/Receiver` | Chunked KV tensor streaming between pipeline stages |
| `astra.inference.AstraTokenizer` | HuggingFace AutoTokenizer wrapper with offline stub fallback |
| `astra.inference.WeightManifest` | SHA-256 weight shard manifest — prevents tampered weight loading |
| `astra.network.AstraDHT` | Peer discovery + generic KV API; drop-in for `hivemind.DHT` |
| `astra.network.PipelineOrchestrator` | DHT → layer coverage → retry-safe N-hop chaining |
| `astra.network.RTTMonitor` | Background TCP/gRPC latency prober; EWMA-smoothed RTT per peer |
| `astra.network.PeerIdentity` / `TrustRegistry` | Ed25519 node signing + TOFU key registry |
| `astra.network.EngramNode` | Storage-only DHT peer: in-memory or on-disk blob store |
| `astra.api.openai_compat` | OpenAI `/v1/chat/completions` + SSE streaming + web UI serving |
| `astra.api.static/index.html` | Web UI: Claude-like chat + sidebar peer panel with layer bars and latency |

---

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, data flow, wire format spec |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Phase-by-phase plan (Phase 1–4, 6 ✓ complete · Phase 5 in progress · Phase 7 hardware-blocked) |
| [docs/TESTING.md](docs/TESTING.md) | Test strategy: 389 tests covered + pending hardware test checklist |
| [docs/SECURITY.md](docs/SECURITY.md) | mTLS encryption, differential privacy, output tamper-proofing |
| [docs/TEE.md](docs/TEE.md) | TEE deployment guide: Intel SGX (Gramine) & AMD SEV-SNP attestation flow |
| [docs/FEASIBILITY.md](docs/FEASIBILITY.md) | Compute thresholds, geo micro-cluster tiers, bandwidth analysis |
| [docs/COMPLIANCE.md](docs/COMPLIANCE.md) | License compliance, DeepSeek model terms, patent analysis |

---

## Implementation Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | Local heterogeneous single-node inference (NumPy stub + SharedExpertCache) | ✅ Complete |
| **Phase 2** | LAN dual-node gRPC pipeline (pack → transmit → compute → receive loop) | ✅ Complete |
| **Phase 3** | Full P2P network: AstraDHT, N-node orchestration, OpenAI API, weight manifest, RTT monitor, peer identity, Engram nodes | ✅ Complete |
| **Phase 4** | Differential privacy (ε/δ budget, per-layer noise), TEE (Intel SGX + AMD SEV-SNP) | ✅ Complete |
| **Phase 5** | gRPC TLS mutual auth + hivemind multi-machine DHT integration | 🔄 In Progress |
| **Phase 6** | SPA dashboard (Chat, Monitor, Identity, Earnings), decentralized challenge-response login, real-time monitoring, contributor token accounting | ✅ Complete |

## Core Innovations

### 1. Geographic Micro-Cluster Scheduling
Node physical location (Haversine great-circle distance + propagation delay estimation) routes MoE expert requests to the nearest available peer, mitigating the blocking effect of high-frequency MoE network I/O.

### 2. Heterogeneous Compute Engine (KTransformers Integration)
- **GPU** handles: MLA attention layers, RoPE, LayerNorm, DSA operators
- **CPU/RAM** handles: MoE expert weight FFN forward computation (all 256 expert weights memory-resident)
- Set `ASTRA_USE_KTRANSFORMERS=1` to activate real C++ kernels; defaults to NumPy stubs for GPU-free development

### 3. Shared Expert Pinning
DeepSeek-V4's 2 shared experts fire on every token. Permanently pinned to GPU VRAM or high-speed RAM, eliminating repeated PCIe data movement overhead entirely.

### 4. Decoupled Storage (Engram Memory Nodes)
Built on AstraDHT (a hivemind DHT drop-in replacement), compute nodes and Engram storage nodes are fully decoupled — enabling independent scaling of distributed KV caches and model weight shards.

---

## Patent Protection

This project is licensed under **Apache License 2.0**. Any entity that initiates patent litigation against the project or its contributors automatically forfeits all patent rights granted herein. See [LICENSE](LICENSE) for full terms.

---

## Licensing

Licensed under **Apache License 2.0**. See [LICENSE](LICENSE).

Incorporates ideas from [Petals](https://github.com/bigscience-workshop/petals) and [KTransformers](https://github.com/kvcache-ai/ktransformers) (both Apache 2.0). All modifications are described in [NOTICE](NOTICE) and per-file headers.

---

## Contributing

PRs welcome. Include Apache 2.0 headers in new files and describe modifications per the [NOTICE](NOTICE) pattern.
