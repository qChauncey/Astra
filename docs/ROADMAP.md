# Astra — Implementation Roadmap

> Version 0.1 · April 2025 · Apache License 2.0

---

## Overview

Astra is developed in six phases, each building on the previous.  The goal of each phase is a **runnable, testable artifact** — not just design work.

---

## Phase 1 — Local Heterogeneous Single-Node (COMPLETE ✓)

**Goal:** Prove the CPU/GPU split works end-to-end on a single machine.

| Task | Status | Module |
|------|--------|--------|
| TensorPacket wire format (serialize / deserialize) | ✓ Done | `astra/serialization/tensor_pack.py` |
| Binary round-trip test (pack → unpack → verify) | ✓ Done | `mock_pipeline.py` Phase 1 Step A |
| SharedExpertCache with pinned shared experts 0 & 1 | ✓ Done | `astra/inference/shared_expert_cache.py` |
| HeterogeneousEngine: attention stub + MoE CPU path | ✓ Done | `astra/inference/heterogeneous.py` |
| KTransformersStub (numpy fallback for dev/CI) | ✓ Done | `astra/inference/heterogeneous.py` |
| GeoAwareMoERouter: haversine RTT + gate + dispatch | ✓ Done | `astra/routing/geo_router.py` |
| mock_pipeline.py Phase 1 runner | ✓ Done | `mock_pipeline.py` |
| Environment checker (`scripts/check_env.py`) | ✓ Done | `scripts/check_env.py` |
| Apache 2.0 compliance (headers, NOTICE, LICENSE) | ✓ Done | root + all source files |

**Milestone test:**
```bash
python mock_pipeline.py --phase 1 --seq-len 16 --hidden-dim 256
# Expected: "Phase 1 COMPLETE ✓"
```

---

## Phase 2 — Dual-Node LAN Pipeline (COMPLETE ✓)

**Goal:** Two nodes on localhost exchange TensorPackets over gRPC, completing the "pack → transmit → compute → receive" loop.

| Task | Status | Module |
|------|--------|--------|
| `inference.proto` gRPC service definition | ✓ Done | `astra/rpc/proto/inference.proto` |
| Compiled pb2 Python stubs | ✓ Done | `astra/rpc/generated/` |
| InferenceServer (gRPC servicer + lifecycle) | ✓ Done | `astra/rpc/server.py` |
| InferenceClient (serialize → RPC → deserialize) | ✓ Done | `astra/rpc/client.py` |
| CRC32 integrity check on wire | ✓ Done | `astra/rpc/client.py` |
| Ping / capability advertisement | ✓ Done | `InferenceServer._servicer.Ping` |
| mock_pipeline.py Phase 2 runner (2 threaded servers) | ✓ Done | `mock_pipeline.py` |

**Milestone test:**
```bash
python mock_pipeline.py --phase 2 --seq-len 16 --hidden-dim 256
# Expected: "Phase 2 COMPLETE ✓" with RTT numbers for both nodes
```

---

## Phase 3 — Full P2P Network (SOFTWARE-COMPLETE ✓)

**Goal:** Real multi-machine cluster with hivemind DHT discovery, peer
authentication, weight integrity, and storage/compute role separation.

> **Scope split (April 2026):** The original Phase 3 plan bundled performance
> optimizations (continuous batching, speculative decoding, expert
> replication, adaptive load balancing) with the P2P infrastructure. Those
> optimizations cannot be validated without real model weights and a GPU
> cluster, so they have been moved to **[Phase 7 — Inference Performance
> Tuning](#phase-7--inference-performance-tuning-blocked-on-hardware)** and
> tagged as hardware-blocked. The P2P infrastructure portion is now complete.

### 3.1 P2P Node Discovery

| Task | Status | Notes |
|------|--------|-------|
| Integrate `hivemind.DHT` for peer discovery | ✓ Done | `astra/network/dht.py` |
| DHT-based expert shard advertisement | ✓ Done | Nodes publish `{expert_ids, layer_range, region}` |
| Dynamic node join/leave handling in `GeoAwareMoERouter` | ✓ Done | DHT event callbacks |
| **Engram memory node (storage-only DHT peers)** | ✓ Done | `astra/network/engram.py` — InMemory + Disk stores, DHT discovery |

### 3.2 Production Inference Engine (P2P infrastructure portion)

| Task | Status | Notes |
|------|--------|-------|
| KV-cache streaming between nodes (`TransferKVCache` RPC) | ✓ Done | `astra/rpc/kv_transfer.py` — chunked ≤3 MB streaming |
| DeepSeek-V4 checkpoint loader (safetensors) | ✓ Done | `astra/inference/weight_loader.py` |
| Tokenizer integration (HuggingFace + stub fallback) | ✓ Done | `astra/inference/tokenizer.py` |
| Real KTransformers C++ binding integration | → Phase 7 | Hardware-blocked |
| Speculative decoding | → Phase 7 | Hardware-blocked |
| Continuous batching | → Phase 7 | Hardware-blocked |

### 3.3 Geographic Micro-Cluster Optimization (foundation)

| Task | Status | Notes |
|------|--------|-------|
| **Real RTT measurement replacing haversine estimate** | ✓ Done | `astra/network/rtt.py` — TCP/gRPC probes, EWMA smoothing |
| `GeoAwareMoERouter` consumes measured RTT | ✓ Done | Falls back to haversine when no measurement available |
| Cluster-affinity grouping (nodes within N ms latency) | → Phase 7 | Needs production traffic patterns |
| Expert shard replication for hot experts | → Phase 7 | Hardware-blocked |
| Adaptive load balancing across nodes | → Phase 7 | Hardware-blocked |

### 3.4 P2P Security

| Task | Status | Notes |
|------|--------|-------|
| gRPC TLS + mutual certificate auth | → Phase 5 | `astra/rpc/tls.py` |
| **Peer identity via Ed25519 key pairs** | ✓ Done | `astra/network/identity.py` — sign/verify advertisements + TOFU registry |
| **Weight shard integrity (SHA-256 manifest)** | ✓ Done | `astra/inference/weight_manifest.py` — verified by `WeightLoader` on every shard load |

### 3.5 CLI & API Infrastructure

| Task | Status | Notes |
|------|--------|-------|
| OpenAI-compatible API endpoint (SSE streaming) | ✓ Done | `astra/api/openai_compat.py` |
| PipelineOrchestrator (N-node DHT chaining) | ✓ Done | `astra/network/orchestrator.py` |
| run_node CLI (production node launcher) | ✓ Done | `scripts/run_node.py` (--mode offline / p2p) |
| run_cluster CLI (single-machine multi-node cluster) | ✓ Done | `scripts/run_cluster.py` |

### 3.6 Frontend Portal

| Task | Status | Notes |
|------|--------|-------|
| All Phase 3.6 items | → Phase 6 | Completed in Phase 6 — see below |

---

## Phase 4 — Security & Privacy Hardening (COMPLETE ✓)

**Goal:** Hardened inference with differential privacy protections and Trusted Execution Environment support.

### 4.1 Differential Privacy (Hidden-State Noise Injection)

| Task | Status | Module |
|------|--------|--------|
| `PrivacyBudget` (ε/δ tracking, exhaustion check) | ✓ Done | `astra/inference/differential_privacy.py` |
| `MomentsAccountant` (RDP → (ε,δ) conversion via Rényi divergence) | ✓ Done | `astra/inference/differential_privacy.py` |
| `DPController` (Gaussian + Laplace mechanisms, utility verification) | ✓ Done | `astra/inference/differential_privacy.py` |
| `LayerDPInjector` (per-layer epsilon splitting across 61 layers) | ✓ Done | `astra/inference/differential_privacy.py` |
| `HeterogeneousEngine` integration (`dp_injector` parameter) | ✓ Done | `astra/inference/heterogeneous.py` |
| DP unit tests (budget accounting, noise calibration, utility thresholds) | ✓ Done | `tests/test_differential_privacy.py` |

### 4.2 Trusted Execution Environment (TEE)

| Task | Status | Module |
|------|--------|--------|
| `TEEBackend` abstract interface (status, attest, seal, unseal, get_quote) | ✓ Done | `astra/tee/__init__.py` |
| `GramineBackend` — Intel SGX via Gramine Library OS | ✓ Done | `astra/tee/gramine.py` |
| `SevBackend` — AMD SEV-SNP confidential computing | ✓ Done | `astra/tee/amd_sev.py` |
| TEE deployment guide (hardware requirements, manifest generation, attestation flow) | ✓ Done | `docs/TEE.md` |

### 4.3 Documentation & Testing

| Task | Status | Notes |
|------|--------|-------|
| Phase 4 roadmap entry | ✓ Done | This document |
| Security roadmap update (DP + TEE marked complete) | ✓ Done | `docs/SECURITY.md` |
| Testing matrix update (DP test file added) | ✓ Done | `docs/TESTING.md` |

---

## Phase 5 — gRPC TLS + hivemind Multi-Machine DHT (IN PROGRESS)

**Goal:** Secure all inter-node communication with mutual TLS and integrate live hivemind DHT for real multi-machine discovery.

### 5.1 gRPC Transport Security

| Task | Status | Module |
|------|--------|--------|
| Generate per-node TLS certificates (X.509 self-signed) | In Progress | `astra/rpc/` |
| Exchange `secure_channel` with mutual TLS credentials | In Progress | `astra/rpc/server.py`, `client.py` |
| Certificate pinning / TOFU trust model for P2P bootstrap | Pending | `astra/network/` |
| gRPC TLS integration tests (encrypted RPC round-trip) | ✅ Done | `tests/test_tls.py` |

### 5.2 hivemind Multi-Machine DHT

| Task | Status | Module |
|------|--------|--------|
| Replace in-memory `AstraDHT` store with live `hivemind.DHT` | In Progress | `astra/network/dht.py` |
| Multi-machine DHT bootstrap (initial peer rendezvous) | Pending | `astra/network/` |
| Cross-machine expert shard advertisement via DHT | Pending | `astra/network/dht.py` |
| DHT-based KV cache location lookup | Pending | `astra/network/dht.py` |
| hivemind DHT integration tests (multi-node discovery) | Pending | `tests/` |

### 5.3 Documentation

| Task | Status | Notes |
|------|--------|-------|
| TLS deployment guide | ✅ Done | `docs/TLS.md` — Certificate generation + distribution |
| hivemind DHT configuration guide | ✅ Done | `docs/HIVEMIND.md` — Bootstrap peer setup, NAT traversal |

---

## Phase 6 — Frontend Portal (COMPLETE ✓)

**Goal:** Build a user-facing web portal with decentralized login and real-time monitoring.

| Task | Status | Module |
|------|--------|--------|
| SPA dashboard (Chat, Monitor, Identity, Earnings) | ✓ Done | `astra/api/static/index.html` |
| Real-time compute / VRAM / RTT monitoring (`/api/monitor`) | ✓ Done | `astra/api/openai_compat.py` — live Ping aggregation |
| Decentralized challenge-response login (`/api/login`) | ✓ Done | `astra/api/openai_compat.py` — HMAC-SHA256 nonce-based |
| Contributor earnings / token accounting (`/api/earnings`) | ✓ Done | `astra/api/openai_compat.py` — in-process ledger |
| Phase 6 unit tests (25 items) | ✓ Done | `tests/test_phase6.py` |

---

## Phase 7 — Inference Performance Tuning (BLOCKED ON HARDWARE)

**Goal:** Performance optimizations that require real model weights, real
GPU hardware, and real production workloads to design and validate.

> **Why blocked:** Each item below makes a quantitative trade-off (latency
> vs throughput, memory vs replication overhead, draft-model accuracy vs
> speedup). Without real measurements these decisions devolve into
> guesswork, and any code we write would be untested. The infrastructure
> they plug into (HeterogeneousEngine, GeoAwareMoERouter, KVCacheSender)
> is already in place; what's missing is the data to drive the design.

### 7.1 Inference Engine

| Task | Status | Prerequisite |
|------|--------|--------------|
| Real KTransformers C++ binding integration | Pending | KTransformers compiled for target CUDA arch + 580 GB DeepSeek-V4 weights |
| Continuous batching across pipeline stages | Pending | Real model + observable production traffic |
| Speculative decoding with draft model on fast node | Pending | Real model + draft model checkpoint |

### 7.2 Routing & Load Distribution

| Task | Status | Prerequisite |
|------|--------|--------------|
| Cluster-affinity grouping (nodes within N ms latency) | Pending | Multi-region production deployment to derive thresholds |
| Expert shard replication for hot experts | Pending | Production expert-frequency telemetry |
| Adaptive load balancing across nodes | Pending | Real GPU utilization measurements |

### 7.3 Hardware CI

| Task | Status | Prerequisite |
|------|--------|--------------|
| Self-hosted GPU runner (`hardware_test.yml`) | Pending | Access to a CUDA-capable runner — typically not free |
| Real-weight numerical alignment tests vs reference impl | Pending | Self-hosted runner above |

---

## Dependency Upgrade Path

| Component | Current (mock) | Production target |
|-----------|---------------|-------------------|
| Tensor compute | numpy stub | KTransformers C++ + CUDA |
| Attention kernel | numpy `@` matmul | `ktransformers.ops.mla_forward` |
| DHT | in-memory dict | `hivemind.DHT` |
| Transport | insecure gRPC | gRPC with mTLS |
| Model weights | random arrays | DeepSeek-V4 safetensors shards |
| Memory | 16–64 GB RAM | 512 GB+ NVMe-backed mmap |

---

## Testing Strategy

> 详细方案见 [docs/TESTING.md](TESTING.md)

| 层级 | 工具 | 当前状态 | 覆盖目标 |
|-----|------|---------|---------|
| 单元测试（CPU） | pytest | ✅ 389 个，全通过 | 序列化、LRU 缓存、Haversine + 真实 RTT、DHT、Engram、Peer Identity、Weight Manifest、gRPC TLS、HeterogeneousEngine、Tokenizer、KVTransfer、OpenAI API、Phase 6 dashboard |
| 集成测试（本地） | pytest + threading | ✅ 已覆盖 | mock_pipeline.py Phase 1 & 2 |
| 硬件集成测试 | 自托管 GPU Runner | ❌ 未配置 | KTransformers C++ 内核、真实权重数值对齐 |
| 负载测试 | locust / 自定义 | ❌ 未实现 | 100 并发请求，吞吐量与 P99 延迟 |

### 待完成测试项（Pending）

| 测试文件 | 状态 | 说明 |
|---------|------|-----|
| `tests/test_heterogeneous.py` | ✅ 完成 | `HeterogeneousEngine` 直接单元测试（23 项） |
| `tests/test_kv_transfer.py` | ✅ 完成 | KV 缓存分块传输与重组（20 项） |
| `tests/test_api.py` | ✅ 完成 | OpenAI API 端点（httpx AsyncClient）（23 项） |
| `.github/workflows/hardware_test.yml` | ❌ 待创建 | 自托管 GPU Runner CI 配置 |

---

## 配套文档

| 文档 | 内容 |
|-----|-----|
| [docs/TESTING.md](TESTING.md) | 完整测试方案，含待完成项与硬件测试要求 |
| [docs/SECURITY.md](SECURITY.md) | 加密方案、威胁模型、差分隐私、mTLS 实施路线 |
| [docs/FEASIBILITY.md](FEASIBILITY.md) | 算力门槛、地理微集群划分、带宽需求、风险分析 |
| [docs/COMPLIANCE.md](COMPLIANCE.md) | 许可证合规、DeepSeek 模型使用条款、专利分析 |

---

## Known Limitations (Alpha)

1. **No real model weights** — all tensors are random. The infrastructure
   (loader, manifest verification, tokenizer, gRPC, DHT) is ready to accept
   real weights, but Phase 7 work (real KTransformers + GPU validation)
   remains.
2. **KTransformersStub is numpy** — ~100× slower than C++ CUDA kernels.
   Use for correctness testing only.
3. **DHT is in-memory** — `AstraDHT` uses an in-memory store as the default
   backend. The `hivemind.DHT` bridge exists (`astra/network/hivemind_bridge.py`)
   but multi-machine bootstrap is still in progress (Phase 5).
4. **TLS available but not enforced by default** — `astra/rpc/tls.py` ships
   the certificate machinery; production deployments must opt in. See
   `docs/TLS.md`.
5. **No hardware CI** — GPU integration tests require a self-hosted runner.
   See [docs/TESTING.md](TESTING.md) for the pending hardware test plan.