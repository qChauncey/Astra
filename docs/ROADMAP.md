# Astra — Implementation Roadmap

> Version 0.1 · April 2025 · Apache License 2.0

---

## Overview

Astra is developed in seven phases, each building on the previous.  The goal of each phase is a **runnable, testable artifact** — not just design work.

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

## Phase 5 — gRPC TLS + hivemind Multi-Machine DHT (SOFTWARE-COMPLETE ✓)

**Goal:** Secure all inter-node communication with mutual TLS and integrate live hivemind DHT for real multi-machine discovery.

> **Scope note (April 2026):** All code deliverables — TLS certificate generation,
> mTLS server/client integration, TOFU trust store, hivemind DHT bridge, and the
> `create_dht()` factory with graceful degradation — are complete and tested
> (389 passed, 1 skipped). Multi-machine bootstrap validation (multiple physical
> nodes) is a deployment verification step, not a software gap. See
> [docs/HIVEMIND.md](HIVEMIND.md) §8 Production Checklist for the per-node setup
> procedure.

### 5.1 gRPC Transport Security

| Task | Status | Module |
|------|--------|--------|
| Generate per-node TLS certificates (X.509 self-signed) | ✓ Done | `astra/rpc/tls.py` |
| Exchange `secure_channel` with mutual TLS credentials | ✓ Done | `astra/rpc/server.py`, `client.py` |
| Certificate pinning / TOFU trust model for P2P bootstrap | ✓ Done | `astra/rpc/tls.py` — `TofuTrustStore` with serialization |
| gRPC TLS integration tests (encrypted RPC round-trip) | ✓ Done | `tests/test_tls.py` (19 items) |

### 5.2 hivemind Multi-Machine DHT

| Task | Status | Module |
|------|--------|--------|
| Replace in-memory `AstraDHT` store with live `hivemind.DHT` | ✓ Done | `astra/network/hivemind_bridge.py` — `HivemindDHT` + `create_dht()` factory |
| Multi-machine DHT bootstrap (initial peer rendezvous) | ✓ Done | `HivemindDHT.__init__` accepts `initial_peers` multiaddr list |
| Cross-machine expert shard advertisement via DHT | ✓ Done | `HivemindDHT.announce()` publishes `expert_shards` |
| DHT-based KV cache location lookup | ✓ Done | `HivemindDHT.store()` / `fetch()` generic KV API |
| hivemind DHT integration tests (multi-node discovery) | ✓ Done | `tests/test_hivemind_bridge.py` (21 items) |

### 5.3 Documentation

| Task | Status | Notes |
|------|--------|-------|
| TLS deployment guide | ✓ Done | `docs/TLS.md` — Certificate generation + distribution |
| hivemind DHT configuration guide | ✓ Done | `docs/HIVEMIND.md` — Bootstrap peer setup, NAT traversal |

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

## Phase 7 — Inference Performance Tuning (IN PROGRESS)

**Goal:** Connect the numpy stub to real GPU hardware, validate multi-machine
P2P operation, and implement production inference optimizations (continuous
batching, speculative decoding, expert replication).

> **Two distinct work categories remain:** items that only require hardware
> to *run* (validation/configuration — code is already written), and items
> that require hardware to *design* (new software must be built alongside
> measurement data). Both categories are blocked until a GPU cluster is
> available.

### 7.1 Soft Deliverables ✓ COMPLETE

| Task | Status | Artifact |
|------|--------|---------|
| CI workflow: `.github/workflows/hardware_test.yml` | ✓ Done | Self-hosted runner config — attach runner to activate |
| Benchmark tooling: `scripts/benchmark.py` | ✓ Done | Token/s, P50/P95/P99; single / gRPC / API modes |
| Docker Compose multi-node deployment | ✓ Done | `docker-compose.yml` — 4-service cluster (dht-seed + 3 nodes + gateway) |
| Load-test script: `scripts/load_test.py` | ✓ Done | asyncio+httpx concurrent driver; SLA thresholds; JSON output |

---

### 7.2 Validation Only 🔒 Hardware-Blocked
> Code is already written. These tasks close once hardware is available and
> the existing code is executed against it. No new software required.

| Task | Nature | Prerequisite |
|------|--------|--------------|
| Register self-hosted GPU runner | Config | 1× Linux machine + CUDA GPU; runner registered in GitHub Actions Settings |
| Run `hardware_test.yml` → real-weight numerical alignment | Test | Runner above + DeepSeek-V4-Flash weights (580 GB, HuggingFace) |
| Multi-machine DHT bootstrap (3+ physical nodes) | Test | 3 machines; run `create_dht(use_hivemind=True)` on each with shared `initial_peers` |
| Cross-machine KV-cache transfer validation | Test | 2-node cluster; observe `KVCacheSender/Receiver` logs |
| Multi-machine gRPC latency/throughput benchmark | Test | 2+ machines, ≥1 Gbps LAN; run `scripts/benchmark.py --mode grpc` |
| Intel SGX quote verification | Test | Intel SGX CPU + PCCS service; run `GramineBackend.attest()` |
| AMD SEV-SNP attestation report validation | Test | AMD EPYC Milan/Genoa + SEV-SNP firmware; run `SevBackend.attest()` |

---

### 7.3 New Code Required 🔒 Hardware-Blocked
> These items need new software. Hardware is needed both to guide design
> decisions (latency thresholds, batch sizes, expert frequencies) and to
> validate the implementation once built.
>
> **7.3.2–7.3.4 software is complete** (86 tests passing). Only 7.3.1
> (KTransformers C++ binding) and the hardware prerequisites remain.

#### 7.3.1 KTransformers C++ Binding
**Effort:** Medium — adapter layer only; inference contract already defined.

| Task | Status | Notes |
|------|--------|-------|
| Replace numpy stub in `HeterogeneousEngine._attention_forward()` | 🔒 Blocked | Call `ktransformers.ops.mla_forward(q, k, v, ...)` instead of numpy matmul |
| Replace numpy stub in `HeterogeneousEngine._moe_forward()` | 🔒 Blocked | Call `ktransformers.ops.expert_forward(hidden, weight_path, ...)` |
| Handle CUDA tensor lifecycle (device placement, dtype casting) | 🔒 Blocked | Tensors must stay on GPU between attention and MoE to avoid PCIe round-trip |
| Update `HeterogeneousEngine.from_gpu_config()` to pass CUDA device | 🔒 Blocked | Currently passes dummy device string |
| **Prerequisite** | 🔒 Blocked | `ktransformers` compiled for target CUDA arch + DeepSeek-V4 safetensors shards |

#### 7.3.2 Continuous Batching
**Effort:** Large — touches scheduler, server, orchestrator, and KV-cache.

| Task | Status | Notes |
|------|--------|-------|
| Add `BatchScheduler` in `astra/inference/` | ✓ Complete | `astra/inference/batch_scheduler.py` — dynamic batch window; length binning; SLA tracking |
| Modify `InferenceServer` to accept batched `TensorPacket` | ✓ Complete | `astra/rpc/server.py` — `forward_batch` endpoint with per-request KV-cache isolation |
| Implement padding/unpadding for variable-length sequences | ✓ Complete | `astra/inference/batch_utils.py` — `pad_sequences`, `unpad_output`, attention masks |
| Extend `PipelineOrchestrator` to fan-out batch across nodes | ✓ Complete | `astra/network/orchestrator.py` — multi-node batch dispatch |
| **Prerequisite** | 🔒 Blocked | Real model + observable production traffic to calibrate batch window and timeout |

#### 7.3.3 Speculative Decoding
**Effort:** Large — requires draft model pipeline running in parallel.

| Task | Status | Notes |
|------|--------|-------|
| Add `DraftModelRunner` in `astra/inference/` | ✓ Complete | `astra/inference/speculative.py` — stub draft model with configurable K, temperature, seed |
| Implement token acceptance/rejection sampling | ✓ Complete | `SpeculativePipeline` — strict (exact match) + relaxed (rejection sampling) verifiers |
| Wire async draft+verify pipeline into `PipelineOrchestrator` | ✓ Complete | `SpeculativePipeline.step()` — draft → verify → merge with stats tracking |
| **Prerequisite** | 🔒 Blocked | Full model checkpoint + draft model checkpoint |

#### 7.3.4 Expert Shard Replication & Adaptive Load Balancing
**Effort:** Medium — extends existing `GeoAwareMoERouter` and `EngramNode`.

| Task | Status | Notes |
|------|--------|-------|
| Add expert access frequency telemetry to `GeoAwareMoERouter` | ✓ Complete | `astra/routing/expert_telemetry.py` — `ExpertTelemetry` with hot-expert detection, pruning, snapshots |
| Implement hot-expert replica placement in `EngramNode` | ✓ Complete | `ExpertTelemetry.get_replica_targets()` — top-K experts for replica placement |
| Add replica-aware routing in `GeoAwareMoERouter._best_node_for_expert()` | ✓ Complete | `astra/routing/geo_router.py` — replica-aware dispatch with telemetry recording |
| Implement GPU utilisation-based load shedding in `PipelineOrchestrator` | ✓ Complete | `astra/network/orchestrator.py` — `_is_node_overloaded()`, `_node_load_score()`, threshold config |
| Add cluster-affinity grouping threshold to `GeoAwareMoERouter` | ✓ Complete | `astra/routing/cluster_affinity.py` — `ClusterAffinity` with EMA RTT, proximity groups |
| **Prerequisite** | 🔒 Blocked | Multi-node deployment with real GPU utilisation data and expert-frequency telemetry |

---

## Dependency Upgrade Path

| Component | Current (mock) | Production target |
|-----------|---------------|-------------------|
| Tensor compute | numpy stub | KTransformers C++ + CUDA |
| Attention kernel | numpy `@` matmul | `ktransformers.ops.mla_forward` |
| DHT | in-memory dict / `HivemindDHT` (Phase 5 done) | `hivemind.DHT` |
| Transport | gRPC | gRPC with mTLS (done — Phase 5) ✅ |
| Model weights | random arrays | DeepSeek-V4 safetensors shards |
| Memory | 16–64 GB RAM | 512 GB+ NVMe-backed mmap |

---

## Testing Strategy

> See [docs/TESTING.md](TESTING.md) for detailed plan

| Layer | Tool | Current Status | Coverage Target |
|-----|------|---------|---------|
| Unit (CPU) | pytest | ✅ 389 passed + 1 skipped | Serialization, LRU cache, Haversine + real RTT, DHT, Engram, Peer Identity, Weight Manifest, gRPC TLS, HeterogeneousEngine, Tokenizer, KVTransfer, OpenAI API, Phase 6 dashboard |
| Integration (local) | pytest + threading | ✅ Covered | mock_pipeline.py Phase 1 & 2 |
| Hardware Integration | Self-hosted GPU Runner | ❌ Not configured | KTransformers C++ kernels, real-weight numerical alignment |
| Load Test | scripts/load_test.py (asyncio+httpx) | ✅ Implemented | 100 concurrent requests, throughput & P99 latency |

### Pending Test Items

| Test File | Status | Notes |
|---------|------|-----|
| `tests/test_heterogeneous.py` | ✅ Done | `HeterogeneousEngine` direct unit tests (23 items) |
| `tests/test_kv_transfer.py` | ✅ Done | KV-cache chunked transfer & reassembly (20 items) |
| `tests/test_api.py` | ✅ Done | OpenAI API endpoints (httpx AsyncClient) (23 items) |
| `.github/workflows/hardware_test.yml` | ✅ Done | Self-hosted GPU Runner CI config — attach runner to activate |

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
3. **DHT bridge ready, multi-machine validation pending** — The hivemind DHT
   bridge (`astra/network/hivemind_bridge.py`) is fully implemented and tested.
   Multi-machine bootstrap and cross-machine discovery require multiple physical
   nodes for validation; see `docs/HIVEMIND.md` §8 Production Checklist.
4. **TLS available but not enforced by default** — `astra/rpc/tls.py` ships
   the certificate machinery; production deployments must opt in. See
   `docs/TLS.md`.
5. **No hardware CI** — GPU integration tests require a self-hosted runner.
   See [docs/TESTING.md](TESTING.md) for the pending hardware test plan.