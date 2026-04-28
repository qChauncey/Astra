# Astra — Implementation Roadmap

> Version 0.2 · April 2026 · Apache License 2.0

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
| Real KTransformers C++ binding integration | → Phase 7 | Hardware-blocked (adapter layer done — torch_fallback validated on WSL2 GPU) |
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
> (486 passed, 3 skipped). Multi-machine bootstrap validation (multiple physical
> nodes) is a deployment verification step, not a software gap. See
> [docs/HIVEMIND.md](HIVEMIND.md) §8 Production Checklist for the per-node setup
> procedure.

### 5.1 gRPC Transport Security

| Task | Status | Module |
|------|--------|--------|
| Generate per-node TLS certificates (X.509 self-signed) | ✓ Done | `astra/rpc/tls.py` |
| Exchange `secure_channel` with mutual TLS credentials | ✓ Done | `astra/rpc/server.py`, `client.py` |
| Certificate pinning / TOFU trust model for P2P bootstrap | ✓ Done | `astra/rpc/tls.py` — `TofuTrustStore` with serialization |
| gRPC TLS integration tests (encrypted RPC round-trip) | ✓ Done | `tests/test_tls.py` (20 items) |

### 5.2 hivemind Multi-Machine DHT

| Task | Status | Module |
|------|--------|--------|
| Replace in-memory `AstraDHT` store with live `hivemind.DHT` | ✓ Done | `astra/network/hivemind_bridge.py` — `HivemindDHT` + `create_dht()` factory |
| Multi-machine DHT bootstrap (initial peer rendezvous) | ✓ Done | `HivemindDHT.__init__` accepts `initial_peers` multiaddr list |
| Cross-machine expert shard advertisement via DHT | ✓ Done | `HivemindDHT.announce()` publishes `expert_shards` |
| DHT-based KV cache location lookup | ✓ Done | `HivemindDHT.store()` / `fetch()` generic KV API |
| hivemind DHT integration tests (multi-node discovery) | ✓ Done | `tests/test_hivemind_bridge.py` (16 items, 15 passed + 1 skipped) |

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

## Phase 7 — Inference Performance Tuning (✓ COMPLETE — software; hardware integration pending)

**Goal:** Connect the numpy stub to real GPU hardware, validate multi-machine
P2P operation, and implement production inference optimizations (continuous
batching, speculative decoding, expert replication).

> **Current validation target: MiniMax-M2.5** (126 GB, 62 layers, GQA, 200K vocab).
> Real-weight loading, GQA attention, MoE expert dequant, and forward pass have
> been verified end-to-end. KTransformers C++ binding, continuous batching,
> speculative decoding, and expert replication are being validated against
> MiniMax-M2.5.
>
> **DeepSeek-V4** support is planned but blocked pending KTransformers upstream
> V4 architecture adaptation. Once KTransformers adds V4 MLA kernel support,
> the validation target will shift to DeepSeek-V4.

### 7.1 Soft Deliverables ✓ COMPLETE

| Task | Status | Artifact |
|------|--------|---------|
| CI workflow: `.github/workflows/hardware_test.yml` | ✓ Done | Self-hosted runner config — attach runner to activate |
| Benchmark tooling: `scripts/benchmark.py` | ✓ Done | Token/s, P50/P95/P99; single / gRPC / API modes |
| Docker Compose multi-node deployment | ✓ Done | `docker-compose.yml` — 4-service cluster (dht-seed + 3 nodes + gateway) |
| GPU Docker variant | ✓ Done | `Dockerfile.gpu` + `docker-compose.gpu.yml` — CUDA 12.4; per-node GPU device assignment |
| Load-test script: `scripts/load_test.py` | ✓ Done | asyncio+httpx concurrent driver; SLA thresholds; JSON output |
| MiniMax-M2.5 shard integrity verification | ✓ Done | `scripts/verify_minimax_m2.py` — ModelIndex + shard count validation |
| MiniMax-M2.5 end-to-end integration test | ✓ Done | `scripts/test_real_minimax_m2.py` — real-weight GQA forward pass + MoE dequant (requires local weights) |

---

### 7.2 Validation Only 🔒 Hardware-Blocked
> Code is already written. These tasks close once hardware is available and
> the existing code is executed against it. No new software required.

| Task | Nature | Prerequisite |
|------|--------|--------------|
| Register self-hosted GPU runner | Config | 1× Linux machine + CUDA GPU; runner registered in GitHub Actions Settings |
| Run `hardware_test.yml` → real-weight numerical alignment | Test | Runner above + MiniMax-M2.5 weights (126 GB, HuggingFace); DeepSeek-V4 pending KTransformers upstream V4 support |
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
**Effort:** Medium — adapter layer already built; C++ binding still pending.

| Task | Status | Notes |
|------|--------|-------|
| `KTransformersAdapter` abstraction layer (`detect_ktransformers`, MLA, RMSNorm, RoPE, matmul) | ✓ Complete | `astra/inference/ktransformers_adapter.py` — GPU-accelerated torch fallback; validated on WSL2 + NVIDIA RTX 5070 Ti |
| `HeterogeneousEngine` wired to dispatch through `KTransformersGPUWrapper` | ✓ Complete | `astra/inference/heterogeneous.py` — `KTransformersGPUWrapper` created in `_init_ktransformers()` |
| End-to-end smoke test on real GPU hardware | ✓ Complete | `scripts/smoke_kt_adapter.py` — MLA, RMSNorm, RoPE, matmul all validated (correct shapes, dtypes, no NaN) |
| Replace torch fallback with real `ktransformers.ops.mla_forward` | 🔒 Blocked | Requires `ktransformers` compiled for target CUDA arch + MiniMax-M2.5 safetensors shards |
| Handle CUDA tensor lifecycle (device placement, dtype casting) | 🔒 Blocked | Tensors must stay on GPU between attention and MoE to avoid PCIe round-trip |
| **Prerequisite** | 🔒 Blocked | `ktransformers` compiled for target CUDA arch + MiniMax-M2.5 safetensors shards; DeepSeek-V4 pending KTransformers upstream V4 MLA kernel |

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

## Phase 8 — Advanced Frontend UI (PLANNED)

**Goal:** Redesign the Astra web portal with a professional chat interface, real-time
model/device telemetry, mode switching, and comprehensive system monitoring.

### 8.1 Chat Interface

| Task | Status | Notes |
|------|--------|-------|
| Professional chat UI with streaming token animation | Planned | Build on existing SSE streaming from Phase 6 |
| Message history with markdown rendering | Planned | Syntax highlighting for code blocks |
| Prompt template library (predefined starter prompts) | Planned | Expand existing starter grid |
| Conversation branching / edit & resend | Planned | Edit previous messages and re-send |
| Token cost / credit display per message | Planned | Show compute cost per interaction |

### 8.2 Mode Switching (Offline / P2P)

| Task | Status | Notes |
|------|--------|-------|
| Toggle switch UI for offline ↔ P2P mode | Planned | Persist preference in localStorage |
| Offline mode indicator with visual distinction | Planned | Yellow badge + banner message |
| Mode-aware API routing (single-node vs. DHT orchestration) | Planned | Backend already supports mode via `create_app(mode=...)` |
| Auto-detect available peers on mode switch | Planned | Trigger DHT refresh on P2P activation |

### 8.3 Model Information Panel

| Task | Status | Notes |
|------|--------|-------|
| Model name, version, architecture display | Planned | e.g., MiniMax-M2.5, DeepSeek-V4-Flash |
| Parameter count (total / active per token) | Planned | 284B total, 12B active per token (MoE) |
| Backend engine (KTransformers / NumPy stub / PyTorch) | Planned | Detect from `KTransformersAdapter` |
| Context window size | Planned | e.g., 32K tokens |
| `/api/model-info` backend endpoint | Planned | Add to `astra/api/openai_compat.py` |

### 8.4 Token Output Speed

| Task | Status | Notes |
|------|--------|-------|
| Real-time tokens-per-second (TPS) meter | Planned | EWMA-smoothed, updated per streaming chunk |
| Latency breakdown (TTFT, inter-token latency) | Planned | Time-to-first-token + per-token timing |
| `/api/token-speed` backend endpoint | Planned | Add to `astra/api/openai_compat.py` |

### 8.5 Device Information

| Task | Status | Notes |
|------|--------|-------|
| CPU model, core count, RAM display | Planned | `psutil` / `platform` based |
| GPU model, VRAM, CUDA version display | Planned | `torch.cuda` + `nvidia-smi` info |
| OS / Python version | Planned | System info from `platform` module |
| Disk space available for model weights | Planned | `shutil.disk_usage` |
| `/api/device-info` backend endpoint | Planned | Add to `astra/api/openai_compat.py` |

### 8.6 UI Polish

| Task | Status | Notes |
|------|--------|-------|
| Responsive layout (desktop + tablet) | Planned | CSS Grid + flexbox |
| Dark theme (default) + light theme toggle | Planned | CSS custom properties; localStorage preference |
| Keyboard shortcuts (Ctrl+Enter send, Ctrl+K palette) | Planned | Global keydown listener |
| System notification on completion | Planned | Web Notification API |
| Export conversation (JSON / Markdown) | Planned | Download button in chat header |

---

## Phase 9 — Production Launch & Ecosystem (PLANNED)

**Goal:** Go from a validated multi-machine cluster to a publicly usable, economically
sustainable decentralized inference network.

> **Prerequisite:** Phase 7 hardware-blocked items resolved — at least one KTransformers
> GPU node running MiniMax-M2.5 end-to-end with continuous batching active.

### 9.1 Multi-Model Support

| Task | Status | Notes |
|------|--------|-------|
| DeepSeek-V4 MLA kernel integration | 🔒 Blocked | Pending KTransformers upstream V4 MLA kernel support |
| Per-model `DeviceMap` profiles (memory footprint, layer count, head config) | Planned | Separate profile files under `astra/inference/profiles/` |
| Model registry API (`GET /v1/models` returns all loaded checkpoints) | Planned | Extend `openai_compat.py` — already returns one model |
| Hot-swap model loading (load new checkpoint without restarting nodes) | Planned | WeightLoader + signal-based reload |

### 9.2 Economic Model & Contributor Incentives

| Task | Status | Notes |
|------|--------|-------|
| Formal tokenomics design (token issuance, burn, staking) | Planned | Currently: in-process ledger only (`/api/earnings`) |
| On-chain earnings settlement (EVM / Solana) | Planned | Requires smart contract + wallet integration |
| Contribution scoring (tokens/s, uptime SLA, geographic diversity) | Planned | Telemetry data already collected in `expert_telemetry.py` |
| Anti-sybil / stake-to-participate mechanism | Planned | Prevents free-riding nodes |

### 9.3 Operational Hardening

| Task | Status | Notes |
|------|--------|-------|
| API rate limiting & quota enforcement | Planned | `openai_compat.py` currently has no rate limiting |
| Graceful node drain (SIGTERM → finish in-flight requests → leave DHT) | Planned | `run_node.py` handles SIGTERM but does not drain |
| Automatic TLS certificate rotation | Planned | `TofuTrustStore` pins certs; rotation needs out-of-band signaling |
| Structured logging + OpenTelemetry traces | Planned | Currently uses Python `logging`; no trace propagation across nodes |
| Prometheus metrics endpoint (`/metrics`) | Planned | Stats available via `engine.stats()` but not exported |

### 9.4 Compliance & Privacy

| Task | Status | Notes |
|------|--------|-------|
| GDPR data-subject request handling (prompt deletion) | Planned | No user-data retention layer currently |
| Configurable DP budget per API key | Planned | `DPController` exists; budget is global, not per-user |
| Audit log for TEE attestation events | Planned | `GramineBackend.attest()` returns report; no persistent log |

---

## Dependency Upgrade Path

| Component | Current (mock) | Production target |
|-----------|---------------|-------------------|
| Tensor compute | numpy stub / torch_fallback (GPU-accelerated via adapter) | KTransformers C++ + CUDA |
| Attention kernel | numpy `@` matmul / PyTorch GPU (via `KTransformersAdapter`) | `ktransformers.ops.mla_forward` |
| DHT | in-memory dict / `HivemindDHT` (Phase 5 done) | `hivemind.DHT` |
| Transport | gRPC | gRPC with mTLS (done — Phase 5) ✅ |
| Model weights | random arrays | MiniMax-M2.5 safetensors shards (primary); DeepSeek-V4 pending KTransformers upstream V4 adaptation |
| Memory | 16–64 GB RAM | 512 GB+ NVMe-backed mmap |

---

## Testing Strategy

> See [docs/TESTING.md](TESTING.md) for detailed plan

| Layer | Tool | Current Status | Coverage Target |
|-----|------|---------|---------|
| Unit (CPU) | pytest | ✅ 486 passed + 3 skipped | Serialization, LRU cache, Haversine + real RTT, DHT, Engram, Peer Identity, Weight Manifest, gRPC TLS, HeterogeneousEngine, Tokenizer, KVTransfer, OpenAI API, Phase 6 dashboard, Continuous Batching, Speculative Decoding, Expert Replication, Weight Loader |
| Integration (local) | pytest + threading | ✅ Covered | mock_pipeline.py Phase 1 & 2 |
| Hardware Integration | Self-hosted GPU Runner | ❌ Not configured | KTransformers C++ kernels, real-weight numerical alignment |
| Load Test | scripts/load_test.py (asyncio+httpx) | ✅ Implemented | 100 concurrent requests, throughput & P99 latency |

### Pending Test Items

| Test File | Status | Notes |
|---------|------|-----|
| `tests/test_heterogeneous.py` | ✅ Done | `HeterogeneousEngine` direct unit tests (23 items) |
| `tests/test_kv_transfer.py` | ✅ Done | KV-cache chunked transfer & reassembly (15 items) |
| `tests/test_api.py` | ✅ Done | OpenAI API endpoints (httpx AsyncClient) (22 items) |
| `.github/workflows/hardware_test.yml` | ✅ Done | Self-hosted GPU Runner CI config — attach runner to activate |

---

## 配套文档

| 文档 | 内容 |
|-----|-----|
| [docs/ARCHITECTURE.md](ARCHITECTURE.md) | 系统架构、设计决策、模块依赖图、核心挑战 |
| [docs/INSTALL.md](INSTALL.md) | 各平台安装指南（Linux / macOS / Windows WSL2） |
| [docs/TESTING.md](TESTING.md) | 完整测试方案，含待完成项与硬件测试要求 |
| [docs/SECURITY.md](SECURITY.md) | 加密方案、威胁模型、差分隐私、mTLS 实施路线 |
| [docs/FEASIBILITY.md](FEASIBILITY.md) | 算力门槛、地理微集群划分、带宽需求、风险分析 |
| [docs/COMPLIANCE.md](docs/COMPLIANCE.md) | 许可证合规、DeepSeek 模型使用条款、专利分析 |
| [docs/TEE.md](docs/TEE.md) | TEE 部署指南（Intel SGX + AMD SEV-SNP 硬件要求、manifest 生成） |
| [docs/TLS.md](docs/TLS.md) | gRPC mTLS 证书生成与分发流程 |
| [docs/HIVEMIND.md](docs/HIVEMIND.md) | hivemind DHT 配置、Bootstrap peer 设置、NAT 穿透 |

---

## Known Limitations (Alpha)

1. **MiniMax-M2.5 validated; DeepSeek-V4 pending** — Real-weight loading, GQA attention, MoE expert dequant, and forward pass have been verified end-to-end with MiniMax-M2.5 (126 GB, 62 layers). Phase 7 software optimizations (continuous batching, speculative decoding, expert replication, weight loader, tokenizer, weight manifest) are complete on CPU; KTransformers C++ binding integration is blocked pending hardware. DeepSeek-V4 support is planned but blocked pending KTransformers upstream V4 architecture adaptation.
2. **KTransformersStub is numpy / torch_fallback** — The `KTransformersAdapter` now provides GPU-accelerated torch fallback when PyTorch + CUDA are available (validated on WSL2 + NVIDIA RTX 5070 Ti). This is faster than pure numpy but still ~10–20× slower than KTransformers C++ CUDA kernels. Use for correctness testing only.
3. **DHT bridge ready, multi-machine validation pending** — The hivemind DHT
   bridge (`astra/network/hivemind_bridge.py`) is fully implemented and tested.
   Multi-machine bootstrap and cross-machine discovery require multiple physical
   nodes for validation; see `docs/HIVEMIND.md` §8 Production Checklist.
4. **TLS available but not enforced by default** — `astra/rpc/tls.py` ships
   the certificate machinery; production deployments must opt in. See
   `docs/TLS.md`.
5. **No hardware CI** — GPU integration tests require a self-hosted runner.
   See [docs/TESTING.md](TESTING.md) for the pending hardware test plan.
6. **No rate limiting** — The OpenAI-compatible API has no per-key quota enforcement. Phase 9.3 covers this.
7. **Earnings ledger is in-process only** — Token accounting resets on restart; on-chain settlement is a Phase 9.2 item.