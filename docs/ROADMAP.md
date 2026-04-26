# Astra — Implementation Roadmap

> Version 0.1 · April 2025 · Apache License 2.0

---

## Overview

Astra is developed in three phases, each building on the previous.  The goal of each phase is a **runnable, testable artifact** — not just design work.

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

## Phase 3 — Full P2P Network + Frontend Portal (IN PROGRESS)

**Goal:** Real multi-machine cluster with hivemind DHT discovery and a user-facing interface.

### 3.1 P2P Node Discovery

| Task | Status | Notes |
|------|--------|-------|
| Integrate `hivemind.DHT` for peer discovery | Pending | Replace mock `REGIONS` dict |
| DHT-based expert shard advertisement | Pending | Nodes publish `{expert_ids, layer_range, region}` |
| Dynamic node join/leave handling in `GeoAwareMoERouter` | Pending | Hook into DHT event callbacks |
| Engram memory node (storage-only DHT peers) | Pending | Separate from compute nodes |

### 3.2 Production Inference Engine

| Task | Status | Notes |
|------|--------|-------|
| Real KTransformers C++ binding integration | Pending | Set `ASTRA_USE_KTRANSFORMERS=1` |
| DeepSeek-V4 checkpoint loader (safetensors / GGUF) | Pending | Weight shard mapping to nodes |
| KV-cache streaming between nodes (`TransferKVCache` RPC) | Pending | Proto stub exists |
| Speculative decoding support | Pending | Draft model on single fast node |
| Continuous batching across pipeline stages | Pending | Micro-batch interleaving |

### 3.3 Geographic Micro-Cluster Optimization

| Task | Status | Notes |
|------|--------|-------|
| Real RTT measurement replacing haversine estimate | Pending | Active probe via Ping |
| Cluster-affinity grouping (nodes within N ms latency) | Pending | Refine `GeoAwareMoERouter` |
| Expert shard replication for hot experts | Pending | Frequency-based replication |
| Adaptive load balancing across nodes | Pending | Weight dispatch by utilization |

### 3.4 Security

| Task | Status | Notes |
|------|--------|-------|
| gRPC TLS + mutual certificate auth | Pending | Replace `insecure_channel` |
| Peer identity via libp2p-style key pairs | Pending | DHT node authentication |
| Weight shard integrity (SHA-256 manifest) | Pending | Prevent weight tampering |

### 3.5 Frontend Portal

| Task | Status | Notes |
|------|--------|-------|
| Next.js / Electron UI scaffold | Pending | Decentralized login |
| Real-time compute / VRAM / RTT monitoring dashboard | Pending | Pulls stats from Ping RPCs |
| Inference API endpoint (OpenAI-compatible) | Pending | FastAPI wrapper over `InferenceClient` |
| Contributor earnings / token accounting | Pending | For optional incentive layer |

---

## Dependency Upgrade Path

| Component | Current (mock) | Production target |
|-----------|---------------|-------------------|
| Tensor compute | numpy stub | KTransformers C++ + CUDA |
| Attention kernel | numpy `@` matmul | `ktransformers.ops.mla_forward` |
| DHT | in-memory dict | `hivemind.DHT` |
| Transport | insecure gRPC | gRPC TLS |
| Model weights | random arrays | DeepSeek-V4 safetensors shards |
| Memory | 16–64 GB RAM | 512 GB+ NVMe-backed mmap |

---

## Testing Strategy

| Level | Tool | Coverage target |
|-------|------|----------------|
| Unit | pytest | serialization round-trip, cache eviction, haversine |
| Integration | pytest + threading | mock_pipeline.py Phases 1 & 2 |
| E2E | pytest + real gRPC | two-process test on localhost |
| Load | locust or custom | 100 concurrent requests, measure throughput |
| Hardware CI | self-hosted runner with GPU | KTransformers kernel correctness |

---

## Known Limitations (Alpha)

1. **No real model weights** — all tensors are zero/random. Output is numerically meaningless.
2. **KTransformersStub is numpy** — ~100× slower than C++ CUDA kernels. Use for correctness testing only.
3. **No checkpoint loading** — weight sharding and loading from safetensors/GGUF is not yet implemented.
4. **DHT is mocked** — `GeoAwareMoERouter.register_node()` must be called manually; no automatic discovery.
5. **No authentication** — gRPC connections are insecure. Do not expose ports to the public internet.
