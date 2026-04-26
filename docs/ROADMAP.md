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

> 详细方案见 [docs/TESTING.md](TESTING.md)

| 层级 | 工具 | 当前状态 | 覆盖目标 |
|-----|------|---------|---------|
| 单元测试（CPU） | pytest | ✅ 70 个，全通过 | 序列化、LRU 缓存、Haversine、DHT、gRPC |
| 待补充单元测试 | pytest | ❌ 缺失 | `HeterogeneousEngine`、`KVTransfer`、OpenAI API |
| 集成测试（本地） | pytest + threading | ✅ 已覆盖 | mock_pipeline.py Phase 1 & 2 |
| 硬件集成测试 | 自托管 GPU Runner | ❌ 未配置 | KTransformers C++ 内核、真实权重数值对齐 |
| 负载测试 | locust / 自定义 | ❌ 未实现 | 100 并发请求，吞吐量与 P99 延迟 |

### 待完成测试项（Pending）

| 测试文件 | 状态 | 说明 |
|---------|------|-----|
| `tests/test_heterogeneous.py` | ❌ 待编写 | `HeterogeneousEngine` 直接单元测试 |
| `tests/test_kv_transfer.py` | ❌ 待编写 | KV 缓存分块传输与重组 |
| `tests/test_api.py` | ❌ 待编写 | OpenAI API 端点（httpx AsyncClient） |
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

1. **No real model weights** — all tensors are zero/random. Output is numerically meaningless.
2. **KTransformersStub is numpy** — ~100× slower than C++ CUDA kernels. Use for correctness testing only.
3. **No checkpoint loading** — weight sharding and loading from safetensors/GGUF is not yet implemented.
4. **DHT is mocked** — `GeoAwareMoERouter.register_node()` must be called manually; no automatic discovery.
5. **No authentication** — gRPC connections are insecure. Do not expose ports to the public internet.
6. **Test coverage gaps** — `HeterogeneousEngine`, `KVTransfer`, and API endpoints have no direct unit tests. See [docs/TESTING.md](TESTING.md) for the full pending test plan.
