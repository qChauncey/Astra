# Astra — System Architecture

> Version 0.2 · April 2026 · Apache License 2.0

---

## 1. Vision & Objectives

Astra enables **distributed inference of large MoE transformers** on clusters of commodity PCs equipped with consumer GPUs (e.g. RTX 5070 Ti, 16 GB VRAM).

**Current validation target:** MiniMax-M2.5 (126 GB, 62 layers, GQA, 200K vocab) — real-weight loading and forward pass verified end-to-end.
**Planned target:** DeepSeek-V4 (284B parameters, MLA attention) — blocked pending KTransformers upstream V4 MLA kernel support.

Astra achieves distributed inference by:

1. **Pipeline parallelism** — slicing the 61-layer transformer across multiple nodes over gRPC.
2. **Heterogeneous compute** — routing Attention to GPU, MoE FFN experts to CPU RAM, per the KTransformers model.
3. **Geographic micro-clustering** — grouping physically close nodes to minimize MoE network I/O latency.
4. **P2P DHT discovery** — using hivemind to eliminate centralized coordination.

---

## 2. Core Challenges & Design Responses

### 2.1 MoE I/O Bottleneck

DeepSeek-V4 uses 256 routed experts + 2 shared experts per MoE layer, with top-8 routing per token. At bf16, the expert weights alone require ~568 GB, far exceeding any single GPU. Astra's response:

- **CPU RAM offload**: Expert weight tensors live in system RAM. KTransformers C++ kernels execute FFN passes directly on CPU without GPU round-trips.
- **Shared Expert Pinning**: Experts 0 and 1 (shared, fire on every token) are pinned in the fastest available memory (GPU VRAM or locked RAM) and never evicted from `SharedExpertCache`.
- **Geo-Aware Dispatch**: The `GeoAwareMoERouter` prefers nodes in the same geographic cluster to minimise PCIe/NIC latency for expert weight transfer.

### 2.2 Pipeline Latency vs Throughput

Naive pipeline parallelism stalls on the slowest stage. Astra mitigates this via:

- **Micro-batch streaming** (`RunLayerStream`): a bidirectional gRPC stream allows overlapping compute and transfer across stages.
- **KV-cache transfer** (Phase 3): partial KV caches are streamed peer-to-peer rather than recomputed.

### 2.3 Heterogeneous Node Capabilities

Real-world P2P clusters contain nodes with wildly different hardware. `DeviceMap` encodes per-node capabilities:

```python
DeviceMap(
    attention_on_gpu = True,   # GPU handles MLA + RoPE + norm
    moe_on_cpu       = True,   # MoE FFN on CPU RAM
    gpu_device       = "cuda:0",
    num_layers       = 61,
    hidden_dim       = 7168,
)
```

Nodes without GPUs can still contribute CPU-only MoE compute.

---

## 3. Module Architecture

### 3.1 Serialization Layer (`astra.serialization`)

```
TensorPacket
 ├── tensor          np.ndarray (seq_len, hidden_dim), float16
 ├── layer_start / layer_end
 ├── token_ids       list[int]   — for KV-cache alignment
 ├── selected_experts np.ndarray (seq_len, top_k)
 ├── geo_region      str
 └── metadata        dict[str, str]

TensorSerializer
 ├── serialize(pkt)  → bytes   (Astra binary wire format v1)
 └── deserialize(b)  → TensorPacket
```

Wire format design goals:
- Zero-copy path for large tensors (raw numpy buffer, no base64)
- CRC32 field validated by `InferenceClient` on receipt
- Extensible via `metadata` map without protocol version bump

### 3.2 RPC Layer (`astra.rpc`)

```
inference.proto
 └── service InferenceService
       ├── RunLayer(InferenceRequest)        → InferenceResponse
       ├── RunLayerStream(stream …)          → stream …
       ├── Ping(PingRequest)                 → PingResponse
       └── TransferKVCache(stream KVCacheChunk) → PingResponse  [Phase 3]

InferenceServer
 └── _InferenceServicer
       ├── RunLayer  → deserialize → engine.forward → serialize → response
       └── Ping      → capability advertisement

InferenceClient
 ├── run_layer(packet, layer_start, layer_end) → TensorPacket
 ├── run_layer_stream([packets], …)           → list[TensorPacket]
 └── ping()                                   → dict
```

gRPC was chosen over REST/FastAPI for the hot path because:
- Native binary framing avoids HTTP/JSON overhead
- Bidirectional streaming supports micro-batch pipelining
- Built-in deadline propagation for latency SLOs

### 3.3 Inference Engine (`astra.inference`)

```
HeterogeneousEngine
 ├── _attention_forward(hidden, layer_idx)   → GPU path
 │    └── KTransformersStub.multi_latent_attention (numpy stub in dev)
 ├── _moe_forward(hidden, selected_experts)  → CPU path
 │    └── SharedExpertCache.forward(expert_id, hidden)
 └── forward(packet, layer_indices)          → TensorPacket

SharedExpertCache
 ├── pin(expert_id, weights)   — permanent, never evicted
 ├── load(expert_id, weights)  — LRU managed
 └── forward(expert_id, hidden_states) → np.ndarray
      └── SiLU-gated MLP: down(silu(gate(x)) * up(x))
```

**KTransformers integration path:**

```
ASTRA_USE_KTRANSFORMERS=0 (default)
  → KTransformersStub  (pure numpy, CPU)
  → runs on any machine, useful for development and CI

ASTRA_USE_KTRANSFORMERS=1
  → import ktransformers as _kt
  → _kt.ops.mla_forward()      # GPU CUDA kernel
  → _kt.ops.rms_norm()
  → Expert FFN still runs on CPU RAM via KTransformers CPU offload
```

### 3.4 Routing Layer (`astra.routing`)

```
GeoAwareMoERouter
 ├── register_node(NodeInfo)
 ├── gate(hidden_states, layer_idx)  → selected_experts (seq, top_k)
 ├── dispatch(selected_experts)      → DispatchPlan
 └── route(packet, layer_idx)        → (packet, DispatchPlan)

NodeInfo
 ├── node_id, region: GeoRegion
 ├── layer_start / layer_end
 └── expert_shards: list[int]

GeoRegion
 ├── lat, lon  (degrees)
 ├── distance_km(other)  — haversine
 └── rtt_ms(other)       — 2×propagation + base overhead
```

Token-level dispatch flow:

```
For each token t in sequence:
  For each k in top_k:
    expert_id = selected_experts[t, k]
    if expert_id < 2:           ← shared expert, always local
      run on local SharedExpertCache
    else:
      candidates = nodes that host expert_id and are available
      best_node  = argmin(local_region.rtt_ms(node.region))
      dispatch (t, expert_id) → best_node
```

---

## 4. Data Flow: Full Pipeline

```
[User tokens]
     │
     ▼
TensorPacket.make_input(token_ids)
     │  seq_len=S, hidden_dim=7168, dtype=float16
     ▼
GeoAwareMoERouter.gate()        ← MoE gate logits
     │  selected_experts: (S, 8)
     ▼
TensorSerializer.serialize()    ← Astra wire format
     │  ~S × 7168 × 2 bytes  +  header
     ▼
InferenceClient.run_layer()     ← gRPC RunLayer
     │  → Node A (layers 0–20)
     │
     ▼  [Node A]
InferenceServer._servicer.RunLayer()
  ├─ attention_forward()  [GPU: MLA + RoPE]
  └─ moe_forward()        [CPU: expert FFN × top_k]
     │
     ▼
TensorSerializer.serialize()    ← forward result
     │
     ▼
InferenceClient.run_layer()     ← gRPC RunLayer
     │  → Node B (layers 21–40)  ...
     │
     ▼
[Final TensorPacket: logits or next-layer hidden states]
```

---

## 5. Wire Format Specification (v1)

```
Offset  Field             Type     Size
------  -----             ----     ----
0       Magic             bytes    4    "ASTR"
4       Version           uint32   4    1
8       Header length     uint32   4
12      Header            JSON     var  {packet_id, layer_*, token_ids, geo_region, ...}
12+H    Tensor ndim       uint8    1
13+H    Tensor shape[i]   uint64   8×ndim
..      Dtype length      uint8    1
..      Dtype string      bytes    var  e.g. "float16"
..      Tensor byte len   uint64   8
..      Tensor data       bytes    var  raw numpy buffer, little-endian
..      Has experts flag  uint8    1
..      [Expert tensor — same layout as tensor section, if flag=1]
```

---

## 6. Security & Trust Model (Phase 3 planning)

- **Transport**: gRPC TLS with mutual certificate authentication between trusted peers.
- **Authorization**: hivemind DHT namespace scoping; only nodes with valid peer IDs can join a shard group.
- **Integrity**: CRC32 on every response; SHA-256 manifest for model weight shards.
- **No arbitrary code execution**: nodes execute only pre-defined operator graphs; no model weights contain executable code.

---

## 7. Scalability Targets

| Cluster size | Model | Target throughput |
|---|---|---|
| 2 nodes × 16 GB GPU | DeepSeek-V4-Flash 284B | ~2 tok/s (draft) |
| 4 nodes × 16 GB GPU | DeepSeek-V4-Flash 284B | ~5 tok/s |
| 8 nodes × 24 GB GPU | DeepSeek-V4-Flash 284B | ~12 tok/s |

Numbers are estimates based on KTransformers benchmarks and Petals measurements. Actual performance depends on inter-node bandwidth and CPU RAM speed.
