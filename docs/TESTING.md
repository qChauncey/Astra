# Astra — 测试方案

> 版本 0.1 · 2025 年 4 月 · Apache License 2.0

---

## 1. 当前测试状态（诚实评估）

### 1.1 已通过的自动化测试（70 个，可在 CI 中运行）

```
python -m pytest tests/ -v
# 70 passed in ~3s（纯 CPU / NumPy 环境）
```

| 测试文件 | 覆盖范围 | 测试数 |
|---------|---------|--------|
| `test_serialization.py` | TensorPacket 序列化往返、边界情况、CRC 校验 | 14 |
| `test_shared_expert_cache.py` | LRU 淘汰、固定策略、SiLU FFN 前向计算 | 11 |
| `test_geo_router.py` | Haversine 距离、门控输出形状、地理最近节点分发 | 12 |
| `test_dht.py` | TTL 过期、订阅回调、节点注销、专家/层查询 | 14 |
| `test_pipeline_grpc.py` | gRPC 单跳/双跳/流式、Ping、CRC 校验（间接调用 HeterogeneousEngine） | 10 |
| `test_orchestrator.py` | N 节点串联、覆盖缺口检测、重试路径 | 9 |

### 1.2 明确的覆盖空缺（待补充）

| 模块 | 缺失内容 | 原因 |
|------|---------|------|
| `inference/heterogeneous.py` | `_attention_forward`、`_moe_forward`、KV cache 累积、`DeviceMap` 各配置 | 没有独立 `test_heterogeneous.py` |
| `rpc/kv_transfer.py` | `KVCacheSender`、`KVCacheReceiver` 重组、分块逻辑 | 没有对应测试文件 |
| `api/openai_compat.py` | `/v1/chat/completions`（普通 + 流式）、`/health`、`/v1/pipeline/topology` | 没有对应测试文件 |

### 1.3 无法在当前环境运行的测试（需真实硬件）

以下测试项目**不能**在 CI（无 GPU、无模型权重）中执行，必须在配备对应硬件的机器上手动或通过自托管 Runner 验证：

| 测试项 | 所需条件 | 验证方式 |
|-------|---------|---------|
| KTransformers C++ 内核正确性 | CUDA GPU + 编译好的 ktransformers | 与 NumPy 存根输出做数值对比 |
| 真实 MLA 注意力数值精度 | torch + CUDA | 与 HuggingFace 参考实现对比（atol=1e-3） |
| GPU/CPU 显存占用 | 16 GB VRAM + 64 GB RAM | `nvidia-smi` + `/proc/meminfo` 监控 |
| DeepSeek-V4 权重加载 | 权重文件（safetensors）+ 512 GB 磁盘 | 对已知 prompt 的输出做 top-1 token 一致性检查 |
| 多机 gRPC 延迟基准 | 两台物理机 + 1 Gbps 网络 | 测量 RTT、吞吐量、P99 延迟 |
| KV 缓存跨节点传输完整性 | 两节点集群 | 比对传输前后 K/V 张量的逐元素差值 |

---

## 2. 待完成测试计划（Pending）

### 2.1 `test_heterogeneous.py`（待编写）

```python
# 目标测试用例清单
class TestHeterogeneousEngine:
    test_attention_forward_shape()          # 输出形状 == 输入形状
    test_attention_forward_residual()       # 残差连接：||out - in|| > 0
    test_moe_forward_shape()               # MoE 输出形状保持不变
    test_moe_forward_shared_experts_always_fire()  # 专家 0/1 必触发
    test_kv_cache_accumulates()            # 多次前向后 cache.k.shape[0] 递增
    test_kv_cache_clear()                  # clear() 后 kv_cache 为空
    test_forward_full_packet()             # 端到端：输入 packet → 输出 packet
    test_device_map_cpu_only()             # CPU-only 模式不抛异常
    test_device_map_for_16gb_gpu()         # GPU 模式配置正确（存根下不实际调用 CUDA）
    test_engine_stats_keys()              # stats() 包含必要字段
```

### 2.2 `test_kv_transfer.py`（待编写）

```python
class TestKVTransfer:
    # 需要启动一个 InferenceServer + KVCacheSender → 验证接收端 engine 内缓存
    test_sender_push_empty_cache()          # 空 cache 时不抛异常，返回 True
    test_chunk_size_within_limit()          # 每块 <= MAX_CHUNK_BYTES (3 MB)
    test_receiver_reassemble_shape()        # 重组后 k.shape == 原始 k.shape
    test_receiver_dtype_preserved()         # float16 传输后 dtype 不变
    test_multi_layer_transfer()            # 3 层 KV cache 全部正确应用
    test_crc_on_chunk_corruption()         # 模拟数据损坏时抛出错误（Phase 3）
```

### 2.3 `test_api.py`（待编写，需 httpx AsyncClient）

```python
class TestOpenAIAPI:
    test_chat_completions_basic()           # 返回正确 JSON 结构
    test_chat_completions_streaming()       # SSE 流以 [DONE] 结束
    test_models_list()                      # /v1/models 返回 deepseek-v4-flash
    test_health_endpoint()                  # /health 返回 {"status": "ok"}
    test_topology_endpoint()               # /v1/pipeline/topology 包含 peers 字段
    test_invalid_model_graceful()          # 未知 model 不崩溃
    test_max_tokens_respected()            # 输出 token 数量 <= max_tokens
```

### 2.4 硬件集成测试（自托管 Runner，待配置）

```yaml
# .github/workflows/hardware_test.yml（待创建）
# 触发条件：手动 dispatch 或 tag push
jobs:
  gpu-integration:
    runs-on: [self-hosted, gpu]
    steps:
      - name: KTransformers kernel correctness
        # 比对 numpy stub vs C++ kernel 输出，atol=1e-2
      - name: VRAM usage within budget
        # 16 GB GPU 运行 10 层注意力，nvidia-smi 峰值显存 < 15 GB
      - name: Two-machine gRPC benchmark
        # 局域网两节点，测量 P50/P99 延迟和 token/s 吞吐
```

---

## 3. 测试分层策略

```
┌─────────────────────────────────────────────────┐
│  Layer 4: 硬件集成测试（自托管 GPU Runner）      │  ← 手动触发
│  真实 KTransformers + 真实权重 + 多机网络        │
├─────────────────────────────────────────────────┤
│  Layer 3: 端到端集成（本地双进程 gRPC）          │  ← PR 合并前
│  mock_pipeline.py Phase 1 & 2 作为 pytest 用例  │
├─────────────────────────────────────────────────┤
│  Layer 2: 组件集成（现有 70 个测试）             │  ← 每次 push
│  序列化 · gRPC · DHT · Orchestrator             │
├─────────────────────────────────────────────────┤
│  Layer 1: 纯单元测试（待补充）                   │  ← 每次 push
│  HeterogeneousEngine · KVTransfer · API          │
└─────────────────────────────────────────────────┘
```

---

## 4. 如何在本地运行现有测试

```bash
# 安装依赖
pip install -e ".[proto,dev]"

# 运行全部可在 CPU 环境执行的测试
python -m pytest tests/ -v

# 运行单个测试文件
python -m pytest tests/test_pipeline_grpc.py -v

# 运行 mock pipeline 模拟（Phase 1 & 2 端到端脚本，非 pytest）
python mock_pipeline.py --phase 1 --seq-len 16 --hidden-dim 256
python mock_pipeline.py --phase 2 --seq-len 16 --hidden-dim 256
```
