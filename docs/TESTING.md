# Astra — 测试方案

> 版本 0.2 · 2026 年 4 月 · Apache License 2.0

---

## 1. 当前测试状态（诚实评估）

### 1.1 已通过的自动化测试（430 个，可在 CI 中运行）

```
python -m pytest tests/ -v
# 430 passed, 3 skipped in ~10s（纯 CPU / NumPy 环境）
```

| 测试文件 | 覆盖范围 | 测试数 |
|---------|---------|--------|
| `test_serialization.py` | TensorPacket 序列化往返、边界情况、CRC 校验 | 14 |
| `test_shared_expert_cache.py` | LRU 淘汰、固定策略、SiLU FFN 前向计算 | 11 |
| `test_geo_router.py` | Haversine 距离、门控输出形状、地理最近节点分发 | 12 |
| `test_dht.py` | TTL 过期、订阅回调、节点注销、专家/层查询 | 14 |
| `test_pipeline_grpc.py` | gRPC 单跳/双跳/流式、Ping、CRC 校验（间接调用 HeterogeneousEngine） | 10 |
| `test_orchestrator.py` | N 节点串联、覆盖缺口检测、重试路径 | 9 |
| `test_heterogeneous.py` | `_attention_forward`、`_moe_forward`、KV cache 累积、`DeviceMap` 配置 | 23 |
| `test_kv_transfer.py` | `KVCacheSender`、`KVCacheReceiver` 重组、分块逻辑 | 20 |
| `test_api.py` | `/v1/chat/completions`（普通 + 流式）、`/health`、`/v1/pipeline/topology` | 23 |
| `test_check_env.py` | 环境检查工具逻辑验证 | 14 |
| `test_differential_privacy.py` | 隐私预算会计、噪声校准、精度退化阈值、DP 注入到推理管道 | 18 |
| `test_tee.py` | TEE 后端初始化、Gramine manifest 生成、SEV attestation、安全信道、密封密钥 | 15 |
| `test_tls.py` | mTLS 证书生成、TLSConfig 服务端/客户端、双向认证、证书过期/域名不匹配拒绝 | 15 |
| `test_hivemind_bridge.py` | hivemind DHT 桥接、节点加入/发现、多进程协调、KV 存储 CRUD、层覆盖查询 | 15 |
| `test_phase6.py` | SPA 静态文件服务、挑战-应答认证、代币账户、模型信息、设备信息、Token 速度、模式切换 | 45 |
| `test_identity.py` | PeerIdentity 密钥对生成、签名/验证、TrustRegistry 信任管理、SignedPayload 往返 | 17 |
| `test_rtt.py` | RTT 探针、PeerRTT 统计、RTTMonitor 周期采样、超时/失败降级 | 14 |
| `test_engram.py` | EngramNode 能力声明、InMemoryEngramStore CRUD、DiskEngramStore 持久化、discover_engrams 过滤 | 18 |
| `test_weight_loader.py` | ModelIndex 分片映射、WeightLoader 完整性校验、manifest 篡改检测、分片缓存 | 13 |
| `test_weight_manifest.py` | WeightManifest 创建、哈希计算、保存/加载往返、篡改检测 | 18 |
| `test_tokenizer.py` | AstraTokenizer 编码/解码、特殊 token、批处理、截断、加载工厂函数 | 23 |

### 1.2 已填补的覆盖空缺（Phase 3–8 完成）

所有在 Phase 1/2 标记为"待补充"的测试文件现已编写并纳入 CI 流水线。`test_heterogeneous.py`、`test_kv_transfer.py`、`test_api.py`、`test_differential_privacy.py`、`test_tee.py`、`test_tls.py`、`test_hivemind_bridge.py`、`test_phase6.py`、`test_identity.py`、`test_rtt.py`、`test_engram.py`、`test_weight_loader.py`、`test_weight_manifest.py`、`test_tokenizer.py` 直接测试各自模块，共计 430 个测试（Phase 1/2: 168 + Phase 3: 84 + Phase 4: 33 + Phase 5: 30 + Phase 6: 25 + Phase 7: 85 + Phase 8: 20）。

#### 剩余覆盖空缺（待补充）

| 模块 | 缺失内容 | 原因 |
|------|---------|------|
| `scripts/run_node.py` | CLI 参数解析、环境变量绑定、信号处理 | 需要集成测试环境（非纯单元测试） |
| `scripts/run_cluster.py` | 多进程启动、端口分配、清理逻辑 | 同上 |
| `astra/tee/gramine.py` | SGX 硬件功能检测、manifest 生成、quote 获取 | 需要 Intel SGX 硬件 |
| `astra/tee/amd_sev.py` | SEV-SNP 平台检测、attestation report 获取 | 需要 AMD EPYC SEV 硬件 |
| `astra/rpc/tls.py` | 真实 TLS 证书链（非自签名）、CA 签发流程 | 需要 PKI 基础设施 |
| `astra/network/hivemind_bridge.py` | 跨主机多进程 DHT、NAT 穿透 | 需要多台物理机 |

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
| TEE attestation 验证 | Intel SGX 或 AMD SEV-SNP 硬件 | 验证 quote 签名、measurement 匹配 |

---

## 2. 测试计划状态

### 2.1 `test_heterogeneous.py`（✅ 已完成）

```python
# 已实现的测试用例
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

### 2.2 `test_kv_transfer.py`（✅ 已完成）

```python
class TestKVTransfer:
    test_sender_push_empty_cache()          # 空 cache 时不抛异常，返回 True
    test_chunk_size_within_limit()          # 每块 <= MAX_CHUNK_BYTES (3 MB)
    test_receiver_reassemble_shape()        # 重组后 k.shape == 原始 k.shape
    test_receiver_dtype_preserved()         # float16 传输后 dtype 不变
    test_multi_layer_transfer()            # 3 层 KV cache 全部正确应用
    test_crc_on_chunk_corruption()         # 模拟数据损坏时抛出错误（Phase 3）
```

### 2.3 `test_api.py`（✅ 已完成，需 httpx AsyncClient）

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

### 2.4 `test_differential_privacy.py`（✅ 已完成 — Phase 4）

```python
class TestPrivacyBudget:
    test_initial_balance()                  # 初始余额 == target_epsilon
    test_consume_reduces_balance()          # 消耗后余额减少
    test_consumes_all_remaining()           # 请求超预算时消耗全部剩余
    test_depleted_returns_zero_epsilon()    # 资源耗尽后 consume() 返回 0.0
    test_delta_default_value()              # δ 默认值 1e-5

class TestMomentsAccountant:
    test_initial_rdp_summary_zero()         # 初始 (ε,δ) ≈ (0,0)
    test_single_gaussian_step()             # 单步高斯后生成合理 (ε,δ)
    test_multiple_steps_accumulate()        # 多步累积，ε 单调递增
    test_laplace_vs_gaussian_difference()   # 拉普拉斯与高斯产生不同 (ε,δ)

class TestDPController:
    test_gaussian_mechanism_adds_noise()    # 输出 ≠ 输入（但形状一致）
    test_laplace_mechanism_adds_noise()     # 拉普拉斯机制产生非零差异
    test_noise_increases_with_small_epsilon()  # ε 越小 → 噪声越大
    test_zero_epsilon_returns_zero_noise()  # ε=0 时返回零噪声（安全降级）
    test_utility_preservation()             # ε≥10 时 SNR > 10 dB（高信噪比）

class TestLayerDPInjector:
    test_inject_applies_noise()             # noise_injected ≠ original
    test_different_layers_get_different_noise()  # 不同层得到不同噪声实例
    test_budget_tracks_epsilon()            # 预算余额随层数递减
    test_shape_preserved()                  # 输出形状与输入相同
```

### 2.5 `test_tee.py`（✅ 已完成 — Phase 4）

```python
class TestTEEBackend:
    test_gramine_backend_creation()          # GramineBackend 初始化成功
    test_sev_backend_creation()              # SevBackend 初始化成功
    test_backend_is_available_stub()         # 无硬件时 is_available()=False
    test_gramine_manifest_generation()       # manifest 模板生成
    test_sev_attestation_stub()              # SEV attestation 存根
    test_secure_channel_establishment()      # 安全信道建立
    test_sealed_key_wrap_unwrap()            # 密封密钥包装/解包
```

### 2.6 `test_tls.py`（✅ 已完成 — Phase 5）

```python
class TestTLSConfig:
    test_tls_config_creation()               # TLSConfig 正确初始化
    test_certificate_generation()            # 自签名证书生成
    test_mutual_tls_server_client()          # 双向 mTLS 握手
    test_certificate_expiry_rejection()      # 过期证书被拒绝
    test_hostname_mismatch_rejection()       # 域名不匹配被拒绝
    test_ca_bundle_verification()            # CA 证书链验证
```

### 2.7 `test_hivemind_bridge.py`（✅ 已完成 — Phase 5）

```python
class TestHivemindBridge:
    test_dht_join_and_discover()             # DHT 加入与节点发现
    test_multi_process_coordination()        # 多进程协调
    test_kv_store_crud()                     # KV 存储增删查改
    test_layer_coverage_query()              # 层覆盖范围查询
    test_peer_heartbeat()                    # 心跳保活
```

### 2.8 硬件集成测试（自托管 Runner，待配置）

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
│  Layer 2: 组件集成（现有 430 个测试）            │  ← 每次 push
│  序列化 · gRPC · DHT · Orchestrator ·           │
│  HeterogeneousEngine · KVTransfer · API · DP ·  │
│  TEE · TLS · HivemindBridge · Phase6 · Phase8 · │
│  Identity · RTT · Engram · WeightLoader         │
├─────────────────────────────────────────────────┤
│  Layer 1: 纯单元测试（✅ 已完成）                │  ← 每次 push
│  HeterogeneousEngine · KVTransfer · API · DP ·  │
│  TEE · TLS · Hivemind · Phase6 · Phase8         │
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

# 运行 DP 测试（Phase 4）
python -m pytest tests/test_differential_privacy.py -v

# 运行 TEE 测试（Phase 4）
python -m pytest tests/test_tee.py -v

# 运行 TLS 测试（Phase 5）
python -m pytest tests/test_tls.py -v

# 运行 hivemind 桥接测试（Phase 5）
python -m pytest tests/test_hivemind_bridge.py -v

# 运行 Phase 6 测试
python -m pytest tests/test_phase6.py -v

# 运行 Phase 8 测试
python -m pytest tests/test_phase6.py -v -k "Phase8"

# 运行 mock pipeline 模拟（Phase 1 & 2 端到端脚本，非 pytest）
python mock_pipeline.py --phase 1 --seq-len 16 --hidden-dim 256
python mock_pipeline.py --phase 2 --seq-len 16 --hidden-dim 256