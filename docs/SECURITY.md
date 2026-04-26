# Astra — 安全与加密方案

> 版本 0.1 · 2025 年 4 月 · Apache License 2.0

---

## 1. 威胁模型

在 P2P 分布式推理场景中，参与计算的节点由不同主体（个人、机构）运营，存在以下威胁面：

| 威胁类型 | 描述 | 风险等级 |
|---------|------|---------|
| **窃听传输内容** | 攻击者监听节点间的 gRPC 流量，获取 TensorPacket | 高 |
| **隐藏状态逆向** | 节点获取经过其计算的中间隐藏状态，尝试反推原始输入 | 中-高 |
| **输出篡改** | 恶意节点修改前向计算结果，污染最终输出 | 高 |
| **身份伪造** | 伪装成合法节点加入集群，拦截或注入数据 | 高 |
| **权重投毒** | 节点加载被篡改的模型权重分片，影响推理结果 | 中 |
| **元数据泄露** | 即使内容加密，token 数量、请求频率等元数据仍可暴露用户意图 | 低-中 |

---

## 2. 传输层加密（TLS 双向认证）

### 2.1 方案

所有节点间的 gRPC 连接必须使用 **mTLS（Mutual TLS）**：

```
Client                          Server
  │── ClientHello ────────────────►│
  │◄── ServerHello + Cert ─────────│
  │── ClientCert + Finished ──────►│  ← 双向认证
  │◄── Finished ───────────────────│
  │═══════ TLS 1.3 加密通道 ════════│
  │── InferenceRequest(encrypted) ►│
  │◄── InferenceResponse(encrypted)│
```

**证书体系：**
- 集群根 CA 由项目维护者或 DAO 管理，签发节点证书
- 每个节点持有唯一的 `node_id.crt` + `node_id.key`
- 证书包含节点 ID、层范围、区域等信息，供路由层验证
- 证书有效期 90 天，支持自动轮换（ACME 协议）

**当前代码状态：**
```python
# astra/rpc/server.py — 当前（不安全，仅用于开发）
self._grpc_server.add_insecure_port(f"[::]:{port}")

# 目标（Phase 3 安全加固）
credentials = grpc.ssl_server_credentials(
    [(private_key, certificate_chain)],
    root_certificates=ca_cert,
    require_client_auth=True,   # mTLS
)
self._grpc_server.add_secure_port(f"[::]:{port}", credentials)
```

### 2.2 密码套件要求

```
TLS 1.3 强制要求，禁用 TLS 1.2 及以下
允许套件：
  TLS_AES_256_GCM_SHA384
  TLS_CHACHA20_POLY1305_SHA256
禁止：RC4、3DES、NULL、EXPORT 套件
```

---

## 3. 隐藏状态隐私保护

### 3.1 分层推理的天然隔离

在流水线并行中，每个节点**只能看到自己负责层的隐藏状态**，看不到：

```
原始输入 tokens → [嵌入层] → 隐藏状态 h₀
                              │
            Node A (L0–L20) ◄─┘  只看到 h₀（维度 7168 的浮点向量）
                              │
            Node B (L21–L40)◄─┘  只看到 h₂₀
                              │
            Node C (L41–L60)◄─┘  只看到 h₄₀
                              │
                         最终输出 logits
```

Node B 和 Node C **无法直接读取原始 prompt**，只能拿到上游节点输出的激活值。

### 3.2 模型逆向攻击风险与缓解

**风险：** 研究表明，在已知模型权重的情况下，中间层激活值可被优化算法近似逆推出输入文本（准确率随层数增加而下降）。

**缓解措施（阶段性实施）：**

| 措施 | 效果 | 实现成本 | 优先级 |
|-----|------|---------|-------|
| **激活值加噪（差分隐私）** | 向隐藏状态注入校准高斯噪声 σ，使逆推不可行，但会轻微降低精度 | 低 | Phase 3 |
| **同态加密（HE）** | 节点在密文上计算，永远看不到明文激活值 | 极高（10000× 慢，不实用） | 长期研究 |
| **可信执行环境（TEE）** | Intel SGX / AMD SEV 硬件隔离，节点代码在加密内存中运行 | 高（需特定硬件） | Phase 4+ |
| **安全多方计算（MPC）** | 激活值拆分成秘密份额分发给多个节点，无单节点能重建完整值 | 高（通信开销大） | 研究方向 |
| **节点最小权限** | 节点只接收和发送 TensorPacket，无法访问原始 token IDs | 已实现（协议设计） | ✅ 已完成 |

**差分隐私实施草案（Phase 3）：**
```python
# astra/inference/heterogeneous.py — 待添加
def _add_dp_noise(hidden: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Gaussian mechanism: sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
    sensitivity ≈ max L2 norm of one token's hidden state
    """
    sensitivity = np.linalg.norm(hidden, axis=-1).max()
    delta = 1e-5
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return hidden + np.random.normal(0, sigma, hidden.shape).astype(hidden.dtype)
```

---

## 4. 输出完整性验证（防篡改）

### 4.1 当前机制

每个 `InferenceResponse` 已包含 CRC32 校验：
```python
# astra/rpc/client.py — 已实现
received_crc = zlib.crc32(out_bytes) & 0xFFFFFFFF
if received_crc != response.output_states.crc32:
    raise ValueError(f"CRC32 mismatch ...")
```

CRC32 可检测**意外损坏**，但**无法防御主动篡改**（攻击者可同时篡改数据和 CRC）。

### 4.2 加密哈希链（待实现）

```
每个 TensorPacket 携带：
  signature = HMAC-SHA256(tensor_bytes, session_key)
  session_key 在 mTLS 握手时协商，每次推理会话独立

验证链：
  Client → Node A: 验证上游签名 → 计算本节点输出签名 → 转发
  Client 最终验证完整签名链，任何节点的篡改均可被检测
```

### 4.3 权重分片完整性

```
模型权重分片分发时携带 SHA-256 manifest：
  shard_0.safetensors → sha256: a3f2...
  shard_1.safetensors → sha256: 7c91...
  ...
manifest 由项目官方签名（Ed25519），节点启动时验证后才加载
```

---

## 5. 节点身份与准入控制

### 5.1 节点注册流程（Phase 3 目标）

```
1. 节点生成 Ed25519 密钥对
2. 向集群注册服务提交公钥 + 硬件证明（算力达标）
3. 注册服务签发节点证书（含层范围、区域、有效期）
4. 节点持证书加入 DHT，其他节点验证证书后建立 mTLS 连接
5. 定期心跳续约；证书过期或算力不达标则自动下线
```

### 5.2 Sybil 攻击防护

- 新节点需提供**工作量证明**（Proof of Compute）：在随机挑战数据上运行指定层的推理，返回结果供集群验证
- 证书绑定硬件指纹（GPU UUID + CPU ID），防止同一设备注册多个身份
- DHT 中节点记录包含信誉分，异常输出（签名校验失败、RTT 异常）会扣分并触发剔除

---

## 6. 安全实施路线图

| 阶段 | 安全措施 | 状态 |
|-----|---------|------|
| Phase 1 & 2 | CRC32 传输完整性 | ✅ 已完成 |
| Phase 1 & 2 | 节点只接收隐藏状态，不接收原始 tokens | ✅ 协议设计保证 |
| Phase 3 | gRPC mTLS 双向认证 | 📋 待实现 |
| Phase 3 | HMAC-SHA256 签名链（防主动篡改） | 📋 待实现 |
| Phase 3 | 权重分片 SHA-256 manifest + Ed25519 签名 | 📋 待实现 |
| Phase 3 | 节点证书体系 + Sybil 防护 | 📋 待实现 |
| Phase 4 | 差分隐私激活值加噪 | 📋 待研究 |
| Phase 4 | TEE（Intel SGX / AMD SEV）支持 | 📋 待评估 |
| 长期 | 同态加密 / 安全多方计算 | 🔬 研究方向 |

---

## 7. 安全披露

发现安全漏洞请通过 GitHub Security Advisory 私下披露，勿公开 issue。
