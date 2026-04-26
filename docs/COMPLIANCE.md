# Astra — 开源许可合规分析

> 版本 0.1 · 2025 年 4 月 · Apache License 2.0

---

## 1. 本项目许可证

**Astra 采用 Apache License 2.0 发布。**

Apache 2.0 的核心权利与义务：

| 权利 | 义务 |
|-----|-----|
| 自由使用、复制、分发、修改 | 分发时必须保留原始版权声明 |
| 商业用途不受限制 | 修改过的文件必须注明已被修改 |
| 专利授权（贡献者授予用户专利使用权） | 分发时必须包含 LICENSE 文件 |
| 可以不开源修改后的代码（弱 Copyleft） | 若存在 NOTICE 文件，分发时须一并包含 |

---

## 2. 直接依赖库合规性

### 2.1 Python 运行时依赖

| 库 | 许可证 | 兼容 Apache 2.0 | 使用方式 | 义务 |
|---|------|---------------|---------|-----|
| numpy | BSD-3-Clause | ✅ 兼容 | 张量计算 | 分发时保留版权声明 |
| grpcio | Apache 2.0 | ✅ 兼容 | gRPC 传输层 | 无额外义务 |
| grpcio-tools | Apache 2.0 | ✅ 兼容 | proto 编译工具（开发用） | 无额外义务 |
| protobuf | BSD-3-Clause | ✅ 兼容 | 消息序列化 | 分发时保留版权声明 |
| fastapi | MIT | ✅ 兼容 | API 网关 | 分发时保留版权声明 |
| uvicorn | BSD-3-Clause | ✅ 兼容 | ASGI 服务器 | 分发时保留版权声明 |
| httpx | BSD-3-Clause | ✅ 兼容 | HTTP 客户端 | 分发时保留版权声明 |
| pydantic | MIT | ✅ 兼容 | 数据验证 | 分发时保留版权声明 |
| starlette | BSD-3-Clause | ✅ 兼容 | FastAPI 底层 | 分发时保留版权声明 |

**结论：所有运行时依赖均与 Apache 2.0 兼容，无 GPL/LGPL 感染风险。**

### 2.2 可选依赖

| 库 | 许可证 | 兼容性 | 备注 |
|---|------|-------|-----|
| torch（PyTorch） | BSD-3-Clause | ✅ 兼容 | 可选，GPU 推理 |
| hivemind | MIT | ✅ 兼容 | 可选，Phase 3 DHT |
| ktransformers | Apache 2.0 | ✅ 兼容 | 可选，C++ 内核 |

---

## 3. 深度借鉴项目的合规处理

### 3.1 Petals（Apache 2.0）

**借鉴内容：** P2P 流水线并行的架构思路、DHT 节点发现的 API 设计参考  
**处理方式：**
- Astra 代码为**独立重写**，未直接复制 Petals 源文件
- `NOTICE` 文件中已注明借鉴关系
- 所有源文件头部包含 Astra 的 Apache 2.0 声明及修改说明

**合规状态：✅ 符合要求**

### 3.2 KTransformers（Apache 2.0）

**借鉴内容：** GPU/CPU 异构计算拆分的设计模式、共享专家固定策略的概念  
**处理方式：**
- `HeterogeneousEngine`、`SharedExpertCache` 均为独立实现
- C++ 绑定通过 `import ktransformers` 调用，作为外部库依赖，非代码复制
- `NOTICE` 文件中已注明借鉴关系

**合规状态：✅ 符合要求**

### 3.3 hivemind（MIT）

**借鉴内容：** DHT API 接口设计参考（`announce`/`get_all_peers` 等）  
**处理方式：**
- `AstraDHT` 为完整重写的纯 Python 实现
- MIT 许可证与 Apache 2.0 兼容，无需额外义务
- `NOTICE` 文件中已注明

**合规状态：✅ 符合要求**

---

## 4. DeepSeek 模型使用合规性

### 4.1 DeepSeek-V4 模型许可证

DeepSeek 系列模型使用 **DeepSeek 模型许可协议**（非标准开源许可），关键条款如下：

| 条款 | 内容 | 对 Astra 的影响 |
|-----|------|--------------|
| **使用权** | 允许研究和商业用途 | ✅ 不影响 |
| **分发限制** | 不得将模型权重作为其他模型的训练数据 | ✅ Astra 仅做推理，不涉及训练 |
| **服务限制** | 月活用户 ≥1 亿时需向 DeepSeek 申请额外授权 | ⚠️ 大规模商业部署需注意 |
| **禁止行为** | 不得用于违法、欺诈、军事武器开发等 | ✅ 不影响 |
| **归属声明** | 公开使用需注明基于 DeepSeek 模型 | 📋 需在用户文档中注明 |

### 4.2 合规行动项

- [x] `NOTICE` 文件注明 Astra 兼容 DeepSeek-V4 模型（推理框架，非权重再分发）
- [ ] **用户文档中添加**：使用 Astra 加载 DeepSeek 权重时需遵守 DeepSeek 模型许可协议
- [ ] **大规模部署前**：若节点网络月活超过 1 亿，需联系 DeepSeek 申请授权
- [ ] Astra **不分发模型权重**，用户需自行从官方渠道下载（Hugging Face / DeepSeek 官网）

### 4.3 其他可兼容模型

Astra 的推理框架本身与模型无关，下表列出其他已确认许可合规的模型：

| 模型 | 许可证 | 商业使用 | 兼容 Astra |
|-----|------|---------|-----------|
| Llama 3（Meta） | Llama 3 Community License | ✅（月活<7亿） | ✅ |
| Mistral 系列 | Apache 2.0 | ✅ 无限制 | ✅ |
| Qwen 2.5 系列 | Apache 2.0 | ✅ 无限制 | ✅ |
| Falcon 系列 | Apache 2.0 | ✅ 无限制 | ✅ |

---

## 5. 专利合规

### 5.1 Apache 2.0 专利条款

Apache 2.0 第 3 条规定：每位贡献者向用户授予**永久的、全球的、免版税的专利许可**，覆盖其贡献中必然涉及的专利权利。

同时包含**专利反击条款**：若任何一方对项目提起专利诉讼，其在 Apache 2.0 下获得的专利许可自动终止。

**含义：** Astra 贡献者受到专利保护，第三方不能用专利诉讼阻止项目的正常使用和分发。

### 5.2 潜在专利风险点

| 技术点 | 已知专利持有方 | 风险评估 |
|-------|-------------|---------|
| Transformer 注意力机制 | Google（已有部分到期） | 低（学术公开领域） |
| MoE 路由算法 | Google、多所大学 | 低（多种实现方式） |
| 流水线并行推理 | Microsoft、Google | 中（需关注具体实现） |
| KV 缓存优化 | 多家云厂商 | 中（PagedAttention 等） |

> **建议：** 在商业部署前，咨询专业知识产权律师对关键技术点做专利检索。

---

## 6. 合规检查清单

### 代码级别

- [x] 所有新建源文件包含 Apache 2.0 License Header
- [x] 修改借鉴文件时在头部注明修改内容
- [x] `NOTICE` 文件完整记录上游项目的版权信息
- [x] `LICENSE` 文件为完整的 Apache 2.0 文本
- [x] `requirements.txt` 中无 GPL/AGPL 许可证的依赖

### 分发级别

- [x] 分发时包含 `LICENSE` 文件
- [x] 分发时包含 `NOTICE` 文件
- [ ] 打包发布到 PyPI 前：确认 `pyproject.toml` 的 `license` 字段正确
- [ ] 如提供 Docker 镜像：镜像中包含所有依赖库的许可声明

### 模型使用级别

- [ ] 用户文档中告知 DeepSeek 模型许可证要求
- [ ] 明确说明 Astra 不提供、不分发模型权重
- [ ] 大规模商业部署前完成 DeepSeek 授权确认

---

## 7. 许可证兼容性总结

```
Astra (Apache 2.0)
  ├── 直接依赖 (BSD-3 / MIT / Apache 2.0)  ✅ 全部兼容
  ├── 借鉴来源 (Petals, KTransformers: Apache 2.0)  ✅ 已合规处理
  ├── 可选依赖 (PyTorch BSD-3, hivemind MIT)  ✅ 兼容
  └── 运行模型 (DeepSeek: 独立许可)  ⚠️ 用户需自行遵守
```

**结论：Astra 框架代码本身的许可合规性良好，无 GPL 感染风险。使用 DeepSeek 模型时需额外遵守其许可协议。**
