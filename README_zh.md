# Astra — 面向大型 MoE 模型的分布式 P2P 推理框架

<div align="right">
  <a href="README.md"><b>English</b></a> ·
  <a href="README_zh.md">中文</a>
</div>

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-389%20passed-brightgreen)]()
[![CI](https://github.com/qchauncey/astra/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Status](https://img.shields.io/badge/status-Phase%201--6%20完成%2C%20Phase%207%20进行中-blue)]()

**Astra** 是一个开源 P2P 分布式推理框架，可在普通 PC 集群（如 RTX 5070 Ti，每台 16 GB 显存）上运行大型 MoE 模型，融合了三大核心思路：

- **[Petals](https://github.com/bigscience-workshop/petals)** 式的去中心化流水线并行
- **[KTransformers](https://github.com/kvcache-ai/ktransformers)** 式的异构 GPU/CPU 计算拆分
- **[hivemind](https://github.com/learning-at-home/hivemind)** DHT 用于节点发现和键值存储

> **Alpha 阶段。** Phase 1–6 已完成并通过测试（389 通过，1 跳过，CPU/NumPy CI 全部通过）。当前验证目标：**MiniMax-M2.5**（126 GB，62 层，GQA，20 万词表）——真权加载、GQA 注意力、MoE 专家反量化及前向推理已端到端验证通过。Phase 7（KTransformers C++ 绑定、连续批处理、投机解码、专家复制）以 MiniMax-M2.5 为主要基准模型进行中。**DeepSeek-V4** 支持已规划，但需等待 KTransformers 上游完成 V4 架构适配后方可推进。

---

## 各阶段状态

| 阶段 | 范围 | 状态 |
|------|------|------|
| **Phase 1** | 本地异构单节点推理（NumPy 存根 + SharedExpertCache） | ✅ 已完成 |
| **Phase 2** | 局域网双节点 gRPC 流水线（打包 → 传输 → 计算 → 接收 循环） | ✅ 已完成 |
| **Phase 3** | 完整 P2P 网络：AstraDHT、N 节点编排、OpenAI API、权重清单、RTT 监控、节点身份、Engram 节点 | ✅ 已完成 |
| **Phase 4** | 差分隐私（ε/δ 预算、逐层噪声）、TEE（Intel SGX + AMD SEV-SNP） | ✅ 已完成 |
| **Phase 5** | gRPC TLS 双向认证 + hivemind 多机 DHT 集成 | ✅ 已完成 |
| **Phase 6** | SPA 仪表盘（聊天、监控、身份、收益）、挑战-应答登录、实时监控、代币记账 | ✅ 已完成 |
| **Phase 7** | 推理引擎（MiniMax-M2.5 验证、KTransformers C++ 绑定、连续批处理、投机解码、专家复制） | 🔧 进行中 |

> 逐项任务分解和前置条件详见 [docs/ROADMAP.md](docs/ROADMAP.md)

---

## 系统架构

```mermaid
graph TD
    User["🖥️ 用户 / OpenAI SDK"] -->|HTTP/SSE| Gateway["🌐 API 网关<br/>/v1/chat/completions"]
    Gateway -->|TensorPacket gRPC| Orch["⚙️ 流水线编排器<br/>DHT → 层覆盖率 → N 跳链路"]
    Orch --> NodeA["🖥️ 节点 A<br/>GPU: MLA<br/>CPU: MoE"]
    Orch --> NodeB["🖥️ 节点 B<br/>GPU: MLA<br/>CPU: MoE"]
    Orch --> NodeC["🖥️ 节点 C<br/>GPU: MLA<br/>CPU: MoE"]
    NodeA -->|KV-cache| NodeB
    NodeB -->|KV-cache| NodeC
    NodeA <-->|节点发现| DHT["🔗 hivemind DHT 网格<br/>节点发现 + KV 存储"]
    NodeB <-->|节点发现| DHT
    NodeC <-->|节点发现| DHT
```

**单节点计算拆分（KTransformers 模型）：** GPU 处理 MLA 注意力、RoPE、LayerNorm → 隐状态流入 CPU RAM → CPU 处理 MoE FFN（共享专家 0 和 1 固定常驻，路由专家 LRU 换页）→ TensorPacket 发往下个节点。

---

## 核心模块

### 🧠 推理引擎

| 模块 | 功能 |
|------|------|
| `astra.inference.HeterogeneousEngine` | GPU 注意力 + CPU MoE FFN 计算拆分 |
| `astra.inference.SharedExpertCache` | LRU 缓存；专家 0 和 1 永久固定 |

### 🔐 安全与隐私

| 模块 | 功能 |
|------|------|
| `astra.inference.DPController` | 差分隐私：逐层噪声注入、ε/δ 预算追踪 |
| `astra.tee.GramineBackend` | Intel SGX TEE：远程证明、模型密封 |
| `astra.tee.SevBackend` | AMD SEV-SNP：远程证明、安全模型加载 |
| `astra.rpc.TLSConfig` | mTLS 证书管理、双向认证 |

### 🗺️ 路由与编排

| 模块 | 功能 |
|------|------|
| `astra.routing.GeoAwareMoERouter` | Token 级 `(token, expert_id) → nearest_node`，基于 haversine RTT |
| `astra.network.PipelineOrchestrator` | DHT → 层覆盖率 → 防重试 N 跳链路编排 |

### 🌐 P2P 网络

| 模块 | 功能 |
|------|------|
| `astra.network.AstraDHT` | 节点发现 + 通用 KV API（兼容 hivemind） |
| `astra.network.HivemindBridge` | 多机 DHT 引导和跨机器发现 |
| `astra.network.PeerIdentity` | Ed25519 节点签名 + TOFU 密钥注册表 |
| `astra.network.EngramNode` | 仅存储 DHT 节点：KV 缓存 / 权重分片 |

### 🔌 RPC 传输

| 模块 | 功能 |
|------|------|
| `astra.rpc.InferenceServer/Client` | gRPC 流水线：打包 → CRC32 校验 → 计算 → 反序列化 |

### 🎨 API 与界面

| 模块 | 功能 |
|------|------|
| `astra.api.openai_compat` | OpenAI `/v1/chat/completions` + SSE 流式输出 |
| `astra.api.static/index.html` | SPA 仪表盘：聊天、监控、登录、收益 |

---

## 快速开始

跳转到对应平台的安装指南 → **[docs/INSTALL.md](docs/INSTALL.md)**

| 平台 | 指南章节 |
|------|----------|
| 🐧 **Linux** | [Linux 安装](docs/INSTALL.md#linux) |
| 🍎 **macOS** | [macOS 安装](docs/INSTALL.md#macos) |
| 🪟 **Windows（无 GPU）** | [Windows 原生](docs/INSTALL.md#windows-原生) |
| 🪟 **Windows + GPU（WSL2）** | [WSL2 + CUDA](docs/INSTALL.md#windows-gpu-wsl2) |
| 🚀 **Windows 一键安装器** | [一键安装](docs/INSTALL.md#一键安装windows) |

安装完成后，运行模拟管线验证环境：

```bash
# Phase 1 — 单节点异构流水线
python mock_pipeline.py --phase 1 --seq-len 16 --hidden-dim 256

# Phase 2 — 双节点 gRPC 流水线
python mock_pipeline.py --phase 2 --seq-len 16 --hidden-dim 256

# 完整测试套件（389 通过，1 跳过，仅需 CPU）
python -m pytest tests/ -v
```

---

## 项目结构

```
astra/
├── serialization/        # TensorPacket 有线格式 v1
├── inference/            # HeterogeneousEngine、SharedExpertCache、差分隐私、分词器
├── tee/                  # Intel SGX (Gramine) + AMD SEV-SNP 后端
├── routing/              # GeoAwareMoERouter（haversine RTT + gate + dispatch）
├── rpc/                  # gRPC proto、服务端/客户端、TLS、KV-cache 传输
├── network/              # AstraDHT、HivemindBridge、编排器、RTT、身份、Engram
└── api/                  # 兼容 OpenAI 的 FastAPI + SPA 仪表盘

mock_pipeline.py          # Phase 1 和 2 本地模拟框架
scripts/                  # run_node.py、run_cluster.py、check_env.py
installer/                # 一键安装器（install.bat/.ps1/.sh、start.bat）
tests/                    # 389 个 pytest 测试 + 1 跳过（CPU/NumPy CI 全部通过）
docs/                     # ARCHITECTURE、ROADMAP、TESTING、INSTALL、SECURITY 等
```

---

## 文档

| 文档 | 内容 |
|------|------|
| [docs/INSTALL.md](docs/INSTALL.md) | 各平台安装指南 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 系统设计、数据流、有线格式规范 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 分阶段计划（Phase 1–6 ✓ · Phase 7 🔧 进行中 — MiniMax-M2.5 验证） |
| [docs/TESTING.md](docs/TESTING.md) | 测试策略：389 项测试 + 硬件测试清单 |
| [docs/SECURITY.md](docs/SECURITY.md) | mTLS、差分隐私、TEE 远程证明 |
| [docs/TEE.md](docs/TEE.md) | TEE 部署：Intel SGX（Gramine）和 AMD SEV-SNP |
| [docs/TLS.md](docs/TLS.md) | mTLS 搭建和配置指南 |
| [docs/HIVEMIND.md](docs/HIVEMIND.md) | 多机 DHT 引导和运维 |
| [docs/FEASIBILITY.md](docs/FEASIBILITY.md) | 算力阈值、地理微集群、带宽分析 |
| [docs/COMPLIANCE.md](docs/COMPLIANCE.md) | 许可证合规、DeepSeek 模型条款、专利分析 |

---

## 核心创新

### 1. 地理微集群调度
通过节点物理位置（Haversine 大圆距离 + 传播延迟估算）将 MoE 专家请求路由到最近的可用节点，缓解高频 MoE 网络 I/O 的阻塞效应。

### 2. 异构计算引擎（KTransformers 集成）
- **GPU** 处理：MLA 注意力层、RoPE、LayerNorm
- **CPU/RAM** 处理：MoE 专家权重 FFN（全部 256 个专家权重常驻内存）
- 设置 `ASTRA_USE_KTRANSFORMERS=1` 激活真实 C++ 内核；默认为 NumPy 存根，支持无 GPU 环境下开发

### 3. 共享专家常驻
每个 token 都会触发共享专家（数量视模型而定，如 DeepSeek-V4 为 2 个）。将其永久固定在 GPU 显存或高速 RAM 中，消除重复的 PCIe 数据传输。

### 4. 去耦存储（Engram 记忆节点）
基于 AstraDHT（hivemind DHT 替代方案），计算节点和 Engram 存储节点完全解耦——分布式 KV 缓存和模型权重分片可独立扩容。

---

## 专利保护

本项目采用 **Apache License 2.0**。任何实体对本项目或其贡献者发起专利诉讼，将自动丧失本协议授予的所有专利权利。完整条款见 [LICENSE](LICENSE)。

---

## 许可证

采用 **Apache License 2.0**。详见 [LICENSE](LICENSE)。

融合了 [Petals](https://github.com/bigscience-workshop/petals) 和 [KTransformers](https://github.com/kvcache-ai/ktransformers) 的思路（二者均为 Apache 2.0）。所有修改详见 [NOTICE](NOTICE) 和各文件头部声明。

---

## 参与贡献

欢迎提交 PR。新文件请添加 Apache 2.0 头声明，参照 [NOTICE](NOTICE) 格式描述修改内容。