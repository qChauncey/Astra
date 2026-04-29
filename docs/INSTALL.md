# Astra — Installation Guide

> 按平台分类的详细安装说明 · Detailed per-platform installation instructions

---

## 平台支持 · Platform Support

| Component | Linux | macOS | Windows |
|-----------|:-----:|:-----:|:-------:|
| Development & testing | ✅ | ✅ | ✅ |
| API Gateway node | ✅ | ✅ | ✅ |
| DHT discovery node | ✅ | ✅ | ✅ |
| **Inference node** (real compute) | ✅ CUDA | ❌ | ⚠️ WSL2 + CUDA |
| KTransformers C++ kernel | ✅ Native | ❌ | ⚠️ WSL2 |

### 各配置能力 · What each config can do

| Config | Inference | API Gateway | DHT node | Dev / test |
|--------|:---------:|:-----------:|:--------:|:----------:|
| Linux + NVIDIA GPU | ✅ | ✅ | ✅ | ✅ |
| Linux CPU-only | ❌ | ✅ | ✅ | ✅ |
| macOS (any) | ❌ | ✅ | ✅ | ✅ |
| Windows + GPU (WSL2) | ✅ | ✅ | ✅ | ✅ |
| Windows no GPU (native) | ❌ | ✅ | ✅ | ✅ |

> **无 GPU = 无推理贡献。** NumPy 存根仅用于开发和测试。  
> **No GPU = no inference contribution.** The numpy stub produces random outputs — development and testing only.  
> 要在 Windows 上贡献算力，请使用 [WSL2 + CUDA](#windows--gpu-推理-via-wsl2)。  
> 没有 GPU 仍然可以运行 **API 网关**（接收 HTTP 请求，路由到 GPU 节点）或 **DHT 节点**（仅节点发现）。

---

<a id="one-click-windows"></a>
## 一键安装（Windows）· One-Click Install (Windows)

1. 克隆或下载此仓库 · Clone or download this repository
2. 双击 **`installer\install.bat`**（或在 PowerShell 中运行 `installer\install.ps1`）
3. 安装完成后双击 **`installer\start.bat`** 启动

启动器以**离线模式**运行 Astra（单机，所有层本地运行），并自动在 `http://localhost:8080` 打开 Web UI。

> 当前版本使用 NumPy 推理存根 — 响应为占位文本。真实 AI 输出需要 GPU + 模型权重（Phase 7）。

---

## Linux

**前提·Prerequisites:** Python 3.10+, Git

```bash
# 1. 克隆并安装 · Clone and install
git clone https://github.com/qchauncey/astra.git && cd astra
pip install -e ".[proto]"
pip install uvicorn   # Web UI 所需

# 2. 安装 CUDA Toolkit（GPU 推理必须）· Install CUDA Toolkit (required for GPU inference)
#    Ubuntu 22.04/24.04:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6 build-essential cmake git

# 3. 安装 PyTorch with CUDA · Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 4. 编译 KTransformers GPU 内核 · Build KTransformers GPU kernels
#    （每次新增推理节点时必须执行 · required for every new inference node）
bash scripts/build_ktransformers.sh

# 5. 检查环境 · Check environment
python scripts/check_env.py

# 6. GPU 离线模式 — 单机，Web UI 端口 8080
#    GPU Offline mode — single machine, Web UI on port 8080
python scripts/run_node.py --mode offline --api-port 8080

# 7. P2P 模式 — 贡献层切片到共享集群
#    Contribute a layer slice to a shared cluster
python scripts/run_node.py --node-id node-A --port 50051 \
    --layer-start 0 --layer-end 30 --hidden-dim 256 --api-port 8080

# 8. GPU 模式（需要 CUDA + KTransformers）
#    GPU mode (requires CUDA + KTransformers)
python scripts/run_node.py --node-id node-A --port 50051 \
    --layer-start 0 --layer-end 30 --gpu --api-port 8080

# 9. 单机多节点集群（无需真实 GPU 即可验证完整 P2P 管线）
#    Single-machine multi-node cluster (validates full P2P pipeline)
python scripts/run_cluster.py --nodes 3 --hidden-dim 256 --validate-only
python scripts/run_cluster.py --nodes 3 --hidden-dim 256 --api-port 8080
```

> **存根说明：** 所有推理输出目前为随机 NumPy 数组。Web UI、流式、节点发现和 gRPC 管线功能完整 — 仅缺少模型权重（Phase 7）。
> **Stub note:** All inference outputs are currently random NumPy arrays. Web UI, streaming, node discovery, and gRPC pipeline are fully functional — only model weights are missing (Phase 7).

---

## macOS

KTransformers C++ 内核需要 CUDA，macOS 不可用。Mac 节点**不能贡献推理算力**。有效角色：API 网关、DHT 发现节点、开发测试。

```bash
# 安装 Homebrew（如需）· Install Homebrew if needed (https://brew.sh)
brew install python@3.11 git

git clone https://github.com/qchauncey/astra.git && cd astra
pip3 install -e ".[proto]"

# 环境检查与本地测试 · Environment check and local testing
python scripts/check_env.py
python mock_pipeline.py --seq-len 32 --hidden-dim 256

# 角色 1：API 网关 — 接收用户请求，路由到 GPU 节点
# Role 1: API Gateway — receives user requests, routes to GPU peers
python scripts/run_node.py --node-id gateway --port 50051 --api-port 8080

# 角色 2：DHT 发现节点（无 API，无推理）
# Role 2: DHT discovery node only (no API, no inference)
python scripts/run_node.py --node-id dht-node --port 50051
```

> **Apple Silicon 说明：** MPS（Metal Performance Shaders）后端尚未集成。Mac 上的完整 GPU 推理需要未来的 MPS 适配器。欢迎贡献。

---

<a id="windows-native"></a>
## Windows — 无 GPU（原生）· No GPU (Native)

无 GPU 的 Windows 节点**不能贡献推理算力**。有效角色：API 网关、DHT 发现节点、开发测试。  
要贡献真实算力，请使用 [WSL2 + CUDA](#windows--gpu-推理-via-wsl2)。

**选项 A — 一键安装器（推荐）· One-click installer (recommended)**

1. 克隆此仓库并双击 `installer\install.bat`
2. 双击 `installer\start.bat` — 以离线模式启动并打开 Web UI

**选项 B — 手动 · Manual**

```powershell
# 从 https://python.org 安装 Python 3.10+（勾选"Add to PATH"）
# 从 https://git-scm.com 安装 Git

git clone https://github.com/qchauncey/astra.git
cd astra
pip install -e ".[proto]"
pip install uvicorn

# 环境检查
python scripts/check_env.py

# 离线模式 — 单机，所有层本地，Web UI 端口 8080
python scripts/run_node.py --mode offline --api-port 8080

# 角色 1：API 网关 — 接收用户 HTTP 请求，路由到集群中的 GPU 节点
python scripts/run_node.py --node-id gateway --port 50051 --api-port 8080

# 角色 2：DHT 发现节点（节点发现，无推理）
python scripts/run_node.py --node-id dht-node --port 50051
```

---

<a id="windows-gpu-wsl2"></a>
## Windows — GPU 推理 via WSL2 · GPU Inference via WSL2

KTransformers 需要 Linux + CUDA。在 Windows 上，WSL2 提供带 GPU 透传的完整 Linux 内核 — Astra 在 WSL2 内与原生 Linux 运行一致。

**前提 · Prerequisites**
- Windows 10 版本 21H2 或更高 / Windows 11
- NVIDIA GPU，驱动 ≥ 535（在 PowerShell 中验证：`nvidia-smi`）

**步骤 1 — 启用 WSL2** *（以管理员身份运行 PowerShell）*

```powershell
wsl --install -d Ubuntu-22.04
# 提示时重启 Windows，然后从开始菜单打开"Ubuntu 22.04"
```

**步骤 2 — 安装 NVIDIA WSL2 CUDA 驱动** *（在 Windows 主机上，不在 WSL 内）*

1. 从此处下载 WSL2 兼容的显示驱动：https://developer.nvidia.com/cuda/wsl  
2. 在 **Windows** 上作为正常驱动更新安装。  
3. **不要**在 Windows 上安装 CUDA Toolkit — 它仅在 WSL2 内存在。

**步骤 3 — 在 WSL2 Ubuntu 内安装 CUDA Toolkit**

```bash
# 在 WSL2 Ubuntu 终端内运行
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4 build-essential python3-pip git cmake

# 编译 KTransformers CUDA 内核
bash scripts/build_ktransformers.sh

# 验证 GPU 可见
nvidia-smi
```

预期输出：你的 GPU 名称、驱动版本和 CUDA 版本。

**步骤 4 — 在 WSL2 内克隆并运行 Astra**

```bash
git clone https://github.com/qchauncey/astra.git && cd astra
pip3 install -e ".[proto]"

# 环境检查
python scripts/check_env.py

# Mock pipeline（纯 CPU，健全性检查）
python mock_pipeline.py --seq-len 32 --hidden-dim 256

# 启动带 GPU 的节点
python scripts/run_node.py --node-id node-A --port 50051 \
    --layer-start 0 --layer-end 30 --gpu --api-port 8080
```

**WSL2 技巧 · Tips**

| 主题 · Topic | 详情 · Detail |
|-------|--------|
| 访问 Windows 文件 | 在 WSL2 内通过 `/mnt/c/`、`/mnt/d/` 等访问 |
| 网络 | WSL2 端口可通过 `localhost:<port>` 从 Windows 访问 — 无需额外配置 |
| GPU 驱动 | 从 Windows 主机共享；**不要**在 WSL2 内安装 GPU 驱动 |
| 多机 | 每台 Windows 机器运行自己的 WSL2；gRPC 管线与 Linux 运行一致 |
| 性能 | 相比裸机 Linux 约 3–5% 开销；对于内存受限的 MoE 工作负载可忽略不计 |