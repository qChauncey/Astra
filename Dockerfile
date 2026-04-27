# Copyright 2025 Project Astra Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ── Astra CPU-only image (numpy stub, no GPU required) ────────────────────────
#
# Build:  docker build -t astra:latest .
# Run:    docker run -p 8080:8080 astra:latest \
#             python scripts/run_node.py --mode offline --api-port 8080
#
# GPU image: use Dockerfile.gpu (adds CUDA base + torch[cuda] + ktransformers)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Build-time args
ARG ASTRA_VERSION=0.1.0-alpha
LABEL org.opencontainers.image.title="Astra"
LABEL org.opencontainers.image.description="Distributed P2P inference for DeepSeek-V4"
LABEL org.opencontainers.image.version="${ASTRA_VERSION}"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Non-root user for security
RUN groupadd -r astra && useradd -r -g astra -d /app astra

WORKDIR /app

# Install OS-level build deps (for grpcio compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy only dependency manifests first (Docker cache layer)
COPY pyproject.toml requirements.txt ./

# Install Python dependencies (CPU-only, no torch)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[proto]" && \
    pip install --no-cache-dir uvicorn[standard]

# Copy source
COPY --chown=astra:astra . .

# Switch to non-root
USER astra

# Expose gRPC + HTTP ports
EXPOSE 50050 50051 50052 50053 50054 50055 8080

# Health check via /health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=4)" \
    || exit 1

# Default: offline single-node mode with web UI
CMD ["python", "scripts/run_node.py", "--mode", "offline", "--api-port", "8080"]
