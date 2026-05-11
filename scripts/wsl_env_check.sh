#!/usr/bin/env bash
set -euo pipefail
echo "=== OS ==="
cat /etc/os-release | head -5
echo ""
echo "=== nvcc search ==="
for d in /usr/local/cuda*/bin/nvcc /usr/lib/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
    if [ -x "$d" ]; then
        echo "FOUND: $d"
        "$d" --version | grep release
    fi
done
echo ""
echo "=== cuda-toolkit dpkg ==="
dpkg -l 2>/dev/null | grep -i cuda-toolkit || echo "no cuda-toolkit package"
echo ""
echo "=== build tools ==="
echo "gcc: $(which gcc 2>/dev/null || echo NOT_FOUND)"
echo "g++: $(which g++ 2>/dev/null || echo NOT_FOUND)"
echo "cmake: $(which cmake 2>/dev/null || echo NOT_FOUND)"
echo "git: $(which git 2>/dev/null || echo NOT_FOUND)"
echo ""
echo "=== RAM ==="
free -h | head -2
echo ""
echo "=== disk ==="
df -h / /home 2>/dev/null