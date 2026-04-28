#!/bin/bash
# Read kt_kernel model registry and expert config from installed package

PKG_DIR="/home/chauncey/.local/lib/python3.10/site-packages/kt_kernel"

echo "=== 1. model_registry.py (first 150 lines) ==="
head -150 "${PKG_DIR}/cli/utils/model_registry.py"

echo ""
echo "=== 2. experts_base.py (first 120 lines) ==="
head -120 "${PKG_DIR}/experts_base.py"

echo ""
echo "=== 3. experts.py (first 120 lines) ==="
head -120 "${PKG_DIR}/experts.py"

echo ""
echo "=== 4. model_discovery.py (first 100 lines) ==="
head -100 "${PKG_DIR}/cli/utils/model_discovery.py"

echo ""
echo "=== 5. config/settings.py (first 100 lines) ==="
head -100 "${PKG_DIR}/cli/config/settings.py"