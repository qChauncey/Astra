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

"""Tests for scripts/check_env.py — environment checks and node eligibility."""

from __future__ import annotations

import sys
import os
import types
from unittest.mock import MagicMock, patch

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import check_env  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def _results(overrides: dict | None = None) -> dict:
    """Return a base results dict where all required deps are satisfied."""
    base = {
        "Python >=3.10": {"ok": True,  "detail": "3.11.0 ✓"},
        "numpy":         {"ok": True,  "detail": "1.26.0"},
        "psutil":        {"ok": True,  "detail": "5.9.0"},
        "grpcio":        {"ok": True,  "detail": "1.60.0"},
        "astra package": {"ok": True,  "detail": "astra 0.1.0-alpha"},
        "PyTorch + CUDA":{"ok": False, "detail": "NOT INSTALLED"},
        "System RAM":    {"ok": False, "detail": "16.0 GB total"},
        "Disk / NVMe":   {"ok": True,  "detail": "/ free=0.50 TB"},
    }
    if overrides:
        base.update(overrides)
    return base


# ── check_python ─────────────────────────────────────────────────────────────

class TestCheckPython:
    def test_current_version_passes(self):
        ok, detail = check_env.check_python()
        assert ok is True
        assert "3." in detail

    def test_detail_contains_version(self):
        ok, detail = check_env.check_python()
        major = sys.version_info.major
        assert str(major) in detail


# ── check_package ─────────────────────────────────────────────────────────────

class TestCheckPackage:
    def test_installed_package_returns_true(self):
        ok, detail = check_env.check_package("sys", attr="version")
        assert ok is True

    def test_missing_package_returns_false(self):
        ok, detail = check_env.check_package("_nonexistent_package_xyz")
        assert ok is False
        assert "NOT INSTALLED" in detail


# ── check_ram ────────────────────────────────────────────────────────────────

class TestCheckRam:
    def test_returns_tuple(self):
        ok, detail = check_env.check_ram()
        assert isinstance(ok, bool)
        assert isinstance(detail, str)

    def test_detail_contains_gb(self):
        _, detail = check_env.check_ram()
        assert "GB" in detail

    def test_large_ram_passes(self):
        mock_mem = MagicMock()
        mock_mem.total = 128 * 1024 ** 3
        mock_mem.available = 64 * 1024 ** 3
        with patch("psutil.virtual_memory", return_value=mock_mem):
            ok, detail = check_env.check_ram()
        assert ok is True
        assert "128" in detail

    def test_small_ram_fails(self):
        mock_mem = MagicMock()
        mock_mem.total = 8 * 1024 ** 3
        mock_mem.available = 4 * 1024 ** 3
        with patch("psutil.virtual_memory", return_value=mock_mem):
            ok, detail = check_env.check_ram()
        assert ok is False


# ── check_nvme ───────────────────────────────────────────────────────────────

class TestCheckNvme:
    def test_returns_tuple(self):
        ok, detail = check_env.check_nvme()
        assert isinstance(ok, bool)
        assert isinstance(detail, str)

    def test_ample_disk_passes(self):
        mock_usage = MagicMock()
        mock_usage.total = 2 * 1024 ** 4
        mock_usage.free  = 1 * 1024 ** 4
        with patch("psutil.disk_usage", return_value=mock_usage):
            ok, _ = check_env.check_nvme()
        assert ok is True

    def test_nearly_full_disk_fails(self):
        mock_usage = MagicMock()
        mock_usage.total = 500 * 1024 ** 3
        mock_usage.free  = 512 * 1024 ** 2   # 0.5 GB — below 1 GB threshold
        with patch("psutil.disk_usage", return_value=mock_usage):
            ok, _ = check_env.check_nvme()
        assert ok is False


# ── check_inference_eligibility ──────────────────────────────────────────────

class TestInferenceEligibility:

    # --- missing required deps → dev ----------------------------------------

    def test_missing_required_dep_returns_dev(self):
        r = _results({"numpy": {"ok": False, "detail": "NOT INSTALLED"}})
        role, reason = check_env.check_inference_eligibility(r)
        assert role == "dev"
        assert "required" in reason.lower()

    def test_all_required_missing_returns_dev(self):
        r = {k: {"ok": False, "detail": ""} for k in _results()}
        role, _ = check_env.check_inference_eligibility(r)
        assert role == "dev"

    # --- no GPU → gateway ---------------------------------------------------

    def test_no_gpu_returns_gateway(self):
        r = _results()   # PyTorch not installed, RAM too small
        role, reason = check_env.check_inference_eligibility(r)
        assert role == "gateway"
        assert "inference" in reason.lower() or "gpu" in reason.lower() or "cuda" in reason.lower()

    # --- GPU present but RAM too small → gateway ----------------------------

    def test_gpu_ok_but_ram_too_small_returns_gateway(self):
        mock_torch = types.ModuleType("torch")
        mock_torch.cuda = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        props = MagicMock()
        props.total_memory = 24 * 1024 ** 3   # 24 GB VRAM ✓
        mock_torch.cuda.get_device_properties.return_value = props

        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024 ** 3        # 16 GB RAM ✗

        mock_disk = MagicMock()
        mock_disk.free = 500 * 1024 ** 3       # 500 GB ✓

        with patch.dict("sys.modules", {"torch": mock_torch}), \
             patch("psutil.virtual_memory", return_value=mock_mem), \
             patch("psutil.disk_usage", return_value=mock_disk):
            role, reason = check_env.check_inference_eligibility(_results())
        assert role == "gateway"
        assert "RAM" in reason

    # --- all thresholds met → inference -------------------------------------

    def test_all_thresholds_met_returns_inference(self):
        mock_torch = types.ModuleType("torch")
        mock_torch.cuda = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        props = MagicMock()
        props.total_memory = 24 * 1024 ** 3   # 24 GB VRAM ✓
        mock_torch.cuda.get_device_properties.return_value = props

        mock_mem = MagicMock()
        mock_mem.total = 128 * 1024 ** 3       # 128 GB RAM ✓

        mock_disk = MagicMock()
        mock_disk.free = 500 * 1024 ** 3       # 500 GB disk ✓

        with patch.dict("sys.modules", {"torch": mock_torch}), \
             patch("psutil.virtual_memory", return_value=mock_mem), \
             patch("psutil.disk_usage", return_value=mock_disk):
            role, reason = check_env.check_inference_eligibility(_results())
        assert role == "inference"
        assert "eligible" in reason.lower()

    # --- exactly at thresholds → inference ----------------------------------

    def test_exactly_at_vram_threshold(self):
        mock_torch = types.ModuleType("torch")
        mock_torch.cuda = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        props = MagicMock()
        props.total_memory = int(check_env._MIN_VRAM_GB * 1024 ** 3)  # exactly 16 GB
        mock_torch.cuda.get_device_properties.return_value = props

        mock_mem = MagicMock()
        mock_mem.total = int(check_env._MIN_RAM_GB * 1024 ** 3)        # exactly 64 GB

        mock_disk = MagicMock()
        mock_disk.free = int(check_env._MIN_DISK_GB * 1024 ** 3)       # exactly 100 GB

        with patch.dict("sys.modules", {"torch": mock_torch}), \
             patch("psutil.virtual_memory", return_value=mock_mem), \
             patch("psutil.disk_usage", return_value=mock_disk):
            role, _ = check_env.check_inference_eligibility(_results())
        assert role == "inference"

    # --- just below thresholds → gateway ------------------------------------

    def test_just_below_vram_threshold(self):
        mock_torch = types.ModuleType("torch")
        mock_torch.cuda = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        props = MagicMock()
        props.total_memory = int((check_env._MIN_VRAM_GB - 0.1) * 1024 ** 3)  # 15.9 GB
        mock_torch.cuda.get_device_properties.return_value = props

        mock_mem = MagicMock()
        mock_mem.total = int(check_env._MIN_RAM_GB * 1024 ** 3)

        mock_disk = MagicMock()
        mock_disk.free = int(check_env._MIN_DISK_GB * 1024 ** 3)

        with patch.dict("sys.modules", {"torch": mock_torch}), \
             patch("psutil.virtual_memory", return_value=mock_mem), \
             patch("psutil.disk_usage", return_value=mock_disk):
            role, reason = check_env.check_inference_eligibility(_results())
        assert role == "gateway"
        assert "VRAM" in reason

    # --- disk too small → gateway -------------------------------------------

    def test_disk_too_small_returns_gateway(self):
        mock_torch = types.ModuleType("torch")
        mock_torch.cuda = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        props = MagicMock()
        props.total_memory = 24 * 1024 ** 3
        mock_torch.cuda.get_device_properties.return_value = props

        mock_mem = MagicMock()
        mock_mem.total = 128 * 1024 ** 3

        mock_disk = MagicMock()
        mock_disk.free = 50 * 1024 ** 3   # 50 GB — below 100 GB threshold

        with patch.dict("sys.modules", {"torch": mock_torch}), \
             patch("psutil.virtual_memory", return_value=mock_mem), \
             patch("psutil.disk_usage", return_value=mock_disk):
            role, reason = check_env.check_inference_eligibility(_results())
        assert role == "gateway"
        assert "disk" in reason.lower()

    # --- multi-GPU: pick the best card --------------------------------------

    def test_multi_gpu_uses_best_vram(self):
        mock_torch = types.ModuleType("torch")
        mock_torch.cuda = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        props_small = MagicMock()
        props_small.total_memory = 8 * 1024 ** 3    # 8 GB GPU 0
        props_large = MagicMock()
        props_large.total_memory = 24 * 1024 ** 3   # 24 GB GPU 1
        mock_torch.cuda.get_device_properties.side_effect = [props_small, props_large]

        mock_mem = MagicMock()
        mock_mem.total = 128 * 1024 ** 3

        mock_disk = MagicMock()
        mock_disk.free = 500 * 1024 ** 3

        with patch.dict("sys.modules", {"torch": mock_torch}), \
             patch("psutil.virtual_memory", return_value=mock_mem), \
             patch("psutil.disk_usage", return_value=mock_disk):
            role, _ = check_env.check_inference_eligibility(_results())
        assert role == "inference"   # 24 GB best card qualifies
