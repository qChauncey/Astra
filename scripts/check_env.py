#!/usr/bin/env python3
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

"""
Environment readiness checker for Astra.

Validates:
  - Python version
  - Core Python dependencies (numpy, grpcio, fastapi, httpx)
  - Optional: torch + CUDA availability
  - Optional: ktransformers C++ binding
  - Optional: hivemind DHT library
  - System RAM available for CPU-side MoE weight storage
  - VRAM check (via nvidia-smi)

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --json     # machine-readable output
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
from typing import Any, Dict, List, Tuple


def _check(label: str, fn, *args) -> Tuple[bool, str]:
    try:
        detail = fn(*args)
        return True, detail
    except Exception as exc:
        return False, str(exc)


def check_python() -> Tuple[bool, str]:
    v = sys.version_info
    ok = v >= (3, 9)
    return ok, f"{v.major}.{v.minor}.{v.micro} {'✓' if ok else '✗ (need ≥3.9)'}"


def check_package(name: str, attr: str = "__version__") -> Tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, attr, "unknown")
        return True, f"{ver}"
    except ImportError:
        return False, "NOT INSTALLED"


def check_torch() -> Tuple[bool, str]:
    try:
        import torch  # type: ignore
        cuda = torch.cuda.is_available()
        devices = torch.cuda.device_count() if cuda else 0
        detail = f"{torch.__version__}  cuda={cuda}  devices={devices}"
        if cuda:
            for i in range(devices):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / 1024 ** 3
                detail += f"\n    GPU[{i}]: {props.name}  VRAM={vram_gb:.1f} GB"
        return True, detail
    except ImportError:
        return False, "NOT INSTALLED (required for GPU inference)"


def check_ktransformers() -> Tuple[bool, str]:
    try:
        import ktransformers as kt  # type: ignore
        return True, f"{getattr(kt, '__version__', 'unknown')} (C++ kernels available)"
    except ImportError:
        return False, (
            "NOT INSTALLED — using numpy stub.\n"
            "    Install: https://github.com/kvcache-ai/ktransformers"
        )


def check_hivemind() -> Tuple[bool, str]:
    try:
        import hivemind  # type: ignore
        return True, f"{hivemind.__version__} (DHT available)"
    except ImportError:
        return False, (
            "NOT INSTALLED — DHT node discovery unavailable (Phase 3).\n"
            "    Install: pip install hivemind"
        )


def check_grpc() -> Tuple[bool, str]:
    try:
        import grpc  # type: ignore
        return True, f"{grpc.__version__}"
    except ImportError:
        return False, "NOT INSTALLED — RPC layer disabled"


def check_ram() -> Tuple[bool, str]:
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    except Exception:
        pass

    # /proc/meminfo fallback
    mem_gb = None
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    mem_gb = kb / 1024 / 1024
                    break

    if mem_gb is not None:
        ok = mem_gb >= 64.0
        status = f"{mem_gb:.1f} GB {'✓' if ok else '✗ (≥64 GB recommended for 284B MoE weights)'}"
        return ok, status
    return True, "unknown (could not read /proc/meminfo)"


def check_nvme() -> Tuple[bool, str]:
    """Check NVMe for potential weight mmap from disk."""
    try:
        result = subprocess.run(
            ["df", "-h", "/"], capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) >= 2:
            return True, lines[1]
        return True, "disk info unavailable"
    except Exception as exc:
        return True, f"(disk check skipped: {exc})"


def check_nvidia_smi() -> Tuple[bool, str]:
    if shutil.which("nvidia-smi") is None:
        return False, "nvidia-smi not found (no GPU driver or no GPU)"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader"],
            timeout=5,
        ).decode().strip()
        return True, out
    except Exception as exc:
        return False, str(exc)


def check_astra_import() -> Tuple[bool, str]:
    try:
        import astra  # type: ignore
        from astra.serialization import TensorPacket, TensorSerializer
        from astra.inference import HeterogeneousEngine, DeviceMap
        from astra.routing import GeoAwareMoERouter
        from astra.rpc import InferenceServer, InferenceClient
        return True, f"astra {astra.__version__} — all submodules importable"
    except Exception as exc:
        return False, str(exc)


# ─────────────────────────────────────────────────────────────────────────── #

def run_checks() -> Dict[str, Any]:
    checks = [
        ("Python ≥3.9",         check_python),
        ("numpy",               lambda: check_package("numpy")),
        ("grpcio",              check_grpc),
        ("fastapi",             lambda: check_package("fastapi")),
        ("httpx",               lambda: check_package("httpx")),
        ("uvicorn",             lambda: check_package("uvicorn")),
        ("PyTorch + CUDA",      check_torch),
        ("KTransformers C++",   check_ktransformers),
        ("hivemind DHT",        check_hivemind),
        ("System RAM",          check_ram),
        ("Disk / NVMe",         check_nvme),
        ("GPU (nvidia-smi)",    check_nvidia_smi),
        ("astra package",       check_astra_import),
    ]

    results = {}
    for label, fn in checks:
        ok, detail = fn()
        results[label] = {"ok": ok, "detail": detail}
    return results


def print_report(results: Dict[str, Any]) -> None:
    PASS = "\033[92m PASS\033[0m"
    FAIL = "\033[91m FAIL\033[0m"
    WARN = "\033[93m WARN\033[0m"

    # Required vs optional
    required = {"Python ≥3.9", "numpy", "grpcio", "astra package"}
    optional_critical = {"PyTorch + CUDA", "KTransformers C++"}

    print("\n" + "=" * 72)
    print("  Astra Environment Check".center(72))
    print("=" * 72)
    print(f"  {'Check':<28} {'Status':<6}  Detail")
    print("-" * 72)

    all_required_ok = True
    for label, info in results.items():
        ok = info["ok"]
        detail = info["detail"].splitlines()[0]   # first line only for table
        if label in required:
            tag = PASS if ok else FAIL
            if not ok:
                all_required_ok = False
        elif label in optional_critical:
            tag = PASS if ok else WARN
        else:
            tag = PASS if ok else WARN
        print(f"  {label:<28}{tag}  {detail}")

    print("=" * 72)
    if all_required_ok:
        print("  ✓ Required dependencies satisfied — ready to run mock_pipeline.py")
    else:
        print("  ✗ Some required dependencies missing — see FAIL items above")
    print()

    # Recommendations
    recs = []
    if not results.get("PyTorch + CUDA", {}).get("ok"):
        recs.append("• Install PyTorch with CUDA: https://pytorch.org/get-started/locally/")
    if not results.get("KTransformers C++", {}).get("ok"):
        recs.append("• Build KTransformers for full GPU/CPU heterogeneous performance.")
    if not results.get("hivemind DHT", {}).get("ok"):
        recs.append("• Install hivemind for Phase 3 P2P node discovery.")
    if recs:
        print("  Recommendations:")
        for r in recs:
            print(f"    {r}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Astra environment checker")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    results = run_checks()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results)

    required = {"Python ≥3.9", "numpy", "grpcio", "astra package"}
    if not all(results[k]["ok"] for k in required if k in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
