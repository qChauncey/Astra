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
KTransformers C++ Adapter — bridges Astra's numpy-based HeterogeneousEngine to
real CUDA kernels provided by the ktransformers/kt_kernel C++ extension.

Overview
--------
When ``ASTRA_USE_KTRANSFORMERS=1`` is set and the ktransformers package is
installed, this module provides:

    * :class:`KTransformersAdapter` — numpy-compatible wrapper around
      CUDA-resident attention, RMSNorm, RoPE, and matmul kernels.
    * :func:`detect_ktransformers` — probes the runtime for available kernels
      and returns readiness status.

Architecture
------------

::

    HeterogeneousEngine
          │
          └── KTransformersGPUWrapper  (dispatcher in heterogeneous.py)
                    │
                    ├── pytorch_cuda  → torch.nn.functional.scaled_dot_product_attention
                    ├── cupy          → CuPy stubs
                    ├── numpy_stub    → KTransformersStub (pure-Python)
                    └── ktransformers_cpp  → KTransformersAdapter (THIS MODULE)
                              │
                              ├── ktransformers.ops.mla_forward  (C++ CUDA)
                              ├── ktransformers.ops.rms_norm
                              ├── ktransformers.ops.rope
                              └── torch.matmul  (fallback for general matmul)

Error Handling
--------------
If ktransformers or its kernels are unavailable, the adapter reports
:attr:`KTransformersAdapter.available == False` and every method raises
``RuntimeError``.  Callers should check ``available`` before invoking methods.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("astra.inference.ktransformers")


# ------------------------------------------------------------------ #
# Runtime detection                                                    #
# ------------------------------------------------------------------ #

def detect_ktransformers() -> dict[str, Any]:
    """
    Probe the Python environment for ktransformers kernel availability.

    Returns a dict with keys:
        available : bool
            ``True`` if at least one kernel backend is usable.
        backend : str
            ``"ktransformers_cpp"``, ``"kt_kernel"``, ``"torch_fallback"``,
            or ``"unavailable"``.
        module : object or None
            The importable module (ktransformers, kt_kernel, or torch).
        mla_forward : callable or None
            The attention kernel callable (may be None if not found).
        rms_norm : callable or None
            The RMSNorm kernel callable.
        rope : callable or None
            The RoPE kernel callable.
        error : str or None
            Error message if detection failed.
    """
    result: dict[str, Any] = {
        "available": False,
        "backend": "unavailable",
        "module": None,
        "mla_forward": None,
        "rms_norm": None,
        "rope": None,
        "error": None,
    }

    # ---- Tier 1: ktransformers.ops (high-level ops module) ----
    try:
        import ktransformers  # type: ignore
        result["module"] = ktransformers

        # Probe for MLA attention kernel
        ops = getattr(ktransformers, "ops", None)
        if ops is not None:
            mla = getattr(ops, "mla_forward", None)
            if mla is not None:
                result["mla_forward"] = mla
            rms = getattr(ops, "rms_norm", None)
            if rms is not None:
                result["rms_norm"] = rms
            rope_fn = getattr(ops, "rope", None) or getattr(ops, "rope_embedding", None)
            if rope_fn is not None:
                result["rope"] = rope_fn

        # Probe for low-level registered PyTorch custom ops
        try:
            import torch
            if hasattr(torch.ops, "ktransformers"):
                kt_ops = torch.ops.ktransformers
                if result["mla_forward"] is None:
                    result["mla_forward"] = getattr(kt_ops, "mla_forward", None)
                if result["rms_norm"] is None:
                    result["rms_norm"] = getattr(kt_ops, "rms_norm", None)
                if result["rope"] is None:
                    result["rope"] = getattr(kt_ops, "rope", None) or getattr(
                        kt_ops, "rope_embedding", None
                    )
        except ImportError:
            pass

        if result["mla_forward"] is not None or result["rms_norm"] is not None:
            result["available"] = True
            result["backend"] = "ktransformers_cpp"
            logger.info(
                "KTransformersAdapter: ktransformers_cpp backend ready "
                "(mla=%s, rms=%s, rope=%s)",
                result["mla_forward"] is not None,
                result["rms_norm"] is not None,
                result["rope"] is not None,
            )
            return result

    except ImportError:
        pass

    # ---- Tier 2: kt_kernel (low-level C++ extension) ----
    try:
        import kt_kernel  # type: ignore
        result["module"] = kt_kernel
        mla = getattr(kt_kernel, "mla_forward", None)
        rms = getattr(kt_kernel, "rms_norm", None)
        rope_fn = getattr(kt_kernel, "rope", None) or getattr(
            kt_kernel, "rope_embedding", None
        )
        if mla is not None:
            result["mla_forward"] = mla
        if rms is not None:
            result["rms_norm"] = rms
        if rope_fn is not None:
            result["rope"] = rope_fn

        if mla is not None or rms is not None:
            result["available"] = True
            result["backend"] = "kt_kernel"
            logger.info(
                "KTransformersAdapter: kt_kernel backend ready "
                "(mla=%s, rms=%s, rope=%s)",
                result["mla_forward"] is not None,
                result["rms_norm"] is not None,
                result["rope"] is not None,
            )
            return result

    except ImportError:
        pass

    # ---- Tier 3: PyTorch CUDA fallback (no custom kernels) ----
    try:
        import torch
        if torch.cuda.is_available():
            result["available"] = True
            result["backend"] = "torch_fallback"
            result["module"] = torch
            logger.info(
                "KTransformersAdapter: torch_fallback backend ready "
                "(no ktransformers-specific kernels found)"
            )
            return result
    except ImportError:
        pass

    result["error"] = (
        "ktransformers, kt_kernel, and PyTorch CUDA are all unavailable. "
        "Install ktransformers per https://github.com/kvcache-ai/ktransformers "
        "or set ASTRA_USE_KTRANSFORMERS=0 to use the NumPy stub."
    )
    return result


# ------------------------------------------------------------------ #
# Adapter class                                                        #
# ------------------------------------------------------------------ #

class KTransformersAdapter:
    """
    Numpy-compatible wrapper around ktransformers CUDA kernels.

    Every method accepts numpy arrays (float16 or float32), copies them to
    GPU via PyTorch, invokes the fastest available kernel, and returns a
    numpy array of the same dtype.

    Parameters
    ----------
    probe : dict or None
        Result of :func:`detect_ktransformers`.  If ``None``, detection
        runs automatically at init time.

    Attributes
    ----------
    available : bool
        ``True`` when at least one GPU kernel backend is functional.
    backend_name : str
        ``"ktransformers_cpp"``, ``"kt_kernel"``, ``"torch_fallback"``,
        or ``"unavailable"``.
    """

    def __init__(self, probe: Optional[dict[str, Any]] = None) -> None:
        info = probe or detect_ktransformers()
        self._info = info
        self._mla_fn = info.get("mla_forward")
        self._rms_fn = info.get("rms_norm")
        self._rope_fn = info.get("rope")
        self._torch: Any = None

        try:
            import torch as _t
            self._torch = _t
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def available(self) -> bool:
        """Whether at least one GPU kernel backend is ready."""
        return bool(self._info.get("available", False))

    @property
    def backend_name(self) -> str:
        """Name of the active backend."""
        return self._info.get("backend", "unavailable")

    @property
    def has_mla(self) -> bool:
        """Whether a dedicated MLA attention kernel was found."""
        return self._mla_fn is not None

    @property
    def has_rms_norm(self) -> bool:
        """Whether a dedicated RMSNorm kernel was found."""
        return self._rms_fn is not None

    @property
    def has_rope(self) -> bool:
        """Whether a dedicated RoPE kernel was found."""
        return self._rope_fn is not None

    # ------------------------------------------------------------------ #
    # Public kernel API                                                   #
    # ------------------------------------------------------------------ #

    def multi_latent_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
        head_dim: int = 128,
    ) -> np.ndarray:
        """
        Multi-Latent Attention forward pass via ktransformers C++ kernel.

        When a dedicated ``mla_forward`` kernel is available (ktransformers_cpp
        or kt_kernel), it is called directly.  Otherwise falls back to
        PyTorch ``scaled_dot_product_attention`` (FlashAttention-2).

        Parameters
        ----------
        query : np.ndarray, shape (batch, seq, num_heads * head_dim)
        key : np.ndarray, shape (batch, cached_seq, num_kv_heads * head_dim)
        value : np.ndarray, shape (batch, cached_seq, num_kv_heads * head_dim)
        mask : np.ndarray or None
        head_dim : int

        Returns
        -------
        np.ndarray, shape (batch, seq, num_heads * head_dim)
        """
        self._require_kernel()

        if self._mla_fn is not None:
            # ---- Dedicated ktransformers MLA kernel ----
            return self._mla_forward_kt(query, key, value, mask, head_dim)
        else:
            # ---- PyTorch fallback (FlashAttention-2) ----
            return self._mla_forward_torch(query, key, value, mask, head_dim)

    def rms_layer_norm(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        RMSNorm via ktransformers kernel or PyTorch fallback.

        Parameters
        ----------
        x : np.ndarray, shape (..., hidden_dim)
        weight : np.ndarray, shape (hidden_dim,)
        eps : float

        Returns
        -------
        np.ndarray, same shape and dtype as ``x``
        """
        self._require_kernel()

        if self._rms_fn is not None:
            return self._rms_norm_kt(x, weight, eps)
        else:
            return self._rms_norm_torch(x, weight, eps)

    def rope_embedding(
        self,
        x: np.ndarray,
        position_ids: np.ndarray,
        theta: float = 10000.0,
    ) -> np.ndarray:
        """
        RoPE (Rotary Position Embedding) via ktransformers kernel or PyTorch.

        Parameters
        ----------
        x : np.ndarray, shape (seq, dim)
        position_ids : np.ndarray, shape (seq,)
        theta : float

        Returns
        -------
        np.ndarray, same shape and dtype as ``x``
        """
        self._require_kernel()

        if self._rope_fn is not None:
            return self._rope_forward_kt(x, position_ids, theta)
        else:
            return self._rope_forward_torch(x, position_ids, theta)

    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        General matrix multiply on GPU via PyTorch.

        Parameters
        ----------
        a : np.ndarray
        b : np.ndarray

        Returns
        -------
        np.ndarray
        """
        self._require_kernel()
        t = self._torch
        a_t = t.tensor(a, device="cuda", dtype=t.float32)
        b_t = t.tensor(b, device="cuda", dtype=t.float32)
        out = t.matmul(a_t, b_t)
        return out.detach().cpu().numpy().astype(a.dtype)

    # ------------------------------------------------------------------ #
    # Kernel-specific implementations                                    #
    # ------------------------------------------------------------------ #

    def _mla_forward_kt(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray],
        head_dim: int,
    ) -> np.ndarray:
        """Invoke the dedicated ktransformers MLA C++ kernel."""
        t = self._torch
        q = t.tensor(query, device="cuda", dtype=t.float16)
        k = t.tensor(key, device="cuda", dtype=t.float16)
        v = t.tensor(value, device="cuda", dtype=t.float16)

        # The ktransformers mla_forward signature varies by version.
        # Common forms:
        #   mla_forward(q, k, v, head_dim=head_dim)
        #   mla_forward(q, k, v, mask, head_dim)
        # We try keyword-first call, fall back to positional.
        try:
            out = self._mla_fn(q, k, v, head_dim=head_dim)  # type: ignore[misc]
        except TypeError:
            try:
                out = self._mla_fn(q, k, v, head_dim)  # type: ignore[misc]
            except TypeError:
                out = self._mla_fn(q, k, v)  # type: ignore[misc]

        if isinstance(out, t.Tensor):
            return out.detach().cpu().numpy().astype(query.dtype)
        return np.asarray(out, dtype=query.dtype)

    def _mla_forward_torch(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray],
        head_dim: int,
    ) -> np.ndarray:
        """Standard PyTorch scaled_dot_product_attention (FlashAttention-2)."""
        t = self._torch
        q = t.tensor(query, device="cuda", dtype=t.float16)
        k = t.tensor(key, device="cuda", dtype=t.float16)
        v = t.tensor(value, device="cuda", dtype=t.float16)

        mask_t = None
        if mask is not None:
            mask_t = t.tensor(mask, device="cuda", dtype=t.float16)

        out = t.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask_t, dropout_p=0.0, is_causal=False,
        )
        return out.detach().cpu().numpy().astype(query.dtype)

    def _rms_norm_kt(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """Invoke the dedicated ktransformers RMSNorm C++ kernel."""
        t = self._torch
        x_t = t.tensor(x, device="cuda", dtype=t.float32)
        w_t = t.tensor(weight, device="cuda", dtype=t.float32)

        try:
            out = self._rms_fn(x_t, w_t, eps)  # type: ignore[misc]
        except TypeError:
            out = self._rms_fn(x_t, w_t)  # type: ignore[misc]

        if isinstance(out, t.Tensor):
            return out.detach().cpu().numpy().astype(x.dtype)
        return np.asarray(out, dtype=x.dtype)

    def _rms_norm_torch(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """PyTorch RMSNorm fallback."""
        t = self._torch
        x_t = t.tensor(x, device="cuda", dtype=t.float32)
        w_t = t.tensor(weight, device="cuda", dtype=t.float32)
        rms = t.sqrt(t.mean(x_t ** 2, dim=-1, keepdim=True) + eps)
        out = (x_t / rms) * w_t
        return out.detach().cpu().numpy().astype(x.dtype)

    def _rope_forward_kt(
        self,
        x: np.ndarray,
        position_ids: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """Invoke the dedicated ktransformers RoPE C++ kernel."""
        t = self._torch
        x_t = t.tensor(x, device="cuda", dtype=t.float16)
        pos = t.tensor(position_ids, device="cuda", dtype=t.int64)

        try:
            out = self._rope_fn(x_t, pos, theta)  # type: ignore[misc]
        except TypeError:
            try:
                out = self._rope_fn(x_t, pos)  # type: ignore[misc]
            except TypeError:
                # kt_kernel may expect a different signature
                out = self._rope_fn(x_t, position_ids=pos.cpu().numpy())  # type: ignore[misc]

        if isinstance(out, t.Tensor):
            return out.detach().cpu().numpy().astype(x.dtype)
        return np.asarray(out, dtype=x.dtype)

    def _rope_forward_torch(
        self,
        x: np.ndarray,
        position_ids: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """PyTorch RoPE fallback."""
        t = self._torch
        x_t = t.tensor(x, device="cuda", dtype=t.float16)
        seq_len, dim = x_t.shape
        half_dim = dim // 2
        freqs = 1.0 / (
            theta
            ** (
                t.arange(0, half_dim, device="cuda", dtype=t.float32)
                / half_dim
            )
        )
        pos = t.tensor(position_ids, device="cuda", dtype=t.float32)
        angles = t.outer(pos, freqs)
        cos = t.cos(angles).to(t.float16)
        sin = t.sin(angles).to(t.float16)
        x1, x2 = x_t[..., :half_dim], x_t[..., half_dim:]
        rotated = t.cat([-x2, x1], dim=-1)
        cos_cat = t.cat([cos, cos], dim=-1)
        sin_cat = t.cat([sin, sin], dim=-1)
        out = x_t * cos_cat + rotated * sin_cat
        return out.detach().cpu().numpy().astype(x.dtype)

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _require_kernel(self) -> None:
        """Raise RuntimeError if no GPU backend is available."""
        if not self.available:
            raise RuntimeError(
                "KTransformersAdapter: no GPU backend available. "
                "Check that ktransformers is installed and CUDA is "
                "accessible, or set ASTRA_USE_KTRANSFORMERS=0."
            )

    def summary(self) -> str:
        """Human-readable summary of adapter status."""
        return (
            f"KTransformersAdapter(available={self.available}, "
            f"backend={self.backend_name}, "
            f"mla={self.has_mla}, "
            f"rms_norm={self.has_rms_norm}, "
            f"rope={self.has_rope})"
        )
