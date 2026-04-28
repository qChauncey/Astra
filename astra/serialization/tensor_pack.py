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
#
# MODIFICATIONS (Astra project):
#   - Designed from scratch for model-agnostic token-level distribution.
#   - Supports KTransformers-compatible float16/bfloat16 wire format.
#   - Adds geographic routing metadata fields for micro-cluster dispatch.
#   - Model constants are now sourced from astra.config.model_config.

"""
Task A: Tensor serialization and packaging for token-level P2P distribution.

TensorPacket is the fundamental unit of data transferred between Astra nodes.
It wraps hidden-state tensors with routing metadata so the MoE router can make
token-level dispatch decisions at each pipeline stage.

Wire format (little-endian binary):
  [4B magic] [4B version] [8B packet_id] [header_len:4B] [header JSON]
  [tensor_bytes_len:8B] [tensor_bytes]

KTransformers compatibility: tensors are serialized as raw numpy buffers in
float16 (default) or bfloat16 to match ktransformers C++ kernel expectations.
"""

from __future__ import annotations

import io
import json
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from astra.config.model_config import get_model_config, ModelConfig

_MAGIC = b"ASTR"
_VERSION = 1
_HEADER_STRUCT = struct.Struct("<4sI")   # magic(4) + version(4)
_LEN4 = struct.Struct("<I")
_LEN8 = struct.Struct("<Q")

# Backward-compatible aliases (deprecated — use get_model_config() instead)
def _get_default_cfg() -> ModelConfig:
    return get_model_config()

DEEPSEEK_V4_NUM_LAYERS = _get_default_cfg().num_layers
DEEPSEEK_V4_HIDDEN_DIM = _get_default_cfg().hidden_dim
DEEPSEEK_V4_NUM_EXPERTS = _get_default_cfg().num_local_experts
DEEPSEEK_V4_TOP_K_EXPERTS = _get_default_cfg().num_experts_per_tok
DEEPSEEK_V4_SHARED_EXPERTS = _get_default_cfg().num_shared_experts


@dataclass
class TensorPacket:
    """
    Serializable unit carrying hidden states between pipeline stages.

    Attributes:
        packet_id:   Unique request identifier (UUIDv4 hex).
        tensor:      Hidden state array, shape (seq_len, hidden_dim) or
                     (batch, seq_len, hidden_dim).  dtype float16 / bfloat16.
        layer_start: First transformer layer index this packet covers.
        layer_end:   Last transformer layer index (exclusive).
        token_ids:   Original token positions in the full sequence (for KV-cache
                     alignment across nodes).
        selected_experts: Per-token expert indices selected by the MoE gate,
                     shape (seq_len, top_k).  -1 means "unrouted / shared".
        geo_region:  Geographic cluster tag of the originating node.
        src_node:    Originating node peer-id string.
        dst_node:    Target node peer-id string (empty = broadcast to cluster).
        timestamp:   Unix timestamp (float) when this packet was created.
        metadata:    Arbitrary string→string map for extensibility.
    """

    packet_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    tensor: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float16))
    layer_start: int = 0
    layer_end: int = 0
    token_ids: List[int] = field(default_factory=list)
    selected_experts: Optional[np.ndarray] = None   # shape (seq_len, top_k)
    geo_region: str = "default"
    src_node: str = ""
    dst_node: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Derived properties                                                    #
    # ------------------------------------------------------------------ #

    @property
    def seq_len(self) -> int:
        return self.tensor.shape[-2] if self.tensor.ndim >= 2 else 0

    @property
    def hidden_dim(self) -> int:
        return self.tensor.shape[-1] if self.tensor.ndim >= 1 else 0

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start

    def byte_size(self) -> int:
        """Approximate wire size in bytes (tensor only)."""
        size = self.tensor.nbytes
        if self.selected_experts is not None:
            size += self.selected_experts.nbytes
        return size

    # ------------------------------------------------------------------ #
    # Factory helpers                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def make_input(
        cls,
        token_ids: List[int],
        hidden_dim: Optional[int] = None,
        model_id: Optional[str] = None,
        dtype: np.dtype = np.float16,
        geo_region: str = "default",
        src_node: str = "",
    ) -> "TensorPacket":
        """Create an initial embedding-level packet from raw token IDs.

        If *hidden_dim* is not provided, it is resolved from the current
        default model configuration (see ``astra.config.model_config``).

        *model_id* can be used to override the default model.
        """
        if hidden_dim is None:
            hidden_dim = get_model_config(model_id).hidden_dim
        seq_len = len(token_ids)
        # In production: embed tokens via embedding table.  Here we use zeros
        # as a stand-in that keeps the tensor shape contract intact.
        tensor = np.zeros((seq_len, hidden_dim), dtype=dtype)
        return cls(
            tensor=tensor,
            layer_start=0,
            layer_end=0,
            token_ids=token_ids,
            geo_region=geo_region,
            src_node=src_node,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "TensorPacket":
        return TensorSerializer.deserialize(data)

    def to_bytes(self) -> bytes:
        return TensorSerializer.serialize(self)

    def split_by_expert(
        self, expert_id: int
    ) -> Tuple["TensorPacket", np.ndarray]:
        """
        Extract tokens routed to a specific expert.

        Returns a sub-packet containing only the tokens that selected
        `expert_id`, plus the original row indices for scatter-back.
        """
        if self.selected_experts is None:
            raise ValueError("selected_experts not set; run MoE gate first")
        mask = np.any(self.selected_experts == expert_id, axis=-1)
        indices = np.where(mask)[0]
        sub = TensorPacket(
            packet_id=self.packet_id,
            tensor=self.tensor[indices],
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            token_ids=[self.token_ids[i] for i in indices],
            selected_experts=self.selected_experts[indices],
            geo_region=self.geo_region,
            src_node=self.src_node,
            dst_node=self.dst_node,
            timestamp=self.timestamp,
            metadata=dict(self.metadata, expert_id=str(expert_id)),
        )
        return sub, indices

    def __repr__(self) -> str:
        return (
            f"TensorPacket(id={self.packet_id[:8]}…, "
            f"layers={self.layer_start}:{self.layer_end}, "
            f"shape={self.tensor.shape}, dtype={self.tensor.dtype}, "
            f"region={self.geo_region})"
        )


class TensorSerializer:
    """
    Binary serializer for TensorPacket.

    Wire format:
        [magic:4B "ASTR"] [version:4B] [header_len:4B] [header:JSON bytes]
        [tensor_ndim:1B] [shape:ndim×8B] [dtype_len:1B] [dtype:str]
        [tensor_bytes_len:8B] [tensor_bytes]
        [has_experts:1B] [experts_bytes_len:8B?] [experts_bytes?]
    """

    @staticmethod
    def serialize(pkt: TensorPacket) -> bytes:
        buf = io.BytesIO()

        # -- header --
        header = {
            "packet_id": pkt.packet_id,
            "layer_start": pkt.layer_start,
            "layer_end": pkt.layer_end,
            "token_ids": pkt.token_ids,
            "geo_region": pkt.geo_region,
            "src_node": pkt.src_node,
            "dst_node": pkt.dst_node,
            "timestamp": pkt.timestamp,
            "metadata": pkt.metadata,
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode()

        buf.write(_HEADER_STRUCT.pack(_MAGIC, _VERSION))
        buf.write(_LEN4.pack(len(header_bytes)))
        buf.write(header_bytes)

        # -- hidden-state tensor --
        tensor = np.ascontiguousarray(pkt.tensor)
        dtype_str = str(tensor.dtype).encode()
        buf.write(bytes([tensor.ndim]))
        for s in tensor.shape:
            buf.write(_LEN8.pack(s))
        buf.write(bytes([len(dtype_str)]))
        buf.write(dtype_str)
        tensor_bytes = tensor.tobytes()
        buf.write(_LEN8.pack(len(tensor_bytes)))
        buf.write(tensor_bytes)

        # -- optional expert routing tensor --
        if pkt.selected_experts is not None:
            experts = np.ascontiguousarray(pkt.selected_experts)
            exp_bytes = experts.tobytes()
            dtype_e = str(experts.dtype).encode()
            buf.write(bytes([1]))  # has_experts flag
            buf.write(bytes([experts.ndim]))
            for s in experts.shape:
                buf.write(_LEN8.pack(s))
            buf.write(bytes([len(dtype_e)]))
            buf.write(dtype_e)
            buf.write(_LEN8.pack(len(exp_bytes)))
            buf.write(exp_bytes)
        else:
            buf.write(bytes([0]))

        return buf.getvalue()

    @staticmethod
    def deserialize(data: bytes) -> TensorPacket:
        buf = io.BytesIO(data)

        magic, version = _HEADER_STRUCT.unpack(buf.read(_HEADER_STRUCT.size))
        if magic != _MAGIC:
            raise ValueError(f"Bad magic bytes: {magic!r}")
        if version != _VERSION:
            raise ValueError(f"Unsupported packet version: {version}")

        header_len = _LEN4.unpack(buf.read(4))[0]
        header = json.loads(buf.read(header_len).decode())

        # -- tensor --
        ndim = buf.read(1)[0]
        shape = tuple(_LEN8.unpack(buf.read(8))[0] for _ in range(ndim))
        dtype_len = buf.read(1)[0]
        dtype = np.dtype(buf.read(dtype_len).decode())
        tensor_len = _LEN8.unpack(buf.read(8))[0]
        tensor = np.frombuffer(buf.read(tensor_len), dtype=dtype).reshape(shape)

        # -- experts --
        has_experts = buf.read(1)[0]
        selected_experts = None
        if has_experts:
            e_ndim = buf.read(1)[0]
            e_shape = tuple(_LEN8.unpack(buf.read(8))[0] for _ in range(e_ndim))
            e_dtype_len = buf.read(1)[0]
            e_dtype = np.dtype(buf.read(e_dtype_len).decode())
            e_len = _LEN8.unpack(buf.read(8))[0]
            selected_experts = np.frombuffer(buf.read(e_len), dtype=e_dtype).reshape(e_shape)

        return TensorPacket(
            packet_id=header["packet_id"],
            tensor=tensor,
            layer_start=header["layer_start"],
            layer_end=header["layer_end"],
            token_ids=header["token_ids"],
            selected_experts=selected_experts,
            geo_region=header["geo_region"],
            src_node=header["src_node"],
            dst_node=header["dst_node"],
            timestamp=header["timestamp"],
            metadata=header["metadata"],
        )
