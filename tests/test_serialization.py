# Copyright 2025 Project Astra Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for astra.serialization.tensor_pack."""

import numpy as np
import pytest

from astra.serialization.tensor_pack import (
    TensorPacket,
    TensorSerializer,
    DEEPSEEK_V4_HIDDEN_DIM,
    DEEPSEEK_V4_TOP_K_EXPERTS,
)


# ─────────────────────────────────────────────────────────────────────────── #
# TensorPacket construction                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

class TestTensorPacketConstruction:
    def test_make_input_shape(self):
        pkt = TensorPacket.make_input([0, 1, 2, 3], hidden_dim=64)
        assert pkt.tensor.shape == (4, 64)
        assert pkt.tensor.dtype == np.float16

    def test_make_input_default_dim(self):
        pkt = TensorPacket.make_input([0], hidden_dim=DEEPSEEK_V4_HIDDEN_DIM)
        assert pkt.hidden_dim == DEEPSEEK_V4_HIDDEN_DIM

    def test_seq_len_property(self):
        pkt = TensorPacket.make_input(list(range(8)), hidden_dim=32)
        assert pkt.seq_len == 8

    def test_packet_id_unique(self):
        ids = {TensorPacket.make_input([0]).packet_id for _ in range(100)}
        assert len(ids) == 100

    def test_byte_size(self):
        pkt = TensorPacket.make_input([0, 1], hidden_dim=16)
        assert pkt.byte_size() == 2 * 16 * 2   # seq=2, dim=16, float16=2B


# ─────────────────────────────────────────────────────────────────────────── #
# Serialization round-trips                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

class TestRoundTrip:
    @pytest.fixture
    def packet(self):
        pkt = TensorPacket.make_input([10, 20, 30], hidden_dim=32, geo_region="eu-central")
        pkt.layer_start = 3
        pkt.layer_end = 7
        pkt.metadata = {"foo": "bar"}
        return pkt

    def test_basic_roundtrip(self, packet):
        raw = TensorSerializer.serialize(packet)
        recovered = TensorSerializer.deserialize(raw)
        assert recovered.packet_id == packet.packet_id
        assert recovered.layer_start == packet.layer_start
        assert recovered.layer_end == packet.layer_end
        assert recovered.geo_region == packet.geo_region
        assert recovered.metadata == packet.metadata
        assert np.array_equal(recovered.tensor, packet.tensor)

    def test_tensor_values_preserved(self):
        rng = np.random.default_rng(42)
        t = rng.standard_normal((4, 32)).astype(np.float16)
        pkt = TensorPacket(tensor=t, layer_start=0, layer_end=1, token_ids=[0, 1, 2, 3])
        raw = TensorSerializer.serialize(pkt)
        rec = TensorSerializer.deserialize(raw)
        assert np.allclose(rec.tensor.astype(np.float32), t.astype(np.float32))

    def test_with_selected_experts(self):
        pkt = TensorPacket.make_input([1, 2], hidden_dim=16)
        pkt.selected_experts = np.array([[0, 1], [2, 3]], dtype=np.int32)
        raw = TensorSerializer.serialize(pkt)
        rec = TensorSerializer.deserialize(raw)
        assert rec.selected_experts is not None
        assert np.array_equal(rec.selected_experts, pkt.selected_experts)

    def test_without_selected_experts(self):
        pkt = TensorPacket.make_input([1], hidden_dim=8)
        pkt.selected_experts = None
        raw = TensorSerializer.serialize(pkt)
        rec = TensorSerializer.deserialize(raw)
        assert rec.selected_experts is None

    def test_bad_magic_raises(self):
        raw = b"XXXX" + b"\x00" * 100
        with pytest.raises(ValueError, match="Bad magic"):
            TensorSerializer.deserialize(raw)

    def test_wire_size_reasonable(self):
        pkt = TensorPacket.make_input(list(range(16)), hidden_dim=256)
        raw = TensorSerializer.serialize(pkt)
        tensor_bytes = 16 * 256 * 2
        assert len(raw) < tensor_bytes * 1.1   # <10% overhead

    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_dtype_preserved(self, dtype):
        t = np.zeros((2, 8), dtype=dtype)
        pkt = TensorPacket(tensor=t)
        rec = TensorSerializer.deserialize(TensorSerializer.serialize(pkt))
        assert rec.tensor.dtype == dtype

    def test_split_by_expert(self):
        pkt = TensorPacket.make_input([0, 1, 2, 3], hidden_dim=8)
        # token 0 and 2 route to expert 5
        pkt.selected_experts = np.array([[5, 6], [7, 8], [5, 9], [10, 11]], dtype=np.int32)
        sub, indices = pkt.split_by_expert(5)
        assert list(indices) == [0, 2]
        assert sub.tensor.shape == (2, 8)


# ─────────────────────────────────────────────────────────────────────────── #
# from_bytes / to_bytes convenience                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class TestConvenienceMethods:
    def test_to_bytes_from_bytes(self):
        pkt = TensorPacket.make_input([1, 2], hidden_dim=16)
        pkt2 = TensorPacket.from_bytes(pkt.to_bytes())
        assert pkt2.packet_id == pkt.packet_id
