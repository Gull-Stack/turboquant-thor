"""Tests for bit packing utilities."""

import numpy as np
import mlx.core as mx
import pytest

from core.packing import pack_indices, unpack_indices, packed_size, _vals_per_word


class TestPacking:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_round_trip(self, bits):
        """Pack then unpack should return original indices."""
        D = 128
        max_val = 2 ** bits
        indices = (mx.random.uniform(shape=(4, D)) * max_val).astype(mx.uint32)
        indices = mx.minimum(indices, max_val - 1)

        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, D)

        np.testing.assert_array_equal(np.array(unpacked), np.array(indices))

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_round_trip_batched(self, bits):
        """Should work with multiple batch dimensions."""
        D = 128
        max_val = 2 ** bits
        indices = (mx.random.uniform(shape=(2, 3, D)) * max_val).astype(mx.uint32)
        indices = mx.minimum(indices, max_val - 1)

        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, D)

        np.testing.assert_array_equal(np.array(unpacked), np.array(indices))

    def test_3bit_vals_per_word(self):
        """3-bit must pack 10 values per uint32 word, not 8."""
        assert _vals_per_word(3) == 10

    def test_2bit_vals_per_word(self):
        assert _vals_per_word(2) == 16

    def test_4bit_vals_per_word(self):
        assert _vals_per_word(4) == 8

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_packed_size(self, bits):
        """Packed size should be ceil(D / vals_per_word)."""
        D = 128
        vpw = _vals_per_word(bits)
        expected_words = (D + vpw - 1) // vpw
        assert packed_size(D, bits) == expected_words

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_compression_ratio(self, bits):
        """Packed representation should be smaller than raw indices."""
        D = 128
        indices = mx.zeros((1, D), dtype=mx.uint32)
        packed = pack_indices(indices, bits)
        # Each uint32 word holds multiple indices
        assert packed.shape[-1] < D

    def test_3bit_compression(self):
        """3-bit at D=128: should use 13 words (128/10 = 12.8, ceil = 13)."""
        D = 128
        indices = mx.zeros((1, D), dtype=mx.uint32)
        packed = pack_indices(indices, 3)
        assert packed.shape[-1] == 13  # ceil(128/10)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_all_zeros(self, bits):
        D = 128
        indices = mx.zeros((D,), dtype=mx.uint32)
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, D)
        np.testing.assert_array_equal(np.array(unpacked), np.zeros(D))

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_max_values(self, bits):
        """All-max indices should round-trip correctly."""
        D = 128
        max_val = 2 ** bits - 1
        indices = mx.full((D,), max_val, dtype=mx.uint32)
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, D)
        np.testing.assert_array_equal(np.array(unpacked), np.full(D, max_val))
