"""Tests for Lloyd-Max codebooks."""

import numpy as np
import mlx.core as mx
import pytest

from core.codebook import get_codebook, get_codebook_unscaled, quantize_to_indices


class TestCodebook:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_centroid_count(self, bits):
        c, b = get_codebook_unscaled(bits)
        assert c.shape[0] == 2 ** bits
        assert b.shape[0] == 2 ** bits - 1

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_centroids_symmetric(self, bits):
        """Codebooks for symmetric distributions must be symmetric."""
        c, _ = get_codebook_unscaled(bits)
        c_np = np.array(c)
        np.testing.assert_allclose(c_np, -c_np[::-1], atol=1e-6)

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_boundaries_are_midpoints(self, bits):
        """Decision boundaries must be midpoints between consecutive centroids."""
        c, b = get_codebook_unscaled(bits)
        c_np = np.array(c)
        b_np = np.array(b)
        expected = (c_np[:-1] + c_np[1:]) / 2.0
        np.testing.assert_allclose(b_np, expected, atol=1e-6)

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_centroids_sorted(self, bits):
        c, _ = get_codebook_unscaled(bits)
        c_np = np.array(c)
        assert np.all(np.diff(c_np) > 0), "Centroids must be sorted ascending"

    def test_scaling(self):
        """Scaled codebook should be 1/sqrt(head_dim) of unscaled."""
        c_unscaled, _ = get_codebook_unscaled(3)
        c_scaled, _ = get_codebook(3, 128)
        ratio = np.array(c_unscaled) / np.array(c_scaled)
        np.testing.assert_allclose(ratio, np.sqrt(128), atol=1e-4)

    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            get_codebook_unscaled(5)
        with pytest.raises(ValueError):
            get_codebook(0, 128)


class TestQuantizeToIndices:
    def test_basic(self):
        """Values should map to correct buckets."""
        _, boundaries = get_codebook(2, 128)
        # Create values clearly in each bucket
        c, _ = get_codebook(2, 128)
        values = c.reshape(1, -1)  # (1, 4)
        indices = quantize_to_indices(values, boundaries)
        expected = mx.array([[0, 1, 2, 3]], dtype=mx.uint32)
        np.testing.assert_array_equal(np.array(indices), np.array(expected))

    def test_batch_dims(self):
        """Should work with arbitrary batch dimensions."""
        _, boundaries = get_codebook(3, 128)
        values = mx.random.normal(shape=(2, 4, 128)) / np.sqrt(128)
        indices = quantize_to_indices(values, boundaries)
        assert indices.shape == (2, 4, 128)
        assert mx.max(indices).item() <= 7
        assert mx.min(indices).item() >= 0
