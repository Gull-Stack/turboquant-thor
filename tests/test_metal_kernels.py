"""Tests for fused Metal kernels — verify they match Python implementations."""

import numpy as np
import mlx.core as mx
import pytest

from core.metal_kernels import (
    fused_quantize_pack,
    fused_dequant_unpack,
    fused_normalize_signflip,
    fused_signflip_scale,
)
from core.codebook import get_codebook, quantize_to_indices
from core.packing import pack_indices, unpack_indices
from core.rotation import generate_rotation, rotate_forward, rotate_inverse, safe_normalize


class TestFusedQuantizePack:
    @pytest.mark.parametrize("bits", [2, 3, 4, 5])
    def test_matches_python(self, bits):
        """Fused kernel must produce identical output to Python path."""
        D = 128
        centroids, boundaries = get_codebook(bits, D)

        # Random rotated values (simulating post-WHT)
        values = mx.random.normal(shape=(D,)) / np.sqrt(D)
        mx.eval(values)

        # Python path
        indices = quantize_to_indices(values.reshape(1, -1), boundaries)
        packed_py = pack_indices(indices.reshape(1, -1), bits)
        mx.eval(packed_py)

        # Metal path
        packed_metal = fused_quantize_pack(values, boundaries, bits)
        mx.eval(packed_metal)

        np.testing.assert_array_equal(
            np.array(packed_py.reshape(-1)),
            np.array(packed_metal),
        )

    @pytest.mark.parametrize("bits", [2, 3, 4, 5])
    def test_batch(self, bits):
        """Should work with batched (N, D) input."""
        D = 128
        N = 50
        _, boundaries = get_codebook(bits, D)

        values = mx.random.normal(shape=(N, D)) / np.sqrt(D)
        mx.eval(values)

        # Python path
        indices = quantize_to_indices(values, boundaries)
        packed_py = pack_indices(indices, bits)
        mx.eval(packed_py)

        # Metal path
        packed_metal = fused_quantize_pack(values, boundaries, bits)
        mx.eval(packed_metal)

        np.testing.assert_array_equal(
            np.array(packed_py.reshape(-1)),
            np.array(packed_metal.reshape(-1)),
        )


class TestFusedDequantUnpack:
    @pytest.mark.parametrize("bits", [2, 3, 4, 5])
    def test_matches_python(self, bits):
        """Fused kernel must produce identical output to Python path."""
        D = 128
        centroids, boundaries = get_codebook(bits, D)

        # Create packed data via Python path
        values = mx.random.normal(shape=(1, D)) / np.sqrt(D)
        indices = quantize_to_indices(values, boundaries)
        packed = pack_indices(indices, bits)
        mx.eval(packed)

        # Python dequant
        unpacked = unpack_indices(packed, bits, D)
        y_py = centroids[unpacked].reshape(-1)
        mx.eval(y_py)

        # Metal dequant
        y_metal = fused_dequant_unpack(packed.reshape(-1), centroids, bits, D)
        mx.eval(y_metal)

        np.testing.assert_allclose(
            np.array(y_py),
            np.array(y_metal),
            atol=1e-6,
        )

    @pytest.mark.parametrize("bits", [2, 3, 4, 5])
    def test_round_trip(self, bits):
        """quantize_pack → dequant_unpack should produce valid centroids."""
        D = 128
        centroids, boundaries = get_codebook(bits, D)

        values = mx.random.normal(shape=(D,)) / np.sqrt(D)
        mx.eval(values)

        packed = fused_quantize_pack(values, boundaries, bits)
        mx.eval(packed)

        reconstructed = fused_dequant_unpack(packed, centroids, bits, D)
        mx.eval(reconstructed)

        # Every value should be a valid centroid
        centroids_np = np.array(centroids)
        recon_np = np.array(reconstructed)
        for val in recon_np[:20]:
            dists = np.abs(centroids_np - val)
            assert np.min(dists) < 1e-5, f"Value {val} not a valid centroid"


class TestFusedNormalizeSignflip:
    def test_matches_python(self):
        """Fused normalize+signflip must match sequential Python ops."""
        D = 128
        N = 10
        rotation = generate_rotation(D, seed=42)

        x = mx.random.normal(shape=(N, D))
        mx.eval(x)

        # Python path
        x_unit, norms = safe_normalize(x)
        flipped_py = x_unit * rotation.signs
        mx.eval(flipped_py)

        # Metal path
        norms_flat = mx.linalg.norm(x, axis=-1)
        mx.eval(norms_flat)
        flipped_metal = fused_normalize_signflip(x, norms_flat, rotation.signs, D)
        mx.eval(flipped_metal)

        np.testing.assert_allclose(
            np.array(flipped_py),
            np.array(flipped_metal),
            atol=1e-5,
        )


class TestFusedSignflipScale:
    def test_matches_python(self):
        """Fused signflip+scale must match sequential Python ops."""
        D = 128
        N = 10
        rotation = generate_rotation(D, seed=42)

        y = mx.random.normal(shape=(N, D))
        norms = mx.random.uniform(shape=(N,)) * 5.0 + 0.1
        mx.eval(y, norms)

        # Python path
        result_py = y * rotation.signs * norms[:, None]
        mx.eval(result_py)

        # Metal path
        result_metal = fused_signflip_scale(y, rotation.signs, norms, D)
        mx.eval(result_metal)

        np.testing.assert_allclose(
            np.array(result_py),
            np.array(result_metal),
            atol=1e-5,
        )


class TestBenchmark:
    """Verify kernels work at scale (not a speed test, just correctness)."""

    def test_large_batch(self):
        """Test with realistic cache sizes."""
        D = 256  # Qwen3.5 head_dim
        N = 1000  # tokens
        bits = 4

        centroids, boundaries = get_codebook(bits, D)
        values = mx.random.normal(shape=(N * D,)) / np.sqrt(D)
        mx.eval(values)

        packed = fused_quantize_pack(values, boundaries, bits)
        mx.eval(packed)

        reconstructed = fused_dequant_unpack(packed, centroids, bits, N * D)
        mx.eval(reconstructed)

        assert reconstructed.shape[0] == N * D
