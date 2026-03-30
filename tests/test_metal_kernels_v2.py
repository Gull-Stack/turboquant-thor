"""Tests for v2 Metal kernels — WHT butterfly + fused pipelines."""

import numpy as np
import mlx.core as mx
import pytest

from core.metal_kernels_v2 import (
    fused_wht,
    fused_full_quantize,
    fused_full_dequantize,
    fused_compressed_attention,
)
from core.rotation import generate_rotation, rotate_forward, rotate_inverse, safe_normalize
from core.codebook import get_codebook
from core.quantizer import TurboQuantMSE


class TestFusedWHT:
    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_matches_matmul(self, d):
        """Butterfly WHT must match matmul WHT."""
        rotation = generate_rotation(d, seed=42)
        x = mx.random.normal(shape=(10, d))
        mx.eval(x)

        # Matmul path (existing)
        y_matmul = x @ rotation.hadamard.T
        mx.eval(y_matmul)

        # Butterfly path (new)
        y_butterfly = fused_wht(x, forward=True)
        mx.eval(y_butterfly)

        np.testing.assert_allclose(
            np.array(y_matmul), np.array(y_butterfly), atol=1e-4, rtol=1e-4,
        )

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_inverse_round_trip(self, d):
        """Forward then inverse should recover original."""
        x = mx.random.normal(shape=(5, d))
        mx.eval(x)

        y = fused_wht(x, forward=True)
        x_hat = fused_wht(y, forward=False)
        mx.eval(x_hat)

        np.testing.assert_allclose(
            np.array(x), np.array(x_hat), atol=1e-4,
        )

    def test_single_vector(self):
        """Should work with a single vector."""
        x = mx.random.normal(shape=(1, 128))
        y = fused_wht(x)
        mx.eval(y)
        assert y.shape == (1, 128)

    def test_preserves_norm(self):
        """WHT is orthogonal — should preserve vector norms."""
        x = mx.random.normal(shape=(20, 256))
        mx.eval(x)
        y = fused_wht(x)
        mx.eval(y)

        x_norms = np.linalg.norm(np.array(x), axis=-1)
        y_norms = np.linalg.norm(np.array(y), axis=-1)
        np.testing.assert_allclose(x_norms, y_norms, rtol=1e-3)


class TestFusedFullQuantize:
    @pytest.mark.parametrize("bits", [3, 4, 5])
    def test_matches_python_pipeline(self, bits):
        """Full fused quantize must match the Python quantizer."""
        d = 128
        N = 20
        rotation = generate_rotation(d, seed=42)
        centroids, boundaries = get_codebook(bits, d)

        x = mx.random.normal(shape=(N, d))
        mx.eval(x)

        # Python path (existing quantizer)
        q = TurboQuantMSE(head_dim=d, bits=bits, seed=42)
        x_unit, norms_py = safe_normalize(x)
        y = rotate_forward(x_unit, rotation)
        mx.eval(y)

        from core.codebook import quantize_to_indices
        from core.packing import pack_indices
        indices = quantize_to_indices(y, boundaries)
        packed_py = pack_indices(indices, bits)
        mx.eval(packed_py, norms_py)

        # Fused Metal path
        packed_metal, norms_metal = fused_full_quantize(x, rotation.signs, boundaries, bits)
        mx.eval(packed_metal, norms_metal)

        # Norms should match
        np.testing.assert_allclose(
            np.array(norms_py).reshape(-1), np.array(norms_metal), atol=1e-4,
        )

        # Packed indices should match
        np.testing.assert_array_equal(
            np.array(packed_py), np.array(packed_metal),
        )

    @pytest.mark.parametrize("d", [128, 256])
    def test_dequantize_round_trip(self, d):
        """Quantize → dequantize should produce valid reconstruction."""
        bits = 4
        rotation = generate_rotation(d, seed=42)
        centroids, boundaries = get_codebook(bits, d)

        x = mx.random.normal(shape=(10, d))
        mx.eval(x)

        packed, norms = fused_full_quantize(x, rotation.signs, boundaries, bits)
        mx.eval(packed, norms)

        x_hat = fused_full_dequantize(packed, centroids, rotation.signs, norms, bits, d)
        mx.eval(x_hat)

        # Cosine similarity should be high at 4-bit
        x_np = np.array(x)
        xh_np = np.array(x_hat)
        for i in range(x_np.shape[0]):
            cos = np.dot(x_np[i], xh_np[i]) / (
                np.linalg.norm(x_np[i]) * np.linalg.norm(xh_np[i]) + 1e-10
            )
            assert cos > 0.95, f"Row {i} cosine similarity {cos:.4f} too low"


class TestFusedFullDequantize:
    @pytest.mark.parametrize("bits", [3, 4, 5])
    def test_matches_python_pipeline(self, bits):
        """Full fused dequantize must match Python pipeline."""
        d = 128
        N = 10
        rotation = generate_rotation(d, seed=42)
        centroids, boundaries = get_codebook(bits, d)

        # Quantize with fused kernel
        x = mx.random.normal(shape=(N, d))
        mx.eval(x)
        packed, norms = fused_full_quantize(x, rotation.signs, boundaries, bits)
        mx.eval(packed, norms)

        # Python dequantize
        from core.packing import unpack_indices
        indices = unpack_indices(packed, bits, d)
        y_hat = centroids[indices]
        x_py = rotate_inverse(y_hat, rotation) * norms[:, None]
        mx.eval(x_py)

        # Fused Metal dequantize
        x_metal = fused_full_dequantize(packed, centroids, rotation.signs, norms, bits, d)
        mx.eval(x_metal)

        np.testing.assert_allclose(
            np.array(x_py), np.array(x_metal), atol=1e-3,
        )


class TestFusedCompressedAttention:
    def test_output_shape(self):
        """Fused attention output shape must be (n_heads, d)."""
        d = 128
        n_heads = 4
        n_tokens = 20
        k_bits, v_bits = 4, 4

        rotation = generate_rotation(d, seed=42)
        k_centroids, k_boundaries = get_codebook(k_bits, d)
        v_centroids, v_boundaries = get_codebook(v_bits, d)

        # Create compressed cache
        keys = mx.random.normal(shape=(n_heads, n_tokens, d))
        values = mx.random.normal(shape=(n_heads, n_tokens, d))
        query = mx.random.normal(shape=(n_heads, d))
        mx.eval(keys, values, query)

        # Quantize keys and values
        k_flat = keys.reshape(-1, d)
        v_flat = values.reshape(-1, d)
        k_packed, k_norms = fused_full_quantize(k_flat, rotation.signs, k_boundaries, k_bits)
        v_packed, v_norms = fused_full_quantize(v_flat, rotation.signs, v_boundaries, v_bits)
        mx.eval(k_packed, k_norms, v_packed, v_norms)

        n_k_words = k_packed.shape[-1]
        n_v_words = v_packed.shape[-1]
        k_packed = k_packed.reshape(n_heads, n_tokens, n_k_words)
        v_packed = v_packed.reshape(n_heads, n_tokens, n_v_words)
        k_norms = k_norms.reshape(n_heads, n_tokens)
        v_norms = v_norms.reshape(n_heads, n_tokens)

        # Pre-rotate query
        q_rotated = (query * rotation.signs) @ rotation.hadamard.T
        mx.eval(q_rotated)

        # Run fused attention
        out = fused_compressed_attention(
            q_rotated, k_packed, k_centroids, k_norms,
            v_packed, v_centroids, v_norms, d, k_bits, v_bits,
        )
        mx.eval(out)

        assert out.shape == (n_heads, d)

    def test_attention_reasonable(self):
        """Output should be bounded and not NaN/Inf."""
        d = 128
        n_heads = 2
        n_tokens = 10
        k_bits, v_bits = 4, 4

        rotation = generate_rotation(d, seed=42)
        k_centroids, k_boundaries = get_codebook(k_bits, d)
        v_centroids, v_boundaries = get_codebook(v_bits, d)

        keys = mx.random.normal(shape=(n_heads * n_tokens, d))
        values = mx.random.normal(shape=(n_heads * n_tokens, d))
        query = mx.random.normal(shape=(n_heads, d))
        mx.eval(keys, values, query)

        k_packed, k_norms = fused_full_quantize(keys, rotation.signs, k_boundaries, k_bits)
        v_packed, v_norms = fused_full_quantize(values, rotation.signs, v_boundaries, v_bits)
        mx.eval(k_packed, k_norms, v_packed, v_norms)

        n_k_words = k_packed.shape[-1]
        n_v_words = v_packed.shape[-1]
        k_packed = k_packed.reshape(n_heads, n_tokens, n_k_words)
        v_packed = v_packed.reshape(n_heads, n_tokens, n_v_words)
        k_norms = k_norms.reshape(n_heads, n_tokens)
        v_norms = v_norms.reshape(n_heads, n_tokens)

        q_rotated = (query * rotation.signs) @ rotation.hadamard.T
        mx.eval(q_rotated)

        out = fused_compressed_attention(
            q_rotated, k_packed, k_centroids, k_norms,
            v_packed, v_centroids, v_norms, d, k_bits, v_bits,
        )
        mx.eval(out)

        out_np = np.array(out)
        assert not np.any(np.isnan(out_np)), "Output contains NaN"
        assert not np.any(np.isinf(out_np)), "Output contains Inf"
        assert np.all(np.abs(out_np) < 100), "Output values unreasonably large"
