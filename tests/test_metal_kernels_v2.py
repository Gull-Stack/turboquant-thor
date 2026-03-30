"""Tests for v2 Metal kernels — WHT butterfly + fused pipelines."""

import numpy as np
import mlx.core as mx
import pytest

from core.metal_kernels_v2 import (
    fused_wht,
    fused_full_quantize,
    fused_full_dequantize,
    fused_compressed_score,
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

    def test_with_signs_folded(self):
        """WHT with signs folded in should match separate sign-flip + WHT."""
        d = 128
        rotation = generate_rotation(d, seed=42)
        x = mx.random.normal(shape=(10, d))
        mx.eval(x)

        # Separate: sign-flip then WHT
        flipped = x * rotation.signs
        y_separate = fused_wht(flipped)
        mx.eval(y_separate)

        # Fused: signs folded into first-stage load
        y_fused = fused_wht(x, signs=rotation.signs)
        mx.eval(y_fused)

        np.testing.assert_allclose(
            np.array(y_separate), np.array(y_fused), atol=1e-5,
        )


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


class TestFusedCompressedScore:
    def test_output_shape(self):
        """Fused scores must be (n_heads, n_tokens)."""
        d = 128
        n_heads = 4
        n_tokens = 20
        k_bits = 4

        rotation = generate_rotation(d, seed=42)
        k_centroids, k_boundaries = get_codebook(k_bits, d)

        keys = mx.random.normal(shape=(n_heads * n_tokens, d))
        query = mx.random.normal(shape=(n_heads, d))
        mx.eval(keys, query)

        k_packed, k_norms = fused_full_quantize(keys, rotation.signs, k_boundaries, k_bits)
        mx.eval(k_packed, k_norms)

        n_k_words = k_packed.shape[-1]
        k_packed = k_packed.reshape(n_heads, n_tokens, n_k_words)
        k_norms = k_norms.reshape(n_heads, n_tokens)

        # Pre-rotate query (sign flip folded into butterfly)
        q_rotated = fused_wht(query, signs=rotation.signs)
        mx.eval(q_rotated)

        scores = fused_compressed_score(
            q_rotated, k_packed, k_centroids, k_norms, d, k_bits,
        )
        mx.eval(scores)

        assert scores.shape == (n_heads, n_tokens)

    def test_scores_match_reference(self):
        """Fused scores must match explicit decompress-then-dot."""
        d = 128
        n_heads = 2
        n_tokens = 10
        k_bits = 4

        rotation = generate_rotation(d, seed=42)
        k_centroids, k_boundaries = get_codebook(k_bits, d)

        keys = mx.random.normal(shape=(n_heads * n_tokens, d))
        query = mx.random.normal(shape=(n_heads, d))
        mx.eval(keys, query)

        # Quantize keys
        k_packed, k_norms = fused_full_quantize(keys, rotation.signs, k_boundaries, k_bits)
        mx.eval(k_packed, k_norms)

        # Reference: decompress keys, then dot with query in rotated space
        k_decompressed = fused_full_dequantize(
            k_packed, k_centroids, rotation.signs, k_norms, k_bits, d
        )
        mx.eval(k_decompressed)
        k_dec = k_decompressed.reshape(n_heads, n_tokens, d)
        # scores_ref[h, t] = query[h] . k_dec[h, t]
        scores_ref = mx.sum(query[:, None, :] * k_dec, axis=-1)
        mx.eval(scores_ref)

        # Fused path: pre-rotate query, score in rotated space
        n_k_words = k_packed.shape[-1]
        k_packed_3d = k_packed.reshape(n_heads, n_tokens, n_k_words)
        k_norms_2d = k_norms.reshape(n_heads, n_tokens)

        q_rotated = fused_wht(query, signs=rotation.signs)
        mx.eval(q_rotated)

        scores_fused = fused_compressed_score(
            q_rotated, k_packed_3d, k_centroids, k_norms_2d, d, k_bits,
        )
        mx.eval(scores_fused)

        np.testing.assert_allclose(
            np.array(scores_ref), np.array(scores_fused), atol=1e-2, rtol=1e-2,
        )

    def test_no_nan_inf(self):
        """Scores should not contain NaN or Inf."""
        d = 256
        n_heads = 4
        n_tokens = 50
        k_bits = 4

        rotation = generate_rotation(d, seed=42)
        k_centroids, k_boundaries = get_codebook(k_bits, d)

        keys = mx.random.normal(shape=(n_heads * n_tokens, d))
        query = mx.random.normal(shape=(n_heads, d))
        mx.eval(keys, query)

        k_packed, k_norms = fused_full_quantize(keys, rotation.signs, k_boundaries, k_bits)
        mx.eval(k_packed, k_norms)
        n_k_words = k_packed.shape[-1]
        k_packed = k_packed.reshape(n_heads, n_tokens, n_k_words)
        k_norms = k_norms.reshape(n_heads, n_tokens)

        q_rotated = fused_wht(query, signs=rotation.signs)
        mx.eval(q_rotated)

        scores = fused_compressed_score(
            q_rotated, k_packed, k_centroids, k_norms, d, k_bits,
        )
        mx.eval(scores)

        scores_np = np.array(scores)
        assert not np.any(np.isnan(scores_np)), "Scores contain NaN"
        assert not np.any(np.isinf(scores_np)), "Scores contain Inf"
