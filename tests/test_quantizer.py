"""Tests for TurboQuant MSE quantizer."""

import numpy as np
import mlx.core as mx
import pytest

from core.quantizer import TurboQuantMSE, AsymmetricQuantizer


class TestTurboQuantMSE:
    @pytest.mark.parametrize("bits,expected_mse", [
        (1, 0.3634),
        (2, 0.1175),
        (3, 0.03045),
        (4, 0.00883),
    ])
    def test_mse_matches_paper(self, bits, expected_mse):
        """MSE distortion should match paper's theoretical bounds.

        Theorem 1: D_mse ≈ {0.36, 0.117, 0.03, 0.009} for b={1,2,3,4}.
        We allow 30% tolerance for finite-sample effects.
        """
        d = 128
        n = 2000
        q = TurboQuantMSE(head_dim=d, bits=bits, seed=42)

        # Random unit vectors
        x = mx.random.normal(shape=(n, d))
        x = x / mx.linalg.norm(x, axis=-1, keepdims=True)

        # Quantize and dequantize
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)

        # Compute MSE
        mse = np.mean(np.sum((np.array(x) - np.array(x_hat)) ** 2, axis=-1))

        # Allow 30% tolerance (finite sample + WHT vs QR differences)
        assert mse < expected_mse * 1.3, \
            f"MSE {mse:.4f} exceeds {expected_mse * 1.3:.4f} (paper: {expected_mse})"
        assert mse > expected_mse * 0.5, \
            f"MSE {mse:.4f} suspiciously low vs paper: {expected_mse}"

    def test_norm_preservation(self):
        """Dequantized vectors should have approximately correct norms."""
        d = 128
        q = TurboQuantMSE(head_dim=d, bits=3)
        x = mx.random.normal(shape=(100, d)) * 2.5  # non-unit norms
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)

        norms_orig = np.linalg.norm(np.array(x), axis=-1)
        norms_recon = np.linalg.norm(np.array(x_hat), axis=-1)

        # Norms should be well-preserved (stored as float32)
        np.testing.assert_allclose(norms_orig, norms_recon, rtol=0.15)

    def test_inner_product_preservation(self):
        """Inner products (attention scores) should be well-preserved.

        This is the key metric — TurboQuant preserves dot products even
        when individual vectors have 23-44% reconstruction error.
        """
        d = 128
        n = 200
        q = TurboQuantMSE(head_dim=d, bits=3)

        # Simulate queries and keys
        queries = mx.random.normal(shape=(n, d))
        keys = mx.random.normal(shape=(n, d))

        # True attention scores
        true_scores = np.sum(np.array(queries) * np.array(keys), axis=-1)

        # Compressed scores
        qt_keys = q.quantize(keys)
        keys_hat = q.dequantize(qt_keys)
        compressed_scores = np.sum(np.array(queries) * np.array(keys_hat), axis=-1)

        # Cosine similarity between score vectors should be high
        cos_sim = np.dot(true_scores, compressed_scores) / (
            np.linalg.norm(true_scores) * np.linalg.norm(compressed_scores)
        )
        assert cos_sim > 0.95, f"Score cosine similarity {cos_sim:.4f} too low"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_round_trip_shape(self, bits):
        d = 128
        q = TurboQuantMSE(head_dim=d, bits=bits)
        x = mx.random.normal(shape=(2, 4, d))
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        assert x_hat.shape == x.shape

    def test_compression_ratio(self):
        q = TurboQuantMSE(head_dim=128, bits=3)
        ratio = q.compression_ratio()
        # 3 bits + ~0.25 amortized norm ≈ 3.25 effective → 16/3.25 ≈ 4.9x
        assert ratio > 4.0
        assert ratio < 6.0

    def test_dequantize_rotated(self):
        """Dequantize in rotated space should give centroids (no inverse rotation)."""
        d = 128
        q = TurboQuantMSE(head_dim=d, bits=3)
        x = mx.random.normal(shape=(10, d))
        x = x / mx.linalg.norm(x, axis=-1, keepdims=True)
        qt = q.quantize(x)
        y_hat = q.dequantize_rotated(qt)
        # Values should be from the codebook
        centroids_np = np.array(q.centroids)
        y_np = np.array(y_hat)
        for val in y_np.flatten()[:20]:
            dists = np.abs(centroids_np - val)
            assert np.min(dists) < 1e-6, f"Value {val} not in codebook"


class TestAsymmetricQuantizer:
    def test_different_bits(self):
        aq = AsymmetricQuantizer(head_dim=128, key_bits=3, value_bits=4)
        keys = mx.random.normal(shape=(4, 128))
        values = mx.random.normal(shape=(4, 128))
        q_keys, q_values = aq.quantize_kv(keys, values)
        assert q_keys.bits == 3
        assert q_values.bits == 4

    def test_values_more_accurate(self):
        """Values at 4-bit should have lower MSE than keys at 3-bit."""
        d = 128
        n = 500
        aq = AsymmetricQuantizer(head_dim=d, key_bits=3, value_bits=4)

        x = mx.random.normal(shape=(n, d))
        x = x / mx.linalg.norm(x, axis=-1, keepdims=True)

        q_k, q_v = aq.quantize_kv(x, x)
        k_hat = aq.key_quantizer.dequantize(q_k)
        v_hat = aq.value_quantizer.dequantize(q_v)

        mse_k = np.mean(np.sum((np.array(x) - np.array(k_hat)) ** 2, axis=-1))
        mse_v = np.mean(np.sum((np.array(x) - np.array(v_hat)) ** 2, axis=-1))

        assert mse_v < mse_k, "4-bit values should have lower MSE than 3-bit keys"

    def test_effective_bits(self):
        aq = AsymmetricQuantizer(key_bits=3, value_bits=4)
        assert aq.effective_bits() == 3.5
