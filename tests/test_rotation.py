"""Tests for WHT rotation."""

import numpy as np
import mlx.core as mx
import pytest

from core.rotation import (
    generate_rotation, rotate_forward, rotate_inverse,
    safe_normalize, _hadamard_matrix,
)


class TestHadamard:
    @pytest.mark.parametrize("d", [1, 2, 4, 8, 16, 64, 128])
    def test_orthogonal(self, d):
        """H @ H^T should be identity (Hadamard is orthogonal)."""
        H = _hadamard_matrix(d)
        H_np = np.array(H)
        product = H_np @ H_np.T
        np.testing.assert_allclose(product, np.eye(d), atol=1e-5)

    @pytest.mark.parametrize("d", [1, 2, 4, 8, 128])
    def test_symmetric(self, d):
        """Normalized Hadamard is symmetric: H = H^T."""
        H = _hadamard_matrix(d)
        H_np = np.array(H)
        np.testing.assert_allclose(H_np, H_np.T, atol=1e-6)

    def test_not_power_of_2(self):
        with pytest.raises(ValueError):
            _hadamard_matrix(3)


class TestRotation:
    def test_round_trip(self):
        """rotate_forward then rotate_inverse should be identity."""
        rot = generate_rotation(128, seed=42)
        x = mx.random.normal(shape=(4, 128))
        y = rotate_forward(x, rot)
        x_back = rotate_inverse(y, rot)
        np.testing.assert_allclose(np.array(x), np.array(x_back), atol=1e-5)

    def test_norm_preservation(self):
        """Rotation should preserve vector norms (orthogonal transform)."""
        rot = generate_rotation(128, seed=42)
        x = mx.random.normal(shape=(10, 128))
        y = rotate_forward(x, rot)
        norms_x = np.linalg.norm(np.array(x), axis=-1)
        norms_y = np.linalg.norm(np.array(y), axis=-1)
        np.testing.assert_allclose(norms_x, norms_y, atol=1e-5)

    def test_gaussianization(self):
        """After rotating unit vectors, coordinates should be ~ N(0, 1/d).

        We generate many random unit vectors, rotate them, and check
        that the coordinate distribution has the expected mean and variance.
        """
        d = 128
        n = 5000
        rot = generate_rotation(d, seed=42)

        # Random unit vectors
        x = mx.random.normal(shape=(n, d))
        x = x / mx.linalg.norm(x, axis=-1, keepdims=True)

        # Rotate
        y = rotate_forward(x, rot)
        y_np = np.array(y)

        # Mean should be ~0
        mean = np.mean(y_np)
        assert abs(mean) < 0.01, f"Mean too far from 0: {mean}"

        # Variance should be ~1/d
        var = np.var(y_np)
        expected_var = 1.0 / d
        assert abs(var - expected_var) / expected_var < 0.15, \
            f"Variance {var} too far from {expected_var}"

    def test_kurtosis_reduction(self):
        """Rotation should reduce kurtosis toward 3.0 (Gaussian).

        Real KV cache tensors have high kurtosis (heavy tails).
        After rotation, kurtosis should approach the Gaussian value of 3.
        """
        from scipy.stats import kurtosis as scipy_kurtosis

        d = 128
        n = 2000
        rot = generate_rotation(d, seed=42)

        # Create vectors with high kurtosis (simulate real KV tensors)
        x = mx.random.normal(shape=(n, d))
        # Add some outliers to create heavy tails
        outlier_mask = mx.random.uniform(shape=(n, d)) > 0.95
        x = mx.where(outlier_mask, x * 5.0, x)
        x = x / mx.linalg.norm(x, axis=-1, keepdims=True)

        y = rotate_forward(x, rot)
        y_np = np.array(y).flatten()

        kurt = scipy_kurtosis(y_np, fisher=True)  # excess kurtosis, Gaussian=0
        # After rotation, kurtosis should be close to 0 (Gaussian)
        assert abs(kurt) < 0.5, f"Kurtosis {kurt} too far from 0 (Gaussian)"

    def test_batch_dims(self):
        """Should work with arbitrary batch dimensions."""
        rot = generate_rotation(128, seed=42)
        x = mx.random.normal(shape=(2, 3, 128))
        y = rotate_forward(x, rot)
        assert y.shape == (2, 3, 128)
        x_back = rotate_inverse(y, rot)
        np.testing.assert_allclose(np.array(x), np.array(x_back), atol=1e-5)

    def test_deterministic(self):
        """Same seed should give same rotation."""
        rot1 = generate_rotation(128, seed=42)
        rot2 = generate_rotation(128, seed=42)
        np.testing.assert_array_equal(np.array(rot1.signs), np.array(rot2.signs))


class TestSafeNormalize:
    def test_unit_norm(self):
        x = mx.random.normal(shape=(10, 128))
        x_norm, norms = safe_normalize(x)
        computed_norms = np.linalg.norm(np.array(x_norm), axis=-1)
        np.testing.assert_allclose(computed_norms, 1.0, atol=1e-5)

    def test_zero_vector(self):
        """Zero vectors should not produce NaN."""
        x = mx.zeros((1, 128))
        x_norm, norms = safe_normalize(x)
        assert not np.any(np.isnan(np.array(x_norm)))

    def test_norm_values(self):
        x = mx.array([[3.0, 4.0]])
        _, norms = safe_normalize(x)
        np.testing.assert_allclose(np.array(norms), [[5.0]], atol=1e-5)
