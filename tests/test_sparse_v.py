"""Tests for sparse V dequantization."""

import numpy as np
import mlx.core as mx
import pytest

from core.sparse_v import (
    SparseVConfig, compute_sparse_v_mask,
    make_adaptive_config, apply_sparse_v,
)


class TestSparseVMask:
    def test_fixed_threshold(self):
        config = SparseVConfig(mode="fixed", threshold=0.1)
        weights = mx.array([0.01, 0.05, 0.2, 0.5, 0.24])
        mask = compute_sparse_v_mask(weights, config)
        expected = np.array([False, False, True, True, True])
        np.testing.assert_array_equal(np.array(mask), expected)

    def test_percentile(self):
        config = SparseVConfig(mode="percentile", percentile=50.0)
        weights = mx.array([0.01, 0.02, 0.03, 0.04, 0.1, 0.2, 0.3, 0.3])
        mask = compute_sparse_v_mask(weights, config)
        # Bottom 50% should be masked out (4 of 8 values)
        assert np.sum(np.array(mask)) >= 4

    def test_adaptive(self):
        config = make_adaptive_config(n_layers=10, base_threshold=1e-6)
        weights = mx.array([1e-7, 1e-5, 0.1, 0.5])
        # Early layer (high multiplier → more skipping)
        mask_early = compute_sparse_v_mask(weights, config, layer_idx=0)
        # Late layer (low multiplier → less skipping)
        mask_late = compute_sparse_v_mask(weights, config, layer_idx=9)
        # Early layer should skip more
        assert np.sum(np.array(mask_early)) <= np.sum(np.array(mask_late))


class TestApplySparseV:
    def test_no_config_is_standard(self):
        """Without config, should be standard matmul."""
        weights = mx.softmax(mx.random.normal(shape=(1, 4, 10, 20)), axis=-1)
        values = mx.random.normal(shape=(1, 4, 20, 64))
        out_standard = weights @ values
        out_sparse = apply_sparse_v(weights, values, config=None)
        np.testing.assert_allclose(
            np.array(out_standard), np.array(out_sparse), atol=1e-6
        )

    def test_sparse_v_close_to_standard(self):
        """Sparse V output should be very close to standard (skipped mass is tiny)."""
        # Create realistic attention weights (most mass on few positions)
        logits = mx.random.normal(shape=(1, 4, 1, 64))
        weights = mx.softmax(logits * 3.0, axis=-1)  # sharp attention
        values = mx.random.normal(shape=(1, 4, 64, 128))

        config = SparseVConfig(mode="fixed", threshold=1e-6)
        out_standard = weights @ values
        out_sparse = apply_sparse_v(weights, values, config=config)

        # Should be nearly identical
        diff = np.max(np.abs(np.array(out_standard) - np.array(out_sparse)))
        assert diff < 1e-4, f"Sparse V diff {diff} too large"

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            SparseVConfig(mode="invalid")


class TestMakeAdaptiveConfig:
    def test_layer_count(self):
        config = make_adaptive_config(n_layers=40)
        assert len(config.layer_thresholds) == 40

    def test_decreasing_multipliers(self):
        """Earlier layers should have higher multipliers (skip more)."""
        config = make_adaptive_config(n_layers=10, early_multiplier=10.0, late_multiplier=1.0)
        thresholds = config.layer_thresholds
        assert thresholds[0] > thresholds[-1]
        # Should be monotonically decreasing
        for i in range(len(thresholds) - 1):
            assert thresholds[i] >= thresholds[i + 1]
