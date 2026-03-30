"""Tests for TurboQuant KV cache."""

import numpy as np
import mlx.core as mx
import pytest

from mlx_integration.cache import TurboQuantKVCache


class TestTurboQuantKVCache:
    def test_update_and_fetch_shapes(self):
        """Output shapes must match input shapes."""
        cache = TurboQuantKVCache(key_bits=3, value_bits=4, head_dim=128)
        B, H, L, D = 1, 4, 10, 128
        keys = mx.random.normal(shape=(B, H, L, D))
        values = mx.random.normal(shape=(B, H, L, D))

        cached_k, cached_v = cache.update_and_fetch(keys, values)
        assert cached_k.shape == (B, H, L, D)
        assert cached_v.shape == (B, H, L, D)
        assert cache.offset == L

    def test_incremental_update(self):
        """Cache should accumulate tokens across calls."""
        cache = TurboQuantKVCache(key_bits=3, value_bits=4, head_dim=128)
        B, H, D = 1, 4, 128

        # First batch of 5 tokens
        k1 = mx.random.normal(shape=(B, H, 5, D))
        v1 = mx.random.normal(shape=(B, H, 5, D))
        ck1, cv1 = cache.update_and_fetch(k1, v1)
        assert ck1.shape == (B, H, 5, D)
        assert cache.offset == 5

        # Second batch of 1 token (decode step)
        k2 = mx.random.normal(shape=(B, H, 1, D))
        v2 = mx.random.normal(shape=(B, H, 1, D))
        ck2, cv2 = cache.update_and_fetch(k2, v2)
        assert ck2.shape == (B, H, 6, D)
        assert cv2.shape == (B, H, 6, D)
        assert cache.offset == 6

    def test_reconstruction_quality(self):
        """Reconstructed values should be close to originals."""
        cache = TurboQuantKVCache(key_bits=4, value_bits=4, head_dim=128)
        B, H, L, D = 1, 4, 20, 128
        keys = mx.random.normal(shape=(B, H, L, D))
        values = mx.random.normal(shape=(B, H, L, D))

        cached_k, cached_v = cache.update_and_fetch(keys, values)

        # Compute cosine similarity
        k_np = np.array(keys).reshape(-1, D)
        ck_np = np.array(cached_k).reshape(-1, D)

        cos_sims = np.sum(k_np * ck_np, axis=-1) / (
            np.linalg.norm(k_np, axis=-1) * np.linalg.norm(ck_np, axis=-1) + 1e-10
        )
        mean_cos = np.mean(cos_sims)
        # At 4-bit, cosine similarity should be very high
        assert mean_cos > 0.98, f"Key cosine similarity {mean_cos:.4f} too low"

    def test_memory_compression(self):
        """Compressed cache should use significantly less memory than fp16."""
        cache = TurboQuantKVCache(key_bits=3, value_bits=4, head_dim=128)
        B, H, L, D = 1, 8, 100, 128
        keys = mx.random.normal(shape=(B, H, L, D))
        values = mx.random.normal(shape=(B, H, L, D))

        cache.update_and_fetch(keys, values)
        mx.eval(cache._k_packed, cache._v_packed, cache._k_norms, cache._v_norms)

        ratio = cache.compression_ratio
        # Should achieve at least 2x compression
        assert ratio > 2.0, f"Compression ratio {ratio:.2f}x too low"

    def test_empty_and_size(self):
        cache = TurboQuantKVCache(head_dim=128)
        assert cache.empty()
        assert cache.size() == 0

        keys = mx.random.normal(shape=(1, 4, 5, 128))
        values = mx.random.normal(shape=(1, 4, 5, 128))
        cache.update_and_fetch(keys, values)

        assert not cache.empty()
        assert cache.size() == 5

    def test_trim(self):
        cache = TurboQuantKVCache(head_dim=128)
        keys = mx.random.normal(shape=(1, 4, 10, 128))
        values = mx.random.normal(shape=(1, 4, 10, 128))
        cache.update_and_fetch(keys, values)

        trimmed = cache.trim(3)
        assert trimmed == 3
        assert cache.offset == 7
        assert cache._k_packed.shape[2] == 7

    def test_layer_adaptive_precision(self):
        """Last 20% of layers should get +1 bit."""
        n_layers = 10
        # Layer 7 is at 70% — should use default bits
        cache_early = TurboQuantKVCache(
            key_bits=3, value_bits=4, head_dim=128,
            layer_idx=7, n_layers=n_layers, layer_adaptive=True,
        )
        assert cache_early.key_bits == 3
        assert cache_early.value_bits == 4

        # Layer 8 is at 80% — should get +1 bit
        cache_late = TurboQuantKVCache(
            key_bits=3, value_bits=4, head_dim=128,
            layer_idx=8, n_layers=n_layers, layer_adaptive=True,
        )
        assert cache_late.key_bits == 4
        assert cache_late.value_bits == 4  # capped at 4

    def test_state_roundtrip(self):
        """State get/set should preserve cache contents."""
        cache = TurboQuantKVCache(key_bits=3, value_bits=4, head_dim=128)
        keys = mx.random.normal(shape=(1, 4, 5, 128))
        values = mx.random.normal(shape=(1, 4, 5, 128))
        cache.update_and_fetch(keys, values)

        state = cache.state
        assert state is not None

        cache2 = TurboQuantKVCache(key_bits=3, value_bits=4, head_dim=128)
        cache2.state = state
        assert cache2.offset == 5


class TestTurboQuantSDPA:
    def test_attention_output_shape(self):
        """SDPA should produce correct output shape."""
        from mlx_integration.attention import turboquant_sdpa

        B, n_heads, T_q, T_kv, D = 1, 8, 1, 20, 128
        queries = mx.random.normal(shape=(B, n_heads, T_q, D))
        keys = mx.random.normal(shape=(B, n_heads, T_kv, D))
        values = mx.random.normal(shape=(B, n_heads, T_kv, D))
        scale = D ** -0.5

        output = turboquant_sdpa(queries, keys, values, scale=scale)
        assert output.shape == (B, n_heads, T_q, D)

    def test_gqa_support(self):
        """Should handle grouped query attention (more Q heads than KV heads)."""
        from mlx_integration.attention import turboquant_sdpa

        B, n_q_heads, n_kv_heads, T_q, T_kv, D = 1, 8, 2, 1, 20, 128
        queries = mx.random.normal(shape=(B, n_q_heads, T_q, D))
        keys = mx.random.normal(shape=(B, n_kv_heads, T_kv, D))
        values = mx.random.normal(shape=(B, n_kv_heads, T_kv, D))
        scale = D ** -0.5

        output = turboquant_sdpa(queries, keys, values, scale=scale)
        assert output.shape == (B, n_q_heads, T_q, D)

    def test_attention_scores_reasonable(self):
        """Output should be a weighted average of values (bounded)."""
        from mlx_integration.attention import turboquant_sdpa

        B, H, T_q, T_kv, D = 1, 4, 1, 10, 64
        queries = mx.random.normal(shape=(B, H, T_q, D))
        keys = mx.random.normal(shape=(B, H, T_kv, D))
        values = mx.random.normal(shape=(B, H, T_kv, D))
        scale = D ** -0.5

        output = turboquant_sdpa(queries, keys, values, scale=scale)
        out_np = np.array(output)
        val_np = np.array(values)

        # Output norms should be bounded by max value norms
        out_norms = np.linalg.norm(out_np, axis=-1)
        val_norms = np.max(np.linalg.norm(val_np, axis=-1))
        assert np.all(out_norms <= val_norms * 1.1)  # small tolerance
