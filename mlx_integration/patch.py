"""Monkey-patch mlx-lm to use TurboQuant compressed KV cache.

Usage:
    from mlx_lm.utils import load
    from mlx_integration.patch import apply_turboquant
    from mlx_integration.cache import make_turboquant_cache

    model, tokenizer = load("mlx-community/Qwen3.5-27B-4bit")
    cache = make_turboquant_cache(model, key_bits=3, value_bits=4)

    # Patch the model to use TurboQuant SDPA when our cache is detected
    apply_turboquant(model)

    # Now use model normally — mlx-lm's generate() works with our cache
"""

from typing import Optional

import mlx.core as mx

from mlx_integration.cache import TurboQuantKVCache
from mlx_integration.attention import turboquant_sdpa
from core.sparse_v import SparseVConfig, make_adaptive_config


# Store original SDPA for fallback
_original_sdpa = None


def _patched_sdpa(
    queries,
    keys,
    values,
    cache=None,
    scale: float = 1.0,
    mask=None,
    sinks=None,
):
    """Replacement SDPA that routes to TurboQuant when our cache is detected."""
    if isinstance(cache, TurboQuantKVCache):
        # Use TurboQuant SDPA with sparse V
        return turboquant_sdpa(
            queries=queries,
            keys=keys,
            values=values,
            scale=scale,
            mask=mask,
            sparse_v_config=_sparse_v_config,
            layer_idx=cache.layer_idx,
        )
    else:
        # Fall back to original SDPA
        return _original_sdpa(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)


# Module-level sparse V config (set by apply_turboquant)
_sparse_v_config: Optional[SparseVConfig] = None


def apply_turboquant(
    model=None,
    sparse_v: bool = True,
    n_layers: Optional[int] = None,
):
    """Patch mlx-lm's SDPA dispatch to use TurboQuant attention.

    This replaces the `scaled_dot_product_attention` function in mlx-lm's
    base module so that when a TurboQuantKVCache is detected, it routes
    to our optimized SDPA with sparse V support.

    Args:
        model: Optional model (used to detect n_layers for adaptive sparse V)
        sparse_v: Enable sparse V dequantization skipping
        n_layers: Number of layers (auto-detected from model if not provided)
    """
    global _original_sdpa, _sparse_v_config

    import mlx_lm.models.base as base_module

    # Save original
    if _original_sdpa is None:
        _original_sdpa = base_module.scaled_dot_product_attention

    # Configure sparse V
    if sparse_v:
        if n_layers is None and model is not None:
            n_layers = len(model.layers)
        if n_layers is not None:
            _sparse_v_config = make_adaptive_config(n_layers)
        else:
            _sparse_v_config = SparseVConfig(mode="percentile", percentile=50.0)
    else:
        _sparse_v_config = None

    # Monkey-patch
    base_module.scaled_dot_product_attention = _patched_sdpa


def remove_turboquant():
    """Restore original mlx-lm SDPA (undo the patch)."""
    global _original_sdpa, _sparse_v_config

    if _original_sdpa is not None:
        import mlx_lm.models.base as base_module
        base_module.scaled_dot_product_attention = _original_sdpa
        _original_sdpa = None
        _sparse_v_config = None
