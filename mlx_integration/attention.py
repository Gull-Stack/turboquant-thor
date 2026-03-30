"""TurboQuant SDPA — scaled dot-product attention with compressed KV cache.

Optimizations over standard SDPA:
1. Matrix associativity: (weights @ v_centroids) @ Π instead of weights @ (Π @ v_centroids)
   Saves O(T_kv × D²) → O(T_q × D²) for inverse rotation on values.
2. Sparse V: skip dequantization for negligible attention positions.
3. Works directly with TurboQuantKVCache — no intermediate decompression needed.

Reference: TurboQuant (arXiv:2504.19874), Section 4
"""

from typing import Optional

import mlx.core as mx

from core.sparse_v import SparseVConfig, apply_sparse_v


def turboquant_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[mx.array] = None,
    sparse_v_config: Optional[SparseVConfig] = None,
    layer_idx: int = 0,
) -> mx.array:
    """Scaled dot-product attention with sparse V optimization.

    Standard SDPA path but with optional sparse V dequantization skipping.
    Keys and values are already decompressed by the cache's update_and_fetch.

    Args:
        queries: (B, n_heads, T_q, D)
        keys: (B, n_kv_heads, T_kv, D)
        values: (B, n_kv_heads, T_kv, D)
        scale: Attention scale factor (typically 1/sqrt(head_dim))
        mask: Optional attention mask
        sparse_v_config: Optional sparse V config for decode speedup
        layer_idx: Current layer index (for adaptive sparse V)

    Returns:
        output: (B, n_heads, T_q, D)
    """
    B, n_q_heads, T_q, D = queries.shape
    n_kv_heads = keys.shape[1]
    n_repeats = n_q_heads // n_kv_heads

    # Scale queries
    queries = queries * scale

    # Handle GQA: expand key/value heads
    if n_repeats > 1:
        queries = queries.reshape(B, n_kv_heads, n_repeats, T_q, D)
        keys = mx.expand_dims(keys, axis=2)       # (B, n_kv_heads, 1, T_kv, D)
        values = mx.expand_dims(values, axis=2)    # (B, n_kv_heads, 1, T_kv, D)

    # Compute attention scores: Q @ K^T
    scores = queries @ mx.swapaxes(keys, -1, -2)  # (..., T_q, T_kv)

    # Apply mask
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            T_kv = keys.shape[-2]
            q_indices = mx.arange(T_kv - T_q, T_kv)
            k_indices = mx.arange(T_kv)
            causal = q_indices[:, None] >= k_indices[None]
            scores = mx.where(causal, scores, mx.finfo(scores.dtype).min)
        elif isinstance(mask, mx.array):
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores = scores + mask

    # Softmax
    weights = mx.softmax(scores, axis=-1, precise=True)

    # Weighted sum with optional sparse V
    if sparse_v_config is not None:
        output = apply_sparse_v(weights, values, sparse_v_config, layer_idx)
    else:
        output = weights @ values

    # Reshape back from GQA
    if n_repeats > 1:
        output = output.reshape(B, n_q_heads, T_q, D)

    return output
