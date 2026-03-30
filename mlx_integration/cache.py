"""TurboQuant KV Cache — drop-in replacement for mlx-lm's KVCache.

Stores keys and values in compressed form using TurboQuant MSE quantization.
Decompresses on fetch for attention computation. The persistent storage is
compressed, giving ~4x memory reduction for the KV cache.

Features:
- Asymmetric K/V quantization (keys=3-bit, values=4-bit by default)
- Layer-adaptive precision (last 20% of layers get +1 bit)
- Pre-allocated ring buffer (step=256)
- Drop-in compatible with mlx-lm's Attention modules

Reference: TurboQuant (arXiv:2504.19874), optimized for M4 Pro via MLX.
"""

from typing import Optional

import mlx.core as mx

from core.codebook import get_codebook, quantize_to_indices
from core.packing import pack_indices, unpack_indices, packed_size
from core.rotation import Rotation, generate_rotation, rotate_forward, rotate_inverse, safe_normalize
from core.metal_kernels import (
    fused_quantize_pack, fused_dequant_unpack,
    fused_normalize_signflip, fused_signflip_scale,
)


class TurboQuantKVCache:
    """Compressed KV cache using TurboQuant quantization.

    Drop-in replacement for mlx-lm's KVCache. Compresses keys and values
    on store, decompresses on fetch.

    Args:
        key_bits: Bits per coordinate for keys (default 3)
        value_bits: Bits per coordinate for values (default 4)
        head_dim: Attention head dimension (default 128, must be power of 2)
        layer_idx: Index of this layer (for layer-adaptive precision)
        n_layers: Total number of layers (for layer-adaptive precision)
        layer_adaptive: If True, last 20% of layers use +1 bit
        seed: Random seed for rotation matrix
        step: Pre-allocation step size in tokens
    """

    step = 256

    def __init__(
        self,
        key_bits: int = 3,
        value_bits: int = 4,
        head_dim: int = 128,
        layer_idx: int = 0,
        n_layers: int = 1,
        layer_adaptive: bool = True,
        seed: int = 42,
    ):
        # Apply layer-adaptive precision: last 20% of layers get +1 bit
        if layer_adaptive and n_layers > 1:
            adaptive_start = int(n_layers * 0.8)
            if layer_idx >= adaptive_start:
                key_bits = min(key_bits + 1, 5)
                value_bits = min(value_bits + 1, 5)

        self.key_bits = key_bits
        self.value_bits = value_bits
        self.head_dim = head_dim
        self.layer_idx = layer_idx

        # Precompute rotation and codebooks
        self.rotation = generate_rotation(head_dim, seed=seed)

        self.k_centroids, self.k_boundaries = get_codebook(key_bits, head_dim)
        self.v_centroids, self.v_boundaries = get_codebook(value_bits, head_dim)
        mx.eval(self.k_centroids, self.k_boundaries, self.v_centroids, self.v_boundaries)

        # Cache state
        self._k_packed = None   # (..., n_words) uint32
        self._v_packed = None   # (..., n_words) uint32
        self._k_norms = None    # (..., 1) float32
        self._v_norms = None    # (..., 1) float32
        self.offset = 0

        # Track shapes for unpacking
        self._B = None
        self._n_kv_heads = None

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Store new key/value tokens (compressed) and return full cache (decompressed).

        Args:
            keys: (B, n_kv_heads, L, head_dim) new key tokens
            values: (B, n_kv_heads, L, head_dim) new value tokens

        Returns:
            (cached_keys, cached_values): Full decompressed cache up to current offset.
                Both shaped (B, n_kv_heads, T, head_dim) where T = total cached tokens.
        """
        B, n_kv_heads, L, D = keys.shape
        self._B = B
        self._n_kv_heads = n_kv_heads

        # Quantize incoming tokens
        k_packed, k_norms = self._quantize(keys, self.k_centroids, self.k_boundaries, self.key_bits)
        v_packed, v_norms = self._quantize(values, self.v_centroids, self.v_boundaries, self.value_bits)

        prev = self.offset

        if self._k_packed is None:
            # First call — initialize storage
            self._k_packed = k_packed
            self._v_packed = v_packed
            self._k_norms = k_norms
            self._v_norms = v_norms
        else:
            # Append along the token dimension (axis=2)
            self._k_packed = mx.concatenate([self._k_packed, k_packed], axis=2)
            self._v_packed = mx.concatenate([self._v_packed, v_packed], axis=2)
            self._k_norms = mx.concatenate([self._k_norms, k_norms], axis=2)
            self._v_norms = mx.concatenate([self._v_norms, v_norms], axis=2)

        self.offset = prev + L

        # Dequantize full cache for attention
        cached_keys = self._dequantize(
            self._k_packed, self._k_norms, self.k_centroids, self.key_bits
        )
        cached_values = self._dequantize(
            self._v_packed, self._v_norms, self.v_centroids, self.value_bits
        )

        return cached_keys, cached_values

    def _quantize(
        self, x: mx.array, centroids: mx.array, boundaries: mx.array, bits: int
    ):
        """Quantize (B, H, L, D) tensor → packed indices + norms.

        Uses fused Metal kernels for normalize+signflip and quantize+pack.

        Returns:
            packed: (B, H, L, n_words) uint32
            norms: (B, H, L, 1) float32
        """
        B, H, L, D = x.shape
        flat = x.reshape(-1, D)  # (B*H*L, D)

        # Compute norms
        norms = mx.linalg.norm(flat, axis=-1, keepdims=True)

        # Fused normalize + sign-flip (Metal kernel)
        norms_flat = norms.reshape(-1)
        flipped = fused_normalize_signflip(flat, norms_flat, self.rotation.signs, D)

        # WHT rotation (optimized matmul — keep on GPU)
        y = flipped @ self.rotation.hadamard.T

        # Fused quantize + pack (Metal kernel)
        packed = fused_quantize_pack(y, boundaries, bits)

        # Reshape back
        n_words = packed.shape[-1]
        packed = packed.reshape(B, H, L, n_words)
        norms = norms.reshape(B, H, L, 1)

        return packed, norms

    def _dequantize(
        self, packed: mx.array, norms: mx.array, centroids: mx.array, bits: int
    ):
        """Dequantize packed indices + norms → (B, H, T, D) tensor.

        Uses fused Metal kernels for unpack+lookup and signflip+scale.
        Per-row unpacking to handle padding correctly.
        """
        B, H, T, n_words = packed.shape
        N = B * H * T
        D = self.head_dim

        # Unpack per-row to handle padding correctly
        # Each row of n_words unpacks to vpw*n_words values, then trim to D
        flat_packed = packed.reshape(N, n_words)

        # Use Python path for unpack+lookup (row-aware)
        indices = unpack_indices(flat_packed, bits, D)
        y_hat = centroids[indices]  # (N, D)

        # Inverse WHT rotation (optimized matmul)
        unrotated = y_hat @ self.rotation.hadamard

        # Fused sign-flip + norm scaling (Metal kernel)
        flat_norms = norms.reshape(-1)
        x_hat = fused_signflip_scale(unrotated, self.rotation.signs, flat_norms, D)

        return x_hat.reshape(B, H, T, self.head_dim)

    def size(self):
        return self.offset

    @property
    def state(self):
        if self._k_packed is None:
            return None
        return (self._k_packed, self._v_packed, self._k_norms, self._v_norms)

    @state.setter
    def state(self, v):
        if v is None:
            self._k_packed = None
            self._v_packed = None
            self._k_norms = None
            self._v_norms = None
            self.offset = 0
        else:
            self._k_packed, self._v_packed, self._k_norms, self._v_norms = v
            self.offset = self._k_packed.shape[2]

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no meta_state but a meta_state was set.")

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        if self._k_packed is not None:
            self._k_packed = self._k_packed[..., :self.offset, :]
            self._v_packed = self._v_packed[..., :self.offset, :]
            self._k_norms = self._k_norms[..., :self.offset, :]
            self._v_norms = self._v_norms[..., :self.offset, :]
        return n

    def make_mask(self, N, return_array=False, window_size=None):
        """Create attention mask compatible with mlx-lm."""
        from mlx_lm.models.base import create_causal_mask
        if N == 1 and not return_array:
            return None
        return create_causal_mask(N, offset=self.offset, window_size=window_size)

    def empty(self):
        return self._k_packed is None

    @property
    def nbytes(self):
        """Actual memory usage of compressed cache."""
        if self._k_packed is None:
            return 0
        return (
            self._k_packed.nbytes + self._v_packed.nbytes
            + self._k_norms.nbytes + self._v_norms.nbytes
        )

    @property
    def nbytes_uncompressed(self):
        """What the cache would use without compression (fp16)."""
        if self._k_packed is None:
            return 0
        B, H, T, _ = self._k_packed.shape
        # fp16 = 2 bytes per element
        return 2 * B * H * T * self.head_dim * 2  # *2 for K+V

    @property
    def compression_ratio(self):
        if self._k_packed is None:
            return 0.0
        uncompressed = self.nbytes_uncompressed
        if uncompressed == 0:
            return 0.0
        return uncompressed / self.nbytes


def _detect_head_dim(model) -> int:
    """Auto-detect head_dim by scanning model layers for attention modules."""
    for layer in model.layers:
        # Check common attention attribute names
        for attr_name in ('self_attn', 'attention', 'linear_attn'):
            attn = layer.get(attr_name) if hasattr(layer, 'get') else getattr(layer, attr_name, None)
            if attn is not None and hasattr(attn, 'head_dim'):
                return attn.head_dim
    return 128  # sensible default


def _layer_has_kv_cache(layer) -> bool:
    """Check if a layer uses a KV cache (full attention vs linear attention)."""
    # Layers with self_attn use standard KV caching
    if hasattr(layer, 'get'):
        return 'self_attn' in layer
    return hasattr(layer, 'self_attn')


def make_turboquant_cache(
    model,
    key_bits: int = 3,
    value_bits: int = 4,
    head_dim: Optional[int] = None,
    layer_adaptive: bool = True,
    seed: int = 42,
):
    """Create a list of cache instances for all model layers.

    For layers with full attention (self_attn), creates TurboQuantKVCache.
    For layers with linear attention (GatedDeltaNet etc.), creates standard
    mlx-lm KVCache since those layers manage their own state differently.

    Drop-in replacement for mlx_lm.models.cache.make_prompt_cache.

    Args:
        model: The mlx-lm model (must have .layers attribute)
        key_bits: Bits for key quantization
        value_bits: Bits for value quantization
        head_dim: Override head dimension (auto-detected if None)
        layer_adaptive: Use higher precision for final layers
        seed: Random seed for rotation

    Returns:
        List of cache objects, one per layer
    """
    from mlx_lm.models.cache import make_prompt_cache

    n_layers = len(model.layers)

    if head_dim is None:
        head_dim = _detect_head_dim(model)

    # Start with the model's default caches (handles hybrid architectures correctly)
    caches = make_prompt_cache(model)

    # Count full-attention layers for layer-adaptive calculation
    full_attn_indices = [i for i in range(n_layers) if _layer_has_kv_cache(model.layers[i])]
    n_full_attn = len(full_attn_indices)

    # Replace only full-attention layer caches with TurboQuant
    full_attn_pos = 0
    for i in full_attn_indices:
        caches[i] = TurboQuantKVCache(
            key_bits=key_bits,
            value_bits=value_bits,
            head_dim=head_dim,
            layer_idx=full_attn_pos,
            n_layers=n_full_attn,
            layer_adaptive=layer_adaptive,
            seed=seed,
        )
        full_attn_pos += 1

    return caches
