"""Fused Metal kernels for TurboQuant quantize/dequantize hot paths.

Replaces the Python-level quantize_to_indices + pack_indices and
unpack_indices + centroid_lookup chains with single Metal kernel calls,
eliminating intermediate array allocations and kernel launch overhead.

Uses mx.fast.metal_kernel for JIT-compiled Metal Shading Language.
"""

import mlx.core as mx

# ---------------------------------------------------------------------------
# Fused quantize + pack kernel
#
# Input:  rotated values (float32), boundaries (float32)
# Output: packed indices (uint32), matching the bit-packing scheme
#
# Each thread processes one uint32 output word (multiple values packed).
# ---------------------------------------------------------------------------

_FUSED_QUANTIZE_PACK_SOURCE = """
    uint word_idx = thread_position_in_grid.x;

    // Determine which values this word covers
    uint val_start = word_idx * vals_per_word;

    uint32_t packed = 0;
    for (uint i = 0; i < vals_per_word; i++) {
        uint val_idx = val_start + i;
        float val = (val_idx < n_values) ? rotated[val_idx] : 0.0f;

        // Binary search: count how many boundaries this value exceeds
        uint idx = 0;
        for (uint b = 0; b < n_boundaries; b++) {
            if (val > boundaries[b]) {
                idx = b + 1;
            }
        }

        packed |= (idx << (i * bits_param));
    }

    out[word_idx] = packed;
"""

_FUSED_QUANTIZE_PACK_HEADER = """
    constant uint bits_param [[function_constant(0)]];
    constant uint vals_per_word [[function_constant(1)]];
    constant uint n_values [[function_constant(2)]];
    constant uint n_boundaries [[function_constant(3)]];
"""


def _get_vals_per_word(bits: int) -> int:
    if bits == 1: return 32
    if bits == 2: return 16
    if bits == 3: return 10
    if bits == 4: return 8
    if bits == 5: return 6
    raise ValueError(f"Unsupported bits: {bits}")


# Cache kernels by bits to avoid re-JIT
_quantize_pack_kernels: dict[int, object] = {}
_dequant_unpack_kernels: dict[int, object] = {}


def _make_quantize_pack_kernel(bits: int):
    """Create a fused quantize+pack Metal kernel for the given bit width."""
    vpw = _get_vals_per_word(bits)
    n_centroids = 2 ** bits
    n_boundaries = n_centroids - 1

    # We use template parameters passed via the source directly
    # since function_constants aren't supported via the Python API.
    # Instead, we bake the constants into the source string.
    source = f"""
    uint word_idx = thread_position_in_grid.x;
    uint val_start = word_idx * {vpw}u;

    uint32_t packed = 0;
    for (uint i = 0; i < {vpw}u; i++) {{
        uint val_idx = val_start + i;

        // Padding values beyond n_values get index 0 (matching Python pack_indices)
        uint idx = 0;
        if (val_idx < (uint)n_values) {{
            float val = rotated[val_idx];
            for (uint b = 0; b < {n_boundaries}u; b++) {{
                if (val > boundaries[b]) {{
                    idx = b + 1;
                }}
            }}
        }}

        packed |= (idx << (i * {bits}u));
    }}

    out[word_idx] = packed;
"""

    kernel = mx.fast.metal_kernel(
        name=f"fused_quantize_pack_{bits}bit",
        input_names=["rotated", "boundaries", "n_values"],
        output_names=["out"],
        source=source,
    )
    return kernel


def _make_dequant_unpack_kernel(bits: int):
    """Create a fused unpack+centroid-lookup Metal kernel for the given bit width."""
    vpw = _get_vals_per_word(bits)
    mask = (1 << bits) - 1

    source = f"""
    uint elem = thread_position_in_grid.x;
    uint word_idx = elem / {vpw}u;
    uint pos_in_word = elem % {vpw}u;

    uint32_t word = packed[word_idx];
    uint idx = (word >> (pos_in_word * {bits}u)) & {mask}u;

    out[elem] = centroids[idx];
"""

    kernel = mx.fast.metal_kernel(
        name=f"fused_dequant_unpack_{bits}bit",
        input_names=["packed", "centroids"],
        output_names=["out"],
        source=source,
    )
    return kernel


def fused_quantize_pack(rotated: mx.array, boundaries: mx.array, bits: int) -> mx.array:
    """Fused quantize + pack: rotated float values → packed uint32.

    Replaces: quantize_to_indices() + pack_indices()

    Args:
        rotated: (..., D) float32 rotated values
        boundaries: (2^bits - 1,) float32 decision boundaries
        bits: Quantization bits (1-5)

    Returns:
        packed: (..., n_words_per_row) uint32 packed indices
    """
    if bits not in _quantize_pack_kernels:
        _quantize_pack_kernels[bits] = _make_quantize_pack_kernel(bits)

    kernel = _quantize_pack_kernels[bits]
    vpw = _get_vals_per_word(bits)

    # Handle 1-D input: treat as single row
    if rotated.ndim == 1:
        rotated = rotated.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    # Flatten all batch dims, keep last dim (D)
    orig_shape = rotated.shape
    D = orig_shape[-1]
    batch_size = rotated.reshape(-1, D).shape[0]
    flat = rotated.reshape(batch_size, D)

    # Per-row packing (matching Python pack_indices behavior)
    words_per_row = (D + vpw - 1) // vpw
    n_total_words = batch_size * words_per_row

    # Flatten for kernel but pass D as n_values so padding is per-row
    # We dispatch one thread per output word, and compute row/col from word_idx
    n_values_arr = mx.array(D, dtype=mx.uint32)

    # We need a kernel that knows about rows. Rebuild with row awareness.
    row_source = f"""
    uint word_idx = thread_position_in_grid.x;
    uint words_per_row = {words_per_row}u;
    uint row = word_idx / words_per_row;
    uint col_word = word_idx % words_per_row;
    uint val_start = col_word * {vpw}u;
    uint row_offset = row * (uint)n_values;

    uint32_t packed = 0;
    for (uint i = 0; i < {vpw}u; i++) {{
        uint val_idx = val_start + i;

        uint idx = 0;
        if (val_idx < (uint)n_values) {{
            float val = rotated[row_offset + val_idx];
            for (uint b = 0; b < {2**bits - 1}u; b++) {{
                if (val > boundaries[b]) {{
                    idx = b + 1;
                }}
            }}
        }}

        packed |= (idx << (i * {bits}u));
    }}

    out[word_idx] = packed;
"""
    # Create row-aware kernel (cache by bits)
    row_kernel_key = (bits, "row")
    if row_kernel_key not in _quantize_pack_kernels:
        _quantize_pack_kernels[row_kernel_key] = mx.fast.metal_kernel(
            name=f"fused_quantize_pack_{bits}bit_row",
            input_names=["rotated", "boundaries", "n_values"],
            output_names=["out"],
            source=row_source,
        )
    row_kernel = _quantize_pack_kernels[row_kernel_key]

    outputs = row_kernel(
        inputs=[flat.reshape(-1), boundaries, n_values_arr],
        output_shapes=[(n_total_words,)],
        output_dtypes=[mx.uint32],
        grid=(n_total_words, 1, 1),
        threadgroup=(min(256, n_total_words), 1, 1),
    )

    result = outputs[0].reshape(*orig_shape[:-1], words_per_row)
    if squeeze:
        result = result.reshape(-1)
    return result


def fused_dequant_unpack(packed: mx.array, centroids: mx.array, bits: int, n_values: int) -> mx.array:
    """Fused unpack + centroid lookup: packed uint32 → float32 centroid values.

    Replaces: unpack_indices() + centroids[indices]

    Args:
        packed: (n_words,) uint32 packed indices
        centroids: (2^bits,) float32 centroid values
        bits: Quantization bits (1-5)
        n_values: Original number of values (for trimming padding)

    Returns:
        values: (n_values,) float32 centroid values
    """
    if bits not in _dequant_unpack_kernels:
        _dequant_unpack_kernels[bits] = _make_dequant_unpack_kernel(bits)

    kernel = _dequant_unpack_kernels[bits]
    vpw = _get_vals_per_word(bits)

    # We dispatch enough threads for all values including padding
    n_padded = packed.shape[0] * vpw

    outputs = kernel(
        inputs=[packed, centroids],
        output_shapes=[(n_padded,)],
        output_dtypes=[mx.float32],
        grid=(n_padded, 1, 1),
        threadgroup=(min(256, n_padded), 1, 1),
    )

    # Trim to original size
    return outputs[0][:n_values]


# ---------------------------------------------------------------------------
# Fused normalize + sign-flip kernel (pre-WHT)
# ---------------------------------------------------------------------------

_normalize_signflip_kernel = None

def _get_normalize_signflip_kernel():
    global _normalize_signflip_kernel
    if _normalize_signflip_kernel is None:
        source = """
    uint elem = thread_position_in_grid.x;
    uint vec_idx = elem / head_dim_val;
    uint coord = elem % head_dim_val;

    float val = x[elem];
    float norm = norms[vec_idx];
    float safe_norm = (norm < 1e-8f) ? 1.0f : norm;
    float sign = signs[coord];

    out[elem] = (val / safe_norm) * sign;
"""
        _normalize_signflip_kernel = mx.fast.metal_kernel(
            name="fused_normalize_signflip",
            input_names=["x", "norms", "signs", "head_dim_val"],
            output_names=["out"],
            source=source,
        )
    return _normalize_signflip_kernel


def fused_normalize_signflip(x: mx.array, norms: mx.array, signs: mx.array, head_dim: int) -> mx.array:
    """Fused normalize + sign-flip before WHT.

    Args:
        x: (N, D) input vectors
        norms: (N,) precomputed L2 norms
        signs: (D,) random ±1 signs

    Returns:
        out: (N, D) normalized and sign-flipped vectors
    """
    kernel = _get_normalize_signflip_kernel()
    n_total = x.shape[0] * x.shape[1]
    hd_arr = mx.array(head_dim, dtype=mx.uint32)

    outputs = kernel(
        inputs=[x.reshape(-1), norms, signs, hd_arr],
        output_shapes=[(n_total,)],
        output_dtypes=[mx.float32],
        grid=(n_total, 1, 1),
        threadgroup=(min(256, n_total), 1, 1),
    )
    return outputs[0].reshape(x.shape)


# ---------------------------------------------------------------------------
# Fused sign-flip + norm-scale kernel (post-inverse-WHT)
# ---------------------------------------------------------------------------

_signflip_scale_kernel = None

def _get_signflip_scale_kernel():
    global _signflip_scale_kernel
    if _signflip_scale_kernel is None:
        source = """
    uint elem = thread_position_in_grid.x;
    uint vec_idx = elem / head_dim_val;
    uint coord = elem % head_dim_val;

    float val = y[elem];
    float sign = signs[coord];
    float norm = norms[vec_idx];

    out[elem] = val * sign * norm;
"""
        _signflip_scale_kernel = mx.fast.metal_kernel(
            name="fused_signflip_scale",
            input_names=["y", "signs", "norms", "head_dim_val"],
            output_names=["out"],
            source=source,
        )
    return _signflip_scale_kernel


def fused_signflip_scale(y: mx.array, signs: mx.array, norms: mx.array, head_dim: int) -> mx.array:
    """Fused sign-flip + norm scaling after inverse WHT.

    Args:
        y: (N, D) vectors after inverse WHT
        signs: (D,) random ±1 signs
        norms: (N,) original L2 norms

    Returns:
        out: (N, D) reconstructed vectors
    """
    kernel = _get_signflip_scale_kernel()
    n_total = y.shape[0] * y.shape[1]
    hd_arr = mx.array(head_dim, dtype=mx.uint32)

    outputs = kernel(
        inputs=[y.reshape(-1), signs, norms, hd_arr],
        output_shapes=[(n_total,)],
        output_dtypes=[mx.float32],
        grid=(n_total, 1, 1),
        threadgroup=(min(256, n_total), 1, 1),
    )
    return outputs[0].reshape(y.shape)
