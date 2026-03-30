"""Fused Metal kernels v2 — WHT butterfly + fused attention on compressed cache.

v1 kernels fused element-wise ops but still used d×d matmul for WHT.
v2 replaces WHT matmul with O(d log d) in-place butterfly, and adds
a fused attention kernel that scores/accumulates directly on compressed
cache without materializing full decompressed K/V.

Key insight: pre-rotate query, operate in rotated space, one inverse
WHT on the output. Eliminates per-token WHT from both K scoring and V
accumulation.
"""

import math
import mlx.core as mx


# ---------------------------------------------------------------------------
# 1. Fused WHT butterfly kernel (forward and inverse)
#
# Walsh-Hadamard Transform via butterfly operations in threadgroup memory.
# O(d log d) instead of O(d²) matmul. For d=256: 1024 vs 65536 ops.
#
# Each threadgroup processes one vector of dimension d.
# Threads cooperate on butterfly stages with barriers.
# ---------------------------------------------------------------------------

def _make_wht_kernel(d: int, forward: bool = True):
    """Create a WHT butterfly kernel for dimension d.

    Each threadgroup handles one row. Threads within the group
    cooperate on log2(d) butterfly stages using shared memory.
    """
    log2_d = int(math.log2(d))
    assert 2 ** log2_d == d, f"d must be power of 2, got {d}"

    # Scale factor: 1/sqrt(2) per stage = 1/sqrt(d) total
    # For forward: accumulate scale across stages
    # For inverse: same (WHT is self-inverse up to scale)
    scale_per_stage = "0.70710678118654752f"  # 1/sqrt(2)

    direction = "forward" if forward else "inverse"

    source = f"""
    uint row = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint row_offset = row * {d}u;

    // Load into threadgroup memory
    threadgroup float shared[{d}];
    if (tid < {d}u) {{
        shared[tid] = inp[row_offset + tid];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Butterfly stages
    for (uint stage = 0; stage < {log2_d}u; stage++) {{
        uint stride = 1u << stage;
        uint block = stride << 1;

        if (tid < {d}u) {{
            uint block_idx = tid / block;
            uint pos_in_block = tid % block;

            if (pos_in_block < stride) {{
                uint i = block_idx * block + pos_in_block;
                uint j = i + stride;
                float a = shared[i];
                float b = shared[j];
                shared[i] = (a + b) * {scale_per_stage};
                shared[j] = (a - b) * {scale_per_stage};
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Write back
    if (tid < {d}u) {{
        out[row_offset + tid] = shared[tid];
    }}
"""

    kernel = mx.fast.metal_kernel(
        name=f"wht_butterfly_{direction}_{d}",
        input_names=["inp"],
        output_names=["out"],
        source=source,
    )
    return kernel


# Cache WHT kernels by (d, forward)
_wht_kernels: dict[tuple[int, bool], object] = {}


def fused_wht(x: mx.array, forward: bool = True) -> mx.array:
    """Apply Walsh-Hadamard Transform via butterfly Metal kernel.

    O(d log d) instead of O(d²) matmul.

    Args:
        x: (..., d) float32 input vectors (d must be power of 2)
        forward: True for forward WHT, False for inverse

    Returns:
        y: (..., d) float32 transformed vectors
    """
    d = x.shape[-1]
    key = (d, forward)

    if key not in _wht_kernels:
        _wht_kernels[key] = _make_wht_kernel(d, forward)

    kernel = _wht_kernels[key]

    # Flatten to (N, d)
    orig_shape = x.shape
    flat = x.reshape(-1, d)
    N = flat.shape[0]

    # Threadgroup: d threads per row (or 256, whichever is smaller)
    tg_x = min(d, 256)

    outputs = kernel(
        inputs=[flat],
        output_shapes=[(N * d,)],
        output_dtypes=[mx.float32],
        grid=(tg_x, N, 1),
        threadgroup=(tg_x, 1, 1),
    )

    return outputs[0].reshape(orig_shape)


# ---------------------------------------------------------------------------
# 2. Fused full quantize: normalize → sign-flip → WHT → quantize → pack
#
# Single kernel call replaces the entire quantize pipeline.
# ---------------------------------------------------------------------------

def _make_full_quantize_kernel(d: int, bits: int):
    """Create a fully fused quantize kernel.

    Each threadgroup processes one vector:
    1. Load vector, compute norm (parallel reduction)
    2. Normalize and apply sign flips
    3. WHT butterfly in shared memory
    4. Quantize against boundaries
    5. Pack into uint32 words
    """
    log2_d = int(math.log2(d))
    n_boundaries = 2 ** bits - 1

    vpw_map = {1: 32, 2: 16, 3: 10, 4: 8, 5: 6}
    vpw = vpw_map[bits]
    n_words = (d + vpw - 1) // vpw

    scale = "0.70710678118654752f"

    source = f"""
    uint row = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint row_offset = row * {d}u;

    threadgroup float shared[{d}];
    threadgroup float norm_shared[1];

    // Load vector
    float val = 0.0f;
    if (tid < {d}u) {{
        val = x[row_offset + tid];
        shared[tid] = val * val;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for L2 norm
    for (uint s = {d}u / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared[tid] += shared[tid + s];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (tid == 0) {{
        float norm = sqrt(shared[0]);
        norm_shared[0] = (norm < 1e-8f) ? 1.0f : norm;
        norms_out[row] = norm;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize + sign flip
    if (tid < {d}u) {{
        shared[tid] = (val / norm_shared[0]) * signs[tid];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // WHT butterfly
    for (uint stage = 0; stage < {log2_d}u; stage++) {{
        uint stride = 1u << stage;
        uint block = stride << 1;
        if (tid < {d}u) {{
            uint block_idx = tid / block;
            uint pos_in_block = tid % block;
            if (pos_in_block < stride) {{
                uint i = block_idx * block + pos_in_block;
                uint j = i + stride;
                float a = shared[i];
                float b = shared[j];
                shared[i] = (a + b) * {scale};
                shared[j] = (a - b) * {scale};
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Quantize + pack (one thread per output word)
    if (tid < {n_words}u) {{
        uint val_start = tid * {vpw}u;
        uint32_t packed = 0;
        for (uint i = 0; i < {vpw}u; i++) {{
            uint vi = val_start + i;
            uint idx = 0;
            if (vi < {d}u) {{
                float rotated_val = shared[vi];
                for (uint b = 0; b < {n_boundaries}u; b++) {{
                    if (rotated_val > boundaries[b]) {{
                        idx = b + 1;
                    }}
                }}
            }}
            packed |= (idx << (i * {bits}u));
        }}
        packed_out[row * {n_words}u + tid] = packed;
    }}
"""

    kernel = mx.fast.metal_kernel(
        name=f"full_quantize_{d}d_{bits}bit",
        input_names=["x", "signs", "boundaries"],
        output_names=["packed_out", "norms_out"],
        source=source,
    )
    return kernel, n_words


_full_quantize_kernels: dict[tuple[int, int], tuple] = {}


def fused_full_quantize(x: mx.array, signs: mx.array, boundaries: mx.array,
                        bits: int) -> tuple[mx.array, mx.array]:
    """Fully fused quantize: input vectors → packed uint32 + norms.

    Single GPU dispatch replaces: normalize → sign-flip → WHT → quantize → pack

    Args:
        x: (N, d) float32 input vectors
        signs: (d,) random ±1 signs
        boundaries: (2^bits - 1,) decision boundaries
        bits: Quantization bits

    Returns:
        packed: (N, n_words) uint32
        norms: (N,) float32
    """
    d = x.shape[-1]
    key = (d, bits)

    if key not in _full_quantize_kernels:
        _full_quantize_kernels[key] = _make_full_quantize_kernel(d, bits)

    kernel, n_words = _full_quantize_kernels[key]

    orig_shape = x.shape
    flat = x.reshape(-1, d)
    N = flat.shape[0]

    tg_x = min(d, 256)

    outputs = kernel(
        inputs=[flat, signs, boundaries],
        output_shapes=[(N * n_words,), (N,)],
        output_dtypes=[mx.uint32, mx.float32],
        grid=(tg_x, N, 1),
        threadgroup=(tg_x, 1, 1),
    )

    packed = outputs[0].reshape(N, n_words)
    norms = outputs[1]
    return packed, norms


# ---------------------------------------------------------------------------
# 3. Fused full dequantize: unpack → centroid lookup → inverse WHT → sign-flip → scale
# ---------------------------------------------------------------------------

def _make_full_dequantize_kernel(d: int, bits: int):
    log2_d = int(math.log2(d))
    vpw_map = {1: 32, 2: 16, 3: 10, 4: 8, 5: 6}
    vpw = vpw_map[bits]
    mask = (1 << bits) - 1
    n_words = (d + vpw - 1) // vpw
    scale = "0.70710678118654752f"

    source = f"""
    uint row = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;

    threadgroup float shared[{d}];

    // Unpack + centroid lookup
    if (tid < {d}u) {{
        uint word_idx = tid / {vpw}u;
        uint pos_in_word = tid % {vpw}u;
        uint32_t word = packed[row * {n_words}u + word_idx];
        uint idx = (word >> (pos_in_word * {bits}u)) & {mask}u;
        shared[tid] = centroids[idx];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inverse WHT butterfly (same as forward — WHT is self-inverse up to scale)
    for (uint stage = 0; stage < {log2_d}u; stage++) {{
        uint stride = 1u << stage;
        uint block = stride << 1;
        if (tid < {d}u) {{
            uint block_idx = tid / block;
            uint pos_in_block = tid % block;
            if (pos_in_block < stride) {{
                uint i = block_idx * block + pos_in_block;
                uint j = i + stride;
                float a = shared[i];
                float b = shared[j];
                shared[i] = (a + b) * {scale};
                shared[j] = (a - b) * {scale};
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Sign-flip + norm scale
    if (tid < {d}u) {{
        out[row * {d}u + tid] = shared[tid] * signs[tid] * norms[row];
    }}
"""

    kernel = mx.fast.metal_kernel(
        name=f"full_dequantize_{d}d_{bits}bit",
        input_names=["packed", "centroids", "signs", "norms"],
        output_names=["out"],
        source=source,
    )
    return kernel, n_words


_full_dequantize_kernels: dict[tuple[int, int], tuple] = {}


def fused_full_dequantize(packed: mx.array, centroids: mx.array, signs: mx.array,
                          norms: mx.array, bits: int, head_dim: int) -> mx.array:
    """Fully fused dequantize: packed uint32 + norms → reconstructed vectors.

    Single GPU dispatch replaces: unpack → lookup → inverse WHT → sign-flip → scale

    Args:
        packed: (N, n_words) uint32
        centroids: (2^bits,) float32
        signs: (d,) random ±1 signs
        norms: (N,) float32
        bits: Quantization bits
        head_dim: Vector dimension d

    Returns:
        out: (N, d) float32 reconstructed vectors
    """
    d = head_dim
    key = (d, bits)

    if key not in _full_dequantize_kernels:
        _full_dequantize_kernels[key] = _make_full_dequantize_kernel(d, bits)

    kernel, n_words = _full_dequantize_kernels[key]

    N = packed.shape[0]
    tg_x = min(d, 256)

    outputs = kernel(
        inputs=[packed, centroids, signs, norms],
        output_shapes=[(N * d,)],
        output_dtypes=[mx.float32],
        grid=(tg_x, N, 1),
        threadgroup=(tg_x, 1, 1),
    )

    return outputs[0].reshape(N, d)


# ---------------------------------------------------------------------------
# 4. Fused attention: score keys + accumulate values directly on compressed cache
#
# Pre-rotate query, score against centroids in rotated space, weighted sum
# of value centroids, ONE inverse WHT on output.
#
# For decode (T_q=1), each threadgroup handles one query head.
# ---------------------------------------------------------------------------

def _make_fused_attention_kernel(d: int, k_bits: int, v_bits: int):
    """Fused attention kernel for decode (single query token).

    Operates directly on compressed K/V — no intermediate decompression.

    Per head:
    1. Query is pre-rotated (done outside kernel)
    2. For each cached key: unpack → centroid lookup → dot with query → score
    3. Softmax over scores
    4. For each cached value: unpack → centroid lookup → weighted accumulation
    5. Output is in rotated space (inverse WHT done outside for efficiency)
    """
    vpw_k_map = {1: 32, 2: 16, 3: 10, 4: 8, 5: 6}
    vpw_v_map = vpw_k_map.copy()
    vpw_k = vpw_k_map[k_bits]
    vpw_v = vpw_v_map[v_bits]
    mask_k = (1 << k_bits) - 1
    mask_v = (1 << v_bits) - 1
    n_k_words = (d + vpw_k - 1) // vpw_k
    n_v_words = (d + vpw_v - 1) // vpw_v

    source = f"""
    // Each thread in the group handles part of the dimension
    // Grid: (d, n_heads, 1), Threadgroup: (d, 1, 1)
    uint dim = thread_position_in_threadgroup.x;
    uint head = thread_position_in_grid.y;

    uint head_offset_k = head * (uint)n_tokens * {n_k_words}u;
    uint head_offset_v = head * (uint)n_tokens * {n_v_words}u;
    uint head_offset_norm_k = head * (uint)n_tokens;
    uint head_offset_norm_v = head * (uint)n_tokens;

    float q_val = query_rotated[head * {d}u + dim];

    threadgroup float scores[4096];  // max tokens
    threadgroup float shared_accum[{d}];

    // Phase 1: Compute attention scores (dot product in rotated space)
    // Each thread computes partial dot for one dimension, then reduce
    for (uint t = 0; t < (uint)n_tokens; t++) {{
        // Unpack key centroid for this token and dimension
        uint k_word_idx = dim / {vpw_k}u;
        uint k_pos = dim % {vpw_k}u;
        uint32_t k_word = k_packed[head_offset_k + t * {n_k_words}u + k_word_idx];
        uint k_idx = (k_word >> (k_pos * {k_bits}u)) & {mask_k}u;
        float k_centroid = k_centroids[k_idx];

        // Partial dot product: q[dim] * k_centroid * k_norm
        float partial = q_val * k_centroid * k_norms[head_offset_norm_k + t];

        // Reduce across dimensions using shared memory
        shared_accum[dim] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction
        for (uint s = {d}u / 2; s > 0; s >>= 1) {{
            if (dim < s) {{
                shared_accum[dim] += shared_accum[dim + s];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (dim == 0) {{
            scores[t] = shared_accum[0];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Phase 2: Softmax
    if (dim == 0) {{
        float max_score = -1e30f;
        for (uint t = 0; t < (uint)n_tokens; t++) {{
            max_score = max(max_score, scores[t]);
        }}
        float sum_exp = 0.0f;
        for (uint t = 0; t < (uint)n_tokens; t++) {{
            scores[t] = exp(scores[t] - max_score);
            sum_exp += scores[t];
        }}
        float inv_sum = 1.0f / (sum_exp + 1e-10f);
        for (uint t = 0; t < (uint)n_tokens; t++) {{
            scores[t] *= inv_sum;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Weighted sum of value centroids (in rotated space)
    float accum = 0.0f;
    for (uint t = 0; t < (uint)n_tokens; t++) {{
        uint v_word_idx = dim / {vpw_v}u;
        uint v_pos = dim % {vpw_v}u;
        uint32_t v_word = v_packed[head_offset_v + t * {n_v_words}u + v_word_idx];
        uint v_idx = (v_word >> (v_pos * {v_bits}u)) & {mask_v}u;
        float v_centroid = v_centroids[v_idx];

        accum += scores[t] * v_centroid * v_norms[head_offset_norm_v + t];
    }}

    out[head * {d}u + dim] = accum;
"""

    kernel = mx.fast.metal_kernel(
        name=f"fused_attention_{d}d_k{k_bits}_v{v_bits}",
        input_names=[
            "query_rotated",      # (n_heads, d)
            "k_packed",           # (n_heads, n_tokens, n_k_words) flattened
            "k_centroids",        # (2^k_bits,)
            "k_norms",            # (n_heads, n_tokens) flattened
            "v_packed",           # (n_heads, n_tokens, n_v_words) flattened
            "v_centroids",        # (2^v_bits,)
            "v_norms",            # (n_heads, n_tokens) flattened
            "n_tokens",           # scalar uint32
        ],
        output_names=["out"],     # (n_heads, d)
        source=source,
    )
    return kernel


_fused_attn_kernels: dict[tuple[int, int, int], object] = {}


def fused_compressed_attention(
    query_rotated: mx.array,
    k_packed: mx.array,
    k_centroids: mx.array,
    k_norms: mx.array,
    v_packed: mx.array,
    v_centroids: mx.array,
    v_norms: mx.array,
    head_dim: int,
    k_bits: int,
    v_bits: int,
) -> mx.array:
    """Fused attention scoring + accumulation directly on compressed KV cache.

    Operates in rotated space — query must be pre-rotated.
    Output is in rotated space — caller applies inverse WHT + sign-flip.

    For decode only (single query token, T_q=1).

    Args:
        query_rotated: (n_heads, d) pre-rotated query
        k_packed: (n_heads, n_tokens, n_k_words) packed key indices
        k_centroids: (2^k_bits,) key centroid values
        k_norms: (n_heads, n_tokens) key norms
        v_packed: (n_heads, n_tokens, n_v_words) packed value indices
        v_centroids: (2^v_bits,) value centroid values
        v_norms: (n_heads, n_tokens) value norms
        head_dim: d
        k_bits: key quantization bits
        v_bits: value quantization bits

    Returns:
        out: (n_heads, d) attention output in rotated space
    """
    d = head_dim
    key = (d, k_bits, v_bits)

    if key not in _fused_attn_kernels:
        _fused_attn_kernels[key] = _make_fused_attention_kernel(d, k_bits, v_bits)

    kernel = _fused_attn_kernels[key]
    n_heads = query_rotated.shape[0]
    n_tokens = k_packed.shape[1]
    n_tokens_arr = mx.array(n_tokens, dtype=mx.uint32)

    tg_x = min(d, 256)

    outputs = kernel(
        inputs=[
            query_rotated.reshape(-1),
            k_packed.reshape(-1),
            k_centroids,
            k_norms.reshape(-1),
            v_packed.reshape(-1),
            v_centroids,
            v_norms.reshape(-1),
            n_tokens_arr,
        ],
        output_shapes=[(n_heads * d,)],
        output_dtypes=[mx.float32],
        grid=(tg_x, n_heads, 1),
        threadgroup=(tg_x, 1, 1),
    )

    return outputs[0].reshape(n_heads, d)
