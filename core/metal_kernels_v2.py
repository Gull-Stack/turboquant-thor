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

def _make_wht_kernel(d: int, with_signs: bool = False):
    """Create a WHT butterfly kernel for dimension d.

    Each threadgroup handles one row. Threads within the group
    cooperate on log2(d) butterfly stages using shared memory.

    If with_signs=True, folds the random sign flip into the initial
    load — zero extra cost vs a separate sign-flip pass.
    """
    log2_d = int(math.log2(d))
    assert 2 ** log2_d == d, f"d must be power of 2, got {d}"

    scale_per_stage = "0.70710678118654752f"  # 1/sqrt(2)
    sign_suffix = "_signed" if with_signs else ""

    # Load: fold sign flip into first stage input for free
    if with_signs:
        load_line = "shared[tid] = inp[row_offset + tid] * signs[tid];"
        input_names = ["inp", "signs"]
    else:
        load_line = "shared[tid] = inp[row_offset + tid];"
        input_names = ["inp"]

    source = f"""
    uint row = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint row_offset = row * {d}u;

    threadgroup float shared[{d}];
    if (tid < {d}u) {{
        {load_line}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    if (tid < {d}u) {{
        out[row_offset + tid] = shared[tid];
    }}
"""

    kernel = mx.fast.metal_kernel(
        name=f"wht_butterfly{sign_suffix}_{d}",
        input_names=input_names,
        output_names=["out"],
        source=source,
    )
    return kernel


# Cache WHT kernels by (d, with_signs)
_wht_kernels: dict[tuple[int, bool], object] = {}


def fused_wht(x: mx.array, forward: bool = True, signs: mx.array = None) -> mx.array:
    """Apply Walsh-Hadamard Transform via butterfly Metal kernel.

    O(d log d) instead of O(d²) matmul.

    If signs is provided, the sign flip is folded into the initial load
    at zero extra cost — one fused pass instead of two separate ops.

    Args:
        x: (..., d) float32 input vectors (d must be power of 2)
        forward: True for forward WHT, False for inverse
        signs: Optional (d,) ±1 signs to fold into the transform

    Returns:
        y: (..., d) float32 transformed vectors
    """
    d = x.shape[-1]
    with_signs = signs is not None
    key = (d, with_signs)

    if key not in _wht_kernels:
        _wht_kernels[key] = _make_wht_kernel(d, with_signs=with_signs)

    kernel = _wht_kernels[key]

    orig_shape = x.shape
    flat = x.reshape(-1, d)
    N = flat.shape[0]

    tg_x = min(d, 256)

    inputs = [flat, signs] if with_signs else [flat]

    outputs = kernel(
        inputs=inputs,
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

    Optimizations:
    - Centroid LUTs loaded into threadgroup shared memory (tiny: 32-128 bytes)
    - Precomputed q[dim] * k_centroid[i] score table per dimension
      Turns dot product into: unpack index → table lookup → accumulate
    - Value centroids also in shared memory for weighted accumulation

    Per head:
    1. Load centroid LUTs into shared memory
    2. Precompute q_score_table[i] = q[dim] * k_centroid[i] for each thread's dim
    3. For each cached key: unpack index → table lookup → reduce → score
    4. Softmax over scores
    5. For each cached value: unpack index → shared LUT → weighted accumulation
    6. Output is in rotated space (inverse WHT done outside)
    """
    n_k_centroids = 2 ** k_bits
    n_v_centroids = 2 ** v_bits
    vpw_map = {1: 32, 2: 16, 3: 10, 4: 8, 5: 6}
    vpw_k = vpw_map[k_bits]
    vpw_v = vpw_map[v_bits]
    mask_k = (1 << k_bits) - 1
    mask_v = (1 << v_bits) - 1
    n_k_words = (d + vpw_k - 1) // vpw_k
    n_v_words = (d + vpw_v - 1) // vpw_v

    source = f"""
    uint dim = thread_position_in_threadgroup.x;
    uint head = thread_position_in_grid.y;

    uint head_offset_k = head * (uint)n_tokens * {n_k_words}u;
    uint head_offset_nk = head * (uint)n_tokens;

    // --- Load key centroid LUT into shared memory ({n_k_centroids} × 4 = {n_k_centroids * 4} bytes) ---
    threadgroup float k_lut[{n_k_centroids}];
    if (dim < {n_k_centroids}u) {{
        k_lut[dim] = k_centroids[dim];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Precompute q × centroid score table for this thread's dimension ---
    // Turns the dot product into: unpack index → table[index] → accumulate
    float q_val = query_rotated[head * {d}u + dim];
    float q_score_table[{n_k_centroids}];
    for (uint i = 0; i < {n_k_centroids}u; i++) {{
        q_score_table[i] = q_val * k_lut[i];
    }}

    // Precompute unpack constants for this thread's dimension
    uint k_word_col = dim / {vpw_k}u;
    uint k_shift = (dim % {vpw_k}u) * {k_bits}u;

    threadgroup float scores[4096];
    threadgroup float reduce_buf[{d}];

    // --- Key scoring via LUT ---
    // For each token: unpack key index → score table lookup → reduce across dims
    for (uint t = 0; t < (uint)n_tokens; t++) {{
        uint32_t k_word = k_packed[head_offset_k + t * {n_k_words}u + k_word_col];
        uint k_idx = (k_word >> k_shift) & {mask_k}u;

        // Score = q[dim] * centroid[k_idx] * norm — just a table lookup!
        float partial = q_score_table[k_idx] * k_norms[head_offset_nk + t];

        // Parallel reduction across dimensions
        reduce_buf[dim] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = {d}u / 2; s > 0; s >>= 1) {{
            if (dim < s) {{
                reduce_buf[dim] += reduce_buf[dim + s];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (dim == 0) {{
            scores[t] = reduce_buf[0];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Output raw scores — caller handles softmax + value accumulation
    if (dim == 0) {{
        for (uint t = 0; t < (uint)n_tokens; t++) {{
            out[head * (uint)n_tokens + t] = scores[t];
        }}
    }}
"""

    kernel = mx.fast.metal_kernel(
        name=f"fused_score_lut_{d}d_k{k_bits}",
        input_names=[
            "query_rotated",      # (n_heads, d)
            "k_packed",           # (n_heads, n_tokens, n_k_words) flattened
            "k_centroids",        # (2^k_bits,)
            "k_norms",            # (n_heads, n_tokens) flattened
            "n_tokens",           # scalar uint32
        ],
        output_names=["out"],     # (n_heads, n_tokens) raw scores
        source=source,
    )
    return kernel


_fused_score_kernels: dict[tuple[int, int], object] = {}


def fused_compressed_score(
    query_rotated: mx.array,
    k_packed: mx.array,
    k_centroids: mx.array,
    k_norms: mx.array,
    head_dim: int,
    k_bits: int,
) -> mx.array:
    """Fused key scoring directly on compressed cache via centroid LUT.

    Operates in rotated space — query must be pre-rotated.
    Returns raw attention scores (caller handles softmax + value accumulation).

    For decode only (single query token, T_q=1).

    Optimization: precomputes q[dim] * centroid[i] score table per thread,
    turning the dot product into index → table lookup → accumulate.
    Centroid LUT in threadgroup shared memory ({2**k_bits} × 4 bytes).

    Args:
        query_rotated: (n_heads, d) pre-rotated query
        k_packed: (n_heads, n_tokens, n_k_words) packed key indices
        k_centroids: (2^k_bits,) key centroid values
        k_norms: (n_heads, n_tokens) key norms
        head_dim: d
        k_bits: key quantization bits

    Returns:
        scores: (n_heads, n_tokens) raw attention scores
    """
    d = head_dim
    # We still need v_bits for the kernel signature but only use k_bits for scoring
    key = (d, k_bits)

    if key not in _fused_score_kernels:
        # Pass dummy v_bits — the kernel only uses k_bits for scoring now
        _fused_score_kernels[key] = _make_fused_attention_kernel(d, k_bits, k_bits)

    kernel = _fused_score_kernels[key]
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
            n_tokens_arr,
        ],
        output_shapes=[(n_heads * n_tokens,)],
        output_dtypes=[mx.float32],
        grid=(tg_x, n_heads, 1),
        threadgroup=(tg_x, 1, 1),
    )

    return outputs[0].reshape(n_heads, n_tokens)
