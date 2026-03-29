# TurboQuant-Thor — Build Specification

## What This Is
Custom implementation of Google's TurboQuant (ICLR 2026) KV cache compression, optimized for Apple M4 Pro 64GB. Takes the best pieces from 4 community implementations and the original paper, adds novel optimizations nobody has built yet.

## Target Hardware
- Apple M4 Pro, 64GB unified memory, 1TB SSD
- Metal GPU compute (no CUDA)
- Primary framework: MLX (Apple's ML framework)
- Secondary: llama.cpp fork (future)

## What To Build (Phase 1 — Core + MLX)

### 1. core/codebook.py — Lloyd-Max Optimal Codebooks
Hardcoded optimal centroids and boundaries for N(0,1) distribution at b=1,2,3,4 bits.
These are mathematical constants computed via Lloyd-Max algorithm. No runtime computation.

Centroids for N(0,1) (scale by 1/√head_dim for actual use):
- 1-bit: {±0.7979}
- 2-bit: {±0.4528, ±1.5104}
- 3-bit: {±0.2451, ±0.7560, ±1.3439, ±2.1519}
- 4-bit: 16 centroids (see sharpner/turboquant-mlx codebook.py for exact values)

Functions:
- `get_codebook(bits, head_dim)` → (centroids, boundaries) scaled by 1/√head_dim
- `get_codebook_unscaled(bits)` → (centroids, boundaries) for N(0,1)

### 2. core/rotation.py — Structured Random Rotation
Use Walsh-Hadamard Transform (WHT) + random sign flips instead of dense QR decomposition.
- O(d log d) instead of O(d²) compute
- O(d) storage (just the random signs) instead of O(d²)
- Same Gaussianization quality at d≥64

Functions:
- `generate_rotation(head_dim, seed)` → rotation object (WHT + signs)
- `rotate_forward(x, rotation)` → rotated x
- `rotate_inverse(y, rotation)` → original space
- `generate_jl_matrix(head_dim, seed)` → JL projection matrix S (for optional QJL)
- `safe_normalize(x)` → (normalized, norms)

Implementation: Use the recursive Hadamard definition:
H_1 = [1]
H_2n = [[H_n, H_n], [H_n, -H_n]] / √2
Then multiply element-wise by random ±1 signs.
At d=128: 128 × 7 = 896 multiply-adds vs 128 × 128 = 16,384 for dense QR.

### 3. core/quantizer.py — TurboQuant MSE Quantizer
Implements Algorithm 1 from the paper ONLY. We do NOT implement Algorithm 2 (TurboQuant_prod) because it's proven to degrade quality through softmax amplification.

Class `TurboQuantMSE`:
- `__init__(head_dim, bits, seed)` — precompute rotation + codebook
- `quantize(x)` → QuantizedVector(indices, norms, bits)
  1. Normalize: x_unit = x / ||x||, store ||x||
  2. Rotate: y = WHT(x_unit, signs)
  3. Quantize: idx_j = nearest_centroid(y_j) for each coordinate
  4. Pack indices into bit-packed representation
- `dequantize(qv)` → reconstructed x
  1. Unpack indices → look up centroids
  2. Inverse rotate: x̃ = WHT_inverse(centroids, signs)
  3. Rescale by stored norms
- `quantize_asymmetric(keys, values, key_bits, value_bits)` — different precision for K/V

Bit packing:
- 2-bit: 16 values per uint32
- 3-bit: 10 values per uint32 (NOT 4-bit containers — don't waste 25%)
- 4-bit: 8 values per uint32

### 4. core/sparse_v.py — Adaptive Sparse V Dequantization
Skip V dequantization where attention weight is negligible. Original to TheTom/turboquant_plus, not in the paper.

Class `SparseVConfig`:
- `threshold_mode`: "fixed" (1e-6), "percentile" (bottom N%), or "adaptive" (per-layer)
- `base_percentile`: 50 (skip bottom 50% of attention weights)
- `layer_scale`: array of per-layer multipliers (early layers skip more, later layers skip less)

Function `apply_sparse_v(weights, v_dequant_fn, config)`:
1. After softmax, identify positions where weight < threshold
2. Only call v_dequant_fn for positions above threshold
3. Return weighted sum (skipped positions contribute 0)

Expected: +15-25% decode speed at 16K+ context, 0 quality loss.

### 5. mlx_integration/cache.py — MLX KV Cache with Compression
Drop-in replacement for mlx-lm's KVCache.

Class `TurboQuantKVCache`:
- Pre-allocated ring buffer (step=256)
- Asymmetric: configurable key_bits and value_bits
- Layer-adaptive: constructor takes `layer_idx` and `n_layers`, automatically uses higher precision for last 20% of layers
- Norm-baking: fold L2 norms into quantized scales (eliminates 2 ops in SDPA)
- Temporal decay: old tokens (beyond decay_threshold) get requantized to lower precision

Key methods:
- `update_and_fetch(keys, values)` → stores compressed, returns cache state
- `get_key_centroids()` → dequantized keys for attention scoring
- `get_value_centroids()` → dequantized values for output computation

Config defaults for M4 Pro:
- key_bits=3, value_bits=4 (asymmetric)
- layer_adaptive=True (last 20% at +1 bit)
- sparse_v=True (adaptive percentile)

### 6. mlx_integration/attention.py — TurboQuant SDPA
Modified scaled dot-product attention that works with compressed cache.

Function `turboquant_sdpa(queries, cache, scale, mask)`:
1. Rotate query: q_rot = WHT(q * scale, signs)
2. Score keys: use centroid lookup from cache (no full dequant)
3. Apply mask + softmax
4. Apply sparse V: skip negligible positions
5. Weighted sum of value centroids
6. Inverse rotate output

Matrix associativity optimization (from sharpner):
output = (weights @ v_centroids) @ Π  instead of  weights @ (v_centroids @ Π)
Saves O(T_kv × D²) → O(T_q × D²) for inverse rotation.

### 7. mlx_integration/patch.py — mlx-lm Integration
Monkey-patch mlx-lm's SDPA dispatch to use TurboQuant cache.

Function `apply(model, config)`:
- Detects TurboQuantKVCache objects
- Routes to turboquant_sdpa instead of default SDPA
- Transparent to the rest of mlx-lm

### 8. validation/capture_kv.py — Real Model Validation
Capture real KV cache from a forward pass and compare compressed vs original attention.

Flow:
1. Load model (e.g., Qwen2.5-3B or Llama-3.2-3B via mlx-lm)
2. Run forward pass on test text, hook attention layers to capture K,V tensors
3. Compress captured K,V with TurboQuant at various bit rates
4. Compute attention scores with both original and compressed K,V
5. Report: cosine similarity, top-1/top-5 match, KL divergence

### 9. validation/perplexity.py — Perplexity Benchmark
Wikitext-2 and wikitext-103 perplexity at multiple context lengths.

### 10. validation/niah.py — Needle In A Haystack
Kamradt methodology: 33+ needle positions × multiple context lengths.
Report retrieval accuracy for each (position, context_length, cache_type) triple.

### 11. validation/audit.py — Self-Audit
Adversarial testing of our own claims (inspired by 0xSero/turboquant).
Honestly evaluate: compression ratios (including overhead), speed claims, quality claims.

## Files to Create

```
turboquant-thor/
├── pyproject.toml            # Package config, deps: mlx, mlx-lm, numpy
├── README.md
├── SPEC.md                   # This file
├── core/
│   ├── __init__.py
│   ├── codebook.py           # Lloyd-Max codebooks
│   ├── rotation.py           # WHT + random signs
│   ├── quantizer.py          # TurboQuantMSE
│   ├── packing.py            # Bit packing (2/3/4-bit into uint32)
│   └── sparse_v.py           # Adaptive sparse V
├── mlx_integration/
│   ├── __init__.py
│   ├── cache.py              # TurboQuantKVCache
│   ├── attention.py          # Modified SDPA
│   └── patch.py              # mlx-lm monkey-patch
├── validation/
│   ├── __init__.py
│   ├── capture_kv.py
│   ├── perplexity.py
│   ├── niah.py
│   └── audit.py
├── benchmarks/
│   ├── speed.py
│   └── memory.py
├── tests/
│   ├── test_codebook.py
│   ├── test_rotation.py
│   ├── test_quantizer.py
│   ├── test_packing.py
│   └── test_sparse_v.py
└── run_demo.py               # Quick demo: compress + decompress + measure quality
```

## Dependencies
```
mlx >= 0.22
mlx-lm >= 0.20
numpy >= 1.24
scipy >= 1.10  # for codebook validation only
pytest >= 7.0
```

## Key Design Decisions
1. **No TurboQuant_prod** — proven to degrade through softmax. Use full b-bit MSE.
2. **WHT rotation, not QR** — 18x fewer ops, near-zero storage, same quality.
3. **Asymmetric K/V** — keys at 3-bit, values at 4-bit by default.
4. **Layer-adaptive** — last 20% of layers at higher precision automatically.
5. **Proper 3-bit packing** — 10 values per uint32, not wasteful 4-bit containers.
6. **Sparse V** — adaptive threshold, per-layer, free decode speed.
7. **Norm-baking** — fold norms into quantized representation.
8. **Pre-allocated buffers** — no per-token allocation during inference.

## Quality Targets
- PPL increase vs fp16: < 0.5% at 3.5-bit effective rate
- NIAH single needle: 90%+ at 32K context
- Cosine similarity of attention scores: > 0.995 at 4-bit values
- Decode speed: ≥ 0.9x of fp16 at 8K, ≥ 1.0x at 32K (sparse V advantage)

## Build Order
1. core/codebook.py + tests
2. core/rotation.py + tests (WHT implementation)
3. core/packing.py + tests (bit packing)
4. core/quantizer.py + tests
5. core/sparse_v.py + tests
6. mlx_integration/cache.py
7. mlx_integration/attention.py
8. mlx_integration/patch.py
9. validation/ scripts
10. run_demo.py
11. benchmarks/

Build ALL core/ and tests/ first, verify math is correct, then build mlx_integration/ integration.
