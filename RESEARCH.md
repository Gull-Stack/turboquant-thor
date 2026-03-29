# TurboQuant Deep Analysis — Paper Math, Implementation Review, Custom Build Plan

*Vision — 2026-03-29*

---

## Part 1: The Paper Math (Complete)

### The Core Problem
Compress d-dimensional vectors (KV cache entries, d=128 typically) to b bits per coordinate while preserving:
1. **MSE** — reconstruction accuracy: E[||x - x̃||²]
2. **Inner product** — attention score accuracy: E[|<y,x> - <y,x̃>|²]
3. **Unbiasedness** — E[<y,x̃>] = <y,x> (no systematic error)

### Algorithm 1: TurboQuant_mse (MSE-optimal)

**Step 1: Random Rotation**
- Generate Π ∈ R^{d×d} via QR decomposition of a Gaussian random matrix
- Compute y = Π·x (rotated vector)
- Key insight: Each coordinate y_j follows a Beta distribution:
  f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^{(d-3)/2}
- At d=128, this is very close to N(0, 1/d) = N(0, 1/128)
- Coordinates become nearly independent (not just uncorrelated — actually independent in high d)

**Step 2: Lloyd-Max Scalar Quantization**
- Since distribution is known and coordinates are independent, quantize each coordinate independently
- Solve the continuous 1D k-means problem for the Beta distribution:
  min Σ ∫ |x - c_i|² · f_X(x) dx
- Voronoi tessellation: boundaries are midpoints between consecutive centroids
- These codebooks are precomputed once per (d, b) pair — no data-dependent training

**Optimal centroids for N(0,1) (multiply by 1/√d for actual use):**
- 1-bit: {±0.7979} (2 centroids)
- 2-bit: {±0.4528, ±1.5104} (4 centroids)
- 3-bit: {±0.2451, ±0.7560, ±1.3439, ±2.1519} (8 centroids)
- 4-bit: 16 centroids, values in paper

**MSE distortion bound (Theorem 1):**
- D_mse ≤ (√3·π/2) · 1/4^b ≈ 2.72/4^b
- At b=1,2,3,4: D_mse ≈ 0.36, 0.117, 0.03, 0.009

**Information-theoretic lower bound (Theorem 3):**
- D_mse ≥ 1/4^b
- Gap factor: √3π/2 ≈ 2.72 (at high b), 1.45 (at b=1)
- This is provably near-optimal

**Dequantization:**
- Look up centroids for stored indices
- Rotate back: x̃ = Π^T · ỹ

### Algorithm 2: TurboQuant_prod (Inner-product optimal)

**Why MSE isn't enough:**
The MSE quantizer is biased for inner products. At b=1:
- E[<y, Q_mse^{-1}(Q_mse(x))>] = (2/π) · <y,x> — a multiplicative bias of 0.637!
- Bias decreases at higher b but never reaches zero

**The fix: Two-stage approach**
1. Apply TurboQuant_mse at (b-1) bits → get reconstruction x̃_mse
2. Compute residual: r = x - x̃_mse
3. Apply QJL to residual: qjl = sign(S·r), store ||r||₂
4. Combined estimator is unbiased

**QJL (Quantized Johnson-Lindenstrauss):**
- S ∈ R^{d×d} with i.i.d. N(0,1) entries
- Quantization: Q_qjl(x) = sign(S·x) — 1 bit per dimension
- Dequantization: Q_qjl^{-1}(z) = (√(π/2)/d) · S^T · z
- Unbiased: E[<y, Q_qjl^{-1}(Q_qjl(x))>] = <y,x>
- Variance: Var ≤ (π/2d) · ||y||² — decreases with dimension

**Combined inner product estimator:**
<y, x̃_mse> + ||r||₂ · √(π/2)/d · <S^T · qjl_signs, y>

**Storage for b-bit TurboQuant_prod:**
- (b-1) bits per coordinate for MSE indices
- 1 bit per coordinate for QJL signs
- 1 float for ||r||₂ per vector
- 1 float for ||x||₂ per vector (norm)
Total: b bits per coordinate + 2 floats per vector (amortized over d, negligible)

**Inner product distortion (Theorem 2):**
- D_prod ≤ (√3·π²·||y||²/d) · 1/4^b
- At b=3: D_prod ≈ 0.18/d — with d=128, this is ~0.0014 per score

### The Critical Subtlety: Why TurboQuant_prod Fails in Practice

The paper's math is correct. The theoretical estimator IS unbiased with low variance. But there's a problem the paper doesn't address:

**Softmax amplification.**

Attention scores go through softmax: weights = exp(score_i) / Σ exp(score_j). Small additive errors in scores become multiplicative errors in weights through the exponential. At 3-bit TurboQuant_prod, you're using 2-bit MSE (only 4 centroids). The MSE at 2-bit is 0.117 — that's 3.9x worse than 3-bit MSE (0.030). The QJL correction fixes the *expectation* but the *variance* of individual scores is much higher.

This is exactly what 0xSero and sharpner independently discovered: PPL 19.48 with TurboQuant_prod vs 13.60 with pure 3-bit MSE. The centroid resolution loss at (b-1) bits gets exponentially amplified by softmax.

**The lesson:** For KV cache compression, use TurboQuant_mse at full b bits. Use QJL as an *additive* correction (extra 1-bit on top of b-bit MSE), not as a *replacement* for the last MSE bit. The paper's Algorithm 2 is mathematically beautiful but practically inferior for attention.

---

## Part 2: Implementation Review — What Each Gets Right and Wrong

### TheTom/turboquant_plus (llama.cpp fork)

**Implementation approach:** C port with Metal GPU kernels, Walsh-Hadamard Transform (WHT) instead of QR rotation

**What's correct:**
- Uses WHT (O(d log d)) instead of dense QR rotation (O(d²)) — faster for the same mathematical effect
- Norm correction: stores norm/||centroid_vector|| instead of raw norm — eliminates bias from centroid magnitude mismatch. PPL improves from +1.6% to +1.1% at zero cost. This is NOT in the paper.
- Block-32 storage matches Metal's preferred memory access patterns
- Graph-side WHT rotation: does the rotation before the KV cache write, not during attention — saves O(T_kv) rotations during decode

**What's wrong/questionable:**
- Claims "3.25 bits/val" for turbo3 — the paper doesn't define this rate. Their encoding is 3 bits for MSE + 0.25 bits amortized for norms per group. Not wrong, just non-standard.
- Sparse V threshold at 1e-6 is arbitrary. No theoretical justification for this value. Works empirically but could be better.
- The "turbo4 beats q8_0 on NIAH" claim (31/33 vs 30/33) — N=33 is too small for statistical significance. Likely noise.

**Novel contributions (not from paper):**
1. Sparse V dequant — skipping V dequant where attention weight < threshold. Mathematically sound: if softmax(score_i) ≈ 0, then weight_i · v_i ≈ 0 regardless of v_i accuracy.
2. Layer-adaptive — last 8/40 layers at q8_0, rest at turbo3. Based on the empirical finding that later layers have sharper attention patterns and are more sensitive to quantization noise.
3. Temporal decay — requantize old tokens from turbo3→turbo2 as context grows. Relies on the observation that attention to distant tokens is naturally lower.

### tonbistudio/turboquant-pytorch

**Implementation approach:** Pure PyTorch, faithful to paper algorithms

**What's correct:**
- Lloyd-Max solver using scipy numerical integration — the proper way to compute codebooks
- Correctly implements both Algorithm 1 (MSE) and Algorithm 2 (inner product)
- Properly separates `TurboQuantMSE` and `TurboQuantProd` as the paper defines them
- Asymmetric attention score computation that doesn't dequantize keys fully — computes <y, x̃_mse> + QJL correction directly

**What's wrong:**
- Uses QR rotation via numpy (dense d×d matrix) — correct but slow. For d=128, this is 128×128=16K multiplies per vector. WHT would be 128×log₂(128) = 896 ops — 18x fewer.
- 3-bit packing stores as 4-bit (2 per byte): `bits=4  # round up to 4-bit packing`. This wastes 25% of the compression gain at 3-bit! The whole point of 3-bit is the compression ratio. Should pack 10 values per 32-bit word (3.2 bits/val) or use custom packing.
- No value quantization handling — they only quantize keys with TurboQuant_prod and values with MSE. But the paper doesn't discuss asymmetric K/V treatment, and the code doesn't explore whether values need different precision than keys.

**Key insight they surface:**
"Per-vector reconstruction error is 23-44% but inner product accuracy is 98%." This is the most important conceptual point about TurboQuant and most people miss it.

### 0xSero/turboquant (vLLM integration)

**Implementation approach:** Triton kernels + vLLM monkey-patch, NVIDIA GPU serving

**What's correct:**
- Honest self-audit (`audit_claims.py`) — the only implementation that adversarially tests its own claims
- Properly computes Q·centroid dot products as Triton kernels — avoids full dequantization during decode
- `searchsorted` for bucket assignment — O(log n) per coordinate, matches the paper's Voronoi tessellation
- Correctly identifies that TurboQuant_prod degradation comes from centroid resolution loss through softmax

**What's wrong:**
- Claims "5.1x compression" but admits it doesn't count Π (rotation matrix) and S (JL matrix) storage. At d=128, Π is 128×128×2 bytes = 32KB per attention head. With 32 heads × 40 layers, that's 40MB overhead. At 32K context, KV cache is ~200MB, so overhead is ~20%. Real compression is ~4.3x, not 5.1x.
- Value quantization uses group quantization (per-group scale + zero) instead of TurboQuant's rotation + Lloyd-Max. This is just standard GPTQ-style quantization, NOT TurboQuant for values. The 2-bit value cos_sim of 0.940 is worse than what TurboQuant 2-bit would achieve.
- "Hybrid decode dequantizes all history to float32" — defeats the memory bandwidth savings during decode. The compression only helps with capacity (fitting more tokens), not speed.

**Novel finding:**
Keys at 3-bit cos_sim = 1.000, values at 2-bit cos_sim = 0.940. **Keys are vastly more compressible than values.** This makes sense: keys participate in Q·K dot products (direction matters), values participate in weighted sums (magnitude matters). For our build: compress keys harder, keep values at higher precision.

### sharpner/turboquant-mlx

**Implementation approach:** Apple Silicon via MLX, two-path architecture

**What's correct:**
- Proper Lloyd-Max codebooks hardcoded for N(0,1) at b=1,2,3,4 — these match the paper's values exactly
- Combined rotation + JL matrix `build_combined_rot_jl`: precomputes [Π; S·Π] so query rotation and JL sketch happen in a single matmul. This is a smart optimization — saves one d×d matmul per query.
- Channel splitting for fractional bit rates: after rotation, channels are i.i.d., so a fixed split (e.g., 64@4-bit + 64@3-bit = 3.5 bits) works as well as dynamic outlier detection. Elegant.
- Norm-baking: `dequant(data, norm*scale, norm*bias)` eliminates 2 element-wise ops per token in the attention hot path
- Pre-allocated buffers with step=256 — O(T/256) allocations instead of O(T) concatenations

**What's wrong:**
- QR decomposition forced to CPU: `Q, R = mx.linalg.qr(G, stream=mx.cpu)` — this is an MLX limitation but it means initialization is slow. For a one-time cost this is fine, but if you ever need to regenerate rotations (e.g., per-layer different rotations), it becomes a bottleneck.
- V3 attention does full software dequant: `k_centroids = cache.get_key_centroids()` — materializes all T_kv × D centroids as float32. At 32K context with 128 heads, this is 32K × 128 × 4 = 16MB per head, per attention computation. This negates the memory savings during decode.
- `sign_correction: diag_sign = mx.sign(mx.diag(R)); Q = Q * diag_sign` — this ensures det(Q) = +1 (proper rotation, not reflection). Mathematically unnecessary for TurboQuant (reflection works equally well for uniformizing distribution), but good practice.
- The TurboQuant_prod implementation (QJL as bit replacement) shows the same degradation 0xSero found — confirming this is a fundamental issue, not implementation-specific.

**Novel discovery:**
V2 3-bit rot+QJL BEATS fp16 on Gemma 3 (D=256): PPL 12.05 vs 12.18. The rotation acts as a regularizer at large head dimensions. This is genuinely surprising — quantization normally only hurts quality. It suggests that for models with D≥256, the rotation-induced uniformization actually improves the conditioning of attention computation. Worth investigating further.

---

## Part 3: Mistakes and Gaps Across All Implementations

### Mistake 1: Everyone mishandles the rotation matrix storage
The paper says Π is a d×d dense matrix. Every implementation stores it as float32, which is 128×128×4 = 64KB per head. But Π only needs to be shared across all tokens in a layer (it's a fixed random matrix). The real cost is the matmul during quantization (O(d²) per vector). TheTom uses WHT (O(d log d)) which is better. Nobody uses structured rotation matrices (Hadamard + random sign flips) which give O(d log d) *and* require only d bits of storage (the sign flips).

### Mistake 2: No one exploits the Gaussian approximation properly
The paper proves coordinates are Beta-distributed but notes they converge to N(0, 1/d) at moderate d. At d=128, the Beta and Gaussian codebooks differ by <0.1%. sharpner hardcodes N(0,1) codebooks — correct and simpler. But nobody exploits the next step: since coordinates are ~N(0, 1/d), the optimal boundaries are symmetric, meaning you only need to store b-1 boundary values (the rest are reflections). This halves codebook storage.

### Mistake 3: Value quantization is undertreated
0xSero uses group quantization (not TurboQuant) for values. sharpner and TheTom use the same TurboQuant for both K and V. tonbistudio only compresses keys. But values have different statistical properties than keys — they carry magnitude information for the weighted sum output. Nobody has systematically compared TurboQuant_mse vs standard group quantization vs TurboQuant with asymmetric precision specifically for values.

### Mistake 4: The S matrix (QJL) is too large
The QJL projection matrix S ∈ R^{d×d} has d² random entries. At d=128, that's 64KB per head in float32. For 32 heads × 40 layers = 1280 matrices = 80MB just for QJL projection storage. This is wasteful. The JL property holds for *any* sub-Gaussian random matrix — you could use a sparse random matrix (only √d nonzeros per row) or a structured matrix (Hadamard + random diagonal) and get the same guarantees with O(d) storage.

### Mistake 5: Nobody implements fused attention with compressed keys
The paper's key insight is that you DON'T need to dequantize keys. The attention score <q, k̃> = Σ q_rot_j · centroid[idx_j]. Since centroids are shared across all tokens, you can precompute q_rot · centroid[i] for each of the 2^b centroids (only 8 values at 3-bit), then each key token's score is a sum of table lookups. This is O(d) additions instead of O(d) multiply-adds. TheTom mentions this as "fused Q·centroid decode" (issue #39) but nobody has implemented it.

---

## Part 4: Our Custom Build — M4 Pro 64GB

### Design Philosophy
Take the paper's proven math, strip out the parts that don't work in practice (TurboQuant_prod), add the community's proven optimizations (sparse V, layer-adaptive, norm correction), and optimize specifically for Apple M4 Pro's Metal compute.

### Architecture: "TurboQuant-Thor"

**Two paths, shared core:**

#### Path 1: MLX (Primary — for mlx-lm models)
Fork sharpner/turboquant-mlx. Add:

1. **Fused attention kernel (Metal)** — The #1 priority
   - Precompute `q_rot · centroid[i]` table (8 values at 3-bit, 16 at 4-bit)
   - Each key token's score = sum of table lookups from pre-computed table
   - Eliminates centroid dequant entirely — no float32 materialization
   - Expected: 3-5x decode speedup over V3's software dequant path

2. **Sparse V (from TheTom)** — Free decode speed
   - After softmax, skip V dequant where weight < threshold
   - Context-adaptive threshold: use percentile (bottom 50%) not fixed 1e-6
   - Per-layer thresholds: early layers broader attention → higher threshold
   - Expected: +15-25% decode at 16K+

3. **Asymmetric K/V (from 0xSero's finding)**
   - Keys: turbo3 (3-bit Lloyd-Max) — direction-preserving, cos_sim ≈ 1.0
   - Values: turbo4 (4-bit Lloyd-Max) — magnitude-preserving, cos_sim ≈ 0.997
   - Effective rate: 3.5 bits, compression 4.1x, quality near q8_0
   - Alternatively: keys turbo2, values turbo3 for maximum compression

4. **Layer-adaptive (from TheTom)**
   - Last 20% of layers at turbo4/q8_0
   - Rest at turbo3 K / turbo4 V
   - Effective compression ~3.5x with +0.15% PPL

5. **Structured rotation (optimization)**
   - Replace dense QR with Hadamard + random sign flips
   - O(d log d) instead of O(d²), and only d bits storage
   - Same Gaussianization quality at d=128

6. **Norm-baking (from sharpner)**
   - Fold norms into quantized representation
   - Eliminate 2 element-wise ops per token in SDPA

7. **Pre-allocated ring buffer**
   - Fixed capacity buffer, oldest tokens get temporal decay (turbo3→turbo2)
   - No allocations during inference after warmup

#### Path 2: llama.cpp (Secondary — for GGUF models)
Fork TheTom/llama-cpp-turboquant. Add:

1. Fused Q·centroid decode kernel (Metal) — fix M4 register spill
2. Adaptive sparse V thresholds
3. Asymmetric cache types (`--cache-type-k turbo3 --cache-type-v turbo4`)
4. Layer-adaptive as default (`--cache-adaptive`)
5. KV cache persistence (save/load for system prompts)

### What We DON'T Build
- TurboQuant_prod (Algorithm 2) — proven to be worse in practice
- CUDA backend — M4 Pro only
- Continuous batching — single user local inference
- MoE-aware compression — proven to not work (KV cache is pre-expert-routing)

### Validation Pipeline (from tonbistudio's methodology)
1. Load model, run forward pass, capture real KV cache at each layer
2. Compress with our implementation at each bit rate
3. Compare attention scores: cosine similarity, top-1/top-5 match, KLD
4. Run full perplexity on wikitext-2 (512 chunks) and wikitext-103 (50 chunks at 32K)
5. NIAH at 33+ positions × multiple context lengths (Kamradt methodology)
6. MT-Bench for task quality (reasoning, code, instruction following)
7. Self-audit: adversarial testing of our own claims (0xSero's approach)

### Expected Performance on M4 Pro 64GB

**Model:** Qwen 3.5 35B MoE (Q4_K_M GGUF / 4-bit MLX)
**Config:** turbo3 K + turbo4 V, layer-adaptive, sparse V enabled

| Metric | Expected | vs q8_0 |
|--------|----------|---------|
| KV cache size (32K ctx) | ~2.2 GB | 3.5x smaller |
| Max context | ~96K | ~3x more |
| Prefill tok/s | ~1400 | ~1.0x |
| Decode tok/s (8K) | ~65 | ~0.95x |
| Decode tok/s (32K) | ~45 | ~1.05x (sparse V helps) |
| PPL (wikitext-2) | +0.2% | near-lossless |
| NIAH single needle | 95%+ | comparable |

### File Structure
```
turboquant-thor/
├── core/
│   ├── codebook.py          # Lloyd-Max codebooks (hardcoded N(0,1))
│   ├── rotation.py          # Structured WHT + random signs
│   ├── quantizer.py         # TurboQuant_mse (no _prod)
│   └── sparse_v.py          # Adaptive sparse V dequant
├── mlx/
│   ├── cache.py             # MLX KV cache with ring buffer
│   ├── attention.py         # Fused attention with centroid table lookup
│   ├── kernels/             # Custom Metal shaders
│   │   ├── fused_score.metal    # Q·centroid table + sparse V
│   │   └── qjl_dot.metal       # Fused QJL sign-bit scoring
│   └── patch.py             # mlx-lm monkey-patch
├── llamacpp/
│   ├── ggml-turbo-quant.c   # C quantize/dequantize
│   └── ggml-metal.metal     # Metal GPU kernels (M4 optimized)
├── validation/
│   ├── capture_kv.py        # Real model KV capture
│   ├── compare_attention.py # Attention score comparison
│   ├── perplexity.py        # Wikitext PPL benchmark
│   ├── niah.py              # Needle-in-a-haystack
│   └── audit.py             # Self-audit (adversarial)
├── benchmarks/
│   ├── speed.py             # Prefill/decode benchmarks
│   └── memory.py            # Memory usage profiling
└── README.md
```

### Build Order
1. **Week 1:** Core math (codebook, rotation, quantizer) + validation pipeline
2. **Week 2:** MLX cache + basic attention (V3-style software dequant)
3. **Week 3:** Fused Metal kernel for attention scoring
4. **Week 4:** Sparse V, layer-adaptive, asymmetric K/V
5. **Week 5:** llama.cpp path (fork + M4 kernel fixes)
6. **Week 6:** Benchmarking, self-audit, optimization

---

## Part 5: Summary of Key Findings

1. **TurboQuant's math is sound and near-optimal** — within 2.72x of Shannon's lower bound. The rotation → Beta distribution → Lloyd-Max pipeline is elegant and provably efficient.

2. **Algorithm 2 (TurboQuant_prod) fails in practice** — the (b-1)-bit centroid resolution loss is amplified exponentially by softmax. Two independent implementations confirmed this. Use full b-bit MSE + optional QJL as extra correction.

3. **Keys compress better than values** — cos_sim 1.000 at 3-bit for keys vs 0.940 at 2-bit for values. Always use asymmetric: lower bits for K, higher for V.

4. **Last 20% of layers cause ~100% of quality loss** — the most impactful finding for practical deployment. Keep those layers at higher precision.

5. **Sparse V is a general optimization** — works on any KV cache format, not just TurboQuant. Free decode speed at long context with no quality loss.

6. **Rotation can IMPROVE quality at D≥256** — sharpner's discovery that V2 3-bit beats fp16 on Gemma suggests rotation acts as a regularizer. Prefer models with larger head_dim.

7. **Fused Q·centroid scoring is the #1 unbuilt optimization** — eliminates dequantization entirely during decode. Nobody has built it yet. Highest impact for our M4 Pro.

8. **The S matrix (QJL) should be structured, not dense** — 80MB of random matrices across all layers is wasteful. Use Hadamard + random diagonal for same guarantees at O(d) storage.
