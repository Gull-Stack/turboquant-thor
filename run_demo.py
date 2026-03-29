#!/usr/bin/env python3
"""TurboQuant-Thor demo — compress vectors and measure quality.

Run: python run_demo.py
"""

import time
import numpy as np
import mlx.core as mx

from core.quantizer import TurboQuantMSE, AsymmetricQuantizer


def main():
    print("=" * 60)
    print("TurboQuant-Thor — KV Cache Compression Demo")
    print("=" * 60)

    d = 128
    n = 5000

    # Generate random unit vectors (simulate KV cache entries)
    mx.random.seed(42)
    x = mx.random.normal(shape=(n, d))
    x = x / mx.linalg.norm(x, axis=-1, keepdims=True)
    mx.eval(x)

    print(f"\nInput: {n} vectors, d={d}")
    print(f"Original size: {n * d * 2 / 1024:.1f} KB (fp16)")
    print()

    # Test each bit width
    print(f"{'Bits':>4}  {'MSE':>8}  {'Paper':>8}  {'Ratio':>6}  "
          f"{'CosSim':>7}  {'Size KB':>8}  {'Compress':>8}")
    print("-" * 70)

    for bits in [1, 2, 3, 4]:
        q = TurboQuantMSE(head_dim=d, bits=bits, seed=42)

        # Quantize
        t0 = time.time()
        qt = q.quantize(x)
        mx.eval(qt.packed_indices, qt.norms)
        t_quant = time.time() - t0

        # Dequantize
        x_hat = q.dequantize(qt)
        mx.eval(x_hat)

        # Metrics
        x_np = np.array(x)
        x_hat_np = np.array(x_hat)
        mse = np.mean(np.sum((x_np - x_hat_np) ** 2, axis=-1))

        # Cosine similarity of attention scores
        queries = np.random.randn(100, d).astype(np.float32)
        true_scores = queries @ x_np.T
        approx_scores = queries @ x_hat_np.T
        cos_sims = []
        for i in range(100):
            cs = np.dot(true_scores[i], approx_scores[i]) / (
                np.linalg.norm(true_scores[i]) * np.linalg.norm(approx_scores[i]) + 1e-10
            )
            cos_sims.append(cs)
        avg_cos_sim = np.mean(cos_sims)

        # Compression
        compressed_bytes = qt.packed_indices.nbytes + qt.norms.nbytes
        compressed_kb = compressed_bytes / 1024
        ratio = q.compression_ratio()
        paper_mse = q.theoretical_mse()

        print(f"{bits:>4}  {mse:>8.4f}  {paper_mse:>8.4f}  "
              f"{ratio:>5.1f}x  {avg_cos_sim:>7.4f}  "
              f"{compressed_kb:>7.1f}  {ratio:>7.1f}x")

    # Asymmetric demo
    print()
    print("=" * 60)
    print("Asymmetric K/V Compression (keys=3bit, values=4bit)")
    print("=" * 60)

    aq = AsymmetricQuantizer(head_dim=d, key_bits=3, value_bits=4, seed=42)
    keys = x[:n // 2]
    values = x[n // 2:]
    q_k, q_v = aq.quantize_kv(keys, values)

    keys_hat = aq.key_quantizer.dequantize(q_k)
    values_hat = aq.value_quantizer.dequantize(q_v)

    mse_k = np.mean(np.sum((np.array(keys) - np.array(keys_hat)) ** 2, axis=-1))
    mse_v = np.mean(np.sum((np.array(values) - np.array(values_hat)) ** 2, axis=-1))

    print(f"Key MSE (3-bit):   {mse_k:.4f}")
    print(f"Value MSE (4-bit): {mse_v:.4f}")
    print(f"Effective bits:    {aq.effective_bits():.1f}")
    print(f"Compression ratio: {aq.compression_ratio():.1f}x")

    # Memory savings example
    print()
    print("=" * 60)
    print("Memory Projection — Qwen 3.5 35B MoE on M4 Pro 64GB")
    print("=" * 60)

    n_layers = 40
    n_heads = 32  # KV heads
    context = 32768
    bytes_per_val = 2  # fp16

    fp16_cache = 2 * n_layers * n_heads * context * d * bytes_per_val  # K+V
    turbo_ratio = aq.compression_ratio()
    turbo_cache = fp16_cache / turbo_ratio

    print(f"Context length: {context:,}")
    print(f"fp16 KV cache:  {fp16_cache / 1024**3:.2f} GB")
    print(f"TurboQuant:     {turbo_cache / 1024**3:.2f} GB ({turbo_ratio:.1f}x smaller)")
    print(f"Memory saved:   {(fp16_cache - turbo_cache) / 1024**3:.2f} GB")

    max_ctx_fp16 = int(context * (64 * 1024**3 * 0.3) / fp16_cache)
    max_ctx_turbo = int(max_ctx_fp16 * turbo_ratio)
    print(f"\nMax context (30% of 64GB for KV):")
    print(f"  fp16:       ~{max_ctx_fp16:,} tokens")
    print(f"  TurboQuant: ~{max_ctx_turbo:,} tokens")

    print()
    print("Done. Run `python -m pytest tests/ -v` to verify math.")


if __name__ == "__main__":
    main()
