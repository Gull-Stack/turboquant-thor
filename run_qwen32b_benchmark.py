#!/usr/bin/env python3
"""Qwen3-32B Dense Attention Benchmark — TurboQuant vs Standard KV Cache.

Full dense-attention model (64/64 layers with KV cache).
This is the validation step before scaling to 235B.
"""

import time
import mlx.core as mx

MODEL_ID = "mlx-community/Qwen3-32B-4bit"
PROMPT = "Write a Python function that checks if a number is prime. Include docstring and type hints."
MAX_TOKENS = 300


def generate_with_cache(model, tokenizer, prompt, cache, max_tokens):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = mx.array(tokenizer.encode(text))[None]

    # Prefill
    t0 = time.perf_counter()
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    t_prefill = time.perf_counter() - t0
    prefill_toks = input_ids.shape[1]

    # Decode
    generated_tokens = []
    token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    generated_tokens.append(token.item())

    t_decode_start = time.perf_counter()
    for _ in range(max_tokens - 1):
        logits = model(token, cache=cache)
        mx.eval(logits)
        token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        tok_id = token.item()
        generated_tokens.append(tok_id)
        if tok_id == tokenizer.eos_token_id:
            break
    t_decode = time.perf_counter() - t_decode_start

    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    decode_toks = len(generated_tokens)

    return {
        "text": output_text,
        "prefill_tokens": prefill_toks,
        "prefill_time": t_prefill,
        "prefill_tps": prefill_toks / t_prefill,
        "decode_tokens": decode_toks,
        "decode_time": t_decode,
        "decode_tps": decode_toks / t_decode if t_decode > 0 else 0,
    }


def get_cache_memory(cache_list, cache_type=None):
    total = 0
    for c in cache_list:
        if cache_type is not None and not isinstance(c, cache_type):
            continue
        if hasattr(c, 'nbytes'):
            total += c.nbytes
    return total


def main():
    print("=" * 72)
    print("Qwen3-32B (DENSE) + TurboQuant — Full Compression Benchmark")
    print("=" * 72)
    print()

    from mlx_lm.utils import load
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = load(MODEL_ID)
    n_layers = len(model.layers)
    print(f"  {n_layers} layers, ALL full-attention (dense)")

    mem_model = mx.get_peak_memory() / 1024 / 1024 / 1024
    print(f"  Model weights: ~{mem_model:.1f} GB")
    print()

    # --- Config A: Standard ---
    print("-" * 72)
    print("Config A: STANDARD KV CACHE (bfloat16)")
    print("-" * 72)
    from mlx_lm.models.cache import make_prompt_cache
    std_cache = make_prompt_cache(model)

    std_result = generate_with_cache(model, tokenizer, PROMPT, std_cache, MAX_TOKENS)
    std_mem = get_cache_memory(std_cache)
    total_tokens = std_result['prefill_tokens'] + std_result['decode_tokens']

    print(f"  Prefill: {std_result['prefill_tokens']} tok in {std_result['prefill_time']:.2f}s "
          f"({std_result['prefill_tps']:.1f} tok/s)")
    print(f"  Decode:  {std_result['decode_tokens']} tok in {std_result['decode_time']:.2f}s "
          f"({std_result['decode_tps']:.1f} tok/s)")
    print(f"  KV cache: {std_mem / 1024 / 1024:.2f} MB ({total_tokens} tokens cached)")
    print(f"  Peak GPU memory: {mx.get_peak_memory() / 1024 / 1024 / 1024:.2f} GB")
    print(f"\n  Output preview:\n  {std_result['text'][:400]}")
    print()

    del std_cache
    mx.clear_cache()

    # --- Config B: TurboQuant ---
    print("-" * 72)
    print("Config B: TURBOQUANT (keys=4bit, values=5bit, ~4.5 effective)")
    print("-" * 72)
    from mlx_integration.cache import TurboQuantKVCache, make_turboquant_cache

    tq_cache = make_turboquant_cache(model, key_bits=4, value_bits=5)
    n_tq = sum(1 for c in tq_cache if isinstance(c, TurboQuantKVCache))
    print(f"  {n_tq}/{n_layers} layers using TurboQuant")

    tq_result = generate_with_cache(model, tokenizer, PROMPT, tq_cache, MAX_TOKENS)
    tq_mem = get_cache_memory(tq_cache)
    tq_attn_mem = get_cache_memory(tq_cache, cache_type=TurboQuantKVCache)
    tq_total_tokens = tq_result['prefill_tokens'] + tq_result['decode_tokens']

    print(f"  Prefill: {tq_result['prefill_tokens']} tok in {tq_result['prefill_time']:.2f}s "
          f"({tq_result['prefill_tps']:.1f} tok/s)")
    print(f"  Decode:  {tq_result['decode_tokens']} tok in {tq_result['decode_time']:.2f}s "
          f"({tq_result['decode_tps']:.1f} tok/s)")
    print(f"  KV cache: {tq_mem / 1024 / 1024:.2f} MB ({tq_total_tokens} tokens cached)")
    print(f"  Peak GPU memory: {mx.get_peak_memory() / 1024 / 1024 / 1024:.2f} GB")

    tq_layers = [c for c in tq_cache if isinstance(c, TurboQuantKVCache) and not c.empty()]
    if tq_layers:
        avg_ratio = sum(c.compression_ratio for c in tq_layers) / len(tq_layers)
        print(f"  Avg compression ratio: {avg_ratio:.1f}x")

    print(f"\n  Output preview:\n  {tq_result['text'][:400]}")
    print()

    # --- Comparison ---
    print("=" * 72)
    print("COMPARISON — Qwen3-32B Dense Attention")
    print("=" * 72)
    compression = std_mem / tq_mem if tq_mem > 0 else 0
    print(f"  Decode:     {std_result['decode_tps']:.1f} → {tq_result['decode_tps']:.1f} tok/s")
    print(f"  Prefill:    {std_result['prefill_tps']:.1f} → {tq_result['prefill_tps']:.1f} tok/s")
    print(f"  KV cache:   {std_mem/1024/1024:.1f} MB → {tq_mem/1024/1024:.1f} MB ({compression:.2f}x compression)")

    # Long context projections
    print()
    print("  Context projections (64 dense attention layers):")
    if total_tokens > 0 and tq_total_tokens > 0:
        std_per_tok = std_mem / total_tokens
        tq_per_tok = tq_mem / tq_total_tokens
        for ctx_len in [8192, 16384, 32768, 65536, 131072]:
            std_gb = std_per_tok * ctx_len / 1024 / 1024 / 1024
            tq_gb = tq_per_tok * ctx_len / 1024 / 1024 / 1024
            savings = std_gb - tq_gb
            print(f"    {ctx_len:>6d} tokens: Standard {std_gb:>5.1f} GB → TurboQuant {tq_gb:>5.1f} GB  (saves {savings:.1f} GB)")

    # 235B projections
    print()
    print("  235B-A22B projection (192 GB across 3x M4 Pro):")
    print(f"    Model weights (~4-bit): ~132 GB")
    print(f"    Remaining for KV + OS:  ~50 GB")
    if tq_total_tokens > 0:
        # Scale per-token KV cost for 235B (more layers, more heads)
        # Qwen3-235B-A22B: 94 layers, but MoE so only some have full attention
        # Estimate ~80 attention layers with similar head_dim
        scale = 80 / 64  # more layers than 32B
        tq_per_tok_235b = tq_per_tok * scale
        std_per_tok_235b = std_per_tok * scale
        max_ctx_std = int(50 * 1024 * 1024 * 1024 / std_per_tok_235b)
        max_ctx_tq = int(50 * 1024 * 1024 * 1024 / tq_per_tok_235b)
        print(f"    Max context (standard): ~{max_ctx_std:,} tokens")
        print(f"    Max context (TurboQuant): ~{max_ctx_tq:,} tokens")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
