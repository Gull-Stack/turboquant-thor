#!/usr/bin/env python3
"""Live test: Qwen3.5-27B with TurboQuant compressed KV cache.

Runs the model with both standard and TurboQuant caches, comparing:
- Generation quality (same prompt, both paths)
- Memory usage (compressed vs uncompressed)
- Token generation speed

Usage:
    python run_qwen_turboquant.py
"""

import time
import mlx.core as mx

MODEL_ID = "mlx-community/Qwen3.5-27B-4bit"
PROMPT = "Explain how KV cache compression works in large language models in 3 sentences."
MAX_TOKENS = 200


def generate_with_cache(model, tokenizer, prompt, cache, max_tokens=200):
    """Generate tokens using the given cache, returning text and stats."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = mx.array(tokenizer.encode(text))[None]  # (1, seq_len)

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
    """Get total memory usage of a cache list in bytes.

    Args:
        cache_list: List of cache objects
        cache_type: If provided, only count caches of this type
    """
    total = 0
    for c in cache_list:
        if cache_type is not None and not isinstance(c, cache_type):
            continue
        if hasattr(c, 'nbytes'):
            total += c.nbytes
    return total


def main():
    print("=" * 70)
    print("Qwen3.5-27B + TurboQuant KV Cache — Live Comparison")
    print("=" * 70)
    print()

    # Load model
    print(f"Loading {MODEL_ID}...")
    from mlx_lm.utils import load
    model, tokenizer = load(MODEL_ID)
    n_layers = len(model.layers)
    print(f"  Loaded: {n_layers} layers")
    print()

    # --- Standard cache ---
    print("-" * 70)
    print("1. STANDARD KV CACHE (fp16)")
    print("-" * 70)
    from mlx_lm.models.cache import make_prompt_cache
    std_cache = make_prompt_cache(model)

    std_result = generate_with_cache(model, tokenizer, PROMPT, std_cache, MAX_TOKENS)
    std_mem = get_cache_memory(std_cache)

    print(f"  Prefill: {std_result['prefill_tokens']} tokens in {std_result['prefill_time']:.2f}s "
          f"({std_result['prefill_tps']:.1f} tok/s)")
    print(f"  Decode:  {std_result['decode_tokens']} tokens in {std_result['decode_time']:.2f}s "
          f"({std_result['decode_tps']:.1f} tok/s)")
    print(f"  Total cache memory: {std_mem / 1024 / 1024:.2f} MB")
    print(f"\n  Output:\n  {std_result['text'][:500]}")
    print()

    # Clear standard cache from memory
    del std_cache
    mx.clear_cache()

    # --- TurboQuant cache ---
    print("-" * 70)
    print("2. TURBOQUANT KV CACHE (keys=4bit, values=5bit, ~4.5 effective)")
    print("-" * 70)
    from mlx_integration.cache import TurboQuantKVCache, make_turboquant_cache
    from mlx_integration.patch import apply_turboquant, remove_turboquant

    tq_cache = make_turboquant_cache(model, key_bits=4, value_bits=5)
    apply_turboquant(model, sparse_v=True)

    # Count how many layers use TurboQuant vs standard
    n_tq = sum(1 for c in tq_cache if isinstance(c, TurboQuantKVCache))
    n_std = len(tq_cache) - n_tq
    print(f"  {n_tq} full-attention layers (TurboQuant) + {n_std} linear-attention layers (standard)")

    tq_result = generate_with_cache(model, tokenizer, PROMPT, tq_cache, MAX_TOKENS)
    tq_mem = get_cache_memory(tq_cache)
    tq_attn_mem = get_cache_memory(tq_cache, cache_type=TurboQuantKVCache)

    print(f"  Prefill: {tq_result['prefill_tokens']} tokens in {tq_result['prefill_time']:.2f}s "
          f"({tq_result['prefill_tps']:.1f} tok/s)")
    print(f"  Decode:  {tq_result['decode_tokens']} tokens in {tq_result['decode_time']:.2f}s "
          f"({tq_result['decode_tps']:.1f} tok/s)")
    print(f"  Total cache memory: {tq_mem / 1024 / 1024:.2f} MB")
    print(f"    TurboQuant attention layers: {tq_attn_mem / 1024 / 1024:.2f} MB")

    # Report per-layer compression for TurboQuant layers
    tq_layers = [c for c in tq_cache if isinstance(c, TurboQuantKVCache) and not c.empty()]
    if tq_layers:
        avg_ratio = sum(c.compression_ratio for c in tq_layers) / len(tq_layers)
        print(f"    Avg compression ratio: {avg_ratio:.1f}x")

    print(f"\n  Output:\n  {tq_result['text'][:500]}")
    print()

    # Restore original SDPA
    remove_turboquant()

    # --- Comparison ---
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  Total cache memory:  {std_mem/1024/1024:.2f} MB → {tq_mem/1024/1024:.2f} MB")
    print(f"  Decode speed:  {std_result['decode_tps']:.1f} → {tq_result['decode_tps']:.1f} tok/s")
    print(f"  Prefill speed: {std_result['prefill_tps']:.1f} → {tq_result['prefill_tps']:.1f} tok/s")

    # Memory projection for long context
    total_tokens = std_result['prefill_tokens'] + std_result['decode_tokens']
    print()
    print(f"  Long context projection (32K tokens, {n_layers} layers):")
    if total_tokens > 0:
        std_per_tok = std_mem / total_tokens
        tq_per_tok = tq_mem / total_tokens
        std_32k = std_per_tok * 32768 / 1024 / 1024 / 1024
        tq_32k = tq_per_tok * 32768 / 1024 / 1024 / 1024
        print(f"    Standard: {std_32k:.2f} GB")
        print(f"    TurboQuant: {tq_32k:.2f} GB")
        savings_pct = (1 - tq_32k / std_32k) * 100 if std_32k > 0 else 0
        print(f"    Savings: {std_32k - tq_32k:.2f} GB ({savings_pct:.0f}%)")

    print()
    print("  Note: Qwen3.5 uses hybrid architecture (75% GatedDeltaNet + 25% full attention)")
    print("  TurboQuant compresses the 16 full-attention layer caches. DeltaNet layers")
    print("  have minimal KV cache overhead by design.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
