"""ceiling.py - roofline ceiling for batch-1 LLM inference (Week 1).

Given a model config and a device spec, predicts the best TTFT, TPOT and
peak memory that any runtime could reach on that device. Measured runs are
then reported as "% of ceiling". Week 8 adds per-runtime efficiency factors;
this version is the pure hardware limit.

How it works
------------
1. Parameters, counted from the Hugging Face config.json. Works for
   Llama-style dense decoders (Llama, Qwen2/2.5/3, Mistral, SmolLM, Gemma,
   Phi-3):
     per layer  attention  q, k, v, o projections
                MLP        gate, up, down (3 x hidden x intermediate)
     once       embedding  vocab x hidden
                lm_head    vocab x hidden (the same matrix if tied)
   Norms and biases are under 0.1% of the weights and are ignored. The total
   is printed: compare it with the model card as a sanity check.

2. Bytes, from bits per weight for each part:
     --quant       transformer layers (nf4, q4_k, fp16, ... or a number)
     --head-quant  embedding and lm_head (bitsandbytes leaves them at fp16)
     --kv-bits     KV cache entries
   KV cache per token = 2 (K and V) x layers x kv_heads x head_dim x bytes.

3. Decode (TPOT). At batch 1, every new token must read all layer weights
   and the lm_head once, plus the KV cache so far. The embedding lookup
   reads one row, which is ignored.
     bytes = layer weights + lm_head + context x KV per token
     FLOPs = 2 x (layer + lm_head params) + 4 x layers x heads x head_dim x context
     TPOT  = max(bytes / bandwidth, FLOPs / FLOPS)
   Context is the average over the answer: prompt + output / 2.
   On the MX450 the bytes term wins by a wide margin: decode is memory-bound.
   On the CPU the two terms can be close (FP32 matmul measured only 0.10
   TFLOPS), so read the printed bound. llama.cpp's quantized CPU kernels use
   int8 dot products, not FP32, so the CPU compute term is the least certain.

4. Prefill (TTFT). One forward pass over the P prompt tokens, which also
   produces the first output token.
     FLOPs = 2 x layer params x P
           + 2 x lm_head params x (P with --all-logits, else 1)
           + 2 x layers x heads x head_dim x P^2     (causal attention)
     bytes = all weights once + P x KV per token     (writing the cache)
     TTFT  = max(FLOPs / FLOPS, bytes / bandwidth)
   --all-logits: llama.cpp and HF generate() compute logits only for the
   last prompt token. Stride calls model() directly, which computes them for
   every prompt token, so Stride needs --all-logits.
   FLOPS is the measured matmul rate in the device preset
   (bandwidth/matmul_flops.py); TTFT is null only if a device has none.
   Tokenization, Python and kernel launch overhead are not part of the
   ceiling.

5. Peak memory = weights + KV cache at full context (prompt + output)
   + logits buffer (rows x vocab x 2 bytes, fp16) + --overhead-mib.
   --overhead-mib (runtime context, activations) is 0 by default, so the
   predicted peak is a lower bound.

Units: bandwidth in GB/s (1e9 bytes), FLOPS in TFLOPS (1e12), memory in MiB
(2^20 bytes, the unit nvidia-smi and the MX450's 2048 MiB use).

How to use
----------
    python ceiling.py <path to config.json> [options]

config.json comes from the model's Hugging Face repo. After a download it sits
in ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<hash>/config.json.

Stride's baseline: Qwen2.5-1.5B-Instruct, bitsandbytes NF4 with double quant,
fp16 embedding/lm_head, fp16 KV cache, logits for every prompt token:

    python ceiling.py ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/<hash>/config.json --device mx450 --quant nf4 --head-quant fp16 --all-logits --prompt-tokens 128 --output-tokens 128

That uses the MX450's best measured matmul rate (FP32). Add --tflops 0.35
to model Stride's FP16 matmuls, which run 3x slower on this GPU.

Options:
    --device                 preset from DEVICES below: mx450 (default), i7-1165g7
    --bandwidth GB/s         overrides the preset; use measured READ bandwidth
    --tflops TFLOPS          usable matmul FLOPS; overrides the preset
    --memory-mib MiB         overrides the preset; used for the fits-in-memory check
    --quant, --head-quant    a name from QUANT_BITS or bits per weight as a number.
                             For mixes like Q4_K_M, pass the GGUF file's average:
                             file bytes x 8 / parameter count.
    --kv-bits                default 16
    --prompt-tokens          default 128
    --output-tokens          default 128
    --all-logits             lm_head runs on every prompt token in prefill
    --overhead-mib           fixed memory to add to the peak (default 0)

Output: a breakdown, then the `predicted` block of the results schema:
    {"ttft_ms": ..., "tpot_ms": ..., "peak_mem_mb": ..., "source": "ceiling.py 0.2"}
peak_mem_mb is in MiB.

predict() can also be imported: shape from load_shape(), the rest are numbers.
"""
import argparse
import json
from pathlib import Path

VERSION = "0.2"  # 0.2: measured TFLOPS in the device presets
MIB = 2**20

# Device specs, all measured (bandwidth/README.md).
#   bandwidth_gbs: read bandwidth, from bandwidth/gpu_bandwidth.py and cpu_stream.py
#   tflops:        4096 x 4096 matmul, best of 20, from bandwidth/matmul_flops.py
DEVICES = {
    # Bandwidth: plugged in, 2026-10-16. On battery the memory clock drops
    # and it reads 44.2 GB/s.
    # TFLOPS: FP32 matmul, the fastest path measured (on battery, 2026-10-17;
    # rerun plugged in). The FP16 matmul measured only 0.35 TFLOPS, 3x slower:
    # the MX450 has no tensor cores. Stride computes in FP16, so pass
    # --tflops 0.35 to model Stride's own path instead of the device's best.
    "mx450": {"bandwidth_gbs": 51.6, "tflops": 1.11, "memory_mib": 2048},
    # Bandwidth: 4 threads, 2026-10-17. Noisy (29-34 GB/s across runs);
    # replace after a rerun with other apps closed.
    # TFLOPS: FP32 matmul, 4 threads, on battery, 2026-10-17; rerun plugged in.
    "i7-1165g7": {"bandwidth_gbs": 30.6, "tflops": 0.10, "memory_mib": 16384},
}

# Bits per weight, including the per-block scales each format stores.
QUANT_BITS = {
    "fp32": 32,
    "fp16": 16,
    "bf16": 16,
    "q8_0": 8.5,      # 32 x int8 + one fp16 scale per block
    "q6_k": 6.5625,
    "q5_k": 5.5,
    "q4_k": 4.5,
    "q4_0": 4.5,      # 32 x 4-bit + one fp16 scale per block
    "nf4": 4.127,     # bitsandbytes with double quant: 4 + 8/64 + 32/(64*256)
    "nf4-nodq": 4.5,  # bitsandbytes without double quant: 4 + 32/64
}


def bits(value):
    """Turn a quant name from QUANT_BITS, or a plain number, into bits per weight."""
    if value in QUANT_BITS:
        return QUANT_BITS[value]
    return float(value)  # argparse reports a ValueError as an invalid value


def load_shape(config_path):
    """Read the model dimensions ceiling.py needs from a Hugging Face config.json."""
    cfg = json.loads(Path(config_path).expanduser().read_text())
    cfg = cfg.get("text_config", cfg)  # VLM configs nest the language model here
    if any(cfg.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts")):
        raise SystemExit("mixture-of-experts models are not modeled")
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    return {
        "layers": cfg["num_hidden_layers"],
        "hidden": hidden,
        "heads": heads,
        "kv_heads": cfg.get("num_key_value_heads", heads),  # missing = no GQA
        "head_dim": cfg.get("head_dim") or hidden // heads,
        "intermediate": cfg["intermediate_size"],
        "vocab": cfg["vocab_size"],
        "tied": cfg.get("tie_word_embeddings", False),
    }


def predict(shape, bandwidth_gbs, tflops, quant_bits, head_bits, kv_bits,
            prompt_tokens, output_tokens, all_logits=False, overhead_mib=0):
    """Return the ceiling and its breakdown as a dict. Times are in seconds."""
    s = shape

    # 1. Parameters.
    # q and o are hidden x (heads x head_dim); k and v are hidden x (kv_heads x head_dim).
    attn = s["hidden"] * s["head_dim"] * (2 * s["heads"] + 2 * s["kv_heads"])
    mlp = 3 * s["hidden"] * s["intermediate"]
    layer_params = s["layers"] * (attn + mlp)
    head_params = s["vocab"] * s["hidden"]          # lm_head (= the embedding if tied)
    embed_params = 0 if s["tied"] else head_params  # a separate embedding matrix

    # 2. Bytes.
    layer_bytes = layer_params * quant_bits / 8
    head_bytes = head_params * head_bits / 8
    embed_bytes = embed_params * head_bits / 8
    kv_per_token = 2 * s["layers"] * s["kv_heads"] * s["head_dim"] * kv_bits / 8
    attn_width = s["layers"] * s["heads"] * s["head_dim"]  # used in attention FLOPs

    bw = bandwidth_gbs * 1e9                       # bytes / s
    flops_per_s = tflops * 1e12 if tflops else None

    # 3. Decode: one token at the average context of the answer.
    context = prompt_tokens + output_tokens / 2
    decode_bytes = layer_bytes + head_bytes + context * kv_per_token
    decode_flops = 2 * (layer_params + head_params) + 4 * attn_width * context
    decode_mem_s = decode_bytes / bw
    decode_compute_s = decode_flops / flops_per_s if flops_per_s else None
    tpot_s = max(decode_mem_s, decode_compute_s or 0.0)

    # 4. Prefill: one pass over the prompt.
    logit_rows = prompt_tokens if all_logits else 1
    prefill_flops = (2 * layer_params * prompt_tokens
                     + 2 * head_params * logit_rows
                     + 2 * attn_width * prompt_tokens**2)
    prefill_bytes = layer_bytes + head_bytes + prompt_tokens * kv_per_token
    prefill_mem_s = prefill_bytes / bw
    prefill_compute_s = prefill_flops / flops_per_s if flops_per_s else None
    ttft_s = max(prefill_mem_s, prefill_compute_s) if flops_per_s else None

    # 5. Peak memory.
    weight_bytes = layer_bytes + head_bytes + embed_bytes
    kv_full_bytes = (prompt_tokens + output_tokens) * kv_per_token
    logits_bytes = logit_rows * s["vocab"] * 2  # fp16 logits
    peak_bytes = weight_bytes + kv_full_bytes + logits_bytes + overhead_mib * MIB

    return {
        "layer_params": layer_params, "head_params": head_params, "embed_params": embed_params,
        "layer_bytes": layer_bytes, "head_bytes": head_bytes, "embed_bytes": embed_bytes,
        "kv_per_token": kv_per_token, "context": context,
        "decode_bytes": decode_bytes, "decode_flops": decode_flops,
        "decode_mem_s": decode_mem_s, "decode_compute_s": decode_compute_s, "tpot_s": tpot_s,
        "prefill_bytes": prefill_bytes, "prefill_flops": prefill_flops,
        "prefill_mem_s": prefill_mem_s, "prefill_compute_s": prefill_compute_s, "ttft_s": ttft_s,
        "weight_bytes": weight_bytes, "kv_full_bytes": kv_full_bytes,
        "logits_bytes": logits_bytes, "peak_bytes": peak_bytes,
    }


def bound(mem_s, compute_s):
    """Name the roofline limit for a phase."""
    if compute_s is None:
        return "memory floor only, no --tflops"
    return "memory-bound" if mem_s >= compute_s else "compute-bound"


def main():
    parser = argparse.ArgumentParser(description="Roofline ceiling for batch-1 LLM inference.")
    parser.add_argument("config", help="path to the model's Hugging Face config.json")
    parser.add_argument("--device", choices=DEVICES, default="mx450")
    parser.add_argument("--bandwidth", type=float, help="GB/s, overrides the device preset")
    parser.add_argument("--tflops", type=float, help="usable TFLOPS, overrides the device preset")
    parser.add_argument("--memory-mib", type=float, help="overrides the device preset")
    parser.add_argument("--quant", type=bits, default="fp16",
                        help=f"layer weights: {', '.join(QUANT_BITS)} or bits per weight")
    parser.add_argument("--head-quant", type=bits, default="fp16",
                        help="embedding and lm_head: same choices as --quant")
    parser.add_argument("--kv-bits", type=float, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--all-logits", action="store_true",
                        help="prefill computes logits for every prompt token (Stride does)")
    parser.add_argument("--overhead-mib", type=float, default=0)
    args = parser.parse_args()

    # Command-line values override the device preset.
    dev = dict(DEVICES[args.device])
    for key, value in (("bandwidth_gbs", args.bandwidth), ("tflops", args.tflops),
                       ("memory_mib", args.memory_mib)):
        if value is not None:
            dev[key] = value

    s = load_shape(args.config)
    r = predict(s, dev["bandwidth_gbs"], dev["tflops"], args.quant, args.head_quant,
                args.kv_bits, args.prompt_tokens, args.output_tokens,
                args.all_logits, args.overhead_mib)

    total_params = r["layer_params"] + r["head_params"] + r["embed_params"]
    tflops_text = f"{dev['tflops']} TFLOPS" if dev["tflops"] else "TFLOPS not set"
    print(f"model    {s['layers']} layers, hidden {s['hidden']}, {s['heads']} heads / "
          f"{s['kv_heads']} KV heads, head_dim {s['head_dim']}, vocab {s['vocab']}, "
          f"{'tied' if s['tied'] else 'untied'} embeddings")
    print(f"params   {total_params / 1e9:.3f} B total = {r['layer_params'] / 1e9:.3f} B layers"
          f" + {(r['head_params'] + r['embed_params']) / 1e9:.3f} B embedding/lm_head"
          f"  (check against the model card)")
    print(f"weights  layers {r['layer_bytes'] / MIB:.0f} MiB @ {args.quant} bits, "
          f"lm_head {r['head_bytes'] / MIB:.0f} MiB @ {args.head_quant} bits, "
          f"KV {r['kv_per_token'] / 1024:.1f} KiB/token")
    print(f"device   {args.device}: {dev['bandwidth_gbs']} GB/s, {tflops_text}, "
          f"{dev['memory_mib']:.0f} MiB")
    print(f"workload {args.prompt_tokens} prompt + {args.output_tokens} output tokens"
          f"{', logits for every prompt token' if args.all_logits else ''}\n")

    print(f"decode   TPOT {r['tpot_s'] * 1e3:.2f} ms = {1 / r['tpot_s']:.1f} tokens/s "
          f"({bound(r['decode_mem_s'], r['decode_compute_s'])}; "
          f"{r['decode_bytes'] / 1e6:.0f} MB read per token at context {r['context']:.0f})")
    if r["ttft_s"] is None:
        print(f"prefill  TTFT null: needs --tflops ({r['prefill_flops'] / 1e9:.1f} GFLOPs; "
              f"memory floor alone {r['prefill_mem_s'] * 1e3:.2f} ms)")
    else:
        e2e_s = r["ttft_s"] + (args.output_tokens - 1) * r["tpot_s"]
        print(f"prefill  TTFT {r['ttft_s'] * 1e3:.2f} ms "
              f"({bound(r['prefill_mem_s'], r['prefill_compute_s'])}; "
              f"{r['prefill_flops'] / 1e9:.1f} GFLOPs)")
        print(f"e2e      {e2e_s * 1e3:.0f} ms for the whole answer")

    peak_mib = r["peak_bytes"] / MIB
    print(f"memory   peak {peak_mib:.0f} MiB = weights {r['weight_bytes'] / MIB:.0f} "
          f"+ KV {r['kv_full_bytes'] / MIB:.1f} + logits {r['logits_bytes'] / MIB:.1f} "
          f"+ overhead {args.overhead_mib:.0f}")
    if peak_mib > dev["memory_mib"]:
        print(f"WARNING  does not fit in {dev['memory_mib']:.0f} MiB of device memory")

    predicted = {
        "ttft_ms": round(r["ttft_s"] * 1e3, 2) if r["ttft_s"] is not None else None,
        "tpot_ms": round(r["tpot_s"] * 1e3, 2),
        "peak_mem_mb": round(peak_mib),
        "source": f"ceiling.py {VERSION}",
    }
    print("\n" + json.dumps({"predicted": predicted}, indent=2))


if __name__ == "__main__":
    main()
