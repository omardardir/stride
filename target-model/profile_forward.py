"""
Nsight Systems target — one decode step of the target model
===========================================================

Point nsys at this file, not at target_engine.py: the REPL waits for input,
and model loading would fill the timeline with ~15 s of disk reads.

What it does
------------
1. Loads the model exactly like TargetEngine.from_pretrained() (NF4, SDPA).
2. Prints which SDPA attention backends can run Qwen2.5's GQA attention on
   this GPU (the timeline will show the kernels of whichever one wins).
3. Warm-up, *outside* the capture: the prefill + a few decode steps.  This
   pays the one-time costs (cuBLAS handle creation, the bitsandbytes DLL
   load, the caching allocator growing its pool) and fills the KV cache.
4. cudaProfilerStart -> captured work -> cudaProfilerStop.  --capture picks it:
     decode   (default) --decode-steps decode steps that continue the same
              generation, one NVTX range each: decode_00, decode_01, ...
     prefill  one fresh prefill of the same prompt on an empty cache, in an
              NVTX range named prefill.  This is the TTFT path.
   Both mirror TargetEngine._run_generation (forward + sampling).

Two runs per question
---------------------
NVTX hooks and nsys's CUDA tracing both add CPU time per kernel launch, so a
traced step is slower than BENCH_LOG.md.  GPU kernel times barely change.
    --nvtx step    honest wall time: use it for the bandwidth / overhead split
    --nvtx linear  per-projection ranges: use it to see *where* time goes

NVTX levels (--nvtx)
--------------------
    step    decode_XX, split into forward + sample
    block   + layer_XX, and inside each layer: input_norm, attn, post_norm, mlp
    linear  + every projection: q/k/v/o_proj, gate/up/down_proj, lm_head
            (default; each push/pop costs a few us of CPU, ~1-2 ms per step)

Run it under nsys from the repo root (PowerShell).  --capture-range-end=stop
matters: the default (stop-shutdown) kills the script at cudaProfilerStop, so
the timing summary never prints.

    nsys profile -t cuda,nvtx,cublas -c cudaProfilerApi --capture-range-end=stop
        -s none --cpuctxsw=none --stats=true -o profiles\\forward_%n
        .venv\\Scripts\\python.exe target-model\\profile_forward.py

--warmup-steps sets how many decode steps run before the captured work.
--repeat-kv swaps the MATH attention fallback for the fused EFFICIENT kernel.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

# The model is already in the Hugging Face cache. Without this, from_pretrained
# still asks huggingface.co for optional files, and Python HTTPS fails on this
# laptop (Avast intercepts TLS). Must be set before transformers is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
import transformers.integrations.sdpa_attention as sdpa_attn
from torch.nn.attention import SDPBackend, sdpa_kernel

from target_engine import (
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_P,
    NaiveKVCache,
    TargetEngine,
    _sample_next_token,
)

DEFAULT_PROMPT = (
    "Explain, step by step, how a GPU executes a matrix multiplication, "
    "and why memory bandwidth matters for running large language models."
)

# On Windows, nsys runs the script in its own console window, which closes the
# moment the script exits, so everything printed is also written here.
LOG_PATH = Path(__file__).resolve().parent.parent / "profiles" / "profile_forward.log"


# ---------------------------------------------------------------------------
# Log file (copy of stdout + stderr)
# ---------------------------------------------------------------------------
class _Tee:
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._log_file.write(data)
        self._log_file.flush()

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def start_log() -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    log_file = open(LOG_PATH, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)


# ---------------------------------------------------------------------------
# NVTX helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def nvtx(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def add_module_nvtx(model, level: str) -> None:
    """Wrap decoder layers (and optionally every projection) in NVTX ranges.

    Uses forward pre/post hooks, so the model code is untouched.  Ranges nest,
    so on the timeline you get layer_03 > attn > q_proj > (its kernels).
    """
    if level == "step":
        return

    targets = {"model.embed_tokens": "embed", "model.norm": "final_norm"}
    for i in range(len(model.model.layers)):
        p = f"model.layers.{i}"
        targets[p] = f"layer_{i:02d}"
        targets[f"{p}.input_layernorm"] = "input_norm"
        targets[f"{p}.self_attn"] = "attn"
        targets[f"{p}.post_attention_layernorm"] = "post_norm"
        targets[f"{p}.mlp"] = "mlp"
        if level == "linear":
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                targets[f"{p}.self_attn.{proj}"] = proj
            for proj in ("gate_proj", "up_proj", "down_proj"):
                targets[f"{p}.mlp.{proj}"] = proj
    if level == "linear":
        targets["lm_head"] = "lm_head"

    def push(label):
        def hook(module, args):
            torch.cuda.nvtx.range_push(label)
        return hook

    def pop(module, args, output):
        torch.cuda.nvtx.range_pop()

    modules = dict(model.named_modules())
    for path, label in targets.items():
        modules[path].register_forward_pre_hook(push(label))
        modules[path].register_forward_hook(pop)


# ---------------------------------------------------------------------------
# Which attention kernel will SDPA pick?
# ---------------------------------------------------------------------------
def report_sdpa_backends(model, device: str, kv_len: int = 128) -> None:
    """Try each SDPA backend on decode-shaped Q/K/V and print why it fails.

    transformers calls SDPA with enable_gqa=True for Qwen2.5 (12 query heads,
    2 KV heads) because we pass no attention mask.  The fused kernels have
    restrictions (GPU architecture, GQA support), and when they all refuse,
    PyTorch silently falls back to the unfused MATH backend.
    """
    cfg = model.config
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
    q = torch.randn(1, n_q, 1, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(1, n_kv, kv_len, head_dim, dtype=torch.float16, device=device)
    v = torch.randn_like(k)
    k_rep = k.repeat_interleave(n_q // n_kv, dim=1)
    v_rep = v.repeat_interleave(n_q // n_kv, dim=1)

    cases = [
        (f"GQA ({n_q} q heads, {n_kv} kv heads) - what the model runs", k, v, True),
        (f"K/V repeated to {n_q} heads, no GQA - for comparison", k_rep, v_rep, False),
    ]
    backends = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ]

    print("[SDPA] Backend check (MATH is the slow, unfused fallback):")
    for title, kk, vv, gqa in cases:
        print(f"  {title}")
        for backend in backends:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    with sdpa_kernel(backend):
                        F.scaled_dot_product_attention(q, kk, vv, enable_gqa=gqa)
                    print(f"    {backend.name:<20} usable")
                except RuntimeError:
                    # Keep the real reason; drop the "X not used because:" headers
                    # and the "runtime disabled" lines that sdpa_kernel() itself
                    # causes for every backend we didn't ask for.
                    reasons = [str(w.message).split("(Triggered internally")[0].strip()
                               for w in caught]
                    reasons = [r for r in reasons
                               if r and "runtime disabled" not in r
                               and not r.endswith("not used because:")]
                    reasons = reasons or ["(no reason given)"]
                    print(f"    {backend.name:<20} rejected")
                    for r in reasons:
                        print(f"      - {r.splitlines()[0][:150]}")
    torch.cuda.synchronize()
    print()


# ---------------------------------------------------------------------------
# Prefill and decode steps (mirror TargetEngine._run_generation)
# ---------------------------------------------------------------------------
# Wall times are honest without extra syncs: sampling ends in .item(), which
# waits for the GPU, so each step is finished when its timer stops.

def prefill(
    model,
    input_ids: torch.Tensor,
    generated_ids: list[int],
    args: argparse.Namespace,
    device: str,
) -> tuple[NaiveKVCache, int, float]:
    """Returns (kv cache, next token id, prefill ms)."""
    kv_cache = NaiveKVCache()
    prompt_len = input_ids.shape[1]

    t0 = time.perf_counter()
    with nvtx("prefill"):
        position_ids = torch.arange(prompt_len, dtype=torch.long, device=device).unsqueeze(0)
        with nvtx("forward"):
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )
        with nvtx("sample"):
            next_token_id = _sample_next_token(
                outputs.logits[:, -1, :], args.temperature, args.top_p,
                generated_ids, args.rep_penalty,
            )
    prefill_ms = (time.perf_counter() - t0) * 1e3
    kv_cache.update(outputs.past_key_values)
    generated_ids.append(next_token_id)
    return kv_cache, next_token_id, prefill_ms


def decode_step(
    model,
    kv_cache: NaiveKVCache,
    next_token_id: int,
    generated_ids: list[int],
    args: argparse.Namespace,
    device: str,
    label: str,
) -> tuple[int, float]:
    """One decode step on an existing cache. Returns (next token id, step ms).

    EOS is ignored so the step count is always what was asked for.
    """
    t0 = time.perf_counter()
    with nvtx(label):
        next_input = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        decode_position_ids = torch.tensor(
            [[kv_cache.seq_len()]], dtype=torch.long, device=device
        )
        with nvtx("forward"):
            outputs = model(
                input_ids=next_input,
                position_ids=decode_position_ids,
                past_key_values=kv_cache.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        with nvtx("sample"):
            next_token_id = _sample_next_token(
                outputs.logits[:, -1, :], args.temperature, args.top_p,
                generated_ids, args.rep_penalty,
            )
    step_ms = (time.perf_counter() - t0) * 1e3
    kv_cache.update(outputs.past_key_values)
    generated_ids.append(next_token_id)
    return next_token_id, step_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Profile target model forward passes under nsys.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--capture", choices=("decode", "prefill"), default="decode",
                        help="What runs inside the capture (default: decode).")
    parser.add_argument("--decode-steps", type=int, default=1,
                        help="Decode steps inside the capture (default: 1).")
    parser.add_argument("--warmup-steps", type=int, default=8,
                        help="Un-captured decode steps after the prefill (default: 8).")
    parser.add_argument("--nvtx", choices=("step", "block", "linear"), default="linear",
                        help="How fine the NVTX ranges are (default: linear).")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--rep-penalty", type=float, default=REPETITION_PENALTY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-kv", action="store_true",
                        help="Repeat K/V to all query heads before SDPA so the fused "
                             "EFFICIENT kernel can replace the MATH fallback.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.repeat_kv:
        # transformers checks this before every SDPA call. When it returns False,
        # sdpa_attention_forward runs repeat_kv (2 -> 12 KV heads, a temporary
        # copy per layer) and calls SDPA without enable_gqa, which EFFICIENT accepts.
        sdpa_attn.use_gqa_in_sdpa = lambda *args, **kwargs: False

    engine = TargetEngine.from_pretrained(profiler=None)
    model, tokenizer, device = engine._model, engine._tokenizer, engine._device

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="pt").to(device)["input_ids"]
    capture_desc = (f"{args.decode_steps} decode step(s), no prefill" if args.capture == "decode"
                    else "1 prefill, no decode")
    print(f"[INFO] Prompt: {input_ids.shape[1]} tokens  |  "
          f"capture: {capture_desc}  |  NVTX: {args.nvtx}  |  "
          f"attention: {'repeat_kv + SDPA (EFFICIENT)' if args.repeat_kv else 'GQA SDPA (MATH)'}\n")

    report_sdpa_backends(model, device)
    add_module_nvtx(model, args.nvtx)

    generated_ids: list[int] = []
    with torch.inference_mode():
        # ---- Warm-up (not captured): prefill + a few decode steps ---------
        # Pays the one-time costs and fills the KV cache, so the captured
        # step is a normal mid-generation decode step.
        with nvtx("warmup"):
            kv_cache, next_token_id, prefill_ms = prefill(
                model, input_ids, generated_ids, args, device
            )
            warmup_ms = []
            for i in range(args.warmup_steps):
                next_token_id, ms = decode_step(
                    model, kv_cache, next_token_id, generated_ids, args, device,
                    label=f"warmup_decode_{i:02d}",
                )
                warmup_ms.append(ms)
        torch.cuda.synchronize()
        context_len = kv_cache.seq_len()

        # ---- Captured region: decode steps or one warm prefill -----------
        decode_ms = []
        torch.cuda.profiler.start()          # cudaProfilerStart: nsys starts recording
        with nvtx("profiled"):
            if args.capture == "prefill":
                # Fresh cache and token list: a second request with the same prompt.
                _, _, captured_prefill_ms = prefill(model, input_ids, [], args, device)
            else:
                for i in range(args.decode_steps):
                    next_token_id, ms = decode_step(
                        model, kv_cache, next_token_id, generated_ids, args, device,
                        label=f"decode_{i:02d}",
                    )
                    decode_ms.append(ms)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()           # cudaProfilerStop: nsys stops recording

    print(f"[INFO] First prefill (not captured, cold): {prefill_ms:.1f} ms  "
          f"({input_ids.shape[1]} prompt tokens)")
    if warmup_ms:
        print("[INFO] Warm-up decode steps (not captured): "
              + "  ".join(f"{t:.0f}" for t in warmup_ms) + " ms")
    if args.capture == "prefill":
        print(f"[INFO] Captured prefill (warm): {captured_prefill_ms:.1f} ms  "
              f"({input_ids.shape[1]} prompt tokens)")
    else:
        print(f"[INFO] Captured decode step(s) at context {context_len}: "
              + "  ".join(f"{t:.1f}" for t in decode_ms) + " ms")
    print("[INFO] (Times include NVTX hook overhead, plus nsys overhead when run under "
          "nsys; compare them to each other, not to BENCH_LOG.md.)")
    print(f"[INFO] Text: {tokenizer.decode(generated_ids, skip_special_tokens=True)!r}")


if __name__ == "__main__":
    start_log()
    try:
        main()
    except Exception:
        traceback.print_exc()
        print(f"\n[ERROR] Traceback saved to {LOG_PATH}")
        sys.exit(1)
