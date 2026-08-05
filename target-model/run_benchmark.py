"""
run_benchmark.py — TargetEngine benchmark
==========================================

Sends a fixed suite of prompts through TargetEngine, records per-request
timing metrics (TTFT, decode latency, e2e, throughput), and prints a clean
summary table at the end.  The Profiler runs alongside to give a full
percentile breakdown.

Designed to be stable and repeatable so you can re-run after implementing
paged KV caching or quantisation changes and directly compare the numbers.

Usage
-----
    python target-model/run_benchmark.py [--model HF_ID_OR_PATH]
                                         [--requests N] [--warmup N]
                                         [--max-tokens N] [--temperature T]
                                         [--no-profiler]
"""

from __future__ import annotations

import sys
import os
import json
import time
import datetime
import argparse
import statistics
from dataclasses import dataclass
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Path setup — import engine and profiler from adjacent directories
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)                               # target-model/
sys.path.insert(0, os.path.join(_HERE, ".."))          # project root

from target_engine import (
    TargetEngine,
    MODEL_ID,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
)
from profiler import Profiler

# ---------------------------------------------------------------------------
# Benchmark prompt suite
# ---------------------------------------------------------------------------
# Prompts are chosen to cover a variety of prefill lengths and output styles.
# "Fake" short completions (e.g. "The capital of France is") let the model
# hit EOS quickly — good for measuring TTFT under minimal decode load.
# Longer prompts stress the prefill path; longer continuations stress decode.

PROMPTS: list[tuple[str, str]] = [
    (
        "short-factual",
        "The capital of France is",
    ),
    (
        "short-qa",
        "What is the boiling point of water in Celsius?",
    ),
    (
        "medium-explain",
        (
            "Explain what gradient descent is and why it is used in machine "
            "learning in two to three sentences."
        ),
    ),
    (
        "medium-code",
        (
            "Write a Python function that takes a list of integers and returns "
            "the two that sum to zero, or None if no such pair exists."
        ),
    ),
    (
        "long-reasoning",
        (
            "You are given a sorted array of integers and a target value. "
            "Describe step by step how binary search finds the target, "
            "including what happens when the target is absent."
        ),
    ),
    (
        "long-creative",
        (
            "Write the opening paragraph of a science-fiction short story set "
            "on a generation ship whose crew has completely forgotten their "
            "original destination."
        ),
    ),
    (
        "system-prompt-heavy",
        (
            "Summarise the following paragraph in one sentence: "
            "Speculative decoding is a technique used to accelerate "
            "autoregressive language model inference by using a small, fast "
            "draft model to propose multiple candidate tokens at once, which "
            "are then verified in a single forward pass by the larger target "
            "model, accepting or rejecting them according to a principled "
            "sampling criterion that preserves the target distribution."
        ),
    ),
    (
        "long-list",
        (
            "List ten practical tips for improving sleep quality, "
            "one per line, starting each with a number and a period."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Per-request result record
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    label: str
    prompt_tokens: int       # tokens in the formatted prompt (including chat template)
    output_tokens: int       # tokens actually generated
    ttft_ms: float           # time to first token (prefill wall-clock)
    decode_mean_ms: float    # mean per-token decode latency
    e2e_ms: float            # full wall-clock for generate()
    tok_per_sec: float       # output_tokens / e2e_s


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _count_prompt_tokens(engine: TargetEngine, prompt: str, system: str) -> int:
    """Return the token count of the formatted chat template string."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]
    text = engine._tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(engine._tokenizer.encode(text))


def _snapshot_profiler_timing(
    prof: Profiler, prefix: str, metric: str
) -> Optional[list[float]]:
    """Return the raw samples list for a timing metric, or None."""
    try:
        return prof._timings[prefix][metric].samples
    except KeyError:
        return None


def _run_single_request(
    engine: TargetEngine,
    label: str,
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    rep_penalty: float,
    prof: Optional[Profiler],
) -> RequestResult:
    """Time a single generate() call and build a RequestResult."""

    # Snapshot profiler counters before the call so we can compute deltas
    prefill_count_before = 0
    decode_count_before  = 0
    if prof is not None:
        s = _snapshot_profiler_timing(prof, "target", "prefill_ms")
        prefill_count_before = len(s) if s else 0
        s = _snapshot_profiler_timing(prof, "target", "decode_ms")
        decode_count_before  = len(s) if s else 0

    t0 = time.perf_counter()
    engine.generate(
        prompt=prompt,
        system=system,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rep_penalty,
        verbose=False,   # suppress per-request profiler.report()
    )
    e2e_ms = (time.perf_counter() - t0) * 1000.0

    # --- Derive metrics from profiler samples (delta since before the call) ---
    ttft_ms       = 0.0
    decode_mean_ms = 0.0
    output_tokens  = 0

    if prof is not None:
        prefill_samples = _snapshot_profiler_timing(prof, "target", "prefill_ms") or []
        new_prefill = prefill_samples[prefill_count_before:]
        ttft_ms = new_prefill[-1] if new_prefill else 0.0

        decode_samples = _snapshot_profiler_timing(prof, "target", "decode_ms") or []
        new_decode = decode_samples[decode_count_before:]
        decode_mean_ms = statistics.mean(new_decode) if new_decode else 0.0
        output_tokens  = len(new_decode) + (1 if new_prefill else 0)  # prefill samples 1st token

    prompt_tokens = _count_prompt_tokens(engine, prompt, system)
    tok_per_sec   = (output_tokens / (e2e_ms / 1000.0)) if e2e_ms > 0 else 0.0

    return RequestResult(
        label=label,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        decode_mean_ms=decode_mean_ms,
        e2e_ms=e2e_ms,
        tok_per_sec=tok_per_sec,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_separator(width: int = 100) -> None:
    print("─" * width)


def _print_results_table(results: list[RequestResult]) -> None:
    col = {
        "label":       26,
        "p_tok":        7,
        "o_tok":        7,
        "ttft":        10,
        "decode":      11,
        "e2e":         10,
        "tok_s":        9,
    }
    header = (
        f"  {'Prompt label':<{col['label']}} "
        f"{'P-tok':>{col['p_tok']}} "
        f"{'O-tok':>{col['o_tok']}} "
        f"{'TTFT ms':>{col['ttft']}} "
        f"{'Decode ms/t':>{col['decode']}} "
        f"{'E2E ms':>{col['e2e']}} "
        f"{'Tok/s':>{col['tok_s']}}"
    )
    _print_separator()
    print(header)
    _print_separator()
    for r in results:
        print(
            f"  {r.label:<{col['label']}} "
            f"{r.prompt_tokens:>{col['p_tok']}} "
            f"{r.output_tokens:>{col['o_tok']}} "
            f"{r.ttft_ms:>{col['ttft']}.1f} "
            f"{r.decode_mean_ms:>{col['decode']}.2f} "
            f"{r.e2e_ms:>{col['e2e']}.1f} "
            f"{r.tok_per_sec:>{col['tok_s']}.1f}"
        )
    _print_separator()


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = (p / 100.0) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def _print_aggregate(results: list[RequestResult]) -> None:
    def _stats(vals: list[float], unit: str = "") -> str:
        if not vals:
            return "n/a"
        return (
            f"mean={statistics.mean(vals):.1f}{unit}  "
            f"p50={_percentile(vals, 50):.1f}{unit}  "
            f"p95={_percentile(vals, 95):.1f}{unit}  "
            f"min={min(vals):.1f}{unit}  max={max(vals):.1f}{unit}"
        )

    ttfts      = [r.ttft_ms for r in results if r.ttft_ms > 0]
    decodes    = [r.decode_mean_ms for r in results if r.decode_mean_ms > 0]
    e2es       = [r.e2e_ms for r in results]
    tputs      = [r.tok_per_sec for r in results]
    out_tokens = [r.output_tokens for r in results]

    print("\n  ── Aggregate statistics ─────────────────────────────────────────────")
    print(f"  Requests          : {len(results)}")
    print(f"  Total output toks : {sum(out_tokens)}")
    print(f"  E2E latency       : {_stats(e2es, ' ms')}")
    if ttfts:
        print(f"  TTFT (prefill)    : {_stats(ttfts, ' ms')}")
    if decodes:
        print(f"  Decode latency    : {_stats(decodes, ' ms/tok')}")
    print(f"  Throughput        : {_stats(tputs, ' tok/s')}")
    _print_separator()


# ---------------------------------------------------------------------------
# JSONL metrics log helpers
# ---------------------------------------------------------------------------

def _agg(vals: list[float]) -> dict:
    """Return mean/p50/p95/min/max for a list of floats, or {} if empty."""
    if not vals:
        return {}
    return {
        "mean": round(statistics.mean(vals), 3),
        "p50":  round(_percentile(vals, 50), 3),
        "p95":  round(_percentile(vals, 95), 3),
        "min":  round(min(vals), 3),
        "max":  round(max(vals), 3),
    }


def _append_jsonl(
    path: str,
    description: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    rep_penalty: float,
    n_warmup: int,
    results: list[RequestResult],
    prof: Optional[Profiler],
) -> None:
    """
    Append one flat JSON record to the JSONL metrics log.

    Flat column layout so ``pd.read_json(path, lines=True)`` gives a
    directly queryable DataFrame with one row per run::

        df = pd.read_json("bench-metrics.jsonl", lines=True)
        baseline  = df[df.description == "baseline"].iloc[-1]
        quantized = df[df.description == "quantized-kv-4bit"].iloc[-1]
        pct = (quantized.decode_ms_mean / baseline.decode_ms_mean - 1) * 100
    """
    e2e_agg   = _agg([r.e2e_ms for r in results])
    tput_agg  = _agg([r.tok_per_sec for r in results])
    ttft_vals = [r.ttft_ms for r in results if r.ttft_ms > 0]
    ttft_agg  = _agg(ttft_vals)
    dec_vals  = [r.decode_mean_ms for r in results if r.decode_mean_ms > 0]
    dec_agg   = _agg(dec_vals)

    record: dict = {
        # --- metadata ---
        "description": description,
        "timestamp":   datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "engine":      "target",
        "model":       model_id,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "top_p":       top_p,
        "rep_penalty": rep_penalty,
        "n_warmup":    n_warmup,
        "n_requests":  len(results),
        # --- aggregate (benchmark-measured) ---
        "total_output_tokens": sum(r.output_tokens for r in results),
        **{f"e2e_ms_{k}": v for k, v in e2e_agg.items()},
        **{f"tok_per_sec_{k}": v for k, v in tput_agg.items()},
    }

    if ttft_agg:
        record.update({f"ttft_ms_{k}": v for k, v in ttft_agg.items()})
    if dec_agg:
        # decode_ms_mean is the canonical column for latency comparisons
        record.update({f"decode_ms_{k}": v for k, v in dec_agg.items()})

    # --- profiler aggregate (if enabled) ---
    if prof is not None:
        export   = prof.export()
        dp       = export["prefixes"].get("target", {})
        timings  = dp.get("timings", {})
        counters = dp.get("counters", {})
        gauges   = dp.get("gauges", {})
        for metric in ("decode_ms", "prefill_ms", "sample_ms", "e2e_latency_ms"):
            t = timings.get(metric)
            if t:
                record[f"profiler_{metric}_mean"] = t["mean_ms"]
                record[f"profiler_{metric}_p50"]  = t["p50_ms"]
                record[f"profiler_{metric}_p95"]  = t["p95_ms"]
                record[f"profiler_{metric}_p99"]  = t["p99_ms"]
                record[f"profiler_{metric}_min"]  = t["min_ms"]
                record[f"profiler_{metric}_max"]  = t["max_ms"]
        for ctr in ("tokens_generated", "requests_completed", "eos_hits",
                    "cache_clears", "cache_rollbacks"):
            if ctr in counters:
                record[ctr] = counters[ctr]
        record["tok_throughput_profiler"] = round(dp.get("throughput_tok_per_s", 0.0), 2)
        vram = gauges.get("vram_used_mb")
        if vram:
            record["vram_used_mb_mean"]   = vram["mean"]
            record["vram_used_mb_latest"] = vram["latest"]

    # --- per-request breakdown ---
    record["requests"] = [
        {
            "label":          r.label,
            "prompt_tokens":  r.prompt_tokens,
            "output_tokens":  r.output_tokens,
            "ttft_ms":        round(r.ttft_ms, 3),
            "decode_mean_ms": round(r.decode_mean_ms, 3),
            "e2e_ms":         round(r.e2e_ms, 3),
            "tok_per_sec":    round(r.tok_per_sec, 3),
        }
        for r in results
    ]

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[bench] Metrics appended to {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# VRAM snapshot helper
# ---------------------------------------------------------------------------

def _vram_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{alloc:.2f} GB allocated  |  {reserved:.2f} GB reserved  |  {total:.2f} GB total"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_benchmark(
    model_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    rep_penalty: float,
    n_warmup: int,
    n_requests: int,
    use_profiler: bool,
    system: str,
    description: str = "unnamed",
    jsonl_path: Optional[str] = None,
) -> None:
    prof: Optional[Profiler] = Profiler() if use_profiler else None

    engine = TargetEngine.from_pretrained(model_id=model_id, profiler=prof)

    print(f"\n{'═' * 100}")
    print(f"  Stride  ·  TargetEngine Benchmark")
    print(f"  model={model_id}  |  max_tokens={max_tokens}  |  temperature={temperature}")
    print(f"  warmup={n_warmup}  |  requests={n_requests}  |  profiler={'on' if use_profiler else 'off'}")
    print(f"  VRAM: {_vram_summary()}")
    print(f"{'═' * 100}\n")

    # ------------------------------------------------------------------
    # Warmup — run a short request to prime CUDA kernel caches; discard data
    # ------------------------------------------------------------------
    if n_warmup > 0:
        print(f"  [warmup] Running {n_warmup} warmup request(s) ...")
        warmup_prompt = PROMPTS[0][1]
        for i in range(n_warmup):
            engine.generate(
                prompt=warmup_prompt,
                system=system,
                max_new_tokens=16,
                temperature=0.0,
                top_p=1.0,
                verbose=False,
            )
            print(f"  [warmup] {i + 1}/{n_warmup} done")
        # Reset profiler so warmup data doesn't contaminate benchmark results
        if prof is not None:
            prof.reset()
        print()

    # ------------------------------------------------------------------
    # Main benchmark loop
    # ------------------------------------------------------------------
    results: list[RequestResult] = []

    # Cycle through prompts if more requests than prompts are requested
    prompt_pool = PROMPTS * ((n_requests // len(PROMPTS)) + 1)
    prompt_pool = prompt_pool[:n_requests]

    print(f"  [bench] Running {n_requests} request(s) ...\n")
    for i, (label, prompt) in enumerate(prompt_pool):
        print(f"  [{i + 1:>3}/{n_requests}] {label} ...", end=" ", flush=True)

        result = _run_single_request(
            engine=engine,
            label=label,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            rep_penalty=rep_penalty,
            prof=prof,
        )
        results.append(result)

        print(
            f"{result.output_tokens} tok  |  "
            f"TTFT {result.ttft_ms:.0f} ms  |  "
            f"E2E {result.e2e_ms:.0f} ms  |  "
            f"{result.tok_per_sec:.1f} tok/s"
        )

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print(f"\n{'═' * 100}")
    print("  Per-request results")
    _print_results_table(results)
    _print_aggregate(results)

    print(f"  VRAM (post-bench): {_vram_summary()}")
    _print_separator()

    # ------------------------------------------------------------------
    # Profiler detail report
    # ------------------------------------------------------------------
    if prof is not None:
        print("\n  ── Profiler detail ─────────────────────────────────────────────────")
        prof.report()

    # ------------------------------------------------------------------
    # JSONL metrics log
    # ------------------------------------------------------------------
    if jsonl_path is not None:
        _append_jsonl(
            path=jsonl_path,
            description=description,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            rep_penalty=rep_penalty,
            n_warmup=n_warmup,
            results=results,
            prof=prof,
        )

    print(f"\n{'═' * 100}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TargetEngine benchmark — measures GPU inference latency and throughput"
    )
    parser.add_argument(
        "--model", default=MODEL_ID,
        help="HuggingFace model ID or local path."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128,
        help=f"Max output tokens per request (default: 128)."
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE}; 0=greedy)."
    )
    parser.add_argument(
        "--top-p", type=float, default=TOP_P,
        help=f"Nucleus sampling threshold (default: {TOP_P})."
    )
    parser.add_argument(
        "--rep-penalty", type=float, default=REPETITION_PENALTY,
        help=f"Repetition penalty (default: {REPETITION_PENALTY}; 1.0=off)."
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Warmup requests to discard (default: 2)."
    )
    parser.add_argument(
        "--requests", type=int, default=len(PROMPTS),
        help=f"Benchmark requests to run (default: {len(PROMPTS)})."
    )
    parser.add_argument(
        "--system", default="You are a helpful assistant.",
        help="System prompt for all requests."
    )
    parser.add_argument(
        "--no-profiler", action="store_true",
        help="Disable the Profiler (only show benchmark table)."
    )
    parser.add_argument(
        "--description", default="unnamed",
        help="Label for this run, written to bench-metrics.jsonl (default: \"unnamed\")."
    )
    args = parser.parse_args()

    run_benchmark(
        model_id=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        rep_penalty=args.rep_penalty,
        n_warmup=args.warmup,
        n_requests=args.requests,
        use_profiler=not args.no_profiler,
        system=args.system,
        description=args.description,
        jsonl_path=os.path.join(_HERE, "bench-metrics.jsonl"),
    )


if __name__ == "__main__":
    main()
