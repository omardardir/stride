"""
run_benchmark.py — DraftEngine benchmark
=========================================

Sends a fixed suite of prompts through DraftEngine, collects per-request
timing data independently of the profiler, and prints a clean summary table
at the end.  The Profiler is also active throughout so you get its full
percentile breakdown alongside the benchmark results.

Designed to be stable and repeatable so you can re-run it after implementing
paged KV caching or quantisation changes and directly compare the numbers.

Usage
-----
    python draft-model/run_benchmark.py [--model PATH] [--gamma N]
                                        [--requests N] [--warmup N]
                                        [--max-tokens N] [--no-profiler]
"""

from __future__ import annotations

import sys
import os
import time
import argparse
import statistics
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup — import engine and profiler from adjacent directories
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)                                   # draft-model/
sys.path.insert(0, os.path.join(_HERE, ".."))              # project root

from draft_engine import DraftEngine, DRAFT_MODEL_PATH, N_CTX, N_THREADS, GAMMA
from profiler import Profiler

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

# Prompts that exercise different prefill lengths and output types.
# Using real-ish text keeps token distributions close to production;
# short sequences stress prefill, long ones stress the decode loop.
PROMPTS: list[tuple[str, str]] = [
    (
        "short-factual",
        "The capital of France is",
    ),
    (
        "short-creative",
        "Once upon a time in a land far away, there lived a",
    ),
    (
        "medium-technical",
        (
            "Explain the key difference between a transformer encoder and "
            "a transformer decoder in two sentences."
        ),
    ),
    (
        "medium-code",
        (
            "Write a Python function that computes the nth Fibonacci number "
            "using dynamic programming."
        ),
    ),
    (
        "long-reasoning",
        (
            "You are given a sorted array of integers and a target value. "
            "Describe, step by step, how binary search finds the target, "
            "including what happens when the target is not present."
        ),
    ),
    (
        "long-creative",
        (
            "Write the opening paragraph of a science-fiction short story set "
            "on a generation ship where the crew has forgotten their destination."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Per-request result record
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    label: str
    prompt_tokens: int          # number of tokens in the prompt
    output_tokens: int          # number of tokens actually generated
    prefill_ms: float           # wall-clock prefill duration
    gen_ms: float               # wall-clock for the full generate() call (prefill + decode)
    tok_per_sec: float          # output_tokens / gen_time_s


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _measure_prefill(engine: DraftEngine, text: str) -> tuple[int, float]:
    """
    Run load_prompt() and return (n_prompt_tokens, prefill_wall_ms).

    We measure the wall clock here rather than relying on the profiler so the
    number is self-contained and comparable across profiler-enabled / disabled
    runs.
    """
    token_ids = engine._llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
    engine.kv.clear()
    t0 = time.perf_counter()
    engine._llm.eval(token_ids)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    engine.kv._advance(len(token_ids))
    engine._last_token_id = token_ids[-1]
    return len(token_ids), elapsed_ms


def _run_single_request(
    engine: DraftEngine,
    label: str,
    prompt: str,
    max_tokens: int,
    gamma: int,
) -> RequestResult:
    """Run one full generate() call and return a RequestResult."""
    t0 = time.perf_counter()
    _ = engine.generate(prompt=prompt, max_new_tokens=max_tokens, gamma=gamma, verbose=False)
    gen_ms = (time.perf_counter() - t0) * 1000.0

    output_tokens = engine._llm.n_tokens - engine.kv.seq_len() + 1  # approx
    # More reliable: count how many tokens are in the KV cache beyond the prompt.
    # Since generate() calls load_prompt() internally we can re-tokenise.
    prompt_token_count = len(
        engine._llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
    )
    output_tokens = max(engine.kv.seq_len() - prompt_token_count, 0)

    prefill_ms = 0.0
    if engine._profiler is not None:
        try:
            samples = engine._profiler._timings["draft"]["prefill_ms"].samples
            prefill_ms = samples[-1] if samples else 0.0
        except (KeyError, IndexError):
            pass

    tok_per_sec = (output_tokens / (gen_ms / 1000.0)) if gen_ms > 0 else 0.0

    return RequestResult(
        label=label,
        prompt_tokens=prompt_token_count,
        output_tokens=output_tokens,
        prefill_ms=prefill_ms,
        gen_ms=gen_ms,
        tok_per_sec=tok_per_sec,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_separator(width: int = 90) -> None:
    print("─" * width)


def _print_results_table(results: list[RequestResult]) -> None:
    col = {
        "label":        24,
        "p_tok":         7,
        "o_tok":         7,
        "prefill":      11,
        "gen_ms":       10,
        "tok_s":         9,
    }
    header = (
        f"  {'Prompt label':<{col['label']}} "
        f"{'P-tok':>{col['p_tok']}} "
        f"{'O-tok':>{col['o_tok']}} "
        f"{'Prefill ms':>{col['prefill']}} "
        f"{'Gen ms':>{col['gen_ms']}} "
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
            f"{r.prefill_ms:>{col['prefill']}.1f} "
            f"{r.gen_ms:>{col['gen_ms']}.1f} "
            f"{r.tok_per_sec:>{col['tok_s']}.1f}"
        )
    _print_separator()


def _print_aggregate(results: list[RequestResult]) -> None:
    gen_times   = [r.gen_ms for r in results]
    prefills    = [r.prefill_ms for r in results if r.prefill_ms > 0]
    throughputs = [r.tok_per_sec for r in results]
    out_tokens  = [r.output_tokens for r in results]

    def _stats(vals: list[float]) -> str:
        if not vals:
            return "n/a"
        return (
            f"mean={statistics.mean(vals):.1f}  "
            f"p50={statistics.median(vals):.1f}  "
            f"p95={sorted(vals)[int(len(vals) * 0.95)]:.1f}  "
            f"min={min(vals):.1f}  max={max(vals):.1f}"
        )

    print("\n  ── Aggregate statistics ──────────────────────────────────────")
    print(f"  Requests       : {len(results)}")
    print(f"  Total out-toks : {sum(out_tokens)}")
    print(f"  Gen latency ms : {_stats(gen_times)}")
    if prefills:
        print(f"  Prefill ms     : {_stats(prefills)}")
    print(f"  Throughput t/s : {_stats(throughputs)}")
    _print_separator()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_benchmark(
    model_path: str,
    n_ctx: int,
    n_threads: int,
    gamma: int,
    max_tokens: int,
    n_warmup: int,
    n_requests: int,
    use_profiler: bool,
) -> None:
    prof: Optional[Profiler] = Profiler() if use_profiler else None

    engine = DraftEngine.from_path(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False,
        profiler=prof,
    )

    print(f"\n{'═' * 90}")
    print(f"  Stride  ·  DraftEngine Benchmark")
    print(f"  model={model_path}  |  gamma={gamma}  |  max_tokens={max_tokens}")
    print(f"  n_ctx={n_ctx}  |  n_threads={n_threads}  |  warmup={n_warmup}  |  requests={n_requests}")
    print(f"{'═' * 90}\n")

    # ------------------------------------------------------------------
    # Warmup — discarded, just to prime the llama.cpp runtime caches
    # ------------------------------------------------------------------
    if n_warmup > 0:
        print(f"  [warmup] Running {n_warmup} warmup request(s) ...")
        warmup_prompt = PROMPTS[0][1]
        for i in range(n_warmup):
            engine.generate(prompt=warmup_prompt, max_new_tokens=16, gamma=gamma, verbose=False)
            print(f"  [warmup] {i + 1}/{n_warmup} done")
        # Reset profiler so warmup data doesn't contaminate results
        if prof is not None:
            prof.reset()
        print()

    # ------------------------------------------------------------------
    # Main benchmark loop
    # ------------------------------------------------------------------
    results: list[RequestResult] = []
    prompt_pool = PROMPTS * ((n_requests // len(PROMPTS)) + 1)  # cycle if needed
    prompt_pool = prompt_pool[:n_requests]

    print(f"  [bench] Running {n_requests} request(s) ...\n")
    for i, (label, prompt) in enumerate(prompt_pool):
        print(f"  [{i + 1:>3}/{n_requests}] {label} ...", end=" ", flush=True)
        result = _run_single_request(engine, label, prompt, max_tokens, gamma)
        results.append(result)
        print(
            f"{result.output_tokens} tok  |  {result.gen_ms:.0f} ms  |  {result.tok_per_sec:.1f} tok/s"
        )

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print(f"\n{'═' * 90}")
    print("  Per-request results")
    _print_results_table(results)
    _print_aggregate(results)

    # ------------------------------------------------------------------
    # Profiler detail report (optional)
    # ------------------------------------------------------------------
    if prof is not None:
        print("\n  ── Profiler detail ───────────────────────────────────────────")
        prof.report()

    print(f"{'═' * 90}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DraftEngine benchmark — measures CPU draft throughput"
    )
    parser.add_argument(
        "--model", default=DRAFT_MODEL_PATH,
        help="Path to the GGUF file."
    )
    parser.add_argument(
        "--n-ctx", type=int, default=N_CTX,
        help=f"KV cache context length (default: {N_CTX})."
    )
    parser.add_argument(
        "--n-threads", type=int, default=N_THREADS,
        help=f"CPU threads (default: {N_THREADS})."
    )
    parser.add_argument(
        "--gamma", type=int, default=GAMMA,
        help=f"Draft tokens per speculative step (default: {GAMMA})."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64,
        help="Max output tokens per request (default: 64)."
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
        "--no-profiler", action="store_true",
        help="Disable the Profiler (only show benchmark table)."
    )
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        gamma=args.gamma,
        max_tokens=args.max_tokens,
        n_warmup=args.warmup,
        n_requests=args.requests,
        use_profiler=not args.no_profiler,
    )


if __name__ == "__main__":
    main()
