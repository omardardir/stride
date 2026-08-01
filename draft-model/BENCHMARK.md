# DraftEngine Benchmark

A repeatable CPU inference benchmark for the `DraftEngine` (Qwen2-0.5B-Instruct, GGUF / llama.cpp).

## What it measures

| Metric | Description |
|---|---|
| **P-tok** | Number of tokens in the prompt |
| **O-tok** | Number of tokens generated |
| **Prefill ms** | Wall-clock time for the single prefill forward pass |
| **Gen ms** | Total wall-clock time for the full `generate()` call |
| **Tok/s** | Output tokens per second (`O-tok / gen_time_s`) |

After the per-request table, an **aggregate block** reports mean / p50 / p95 / min / max across all requests.  
A full **profiler detail section** (percentiles for `prefill_ms`, `draft_ms`, `eval_ms`, `sample_ms`, etc.) follows at the end.

## How it works

1. **Load** — `DraftEngine.from_path()` is called once with a `Profiler` instance wired in.
2. **Warmup** — a short request is run (default: 2 rounds) and then `profiler.reset()` is called so warmup noise doesn't contaminate the results.
3. **Benchmark loop** — each prompt in the suite is passed to `engine.generate()`. The script wall-clocks the call and reads per-request deltas from the profiler's internal sample lists.
4. **Report** — results are printed as a fixed-width table, followed by aggregate statistics and the profiler's full percentile breakdown.

## Prompt suite

Six prompts are included, chosen to stress different code paths:

| Label | Focus |
|---|---|
| `short-factual` | Minimal prefill, near-instant EOS |
| `short-creative` | Short prefill, a few decode steps |
| `medium-technical` | Mid-length prompt, moderate output |
| `medium-code` | Code generation path |
| `long-reasoning` | Long prefill, multi-step decode |
| `long-creative` | Long output, tests decode throughput |

If `--requests N` exceeds the suite size, prompts are cycled.

## Usage

```bash
# defaults (6 requests, 2 warmup, max 64 output tokens)
python draft-model/run_benchmark.py --model ./draft-model/weights/qwen2-0_5b-instruct-q4_k_m.gguf

# custom run
python draft-model/run_benchmark.py \
    --model  ./draft-model/weights/qwen2-0_5b-instruct-q4_k_m.gguf \
    --gamma  4 \
    --n-threads 4 \
    --max-tokens 128 \
    --warmup 3 \
    --requests 18

# skip profiler overhead
python draft-model/run_benchmark.py --model ... --no-profiler
```

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `DRAFT_MODEL_PATH` | Path to the GGUF file |
| `--gamma` | `4` | Draft tokens proposed per speculative step |
| `--n-ctx` | `2048` | KV cache context length |
| `--n-threads` | `4` | CPU threads for ggml |
| `--max-tokens` | `64` | Max output tokens per request |
| `--warmup` | `2` | Requests to discard before recording |
| `--requests` | `6` | Total benchmark requests |
| `--no-profiler` | off | Disable profiler instrumentation |

## Re-running after engine changes

Because the benchmark calls only the public interface (`DraftEngine.from_path()` and `engine.generate()`), you can swap in a new KV cache or quantisation implementation inside `draft_engine.py` and re-run with the exact same command to get comparable numbers.
