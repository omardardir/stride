# TargetEngine Benchmark

A repeatable GPU inference benchmark for the `TargetEngine` (Qwen2.5-1.5B-Instruct, 4-bit NF4 / HuggingFace + BitsAndBytes).

## What it measures

| Metric | Description |
|---|---|
| **P-tok** | Token count of the fully-formatted chat-template prompt |
| **O-tok** | Number of tokens generated |
| **TTFT ms** | Time To First Token — wall-clock duration of the prefill pass |
| **Decode ms/tok** | Mean per-token decode step latency |
| **E2E ms** | Total wall-clock time for the full `generate()` call |
| **Tok/s** | Output tokens per second (`O-tok / e2e_time_s`) |

After the per-request table, an **aggregate block** reports mean / p50 / p95 / min / max across all requests.  
A full **profiler detail section** (percentiles for `prefill_ms`, `decode_ms`, `sample_ms`, `e2e_latency_ms`, etc.) follows at the end, along with VRAM usage before and after the run.

## How it works

1. **Load** — `TargetEngine.from_pretrained()` is called once with a `Profiler` instance wired in.
2. **Warmup** — a short greedy request is run (default: 2 rounds) and then `profiler.reset()` is called so warmup noise (CUDA JIT, weight paging) doesn't contaminate the results.
3. **Benchmark loop** — each prompt is passed to `engine.generate()`. Before each call the script snapshots the profiler's internal sample-list lengths; after the call it reads the new samples as deltas, giving per-request breakdowns even though a single profiler accumulates across the whole run.
4. **Report** — results are printed as a fixed-width table, followed by aggregate statistics, VRAM usage, and the profiler's full percentile breakdown.

## Prompt suite

Eight prompts are included, chosen to stress different code paths:

| Label | Focus |
|---|---|
| `short-factual` | Minimal prefill, near-instant EOS |
| `short-qa` | Short prompt, brief factual answer |
| `medium-explain` | Mid-length prompt, a few sentences of output |
| `medium-code` | Code generation path |
| `long-reasoning` | Long prompt, multi-step decode |
| `long-creative` | Long free-form output, stresses decode throughput |
| `system-prompt-heavy` | Long input to summarise, tests prefill at scale |
| `long-list` | Structured list output, many decode steps |

If `--requests N` exceeds the suite size, prompts are cycled.

## Usage

```bash
# defaults (8 requests, 2 warmup, max 128 output tokens)
python target-model/run_benchmark.py

# custom run
python target-model/run_benchmark.py \
    --max-tokens  256  \
    --temperature 0.0  \
    --warmup      3    \
    --requests    16

# skip profiler overhead
python target-model/run_benchmark.py --no-profiler
```

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model ID or local path |
| `--max-tokens` | `128` | Max output tokens per request |
| `--temperature` | `0.7` | Sampling temperature (0 = greedy) |
| `--top-p` | `0.9` | Nucleus sampling threshold |
| `--rep-penalty` | `1.1` | Repetition penalty (1.0 = off) |
| `--warmup` | `2` | Requests to discard before recording |
| `--requests` | `8` | Total benchmark requests |
| `--system` | `"You are a helpful assistant."` | System prompt applied to every request |
| `--no-profiler` | off | Disable profiler instrumentation |

## Re-running after engine changes

Because the benchmark calls only the public interface (`TargetEngine.from_pretrained()` and `engine.generate()`), you can swap in a new KV cache or quantisation implementation inside `target_engine.py` and re-run with the exact same command to get directly comparable numbers.
