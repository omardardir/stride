# TargetEngine Benchmark Log

Benchmark results for `target-model/run_benchmark.py`.  
Each run is appended by the `bench` script with a description, timestamp, and full output.

**Usage:**
```bash
bench "description of this run" target-model/run_benchmark.py [args...]
```

---

---

## baseline 2

> **Date:** 2026-09-14 00:25:30  
> **Script:** `/c/Users/omara/Desktop/projects/stride/target-model/run_benchmark.py`

```

════════════════════════════════════════════════════════════════════════════════════════════════════
  Per-request results
────────────────────────────────────────────────────────────────────────────────────────────────────
  Prompt label                 P-tok   O-tok    TTFT ms Decode ms/t     E2E ms     Tok/s
────────────────────────────────────────────────────────────────────────────────────────────────────
  short-factual                   24       3      821.0      195.68     1450.0       2.1
  short-qa                        29      30      823.8      140.88     4987.5       6.0
  medium-explain                  39      66      832.7      112.34     8301.3       8.0
  medium-code                     46     128      837.0       96.84    13360.3       9.6
  long-reasoning                  52     128      843.5      100.60    13833.5       9.3
  long-creative                   43     128      836.0       99.74    13719.0       9.3
  system-prompt-heavy             92      24     1388.6       83.21     3339.3       7.2
  long-list                       41      87      837.3       93.54     9023.0       9.6
────────────────────────────────────────────────────────────────────────────────────────────────────

  ── Aggregate statistics ─────────────────────────────────────────────
  Requests          : 8
  Total output toks : 594
  E2E latency       : mean=8501.7 ms  p50=8662.2 ms  p95=13793.4 ms  min=1450.0 ms  max=13833.5 ms
  Tokenization      : mean=0.6 ms  p50=0.5 ms  p95=0.8 ms  min=0.4 ms  max=0.8 ms
  TTFT (total)      : mean=902.5 ms  p50=836.5 ms  p95=1197.8 ms  min=821.0 ms  max=1388.6 ms
  Decode latency    : mean=115.4 ms/tok  p50=100.2 ms/tok  p95=176.5 ms/tok  min=83.2 ms/tok  max=195.7 ms/tok
  Throughput        : mean=7.6 tok/s  p50=8.6 tok/s  p95=9.6 tok/s  min=2.1 tok/s  max=9.6 tok/s
────────────────────────────────────────────────────────────────────────────────────────────────────
  VRAM (post-bench): 1.16 GB allocated  |  1.26 GB reserved  |  2.15 GB total
────────────────────────────────────────────────────────────────────────────────────────────────────

  ── Profiler detail ─────────────────────────────────────────────────

======================================================================
  Stride Profiler Report  |  elapsed: 68.02s
======================================================================

  [TARGET]
  metric                     count         mean          p50          p95          p99          min          max
  --------------------------------------------------------------------------------------------------------------
  e2e_latency_ms                 8     8501.09ms     8661.38ms    13792.39ms    13824.49ms     1449.87ms    13832.51ms
  tokenize_ms                    8        0.58ms        0.53ms        0.82ms        0.84ms        0.43ms        0.84ms
  prefill_ms                     8      901.91ms      835.98ms     1197.04ms     1349.58ms      820.52ms     1387.72ms
  decode_ms                    586      101.50ms       95.47ms      146.00ms      209.12ms       80.36ms      315.20ms
  sample_ms                    594        1.86ms        1.35ms        2.35ms        3.05ms        1.14ms      231.63ms

  counter                           value
  ----------------------------------------
  eos_hits                              5
  requests_completed                    8
  tokens_generated                    594
  tok_throughput                    10.0 tok/s
  tpot (time/output token)        101.50 ms
  request_throughput                0.12 req/s

  gauge                            latest       mean
  --------------------------------------------------
  kv_seq_len                       127.0       98.2
  vram_used_mb                    1122.1     1123.6

======================================================================


════════════════════════════════════════════════════════════════════════════════════════════════════
```

