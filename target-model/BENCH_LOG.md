# TargetEngine Benchmark Log

Benchmark results for `target-model/run_benchmark.py`.  
Each run is appended by the `bench` script with a description, timestamp, and full output.

**Usage:**
```bash
bench "description of this run" target-model/run_benchmark.py [args...]
```

---


---

## baseline

> **Date:** 2026-08-05 22:30:35  
> **Script:** `/c/Users/omara/Desktop/projects/stride/target-model/run_benchmark.py`

```

════════════════════════════════════════════════════════════════════════════════════════════════════
  Per-request results
────────────────────────────────────────────────────────────────────────────────────────────────────
  Prompt label                 P-tok   O-tok    TTFT ms Decode ms/t     E2E ms     Tok/s
────────────────────────────────────────────────────────────────────────────────────────────────────
  short-factual                   24       3      501.9       74.17     1056.8       2.8
  short-qa                        29      33      498.1       78.42     3634.4       9.1
  medium-explain                  39      65      495.4       73.25     6120.9      10.6
  medium-code                     46     128      500.9       77.14    11882.5      10.8
  long-reasoning                  52     128      492.0       74.43    11536.2      11.1
  long-creative                   43     128      500.7       74.47    11532.6      11.1
  system-prompt-heavy             92      40      851.6       74.54     4630.7       8.6
  long-list                       41     110      507.9       74.03     9966.7      11.0
────────────────────────────────────────────────────────────────────────────────────────────────────

  ── Aggregate statistics ─────────────────────────────────────────────
  Requests          : 8
  Total output toks : 635
  E2E latency       : mean=7545.1 ms  p50=8043.8 ms  p95=11761.3 ms  min=1056.8 ms  max=11882.5 ms
  TTFT (prefill)    : mean=543.6 ms  p50=500.8 ms  p95=731.3 ms  min=492.0 ms  max=851.6 ms
  Decode latency    : mean=75.1 ms/tok  p50=74.4 ms/tok  p95=78.0 ms/tok  min=73.2 ms/tok  max=78.4 ms/tok
  Throughput        : mean=9.4 tok/s  p50=10.7 tok/s  p95=11.1 tok/s  min=2.8 tok/s  max=11.1 tok/s
────────────────────────────────────────────────────────────────────────────────────────────────────
  VRAM (post-bench): 1.16 GB allocated  |  1.26 GB reserved  |  2.15 GB total
────────────────────────────────────────────────────────────────────────────────────────────────────

  ── Profiler detail ─────────────────────────────────────────────────

======================================================================
  Stride Profiler Report  |  elapsed: 60.36s
======================================================================

  [TARGET]
  metric                     count         mean          p50          p95          p99          min          max
  --------------------------------------------------------------------------------------------------------------
  e2e_latency_ms                 8     7543.98ms     8042.66ms    11759.87ms    11856.82ms     1056.26ms    11881.06ms
  prefill_ms                     8      543.56ms      500.79ms      731.29ms      827.54ms      492.02ms      851.60ms
  decode_ms                    627       75.00ms       73.37ms       83.48ms       92.26ms       70.29ms      111.06ms
  sample_ms                    635       13.96ms       10.12ms       10.47ms      264.55ms       10.02ms      468.70ms

  counter                           value
  ----------------------------------------
  eos_hits                              5
  requests_completed                    8
  tokens_generated                    635
  tok_throughput                    13.5 tok/s
  tpot (time/output token)         75.01 ms
  request_throughput                0.13 req/s

  gauge                            latest       mean
  --------------------------------------------------
  kv_seq_len                       150.0      100.1
  vram_used_mb                    1122.1     1123.6

======================================================================


════════════════════════════════════════════════════════════════════════════════════════════════════
```

