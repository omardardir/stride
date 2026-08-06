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

> **Date:** 2026-08-06 00:12:36  
> **Script:** `/c/Users/omara/Desktop/projects/stride/target-model/run_benchmark.py`

```

════════════════════════════════════════════════════════════════════════════════════════════════════
  Per-request results
────────────────────────────────────────────────────────────────────────────────────────────────────
  Prompt label                 P-tok   O-tok    TTFT ms Decode ms/t     E2E ms     Tok/s
────────────────────────────────────────────────────────────────────────────────────────────────────
  short-factual                   24       8      493.6       76.29     1440.3       5.6
  short-qa                        29      44      506.1       86.55     4937.4       8.9
  medium-explain                  39      82      500.6       80.12     8113.4      10.1
  medium-code                     46     128      501.9       78.48    12072.0      10.6
  long-reasoning                  52     128      498.7       79.36    12184.7      10.5
  long-creative                   43     128      496.8      107.20    15788.0       8.1
  system-prompt-heavy             92      36      875.8      116.87     5820.9       6.2
  long-list                       41      96      501.7       93.42    10642.8       9.0
────────────────────────────────────────────────────────────────────────────────────────────────────

  ── Aggregate statistics ─────────────────────────────────────────────
  Requests          : 8
  Total output toks : 650
  E2E latency       : mean=8874.9 ms  p50=9378.1 ms  p95=14526.9 ms  min=1440.3 ms  max=15788.0 ms
  TTFT (prefill)    : mean=546.9 ms  p50=501.1 ms  p95=746.4 ms  min=493.6 ms  max=875.8 ms
  Decode latency    : mean=89.8 ms/tok  p50=83.3 ms/tok  p95=113.5 ms/tok  min=76.3 ms/tok  max=116.9 ms/tok
  Throughput        : mean=8.6 tok/s  p50=9.0 tok/s  p95=10.6 tok/s  min=5.6 tok/s  max=10.6 tok/s
────────────────────────────────────────────────────────────────────────────────────────────────────
  VRAM (post-bench): 1.16 GB allocated  |  1.26 GB reserved  |  2.15 GB total
────────────────────────────────────────────────────────────────────────────────────────────────────

  ── Profiler detail ─────────────────────────────────────────────────

======================================================================
  Stride Profiler Report  |  elapsed: 71.00s
======================================================================

  [TARGET]
  metric                     count         mean          p50          p95          p99          min          max
  --------------------------------------------------------------------------------------------------------------
  e2e_latency_ms                 8     8873.72ms     9376.92ms    14525.12ms    15534.01ms     1439.76ms    15786.24ms
  prefill_ms                     8      546.90ms      501.14ms      746.39ms      849.90ms      493.59ms      875.77ms
  decode_ms                    642       89.36ms       79.80ms      136.90ms      185.50ms       70.23ms      254.26ms
  sample_ms                    650       14.02ms       10.19ms       11.18ms      259.74ms       10.03ms      471.65ms

  counter                           value
  ----------------------------------------
  eos_hits                              5
  requests_completed                    8
  tokens_generated                    650
  tok_throughput                    11.3 tok/s
  tpot (time/output token)         89.36 ms
  request_throughput                0.11 req/s

  gauge                            latest       mean
  --------------------------------------------------
  kv_seq_len                       136.0       98.2
  vram_used_mb                    1122.1     1123.6

======================================================================


════════════════════════════════════════════════════════════════════════════════════════════════════
```


---

## Switched both prefill_ms and decode_ms to use cuda_event_time to record a torch.cuda.Event before and after the model forward pass

> **Date:** 2026-08-06 18:06:50  
> **Script:** `/c/Users/omara/Desktop/projects/stride/target-model/run_benchmark.py`

```

════════════════════════════════════════════════════════════════════════════════════════════════════
  Per-request results
────────────────────────────────────────────────────────────────────────────────────────────────────
  Prompt label                 P-tok   O-tok    TTFT ms Decode ms/t     E2E ms     Tok/s
────────────────────────────────────────────────────────────────────────────────────────────────────
  short-factual                   24       8      502.9       74.27     1487.0       5.4
  short-qa                        29      30      502.5       74.27     3221.9       9.3
  medium-explain                  39     108      495.1       74.00     9799.3      11.0
  medium-code                     46     128      500.9       75.06    11619.4      11.0
  long-reasoning                  52     128      497.4       77.69    11959.6      10.7
  long-creative                   43     128      500.7       80.96    12395.0      10.3
  system-prompt-heavy             92      41      845.8       82.05     5043.5       8.1
  long-list                       41      92      496.1       77.17     8735.9      10.5
────────────────────────────────────────────────────────────────────────────────────────────────────

  ── Aggregate statistics ─────────────────────────────────────────────
  Requests          : 8
  Total output toks : 663
  E2E latency       : mean=8032.7 ms  p50=9267.6 ms  p95=12242.6 ms  min=1487.0 ms  max=12395.0 ms
  TTFT (prefill)    : mean=542.7 ms  p50=500.8 ms  p95=725.8 ms  min=495.1 ms  max=845.8 ms
  Decode latency    : mean=76.9 ms/tok  p50=76.1 ms/tok  p95=81.7 ms/tok  min=74.0 ms/tok  max=82.0 ms/tok
  Throughput        : mean=9.6 tok/s  p50=10.4 tok/s  p95=11.0 tok/s  min=5.4 tok/s  max=11.0 tok/s
────────────────────────────────────────────────────────────────────────────────────────────────────
  VRAM (post-bench): 1.16 GB allocated  |  1.26 GB reserved  |  2.15 GB total
────────────────────────────────────────────────────────────────────────────────────────────────────

  ── Profiler detail ─────────────────────────────────────────────────

======================================================================
  Stride Profiler Report  |  elapsed: 64.27s
======================================================================

  [TARGET]
  metric                     count         mean          p50          p95          p99          min          max
  --------------------------------------------------------------------------------------------------------------
  e2e_latency_ms                 8     8031.48ms     9266.38ms    12241.02ms    12362.90ms     1486.38ms    12393.36ms
  prefill_ms                     8      542.67ms      500.81ms      725.79ms      821.80ms      495.13ms      845.81ms
  decode_ms                    655       77.22ms       75.58ms       90.63ms       99.85ms       67.86ms      111.54ms
  sample_ms                    663       13.88ms       10.12ms       10.57ms      266.29ms       10.02ms      482.19ms

  counter                           value
  ----------------------------------------
  eos_hits                              5
  requests_completed                    8
  tokens_generated                    663
  tok_throughput                    13.1 tok/s
  tpot (time/output token)         77.22 ms
  request_throughput                0.12 req/s

  gauge                            latest       mean
  --------------------------------------------------
  kv_seq_len                       132.0      100.3
  vram_used_mb                    1122.1     1123.6

======================================================================


════════════════════════════════════════════════════════════════════════════════════════════════════
```

