# DraftEngine Benchmark Log

Benchmark results for `draft-model/run_benchmark.py`.  
Each run is appended by the `bench` script with a description, timestamp, and full output.

**Usage:**
```bash
bench "description of this run" draft-model/run_benchmark.py [args...]
```

---


---

---

## baseline

> **Date:** 2026-08-05 22:26:51  
> **Script:** `/c/Users/omara/Desktop/projects/stride/draft-model/run_benchmark.py`

```

══════════════════════════════════════════════════════════════════════════════════════════
  Per-request results
──────────────────────────────────────────────────────────────────────────────────────────
  Prompt label               P-tok   O-tok  Prefill ms     Gen ms     Tok/s
──────────────────────────────────────────────────────────────────────────────────────────
  short-factual                  5      64        46.0     2043.8      31.3
  short-creative                13      64       123.1     2333.9      27.4
  medium-technical              17      64        90.8     2112.5      30.3
  medium-code                   14      64       101.7     2023.2      31.6
  long-reasoning                36      64       178.5     2230.0      28.7
  long-creative                 23      64       128.4     2393.9      26.7
──────────────────────────────────────────────────────────────────────────────────────────

  ── Aggregate statistics ──────────────────────────────────────
  Requests       : 6
  Total out-toks : 384
  Gen latency ms : mean=2189.5  p50=2171.3  p95=2393.9  min=2023.2  max=2393.9
  Prefill ms     : mean=111.4  p50=112.4  p95=178.5  min=46.0  max=178.5
  Throughput t/s : mean=29.3  p50=29.5  p95=31.6  min=26.7  max=31.6
──────────────────────────────────────────────────────────────────────────────────────────

  ── Profiler detail ───────────────────────────────────────────

======================================================================
  Stride Profiler Report  |  elapsed: 13.14s
======================================================================

  [DRAFT]
  metric                     count         mean          p50          p95          p99          min          max
  --------------------------------------------------------------------------------------------------------------
  e2e_latency_ms                 6     2076.43ms     2034.97ms     2250.20ms     2261.13ms     1919.78ms     2263.86ms
  prefill_ms                     6      111.40ms      112.40ms      165.97ms      175.99ms       45.96ms      178.50ms
  eval_ms                      384       18.04ms       17.41ms       21.80ms       26.62ms       12.79ms      105.48ms

  counter                           value
  ----------------------------------------
  requests_completed                    6
  tokens_generated                    384
  tok_throughput                    55.4 tok/s
  request_throughput                0.48 req/s

  gauge                            latest       mean
  --------------------------------------------------
  kv_seq_len                        87.0       50.0

======================================================================

══════════════════════════════════════════════════════════════════════════════════════════
```

