"""
profiler.py — Unified inference profiler for stride
=====================================================

One Profiler object, shared across the CPU draft engine and the GPU target
engine.  Each engine tags its entries with a prefix ("draft" / "target") so
you can filter, aggregate, or compare per-engine from a single combined log.

Metrics tracked
---------------
Timings (recorded in milliseconds, aggregated into percentiles):

  ttft_ms               — Time to First Token: wall-clock from the moment
                          the engine receives a request (after tokenization
                          if you wrap the tokenizer call too) to the moment
                          the first output token is sampled.  Encompasses
                          queue wait + tokenization + prefill.
  prefill_ms            — forward pass over the full prompt (prefill phase
                          only, no tokenization overhead).
  inter_token_latency_ms— time between emitting token N and token N+1
                          (i.e., one decode-step duration).  Identical in
                          value to decode_ms; kept as a named alias so
                          report output is unambiguous.
  decode_ms             — same samples as inter_token_latency_ms; retained
                          for backwards compatibility with existing call sites.
  e2e_latency_ms        — End-to-end latency per request: wall-clock from
                          request entry to last token emitted.  Use the
                          time_request() context manager to record this
                          automatically.
  draft_ms              — full speculative draft step (γ tokens on CPU)
  eval_ms               — llama.cpp eval() call for one token (CPU engine)
  sample_ms             — sampling / argmax after logit extraction
  verify_ms             — target model verification pass (GPU engine)
  transfer_ms           — host→device / device→host data movement

Derived metrics (computed on demand, not stored as raw samples):

  tpot()                — Time Per Output Token = decode_ms.mean()
                          (total decode time / tokens generated).
                          Equivalent to inter_token_latency mean.
  request_throughput()  — Requests per second = requests_completed /
                          total_e2e_time_s.  Only meaningful when
                          request sizes (prompt + output) are comparable.
  acceptance_rate()     — α = tokens_accepted / (accepted + rejected)
  throughput()          — Output tokens per second (decode phase only)

Counters (monotonically increasing integers):

  tokens_generated    — total output tokens produced
  tokens_accepted     — draft tokens accepted by the target (speculative loop)
  tokens_rejected     — draft tokens rejected
  requests_completed  — number of fully completed requests (incremented
                        automatically by time_request())
  draft_steps         — number of speculative draft rounds
  eos_hits            — how many times EOS stopped generation early
  cache_clears        — KV cache wipes (new requests)
  cache_rollbacks     — KV cache trims (rejected draft tokens)

Gauges (point-in-time snapshots, kept as a time-series list):

  kv_seq_len      — KV cache occupancy in tokens
  vram_used_mb    — VRAM in use (GPU engine; requires torch.cuda)
  ram_used_mb     — RSS RAM in use (CPU engine; requires psutil or tracemalloc)

Usage
-----
    from profiler import Profiler

    profiler = Profiler()                           # create once at top level

    # --- Inside GPU target engine ---
    with profiler.time("target", "prefill_ms"):
        outputs = model(input_ids=..., ...)
    profiler.count("target", "tokens_generated", n)
    profiler.gauge("target", "vram_used_mb", get_vram())

    # --- Inside CPU draft engine ---
    with profiler.time("draft", "draft_ms"):
        result = draft_loop(gamma)
    profiler.count("draft", "tokens_generated", gamma)

    # --- At the end ---
    profiler.report()                               # prints a summary table
    data = profiler.export()                        # returns a plain dict

GPU timing note
---------------
For GPU kernels the wall-clock approach (perf_counter) captures PCIe + kernel
dispatch overhead but not pure kernel time.  When you want kernel-accurate GPU
timing, use the optional `cuda_event_time()` context manager instead:

    with profiler.cuda_event_time("target", "decode_ms"):
        outputs = model(...)

This uses torch.cuda.Event with synchronise() to get true kernel duration.
Falls back to perf_counter if torch/CUDA is unavailable.
"""

from __future__ import annotations

import time
import json
import math
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

# Optional torch import — profiler works without it on CPU-only machines.
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Optional psutil for RAM gauge.
try:
    import psutil as _psutil
    import os as _os
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal storage types
# ---------------------------------------------------------------------------

@dataclass
class _TimingSeries:
    """Ordered list of timing samples (ms) for one (prefix, metric) pair."""
    samples: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def count(self) -> int:
        return len(self.samples)

    def total(self) -> float:
        return sum(self.samples)

    def mean(self) -> float:
        return self.total() / self.count() if self.samples else 0.0

    def min(self) -> float:
        return min(self.samples) if self.samples else 0.0

    def max(self) -> float:
        return max(self.samples) if self.samples else 0.0

    def percentile(self, p: float) -> float:
        """Return the p-th percentile (0–100) using linear interpolation."""
        if not self.samples:
            return 0.0
        sorted_s = sorted(self.samples)
        n = len(sorted_s)
        idx = (p / 100.0) * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return sorted_s[lo] + frac * (sorted_s[hi] - sorted_s[lo])

    def p50(self) -> float:  return self.percentile(50)
    def p95(self) -> float:  return self.percentile(95)
    def p99(self) -> float:  return self.percentile(99)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count":   self.count(),
            "total_ms": round(self.total(), 3),
            "mean_ms": round(self.mean(), 3),
            "min_ms":  round(self.min(), 3),
            "p50_ms":  round(self.p50(), 3),
            "p95_ms":  round(self.p95(), 3),
            "p99_ms":  round(self.p99(), 3),
            "max_ms":  round(self.max(), 3),
        }


@dataclass
class _GaugeSeries:
    """Time-series of point-in-time gauge snapshots."""
    samples: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, value)

    def record(self, value: float) -> None:
        self.samples.append((time.perf_counter(), value))

    def latest(self) -> float:
        return self.samples[-1][1] if self.samples else 0.0

    def mean(self) -> float:
        if not self.samples:
            return 0.0
        return sum(v for _, v in self.samples) / len(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count":  len(self.samples),
            "latest": round(self.latest(), 3),
            "mean":   round(self.mean(), 3),
        }


# ---------------------------------------------------------------------------
# Main Profiler class
# ---------------------------------------------------------------------------

class Profiler:
    """
    Unified profiler for CPU (draft) and GPU (target) inference engines.

    Thread-safe: all mutations go through a reentrant lock so the draft
    engine's background thread and the main GPU thread can record metrics
    concurrently without corruption.

    Parameters
    ----------
    enabled : bool
        Set to False to make every call a no-op (zero overhead in production).
    wall_clock_gpu : bool
        If True (default), GPU timings use perf_counter wall-clock time.
        Set to False to use torch.cuda.Event synchronised GPU kernel time
        inside cuda_event_time() — slightly more expensive but more accurate
        for isolating kernel duration from dispatch overhead.
    """

    # Known timing metric names — used for column ordering in report()
    _TIMING_METRICS = [
        "ttft_ms",
        "e2e_latency_ms",
        "prefill_ms",
        "inter_token_latency_ms",
        "decode_ms",
        "draft_ms",
        "eval_ms",
        "sample_ms",
        "verify_ms",
        "transfer_ms",
    ]

    # Known counter names
    _COUNTER_METRICS = [
        "tokens_generated",
        "tokens_accepted",
        "tokens_rejected",
        "requests_completed",
        "draft_steps",
        "eos_hits",
        "cache_clears",
        "cache_rollbacks",
    ]

    # Known gauge names
    _GAUGE_METRICS = [
        "kv_seq_len",
        "vram_used_mb",
        "ram_used_mb",
    ]

    def __init__(
        self,
        enabled: bool = True,
        wall_clock_gpu: bool = True,
    ) -> None:
        self.enabled = enabled
        self.wall_clock_gpu = wall_clock_gpu
        self._lock = threading.RLock()

        # {prefix: {metric_name: _TimingSeries}}
        self._timings:  dict[str, dict[str, _TimingSeries]]  = defaultdict(lambda: defaultdict(_TimingSeries))
        # {prefix: {metric_name: int}}
        self._counters: dict[str, dict[str, int]]             = defaultdict(lambda: defaultdict(int))
        # {prefix: {metric_name: _GaugeSeries}}
        self._gauges:   dict[str, dict[str, _GaugeSeries]]   = defaultdict(lambda: defaultdict(_GaugeSeries))

        self._start_time: float = time.perf_counter()

    # ------------------------------------------------------------------
    # Public API — timing
    # ------------------------------------------------------------------

    @contextmanager
    def time(self, prefix: str, metric: str) -> Generator[None, None, None]:
        """
        Context manager.  Records wall-clock duration of the block in ms.

            with profiler.time("target", "decode_ms"):
                outputs = model(...)
        """
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            self.record_timing(prefix, metric, ms)

    @contextmanager
    def time_request(self, prefix: str) -> Generator[None, None, None]:
        """
        Context manager that measures end-to-end request latency.

        Wrap the *entire* handling of one request — from the moment the
        engine receives the raw prompt string to the moment the last output
        token is returned.  Records `e2e_latency_ms` and increments
        `requests_completed`.

            with profiler.time_request("target"):
                tokenize → prefill → decode → detokenize

        Include tokenization in the block if you want TTFT and e2e to
        account for that overhead (recommended for realistic benchmarks).
        """
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            self.record_timing(prefix, "e2e_latency_ms", ms)
            self.count(prefix, "requests_completed", 1)

    @contextmanager
    def cuda_event_time(self, prefix: str, metric: str) -> Generator[None, None, None]:
        """
        Context manager using torch.cuda.Event for true GPU kernel timing.

        Falls back to wall-clock (perf_counter) if torch/CUDA is unavailable.
        The end event is synchronised on __exit__ so this blocks until the GPU
        kernel is done — only use where you're already synchronising anyway.

            with profiler.cuda_event_time("target", "decode_ms"):
                outputs = model(...)
        """
        if not self.enabled:
            yield
            return

        if _TORCH_AVAILABLE and _torch.cuda.is_available() and not self.wall_clock_gpu:
            start_ev = _torch.cuda.Event(enable_timing=True)
            end_ev   = _torch.cuda.Event(enable_timing=True)
            start_ev.record()
            try:
                yield
            finally:
                end_ev.record()
                _torch.cuda.synchronize()
                ms = start_ev.elapsed_time(end_ev)  # ms, float
                self.record_timing(prefix, metric, ms)
        else:
            # Fallback: wall-clock
            with self.time(prefix, metric):
                yield

    def record_timing(self, prefix: str, metric: str, ms: float) -> None:
        """Directly record a pre-measured timing in milliseconds."""
        if not self.enabled:
            return
        with self._lock:
            self._timings[prefix][metric].add(ms)

    # ------------------------------------------------------------------
    # Public API — counters
    # ------------------------------------------------------------------

    def count(self, prefix: str, metric: str, delta: int = 1) -> None:
        """Increment a counter by `delta` (default 1)."""
        if not self.enabled:
            return
        with self._lock:
            self._counters[prefix][metric] += delta

    def reset_counter(self, prefix: str, metric: str) -> None:
        """Reset a counter to zero."""
        with self._lock:
            self._counters[prefix][metric] = 0

    def get_counter(self, prefix: str, metric: str) -> int:
        """Read a counter value."""
        with self._lock:
            counter_dict = self._counters.get(prefix)
            return counter_dict.get(metric, 0) if counter_dict else 0

    # ------------------------------------------------------------------
    # Public API — gauges
    # ------------------------------------------------------------------

    def gauge(self, prefix: str, metric: str, value: float) -> None:
        """Record a point-in-time gauge snapshot."""
        if not self.enabled:
            return
        with self._lock:
            self._gauges[prefix][metric].record(value)

    # ------------------------------------------------------------------
    # Convenience: auto-capture hardware gauges
    # ------------------------------------------------------------------

    def snapshot_vram(self, prefix: str = "target") -> None:
        """
        Record current VRAM usage (MB) under `{prefix}.vram_used_mb`.
        No-op if torch/CUDA is unavailable.
        """
        if not self.enabled:
            return
        if _TORCH_AVAILABLE and _torch.cuda.is_available():
            mb = _torch.cuda.memory_allocated() / 1_048_576.0
            self.gauge(prefix, "vram_used_mb", mb)

    def snapshot_ram(self, prefix: str = "draft") -> None:
        """
        Record current process RSS (MB) under `{prefix}.ram_used_mb`.
        Requires psutil; falls back to tracemalloc if unavailable.
        """
        if not self.enabled:
            return
        mb: float = 0.0
        if _PSUTIL_AVAILABLE:
            proc = _psutil.Process(_os.getpid())
            mb = proc.memory_info().rss / 1_048_576.0
        else:
            import tracemalloc
            if tracemalloc.is_tracing():
                current, _ = tracemalloc.get_traced_memory()
                mb = current / 1_048_576.0
        if mb > 0:
            self.gauge(prefix, "ram_used_mb", mb)

    # ------------------------------------------------------------------
    # Derived / computed metrics
    # ------------------------------------------------------------------

    def acceptance_rate(self) -> float:
        """
        Draft token acceptance rate α = accepted / (accepted + rejected).
        Returns 0.0 if no speculative steps have been recorded yet.
        """
        with self._lock:
            accepted = sum(
                self._counters[p].get("tokens_accepted", 0)
                for p in self._counters
            )
            rejected = sum(
                self._counters[p].get("tokens_rejected", 0)
                for p in self._counters
            )
        total = accepted + rejected
        return accepted / total if total > 0 else 0.0

    def throughput(self, prefix: str) -> float:
        """
        Output token throughput for a given prefix, in tokens/second.
        Computed as: tokens_generated / total_per_token_time_s
        (decode phase only — does not include prefill time).

        Looks for decode_ms first (target/GPU engine); falls back to eval_ms
        (draft/CPU engine) so both engines produce a throughput figure.
        """
        with self._lock:
            timing_dict = self._timings.get(prefix)
            # Prefer decode_ms (target engine); fall back to eval_ms (draft engine)
            series = None
            if timing_dict:
                dec = timing_dict.get("decode_ms")
                evl = timing_dict.get("eval_ms")
                if dec and dec.count() > 0:
                    series = dec
                elif evl and evl.count() > 0:
                    series = evl
            total_ms = series.total() if series else 0.0
            counter_dict = self._counters.get(prefix)
            total_tok = counter_dict.get("tokens_generated", 0) if counter_dict else 0
        if total_ms <= 0 or total_tok == 0:
            return 0.0
        return total_tok / (total_ms / 1000.0)

    def tpot(self, prefix: str) -> float:
        """
        Time Per Output Token (ms) — mean time to produce one output token.

        Computed as decode_ms.mean().  Equivalent to the mean of
        inter_token_latency_ms samples.  The two metric names record the
        same underlying value; this method reads whichever has samples.
        """
        with self._lock:
            timing_dict = self._timings.get(prefix)
            if timing_dict is None:
                return 0.0
            # Prefer inter_token_latency_ms if explicitly recorded,
            # fall back to decode_ms (same thing, different call site name).
            itl = timing_dict.get("inter_token_latency_ms")
            dec = timing_dict.get("decode_ms")
            if itl and itl.count() > 0:
                series = itl
            elif dec and dec.count() > 0:
                series = dec
            else:
                return 0.0
        return series.mean()

    def request_throughput(self, prefix: str) -> float:
        """
        Request throughput in requests/second.

        Computed as: requests_completed / total_e2e_latency_s

        Only meaningful when request sizes (prompt length + output length)
        are comparable across requests.  If output lengths vary widely,
        compare requests of similar size separately.
        """
        with self._lock:
            timing_dict = self._timings.get(prefix)
            e2e_series = timing_dict.get("e2e_latency_ms") if timing_dict else None
            total_ms = e2e_series.total() if e2e_series else 0.0
            counter_dict = self._counters.get(prefix)
            n_requests = counter_dict.get("requests_completed", 0) if counter_dict else 0
        if total_ms <= 0 or n_requests == 0:
            return 0.0
        return n_requests / (total_ms / 1000.0)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, prefixes: list[str] | None = None) -> None:
        """
        Print a formatted summary table to stdout.

        Parameters
        ----------
        prefixes : list[str] | None
            Which engine prefixes to include.  None = all recorded prefixes.
        """
        with self._lock:
            all_prefixes = prefixes or sorted(self._timings.keys() | self._counters.keys() | self._gauges.keys())

        elapsed = time.perf_counter() - self._start_time
        print(f"\n{'='*70}")
        print(f"  Stride Profiler Report  |  elapsed: {elapsed:.2f}s")
        print(f"{'='*70}")

        for prefix in all_prefixes:
            print(f"\n  [{prefix.upper()}]")

            # --- Timings ---
            with self._lock:
                timing_keys = sorted(self._timings.get(prefix, {}).keys())
            if timing_keys:
                # Determine column order: known metrics first, then any extras
                ordered = [m for m in self._TIMING_METRICS if m in timing_keys]
                ordered += [m for m in timing_keys if m not in self._TIMING_METRICS]
                header = f"  {'metric':<24} {'count':>7} {'mean':>12} {'p50':>12} {'p95':>12} {'p99':>12} {'min':>12} {'max':>12}"
                print(header)
                print("  " + "-" * (len(header) - 2))
                for metric in ordered:
                    with self._lock:
                        s = self._timings.get(prefix, {}).get(metric)
                        if s is None:
                            continue
                        row = s.to_dict()
                    print(
                        f"  {metric:<24} {row['count']:>7} "
                        f"{row['mean_ms']:>11.2f}ms {row['p50_ms']:>11.2f}ms "
                        f"{row['p95_ms']:>11.2f}ms {row['p99_ms']:>11.2f}ms "
                        f"{row['min_ms']:>11.2f}ms {row['max_ms']:>11.2f}ms"
                    )

            # --- Counters ---
            with self._lock:
                counter_keys = sorted(self._counters.get(prefix, {}).keys())
            if counter_keys:
                print(f"\n  {'counter':<28} {'value':>10}")
                print("  " + "-" * 40)
                for metric in counter_keys:
                    with self._lock:
                        val = self._counters.get(prefix, {}).get(metric, 0)
                    print(f"  {metric:<28} {val:>10,}")

                # Derived metrics
                tput = self.throughput(prefix)
                if tput > 0:
                    print(f"  {'tok_throughput':<28} {tput:>9.1f} tok/s")
                tpot_ms = self.tpot(prefix)
                if tpot_ms > 0:
                    print(f"  {'tpot (time/output token)':<28} {tpot_ms:>9.2f} ms")
                req_tput = self.request_throughput(prefix)
                if req_tput > 0:
                    print(f"  {'request_throughput':<28} {req_tput:>9.2f} req/s")

            # --- Gauges ---
            with self._lock:
                gauge_keys = sorted(self._gauges.get(prefix, {}).keys())
            if gauge_keys:
                print(f"\n  {'gauge':<28} {'latest':>10} {'mean':>10}")
                print("  " + "-" * 50)
                for metric in gauge_keys:
                    with self._lock:
                        g = self._gauges.get(prefix, {}).get(metric)
                        if g is None:
                            continue
                        row = g.to_dict()
                    print(
                        f"  {metric:<28} {row['latest']:>9.1f}  {row['mean']:>9.1f}"
                    )

        # --- Cross-engine speculative decoding summary ---
        ar = self.acceptance_rate()
        if ar > 0:
            print(f"\n  [SPECULATIVE DECODING]")
            print(f"  acceptance_rate α = {ar:.3f}  ({ar*100:.1f}%)")

        print(f"\n{'='*70}\n")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self) -> dict[str, Any]:
        """
        Export all recorded data as a plain nested dict (JSON-serialisable).

        Schema:
            {
              "elapsed_s": float,
              "acceptance_rate": float,
              "prefixes": {
                "<prefix>": {
                  "timings":  { "<metric>": { count, mean_ms, p50_ms, ... } },
                  "counters": { "<metric>": int },
                  "gauges":   { "<metric>": { count, latest, mean } },
                  "throughput_tok_per_s": float,
                }
              }
            }
        """
        with self._lock:
            result: dict[str, Any] = {
                "elapsed_s": round(time.perf_counter() - self._start_time, 3),
                "acceptance_rate": round(self.acceptance_rate(), 4),
                "prefixes": {},
            }
            all_prefixes = sorted(
                self._timings.keys() | self._counters.keys() | self._gauges.keys()
            )
            for prefix in all_prefixes:
                result["prefixes"][prefix] = {
                    "timings": {
                        m: s.to_dict()
                        for m, s in self._timings.get(prefix, {}).items()
                    },
                    "counters": dict(self._counters.get(prefix, {})),
                    "gauges": {
                        m: g.to_dict()
                        for m, g in self._gauges.get(prefix, {}).items()
                    },
                    "throughput_tok_per_s": round(self.throughput(prefix), 2),
                }
        return result

    def export_json(self, path: str) -> None:
        """Write the full export dict to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export(), f, indent=2)
        print(f"[profiler] Saved to {path}")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded data (useful between benchmark runs)."""
        with self._lock:
            self._timings.clear()
            self._counters.clear()
            self._gauges.clear()
            self._start_time = time.perf_counter()


# ---------------------------------------------------------------------------
# Module-level singleton  (optional convenience)
# ---------------------------------------------------------------------------

# Engines can do `from profiler import default_profiler` if they want a
# shared instance without wiring one through the call stack.
# The top-level script can replace this with its own configured instance.
default_profiler = Profiler()
