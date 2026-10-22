"""Measure usable matmul throughput (TFLOPS) on the GPU and CPU (Week 1).

Why
---
Prefill is mostly large matrix multiplies, so its ceiling is
FLOPs / usable FLOPS. Spec sheets give peak FLOPS at maximum boost clock;
this script measures what a real matmul achieves. ceiling.py uses this
number for TTFT.

What it measures
----------------
C = A @ B on square N x N matrices (N = 4096 by default):
  GPU (MX450): FP16, the dtype Stride computes in. bitsandbytes turns the
               NF4 weights back into FP16 before each prefill matmul.
               Also FP32: the MX450 has no tensor cores, and on it the FP16
               matmul measured ~3x slower than FP32 (2026-10-17).
  CPU:         FP32, PyTorch's multithreaded CPU matmul.
One matmul does N^3 multiply-adds = 2 x N^3 FLOPs.

Why the numbers are trustworthy
-------------------------------
- Warmup runs first, so clocks have ramped up and the math library has
  picked its kernel before timing starts.
- Each call is timed with a synchronize before and after, so the time covers
  the whole matmul on the device. A call takes tens of milliseconds, so the
  synchronize overhead (microseconds) is well under 1%.
- The result is checked against a float64 reference computed on the CPU for
  the first 64 rows. The relative error must be small for the dtype (FP16
  output keeps ~3 decimal digits, FP32 ~7).

Output
------
best TFLOPS:    rate of the fastest call (1 TFLOPS = 1e12 FLOP/s).
median TFLOPS:  rate of the typical call. Laptops throttle under sustained
                load, so a big gap between the two means clocks dropped.
rel error:      ||C - reference|| / ||reference|| over the checked rows.

Usage: python bandwidth/matmul_flops.py [--device both|cuda|cpu] [--n 4096] [--iters 20]
"""
import argparse
import statistics
import time

import torch


def sync(device):
    """Wait until the device has finished all queued work (GPU only; CPU ops are synchronous)."""
    if device == "cuda":
        torch.cuda.synchronize()


def time_op(op, device, iters, warmup):
    """Run op() `warmup` times untimed, then `iters` times timed.

    Returns a list of per-call wall-clock times in seconds.
    """
    for _ in range(warmup):
        op()
    times = []
    for _ in range(iters):
        sync(device)  # nothing else in flight when the clock starts
        t0 = time.perf_counter()
        op()          # on the GPU this only queues the matmul...
        sync(device)  # ...so wait for it to finish before stopping the clock
        times.append(time.perf_counter() - t0)
    return times


def measure(device, dtype, n, iters, warmup):
    """Time an N x N matmul on one device; return (best TFLOPS, median TFLOPS, best s, rel error)."""
    torch.manual_seed(0)
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    c = torch.empty(n, n, device=device, dtype=dtype)  # written in place, no new allocation per call

    times = time_op(lambda: torch.matmul(a, b, out=c), device, iters, warmup)
    flops = 2 * n**3  # n^3 multiply-adds, 2 FLOPs each
    best = flops / min(times) / 1e12
    median = flops / statistics.median(times) / 1e12

    # Correctness: recompute the first 64 rows in float64 on the CPU, from the
    # same inputs, and compare. This reference doesn't use the device's
    # matmul at all, so a wrong or skipped kernel can't pass.
    ref = a[:64].cpu().double() @ b.cpu().double()
    err = ((c[:64].cpu().double() - ref).norm() / ref.norm()).item()
    tol = 1e-2 if dtype == torch.float16 else 1e-5
    assert err < tol, f"{device} matmul check failed: relative error {err:.1e} >= {tol:.0e}"
    return best, median, min(times), err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["both", "cuda", "cpu"], default="both")
    parser.add_argument("--n", type=int, default=4096, help="matrix size (N x N)")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=torch.get_num_threads(),
                        help="CPU threads (default: one per physical core)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    # GPU: FP16 (what Stride's prefill runs in) and FP32. CPU: FP32.
    runs = []
    if args.device in ("both", "cuda"):
        runs.append(("cuda", torch.float16))
        runs.append(("cuda", torch.float32))
    if args.device in ("both", "cpu"):
        runs.append(("cpu", torch.float32))

    print(f"{args.n} x {args.n} matmul, {args.iters} timed iters after {args.warmup} warmup\n")
    print(f"{'device':<30} {'dtype':<8} {'best TFLOPS':>11} {'median TFLOPS':>13} "
          f"{'best ms':>8} {'rel error':>10}")
    for device, dtype in runs:
        best, median, t, err = measure(device, dtype, args.n, args.iters, args.warmup)
        name = (torch.cuda.get_device_name(0) if device == "cuda"
                else f"CPU, {torch.get_num_threads()} threads")
        dtype_name = str(dtype).replace("torch.", "")
        print(f"{name:<30} {dtype_name:<8} {best:11.2f} {median:13.2f} {t * 1e3:8.1f} {err:10.1e}")
    print("\ncorrectness checks passed")


if __name__ == "__main__":
    main()
