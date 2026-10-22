"""STREAM-style CPU memory bandwidth test (Week 1).

What it measures
----------------
STREAM is the standard CPU memory bandwidth benchmark. It times four simple
loops over arrays much larger than the caches, so every byte comes from RAM.
This script runs the same four kernels, plus a read-only one:

  kernel  operation      arrays counted   bytes per call
  copy    a = b          2 (read b, write a)        2 x array
  scale   a = q*b        2                          2 x array
  add     a = b + c      3 (read b, c, write a)     3 x array
  triad   a = b + q*c    3                          3 x array
  read    sum(b)         1 (read b)                 1 x array

`read` is the access pattern of decode (read all weights, write almost
nothing), so it is the number to use for the roofline decode floor.

Why PyTorch and not numpy
-------------------------
Each kernel is one PyTorch CPU op. PyTorch splits it across all cores
(multithreaded) and writes straight into `a` (out=a), so no temporary arrays
are created and each array is touched exactly once. numpy runs elementwise
ops on one thread, which understates bandwidth, and it has to do triad in two
passes, which would make the byte count wrong.

How bytes are counted
---------------------
Before a CPU writes to a memory location, it first reads that cache line into
cache ("write-allocate"). STREAM's convention does not count that hidden
read, and neither does this script. So copy/scale/add/triad understate the
real RAM traffic (copy really moves 3 arrays, not 2). The read kernel has no
writes, so it is not affected.

Correctness
-----------
Inputs are fixed values (b = 2.0, c = 0.5, q = 3.0) whose results are exact in
float64. After each kernel, every element of `a` is checked against the
expected value; the sum is checked against the exact total.

Sizes: each array is 256 MiB by default (3 arrays = 768 MiB of RAM), far
above STREAM's rule of 4x the last-level cache (12 MB L3 here).
GB = 1e9 bytes, the unit spec sheets use.

Usage: python bandwidth/cpu_stream.py [--threads N] [--mib 256] [--iters 20]
"""
import argparse
import statistics
import time

import torch


def time_op(op, iters, warmup):
    """Run op() `warmup` times untimed, then `iters` times timed.

    Returns a list of per-call wall-clock times in seconds. PyTorch CPU ops
    are synchronous (they finish before returning), so a CPU timer around
    each call measures the whole operation.
    """
    # Warmup: the first touches of a fresh array trigger OS page faults, and
    # the thread pool needs to start up. Neither should be in the timing.
    for _ in range(warmup):
        op()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()  # highest-resolution clock Python offers
        op()
        times.append(time.perf_counter() - t0)
    return times


def main():
    parser = argparse.ArgumentParser()
    # Default thread count is PyTorch's default: one per physical core (4 here).
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--mib", type=int, default=256, help="size of each array in MiB")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    # Three float64 arrays (8 bytes per element), as in the original STREAM.
    # b and c are inputs and never change; every kernel writes its result
    # into a. These values make every result exact in float64.
    n = args.mib * 2**20 // 8  # number of float64 elements per array
    q = 3.0                    # the STREAM "scalar"
    a = torch.zeros(n, dtype=torch.float64)
    b = torch.full((n,), 2.0, dtype=torch.float64)
    c = torch.full((n,), 0.5, dtype=torch.float64)
    array_bytes = n * a.element_size()  # bytes in one array

    # sum() returns a new tensor; store it so the last result can be checked.
    result = {}
    def read():
        result["sum"] = b.sum()

    # Each entry: name, the operation, how many arrays it moves (STREAM
    # counting), and the value every element of `a` must hold afterwards.
    # The expected values all differ from each other (2.0 -> 6.0 -> 2.5 -> 3.5),
    # so a passing check proves the kernel rewrote every element of `a`
    # rather than leaving the previous kernel's output behind.
    kernels = [
        ("copy", lambda: a.copy_(b), 2, 2.0),                       # a = b
        ("scale", lambda: torch.mul(b, q, out=a), 2, 6.0),          # a = q*b
        ("add", lambda: torch.add(b, c, out=a), 3, 2.5),            # a = b + c
        ("triad", lambda: torch.add(b, c, alpha=q, out=a), 3, 3.5), # a = b + q*c, one pass
        ("read", read, 1, None),                                    # sum(b), checked below
    ]

    print(f"{torch.get_num_threads()} threads, 3 arrays of {args.mib} MiB float64, "
          f"{args.iters} timed iters after {args.warmup} warmup\n")
    print(f"{'test':<6} {'best GB/s':>9} {'median GB/s':>11} {'best ms':>9}")
    for name, op, arrays, expected in kernels:
        times = time_op(op, args.iters, args.warmup)
        nbytes = arrays * array_bytes  # bytes one call moves (STREAM counting)
        # best = fastest call (STREAM's reported number); median = typical call.
        print(f"{name:<6} {nbytes / min(times) / 1e9:9.1f} "
              f"{nbytes / statistics.median(times) / 1e9:11.1f} {min(times) * 1e3:9.2f}")
        # Check every element of a, after timing so it doesn't affect the numbers.
        if expected is not None:
            assert torch.all(a == expected), f"{name} check failed"

    # The sum of n twos is 2n. All partial sums are small integers, which
    # float64 represents exactly, so this check can be exact equality.
    assert result["sum"].item() == 2.0 * n, f"read check failed: {result['sum'].item()} != {2.0 * n}"
    print("\ncorrectness checks passed")


if __name__ == "__main__":
    main()
