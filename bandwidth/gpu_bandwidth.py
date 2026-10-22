"""Measure GPU memory bandwidth with a large device-to-device copy (Week 1).

What it measures
----------------
copy: dst.copy_(src). Each pass reads src once and writes dst once, so it is
      counted as 2 x buffer bytes, the same convention as CUDA's bandwidthTest.
read: src.sum(). Each pass reads src once and writes a single number, so it
      is counted as 1 x buffer bytes. Decode is read-dominated (it reads every
      weight once per token and writes almost nothing), so this is the
      number to use for the roofline decode floor.

Why the numbers are trustworthy
-------------------------------
- Buffers (256 MiB by default) are ~500x larger than the MX450's 512 KB L2
  cache, so every pass comes from GPU DRAM, not from cache.
- Time is measured on the GPU with CUDA events, not with a CPU clock, so the
  asynchronous kernel launches do not distort it.
- Warmup passes run first so clocks have ramped up and allocation is done.
- After timing, the outputs are checked: the copy must reproduce src exactly
  and the sum must equal the number of elements.

Output
------
best GB/s:    rate of the fastest pass (the convention STREAM also uses).
median GB/s:  rate of the typical pass.
spec peak:    theoretical bandwidth computed from the driver-reported memory
              clock and bus width; the measured rate is printed as a % of it.
GB = 1e9 bytes, the unit spec sheets use.

Usage: python bandwidth/gpu_bandwidth.py [--mib 256] [--iters 100] [--warmup 20]
"""
import argparse
import statistics

import torch


def time_op(op, iters, warmup):
    """Run op() `warmup` times untimed, then `iters` times timed.

    Returns a list of per-call GPU times in seconds.
    """
    # Warmup: lets the GPU reach its boost clocks and finishes any one-time
    # setup (kernel loading, allocator work) before we start timing.
    for _ in range(warmup):
        op()

    # CUDA events are timestamps written by the GPU itself into its command
    # stream. start.record() / end.record() bracket the work, so the elapsed
    # time is pure GPU execution time. One pair of events per iteration.
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        op()  # queued on the GPU; returns to Python immediately
        end.record()

    # The calls above only queued the work. Wait until the GPU has finished
    # everything before reading the event timestamps.
    torch.cuda.synchronize()

    # elapsed_time() returns milliseconds; convert to seconds.
    return [s.elapsed_time(e) / 1e3 for s, e in zip(starts, ends)]


def report(name, nbytes, times):
    """Print one result row and return the best rate in GB/s.

    nbytes is the number of bytes one call moves (read + written).
    """
    best = nbytes / min(times) / 1e9                  # fastest pass
    median = nbytes / statistics.median(times) / 1e9  # typical pass
    print(f"{name:<6} {best:8.1f} {median:10.1f} {min(times) * 1e3:10.3f}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib", type=int, default=256, help="size of each buffer in MiB")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    # Theoretical peak from what the driver reports for this exact card:
    #   memory_clock_rate is in kHz (3501000 kHz = 3501 MHz on this MX450),
    #   memory_bus_width is in bits (64).
    # Memory transfers data twice per reported clock, hence the x2;
    # bits / 8 = bytes moved per transfer across the whole bus.
    # MX450: 2 x 3501e6 x 64 / 8 = 56.0 GB/s.
    props = torch.cuda.get_device_properties(0)
    spec_peak = 2 * props.memory_clock_rate * 1e3 * props.memory_bus_width / 8 / 1e9

    # Two float32 buffers (4 bytes per element) on the GPU.
    # src is all ones and dst all zeros, so after the copy we can prove every
    # element was written (dst == src), and the sum of src must equal n.
    # Both buffers live in GPU memory: --mib 256 uses 512 MiB in total.
    n = args.mib * 2**20 // 4  # number of float32 elements per buffer
    src = torch.ones(n, device="cuda")
    dst = torch.zeros_like(src)
    nbytes = n * src.element_size()  # bytes in one buffer

    print(f"{props.name}: {props.memory_bus_width}-bit bus, "
          f"{props.memory_clock_rate / 1e3:.0f} MHz memory clock, "
          f"spec peak {spec_peak:.1f} GB/s")
    print(f"buffer {args.mib} MiB, {args.iters} timed iters after {args.warmup} warmup\n")
    print(f"{'test':<6} {'best GB/s':>8} {'median GB/s':>10} {'best ms':>10}")

    # Copy test: reads src (nbytes) and writes dst (nbytes) -> 2 x nbytes.
    # For two contiguous same-type GPU tensors, PyTorch does this with a
    # device-to-device cudaMemcpy, the same operation bandwidthTest times.
    copy_times = time_op(lambda: dst.copy_(src), args.iters, args.warmup)
    best_copy = report("copy", 2 * nbytes, copy_times)

    # Read test: sum() reads all of src and writes one number -> 1 x nbytes.
    # The result is kept in a dict so we can check the last sum afterwards.
    result = {}
    def read():
        result["sum"] = src.sum()
    read_times = time_op(read, args.iters, args.warmup)
    best_read = report("read", nbytes, read_times)

    # Correctness checks, run after timing so they don't affect the numbers.
    # 1) dst started as zeros; equal to src means the copy wrote every element.
    # 2) The sum of n ones must be n. A small tolerance covers float32
    #    rounding inside the GPU's parallel reduction.
    assert torch.equal(dst, src), "copy check failed"
    assert abs(result["sum"].item() - n) <= 1e-5 * n, f"read check failed: {result['sum'].item()} != {n}"
    print("\ncorrectness checks passed")
    print(f"copy reaches {100 * best_copy / spec_peak:.0f}% and read "
          f"{100 * best_read / spec_peak:.0f}% of spec peak")


if __name__ == "__main__":
    main()
