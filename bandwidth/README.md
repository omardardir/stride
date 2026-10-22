# Week 1 — Measured memory bandwidth and matmul throughput

Session date: 2026-10-16 · Plan: Edge VLM Inference, Week 1 (Roofline) · Status: **measured, CPU rerun pending**

This folder covers the Week 1 task "measure real bandwidth: a large GPU device-to-device copy and a STREAM-style CPU test." From here on, use these numbers instead of spec sheets.

## Headline numbers

| Device | Number to use for the decode floor | Theoretical peak | % of peak |
|---|---|---|---|
| MX450 GPU | **51.6 GB/s** (read) | 56.0 GB/s | 92% |
| i7-1165G7 CPU, 4 threads | **~29–34 GB/s** (read, best) · ~26 GB/s median | 51.2 GB/s | 57–66% |

Decode reads the weights and writes almost nothing, so the **read** test is the roofline number. Copy and triad numbers are kept as reference points.

GB = 1e9 bytes, the unit spec sheets use.

## Files

| File | What it measures |
|---|---|
| [gpu_bandwidth.py](gpu_bandwidth.py) | `dst.copy_(src)` on two 256 MiB buffers (counted as 2× bytes, like CUDA's bandwidthTest), plus a read-only `src.sum()` (1× bytes). Timed with CUDA events: 20 warmup runs, then 100 timed runs. Also prints the theoretical peak from the driver-reported memory clock and bus width. |
| [cpu_stream.py](cpu_stream.py) | The four STREAM kernels (copy, scale, add, triad) plus a read-only sum, on three 256 MiB float64 arrays. Each kernel is one multithreaded PyTorch op with no temporary arrays. Timed with `perf_counter`: 3 warmup runs, then 20 timed runs. `--threads N` sets the thread count. |
| [matmul_flops.py](matmul_flops.py) | Usable matmul throughput: 4096 × 4096 matmul (2 × N³ FLOPs) in FP16 and FP32 on the GPU, and FP32 on the CPU. 5 warmup runs, then 20 timed runs, with a synchronize around each call. The first 64 rows are checked against a float64 CPU reference. |

All three scripts report the best and median rate, then run a correctness check (see **Method**).

### Run

```bash
.venv/Scripts/python.exe bandwidth/gpu_bandwidth.py
```

```bash
.venv/Scripts/python.exe bandwidth/cpu_stream.py
```

```bash
.venv/Scripts/python.exe bandwidth/matmul_flops.py
```

**Dependencies:** only `torch`, which is already in the project `.venv` (2.13.0+cu130). Nothing to install.

## Hardware (resolves Open facts in the Context doc)

| Item | Value | Source |
|---|---|---|
| CPU | Intel Core i7-1165G7 (Tiger Lake), 4 cores / 8 threads, 12 MB L3 | WMI `Win32_Processor` |
| OS | Windows 11 Home (WDDM driver model) | environment |
| RAM | 16 GB = 2 × 8 GB DDR4-3200, **dual-channel** | WMI `Win32_PhysicalMemory`; dual-channel inferred from measurement (below) |
| GPU | NVIDIA GeForce MX450, 2 GB, Turing (sm_75), 14 SMs, 512 KB L2 | `torch.cuda.get_device_properties` |
| GPU memory | 64-bit bus, 3501 MHz memory clock → 7 Gbps per pin | `torch.cuda.get_device_properties`, `nvidia-smi` |
| Power | Plugged in during all runs | WMI `Win32_Battery` |

Intel CPU, so **OpenVINO and the Iris Xe iGPU stay in scope** for Week 2.

### Theoretical peaks

- **GPU:** 2 × 3501 MHz × 64 bit / 8 = **56.0 GB/s**
- **CPU:** 2 channels × 3200 MT/s × 8 B = **51.2 GB/s**

> **Correction to `specs.md`:** it lists the MX450 as "GDDR6, ~80 GB/s". This laptop's card runs at 7 Gbps per pin on a 64-bit bus, which gives 56 GB/s, and it measures 51.6 GB/s. The 80 GB/s belongs to the 10 Gbps GDDR6 variant. That 7 Gbps means GDDR5 is an inference from the clock, not a confirmed spec.

## Results

### Predictions (written before running)

| Test | Predicted | Reasoning |
|---|---|---|
| GPU copy | 45 GB/s | ~80% of 56 GB/s, typical for GDDR5 copies |
| GPU read | 48 GB/s | ~85% of peak |
| CPU read, 4 threads | 38 GB/s | ~75% of 51.2 GB/s |
| CPU triad, 4 threads | 29 GB/s | STREAM doesn't count the write-allocate read (see Method), so the counted rate is lower |
| CPU copy, 4 threads | 25 GB/s | same, with a bigger write-allocate share |

### GPU — stable and repeatable

| Buffer | Test | Best GB/s | Median GB/s | Best ms |
|---|---|---|---|---|
| 256 MiB | copy | 48.4 | 48.1 | 11.09 |
| 256 MiB | read | 51.6 | 51.5 | 5.21 |
| 512 MiB | copy | 48.1 | 48.1 | 22.32 |
| 512 MiB | read | 51.6 | 51.6 | 10.41 |

The measured rates came in 8% above the predictions. Best and median are within 1%, and doubling the buffer doubles the time, so the rate doesn't depend on buffer size.

### CPU — real but noisy

256 MiB arrays unless noted. Rates are best / median in GB/s.

| Test | 1 thread | 4 threads (run A) | 4 threads (run B) | 8 threads |
|---|---|---|---|---|
| copy | 16.4 / 12.7 | 21.4 / 19.7 | 23.4 / 21.5 | 24.5 / 23.8 |
| scale | 15.7 / 13.4 | 20.1 / 18.1 | 23.3 / 22.0 | 25.3 / 23.8 |
| add | 17.2 / 15.6 | 24.1 / 22.6 | 27.1 / 23.6 | 27.5 / 24.7 |
| triad | 17.0 / 15.6 | 24.9 / 23.0 | 26.9 / 24.4 | 27.7 / 23.0 |
| **read** | 15.3 / 13.3 | **29.0 / 25.8** | **33.6 / 26.7** | 25.5 / 16.9 |

- The 4-thread results came in 12–24% below the predictions. Background load is the likely cause (see Caveats).
- One core reaches only about half the 4-thread read rate, so llama.cpp on CPU needs several threads to use the memory bus.
- 8 threads (hyperthreading) doesn't help read, and its median is the worst of all.
- **Discarded run:** 1 GiB arrays, 4 threads, gave read 25.4 / 18.2 and triad 23.1 / 19.6. Only about 4 GB of RAM was free, so this run was likely under memory pressure.

## Matmul throughput (usable FLOPS)

Prefill is mostly matrix multiplies, so its ceiling is FLOPs ÷ usable FLOPS. All runs below are from 2026-10-17, **on battery**: rerun plugged in.

| Device | dtype | Predicted | Best TFLOPS | Median TFLOPS | Relative error vs float64 |
|---|---|---|---|---|---|
| MX450 | FP16 | 4.0 | **0.35** | 0.35 | 2.1e-4 |
| MX450 | FP32 | – | **1.11** | 0.99 | 1.1e-6 |
| i7-1165G7, 4 threads | FP32 | 0.30 | **0.10** | 0.10 | 3.6e-7 |

How the predictions were made:

- **MX450:** 896 cores × 2 FLOPs × 1.785 GHz = 3.2 TFLOPS in FP32. This chip is listed as running FP16 at twice that rate, about 6.4 TFLOPS.
- **CPU:** 4 cores × 32 FP32 FLOPs per cycle × about 3.5 GHz ≈ 0.45 TFLOPS.

Findings:

- **FP16 matmul is 3× slower than FP32 on the MX450.** This is the opposite of what the spec suggests. The MX450 has no tensor cores, and PyTorch's FP16 matmul on this GPU is evidently a slow path; the cause hasn't been diagnosed yet.
  - The SM clock held at 1860 MHz during the run, so this isn't throttling.
  - Stride computes in FP16 (`bnb_4bit_compute_dtype=torch.float16`), so its prefill matmuls run at the slow rate.
- **The GPU's FP32 matmul reaches only about 33% of its 3.3 TFLOPS peak** (896 cores × 2 × 1.86 GHz). Battery power limits (throttle reason `0x4`) may be part of the reason. Rerun plugged in.
- **The CPU came in 3× below prediction.** On battery, Windows holds the CPU to a lower power budget, which is the likely cause. Rerun plugged in.

How these numbers are used in `ceiling.py`:

- The `mx450` preset uses the FP32 rate (1.11), because the ceiling is the best any runtime could do on this device.
- Pass `--tflops 0.35` to model Stride's own FP16 path instead.
- For Stride's workload (128-token prompt, logits for every prompt token, 396.6 GFLOPs of prefill), the TTFT ceiling is about 357 ms at 1.11 TFLOPS and about 1133 ms at 0.35. Prefill is compute-bound either way; the memory floor is 22 ms.

## Method and why the numbers can be trusted

1. **Every element is checked.**
   - GPU copy: `dst` starts at 0 and must equal `src` (all ones) afterwards.
   - CPU kernels: each one writes a value that differs from the previous kernel's output (2.0 → 6.0 → 2.5 → 3.5), and the script checks that every element of the output array changed.
   - Read tests: each sum is compared with the exact expected value.
2. **The buffers far exceed the caches.**
   - GPU: 256 MiB against a 512 KB L2.
   - CPU: 256 MiB per array against 12 MB of L3. STREAM's rule is at least 4× the last-level cache.
3. **The GPU result doesn't depend on buffer size.** The rate was the same at 256 MiB and 512 MiB, so the test is limited by DRAM, not by caches.
4. **The CPU kernels agree once write-allocate is counted.**
   - Before a CPU writes to memory, it first reads the destination cache line (write-allocate). STREAM's convention doesn't count that read.
   - Counting it, copy moves 3 arrays instead of 2: 23.4 × 3/2 ≈ 35 GB/s.
   - Triad moves 4 instead of 3: 26.9 × 4/3 ≈ 36 GB/s.
   - Both match the read test, about 34 GB/s. So the real DRAM throughput is about 34–36 GB/s, roughly 70% of peak.
5. **Independent cross-check.** numpy's single-thread `np.copyto` scored 25.1 GB/s, beating PyTorch's copy at 1 and 4 threads. The likely reason is that numpy's memcpy uses non-temporal stores, which skip the write-allocate read. This supports point 4: PyTorch's copy and triad understate DRAM traffic, while read does not.
6. **The RAM is dual-channel.** A single DDR4-3200 channel can deliver at most 25.6 GB/s. The read test exceeded that (29–34 GB/s), so the RAM must be dual-channel.
7. **Why the CPU test uses PyTorch, not numpy.** numpy runs elementwise ops on a single thread, and it can only do triad in two passes. That would understate bandwidth or miscount the bytes.

## Caveats

- **Rerun the CPU test before recording it.** During these runs the laptop showed 17% CPU load from other apps (VS Code, Claude, Chrome, Avast) and only about 4 GB of 15.8 GB RAM free. The CPU numbers varied about 15% between identical runs. Rerun plugged in, with those apps closed, using the default 256 MiB size, and record the best and median read.
- **Run plugged in.** On battery (2026-10-17), the GPU dropped to power state P3 with the memory at 3000 MHz instead of 3501 MHz, and nvidia-smi reported the software power cap (throttle reason `0x4`). Read fell to 44.2 GB/s and copy to 41.2 GB/s. That is still 92% of the lower 48.0 GB/s peak, but the script's "% of spec peak" line uses the plugged-in 3501 MHz clock, so on battery it shows 79%.
- Turbo and the laptop's power limits aren't controlled. The GPU numbers held steady; the CPU numbers may drift under sustained load (a Week 6 topic).

## Next steps (Week 1)

- [ ] Rerun `cpu_stream.py` on a quiet system and replace the CPU row in **Headline numbers**.
- [ ] Record both bandwidths in the Context doc: tick the Open facts (CPU, OS, RAM, MX450 variant) and fill in the Week 1 status.
- [ ] Fix the MX450 entry in `specs.md` (56 GB/s, 7 Gbps per pin, 64-bit bus).
- [x] Feed bandwidth and matmul TFLOPS into `ceiling.py`'s device presets (version 0.2).
- [ ] Rerun `matmul_flops.py` plugged in and update the TFLOPS in `ceiling.py`'s presets.
- [ ] First roofline check on Stride. Its model, found in `target-model/target_engine.py`, is Qwen2.5-1.5B-Instruct with bitsandbytes NF4 (double quant) and FP16 compute.
  - Decode: hand-calculated ceiling ≈ 22.3 ms/token (1148 MB read per token ÷ 51.6 GB/s), compared with the measured 83.1 ms/token, which is about 27% of the ceiling.
  - Run `ceiling.py` to confirm this number.
- [ ] Explain the ~800 ms TTFT floor from an nsys trace. The FP16 matmul rate (0.35 TFLOPS) is one candidate to check.
