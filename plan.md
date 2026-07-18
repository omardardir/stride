# Week 1 — Baseline engines + profiling + Triton fundamentals

Minimal PyTorch inference engine for the GPU target model (forward pass, naive KV cache, sampling). Minimal CPU draft engine using ggml/llama.cpp bindings — don't hand-roll SIMD/AVX, that's a whole separate low-level skill and it's already solved well. Build a microsecond-precision profiling harness (torch.cuda.Event for GPU, high-res perf_counter for CPU) that separates prefill from decode. Spend a real chunk of the week on the official Triton tutorials (vector-add → softmax → fused attention). Do a roofline calc for your exact card. Deliverable: two working baselines, a profiling harness you trust, and enough Triton to not be starting cold next week.

# Week 2 — Quantized, paged KV cache: design first, kernel second

Implement block-based allocation and 4-bit block-wise quantization in plain PyTorch. Validate correctness, then measure the real accuracy-vs-capacity tradeoff (perplexity vs. context length — a nice callback to your sparse-attention eval). Only then write the Triton kernel: block-wise dequant in shared memory fused into the decode step, built by modifying a known fused-attention tutorial kernel rather than starting from a blank file. Deliverable: measured PyTorch-vs-kernel latency, and an honest answer — not an assumption — about whether it actually crosses into compute-bound territory on your hardware.

# Week 3 — Heterogeneous CPU-draft / GPU-target pipeline

Wire the CPU draft engine to the GPU target engine. Measure round-trip sync latency and CPU throughput directly instead of assuming PCIe bandwidth is the issue. Add async overlap — CPU drafts the next chunk while GPU verifies the current one; a background thread and a queue is sufficient, no need to hand-engineer a custom threading model. Deliverable: a working end-to-end speculative loop at fixed γ, benchmarked against vanilla decode, with a real latency breakdown by component.

# Week 4 — Adaptive γ, benchmarks, write-up

A controller that adjusts speculation window γ from measured acceptance rate α and the latency numbers from week 3 — simple rule-based or PID-style is enough, this is not the place for a full dynamic layer-router. Full benchmark suite: vanilla vs. static-γ vs. adaptive-γ, across a few context lengths, reported honestly including any regime where it doesn't help. Write-up using the problem → hardware-specific insight → design → honest measured results shape from the three references