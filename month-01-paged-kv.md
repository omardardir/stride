# Month 01 — Paged KV Cache (October 2026)

> **One topic only:** a block-based (paged) KV cache in plain PyTorch, plugged into `TargetEngine`, measured against the naive (dynamic) cache and a static cache, and explained in a technical write-up. The angle is **local inference**: when does paging actually help one person on one small GPU, and what could it do when many apps share that GPU?
>
> **Time budget:** ~4–5 h/week → 14 sessions of ~1.5 h + a short buffer (Oct 28–31).
> **How to use this file:** open it, find the first unchecked box, do that session. Don't skip ahead or reorder. Tick boxes as you go and commit this file with your work.

---

## 0. The rules (read once, then follow)

1. **One session = one numbered block below.** Each has a *Done when* line. When it's true, stop, even if you feel like doing more.
2. **Every session ends with a 3-line log** in `journal/paged-kv/log.md`:
   `date · what I did · what confused me / what's next`
3. **Stuck for more than 45 min?** Write the exact question in `journal/paged-kv/notes.md` under `## Open questions`, pick the simplest workaround, and move on. Stuck questions often make good write-up material.
4. **Parking lot:** any idea not in this month's scope goes in section 6 of this file, **not into code**. These are the ones that will tempt you:
   actually building swap-to-RAM, 4-bit KV quantization, a Triton paged-attention kernel, real prefix sharing, batching, the CPU draft model.
5. **Write predictions before measuring.** Every experiment has a *Predict* line. Fill it in *before* running. Being wrong is fine and makes for a more interesting write-up. So is "no difference."
6. **If a week slips,** use the buffer. If the buffer is used up too, cut the ★ stretch items first, then scenario L3, and never the write-up.

---

## 1. Scope

**Three caches you'll compare**

| Cache | What it is | Its weakness |
|---|---|---|
| **Dynamic** (naive, current) | HF `DynamicCache`, grows as tokens arrive | regrowth and allocator churn (you'll confirm how in S2) |
| **Static** | HF `StaticCache`, one contiguous buffer reserved for `max_len` upfront | reserves memory for tokens that may never come; one buffer per sequence |
| **Paged** (yours) | shared pool of fixed-size pages + block tables | page-table bookkeeping + a gather on every read (no custom kernel) |

The central question: **for local inference, when does paged beat *static*?** Static is the fair single-user competitor. Beating dynamic alone would credit paging with wins that really come from preallocating.

**In scope**
- `BlockAllocator`, per-sequence block tables, `PagedCache` (a `transformers` `Cache` subclass that writes into pages and gathers them for SDPA).
- `TargetEngine` flag `--kv dynamic | static | paged`.
- **Simulations** (CPU-only) of local scenarios and of your own multi-app scenario, including sharing and swapping *as simulated policies*.
- One small real measurement: how long moving KV pages GPU↔CPU takes vs. recomputing them (prefill).
- The write-up.

**Out of scope (candidates for later months)**
- Building swap-to-RAM, eviction or prefix sharing into the engine. This month they only exist in the simulator.
- Custom attention kernel (Triton). Attention reads K/V by **gathering pages into a contiguous tensor**, then calls SDPA. Measuring what that gather costs is part of the point.
- KV quantization, continuous batching, non-LLM models.

**Numbers to keep in mind** (Qwen2.5-1.5B, fp16 KV, MX450 2 GB):
- KV bytes per token = 28 layers × 2 (K,V) × 2 kv-heads × 128 dim × 2 B = **28,672 B ≈ 28 KB/token**
- One page of 16 tokens ≈ **448 KB**
- 500 MB of pool ≈ **~18k tokens**; a static 4k-token reservation ≈ **112 MB**
(Check these yourself in Session 3. Re-deriving them is part of the learning.)

---

## 2. The write-up (the end product)

File: `writeups/01-paged-kv-cache.md`. Create the skeleton in Session 0 and fill it **as you go**. The session that fills each section is listed next to it.

```
# Paged KV Cache on a 2 GB GPU: what paging buys a local user, and what it costs

1. Problem                      — how dynamic & static caches waste memory        (S1, S2)
   - how HF DynamicCache grows — what I found in the source
   - static reservation vs. actual usage
2. Hardware context             — MX450 2 GB, model ~1.1 GB, what's left for KV    (S3)
   - bytes/token math, tokens that fit, why this matters on small GPUs
3. Design                       — pages, block tables, allocator, pool layout     (S3–S8)
   - diagram: logical tokens → block table → physical pages
   - decisions + why: page size, pool layout, pool sizing, gather-on-read
4. Correctness                  — how I proved it's right                          (S8, S9)
5. Results
   5a. Local scenarios (E1)     — single chat, parallel sampling, spec-decode      (S6)
   5b. Parity (E2)                                                                 (S9)
   5c. Memory & latency on the real GPU (E3, E4): dynamic vs static vs paged       (S11, S12)
6. Analysis — the honest part   — for ONE local user, is paging worth it over a    (S12)
                                  static cache? Answer: "mostly X, except when Y, Z"
7. Beyond the server:           — many apps, one small GPU: idle-session swapping, (S7, S13)
   paging for multi-app            foreground priority, shared prompts.
   local inference                 Simulation (E5) + measured swap-vs-recompute cost
8. What I learned / surprises   — pulled from notes.md                             (S13)
9. Next                         — fused kernel vs. real swap-to-RAM: which first?  (S13)
```

---

## 3. Experiments (plan them now, run them later)

All simulations share one script and one cost model: 28 KB/token, page size 16, 500 MB pool, max context 4096. They use your real `BlockAllocator`. **Sharing and swapping are simulated policies, not engine features.**

| ID | Question | Setup | Metric | Session |
|----|----------|-------|--------|---------|
| **E1** | In realistic local scenarios, how much memory does static reserve vs. paged? | CPU simulation, 3 scenarios (below). Strategies: **static** (reserve `max_len` per sequence) vs **paged** (and **paged+shared** where it applies). | peak reserved MB, peak used MB, utilization % = used/reserved | S6 |
| **E2** | Is paged output identical to dynamic? | Greedy (temp=0), 5 prompts, 128 tokens | token-for-token match; max abs logit diff per step | S9 |
| **E3** | What does real GPU memory look like while generating? | 1 prompt, generate 1024 tokens; dynamic vs static vs paged | `max_memory_allocated`, `memory_reserved`, sampled every 64 steps | S11 |
| **E4** | What does each cache cost per decode step? | Context {128, 512, 1024, 2048 if it fits}; dynamic vs static vs paged-16 vs paged-64 | mean/p50 `decode_ms` per token (existing profiler) | S11 |
| **E5** | *(my own scenario)* Several apps share one GPU: what does paging + swapping buy? | CPU simulation of 3 apps (below), fed with **real** swap and prefill costs measured in S7 | see below | S7 |

**E1 scenarios** (local, single model):
- **L1 — one long chat:** 1 session, 20 turns; each turn adds a 50–300-token user message + a 100–500-token reply. Stops when it hits 4096.
- **L2 — parallel sampling:** one 500-token prompt, n=4 answers of 100–400 tokens each. Paged+shared stores the prompt's pages once (simulated refcount).
- **L3 — speculative decoding rollback:** generate 1000 tokens; each step appends γ=4 draft tokens, keeps `k ~ Binomial(4, α=0.7)` and rolls back the rest. Metric here: pages allocated/freed per step, and whether peak memory differs at all. *"No meaningful difference" is a valid, useful result.*

**E5 — multi-app scenario (yours):** one 500 MB pool, a 60-minute simulated timeline, 3 apps:
- **Chat assistant:** long sessions (up to 3k tokens), goes idle for minutes at a time
- **Code agent:** bursty, long prompts (~2k tokens), shares a 400-token tool-instructions prefix across its calls
- **Notes summarizer:** short (300–800 tokens), frequent, short-lived

Policies:
- **static:** each app reserves `max_len`, and a request that doesn't fit waits
- **paged + drop:** when the pool is full, evict the least-recently-used idle session and **recompute** it (re-prefill) on resume
- **paged + swap:** same eviction, but pages are **copied to CPU RAM** and copied back on resume
- ★ **+ priority:** the foreground app never gets evicted

Metrics: time requests spend waiting, total recompute ms vs. total swap ms (from the S7 measurements), and peak pool usage.

Results go in `bench/paged-kv/` (scripts + `.jsonl` + plots).

---

## 4. The checklist

### Week 1 (Sep 30 – Oct 6): Understand and design, no big code yet

#### Session 0 — Setup (30 min)
- [x] Commit current WIP (`kv_cache_manager.py`, `test_kv_cache_init.py`, engine diff) as "WIP: KV pool allocation"
- [x] Create `journal/paged-kv/notes.md` with headings: `## Concepts`, `## Transformers internals`, `## Design decisions`, `## Local use cases`, `## Open questions`, `## Surprises`
- [x] Create `journal/paged-kv/log.md` (empty)
- [x] Create `writeups/01-paged-kv-cache.md` with the skeleton from section 2
- [x] Create `bench/paged-kv/` folder

**Done when:** all four files and folders exist and are committed.

#### Session 1 — Read the source: vLLM paper
- [x] Read *Efficient Memory Management for LLM Serving with PagedAttention* (Kwon et al., 2023), **sections 1–4 only**
- [x] In `notes.md → Concepts`, answer in your own words:
  - [x] What are internal fragmentation, external fragmentation, and reservation waste? Give one example of each.
  - [x] What is a block table, and how is it like an OS page table?
  - [x] Why does paging need a custom attention kernel in vLLM?
  - [x] Which of their benefits need **many concurrent requests**, and which help even at batch size 1?
  - [x] What does paging give that a **static per-sequence buffer** can't? (list 3)
- [x] In `notes.md → Local use cases`: for each OS idea in the paper (paging, sharing, swapping, copy-on-write), write one sentence on what it would mean for **several apps on one laptop GPU**. This feeds E5 and write-up section 7.

**Done when:** all 5 questions and the OS-ideas list are answered, ≤ 5 sentences each.

#### Session 2 — Read the source: how transformers handles the cache
- [ ] Open `.venv/Lib/site-packages/transformers/cache_utils.py` (transformers **5.14.1**) and find:
  - [ ] the base `Cache` class and the signature of `update(key_states, value_states, layer_idx, cache_kwargs)`
  - [ ] how `DynamicCache` stores K/V and **how it grows** (torch.cat? preallocated?)
  - [ ] how `StaticCache` preallocates, and what arguments it needs (`max_cache_len`, …). You'll use it as a baseline.
  - [ ] which methods the model calls besides `update` (`get_seq_length`, `get_mask_sizes`, …?)
- [ ] Open `models/qwen2/modeling_qwen2.py` and find where attention calls `past_key_values.update(...)`. Write down the **tensor shape** of `key_states` there (expected `[batch, kv_heads, seq, head_dim]`)
- [ ] Write in `notes.md → Transformers internals`: "To plug in my cache I must implement: …" (a short list of methods)
- [ ] Draft write-up section 1 bullets: "How DynamicCache grows", "What StaticCache reserves"

**Done when:** you can name every method your `PagedCache` must implement.

#### Session 3 — Design on paper and fix the pool sizing
- [ ] Re-derive bytes/token, bytes/page, tokens-per-500MB and static-4k-MB by hand. Put them in write-up section 2.
- [ ] Decide and record in `notes.md → Design decisions` (one line of *why* each):
  - [ ] **Page size**: start with 16
  - [ ] **Pool layout**: the current layout `[pages, 2, layers, page, heads, dim]` vs per-layer `[layers, 2, pages, page, heads, dim]`. `update()` is called **per layer**, so which layout makes a per-layer gather touch less scattered memory? Pick one.
  - [ ] **Read path**: gather-on-read into a contiguous tensor, then SDPA (confirm this is the plan)
- [ ] **Fix the pool sizing trap:** `KVCacheManager.initialize()` currently takes 90% of free VRAM. On 2 GB that leaves nothing for prefill activations, and dynamic/static benchmarks will OOM or be unfair. Change it to take `max_tokens` (e.g. 4096) and size the pool from that. Allocate the pool **only** in paged mode.
- [ ] Draw the diagram (logical tokens → block table → pages) in the write-up. ASCII is fine.

**Done when:** the design-decisions section has 3 decisions with reasons, and the pool is sized by token budget.

---

### Week 2 (Oct 7 – Oct 13): Allocator, block tables, simulations (mostly CPU)

#### Session 4 — `BlockAllocator` (short session)
- [ ] In `target-model/kv_cache_manager.py` (or a new `block_allocator.py`), write a **pure-Python** `BlockAllocator(num_pages)` with no torch or GPU:
  - `allocate() -> int`, `free(page: int)`, `num_free`, raises a clear error when out of pages
- [ ] `tests/test_block_allocator.py`: allocate all, free one, re-allocate gets it back; double-free raises; out-of-pages raises

**Done when:** `pytest tests/test_block_allocator.py` passes on CPU.

#### Session 5 — Block tables and slot mapping
- [ ] Add a sequence-level structure (e.g. `SequenceBlockTable` or methods on the manager):
  - `append_tokens(seq_id, n) -> list[(page, offset)]` allocates new pages only when crossing a page boundary
  - `truncate(seq_id, new_len)` frees tail pages, which L3 rollback needs
  - `get_pages(seq_id) -> list[int]`, `free_sequence(seq_id)` returns all pages to the allocator
- [ ] Tests at the **boundaries**: 15, 16, 17, and 32 tokens with page size 16; appending 1 token at a time vs n at once gives the same mapping; truncate 33→15 frees 2 pages; freeing returns the right count
- [ ] Note in `notes.md` what "internal fragmentation" is for *your* design (the last partially filled page)

**Done when:** boundary tests pass.

#### Session 6 — E1: local-scenario simulation
- [ ] Write *Predict* lines for L1, L2, L3 in section 3 first
- [ ] `bench/paged-kv/sim.py`, CPU only, using your real allocator + block tables:
  - a small `Strategy` interface (static / paged / paged+shared) with `reserve`, `append`, `rollback`, `release`, and peak-MB tracking
  - scenarios L1, L2, L3 as plain functions that drive a strategy (fixed random seed)
- [ ] Output a table: scenario × strategy → peak reserved MB, peak used MB, utilization %
- [ ] Write write-up section 5a: prediction vs actual, and **why**

**Done when:** the E1 table is in the write-up.

#### Session 7 — E5: your multi-app scenario
- [ ] **Measure the two real costs** (GPU, small script `bench/paged-kv/measure_swap.py`):
  - [ ] swap cost: copy the KV of 512 / 1024 / 2048 tokens (use the 28 KB/token size) GPU → pinned CPU → GPU, timed with CUDA events, 10 repeats, report ms and GB/s
  - [ ] recompute cost: prefill ms for 512 / 1024 / 2048-token prompts (reuse the existing `run_benchmark.py` / profiler `prefill_ms`)
  - [ ] Write down: **at what context length is swapping cheaper than recomputing?** (There may be no crossover.)
- [ ] Write the *Predict* line for E5, then add the 3-app timeline + the 3 policies to `sim.py`, reusing the S6 strategies. Plug in the measured ms.
- [ ] Output: waiting time, recompute ms vs swap ms, peak pool usage per policy
- [ ] Draft write-up section 7: the scenario, the 3 use cases (idle swapping, foreground priority, shared prompts), the table, and one paragraph on what would change on a bigger GPU or a unified-memory laptop

**Done when:** the swap-vs-recompute numbers and the E5 table are in the write-up.

---

### Week 3 (Oct 14 – Oct 20): Plug into the model

#### Session 8 — `PagedCache` (tested without the model)
- [ ] Implement `PagedCache(Cache)` using the method list from Session 2:
  - `update(k, v, layer_idx, ...)`: compute slots for the new tokens (only once per step, not once per layer!), write K/V into the pool with advanced indexing, then **gather** this layer's pages up to `seq_len` and return contiguous `[1, kv_heads, seq, dim]` K and V
  - `get_seq_length()` and anything else from your list
- [ ] `tests/test_paged_cache.py` on the GPU with **random tensors, no model**: feed a prefill of 37 tokens, then 20 single-token updates, and compare returned K/V against a reference built with `torch.cat`. Use `torch.equal`; they should be exactly equal.

**Done when:** the roundtrip test passes for all layers.

#### Session 9 — Wire into `TargetEngine` + E2 correctness
- [ ] Add `kv_mode: "dynamic" | "static" | "paged"` to `TargetEngine` and `--kv` to `run_benchmark.py`
  - static: pass a `StaticCache(max_cache_len=4096, ...)` as `past_key_values`
  - paged: pass a `PagedCache`, and free the sequence's pages when the request ends
- [ ] **E2:** write *Predict*, then run greedy on 5 prompts × 128 tokens, dynamic vs paged (and static as a bonus). Tokens must match exactly. Log max abs logit diff.
- [ ] Record the result in write-up section 4

**Done when:** greedy outputs are identical, or the mismatch is understood and written down.

#### Session 10 — Buffer / debug session
Integration almost always breaks something (position ids, attention mask sizes, dtype, the layer-0 slot computation, `StaticCache` mask handling). This session is reserved for that.
- [ ] If Session 9 passed: add a test that runs **two requests back-to-back** and asserts the allocator has all pages free afterward (no leak)
- [ ] Write every bug you hit into `notes.md → Surprises`. These make good write-up material.
- [ ] ★ If there's time left: in the real engine, run two sequences alternately in one pool, free one mid-way, and show its pages get reused (allocator trace)

**Done when:** no page leaks across requests.

---

### Week 4 (Oct 21 – Oct 27): Measure and write

#### Session 11 — Run E3 + E4
- [ ] Write *Predict* lines for E3 and E4 in section 3 first. E4 hint: gather copies the whole context every step, so how should paged `decode_ms` scale with context length compared to static?
- [ ] `bench/paged-kv/run_memory.py` (E3) and `bench/paged-kv/run_latency.py` (E4): reuse `Profiler` / `decode_ms`, call `torch.cuda.reset_peak_memory_stats()` between runs, 2 warmup requests, and append results to `bench/paged-kv/results.jsonl`
- [ ] Run all three caches. Close other GPU apps first (WDDM: the display can steal VRAM).

**Done when:** `results.jsonl` has all E3/E4 rows for dynamic, static and paged.

#### Session 12 — Tables, plots, analysis
- [ ] Make 4 visuals: E1 utilization per scenario, E3 memory over steps, E4 `decode_ms` vs context length (dynamic vs static vs paged-16 vs paged-64), E5 swap vs recompute
- [ ] Write write-up sections 5c and 6. For each result: prediction → measured → **why**. Include the regime where paged is *worse*.
- [ ] Answer section 6 in one sentence you can defend: **"For a single local user on this GPU, paging over a static cache is worth it when ___, and not when ___."**

**Done when:** sections 5–6 are written, even if rough.

#### Session 13 — Finish the write-up
- [ ] Polish section 7 (multi-app) with the final E5 numbers
- [ ] Section 8: pick the 3–5 best entries from `notes.md → Surprises / Open questions`
- [ ] Section 9: two candidate next steps with your own numbers:
  - fused paged-attention kernel: "the gather costs X µs/step at 2k context"
  - real swap-to-RAM: "restoring a 2k-token session costs Y ms vs Z ms to recompute"
  Say which you'd do first and why.
- [ ] Re-read everything once, top to bottom. Cut anything that doesn't support the story.
- [ ] Update `README.md` with a link to the write-up
- [ ] Commit, then `git tag month-01-paged-kv`

**Done when:** the tag exists.

---

### Buffer (Oct 28 – Oct 31)
- [ ] Finish anything that slipped (write-up first, then experiments)
- [ ] If everything is done: pick a ★ item, or rest. Rest is a valid choice.
- [ ] Fill section 5 below and pick Month 02's single topic from the parking lot

---

## 5. Month retro (fill on Oct 31)

- Sessions planned: 14 · Sessions done: __
- Hours actually spent: __
- What took longer than expected:
- What I'd cut next time:
- Month 02 topic:

---

## 6. Parking lot (ideas go here, not into code)

- Triton paged-attention decode kernel (reads pages directly, no gather). **Month 02 candidate.**
- Real swap-to-RAM for idle sessions in the engine, using the E5 policy. **Month 02 candidate.**
- Foreground/background priority eviction in the real engine
- Real prefix sharing with ref-counted pages / copy-on-write (shared system prompts across apps)
- Persisting a session's KV pages to disk, so the cache survives an app restart
- 4-bit block-wise KV quantization (original plan, week 2)
- CPU-draft / GPU-target speculative pipeline (original plan, week 3). The L3 rollback results feed into this.
-
