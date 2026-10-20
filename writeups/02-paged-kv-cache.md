# Paged KV Cache on a 2 GB GPU: what paging buys a local user, and what it costs

give an intro to kv cache.
## 1. Problem — how dynamic & static caches waste memory


### How HF DynamicCache grows — what I found in the source


## 2. Hardware context — MX450 2 GB, model ~1.1 GB, what's left for KV

<!-- S3 -->

### Bytes/token math, tokens that fit, why this matters on small GPUs

## 3. Design — pages, block tables, allocator, pool layout

<!-- S3–S8 -->

### Diagram: logical tokens → block table → physical pages

### Decisions + why: page size, pool layout, pool sizing, gather-on-read

## 4. Correctness — how I proved it's right

<!-- S8, S9 -->

## 5. Results

### 5a. Local scenarios (E1) — single chat, parallel sampling, spec-decode

<!-- S6 -->

### 5b. Parity (E2)

<!-- S9 -->

### 5c. Memory & latency on the real GPU (E3, E4): dynamic vs static vs paged

<!-- S11, S12 -->

## 6. Analysis — the honest part

<!-- S12 -->

For ONE local user, is paging worth it over a static cache? Answer: "mostly X, except when Y, Z"

## 7. Beyond the server: paging for multi-app local inference

<!-- S7, S13 -->

Many apps, one small GPU: idle-session swapping, foreground priority, shared prompts.

### Simulation (E5)

### Measured swap-vs-recompute cost

## 8. What I learned / surprises

<!-- S13 — pulled from notes.md -->

## 9. Next — fused kernel vs. real swap-to-RAM: which first?

<!-- S13 -->
