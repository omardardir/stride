# Stride Engine — Agent Notes & Architecture Guide

## Overview & Current State

`stride` is a custom inference engine designed for running LLMs under tight hardware constraints (e.g., NVIDIA MX450 2GB VRAM + Intel i7 CPU).

The baseline inference pipeline in [`inference.py`](file:///C:/Users/omara/Desktop/projects/stride/inference.py) has been updated from HuggingFace's built-in `model.generate()` to a hand-rolled, explicit prefill-and-decode engine.

---

## Current Implementation Details (`inference.py` Diff Summary)

### 1. Hand-Rolled Generation Loop (`_run_generation`)
- **Reason**: `model.generate()` hides internal state, KV cache handling, and sampling logic. A custom generation loop exposes exact token-by-token control needed for custom KV cache management and speculative decoding.
- **Prefill Phase**: Passes the full tokenized prompt (`[1, prompt_len]`) with `past_key_values=None` to compute initial KV cache in a single parallel forward pass.
- **Decode Phase**: Loops token-by-token (`[1, 1]` tensor inputs) feeding `past_key_values=kv_cache.past_key_values` into `model()`.

### 2. Naive KV Cache Abstraction (`NaiveKVCache`)
- **Reason**: Acts as an interface seam between the model's output `past_key_values` and the generation loop.
- **Design**: Currently wraps the standard HuggingFace `past_key_values` tuple. In Week 2, this will be replaced with block-based paged memory allocation without requiring changes to the core generation loop.

### 3. Explicit Sampling Pipeline (`_sample_next_token`, `_top_p_filter`)
- **Reason**: Replaces opaque generator sampling with explicit logit transformation:
  1. Temperature scaling: `logits / temperature`
  2. Nucleus filtering (`top_p` cumulative probability cutoff)
  3. Softmax & Multinomial sampling: `torch.multinomial(probs, num_samples=1)`

---

## Architectural Decision: Stateless vs. Stateful KV Cache

- **Current Behavior (Stateless KV Cache)**:
  - `_run_generation` instantiates a new `NaiveKVCache` on every call to `generate()`.
  - For multi-turn chats, the entire history is formatted into a single prompt string via `apply_chat_template` and re-prefilled in one parallel forward pass.
  - **Tradeoff**: Very simple and clean; GPU prefill across a few hundred tokens is parallel and fast.
- **Future Extension (Stateful Session Caching)**:
  - To preserve KV cache across user turns without re-prefilling past context, `kv_cache` can be persisted outside `_run_generation`.
  - On follow-up messages, only the incremental new tokens (user prompt + chat template tags) are passed into `model(input_ids=new_tokens, past_key_values=active_kv_cache)`.

---

## Hardware Context & Target Specs

- **GPU**: NVIDIA GeForce MX450 (Turing TU117, 2GB VRAM, Compute Capability 7.5, No Tensor Cores).
- **Target Model**: `Qwen/Qwen2.5-1.5B-Instruct` quantized to 4-bit NF4 via `bitsandbytes` (~0.9 GB weights).
- **CPU Draft Model (Target for Speculative Decoding)**: `Qwen2.5-0.5B` running on Intel Core i7 CPU.
