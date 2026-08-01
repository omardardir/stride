"""
Draft inference engine — Qwen2-0.5B-Instruct  (CPU, ggml/llama.cpp)
====================================================================

Role in speculative decoding
-----------------------------
This engine is the *draft* model.  It runs entirely on the host CPU and
proposes γ candidate tokens per speculative step.  The GPU target model
then verifies those tokens in a single batched forward pass.

Why llama.cpp / ggml instead of PyTorch?
-----------------------------------------
- llama.cpp ships hand-optimised SIMD kernels (AVX2/AVX-512/NEON) for
  every common quantisation format — no need to write them ourselves.
- The Q4_K_M GGUF checkpoint for Qwen2-0.5B is ~350 MB, loads in < 2 s
  on a laptop SSD, and runs at > 100 tok/s on the i7-11xx CPU.
- The Python binding (llama-cpp-python) exposes the full low-level API,
  including per-step logits access and manual KV-cache control.

Requirements
-------------
    pip install llama-cpp-python           # CPU build — no CUDA flags needed
    # Download a Q4_K_M GGUF, e.g. from HuggingFace:
    # huggingface-cli download \
    #     Qwen/Qwen2-0.5B-Instruct-GGUF \
    #     qwen2-0_5b-instruct-q4_k_m.gguf \
    #     --local-dir ./draft-model/weights/

GGUF path
----------
Set the DRAFT_MODEL_PATH constant below, or pass --model on the CLI.

Naive KV cache
--------------
llama.cpp manages the KV cache internally in a fixed ring-buffer allocated
at load time (n_ctx tokens).  Our "naive" layer on top simply:
  1. Calls llama_kv_cache_clear() to wipe the cache for a new sequence.
  2. Tracks the current sequence length in Python so callers can read it.
  3. Exposes a rollback() method that calls llama_kv_cache_seq_rm() to
     trim back to a checkpoint — essential for the speculative loop when
     draft tokens are rejected.

This is intentionally simple.  No paging, no block allocation, no
quantised cache entries.  That complexity lives in Week 2.
"""

from __future__ import annotations

import sys
import os
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional

# Add the project root to sys.path so we can import profiler.py from the
# parent directory regardless of where the script is invoked from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from profiler import Profiler

# ---------------------------------------------------------------------------
# Profiler — one shared instance for the draft engine
# ---------------------------------------------------------------------------
profiler = Profiler()  # enabled=True by default

# llama-cpp-python: pip install llama-cpp-python
try:
    from llama_cpp import Llama, llama_token_eos
except ImportError:
    sys.exit(
        "[ERROR] llama-cpp-python is not installed.\n"
        "        Run: pip install llama-cpp-python"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DRAFT_MODEL_PATH = "./draft-model/weights/qwen2-0_5b-instruct-q4_k_m.gguf"

# Context window allocated inside llama.cpp's ring-buffer KV cache.
# 2048 is enough for speculative decoding drafts; raise if you need longer
# conversations.
N_CTX = 2048

# Number of CPU threads for ggml matrix multiplications.
# Match to physical core count — the i7-11xx has 4 P-cores.
N_THREADS = 4

# Speculation window: how many tokens the draft model proposes per step.
# γ = 4–6 is the typical sweet spot for small draft models.
GAMMA = 4

# Greedy decoding (temperature=0) for the draft — matches the simplest
# speculative-decoding formulation where acceptance is deterministic.
TEMPERATURE = 0.0
TOP_P = 1.0  # disabled when temperature == 0


# ---------------------------------------------------------------------------
# Naive KV cache manager
# ---------------------------------------------------------------------------

class NaiveDraftKVCache:
    """
    Thin Python wrapper around llama.cpp's internal KV cache.

    llama.cpp allocates a ring-buffer of n_ctx KV slots at model-load time.
    We don't allocate anything ourselves — we just track the sequence length
    and call the llama.cpp C-API (via llama_cpp-python) to wipe or trim it.

    Interface mirrors the target model's NaiveKVCache so the speculative
    orchestrator can treat both caches uniformly.

    Methods
    -------
    clear()
        Wipe the entire cache.  Call this before processing a new prompt.
    checkpoint() -> int
        Return the current sequence length as a rollback point.
    rollback(pos: int)
        Trim the cache back to `pos` tokens.  Used when draft tokens are
        rejected by the target model.
    seq_len() -> int
        Number of tokens currently in the cache.
    """

    def __init__(self, llm: Llama) -> None:
        self._llm = llm
        self._seq_len: int = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Wipe the entire KV cache for a fresh sequence."""
        self._llm._ctx.kv_cache_clear()
        self._seq_len = 0

    def checkpoint(self) -> int:
        """Return a rollback point (current sequence length)."""
        return self._seq_len

    def rollback(self, pos: int) -> None:
        """
        Trim the KV cache back to `pos` tokens.

        Internally calls llama_kv_cache_seq_rm(ctx, seq_id=0, p0=pos, p1=-1)
        which removes all KV entries with position >= pos for sequence 0.
        """
        if pos < 0 or pos > self._seq_len:
            raise ValueError(
                f"rollback pos={pos} is out of range [0, {self._seq_len}]"
            )
        # seq_rm(seq_id, p0, p1): remove positions [p0, p1).  p1=-1 means end.
        self._llm._ctx.kv_cache_seq_rm(0, pos, -1)
        self._seq_len = pos

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    def seq_len(self) -> int:
        """Number of tokens currently stored in the KV cache."""
        return self._seq_len

    def is_empty(self) -> bool:
        return self._seq_len == 0

    # ------------------------------------------------------------------
    # Internal: called by DraftEngine after each forward pass
    # ------------------------------------------------------------------

    def _advance(self, n: int = 1) -> None:
        """Record that n new tokens were appended to the cache."""
        self._seq_len += n


# ---------------------------------------------------------------------------
# Draft engine
# ---------------------------------------------------------------------------

@dataclass
class DraftResult:
    """Output of a single speculative drafting step."""
    token_ids: list[int]          # proposed token ids (len <= gamma)
    token_logprobs: list[float]   # log-prob of each proposed token
    wall_ms: float                # total wall-clock time for this draft step


class DraftEngine:
    """
    Minimal CPU draft engine for Qwen2-0.5B-Instruct (GGUF / llama.cpp).

    Usage (standalone)
    ------------------
        engine = DraftEngine.from_path("path/to/model.gguf")
        engine.load_prompt("Hello, my name is")
        result = engine.draft(gamma=4)
        print(result.token_ids)

    Usage (speculative loop integration)
    --------------------------------------
    The orchestrator calls:
        1. engine.load_prompt(text)      -- prefill once per new prompt
        2. result = engine.draft(gamma)  -- propose γ tokens
        3. engine.rollback(accept_count) -- trim rejected tokens from KV
        4. engine.extend(accepted_ids)   -- feed back verified tokens
        5. goto 2

    KV cache lifetime
    ------------------
    The cache is managed by `self.kv` (NaiveDraftKVCache).  Call
    `self.kv.clear()` to reset between independent requests.
    """

    def __init__(self, llm: Llama) -> None:
        self._llm = llm
        self.kv = NaiveDraftKVCache(llm)
        self._last_token_id: Optional[int] = None  # last token in the seq

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_path(
        cls,
        model_path: str,
        n_ctx: int = N_CTX,
        n_threads: int = N_THREADS,
        verbose: bool = False,
    ) -> "DraftEngine":
        """Load a GGUF model and return a ready-to-use DraftEngine."""
        print(f"[draft] Loading {model_path} ...", flush=True)
        t0 = time.perf_counter()
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,      # CPU-only — GPU stays free for target model
            logits_all=False,    # we only need the last logit per step
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t0
        print(f"[draft] Model loaded in {elapsed:.2f}s  |  n_ctx={n_ctx}  "
              f"n_threads={n_threads}", flush=True)
        return cls(llm)

    # ------------------------------------------------------------------
    # Prompt ingestion
    # ------------------------------------------------------------------

    def load_prompt(self, text: str) -> None:
        """
        Tokenise `text` and run a prefill forward pass to populate the KV cache.

        This must be called once before the first draft() call for any new
        conversation.  For a continuation (same conversation, new user turn)
        call extend() with the new tokens instead.
        """
        self.kv.clear()

        token_ids: list[int] = self._llm.tokenize(
            text.encode("utf-8"), add_bos=False, special=True
        )

        if not token_ids:
            raise ValueError("Prompt produced zero tokens.")

        # Run the prefill: eval all prompt tokens in one shot.
        # llama.cpp fills the KV cache internally for positions [0, len).
        with profiler.time("draft", "prefill_ms"):
            self._llm.eval(token_ids)
        self.kv._advance(len(token_ids))
        self._last_token_id = token_ids[-1]
        profiler.gauge("draft", "kv_seq_len", self.kv.seq_len())

    # ------------------------------------------------------------------
    # Core: draft γ tokens
    # ------------------------------------------------------------------

    def draft(self, gamma: int = GAMMA) -> DraftResult:
        """
        Propose up to `gamma` tokens autoregressively using greedy decoding.

        Each step:
          1. Read logits for the current last token.
          2. Argmax → next token id.
          3. Eval that token (appends to KV cache).
          4. Repeat until gamma tokens or EOS.

        Returns a DraftResult with the proposed token ids and their log-probs.
        """
        if self._last_token_id is None:
            raise RuntimeError("Call load_prompt() or extend() before draft().")

        token_ids: list[int] = []
        token_logprobs: list[float] = []
        eos_id: int = self._llm.token_eos()

        with profiler.time("draft", "draft_ms"):
            for _ in range(gamma):
                with profiler.time("draft", "sample_ms"):
                    logits = self._get_logits()    # list[float] of length vocab_size
                    next_id = int(self._argmax(logits))

                    # Log-prob of the chosen token (for acceptance ratio in stochastic SD)
                    import math
                    max_logit = max(logits)
                    log_z = math.log(sum(math.exp(l - max_logit) for l in logits)) + max_logit
                    log_prob = logits[next_id] - log_z

                token_ids.append(next_id)
                token_logprobs.append(log_prob)
                profiler.count("draft", "tokens_generated", 1)

                if next_id == eos_id:
                    # Don't eval past EOS — leave cache at this position
                    self._last_token_id = next_id
                    profiler.count("draft", "eos_hits", 1)
                    break

                # Eval the chosen token so llama.cpp appends it to the KV cache
                with profiler.time("draft", "eval_ms"):
                    self._llm.eval([next_id])
                self.kv._advance(1)
                self._last_token_id = next_id
                profiler.gauge("draft", "kv_seq_len", self.kv.seq_len())

        profiler.count("draft", "draft_steps", 1)
        wall_ms = profiler._timings["draft"]["draft_ms"].samples[-1]  # last recorded draft_ms
        return DraftResult(
            token_ids=token_ids,
            token_logprobs=token_logprobs,
            wall_ms=wall_ms,
        )

    # ------------------------------------------------------------------
    # KV cache rollback / extension
    # ------------------------------------------------------------------

    def rollback(self, accept_count: int) -> None:
        """
        Roll the KV cache back so only `accept_count` draft tokens remain.

        Call this after the target model rejects tokens beyond position
        `accept_count` in the last draft batch.

        Example: draft returned [A, B, C, D], target accepted [A, B] only.
            engine.rollback(accept_count=2)
        The cache is trimmed back to (prompt_len + prior_accepted + 2).
        """
        # Current seq_len includes all gamma tokens we eval-ed in draft().
        # We want to keep only accept_count of them.
        n_to_remove = self.kv.seq_len() - (
            self._checkpoint_before_draft + accept_count
        )
        target_pos = self.kv.seq_len() - n_to_remove
        self.kv.rollback(target_pos)

        # Reset _last_token_id to the last accepted token.
        # The caller must pass the accepted token id explicitly via extend().

    def mark_draft_start(self) -> None:
        """
        Snapshot the KV length before calling draft().

        Must be called immediately before draft() so rollback() knows how far
        back to trim.  In the speculative loop:

            engine.mark_draft_start()
            result = engine.draft(gamma)
            # ... target model verifies ...
            engine.rollback(accept_count)
            engine.extend([last_accepted_id])
        """
        self._checkpoint_before_draft: int = self.kv.seq_len()

    def extend(self, token_ids: list[int]) -> None:
        """
        Feed a list of already-verified token ids into the KV cache.

        Used after rollback() to re-anchor the cache at the last accepted
        token without re-sampling it.  Typically called with a single token
        (the one sampled by the target after rejection).
        """
        if not token_ids:
            return
        self._llm.eval(token_ids)
        self.kv._advance(len(token_ids))
        self._last_token_id = token_ids[-1]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_logits(self) -> list[float]:
        """Read the logits for the last decoded token from the C context.

        llama-cpp-python's `self._llm.scores` array is only populated when
        `logits_all=True`.  With `logits_all=False` (our default) the
        high-level scores array is never written to — the save-logits
        codepath is a `pass`.  So we go through the C API directly:

            llama_get_logits_ith(ctx, -1)

        returns a ctypes pointer to `n_vocab` floats for the last token in
        the batch.  This works regardless of `logits_all`.
        """
        import ctypes
        n_vocab = self._llm._n_vocab
        logits_ptr = self._llm._ctx.get_logits_ith(-1)
        # Build a Python list from the ctypes float array.
        return list(logits_ptr[:n_vocab])

    @staticmethod
    def _argmax(logits: list[float]) -> int:
        """Pure-Python argmax — avoids numpy/torch dependency on this hot path."""
        best_idx = 0
        best_val = logits[0]
        for i, v in enumerate(logits):
            if v > best_val:
                best_val = v
                best_idx = i
        return best_idx

    # ------------------------------------------------------------------
    # Convenience: standalone greedy generation (for testing / REPL)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        gamma: int = GAMMA,
        verbose: bool = True,
    ) -> str:
        """
        Simple greedy autoregressive generation loop (no speculative decoding).

        Useful for verifying that the model is loaded correctly and produces
        sensible output before wiring it into the full speculative loop.
        """
        self.load_prompt(prompt)
        eos_id = self._llm.token_eos()
        generated_ids: list[int] = []

        with profiler.time_request("draft"):
            for _ in range(max_new_tokens):
                next_id = int(self._argmax(self._get_logits()))
                if next_id == eos_id:
                    profiler.count("draft", "eos_hits", 1)
                    break
                generated_ids.append(next_id)
                with profiler.time("draft", "eval_ms"):
                    self._llm.eval([next_id])
                self.kv._advance(1)
                self._last_token_id = next_id
                profiler.count("draft", "tokens_generated", 1)
                profiler.gauge("draft", "kv_seq_len", self.kv.seq_len())

        e2e_ms = profiler._timings["draft"]["e2e_latency_ms"].samples[-1]
        elapsed = e2e_ms / 1000.0
        tok_per_sec = len(generated_ids) / elapsed if elapsed > 0 else 0.0

        text = self._llm.detokenize(generated_ids).decode("utf-8", errors="replace")

        if verbose:
            print(
                f"\n[draft] {len(generated_ids)} tokens in {elapsed:.2f}s  "
                f"({tok_per_sec:.1f} tok/s)",
                flush=True,
            )

        profiler.report()
        return text


# ---------------------------------------------------------------------------
# Interactive REPL  (python inference.py --model <path>)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen2-0.5B draft engine (CPU, llama.cpp)"
    )
    parser.add_argument(
        "--model", default=DRAFT_MODEL_PATH,
        help="Path to the Q4_K_M GGUF file."
    )
    parser.add_argument(
        "--n-ctx", type=int, default=N_CTX,
        help=f"KV cache context length (default: {N_CTX})."
    )
    parser.add_argument(
        "--n-threads", type=int, default=N_THREADS,
        help=f"CPU threads for ggml (default: {N_THREADS})."
    )
    parser.add_argument(
        "--gamma", type=int, default=GAMMA,
        help=f"Draft tokens per speculative step (default: {GAMMA})."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Max new tokens in REPL mode (default: 256)."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show llama.cpp internal logs."
    )
    args = parser.parse_args()

    engine = DraftEngine.from_path(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        verbose=args.verbose,
    )

    print("=" * 60)
    print("  Qwen2-0.5B-Instruct  |  GGUF Q4_K_M  |  CPU draft engine")
    print(f"  gamma={args.gamma}  |  n_ctx={args.n_ctx}  |  threads={args.n_threads}")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("[INFO] Goodbye.")
            break

        # Build a minimal chat prompt in Qwen2 ChatML format
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        print("\nAssistant: ", end="", flush=True)
        response = engine.generate(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            gamma=args.gamma,
            verbose=True,
        )
        print(response)


if __name__ == "__main__":
    main()
