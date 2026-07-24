"""
Inference engine for Qwen2.5-1.5B-Instruct
Quantization: 4-bit NF4 (BitsAndBytes) -- fits comfortably in 2GB VRAM
  - Model weights:  ~0.9 GB
  - KV cache + activations: ~0.5-0.7 GB
  - Total:          ~1.5-1.8 GB

Requirements:
    pip install transformers bitsandbytes accelerate
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID       = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.0
TOP_P          = 0.9
DEVICE         = "cuda"
REPETITION_PENALTY = 1.0  # matches HuggingFace GenerationConfig default

# 4-bit NF4 quantization config -- best quality-per-bit for 2 GB VRAM
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 has better accuracy than fp4
    bnb_4bit_compute_dtype=torch.float16, # compute in fp16 for speed
    bnb_4bit_use_double_quant=True,      # extra ~0.1 bpw savings via nested quant
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    """Load tokenizer and 4-bit quantized model onto GPU."""
    if not torch.cuda.is_available():
        sys.exit("[ERROR] No CUDA GPU detected. This engine is GPU-only.")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {vram_gb:.1f} GB")
    print(f"[INFO] Loading {MODEL_ID} with 4-bit NF4 quantization ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=QUANT_CONFIG,
        device_map=DEVICE,           # force everything onto GPU 0
        trust_remote_code=True,
    )
    model.eval()

    used_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[INFO] Model loaded -- VRAM in use: {used_gb:.2f} GB\n")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Naive KV Cache Manager
# ---------------------------------------------------------------------------
class NaiveKVCache:
    """
    Naive KV cache that wraps the past_key_values tuple returned by the model.

    This is intentionally simple so the generation loop is easy to follow:
      - On the first forward pass (prefill) we store the full KV cache.
      - On every subsequent decode step we append the new single-token KV
        entries returned by the model to the stored cache.
      - We expose the cache as `past_key_values`, which is the exact format
        the HuggingFace model expects as input.

    Later this class can be replaced with a PagedAttention / chunked-prefill
    implementation without touching the generation loop.
    """

    def __init__(self):
        # past_key_values is a tuple of (key, value) pairs, one per layer.
        # Each key/value tensor has shape [batch, heads, seq_len, head_dim].
        self.past_key_values = None

    def update(self, new_past_key_values):
        """Replace the cache with the freshly returned past_key_values.

        Because HuggingFace models return the *full* past_key_values (old +
        new token appended) when use_cache=True, we simply overwrite.
        """
        self.past_key_values = new_past_key_values

    def is_empty(self) -> bool:
        return self.past_key_values is None

    def seq_len(self) -> int:
        """Number of tokens currently cached.

        Handles both formats returned by different transformers versions:
          - Legacy tuple-of-tuples: past_key_values[layer][0].shape[2]
          - DynamicCache (transformers >= 4.38): exposes get_seq_length()
        """
        if self.is_empty():
            return 0
        pkv = self.past_key_values
        # DynamicCache (newer transformers) -- preferred path
        if hasattr(pkv, "get_seq_length"):
            return pkv.get_seq_length()
        # Legacy tuple format: ((key, value), ...) one pair per layer
        # key shape: [batch, heads, seq_len, head_dim]
        return pkv[0][0].shape[2]


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out logits below the nucleus threshold."""
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative prob above top_p
    sorted_indices_to_remove = cumulative_probs - sorted_logits.softmax(dim=-1) > top_p
    sorted_logits[sorted_indices_to_remove] = float("-inf")

    # Scatter back to original indexing
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


def _apply_repetition_penalty(
    logits: torch.Tensor,          # [vocab_size]
    generated_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty (same formula as HuggingFace RepetitionPenaltyLogitsProcessor).

    For every token that has already been generated:
      score /= penalty   if score > 0
      score *= penalty   if score < 0

    A penalty of 1.0 (HuggingFace default) is a no-op.
    """
    if penalty == 1.0 or not generated_ids:
        return logits

    # Clone to escape inference-mode read-only restriction before in-place update
    logits = logits.clone()

    # Gather scores for previously generated tokens
    prev = torch.tensor(generated_ids, dtype=torch.long, device=logits.device)
    scores = logits[prev]
    # Divide positive scores, multiply negative scores -- matches HF exactly
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    logits[prev] = scores
    return logits


def _sample_next_token(
    logits: torch.Tensor,        # shape [1, vocab_size]
    temperature: float,
    top_p: float,
    generated_ids: list[int],
    repetition_penalty: float,
) -> int:
    """Apply repetition penalty + temperature + top-p then sample one token id.

    When temperature == 0 we do true greedy decoding (argmax), which is
    identical to what HuggingFace model.generate(do_sample=False) does
    internally.  Dividing by a tiny epsilon and then calling multinomial
    is NOT equivalent -- floating-point precision in the CUDA softmax
    kernel can cause a different token to win, producing divergent output.
    """
    logits = logits[0]                           # [vocab_size]

    # Repetition penalty (applied before temperature, same as HuggingFace)
    logits = _apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Greedy path: identical to do_sample=False in model.generate()
    if temperature == 0.0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    logits = _top_p_filter(logits, top_p)
    probs  = torch.softmax(logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1).item()
    return token_id


# ---------------------------------------------------------------------------
# Custom generation loop
# ---------------------------------------------------------------------------
def _run_generation(
    model,
    input_ids: torch.Tensor,          # [1, prompt_len]
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
    repetition_penalty: float,  # HuggingFace GenerationConfig default
) -> list[int]:
    """
    Token-by-token generation with a naive KV cache.

    Step 1 – Prefill:  feed the whole prompt in one forward pass, get KV cache.
    Step 2 – Decode:   feed one new token per step, reuse the KV cache.

    position_ids are computed explicitly so the model always knows the absolute
    position of each token, even when feeding a single decode token against a
    non-empty KV cache.

    Returns the list of newly generated token ids (not including the prompt).
    """
    kv_cache = NaiveKVCache()
    generated_ids: list[int] = []
    prompt_len = input_ids.shape[1]

    # ---- Prefill --------------------------------------------------------
    # position_ids: [0, 1, 2, ..., prompt_len-1]
    prefill_position_ids = torch.arange(
        prompt_len, dtype=torch.long, device=DEVICE
    ).unsqueeze(0)  # [1, prompt_len]

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            position_ids=prefill_position_ids,
            past_key_values=None,       # no cache yet
            use_cache=True,
            return_dict=True,
        )

    # outputs.logits: [1, prompt_len, vocab_size]
    # We only need the logits for the last prompt token to sample the first
    # generated token.
    next_token_id = _sample_next_token(
        outputs.logits[:, -1, :],
        temperature,
        top_p,
        generated_ids,
        repetition_penalty,
    )
    kv_cache.update(outputs.past_key_values)
    generated_ids.append(next_token_id)

    if next_token_id == eos_token_id:
        return generated_ids

    # ---- Decode loop ----------------------------------------------------
    for _ in range(max_new_tokens - 1):
        # next input is just the single newly generated token
        next_input = torch.tensor([[next_token_id]], dtype=torch.long, device=DEVICE)

        # position_ids for the new token = total tokens seen so far
        decode_position_ids = torch.tensor(
            [[kv_cache.seq_len()]], dtype=torch.long, device=DEVICE
        )  # [1, 1]

        with torch.inference_mode():
            outputs = model(
                input_ids=next_input,
                position_ids=decode_position_ids,
                past_key_values=kv_cache.past_key_values,
                use_cache=True,
                return_dict=True,
            )

        # outputs.logits: [1, 1, vocab_size]
        next_token_id = _sample_next_token(
            outputs.logits[:, -1, :],
            temperature,
            top_p,
            generated_ids,
            repetition_penalty,
        )
        kv_cache.update(outputs.past_key_values)
        generated_ids.append(next_token_id)

        if next_token_id == eos_token_id:
            break

    return generated_ids


# ---------------------------------------------------------------------------
# Public-facing generate function (streaming)
# ---------------------------------------------------------------------------
def generate(
    tokenizer,
    model,
    prompt: str,
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    repetition_penalty: float = REPETITION_PENALTY,  # matches HuggingFace GenerationConfig default
) -> None:
    """Run the custom generation loop and stream tokens to stdout.

    Args:
        repetition_penalty: Values > 1.0 discourage the model from repeating
            tokens it has already generated. 1.0 (HuggingFace default) is a
            no-op. See https://huggingface.co/docs/transformers/main_classes/text_generation
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]

    # Apply the model's built-in chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    # Run our custom loop
    generated_ids = _run_generation(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
    )

    # Stream-decode: print each token as it is decoded
    for token_id in generated_ids:
        token_str = tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(token_str, end="", flush=True)

    print()  # newline after the response


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------
def main():
    tokenizer, model = load_model()

    system_prompt = "You are a helpful assistant."
    print("=" * 60)
    print("  Qwen2.5-1.5B-Instruct  |  4-bit NF4  |  GPU-only")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'system: <text>' to change the system prompt.")
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
        if user_input.lower().startswith("system:"):
            system_prompt = user_input[len("system:"):].strip()
            print(f"[INFO] System prompt updated: {system_prompt!r}")
            continue

        print("\nAssistant: ", end="", flush=True)
        generate(tokenizer, model, prompt=user_input, system=system_prompt)


if __name__ == "__main__":
    main()
