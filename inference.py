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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID       = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.0
TOP_P          = 0.9
DEVICE         = "cuda"

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
# Inference (streaming)
# ---------------------------------------------------------------------------
def generate(
    tokenizer,
    model,
    prompt: str,
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> None:
    """Stream the assistant reply token-by-token to stdout."""
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

    # TextStreamer prints each token to stdout as it is decoded
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )


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
