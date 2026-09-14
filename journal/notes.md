# Notes


## 06-08-2026
Goal: analyze baseline benchmark results for the target-model
Finding:
- TTFT starts of by being bound by the base overhead of launching CUDA kernels and moving model weights through the memory.
    So for a range of small input tokens, the TTFT stays the same, and as it increases above a certain threshold then it becomes input-token bound & TTFT starts to increase beyond the baseline. Baseline TTFT: 500 ms
- torch.tensor(generated_ids) allocation — this line rebuilds a fresh CPU tensor from the entire Python list and copies it 
    to GPU every single decode step, and the list grows each step.

## 23-08-2026
Goal: Document GPU kernel execution details for the target model.

### GPU Kernels and Execution (Hugging Face `_model`)
- The Hugging Face `transformers` source code (specifically `Qwen2ForCausalLM`) is written in Python and does not contain raw C++/CUDA kernel code.
- However, it delegates all tensor operations to libraries that run compiled GPU kernels:
  - **PyTorch / SDPA**: By setting `attn_implementation="sdpa"`, the model uses PyTorch's native `torch.nn.functional.scaled_dot_product_attention`, executing optimized C++/CUDA kernels for FlashAttention or memory-efficient attention.
  - **Quantization (`bitsandbytes`)**: With 4-bit NF4 quantization enabled via `BitsAndBytesConfig`, linear layers are wrapped to run custom CUDA kernels from the `bitsandbytes` library for on-the-fly weight dequantization and GPU matrix multiplications.
