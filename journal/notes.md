# Notes


## 06-08-2026
Goal: analyze baseline benchmark results for the target-model
Finding:
- TTFT starts of by being bound by the base overhead of launching CUDA kernels and moving model weights through the memory.
    So for a range of small input tokens, the TTFT stays the same, and as it increases above a certain threshold then it becomes input-token bound & TTFT starts to increase beyond the baseline. Baseline TTFT: 500 ms
- torch.tensor(generated_ids) allocation — this line rebuilds a fresh CPU tensor from the entire Python list and copies it 
    to GPU every single decode step, and the list grows each step.

