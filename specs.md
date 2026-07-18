# Laptop Hardware Specifications

> Last updated: 2026-07-18
> Some fields marked `?` are unknown — fill them in with `msinfo32` or `dxdiag`.

---

## CUDA & Driver — Confirmed via `nvidia-smi`

| Property | Value |
|---|---|
| NVIDIA-SMI Version | 592.82 |
| Driver Version | 592.82 |
| CUDA Version | 13.1 |
| Driver Model | WDDM |
| GPU Bus ID | 00000000:01:00.0 |
| Display Active | Off |
| VRAM Total | 2048 MiB (2 GB) — confirmed |
| VRAM Used (idle) | 0 MiB |
| GPU Utilization (idle) | 0% |
| Idle Temperature | 52°C |
| Performance State (idle) | P0 (maximum performance state) |
| Compute Mode | Default |
| MIG Mode | N/A |
| ECC | N/A |
| Active Processes | None |

```
NVIDIA-SMI 592.82   Driver Version: 592.82   CUDA Version: 13.1
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce MX450         WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   52C    P0            N/A  /  N/A  |       0MiB /   2048MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

> **Note on idle P0 state:** The GPU sitting at P0 while idle is normal on Windows WDDM — it means the driver hasn't downclocked it yet. Under load it will stay at P0 and boost clocks.
> **Note on CUDA 13.1:** This is a very recent CUDA toolkit. Ensure your PyTorch build is CUDA-compatible (PyTorch ships its own bundled CUDA runtime, so the system CUDA version only matters for native extensions like `bitsandbytes`).

---

## CPU — Intel Core i7 (11th Gen, Tiger Lake)

| Property | Value |
|---|---|
| Family | Intel Core i7 — 11th Generation |
| Codename | Tiger Lake |
| Microarchitecture | Willow Cove |
| Process Node | Intel 10nm SuperFin |
| Cores / Threads | 4 / 8 |
| L3 Cache | 12 MB |
| TDP | 12–35 W (varies by variant & laptop OEM) |
| Max Turbo Boost | 4.70–5.00 GHz (variant-dependent) |
| Memory Support | DDR4-3200 / LPDDR4x-4267, dual-channel |
| PCIe | Gen 4 |
| Connectivity | Thunderbolt 4, USB 4 |
| Integrated GPU | Intel Iris Xe Graphics (96 EUs) |
| Exact SKU | ? (likely i7-1165G7 / i7-1185G7 / i7-11370H — check `msinfo32`) |

---

## GPU (Discrete) — NVIDIA GeForce MX450

| Property | Value |
|---|---|
| Architecture | Turing (TU117) |
| CUDA Cores | 896 |
| VRAM | 2 GB GDDR6 — **confirmed 2048 MiB** |
| Memory Bus Width | 64-bit |
| Memory Bandwidth | ~80 GB/s |
| Base Clock | ~1,395 MHz |
| Boost Clock | ~1,575 MHz |
| TDP | 25–28.5 W (OEM-tuned) |
| PCIe | Gen 4 |
| Tensor Cores | None (Turing consumer mobile, no RT/Tensor on MX-class) |
| CUDA Compute Capability | 7.5 |
| Driver API | CUDA |

> **Note:** NVIDIA did not publish a single official spec sheet for the MX450.
> Two laptops with an "MX450" can perform differently due to OEM power tuning.

---

## GPU (Integrated) — Intel Iris Xe Graphics

| Property | Value |
|---|---|
| Execution Units | 96 |
| Shared Memory | Uses system RAM |
| API | OpenCL, DirectX 12, Vulkan |
| AI Acceleration | Intel Deep Learning Boost (VNNI) via CPU |

---

## RAM

| Property | Value |
|---|---|
| Type | DDR4 / LPDDR4x |
| Speed | 3200 / 4267 MHz |
| Channels | Dual-channel |
| Total Capacity | ? GB (check `msinfo32`) |

---

## Storage

| Property | Value |
|---|---|
| Type | ? (likely NVMe PCIe Gen 4 SSD) |
| Capacity | ? |

---

## OS & Runtime

| Property | Value |
|---|---|
| Operating System | Windows (confirmed) |
| Python Environment | venv |
| ML Libraries | PyTorch (CUDA), Transformers, BitsAndBytes, Accelerate |

---

## AI Inference Profile

| Role | Hardware | Notes |
|---|---|---|
| **Target / main model** | MX450 GPU (CUDA) | 2 GB VRAM — use 4-bit NF4 quantization |
| **Draft model (speculative decoding)** | i7 CPU | Qwen2.5-0.5B — runs on CPU via PyTorch |
| **Max safe context (GPU)** | ~2K–4K tokens | Larger context risks OOM on 2 GB VRAM |
| **Speculative decoding gamma** | 4–6 tokens | Typical draft lookahead for small draft models |

### Why This Setup Works for Speculative Decoding
- The **MX450 verifies** draft tokens in parallel on GPU — verification is cheap
- The **i7 (0.5B draft model)** generates candidate tokens on CPU — fast at 0.5B scale
- Net effect: fewer GPU forward passes → faster wall-clock generation
- The CPU and GPU can pipeline — while GPU verifies batch N, CPU drafts batch N+1

---

## Quick Reference Commands

```powershell
# Full system info
msinfo32

# GPU info (DirectX)
dxdiag

# GPU VRAM & driver from PowerShell
Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion

# Check CUDA from Python
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 1024**2, 'MB')"
```
