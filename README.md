# Feather

**Feather** is a high-performance emulation library that brings **FP8 (E5M2 & E4M3)** precision arithmetic to older GPU architectures (Ampere, Turing, Volta) that lack native hardware support.

By implementing custom bit-packing and optimized **Triton** kernels, Feather bypasses the memory bandwidth bottleneck, achieving **3x speedups** over native PyTorch FP32 operations on memory-bound workloads.

## Why Feather?

Modern deep learning benefits enormously from low-precision arithmetic, but native hardware support is gated behind the latest GPUs:

  - **Native FP8**: Exclusive to Ada Lovelace (RTX 40-series) and Hopper (H100).
  - **Older Hardware** (RTX 3090, A100, V100): Forced to use FP16 or FP32, leaving huge bandwidth potential on the table.
  - **NOTE**: Also supports FP16 packing.

## Key Features

  - **Software-Based FP8**: Runs `E5M2` and `E4M3` on any CUDA GPU (RTX 20/30 series supported).
  - **3x Speedup**: Validated **3.37x** speedup on large GEMV operations.
  - **Correctness**:
      - **E5M2**: Direct bit-manipulation for maximum speed.
      - **E4M3**: Full software emulation (rebias + denorm flushing) for stability.

## Benchmark Results

Performance measured on **NVIDIA RTX 3050 (Ampere)**, comparing PyTorch Native (FP32) vs. Feather (FP8).
*Task: Large Scale Matrix-Vector Multiplication (GEMV)*

### Summary Table

| Matrix Shape | PyTorch (FP32) | Feather (FP8-E5M2) | Feather (FP8-E4M3) | 
|---|---|---|---|
| **8192 x 8192** | 1,389 $\mu s$ | 431 $\mu s$ | 720 $\mu s$ |
| **16384 x 8192** | 2,862 $\mu s$ | 841 $\mu s$ | 1,368 $\mu s$ |  
| **16384 x 16384** | 5,635 $\mu s$ | 1,679 $\mu s$ | 2,703 $\mu s$ |  

More results are in `results/`

## Installation

### Prerequisites

  - Python 3.10+
  - CUDA-capable GPU
  - [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/SuriyaaMM/feather
cd feather

# Install dependencies
uv sync 
source .venv/bin/activate

# Run benchmarks
uv run pytest benchmark/bench_gemv.py --benchmark-histogram=benchmark_results
uv run pytest benchmark/bench_dot.py --benchmark-histogram=benchmark_results
```

## Usage Example

```python
import torch
import feather

# 1. Create standard data
matrix = torch.randn(16384, 16384, dtype=torch.float16, device='cuda')
vector = torch.randn(16384, dtype=torch.float16, device='cuda')

# 2. Pack data (Offline / Pre-processing step)
# Compresses memory usage by 2x (vs FP16) or 4x (vs FP32)
m_packed = feather.pack_fp8_tensor(matrix, mode="E5M2").cuda()
v_packed = feather.pack_fp8_tensor(vector, mode="E5M2").cuda()

# 3. Run Fast GEMV
# Returns FP32 result with FP32 accumulation
result = feather.gemv(m_packed, v_packed, original_shape=matrix.shape)
```

## Roadmap

  - [x] GEMV
  - [x] E5M2 Support 
  - [x] E4M3 Support
  - [ ] SDPA Kernels

## License

MIT License - see [LICENSE](https://www.google.com/search?q=LICENSE) for details.