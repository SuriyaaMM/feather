# Feather

**Feather** is a high-performance emulation library that brings **FP8** and **FP16** precision arithmetic to older GPU architectures lacking native low-precision support.

By implementing custom bit-packing techniques and optimized Triton kernels, Feather achieves **2x+ speedups** over native PyTorch FP32/FP16 operations on memory-bandwidth-bound workloads—without requiring Ada Lovelace (RTX 40 series) or newer hardware.

## Why Feather?

Modern deep learning benefits enormously from low-precision arithmetic (FP8, FP16), but native hardware support is limited to recent GPU generations:

- **FP8 native support**: Only available on Ada Lovelace+ (RTX 40 series, H100, L40S)
- **Older architectures** (RTX 20/30 series, A100, V100): No native FP8, limited FP16 benefits for memory-bound operations

**Feather bridges this gap** with software-based precision emulation that delivers real performance gains on older hardware.

## Key Features

- **Bit-Packing**: Pack `2xFP16` or `4xFP8` values into single FP32 containers
- **Custom GPU Kernels**: Optimized Triton kernels with FP32 accumulation
- **Memory Bandwidth Optimization**: `2-4x` reduction in data transfer
- **Zero Hardware Requirements**: Works on any CUDA-capable GPU
- **Production-Ready**: Includes NumPy (CPU) and GPU implementations with comprehensive tests

## Benchmark Results

Performance comparison of native PyTorch operations vs Feather's packed precision implementations on GPU.

### Summary Table

| Implementation | 100K Elements | 500K Elements | 1.5M Elements | Speedup (1.5M) |
|---|---|---|---|---|
| **PyTorch FP32** | 8.24 $\mu s$ | 34.16 $\mu s$ | 94.23 $\mu s$ | 1.0x (baseline) |
| **Feather FP16 (packed)** | 26.36 $\mu s$ | 35.36 $\mu s$ | 87.67 $\mu s$ | **1.07×** |
| **Feather FP8 (packed)** | 24.45 $\mu s$ | 25.70 $\mu s$ | 43.71 $\mu s$ | **2.16×** |

### Performance by Scale

#### Small Scale (100K elements)
- Kernel launch overhead dominates
- **Recommendation**: Use native PyTorch for workloads <500K elements

#### Medium Scale (500K elements)
- Feather FP8 shows **25% improvement** over native operations
- Memory bandwidth savings begin offsetting unpacking overhead

#### Large Scale (1.5M elements)
- **Feather FP8**: **2.16× faster** than PyTorch FP32/FP16
- **Feather FP16**: **1.07× faster**, competitive with native implementations
- Memory bandwidth becomes the primary bottleneck—packing delivers maximum benefit

### Memory Bandwidth Comparison

| Precision | Bytes per Element | Total Memory (1.5M elements) | Bandwidth Reduction |
|---|---|---|---|
| FP32 | 4 bytes | 12 MB | 1x (baseline) |
| FP16 (native) | 2 bytes | 6 MB | 2x |
| FP16 (packed) | 2 bytes | 6 MB | 2x |
| **FP8 (packed)** | **1 byte** | **3 MB** | **4x** |

## Key Insights

- **Real speedups on large-scale workloads**: `2x+` performance improvement for arrays >1M elements

= **Unlocks FP8 on older hardware**: Brings modern low-precision benefits to RTX 20/30 series, A100, V100

- **Memory-bandwidth optimization**: Reduces data transfer by up to `4x` for FP8

- **Overhead on small workloads**: Best for arrays >500K elements; use native PyTorch for smaller sizes

- **Precision trade-offs**: FP8 E5M2 format suitable for inference, not training

## Installation

### Prerequisites
- Python 3.11+
- CUDA-capable GPU
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/SuriyaaMM/feather
cd feather

# Install dependencies with uv
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run benchmarks
pytest bench_operations.py --benchmark-only
```

## Architecture

Feather implements three computational backends:

1. **NumPy (CPU)**: Reference implementation for correctness validation
2. **Numba (CPU)**: JIT-compiled accelerated operations
3. **Triton (GPU)**: Custom CUDA kernels with optimal memory access patterns

All implementations use FP32 accumulation to maintain numerical stability while benefiting from reduced memory bandwidth.

## Test Environment

- **GPU**: NVIDIA Ampere/Turing architecture (tested on RTX3050, :( i have only this!))
- **Framework**: PyTorch 2.0+, Triton 2.0+
- **Benchmark Tool**: pytest-benchmark
- **Operations**: Dot product with FP32 accumulation

## Roadmap

- [ ] Matrix multiplication (GEMM) kernels
- [ ] Attention mechanism implementations
- [ ] Batch operations support
- [ ] INT8 quantization support
- [ ] Auto-tuning for optimal block sizes
- [ ] Integration with popular ML frameworks

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:
- [Triton](https://github.com/openai/triton) - GPU kernel language
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [NumPy](https://numpy.org/) - Numerical computing