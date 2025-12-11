# Feather

**Feather** is a lightweight emulation library for **FP16 (Half-Precision)** arithmetic and storage on architectures that primarily support FP32.

It currently implements **bit-packing techniques** (storing two `float16` values inside a single `float32` register) to simulate reduced memory bandwidth usage. While the current implementation is a CPU-based NumPy prototype, the goal is to develop custom GPU kernels (Torch/CUDA) to bring efficient mixed-precision support to older NVIDIA architectures or specific hardware constraints lacking native FP16 Tensor Cores.

### Features

  * **Bit-Packing:** Efficiently pack two `FP16` values into a single `FP32` container.
  * **Zero-Copy Views:** Vectorized unpacking using NumPy memory views for high performance.
  * **Emulated Arithmetic:** Perform `FP16` dot products with `FP32` accumulation (software emulation of mixed-precision MAC operations).


### Results
Yes, you absolutely should. You have empirical proof that your custom kernel beats a highly optimized library (PyTorch) by fundamentally changing how memory is accessed. That is the core value proposition of your project.

Here is a drop-in Markdown section for your `README.md`.

***

## Results

### GPU vs. GPU (1.5M Elements)

*Benchmark run on NVIDIA CUDA GPU using Triton*
```bash
# use this script to get similar results
uv run pytest   benchmark/bench_operations.py::test_dot_feather_gpu   benchmark/bench_operations.py::test_dot_torch   --benchmark-histogram=benchmark_results
```
| Implementation | Precision | Memory Load | Execution Time (Median) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch Native** | FP32 | 32-bit / 1 elems | 77.35 µs | 1.0x |
| **Feather** | **Packed FP16** | **32-bit / 2 elems** | **56.30 µs** | **1.37x** |

### Installation

- Pre-Requisites : `uv`

```bash
git clone https://github.com/SuriyaaMM/feather
cd feather
# sync uv packages
uv sync
source .venv/bin/activate
```
