# Feather

**Feather** is a lightweight emulation library for **FP16 (Half-Precision)** arithmetic and storage on architectures that primarily support FP32.

It currently implements **bit-packing techniques** (storing two `float16` values inside a single `float32` register) to simulate reduced memory bandwidth usage. While the current implementation is a CPU-based NumPy prototype, the goal is to develop custom GPU kernels (Torch/CUDA) to bring efficient mixed-precision support to older NVIDIA architectures or specific hardware constraints lacking native FP16 Tensor Cores.

### Features

  * **Bit-Packing:** Efficiently pack two `FP16` values into a single `FP32` container.
  * **Zero-Copy Views:** Vectorized unpacking using NumPy memory views for high performance.
  * **Emulated Arithmetic:** Perform `FP16` dot products with `FP32` accumulation (software emulation of mixed-precision MAC operations).

### Installation

- Pre-Requisites : `uv`

```bash
git clone https://github.com/SuriyaaMM/feather
cd feather
# sync uv packages
uv sync
source .venv/bin/activate
```
