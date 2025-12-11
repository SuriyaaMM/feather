import time
import numpy as np
from packers import *
from operations import *

def dot_python_loop(a, b):
    acc = 0.0
    for ai, bi in zip(a, b):
        acc += float(ai) * float(bi)
    return acc

N = 2_000_000
print(f"--- Benchmark: N = {N} elements ---\n")

a = np.random.normal(size=(N,)).astype(np.float16)
b = np.random.normal(size=(N,)).astype(np.float16)

# packing
ap = pack_fp16_ndarray(a)
bp = pack_fp16_ndarray(b)

dot_fp16_acc_fp32(ap, bp) 

t0 = time.time()
packed_dot = dot_fp16_acc_fp32_vec(ap, bp)
t1 = time.time()

packed_time = t1 - t0
packed_bytes = ap.nbytes + bp.nbytes
packed_bw = packed_bytes / packed_time / 1e9
packed_eff = (2 * len(ap)) / packed_time / 1e6

dot_python_loop(a, b)

t0 = time.time()
python_dot = dot_python_loop(a, b)
t1 = time.time()

python_time = t1 - t0
python_bytes = a.nbytes + b.nbytes
python_bw = python_bytes / python_time / 1e9
python_eff = N / python_time / 1e6
_FP16_LUT = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)

# warmup
dot_fp16_acc_fp32_numba(ap, bp, _FP16_LUT)

import time
t0 = time.time()
r = dot_fp16_acc_fp32_numba(ap, bp, _FP16_LUT)
t1 = time.time()

dt = t1 - t0

bytes_read = ap.nbytes + bp.nbytes

print("numba packed-dot result:", r)
print("time:", dt)
print("bandwidth (GB/s):", bytes_read / dt / 1e9)
print("effective fp16 elems/s (M):", (2 * len(ap)) / dt / 1e6)


# np standard dot
tmp_a = a.astype(np.float32)
tmp_b = b.astype(np.float32)

np.dot(tmp_a, tmp_b)  

t0 = time.time()
numpy_dot = np.dot(tmp_a, tmp_b)
t1 = time.time()

numpy_time = t1 - t0
numpy_bytes = tmp_a.nbytes + tmp_b.nbytes
numpy_bw = numpy_bytes / numpy_time / 1e9
numpy_eff = N / numpy_time / 1e6


print("=== Packed FP16×2 Dot Product ===")
print(f"result                : {packed_dot}")
print(f"time (s)              : {packed_time}")
print(f"bandwidth (GB/s)      : {packed_bw}")
print(f"effective elems/s (M) : {packed_eff}")
print()

print("=== Pure Python FP16 Dot Product ===")
print(f"result                : {python_dot}")
print(f"time (s)              : {python_time}")
print(f"bandwidth (GB/s)      : {python_bw}")
print(f"effective elems/s (M) : {python_eff}")
print()

print("=== NumPy FP32 Dot Product ===")
print(f"result                : {numpy_dot}")
print(f"time (s)              : {numpy_time}")
print(f"bandwidth (GB/s)      : {numpy_bw}")
print(f"effective elems/s (M) : {numpy_eff}")
print()
