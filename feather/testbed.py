import numpy as np
import torch
from feather.packers.fp8 import *
from feather.packers.fp16 import *
from feather.routines.gemv import *
from feather.routines.dot import *

# a = np.array([4.5], dtype=np.float16)
# b = np.array([1.25], dtype=np.float16)

# packed = pack_fp16_into_fp32(a, b)
# unpacked = unpack_fp32_into_fp16(packed)

# print(f"a = {a}")
# print(f"b = {b}")

# print(f"packed = {packed}")
# print(f"unpacked = {unpacked}")

# a_bits = a.view(np.uint16)[0]
# b_bits = b.view(np.uint16)[0]
# packed_bits = packed.view(np.uint32)[0]
# unpacked_bits = unpacked.view(np.uint16)
# a_unpacked_bits = unpacked.view(np.uint16)[0]
# b_unpacked_bits = unpacked.view(np.uint16)[1]

# print(f"a (FP16) bits: {format(a_bits, '016b')}")
# print(f"b (FP16) bits: {format(b_bits, '016b')}")

# print(f"packed (FP32) bits: {format(packed_bits, '032b')}")
# print(f"unpacked (0) (FP32) bits: {format(unpacked_bits[0], '032b')}")
# print(f"unpacked (1) (FP32) bits: {format(unpacked_bits[0], '032b')}")

# print(f"a (FP32) bits: {format(a_unpacked_bits, '016b')}")
# print(f"b (FP32) bits: {format(b_unpacked_bits, '016b')}")

# a = np.array([4.5, 1.25, 6.175, 2.375], dtype=np.float16)

# packed = pack_fp16_ndarray(a)
# unpacked = unpack_fp32_ndarray(packed)

# print(f"a = {a}")
# print(f"packed = {packed}")
# print(f"unpacked = {unpacked}")

# a = np.random.normal(size=(15000, )).astype(np.float16)
# b = np.random.normal(size=(15000, )).astype(np.float16)

# a_packed = pack_fp16_ndarray(a)
# b_packed = pack_fp16_ndarray(b)

# device = "cuda"

# print(f"dot prod (gpu) = ", dot_fp16_acc_fp32_gpu(a, b))
# print(f"dot prod (np) = ", np.dot(a, b))

# ----- FP8 test bench basic packing & unpacking
# a = np.array([0.275], dtype=np.float16)
# b = np.array([0.275], dtype=np.float16)
# c = np.array([0.275], dtype=np.float16)
# d = np.array([0.275], dtype=np.float16)

# packed = pack_fp8_into_fp32(a, b, c, d)
# unpacked = unpack_fp8_from_fp32(packed)

# print("a = ", format(a.view(np.uint16)[0], "016b"))
# print(f"b = ", format(b.view(np.uint16)[0], "016b"))
# print(f"c = ", format(c.view(np.uint16)[0], "016b"))
# print(f"d = ", format(d.view(np.uint16)[0], "016b"))
# print(f"packed = ", format(packed.view(np.uint32)[0], "032b"))
# print(f"unpacked = {unpacked}")

# ----- FP8 array packing & unpacking
# a = np.random.normal(size=(32,)).astype(np.float16)

# packed = pack_fp8_ndarray(a)
# unpacked = unpack_fp8_ndarray(packed)

# print(f"a = {a}")
# print(f"unpacked = {unpacked}")

# a = np.random.normal(size=(300,)).astype(np.float16)
# b = np.random.normal(size=(300,)).astype(np.float16)

# print(f"dot (np) = ", np.dot(a, b))

# a_packed = pack_fp8_ndarray(a)
# b_packed = pack_fp8_ndarray(b)

# a_tensor = torch.from_numpy(a_packed).view(torch.uint32).to("cuda")
# b_tensor = torch.from_numpy(b_packed).view(torch.uint32).to("cuda")

# print(f"dot (feather) = ", dot_fp8_acc_fp32_gpu(a_tensor, b_tensor))

# a = torch.randint(low=-3, high=3, size=(4, 4), dtype=torch.float16)
# b = torch.randint(low=-3, high=3, size=(4,), dtype=torch.float16)

# a_packed = pack_fp8_tensor(a, mode="E5M2").to("cuda")
# b_packed = pack_fp8_tensor(b, mode="E5M2").to("cuda")

# gemv = gemv_fp8_e5m2_acc_fp32_gpu(a_packed, b_packed, a.shape)
# print(a)
# print(b)
# print(gemv)

# a = np.random.randint(low=-3, high=3, size=(4,)).astype(np.float16)
# b = np.random.randint(low=-3, high=3, size=(4,)).astype(np.float16)

# a_packed = pack_fp16_ndarray(a)
# b_packed = pack_fp16_ndarray(b)

# _FP16_LUT = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
# dot = dot_fp16_acc_fp32_numba(a_packed, b_packed, _FP16_LUT)

# print(a)
# print(b)
# print(dot)
