from packers import *
from operations import *
import numpy as np
import torch

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

a = np.random.normal(size=(15000, )).astype(np.float16)
b = np.random.normal(size=(15000, )).astype(np.float16)

a_packed = pack_fp16_ndarray(a)
b_packed = pack_fp16_ndarray(b)

device = "cuda"

print(f"dot prod (gpu) = ", dot_fp16_acc_fp32_gpu(a, b))
print(f"dot prod (np) = ", np.dot(a, b))