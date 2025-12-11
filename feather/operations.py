import numpy as np
import logging
import triton
import triton.language as tl
import torch
import numba
from feather.packers import *

def add_fp16_acc_fp32(
    x:np.ndarray
):
    """
    helper function to accumulate the packed `np.float16` compressed array
    into `np.float32`
    
    :param x: array to accumulate
    :type x: np.ndarray
    """
    acc = np.float32(0.0)
    for xi in x:
        xi_bits = np.array([xi], dtype=np.float32).view(np.uint32)[0]
        lo_bits = np.uint16(xi_bits & 0xFFFF)
        hi_bits = np.uint16((xi_bits >> 16) & 0xFFFF)
        lo_val = lo_bits.view(np.float16)[0]
        hi_val = hi_bits.view(np.float16)[0]
        acc += lo_val + hi_val
    return acc


def dot_fp16_acc_fp32_vec(
    x1:np.ndarray[np.dtype[np.float32]], 
    x2:np.ndarray[np.dtype[np.float32]]
):
    """
    performs dot product
    
    :param x1: input array 1 `FP16`
    :type x1: np.ndarray[np.dtype[np.float16]]
    :param x2: input array 2 `FP16`
    :type x2: np.ndarray[np.dtype[np.float16]]
    """
    v1 = x1.view(np.float16)
    v2 = x2.view(np.float16)
    
    v1_fp32 = v1.astype(np.float32)
    v2_fp32 = v2.astype(np.float32)

    return np.dot(v1_fp32, v2_fp32)

@numba.njit(fastmath=True, parallel=True)
def dot_fp16_acc_fp32_numba(
    x1:np.ndarray[np.dtype[np.float32]],
    x2:np.ndarray[np.dtype[np.float32]],
    lut:np.ndarray
):
    """
    numba accelerated dot product

    :param x1: input array 1 `FP32`
    :type x1: np.ndarray[np.dtype[np.float32]]
    :param x2: input array 2 `FP32`
    :type x2: np.ndarray[np.dtype[np.float32]]
    :param lut: Lookup table to support `FP16` in numba
    :type lut: np.ndarray
    """
    
    x1_bits = x1.view(np.uint16)
    x2_bits = x2.view(np.uint16)
    
    acc = np.float32(0.0)
    n = x1_bits.size
    
    for i in numba.prange(n):
        val_x = lut[x1_bits[i]]
        val_y = lut[x2_bits[i]]
        acc += val_x * val_y
        
    return acc


@triton.jit
def _dot_fp16_acc_fp32_kernel(
    x1:torch.Tensor,
    x2:torch.Tensor,
    out:torch.Tensor,
    n:int,
    BLOCK_SIZE:tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(start=0, end=BLOCK_SIZE)

    mask = offsets < n

    # raw uint32 pointers
    x1_packed = tl.load(pointer=x1+offsets, mask=mask, other=0)
    x2_packed = tl.load(pointer=x2+offsets, mask=mask, other=0)

    # extract stored floating points
    x1_lower_bits = tl.cast(x1_packed & 0xFFFF, dtype=tl.uint16)
    x2_lower_bits = tl.cast(x2_packed & 0xFFFF, dtype=tl.uint16)
    x1_upper_bits = tl.cast((x1_packed >> 16) & 0xFFFF, dtype=tl.uint16)
    x2_upper_bits = tl.cast((x2_packed >> 16) & 0xFFFF, dtype=tl.uint16)

    # cast raw bits into FP16 values
    x1_lower_val = tl.cast(x1_lower_bits, dtype=tl.float16, bitcast=True)
    x2_lower_val = tl.cast(x2_lower_bits, dtype=tl.float16, bitcast=True)
    x1_upper_val = tl.cast(x1_upper_bits, dtype=tl.float16, bitcast=True)
    x2_upper_val = tl.cast(x2_upper_bits, dtype=tl.float16, bitcast=True)

    # accumulate in float32
    acc = tl.add(x1_lower_val * x2_lower_val, x1_upper_val * x2_upper_val)
    tl.atomic_add(pointer=out, val=tl.sum(acc, dtype=tl.float32))

def dot_fp16_acc_fp32_gpu(
    x1:torch.Tensor,
    x2:torch.Tensor
):
    out = torch.tensor(0.0, dtype=torch.float32).to("cuda")
    n_elements = x1.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _dot_fp16_acc_fp32_kernel[grid](x1, x2, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out