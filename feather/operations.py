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

# ----- triton GPU kernels implementation

@triton.jit
def _dot_fp16_acc_fp32_kernel(
    x1:torch.Tensor,
    x2:torch.Tensor,
    out:torch.Tensor,
    n:int,
    BLOCK_SIZE:tl.constexpr
):
    """Internal Kernel!, do not use, use `dot_fp16_acc_fp32_gpu`"""
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
    acc_block = tl.sum(acc, axis=0, dtype=tl.float32)
    tl.atomic_add(pointer=out, val=acc_block)

@triton.jit
def _dot_fp8_acc_fp32_kernel(
    x1:torch.Tensor,
    x2:torch.Tensor,
    out:torch.Tensor,
    n:int,
    BLOCK_SIZE:tl.constexpr
):
    """Internal Kernel!, do not use, use `dot_fp8_acc_fp32_gpu`"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(start=0, end=BLOCK_SIZE)

    mask = offsets < n

    # raw uint32 pointers
    x1_packed = tl.load(pointer=x1+offsets, mask=mask, other=0)
    x2_packed = tl.load(pointer=x2+offsets, mask=mask, other=0)

    # extract stored floating points
    # a, b, c, d represents the lower -> higher bits with stride 8
    x1_a = tl.cast((x1_packed) & 0xFF, dtype=tl.uint16) << 8
    x1_b = tl.cast((x1_packed >> 8) & 0xFF, dtype=tl.uint16) << 8
    x1_c = tl.cast((x1_packed >> 16) & 0xFF, dtype=tl.uint16) << 8
    x1_d = tl.cast((x1_packed >> 24) & 0xFF, dtype=tl.uint16) << 8

    x2_a = tl.cast((x2_packed) & 0xFF, dtype=tl.uint16) << 8
    x2_b = tl.cast((x2_packed >> 8) & 0xFF, dtype=tl.uint16) << 8
    x2_c = tl.cast((x2_packed >> 16) & 0xFF, dtype=tl.uint16) << 8
    x2_d = tl.cast((x2_packed >> 24) & 0xFF, dtype=tl.uint16) << 8

    # cast raw bits into FP16 values
    x1_a_val = tl.cast(tl.cast(x1_a, dtype=tl.float16, bitcast=True), dtype=tl.float32)
    x1_b_val = tl.cast(tl.cast(x1_b, dtype=tl.float16, bitcast=True), dtype=tl.float32)
    x1_c_val = tl.cast(tl.cast(x1_c, dtype=tl.float16, bitcast=True), dtype=tl.float32)
    x1_d_val = tl.cast(tl.cast(x1_d, dtype=tl.float16, bitcast=True), dtype=tl.float32)

    x2_a_val = tl.cast(tl.cast(x2_a, dtype=tl.float16, bitcast=True), dtype=tl.float32)
    x2_b_val = tl.cast(tl.cast(x2_b, dtype=tl.float16, bitcast=True), dtype=tl.float32)
    x2_c_val = tl.cast(tl.cast(x2_c, dtype=tl.float16, bitcast=True), dtype=tl.float32)
    x2_d_val = tl.cast(tl.cast(x2_d, dtype=tl.float16, bitcast=True), dtype=tl.float32)

    # accumulate in float32
    acc1 = tl.add(x1_a_val * x2_a_val, x1_b_val * x2_b_val)
    acc2 = tl.add(x1_c_val * x2_c_val, x1_d_val * x2_d_val)
    tl.atomic_add(pointer=out, val=tl.add(tl.sum(acc1, dtype=tl.float32), tl.sum(acc2, dtype=tl.float32)))

@triton.jit
def _gemv_fp8_acc_fp32_kernel(
    m: torch.Tensor, 
    v: torch.Tensor, 
    out: torch.Tensor, 
    n_col: int,       
    BLOCK_SIZE: tl.constexpr
):
    """Internal Kernel!, do not use, use `gemv_fp8_acc_fp32_gpu`"""
    pid = tl.program_id(axis=0)
    p_m_begin = m + (pid * n_col)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_col

    m_packed = tl.load(p_m_begin + offsets, mask=mask, other=0)
    v_packed = tl.load(v + offsets, mask=mask, other=0)

    # extract stored floating points
    # a, b, c, d represents the lower -> higher bits with stride 8
    m_a = tl.cast((m_packed) & 0xFF, dtype=tl.uint16) << 8
    m_b = tl.cast((m_packed >> 8) & 0xFF, dtype=tl.uint16) << 8
    m_c = tl.cast((m_packed >> 16) & 0xFF, dtype=tl.uint16) << 8
    m_d = tl.cast((m_packed >> 24) & 0xFF, dtype=tl.uint16) << 8

    v_a = tl.cast((v_packed) & 0xFF, dtype=tl.uint16) << 8
    v_b = tl.cast((v_packed >> 8) & 0xFF, dtype=tl.uint16) << 8
    v_c = tl.cast((v_packed >> 16) & 0xFF, dtype=tl.uint16) << 8
    v_d = tl.cast((v_packed >> 24) & 0xFF, dtype=tl.uint16) << 8

    # cast raw bits into FP16 values & accumulate
    acc = (m_a.to(tl.float16, bitcast=True).to(tl.float32) * v_a.to(tl.float16, bitcast=True).to(tl.float32)) + \
          (m_b.to(tl.float16, bitcast=True).to(tl.float32) * v_b.to(tl.float16, bitcast=True).to(tl.float32)) + \
          (m_c.to(tl.float16, bitcast=True).to(tl.float32) * v_c.to(tl.float16, bitcast=True).to(tl.float32)) + \
          (m_d.to(tl.float16, bitcast=True).to(tl.float32) * v_d.to(tl.float16, bitcast=True).to(tl.float32))

    tl.store(out + pid, tl.sum(acc))

def dot_fp16_acc_fp32_gpu(
    x1:torch.Tensor,
    x2:torch.Tensor
):
    """
    performs dot product on `FP16` packed `FP32` arrays, equivalent to
    `torch.dot(x1, x2)` or `np.dot(x1, x2)`

    NOTE: use `pack_fp16_ndarray` and convert it to `torch.tensor` using `torch.from_numpy()`

    :param x1: tensor 1
    :type x1: torch.Tensor
    :param x2: tensor 2
    :type x2: torch.Tensor
    """
    out = torch.tensor(0.0, dtype=torch.float32).to("cuda")
    n_elements = x1.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _dot_fp16_acc_fp32_kernel[grid](x1, x2, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

def dot_fp8_acc_fp32_gpu(
    x1:torch.Tensor,
    x2:torch.Tensor
):
    """
    performs dot product on `FP16` casted to `FP8` and packed `FP32` arrays, equivalent to
    `torch.dot(x1, x2)` or `np.dot(x1, x2)`

    NOTE: use `pack_fp8_ndarray` and convert it to `torch.tensor` using `torch.from_numpy()`

    :param x1: tensor 1
    :type x1: torch.Tensor
    :param x2: tensor 2
    :type x2: torch.Tensor
    """
    out = torch.tensor(0.0, dtype=torch.float32).to("cuda")
    n_elements = x1.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _dot_fp8_acc_fp32_kernel[grid](x1, x2, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

def gemv_fp8_acc_fp32_gpu(
    m:torch.Tensor,
    v:torch.Tensor,
    m_shape: tuple
):
    """
    performs GEMV on `FP16` casted to `FP8` and packed `FP32` arrays, equivalent to
    `torch.mv(m, v)`

    NOTE: use `pack_fp8_tensor` and convert it to `torch.tensor`

    :param m: matrix
    :type n: torch.Tensor
    :param v: vector
    :type v: torch.Tensor
    """
    out = torch.empty((m_shape[0], ), dtype=torch.float32).to("cuda")

    BLOCK_SIZE = 1024
    grid = (m_shape[0], )
    
    _gemv_fp8_acc_fp32_kernel[grid](m, v, out, m_shape[1], BLOCK_SIZE=BLOCK_SIZE)
    return out