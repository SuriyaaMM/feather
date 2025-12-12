import numba
import numpy as np
import torch
import triton
import triton.language as tl

from feather.packers import *

def dot_fp16_acc_fp32_vec(
    x1: np.ndarray, x2: np.ndarray
) -> np.ndarray:
    """
    Performs `DOT` subroutine on `FP16` packed into `FP32` arrays. 
    Computation-wise should be equivalent to `np.dot(x1, x2)`.

    Parameters
    ----------
    x1 : np.ndarray
        Vector ndarray (packed format).
    x2 : np.ndarray
        Vector ndarray (packed format).
        
    Returns
    -------
    np.ndarray
        Output vector in FP32 format.

    Note
    ----
    - If the input arrays `x1` and `x2` are not packed, then the computation
    will result in garbage.
    
    Examples
    --------
    >>> a = np.random.randint(low=-3, high=3, size=(4,)).astype(np.float16)
    >>> [-3. -3. -3.  1.]
    >>> b = np.random.randint(low=-3, high=3, size=(4,)).astype(np.float16)
    >>> [-3.  1.  2. -1.]
    >>> a_packed = pack_fp16_ndarray(a)
    >>> b_packed = pack_fp16_ndarray(b)
    >>> dot = dot_fp16_acc_fp32_vec(a_packed, b_packed)
    >>> -1.0
    """
    v1 = x1.view(np.float16)
    v2 = x2.view(np.float16)

    v1_fp32 = v1.astype(np.float32)
    v2_fp32 = v2.astype(np.float32)

    return np.dot(v1_fp32, v2_fp32)


@numba.njit(fastmath=True, parallel=True)
def dot_fp16_acc_fp32_numba(
    x1: np.ndarray[np.dtype[np.float32]],
    x2: np.ndarray[np.dtype[np.float32]],
    lut: np.ndarray,
):
    """
    Performs `DOT` subroutine on `FP16` packed into `FP32` arrays.
    Similar to `dot_fp16_acc_fp32_vec` but accelerated using `numba`.
    Computation-wise should be equivalent to `np.dot(x1, x2)`.

    Parameters
    ----------
    x1 : np.ndarray
        Vector ndarray (packed format).
    x2 : np.ndarray
        Vector ndarray (packed format).
        
    Returns
    -------
    np.ndarray
        Output vector in FP32 format.

    Note
    ----
    - If the input arrays `x1` and `x2` are not packed, then the computation
    will result in garbage.
    
    Examples
    --------
    >>> a = np.random.randint(low=-3, high=3, size=(4,)).astype(np.float16)
    >>> [-3. -3. -3.  1.]
    >>> b = np.random.randint(low=-3, high=3, size=(4,)).astype(np.float16)
    >>> [-3.  1.  2. -1.]
    >>> a_packed = pack_fp16_ndarray(a)
    >>> b_packed = pack_fp16_ndarray(b)
    >>> # Lookup table is necessary for numba due to its undefined fp16 types
    >>> _FP16_LUT = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    >>> dot = dot_fp16_acc_fp32_numba(a_packed, b_packed, _FP16_LUT)
    >>> -1.0
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
    x1: torch.Tensor,
    x2: torch.Tensor,
    out: torch.Tensor,
    n: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Internal Kernel!, do not use, use `dot_fp16_acc_fp32_gpu`"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(start=0, end=BLOCK_SIZE)

    mask = offsets < n

    # raw uint32 pointers
    x1_packed = tl.load(pointer=x1 + offsets, mask=mask, other=0)
    x2_packed = tl.load(pointer=x2 + offsets, mask=mask, other=0)

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
def _dot_fp8_e5m2_acc_fp32_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    out: torch.Tensor,
    n: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Internal Kernel!, do not use, use `dot_fp8_acc_fp32_gpu`"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(start=0, end=BLOCK_SIZE)

    mask = offsets < n

    # raw uint32 pointers
    x1_packed = tl.load(pointer=x1 + offsets, mask=mask, other=0)
    x2_packed = tl.load(pointer=x2 + offsets, mask=mask, other=0)

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
    tl.atomic_add(
        pointer=out,
        val=tl.add(tl.sum(acc1, dtype=tl.float32), tl.sum(acc2, dtype=tl.float32)),
    )

def dot_fp16_acc_fp32_gpu(x1: torch.Tensor, x2: torch.Tensor):
    """
    Performs `DOT` subroutine on `FP16` packed into `FP32` arrays. 
    Computation-wise should be equivalent to `torch.dot(x1, x2)`.

    Parameters
    ----------
    x1 : torch.Tensor
        Vector tensor (packed format).
    x2 : torch.Tensor
        Vector tensor (packed format).
        
    Returns
    -------
    torch.Tensor
        Output vector in FP32 format.
    """
    out = torch.tensor(0.0, dtype=torch.float32).to("cuda")
    n_elements = x1.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _dot_fp16_acc_fp32_kernel[grid](x1, x2, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

def dot_fp8_e5m2_acc_fp32_gpu(x1: torch.Tensor, x2: torch.Tensor):
    """
    Performs `DOT` subroutine on `FP8_E5M2` packed into `FP32` arrays. 
    Computation-wise should be equivalent to `torch.dot(x1, x2)`.

    Parameters
    ----------
    x1 : torch.Tensor
        Vector tensor (packed format).
    x2 : torch.Tensor
        Vector tensor (packed format).
        
    Returns
    -------
    torch.Tensor
        Output vector in FP32 format.
        
    Notes
    -----
    - Input tensors must be packed using one of the functions exposed in 
    `feather.packers.fp8` module, else computation is undefined.
    
    Examples
    --------
    >>> a = torch.randint(low=-3, high=3, size=(4, 4), dtype=torch.float16)
    >>> tensor([ 2.,  0., -3.,  1.], dtype=torch.float16)
    >>> b = torch.randint(low=-3, high=3, size=(4,), dtype=torch.float16)
    >>> tensor([-1.,  0., -3.,  1.], dtype=torch.float16)
    >>> a_packed = pack_fp8_tensor(a, mode="E5M2").to("cuda")
    >>> b_packed = pack_fp8_tensor(b, mode="E5M2").to("cuda")
    >>> dot = dot_fp8_e5m2_acc_fp32_gpu(a_packed, b_packed)
    >>> tensor(8., device='cuda:0')
    """
    out = torch.tensor(0.0, dtype=torch.float32).to("cuda")
    n_elements = x1.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _dot_fp8_e5m2_acc_fp32_kernel[grid](x1, x2, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out