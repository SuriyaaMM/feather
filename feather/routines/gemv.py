import numpy as np
import torch
import triton
import triton.language as tl

from feather.packers import *

@triton.jit
def _gemv_fp8_e5m2_acc_fp32_kernel(
    m: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    n_col: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Internal Kernel!, use `gemv_fp8_e5m2_acc_fp32_gpu`"""
    pid = tl.program_id(axis=0)
    m_begin = m + (pid * n_col)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for _offset in range(0, n_col, BLOCK_SIZE):
        offsets = _offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_col

        m_packed = tl.load(m_begin + offsets, mask=mask, other=0)
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

        # cast raw bits into fp16 values & accumulate in fp32
        acc += (
            (
                m_a.to(tl.float16, bitcast=True).to(tl.float32)
                * v_a.to(tl.float16, bitcast=True).to(tl.float32)
            )
            + (
                m_b.to(tl.float16, bitcast=True).to(tl.float32)
                * v_b.to(tl.float16, bitcast=True).to(tl.float32)
            )
            + (
                m_c.to(tl.float16, bitcast=True).to(tl.float32)
                * v_c.to(tl.float16, bitcast=True).to(tl.float32)
            )
            + (
                m_d.to(tl.float16, bitcast=True).to(tl.float32)
                * v_d.to(tl.float16, bitcast=True).to(tl.float32)
            )
        )

    tl.store(out + pid, tl.sum(acc))


@triton.jit
def _unpack_e4m3_to_fp32(x_u8):
    """Internal Kernel!, called by `_gemv_fp8_e4m3_acc_fp32_kernel`"""

    """
    conversion:
    x_u8 is tl.uint16, refer to `_gemv_fp8_e4m3_acc_fp32_kernel`
    
    first starting off with the mantissa correction `(x_u8 & 0x07)`
    extracts mantissa from `E4M3` then `<< 7` shifts it to upper bits of
    `FP16`

    next is bias correction, `FP16` bias = 15, `E4M3` bias = 7, 
    so we add 8 to after extracting the exponent, then place it right 
    position (10 bits after)

    last part is straightforward, just extract the sign bit
    then mash everything together using bitwise or
    """
    x_u16 = (
        ((x_u8 & 0x80) << 8)
        | (((((x_u8 & 0x78) >> 3) + 8) * (((x_u8 & 0x78) >> 3) > 0)) << 10)
        | ((x_u8 & 0x07) << 7)
    )
    return x_u16.to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _gemv_fp8_e4m3_acc_fp32_kernel(
    m: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    n_col: int,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Internal Kernel!, use `gemv_fp8_e4m3_acc_fp32_kernel`
    """
    pid = tl.program_id(axis=0)
    m_begin = m + (pid * n_col)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, n_col, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_col

        m_packed = tl.load(m_begin + offsets, mask=mask, other=0)
        v_packed = tl.load(v + offsets, mask=mask, other=0)

        # extract stored floating points
        m_a = tl.cast((m_packed) & 0xFF, dtype=tl.uint16)
        m_b = tl.cast((m_packed >> 8) & 0xFF, dtype=tl.uint16)
        m_c = tl.cast((m_packed >> 16) & 0xFF, dtype=tl.uint16)
        m_d = tl.cast((m_packed >> 24) & 0xFF, dtype=tl.uint16)

        v_a = tl.cast((v_packed) & 0xFF, dtype=tl.uint16)
        v_b = tl.cast((v_packed >> 8) & 0xFF, dtype=tl.uint16)
        v_c = tl.cast((v_packed >> 16) & 0xFF, dtype=tl.uint16)
        v_d = tl.cast((v_packed >> 24) & 0xFF, dtype=tl.uint16)

        # accumulate in fp32
        acc += (
            (_unpack_e4m3_to_fp32(m_a) * _unpack_e4m3_to_fp32(v_a))
            + (_unpack_e4m3_to_fp32(m_b) * _unpack_e4m3_to_fp32(v_b))
            + (_unpack_e4m3_to_fp32(m_c) * _unpack_e4m3_to_fp32(v_c))
            + (_unpack_e4m3_to_fp32(m_d) * _unpack_e4m3_to_fp32(v_d))
        )

    tl.store(out + pid, tl.sum(acc))


def gemv_fp8_e5m2_acc_fp32_gpu(m: torch.Tensor, v: torch.Tensor, m_shape: tuple):
    """
    Performs `GEMV` subroutine on `FP8_E5M2` packed into `FP32` arrays. 
    Computation-wise should be equivalent to `torch.mv(m, v)`.

    Parameters
    ----------
    m : torch.Tensor
        Matrix tensor (packed format).
    v : torch.Tensor
        Vector tensor (packed format).
    m_shape : tuple
        Shape of the original matrix before packing. Division by 4 will be
        performed internally by this function.
        
    Returns
    -------
    torch.Tensor
        Output vector in FP32 format.
        
    Notes
    -----
    - Parameter `m_shape` must be shape of the original matrix (before packing), division by 4 will be performed by this function itself internally.
    - Input tensors must be packed using one of the functions exposed in `feather.packers.fp8` module, else computation is undefined.
    
    Examples
    --------
    >>> a = torch.randint(low=-3, high=3, size=(4, 4), dtype=torch.float16)
    >>> tensor([[ 1.,  1., -2.,  0.],
        [-3., -3., -2., -3.],
        [ 0.,  1.,  1., -2.],
        [ 2.,  0., -1.,  2.]], dtype=torch.float16)
    >>> b = torch.randint(low=-3, high=3, size=(4,), dtype=torch.float16)
    >>> tensor([ 2.,  2., -1.,  2.], dtype=torch.float16)
    >>> a_packed = pack_fp8_tensor(a, mode="E5M2").to("cuda")
    >>> b_packed = pack_fp8_tensor(b, mode="E5M2").to("cuda")
    >>>
    >>> gemv = gemv_fp8_e5m2_acc_fp32_gpu(a_packed, b_packed, a.shape)
    >>> tensor([  6., -16.,  -3.,   9.], device='cuda:0')
    """
    out = torch.empty((m_shape[0],), dtype=torch.float32).to("cuda")

    BLOCK_SIZE = 1024
    grid = (m_shape[0],)

    _gemv_fp8_e5m2_acc_fp32_kernel[grid](
        m, v, out, m_shape[1] // 4, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def gemv_fp8_e4m3_acc_fp32_gpu(m: torch.Tensor, v: torch.Tensor, m_shape: tuple):
    """
    Performs `GEMV` subroutine on `FP8_E4M3` packed into `FP32` arrays. 
    Computation-wise should be equivalent to `torch.mv(m, v)`.

    Parameters
    ----------
    m : torch.Tensor
        Matrix tensor (packed format).
    v : torch.Tensor
        Vector tensor (packed format).
    m_shape : tuple
        Shape of the original matrix before packing. Division by 4 will be
        performed internally by this function.
        
    Returns
    -------
    torch.Tensor
        Output vector in FP32 format.
        
    Notes
    -----
    - Parameter `m_shape` must be shape of the original matrix (before packing), division by 4 will be performed by this function itself internally.
    - Input tensors must be packed using one of the functions exposed in `feather.packers.fp8` module, else computation is undefined.
    
    Examples
    --------
    >>> a = torch.randint(low=-3, high=3, size=(4, 4), dtype=torch.float16)
    >>> tensor([[ 1.,  1., -2.,  0.],
        [-3., -3., -2., -3.],
        [ 0.,  1.,  1., -2.],
        [ 2.,  0., -1.,  2.]], dtype=torch.float16)
    >>> b = torch.randint(low=-3, high=3, size=(4,), dtype=torch.float16)
    >>> tensor([ 2.,  2., -1.,  2.], dtype=torch.float16)
    >>> a_packed = pack_fp8_tensor(a, mode="E5M2").to("cuda")
    >>> b_packed = pack_fp8_tensor(b, mode="E5M2").to("cuda")
    >>>
    >>> gemv = gemv_fp8_e4m3_acc_fp32_gpu(a_packed, b_packed, a.shape)
    >>> tensor([  6., -16.,  -3.,   9.], device='cuda:0')
    """
    out = torch.empty((m_shape[0],), dtype=torch.float32).to("cuda")
    
    BLOCK_SIZE = 1024
    grid = (m_shape[0],)

    _gemv_fp8_e4m3_acc_fp32_kernel[grid](
        m, v, out, m_shape[1] // 4, BLOCK_SIZE=BLOCK_SIZE
    )
    return out