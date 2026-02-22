import numpy as np
import torch
import triton
import triton.language as tl

from feather.packers import *
from feather.routines.utils import (
    _unpack_e4m3_to_fp32,
    _unpack_e4m3_to_fp16,
    _pack_fp32_to_e4m3,
)


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
def _gemv_fp8_e5m2_out_packed_kernel(
    m: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    n_col: int,
    n_row: int,
    _scale_acc: torch.Tensor,
    _scale_w: torch.Tensor,
    tile_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_a = pid * 4
    row_b = pid * 4 + 1
    row_c = pid * 4 + 2
    row_d = pid * 4 + 3

    acc_a = tl.zeros((tile_size,), dtype=tl.float32)
    acc_b = tl.zeros((tile_size,), dtype=tl.float32)
    acc_c = tl.zeros((tile_size,), dtype=tl.float32)
    acc_d = tl.zeros((tile_size,), dtype=tl.float32)

    scale_w = tl.load(pointer=_scale_w)

    for _offset in range(0, n_col, tile_size):
        offsets = _offset + tl.arange(0, tile_size)
        mask = offsets < n_col

        v_packed = tl.load(pointer=v + offsets, mask=mask, other=0)
        ma_packed = tl.load(pointer=m + row_a * n_col + offsets, mask=mask, other=0)
        mb_packed = tl.load(pointer=m + row_b * n_col + offsets, mask=mask, other=0)
        mc_packed = tl.load(pointer=m + row_c * n_col + offsets, mask=mask, other=0)
        md_packed = tl.load(pointer=m + row_d * n_col + offsets, mask=mask, other=0)

        v_a = (
            (tl.cast((v_packed) & 0xFF, tl.uint16) << 8)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
        v_b = (
            (tl.cast((v_packed >> 8) & 0xFF, tl.uint16) << 8)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
        v_c = (
            (tl.cast((v_packed >> 16) & 0xFF, tl.uint16) << 8)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
        v_d = (
            (tl.cast((v_packed >> 24) & 0xFF, tl.uint16) << 8)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )

        acc_a += (tl.cast((ma_packed) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_a
        acc_a += (tl.cast((ma_packed >> 8) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_b
        acc_a += (tl.cast((ma_packed >> 16) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_c
        acc_a += (tl.cast((ma_packed >> 24) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_d

        acc_b += (tl.cast((mb_packed) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_a
        acc_b += (tl.cast((mb_packed >> 8) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_b
        acc_b += (tl.cast((mb_packed >> 16) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_c
        acc_b += (tl.cast((mb_packed >> 24) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_d

        acc_c += (tl.cast((mc_packed) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_a
        acc_c += (tl.cast((mc_packed >> 8) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_b
        acc_c += (tl.cast((mc_packed >> 16) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_c
        acc_c += (tl.cast((mc_packed >> 24) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_d

        acc_d += (tl.cast((md_packed) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_a
        acc_d += (tl.cast((md_packed >> 8) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_b
        acc_d += (tl.cast((md_packed >> 16) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_c
        acc_d += (tl.cast((md_packed >> 24) & 0xFF, tl.uint16) << 8).to(
            tl.float16, bitcast=True
        ).to(tl.float32) * v_d

    ra = tl.sum(acc_a * scale_w).to(tl.float16)
    rb = tl.sum(acc_b * scale_w).to(tl.float16)
    rc = tl.sum(acc_c * scale_w).to(tl.float16)
    rd = tl.sum(acc_d * scale_w).to(tl.float16)

    max_ab = tl.maximum(tl.abs(ra), tl.abs(rb))
    max_cd = tl.maximum(tl.abs(rc), tl.abs(rd))
    local_max = tl.maximum(max_ab, max_cd)

    tl.atomic_max(pointer=_scale_acc, val=local_max)

    ua = (ra.to(tl.uint16, bitcast=True) >> 8) & 0xFF
    ub = (rb.to(tl.uint16, bitcast=True) >> 8) & 0xFF
    uc = (rc.to(tl.uint16, bitcast=True) >> 8) & 0xFF
    ud = (rd.to(tl.uint16, bitcast=True) >> 8) & 0xFF

    packed = (
        ua.to(tl.int32)
        | (ub.to(tl.int32) << 8)
        | (uc.to(tl.int32) << 16)
        | (ud.to(tl.int32) << 24)
    )

    tl.store(out + pid, packed, mask=row_a < n_row)


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


@triton.jit
def _gemv_fp8_e4m3_out_packed_kernel(
    m: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    n_col: int,
    n_row: int,
    _scale_acc: torch.Tensor,
    _scale_w: torch.Tensor,
    tile_m: tl.constexpr,
    tile_k: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rm_packed = pid * tile_m + tl.arange(start=0, end=tile_m)

    row_a = rm_packed * 4 + 0
    row_b = rm_packed * 4 + 1
    row_c = rm_packed * 4 + 2
    row_d = rm_packed * 4 + 3

    acc_a = tl.zeros((tile_m, tile_k), dtype=tl.float32)
    acc_b = tl.zeros((tile_m, tile_k), dtype=tl.float32)
    acc_c = tl.zeros((tile_m, tile_k), dtype=tl.float32)
    acc_d = tl.zeros((tile_m, tile_k), dtype=tl.float32)

    scale_w = tl.load(pointer=_scale_w)
    rk = tl.arange(start=0, end=tile_k)

    for _offset in tl.range(arg1=0, arg2=n_col, step=tile_k):
        col_offsets = _offset + rk
        mask_k = col_offsets < n_col

        v_packed = tl.load(pointer=v + col_offsets, mask=mask_k, other=0)
        v_a = _unpack_e4m3_to_fp32(tl.cast((v_packed) & 0xFF, tl.uint16))
        v_b = _unpack_e4m3_to_fp32(tl.cast((v_packed >> 8) & 0xFF, tl.uint16))
        v_c = _unpack_e4m3_to_fp32(tl.cast((v_packed >> 16) & 0xFF, tl.uint16))
        v_d = _unpack_e4m3_to_fp32(tl.cast((v_packed >> 24) & 0xFF, tl.uint16))

        mask_ma = (row_a[:, None] < n_row) & mask_k[None, :]
        mask_mb = (row_b[:, None] < n_row) & mask_k[None, :]
        mask_mc = (row_c[:, None] < n_row) & mask_k[None, :]
        mask_md = (row_d[:, None] < n_row) & mask_k[None, :]

        ma_packed = tl.load(
            pointer=m + row_a[:, None] * n_col + col_offsets[None, :],
            mask=mask_ma,
            other=0,
        )
        mb_packed = tl.load(
            pointer=m + row_b[:, None] * n_col + col_offsets[None, :],
            mask=mask_mb,
            other=0,
        )
        mc_packed = tl.load(
            pointer=m + row_c[:, None] * n_col + col_offsets[None, :],
            mask=mask_mc,
            other=0,
        )
        md_packed = tl.load(
            pointer=m + row_d[:, None] * n_col + col_offsets[None, :],
            mask=mask_md,
            other=0,
        )

        # Vectorized Multiply and Accumulate (Notice: NO tl.sum here!)
        acc_a += (
            _unpack_e4m3_to_fp32(tl.cast((ma_packed) & 0xFF, tl.uint16)) * v_a[None, :]
        )
        acc_a += (
            _unpack_e4m3_to_fp32(tl.cast((ma_packed >> 8) & 0xFF, tl.uint16))
            * v_b[None, :]
        )
        acc_a += (
            _unpack_e4m3_to_fp32(tl.cast((ma_packed >> 16) & 0xFF, tl.uint16))
            * v_c[None, :]
        )
        acc_a += (
            _unpack_e4m3_to_fp32(tl.cast((ma_packed >> 24) & 0xFF, tl.uint16))
            * v_d[None, :]
        )

        acc_b += (
            _unpack_e4m3_to_fp32(tl.cast((mb_packed) & 0xFF, tl.uint16)) * v_a[None, :]
        )
        acc_b += (
            _unpack_e4m3_to_fp32(tl.cast((mb_packed >> 8) & 0xFF, tl.uint16))
            * v_b[None, :]
        )
        acc_b += (
            _unpack_e4m3_to_fp32(tl.cast((mb_packed >> 16) & 0xFF, tl.uint16))
            * v_c[None, :]
        )
        acc_b += (
            _unpack_e4m3_to_fp32(tl.cast((mb_packed >> 24) & 0xFF, tl.uint16))
            * v_d[None, :]
        )

        acc_c += (
            _unpack_e4m3_to_fp32(tl.cast((mc_packed) & 0xFF, tl.uint16)) * v_a[None, :]
        )
        acc_c += (
            _unpack_e4m3_to_fp32(tl.cast((mc_packed >> 8) & 0xFF, tl.uint16))
            * v_b[None, :]
        )
        acc_c += (
            _unpack_e4m3_to_fp32(tl.cast((mc_packed >> 16) & 0xFF, tl.uint16))
            * v_c[None, :]
        )
        acc_c += (
            _unpack_e4m3_to_fp32(tl.cast((mc_packed >> 24) & 0xFF, tl.uint16))
            * v_d[None, :]
        )

        acc_d += (
            _unpack_e4m3_to_fp32(tl.cast((md_packed) & 0xFF, tl.uint16)) * v_a[None, :]
        )
        acc_d += (
            _unpack_e4m3_to_fp32(tl.cast((md_packed >> 8) & 0xFF, tl.uint16))
            * v_b[None, :]
        )
        acc_d += (
            _unpack_e4m3_to_fp32(tl.cast((md_packed >> 16) & 0xFF, tl.uint16))
            * v_c[None, :]
        )
        acc_d += (
            _unpack_e4m3_to_fp32(tl.cast((md_packed >> 24) & 0xFF, tl.uint16))
            * v_d[None, :]
        )

    # NOW we reduce outside the loop
    ra = tl.sum(acc_a, axis=1) * scale_w
    rb = tl.sum(acc_b, axis=1) * scale_w
    rc = tl.sum(acc_c, axis=1) * scale_w
    rd = tl.sum(acc_d, axis=1) * scale_w

    max_ab = tl.maximum(tl.abs(ra), tl.abs(rb))
    max_cd = tl.maximum(tl.abs(rc), tl.abs(rd))
    local_max = tl.maximum(max_ab, max_cd)

    tl.atomic_max(pointer=_scale_acc, val=tl.max(local_max))

    ua = _pack_fp32_to_e4m3(ra) & 0xFF
    ub = _pack_fp32_to_e4m3(rb) & 0xFF
    uc = _pack_fp32_to_e4m3(rc) & 0xFF
    ud = _pack_fp32_to_e4m3(rd) & 0xFF

    packed = (
        ua.to(tl.int32)
        | (ub.to(tl.int32) << 8)
        | (uc.to(tl.int32) << 16)
        | (ud.to(tl.int32) << 24)
    )

    tl.store(out + rm_packed, packed, mask=rm_packed < (n_row // 4))


# @triton.jit
# def _gemv_fp8_e4m3_out_packed_kernel(
#     m: torch.Tensor,
#     v: torch.Tensor,
#     out: torch.Tensor,
#     n_col: int,
#     n_row: int,
#     _scale_acc: torch.Tensor,
#     _scale_w: torch.Tensor,
#     tile_size: tl.constexpr,
# ):
#     pid = tl.program_id(axis=0)
#     row_a = pid * 4
#     row_b = pid * 4 + 1
#     row_c = pid * 4 + 2
#     row_d = pid * 4 + 3

#     acc_a = tl.zeros((tile_size,), dtype=tl.float32)
#     acc_b = tl.zeros((tile_size,), dtype=tl.float32)
#     acc_c = tl.zeros((tile_size,), dtype=tl.float32)
#     acc_d = tl.zeros((tile_size,), dtype=tl.float32)

#     scale_w = tl.load(pointer=_scale_w)

#     for _offset in range(0, n_col, tile_size):
#         offsets = _offset + tl.arange(0, tile_size)
#         mask = offsets < n_col

#         v_packed = tl.load(pointer=v + offsets, mask=mask, other=0)
#         ma_packed = tl.load(pointer=m + row_a * n_col + offsets, mask=mask, other=0)
#         mb_packed = tl.load(pointer=m + row_b * n_col + offsets, mask=mask, other=0)
#         mc_packed = tl.load(pointer=m + row_c * n_col + offsets, mask=mask, other=0)
#         md_packed = tl.load(pointer=m + row_d * n_col + offsets, mask=mask, other=0)

#         v_a = _unpack_e4m3_to_fp32(tl.cast((v_packed) & 0xFF, tl.uint16))
#         v_b = _unpack_e4m3_to_fp32(tl.cast((v_packed >> 8) & 0xFF, tl.uint16))
#         v_c = _unpack_e4m3_to_fp32(tl.cast((v_packed >> 16) & 0xFF, tl.uint16))
#         v_d = _unpack_e4m3_to_fp32(tl.cast((v_packed >> 24) & 0xFF, tl.uint16))

#         acc_a += _unpack_e4m3_to_fp32(tl.cast((ma_packed) & 0xFF, tl.uint16)) * v_a
#         acc_a += _unpack_e4m3_to_fp32(tl.cast((ma_packed >> 8) & 0xFF, tl.uint16)) * v_b
#         acc_a += (
#             _unpack_e4m3_to_fp32(tl.cast((ma_packed >> 16) & 0xFF, tl.uint16)) * v_c
#         )
#         acc_a += (
#             _unpack_e4m3_to_fp32(tl.cast((ma_packed >> 24) & 0xFF, tl.uint16)) * v_d
#         )

#         acc_b += _unpack_e4m3_to_fp32(tl.cast((mb_packed) & 0xFF, tl.uint16)) * v_a
#         acc_b += _unpack_e4m3_to_fp32(tl.cast((mb_packed >> 8) & 0xFF, tl.uint16)) * v_b
#         acc_b += (
#             _unpack_e4m3_to_fp32(tl.cast((mb_packed >> 16) & 0xFF, tl.uint16)) * v_c
#         )
#         acc_b += (
#             _unpack_e4m3_to_fp32(tl.cast((mb_packed >> 24) & 0xFF, tl.uint16)) * v_d
#         )

#         acc_c += _unpack_e4m3_to_fp32(tl.cast((mc_packed) & 0xFF, tl.uint16)) * v_a
#         acc_c += _unpack_e4m3_to_fp32(tl.cast((mc_packed >> 8) & 0xFF, tl.uint16)) * v_b
#         acc_c += (
#             _unpack_e4m3_to_fp32(tl.cast((mc_packed >> 16) & 0xFF, tl.uint16)) * v_c
#         )
#         acc_c += (
#             _unpack_e4m3_to_fp32(tl.cast((mc_packed >> 24) & 0xFF, tl.uint16)) * v_d
#         )

#         acc_d += _unpack_e4m3_to_fp32(tl.cast((md_packed) & 0xFF, tl.uint16)) * v_a
#         acc_d += _unpack_e4m3_to_fp32(tl.cast((md_packed >> 8) & 0xFF, tl.uint16)) * v_b
#         acc_d += (
#             _unpack_e4m3_to_fp32(tl.cast((md_packed >> 16) & 0xFF, tl.uint16)) * v_c
#         )
#         acc_d += (
#             _unpack_e4m3_to_fp32(tl.cast((md_packed >> 24) & 0xFF, tl.uint16)) * v_d
#         )

#     ra = tl.sum(acc_a * scale_w)
#     rb = tl.sum(acc_b * scale_w)
#     rc = tl.sum(acc_c * scale_w)
#     rd = tl.sum(acc_d * scale_w)

#     max_ab = tl.maximum(tl.abs(ra), tl.abs(rb))
#     max_cd = tl.maximum(tl.abs(rc), tl.abs(rd))
#     local_max = tl.maximum(max_ab, max_cd)

#     tl.atomic_max(pointer=_scale_acc, val=local_max)

#     ua = _pack_fp32_to_e4m3(ra) & 0xFF
#     ub = _pack_fp32_to_e4m3(rb) & 0xFF
#     uc = _pack_fp32_to_e4m3(rc) & 0xFF
#     ud = _pack_fp32_to_e4m3(rd) & 0xFF

#     packed = (
#         ua.to(tl.int32)
#         | (ub.to(tl.int32) << 8)
#         | (uc.to(tl.int32) << 16)
#         | (ud.to(tl.int32) << 24)
#     )

#     tl.store(out + pid, packed, mask=row_a < n_row)


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


# def gemv_fp8_e4m3_out_packed_gpu(
#     m: torch.Tensor,
#     v: torch.Tensor,
#     m_shape: tuple,
#     scale_acc: torch.Tensor,
#     scale_w: torch.Tensor,
#     out: torch.Tensor
# ) -> torch.Tensor:
#     n_row = m_shape[0]
#     n_col = m_shape[1] // 4
#     tile_size = 1024
#     grid = (n_row // 4,)
#     _gemv_fp8_e4m3_out_packed_kernel[grid](
#         m=m,
#         v=v,
#         out=out,
#         n_col=n_col,
#         n_row=n_row,
#         _scale_acc=scale_acc,
#         _scale_w=scale_w,
#         tile_size=tile_size,
#     )
#     return out


def gemv_fp8_e4m3_out_packed_gpu(
    m: torch.Tensor,
    v: torch.Tensor,
    m_shape: tuple,
    scale_acc: torch.Tensor,
    scale_w: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    n_row = m_shape[0]
    n_col = m_shape[1] // 4
    tile_m = 16
    tile_k = 128

    grid = (triton.cdiv(n_row // 4, tile_m),)
    _gemv_fp8_e4m3_out_packed_kernel[grid](
        m=m,
        v=v,
        out=out,
        n_col=n_col,
        n_row=n_row,
        _scale_acc=scale_acc,
        _scale_w=scale_w,
        tile_m=tile_m,
        tile_k=tile_k,
    )
    return out
