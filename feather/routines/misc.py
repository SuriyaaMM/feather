import torch
import triton
import triton.language as tl
from typing import Tuple

from feather.routines.utils import (
    _unpack_e4m3_to_fp16,
    _pack_fp32_to_e4m3,
    _unpack_e4m3_to_fp32,
)
from feather.packers.fp8 import *


@triton.jit
def _rms_norm_accumulator_fp8_e4m3_acc_fp32_kernel(
    _x: torch.Tensor,
    _norm: torch.Tensor,
    _n: int,
    _scale,
    _tile: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * _tile + tl.arange(start=0, end=_tile)
    mask = offsets < _n

    x_tile = tl.load(pointer=_x + offsets, mask=mask, other=0)

    x_a = _unpack_e4m3_to_fp32(tl.cast((x_tile) & 0xFF, tl.uint16))
    x_b = _unpack_e4m3_to_fp32(tl.cast((x_tile >> 8) & 0xFF, tl.uint16))
    x_c = _unpack_e4m3_to_fp32(tl.cast((x_tile >> 16) & 0xFF, tl.uint16))
    x_d = _unpack_e4m3_to_fp32(tl.cast((x_tile >> 24) & 0xFF, tl.uint16))

    block_sum = tl.sum(x_a * x_a + x_b * x_b + x_c * x_c + x_d * x_d)
    tl.atomic_add(pointer=_norm, val=block_sum)


@triton.jit
def _rms_norm_fp8_e4m3_acc_fp32_out_packed_kernel(
    _x: torch.Tensor,
    _w: torch.Tensor,
    _out: torch.Tensor,
    _norm: torch.Tensor,
    _n: int,
    _scale,
    _eps: tl.constexpr,
    _tile: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * _tile + tl.arange(start=0, end=_tile)
    mask = offsets < _n

    norm = tl.load(pointer=_norm)
    norm = norm / tl.cast(_n * 4, tl.float32)
    norm = tl.rsqrt(norm + _eps)

    scale = tl.load(pointer=_scale)
    x_tile = tl.load(pointer=_x + offsets, mask=mask, other=0)
    w_tile = tl.load(pointer=_w + offsets, mask=mask, other=0)

    x_a = _unpack_e4m3_to_fp32(tl.cast((x_tile) & 0xFF, tl.uint16))
    x_b = _unpack_e4m3_to_fp32(tl.cast((x_tile >> 8) & 0xFF, tl.uint16))
    x_c = _unpack_e4m3_to_fp32(tl.cast((x_tile >> 16) & 0xFF, tl.uint16))
    x_d = _unpack_e4m3_to_fp32(tl.cast((x_tile >> 24) & 0xFF, tl.uint16))

    w_a = _unpack_e4m3_to_fp32(tl.cast((w_tile) & 0xFF, tl.uint16))
    w_b = _unpack_e4m3_to_fp32(tl.cast((w_tile >> 8) & 0xFF, tl.uint16))
    w_c = _unpack_e4m3_to_fp32(tl.cast((w_tile >> 16) & 0xFF, tl.uint16))
    w_d = _unpack_e4m3_to_fp32(tl.cast((w_tile >> 24) & 0xFF, tl.uint16))

    ya = x_a * norm * w_a * scale
    yb = x_b * norm * w_b * scale
    yc = x_c * norm * w_c * scale
    yd = x_d * norm * w_d * scale

    ya_u8 = _pack_fp32_to_e4m3(ya)
    yb_u8 = _pack_fp32_to_e4m3(yb)
    yc_u8 = _pack_fp32_to_e4m3(yc)
    yd_u8 = _pack_fp32_to_e4m3(yd)

    packed_out = (
        ya_u8.to(tl.int32)
        | (yb_u8.to(tl.int32) << 8)
        | (yc_u8.to(tl.int32) << 16)
        | (yd_u8.to(tl.int32) << 24)
    )
    tl.store(pointer=_out + offsets, value=packed_out, mask=mask)


def rms_norm_fp8_e4m3_out_packed_gpu(
    x: torch.Tensor,
    w: torch.Tensor,
    n: int,
    scale: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    norm: torch.Tensor,
):
    scale = scale.to(torch.float32)
    tile = 1024
    norm.zero_()

    _rms_norm_accumulator_fp8_e4m3_acc_fp32_kernel[(triton.cdiv(n, tile),)](
        _x=x, _norm=norm, _n=n, _scale=scale, _tile=tile
    )
    _rms_norm_fp8_e4m3_acc_fp32_out_packed_kernel[(triton.cdiv(n, tile),)](
        _x=x, _w=w, _out=out, _norm=norm, _n=n, _eps=eps, _scale=scale, _tile=tile
    )

    return out


@triton.jit
def _swiglu_fp8_e4m3_acc_fp32_packed_kernel(
    _gate: torch.Tensor,
    _up: torch.Tensor,
    _out: torch.Tensor,
    _n: int,
    _tile_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * _tile_size + tl.arange(start=0, end=_tile_size)
    mask = offsets < _n

    gate_tile = tl.load(pointer=_gate + offsets, mask=mask, other=0)
    up_tile = tl.load(pointer=_up + offsets, mask=mask, other=0)

    g_a = _unpack_e4m3_to_fp16(tl.cast((gate_tile) & 0xFF, tl.uint16)).to(tl.float32)
    g_b = _unpack_e4m3_to_fp16(tl.cast((gate_tile >> 8) & 0xFF, tl.uint16)).to(
        tl.float32
    )
    g_c = _unpack_e4m3_to_fp16(tl.cast((gate_tile >> 16) & 0xFF, tl.uint16)).to(
        tl.float32
    )
    g_d = _unpack_e4m3_to_fp16(tl.cast((gate_tile >> 24) & 0xFF, tl.uint16)).to(
        tl.float32
    )

    u_a = _unpack_e4m3_to_fp16(tl.cast((up_tile) & 0xFF, tl.uint16)).to(tl.float32)
    u_b = _unpack_e4m3_to_fp16(tl.cast((up_tile >> 8) & 0xFF, tl.uint16)).to(tl.float32)
    u_c = _unpack_e4m3_to_fp16(tl.cast((up_tile >> 16) & 0xFF, tl.uint16)).to(
        tl.float32
    )
    u_d = _unpack_e4m3_to_fp16(tl.cast((up_tile >> 24) & 0xFF, tl.uint16)).to(
        tl.float32
    )

    ya = (g_a / (1.0 + tl.exp(-g_a))) * u_a
    yb = (g_b / (1.0 + tl.exp(-g_b))) * u_b
    yc = (g_c / (1.0 + tl.exp(-g_c))) * u_c
    yd = (g_d / (1.0 + tl.exp(-g_d))) * u_d

    ya_u8 = _pack_fp32_to_e4m3(ya)
    yb_u8 = _pack_fp32_to_e4m3(yb)
    yc_u8 = _pack_fp32_to_e4m3(yc)
    yd_u8 = _pack_fp32_to_e4m3(yd)

    packed_out = (
        ya_u8.to(tl.int32)
        | (yb_u8.to(tl.int32) << 8)
        | (yc_u8.to(tl.int32) << 16)
        | (yd_u8.to(tl.int32) << 24)
    )

    tl.store(pointer=_out + offsets, value=packed_out, mask=mask)


def swiglu_fp8_e4m3_packed_gpu(
    gate: torch.Tensor, up: torch.Tensor, n: int
) -> torch.Tensor:
    out = torch.empty(n, dtype=torch.int32, device="cuda")
    tile_size = 1024
    _swiglu_fp8_e4m3_acc_fp32_packed_kernel[(triton.cdiv(n, tile_size),)](
        _gate=gate, _up=up, _out=out, _n=n, _tile_size=tile_size
    )
    return out


@triton.jit
def _rope_fp8_e4m3_inplace_kernel(
    _x: torch.Tensor,
    _cos: torch.Tensor,
    _sin: torch.Tensor,
    _head_dim_packed: tl.constexpr,
    _tile_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row = pid * _head_dim_packed

    offset = tl.arange(start=0, end=_tile_size)
    mask = offset < _head_dim_packed
    half_tile = _head_dim_packed // 2

    x_tile = tl.load(pointer=_x + row + offset, mask=mask, other=0)
    x_a = _unpack_e4m3_to_fp16(tl.cast((x_tile) & 0xFF, tl.uint16))
    x_b = _unpack_e4m3_to_fp16(tl.cast((x_tile >> 8) & 0xFF, tl.uint16))
    x_c = _unpack_e4m3_to_fp16(tl.cast((x_tile >> 16) & 0xFF, tl.uint16))
    x_d = _unpack_e4m3_to_fp16(tl.cast((x_tile >> 24) & 0xFF, tl.uint16))

    partner = tl.where(
        condition=offset < half_tile, x=offset + half_tile, y=offset - half_tile
    )
    sign = tl.where(
        condition=offset < half_tile,
        x=tl.full(shape=(_tile_size,), value=-1.0, dtype=tl.float32),
        y=tl.full(shape=(_tile_size,), value=1.0, dtype=tl.float32),
    )

    rh_tile = tl.load(pointer=_x + row + partner, mask=mask, other=0)
    rh_a = _unpack_e4m3_to_fp16(tl.cast((rh_tile) & 0xFF, tl.uint16)) * sign
    rh_b = _unpack_e4m3_to_fp16(tl.cast((rh_tile >> 8) & 0xFF, tl.uint16)) * sign
    rh_c = _unpack_e4m3_to_fp16(tl.cast((rh_tile >> 16) & 0xFF, tl.uint16)) * sign
    rh_d = _unpack_e4m3_to_fp16(tl.cast((rh_tile >> 24) & 0xFF, tl.uint16)) * sign

    base = offset * 4
    cos_a = tl.load(pointer=_cos + base + 0, mask=mask).to(tl.float32)
    cos_b = tl.load(pointer=_cos + base + 1, mask=mask).to(tl.float32)
    cos_c = tl.load(pointer=_cos + base + 2, mask=mask).to(tl.float32)
    cos_d = tl.load(pointer=_cos + base + 3, mask=mask).to(tl.float32)

    sin_a = tl.load(pointer=_sin + base + 0, mask=mask).to(tl.float32)
    sin_b = tl.load(pointer=_sin + base + 1, mask=mask).to(tl.float32)
    sin_c = tl.load(pointer=_sin + base + 2, mask=mask).to(tl.float32)
    sin_d = tl.load(pointer=_sin + base + 3, mask=mask).to(tl.float32)

    ya = x_a * cos_a + rh_a * sin_a
    yb = x_b * cos_b + rh_b * sin_b
    yc = x_c * cos_c + rh_c * sin_c
    yd = x_d * cos_d + rh_d * sin_d

    ya_u8 = _pack_fp32_to_e4m3(ya)
    yb_u8 = _pack_fp32_to_e4m3(yb)
    yc_u8 = _pack_fp32_to_e4m3(yc)
    yd_u8 = _pack_fp32_to_e4m3(yd)

    packed_out = (
        ya_u8.to(tl.int32)
        | (yb_u8.to(tl.int32) << 8)
        | (yc_u8.to(tl.int32) << 16)
        | (yd_u8.to(tl.int32) << 24)
    )

    tl.store(pointer=_x + row + offset, value=packed_out, mask=mask)


def rope_fp8_e4m3_inplace_gpu(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim_packed: int,
):
    h = x.shape[0]
    tile_size = triton.next_power_of_2(head_dim_packed)
    _rope_fp8_e4m3_inplace_kernel[(h,)](
        _x=x,
        _cos=cos.contiguous(),
        _sin=sin.contiguous(),
        _head_dim_packed=head_dim_packed,
        _tile_size=tile_size,
    )


@triton.jit
def _fused_add_e4m3_acc_fp32_dual_out_kernel(
    _a_original: torch.Tensor,
    _b_packed: torch.Tensor,
    _b_packed_scale: torch.Tensor,  # unsued as of now
    _out_original: torch.Tensor,
    _out_packed: torch.Tensor,
    _n: int,
    _tile_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * _tile_size + tl.arange(start=0, end=_tile_size)
    mask = offsets < (_n // 4)

    offsets_a = offsets * 4 + 0
    offsets_b = offsets * 4 + 1
    offsets_c = offsets * 4 + 2
    offsets_d = offsets * 4 + 3

    b_packed_tile = tl.load(pointer=_b_packed + offsets, mask=mask, other=0)
    b_a = _unpack_e4m3_to_fp16(tl.cast((b_packed_tile) & 0xFF, tl.uint16))
    b_b = _unpack_e4m3_to_fp16(tl.cast((b_packed_tile >> 8) & 0xFF, tl.uint16))
    b_c = _unpack_e4m3_to_fp16(tl.cast((b_packed_tile >> 16) & 0xFF, tl.uint16))
    b_d = _unpack_e4m3_to_fp16(tl.cast((b_packed_tile >> 24) & 0xFF, tl.uint16))

    # note: a here is expected in FP16
    a_a = tl.load(pointer=_a_original + offsets_a, mask=offsets_a < _n, other=0.0).to(
        tl.float32
    )
    a_b = tl.load(pointer=_a_original + offsets_b, mask=offsets_b < _n, other=0.0).to(
        tl.float32
    )
    a_c = tl.load(pointer=_a_original + offsets_c, mask=offsets_c < _n, other=0.0).to(
        tl.float32
    )
    a_d = tl.load(pointer=_a_original + offsets_d, mask=offsets_d < _n, other=0.0).to(
        tl.float32
    )

    out_a = a_a + b_a
    out_b = a_b + b_b
    out_c = a_c + b_c
    out_d = a_d + b_d

    # note: original values are stored
    tl.store(
        pointer=_out_original + offsets_a,
        value=out_a.to(tl.float16),
        mask=offsets_a < _n,
    )
    tl.store(
        pointer=_out_original + offsets_b,
        value=out_b.to(tl.float16),
        mask=offsets_b < _n,
    )
    tl.store(
        pointer=_out_original + offsets_c,
        value=out_c.to(tl.float16),
        mask=offsets_c < _n,
    )
    tl.store(
        pointer=_out_original + offsets_d,
        value=out_d.to(tl.float16),
        mask=offsets_d < _n,
    )

    # packed output
    out_packed_a = _pack_fp32_to_e4m3(out_a)
    out_packed_b = _pack_fp32_to_e4m3(out_b)
    out_packed_c = _pack_fp32_to_e4m3(out_c)
    out_packed_d = _pack_fp32_to_e4m3(out_d)

    out_packed = (
        out_packed_a.to(tl.int32)
        | (out_packed_b.to(tl.int32) << 8)
        | (out_packed_c.to(tl.int32) << 16)
        | (out_packed_d.to(tl.int32) << 24)
    )
    tl.store(pointer=_out_packed + offsets, value=out_packed, mask=mask)


def fused_add_e4m3_acc_fp32_dual_out_gpu(
    a_original: torch.Tensor, b_packed: torch.Tensor, b_packed_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Addition (accumulate & scale), outputs an dual tensor (original, packed).
    Although unnecessary to store the original vector, it helps in debugging
    """

    n = a_original.numel()
    tile_size = 1024

    grid = (triton.cdiv(n // 4, tile_size),)

    out_original = torch.empty_like(input=a_original)
    out_packed = torch.empty_like(input=b_packed)

    _fused_add_e4m3_acc_fp32_dual_out_kernel[grid](
        _a_original=a_original,
        _b_packed=b_packed,
        _b_packed_scale=b_packed_scale,
        _out_original=out_original,
        _out_packed=out_packed,
        _n=n,
        _tile_size=tile_size,
    )

    return out_original, out_packed
