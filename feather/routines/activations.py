import torch
import triton

import triton.language as tl

from feather.packers.fp8 import *

@triton.jit
def _relu_1d_fp8_ret_fp32_kernel(
    _x,         
    _out,        
    _n: tl.constexpr,
    _blk_size: tl.constexpr
):
    _block_id = tl.program_id(axis=0)
    
    _x_ptrs = _block_id * _blk_size + tl.arange(start=0, end=_blk_size)
    _mask = _x_ptrs < _n
    _x_packed = tl.load(_x + _x_ptrs, mask=_mask)

    _x_a = tl.cast((_x_packed) & 0xFF, dtype=tl.uint16) << 8
    _x_b = tl.cast((_x_packed >> 8) & 0xFF, dtype=tl.uint16) << 8
    _x_c = tl.cast((_x_packed >> 16) & 0xFF, dtype=tl.uint16) << 8
    _x_d = tl.cast((_x_packed >> 24) & 0xFF, dtype=tl.uint16) << 8

    _out_base = _x_ptrs * 4
    tl.store(_out + _out_base + 0, tl.maximum(0.0, tl.cast(_x_a, dtype=tl.float16, bitcast=True).to(tl.float32)), mask=_mask)
    tl.store(_out + _out_base + 1, tl.maximum(0.0, tl.cast(_x_b, dtype=tl.float16, bitcast=True).to(tl.float32)), mask=_mask)
    tl.store(_out + _out_base + 2, tl.maximum(0.0, tl.cast(_x_c, dtype=tl.float16, bitcast=True).to(tl.float32)), mask=_mask)
    tl.store(_out + _out_base + 3, tl.maximum(0.0, tl.cast(_x_d, dtype=tl.float16, bitcast=True).to(tl.float32)), mask=_mask)


def relu1d_fp8_ret_fp32_feather_gpu(
    x: torch.Tensor,
    n: int
):
    out = torch.empty(size=(n, ), dtype=torch.float32).cuda()
    block_size = 8192
    grid = (triton.cdiv(x.shape[-1], block_size), )
    
    _relu_1d_fp8_ret_fp32_kernel[grid](x, out, x.shape[-1], block_size)
    return out
