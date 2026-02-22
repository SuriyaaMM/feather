import triton
import triton.language as tl
import torch


# @triton.jit
# def _unpack_e4m3_to_fp16(x_u8):
#     """
#     conversion:
#     x_u8 is tl.uint16, refer to `_gemv_fp8_e4m3_acc_fp32_kernel`

#     first starting off with the mantissa correction `(x_u8 & 0x07)`
#     extracts mantissa from `E4M3` then `<< 7` shifts it to upper bits of
#     `FP16`

#     next is bias correction, `FP16` bias = 15, `E4M3` bias = 7,
#     so we add 8 to after extracting the exponent, then place it right
#     position (10 bits after)

#     last part is straightforward, just extract the sign bit
#     then mash everything together using bitwise or
#     """
#     x_u16 = (
#         ((x_u8 & 0x80) << 8)
#         | (((((x_u8 & 0x78) >> 3) + 8) * (((x_u8 & 0x78) >> 3) > 0)) << 10)
#         | ((x_u8 & 0x07) << 7)
#     )
#     return x_u16.to(tl.float16, bitcast=True)


@triton.jit
def _unpack_e4m3_to_fp32(x_u8):
    exp = tl.cast((x_u8 & 0x78) >> 3, tl.uint32)

    exp_norm = (exp + 120) << 23
    norm_u32 = (
        tl.cast((x_u8 & 0x80), tl.uint32) << 24
        | exp_norm
        | (tl.cast(x_u8 & 0x07, tl.uint32) << 20)
    )
    norm_f32 = norm_u32.to(tl.float32, bitcast=True)

    # 2e-9
    sub_f32 = (tl.cast(x_u8 & 0x07, tl.uint32)).to(tl.float32) * 0.001953125
    sub_f32 = tl.where(condition=(x_u8 & 0x80) != 0, x=-sub_f32, y=sub_f32)

    return tl.where(condition=exp > 0, x=norm_f32, y=sub_f32)


@triton.jit
def _unpack_e4m3_to_fp16(x_u8):
    return _unpack_e4m3_to_fp32(x_u8).to(tl.float16)


@triton.jit
def _pack_fp32_to_e4m3(x_f32):
    x_u32 = x_f32.to(tl.uint32, bitcast=True)
    sign = (x_u32 & 0x80000000) >> 24
    abs_f32 = tl.abs(x_f32)

    x_u32_rounded = x_u32 + (1 << 19)
    exp = (x_u32_rounded & 0x7F800000) >> 23
    mant = x_u32_rounded & 0x007FFFFF

    exp_e4m3 = tl.cast(exp, tl.int32) - 120
    mant_e4m3 = mant >> 20

    sub_mant = tl.cast(abs_f32 * 512.0 + 0.5, tl.uint32)
    is_subnormal = abs_f32 < 0.015625

    exp_e4m3 = tl.maximum(exp_e4m3, 1)
    exp_e4m3 = tl.minimum(exp_e4m3, 15)

    norm_val = sign | (exp_e4m3 << 3) | mant_e4m3
    sub_val = sign | sub_mant

    res = tl.where(is_subnormal, sub_val, norm_val)
    # 0.5 * 2e-9
    res = tl.where(abs_f32 < 0.0009765625, sign, res)

    return tl.cast(res, tl.uint16)


@triton.jit
def _pack_tensor_kernel(
    x: torch.Tensor,
    out: torch.Tensor,
    n: tl.constexpr,
    tile_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    offsets = pid * tile_size + tl.arange(0, tile_size)
    mask = offsets < (n // 4)

    idx_a = offsets * 4 + 0
    idx_b = offsets * 4 + 1
    idx_c = offsets * 4 + 2
    idx_d = offsets * 4 + 3

    val_a = tl.load(pointer=x + idx_a, mask=mask, other=0.0).to(tl.float32)
    val_b = tl.load(pointer=x + idx_b, mask=mask, other=0.0).to(tl.float32)
    val_c = tl.load(pointer=x + idx_c, mask=mask, other=0.0).to(tl.float32)
    val_d = tl.load(pointer=x + idx_d, mask=mask, other=0.0).to(tl.float32)

    u_a = _pack_fp32_to_e4m3(val_a) & 0xFF
    u_b = _pack_fp32_to_e4m3(val_b) & 0xFF
    u_c = _pack_fp32_to_e4m3(val_c) & 0xFF
    u_d = _pack_fp32_to_e4m3(val_d) & 0xFF

    packed = (
        u_a.to(tl.int32)
        | (u_b.to(tl.int32) << 8)
        | (u_c.to(tl.int32) << 16)
        | (u_d.to(tl.int32) << 24)
    )

    tl.store(pointer=out + offsets, value=packed, mask=mask)


def pack_tensor_gpu(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    tile_size = 1024
    grid = (triton.cdiv(n // 4, tile_size),)
    _pack_tensor_kernel[grid](x=x, out=out, n=n, tile_size=tile_size)
    return out
