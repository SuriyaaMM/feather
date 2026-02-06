import triton
import triton.language as tl


@triton.jit
def _unpack_e4m3_to_fp16(x_u8):
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
    return x_u16.to(tl.float16, bitcast=True)
