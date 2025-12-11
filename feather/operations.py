import numpy as np
import logging
import numba
from packers import *

def add_fp16_acc_fp32(
    x: np.ndarray
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
        lo_val = lo_bits.view(np.float16).astype(np.float32)
        hi_val = hi_bits.view(np.float16).astype(np.float32)
        acc += lo_val + hi_val
    return acc

def dot_fp16_acc_fp32(
    x1: np.ndarray,
    x2: np.ndarray
):
    """
    helper function to perform dot product the packed `np.float16` compressed array
    into `np.float32`
    
    :param x1: input array 1
    :type x1: np.ndarray
    :param x2: input array 1
    :type x2: np.ndarray
    """
    acc = np.float32(0.0)
    for x1i, x2i in zip(x1, x2):
        x1i_bits = np.array([x1i], dtype=np.float32).view(np.uint32)[0]
        lo1_bits = np.uint16(x1i_bits & 0xFFFF)
        hi1_bits = np.uint16((x1i_bits >> 16) & 0xFFFF)
        lo1_val = lo1_bits.view(np.float16).astype(np.float32)
        hi1_val = hi1_bits.view(np.float16).astype(np.float32)

        x2i_bits = np.array([x2i], dtype=np.float32).view(np.uint32)[0]
        lo2_bits = np.uint16(x2i_bits & 0xFFFF)
        hi2_bits = np.uint16((x2i_bits >> 16) & 0xFFFF)
        lo2_val = lo2_bits.view(np.float16).astype(np.float32)
        hi2_val = hi2_bits.view(np.float16).astype(np.float32)
        acc += lo1_val * lo2_val + hi1_val * hi2_val
    return acc


def dot_fp16_acc_fp32_vec(x1, x2):
    # reinterpret float32 → uint32
    bits1 = x1.view(np.uint32)
    bits2 = x2.view(np.uint32)

    # extract packed halves
    lo1_bits = (bits1 & 0xFFFF).astype(np.uint16)
    hi1_bits = (bits1 >> 16).astype(np.uint16)

    lo2_bits = (bits2 & 0xFFFF).astype(np.uint16)
    hi2_bits = (bits2 >> 16).astype(np.uint16)

    # reinterpret halves as float16 and convert to float32
    lo1 = lo1_bits.view(np.float16).astype(np.float32)
    hi1 = hi1_bits.view(np.float16).astype(np.float32)

    lo2 = lo2_bits.view(np.float16).astype(np.float32)
    hi2 = hi2_bits.view(np.float16).astype(np.float32)

    # vectorized dot of both halves
    return np.sum(lo1 * lo2 + hi1 * hi2)



@numba.njit(parallel=True, fastmath=True)
def dot_fp16_acc_fp32_numba(x1_u32, x2_u32, lut):
    """
    Numba-accelerated dot product on packed FP16x2 arrays.

    x1_u32, x2_u32: uint32 arrays, each element packs two fp16 values:
        packed = (hi << 16) | lo
    lut: uint16 -> float32 lookup table for fp16 decoding
    """
    acc = 0.0
    n = x1_u32.shape[0]

    for i in numba.prange(n):
        u1 = x1_u32[i]
        u2 = x2_u32[i]

        # decode halves via LUT
        lo1 = lut[u1 & 0xFFFF]
        hi1 = lut[(u1 >> 16) & 0xFFFF]

        lo2 = lut[u2 & 0xFFFF]
        hi2 = lut[(u2 >> 16) & 0xFFFF]

        acc += lo1 * lo2 + hi1 * hi2

    return acc
