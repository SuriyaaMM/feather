import numpy as np
import logging
import numba
from feather.packers import *

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
        lo_val = lo_bits.view(np.float16)[0]
        hi_val = hi_bits.view(np.float16)[0]
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
        lo1_val = lo1_bits.view(np.float16)
        hi1_val = hi1_bits.view(np.float16)

        x2i_bits = np.array([x2i], dtype=np.float32).view(np.uint32)[0]
        lo2_bits = np.uint16(x2i_bits & 0xFFFF)
        hi2_bits = np.uint16((x2i_bits >> 16) & 0xFFFF)
        lo2_val = lo2_bits.view(np.float16)
        hi2_val = hi2_bits.view(np.float16)
        acc += lo1_val * lo2_val + hi1_val * hi2_val
    return acc


def dot_fp16_acc_fp32_vec(
    x1: np.ndarray[np.dtype[np.float16]], 
    x2: np.ndarray[np.dtype[np.float16]]
):
    """
    performs dot product
    
    :param x1: input array 1 `FP16`
    :type x1: np.ndarray[np.dtype[np.float16]]
    :param x2: input array 2 `FP16`
    :type x2: np.ndarray[np.dtype[np.float16]]
    """
    # unpack
    x1_lower, x1_upper = unpack_fp32_into_fp16(x1)
    x2_lower, x2_upper = unpack_fp32_into_fp16(x2)
    
    return np.sum(x1_lower * x2_lower + x1_upper * x2_upper)


@numba.njit(parallel=True, fastmath=True)
def dot_fp16_acc_fp32_numba(
    x1_u32: np.ndarray, 
    x2_u32: np.ndarray, 
    lut: np.ndarray
):
    """
    Numba-accelerated dot product on packed `FP32` arrays packed using `FP16`

    :param x1_u32: uint32 array, each element packs two fp16 values
    :param x2_u32: uint32 array, each element packs two fp16 values
    :param lut: uint16 -> float32 lookup table for fp16 decoding
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
