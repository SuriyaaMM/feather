import numpy as np 
import logging

def pack_fp16_into_fp32(
    a: np.ndarray[np.dtype[np.float16]], 
    b: np.ndarray[np.dtype[np.float16]]
) -> np.ndarray[np.dtype[np.float32]]:
    """
    packs two `FP16` into `FP32`
    
    :param a: first number `FP16`
    :type a: np.ndarray
    :param b: second number `FP16`
    :type b: np.ndarray
    """
    if a.dtype != np.float16 or b.dtype != np.float16:
        logging.warning(f"PARAM a ({a}) or b ({b}) was not float16, routine is undefined!")
    if len(a.shape) > 1 or len(b.shape) > 1:
        logging.warning(f"PARAM a ({a}) and b ({b}) had more than 1 element to pack")
    # view as UINT16
    a_bits = np.array([a], dtype=np.float16).view(np.uint16)[0]
    b_bits = np.array([b], dtype=np.float16).view(np.uint16)[0]
    # pack
    packed_bits = (np.uint32(b_bits) << 16) | np.uint32(a_bits)
    # convert into UINT32 and view as FP32
    return np.array([packed_bits], dtype=np.uint32).view(np.float32)[0]

def unpack_fp32_into_fp16(
    a: np.ndarray[np.dtype[np.float32]]
) -> np.ndarray[np.dtype[np.float16], np.dtype[np.float16]]:
    """
    unpacks 1 `FP32` into 2 `FP16`'s
    
    :param a: number to unpack `FP32`
    :type a: np.ndarray
    """
    if a.dtype != np.float32:
        logging.warning(f"PARAM a ({a}) was not float16, routine is undefined!")
    if len(a.shape) > 1:
        logging.warning(f"PARAM a ({a}) had more than 1 element to pack")
    # view as UINT32
    a_bits = np.array([a], dtype=np.float32).view(np.uint32)[0]
    a_lower = (a_bits & 0xFFFF)
    a_upper = (a_bits >> 16) & 0xFFFF  
    # unpack into UINT16 and view as FP16
    unpacked = np.array([a_lower, a_upper], dtype=np.uint16).view(np.float16)
    return unpacked

def pack_fp16_ndarray(
    x: np.ndarray[np.dtype[np.float16]]
):
    """
    packs an entire `FP16` array into `FP32` array

    :param x: array to compress
    :type x: np.ndarray
    """
    # pre-allocate output array
    out = np.empty((len(x)+1)//2, dtype=np.float32)

    idx = 0
    # TODO: vectorize this loop
    for i in range(0, len(x), 2):
        if i+1 < len(x):
            out[idx] = pack_fp16_into_fp32(x[i], x[i+1])
        else:
            out[idx] = pack_fp16_into_fp32(x[i], np.array([0.0], dtype=np.float16))
        idx += 1
    
    return out

def unpack_fp32_ndarray(
    x: np.ndarray[np.dtype[np.float32]]
) -> np.ndarray[np.dtype[np.float16]]:
    """
    unpacks an entire `FP32` array packed with 2 `FP16` numbers

    :param x: array to unpack
    :type x: np.ndarray
    """
    # pre-allocate output array
    out = np.empty((len(x))*2, dtype=np.float16)

    idx = 0
    # TODO: vectorize this loop
    for i in range(0, len(x)):
        a_unpacked, b_unpacked = unpack_fp32_into_fp16(x[i])
        out[idx] = a_unpacked
        out[idx+1] = b_unpacked
        idx += 2
    
    return out