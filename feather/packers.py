import numpy as np 
import logging

def pack_fp16_into_fp32(
    a: np.ndarray, 
    b: np.ndarray
):
    """
    packs two np.float16 types onto a np.float32 type
    
    :param a: first number (np.float16)
    :type a: np.ndarray
    :param b: second number (np.float16)
    :type b: np.ndarray
    """
    if a.dtype != np.float16 or b.dtype != np.float16:
        logging.warning(f"PARAM a ({a}) or b ({b}) was not float16, routine is undefined!")
    if len(a.shape) > 1 or len(b.shape) > 1:
        logging.warning(f"PARAM a ({a}) and b ({b}) had more than 1 element to pack")
    # view a and b as np.uint16
    a_bits = np.array([a], dtype=np.float16).view(np.uint16)[0]
    b_bits = np.array([b], dtype=np.float16).view(np.uint16)[0]
    # pack a into lower 16-bits & b into upper 16 bits
    packed_bits = (np.uint32(b_bits) << 16) | np.uint32(a_bits)
    # return 32-bits floating point view 
    return np.array([packed_bits], dtype=np.uint32).view(np.float32)[0]

def pack_fp16_ndarray(
    x: np.ndarray
):
    """
    Packs an entire `np.ndarray` into 32-bits floating points

    :param x: array to compress
    :type x: np.ndarray
    """
    if x.dtype != np.float16:
        raise ValueError("PARAM x must be float16")

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