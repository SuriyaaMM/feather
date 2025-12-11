import numpy as np 
import logging

def pack_fp16_into_fp32(
    a:np.ndarray[np.dtype[np.float16]], 
    b:np.ndarray[np.dtype[np.float16]]
) -> np.ndarray[np.dtype[np.float32]]:
    """
    packs `2xFP16` into `FP32` container \\
    **NOTE**:
    - both the numbers must of of `FP16` type
    
    :param a: first number 
    :type a: np.ndarray
    :param b: second number
    :type b: np.ndarray
    :return: `FP32` packed with `2xFP16`
    :rtype: ndarray[dtype[float32], dtype[Any]]
    """
    # sanity checks
    if a.dtype != np.float16 or b.dtype != np.float16:
        logging.warning(f"PARAM a ({a}) or b ({b}) was not float16, routine is undefined!")
    if len(a.shape) > 1 or len(b.shape) > 1:
        logging.warning(f"PARAM a ({a}) and b ({b}) had more than 1 element to pack")
    
    # view as u16
    a_bits = np.array([a], dtype=np.float16).view(np.uint16)[0]
    b_bits = np.array([b], dtype=np.float16).view(np.uint16)[0]
    
    # pack
    packed_bits = (np.uint32(b_bits) << 16) | np.uint32(a_bits)
    
    # convert into u32 and view as fp32
    packed = np.array([packed_bits], dtype=np.uint32).view(np.float32)[0]
    return packed

def pack_fp8_into_fp32(
    a:np.ndarray[np.dtype[np.float16]], 
    b:np.ndarray[np.dtype[np.float16]],
    c:np.ndarray[np.dtype[np.float16]], 
    d:np.ndarray[np.dtype[np.float16]],
    mode: str = "E5M2"
) -> np.ndarray[np.dtype[np.float32]]:
    """
    packs `4xFP16` floats casted into `FP8` into a `FP32` container \\
    **NOTE**:
    - all the numbers must of of `FP16` type

    :param a: first number
    :type a: np.ndarray[np.dtype[np.float16]]
    :param b: second number
    :type b: np.ndarray[np.dtype[np.float16]]
    :param c: third number
    :type c: np.ndarray[np.dtype[np.float16]]
    :param d: fourth number
    :type d: np.ndarray[np.dtype[np.float16]]
    :param mode: FP8 standard, available = {`E5M3`}
    :type mode: str
    :return: `FP32` packed with `4xFP8`
    :rtype: ndarray[dtype[float32], dtype[Any]]
    """
    # sanity checks
    if a.dtype != np.float16 or b.dtype != np.float16 or d.dtype != np.float16 or c.dtype != np.float16:
        logging.warning(f"PARAM a ({a}) or b ({b}) or c({c}) or d ({d}) was not float16, routine is undefined!")
    if len(a.shape) > 1 or len(b.shape) > 1 or len(c.shape) > 1 or len(c.shape) > 1:
        logging.warning(f"PARAM a ({a}) or b ({b}) or c({c}) or d ({d}) had more than 1 element to pack")

    # view as u16
    a_bits = np.array([a], dtype=np.float16).view(np.uint16)[0]
    b_bits = np.array([b], dtype=np.float16).view(np.uint16)[0]
    c_bits = np.array([c], dtype=np.float16).view(np.uint16)[0]
    d_bits = np.array([d], dtype=np.float16).view(np.uint16)[0]

    if mode == "E5M2":
        # E5M2 just strip off 8 bits from mantissa
        a_e5m2 = (a_bits >> 8) & 0xFF
        b_e5m2 = (b_bits >> 8) & 0xFF
        c_e5m2 = (c_bits >> 8) & 0xFF
        d_e5m2 = (d_bits >> 8) & 0xFF
    else:
        raise NotImplementedError(f"PARAM mode {mode} is not implemented yet!")
    
    # convert to u32
    a_e5m2_casted = np.uint32(a_e5m2)
    b_e5m2_casted = np.uint32(b_e5m2)
    c_e5m2_casted = np.uint32(c_e5m2)
    d_e5m2_casted = np.uint32(d_e5m2)    
    
    # pack
    packed_bits = a_e5m2_casted | (b_e5m2_casted << 8) | (c_e5m2_casted << 16) | (d_e5m2_casted << 24)
    
    # convert into u32 and view as fp32
    packed = np.array([packed_bits], dtype=np.uint32).view(np.float32)[0]
    return packed

def unpack_fp32_into_fp16(
    a:np.ndarray[np.dtype[np.float32]]
) -> np.ndarray[np.dtype[np.float16], np.dtype[np.float16]]:
    """
    unpacks the given `FP32` into `2xFP16`

    :param a: packed `FP32`
    :type a: np.ndarray[np.dtype[np.float32]]
    :return: unpacked `2xFP16`
    :rtype: ndarray[dtype[float16], dtype[float16]]
    """
    # sanity checks
    if a.dtype != np.float32:
        logging.warning(f"PARAM a ({a}) was not float32, routine is undefined!")
    if len(a.shape) > 1:
        logging.warning(f"PARAM a ({a}) had more than 1 element to pack")

    # view as u32
    a_bits = np.array([a], dtype=np.float32).view(np.uint32)[0]
    a_lower = (a_bits & 0xFFFF)
    a_upper = (a_bits >> 16) & 0xFFFF  
    
    # unpack into u16 and view as f16
    unpacked = np.array([a_lower, a_upper], dtype=np.uint16).view(np.float16)
    return unpacked

def unpack_fp8_from_fp32(
    a:np.ndarray[np.dtype[np.float32]],
    mode: str = "E5M2"
) -> np.ndarray[np.dtype[np.float32]]:
    """
    unpacks the given `FP32` into `4xFP8`, since `FP8` isn't natively
    supported on most architecture, it is casted back into `FP16`.

    :param a: packed `FP32`
    :type a: np.ndarray[np.dtype[np.float32]]
    :return: unpacked `4xFP8` casted into `FP16`
    :rtype: ndarray[dtype[float16], dtype[float16]]
    """
    # sanity checks
    if a.dtype != np.float32:
        logging.warning(f"PARAM a ({a}) was not float32, routine is undefined!")
    if len(a.shape) > 1:
        logging.warning(f"PARAM a ({a}) had more than 1 element to pack")

    # view as u32
    a_bits = np.array([a], dtype=np.float32).view(np.uint32)[0]
    
    # extract bits as packed
    x_bits = (a_bits) & 0xFF
    b_bits = (a_bits >> 8) & 0xFF
    c_bits = (a_bits >> 16) & 0xFF
    d_bits = (a_bits >> 24) & 0xFF

    if mode == "E5M2":
        # shift to upper 8 bits to fill in the lower 8 with zeros
        x_e5m2 = x_bits << 8
        b_e5m2 = b_bits << 8
        c_e5m2 = c_bits << 8
        d_e5m2 = d_bits << 8

    # convert into u16, view as fp16 and cast to fp32
    unpacked = np.array([x_e5m2, b_e5m2, c_e5m2, d_e5m2], dtype=np.uint16).view(np.float16).astype(np.float32)
    return unpacked

def pack_fp16_ndarray(
    x:np.ndarray[np.dtype[np.float16]]
) -> np.ndarray[np.dtype[np.float32]]:
    """
    packs the given `FP16` array into `n/2` `FP32` array, here `n` is the
    number of elements present in parameter `x`
    
    :param x: packed `FP16`
    :type x: np.ndarray[np.dtype[np.float16]]
    :return: compressed array of `FP16` stored as `FP32`
    :rtype: ndarray[dtype[float32], dtype[float32]]
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

def pack_fp8_ndarray(
    x:np.ndarray[np.dtype[np.float16]]
) -> np.ndarray[np.dtype[np.float32]]:
    """
    packs the given `FP16` array into `n/4` `FP32` array, here `n` is the
    number of elements present in parameter `x`, but casts each `FP16` into
    `FP8` before compressing
    
    :param x: packed `FP16` casted as `FP8`
    :type x: np.ndarray[np.dtype[np.float16]]
    :return: compressed array of `FP8` stored as `FP32`
    :rtype: ndarray[dtype[float32], dtype[float32]]
    """
    # pre-allocate output array
    out = np.empty((len(x)+3)//4, dtype=np.float32)

    idx = 0
    # TODO: vectorize this loop
    for i in range(0, len(x), 4):
        out[idx] = pack_fp8_into_fp32(x[i], x[i+1], x[i+2], x[i+3])
        idx += 1
    
    return out

def unpack_into_fp16_ndarray(
    x:np.ndarray[np.dtype[np.float32]]
) -> np.ndarray[np.dtype[np.float16]]:
    """
    unpacks the given `FP32` array into `2*n` `FP16` array, here `n` is the
    number of elements present in parameter `x` \\
    reverse of `pack_fp32_ndarray`
    
    :param x: packed `FP32`
    :type x: np.ndarray[np.dtype[np.float16]]
    :return: de-compressed array of `FP32` stored as `FP16`
    :rtype: ndarray[dtype[float16], dtype[float16]]
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

def unpack_into_fp8_ndarray(
    x:np.ndarray[np.dtype[np.float16]]
) -> np.ndarray[np.dtype[np.float32]]:
    """
    unpacks the given `FP32` array into `4*n` `FP16` array, here `n` is the
    number of elements present in parameter `x` but casts to `FP8` \\
    reverse of `pack_fp32_ndarray`
    
    :param x: packed `FP32`
    :type x: np.ndarray[np.dtype[np.float16]]
    :return: de-compressed array of `FP32` stored as `FP8`
    :rtype: ndarray[dtype[float16], dtype[float16]]
    """
    # pre-allocate output array
    out = np.empty((len(x))*4, dtype=np.float16)

    idx = 0
    # TODO: vectorize this loop
    for i in range(0, len(x)):
        a_unpacked, b_unpacked, c_unpacked, d_unpacked = unpack_fp8_from_fp32(x[i])
        out[idx] = a_unpacked
        out[idx+1] = b_unpacked 
        out[idx+2] = c_unpacked
        out[idx+3] = d_unpacked
        idx += 4
    
    return out