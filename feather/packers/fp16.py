import logging

import numpy as np
import torch
import torch.functional as F


def pack_fp16_into_fp32(
    a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    Packs 2 `FP16` values into 1 `FP32` container.

    Parameters
    ----------
    a: np.ndarray
        Single element `ndarray` in `np.float16` format
    b: np.ndarray
        Single element `ndarray` in `np.float16` format
        
    Returns
    -------
    np.ndarray
        Single element `ndarray` containing the packed `np.float32` value
    """
    # sanity checks
    if a.dtype != np.float16 or b.dtype != np.float16:
        logging.warning(
            f"PARAM a ({a}) or b ({b}) was not float16, routine is undefined!"
        )
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


def unpack_fp32_into_fp16(
    a: np.ndarray[np.dtype[np.float32]],
) -> np.ndarray[np.dtype[np.float16], np.dtype[np.float16]]:
    """
    Unpacks 1 `FP32` values into 2 `FP16` container.

    Parameters
    ----------
    a: np.ndarray
        Single element `ndarray` in `np.float32` format
        
    Returns
    -------
    np.ndarray
        Two element `ndarray` containing the unpacked `np.float16` values

    Note
    ----
    - If the element was not packed using `feather.packers` functions, then
    there is no guarantee that unpacking will produce expected result
    """
    # sanity checks
    if a.dtype != np.float32:
        logging.warning(f"PARAM a ({a}) was not float32, routine is undefined!")
    if len(a.shape) > 1:
        logging.warning(f"PARAM a ({a}) had more than 1 element to pack")

    # view as u32
    a_bits = np.array([a], dtype=np.float32).view(np.uint32)[0]
    a_lower = a_bits & 0xFFFF
    a_upper = (a_bits >> 16) & 0xFFFF

    # unpack into u16 and view as f16
    unpacked = np.array([a_lower, a_upper], dtype=np.uint16).view(np.float16)
    return unpacked


def pack_fp16_ndarray(
    x: np.ndarray
) -> np.ndarray:
    """
    Packs an entire `ndarray` containing `np.float16` values into `np.float32` ndarray.
    Return `ndarray` is of the size `n/2`.

    Parameters
    ----------
    a: np.ndarray
        `ndarray` in `np.float16` format
        
    Returns
    -------
    np.ndarray
        `ndarray` containing the packed `np.float32` values

    Note
    ----
    - Pads with zeros if it is not perfectly divisible by 2
    """
    # pre-allocate output array
    out = np.empty((len(x) + 1) // 2, dtype=np.float32)

    idx = 0
    # TODO: vectorize this loop
    for i in range(0, len(x), 2):
        if i + 1 < len(x):
            out[idx] = pack_fp16_into_fp32(x[i], x[i + 1])
        else:
            out[idx] = pack_fp16_into_fp32(x[i], np.array([0.0], dtype=np.float16))
        idx += 1

    return out


def unpack_into_fp16_ndarray(
    x: np.ndarray[np.dtype[np.float32]],
) -> np.ndarray[np.dtype[np.float16]]:
    """
    UnPacks an entire `ndarray` containing `np.float32` values into `np.float16` ndarray.
    Return `ndarray` is of the size `2*n`.

    Parameters
    ----------
    a: np.ndarray
        `ndarray` in `np.float32` format
        
    Returns
    -------
    np.ndarray
        `ndarray` containing the packed `np.float16` values

    Note
    ----
    - Pads with zeros if it is not perfectly divisible by 2
    """
    # pre-allocate output array
    out = np.empty((len(x)) * 2, dtype=np.float16)

    idx = 0
    # TODO: vectorize this loop
    for i in range(0, len(x)):
        a_unpacked, b_unpacked = unpack_fp32_into_fp16(x[i])
        out[idx] = a_unpacked
        out[idx + 1] = b_unpacked
        idx += 2

    return out
