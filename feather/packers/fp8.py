import logging

import ml_dtypes
import numpy as np
import torch
import torch.functional as F

def pack_fp8_into_fp32_legacy(
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
    - mode is unused, supports only `E5M3`

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
    a_u16 = np.array([a], dtype=np.float16).view(np.uint16)[0]
    b_u16 = np.array([b], dtype=np.float16).view(np.uint16)[0]
    c_u16 = np.array([c], dtype=np.float16).view(np.uint16)[0]
    d_u16 = np.array([d], dtype=np.float16).view(np.uint16)[0]

    if mode == "E5M2":
        # E5M2 just strip off 8 upper bits from mantissa
        a_u8 = (a_u16 >> 8) & 0xFF
        b_u8 = (b_u16 >> 8) & 0xFF
        c_u8 = (c_u16 >> 8) & 0xFF
        d_u8 = (d_u16 >> 8) & 0xFF
    else:
        raise NotImplementedError(f"PARAM mode {mode} is not supported")

    # convert to u32
    a_u32 = np.uint32(a_u8)
    b_u32 = np.uint32(b_u8)
    c_u32 = np.uint32(c_u8)
    d_u32 = np.uint32(d_u8)

    # pack
    packed_bits = a_u32 | (b_u32 << 8) | (c_u32 << 16) | (d_u32 << 24)

    # convert into u32 and view as fp32
    packed = np.array([packed_bits], dtype=np.uint32).view(np.float32)[0]
    return packed


def pack_fp8_into_fp32(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, mode: str = "E5M2"
) -> np.ndarray:
    """
    Packs 4 `FP8` values into 1 `FP32` container.

    Parameters
    ----------
    a: np.ndarray
        Single element `ndarray` in `np.float16` format
    b: np.ndarray
        Single element `ndarray` in `np.float16` format
    c: np.ndarray
        Single element `ndarray` in `np.float16` format
    d: np.ndarray
        Single element `ndarray` in `np.float16` format
    mode: str
        FP8 standard to use, available options:
        - E4M3
        - E5M2
    Returns
    -------
    np.ndarray
        Single element `ndarray` containing the packed `np.float32` value
    """
    # sanity checks
    if (
        a.dtype != np.float16
        or b.dtype != np.float16
        or d.dtype != np.float16
        or c.dtype != np.float16
    ):
        logging.warning(
            f"PARAM a ({a}) or b ({b}) or c({c}) or d ({d}) was not float16, routine is undefined!"
        )
    if len(a.shape) > 1 or len(b.shape) > 1 or len(c.shape) > 1 or len(c.shape) > 1:
        logging.warning(
            f"PARAM a ({a}) or b ({b}) or c({c}) or d ({d}) had more than 1 element to pack"
        )

    if mode == "E5M2":
        target_dtype = ml_dtypes.float8_e5m2
    elif mode == "E4M3":
        target_dtype = ml_dtypes.float8_e4m3fn
    else:
        raise NotImplementedError(f"Mode {mode} not supported")

    # cast as target datatype
    a_fp8 = a.astype(target_dtype)
    b_fp8 = b.astype(target_dtype)
    c_fp8 = c.astype(target_dtype)
    d_fp8 = d.astype(target_dtype)

    # convert into u8 -> u32 & pack them
    a_u8 = a_fp8.view(np.uint8)
    b_u8 = b_fp8.view(np.uint8)
    c_u8 = c_fp8.view(np.uint8)
    d_u8 = d_fp8.view(np.uint8)

    a_u32 = a_u8.astype(np.uint32)
    b_u32 = b_u8.astype(np.uint32)
    c_u32 = c_u8.astype(np.uint32)
    d_u32 = d_u8.astype(np.uint32)

    # convert into u32 and view as fp32
    packed_bits = a_u32 | (b_u32 << 8) | (c_u32 << 16) | (d_u32 << 24)
    packed = np.array([packed_bits], dtype=np.uint32).view(np.float32)[0]
    return packed


def pack_fp8_ndarray_legacy(
    x:np.ndarray[np.dtype[np.float16]],
    mode: str = "E5M2"
) -> np.ndarray[np.dtype[np.float32]]:
    """
    Packs an entire `ndarray` of `FP16` values into 1 `FP32` container.
    Casts from `FP16` into `FP8` before packing.
    Packed array contains `n\\4` elements. 

    Parameters
    ----------
    x: np.ndarray
        `ndarray` in `np.float16` format
    mode: str
        FP8 standard to use, available options:
        - E4M3
        - E5M2

    Returns
    -------
    np.ndarray
        `ndarray` containing the packed `np.float32` value

    Note
    ----
    - The Implementation is naive python for loops, do not use this.
    """
    # pre-allocate output array
    out = np.empty((len(x)+3)//4, dtype=np.float32)

    idx = 0
    # TODO: vectorize this loop
    for i in range(0, len(x), 4):
        out[idx] = pack_fp8_into_fp32(x[i], x[i+1], x[i+2], x[i+3], mode)
        idx += 1

    return out


def pack_fp8_ndarray(
    x: np.ndarray, mode: str = "E5M2"
) -> np.ndarray:
    """
    Packs an entire `ndarray` of `FP16` values into 1 `FP32` container.
    Casts from `FP16` into `FP8` before packing.
    Packed array contains `n\\4` elements.

    Parameters
    ----------
    x: np.ndarray
        `ndarray` in `np.float16` format
    mode: str
        FP8 standard to use, available options:
        - E4M3
        - E5M2

    Returns
    -------
    np.ndarray
        `ndarray` containing the packed `np.float32` value
    """
    x = x.flatten()
    rem = x.size % 4
    if rem > 0:
        pad_len = 4 - rem
        x = np.pad(x, (0, pad_len), mode="constant", constant_values=0)

    x_reshaped = x.reshape(-1, 4)

    return pack_fp8_into_fp32(
        x_reshaped[:, 0], x_reshaped[:, 1], x_reshaped[:, 2], x_reshaped[:, 3], mode
    )


def pack_fp8_tensor_legacy(
    x: torch.Tensor,
    mode: str = "E5M2"
) -> torch.Tensor:
    """
    Packs an entire `torch.Tensor` of `FP16` values into 1 `FP32` container.
    Casts from `FP16` into `FP8` before packing.
    Packed array contains `n\\4` elements.

    Parameters
    ----------
    x: torch.Tensor
        Tensor in `np.float16` format
    mode: str
        FP8 standard to use, available options:
        - E4M3
        - E5M2

    Returns
    -------
    torch.Tensor
        Tensor containing the packed `np.float32` value

    Note
    ----
    - Implementation is limited to E5M2, do not use this
    """
    # pad with zeros to prevent out of bounds
    x_flat = x.flatten()
    pad_len = (4 - (x_flat.numel() % 4)) % 4
    if pad_len > 0:
        x_flat = F.pad(x_flat, (0, pad_len), value=0.0)

    # NOTE: torch.int<bits> variants are used here because of the limited
    # support for rshift & lshift from torch
    # view as u16
    x_u16 = x_flat.view(torch.int16)
    # shift to remove mantissa
    x_u8 = (x_u16 >> 8) & 0xFF
    # compress into fp32 tensor
    x_u32 = x_u8.to(torch.int32)
    x_reshaped = x_u32.view(-1, 4)
    # pack them
    packed_ints = (x_reshaped[:, 0]) | (x_reshaped[:, 1] << 8) | (x_reshaped[:, 2] << 16) | (x_reshaped[:, 3] << 24)
    return packed_ints.view(torch.float32)


def pack_fp8_tensor(x: torch.Tensor, mode: str = "E5M2") -> torch.Tensor:
    """
    Packs an entire `torch.Tensor` of `FP16` values into 1 `FP32` container.
    Casts from `FP16` into `FP8` before packing.
    Packed array contains `n\\4` elements.

    Parameters
    ----------
    x: torch.Tensor
        Tensor in `np.float16` format
    mode: str
        FP8 standard to use, available options:
        - E4M3
        - E5M2

    Returns
    -------
    torch.Tensor
        Tensor containing the packed `np.float32` value
    """
    x_np = x.flatten().cpu().numpy()
    return torch.from_numpy(pack_fp8_ndarray(x_np, mode=mode).view(np.uint32))


def unpack_fp8_from_fp32_legacy(
    a:np.ndarray,
    mode: str = "E5M2"
) -> np.ndarray:
    """
    UnPacks 4 `FP8` values from 1 `FP32` container.

    Parameters
    ----------
    a: np.ndarray
        Single packed element of FP32
    mode: str
        FP8 standard to use, available options:
        - E4M3
        - E5M2
    Returns
    -------
    np.ndarray
        Four elements `ndarray` containing the unpacked `np.float16` value

    Note
    ----
    - Unpacked FP8's are upcasted to FP16 because FP8 isn't supported natively by numpy
    - Implementation Limited to E5M2, do not use this
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


def unpack_fp8_from_fp32(
    a: np.ndarray, mode: str = "E5M2"
) -> np.ndarray:
    """
    UnPacks 4 `FP8` values from 1 `FP32` container.

    Parameters
    ----------
    a: np.ndarray
        Single packed element of FP32
    mode: str
        FP8 standard to use, available options:
        - E4M3
        - E5M2
    Returns
    -------
    np.ndarray
        Four elements `ndarray` containing the unpacked `np.float16` value

    Note
    ----
    - Unpacked FP8's are upcasted to FP16 because FP8 isn't supported natively by numpy
    """
    # sanity checks
    if a.dtype != np.float32:
        logging.warning(f"PARAM a ({a}) was not float32, routine is undefined!")
    if len(a.shape) > 1:
        logging.warning(f"PARAM a ({a}) had more than 1 element to pack")

    if mode == "E5M2":
        target_dtype = ml_dtypes.float8_e5m2
    elif mode == "E4M3":
        target_dtype = ml_dtypes.float8_e4m3fn
    else:
        raise NotImplementedError(f"Mode {mode} not supported")

    # view as u32
    a_u32 = np.array([a], dtype=np.float32).view(np.uint32)[0]

    # extract bits as packed
    x_u32 = (a_u32) & 0xFF
    b_u32 = (a_u32 >> 8) & 0xFF
    c_u32 = (a_u32 >> 16) & 0xFF
    d_u32 = (a_u32 >> 24) & 0xFF

    x_fp8 = np.array([x_u32], dtype=np.uint8).view(target_dtype)
    b_fp8 = np.array([b_u32], dtype=np.uint8).view(target_dtype)
    c_fp8 = np.array([c_u32], dtype=np.uint8).view(target_dtype)
    d_fp8 = np.array([d_u32], dtype=np.uint8).view(target_dtype)

    # convert into u16, view as fp16 and cast to fp32
    unpacked = (
        np.array([x_fp8, b_fp8, c_fp8, d_fp8], dtype=target_dtype)
        .astype(np.float16)
        .astype(np.float32)
    )
    return unpacked


def unpack_into_fp8_ndarray(
    x: np.ndarray, mode: str = "E5M2"
) -> np.ndarray:
    """
    UnPacks an entire `ndarray` of `FP8` values into 4 `FP16` container.
    Casts from `FP8` into `FP16` before packing.
    Packed array contains `n*4` elements.

    Parameters
    ----------
    x: np.ndarray
        `ndarray` in `np.float16` format
    mode: str
        FP8 standard to use, available options:
        - E4M3
        - E5M2

    Returns
    -------
    np.ndarray
        `ndarray` containing the unpacked `np.float32` value
    """
    # pre-allocate output array
    out = np.empty((len(x)) * 4, dtype=np.float16)

    idx = 0
    # TODO: vectorize this loop
    for i in range(0, len(x)):
        a_unpacked, b_unpacked, c_unpacked, d_unpacked = unpack_fp8_from_fp32(
            x[i], mode
        )
        out[idx] = a_unpacked
        out[idx + 1] = b_unpacked
        out[idx + 2] = c_unpacked
        out[idx + 3] = d_unpacked
        idx += 4

    return out