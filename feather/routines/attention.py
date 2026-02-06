import torch
import triton
import triton.language as tl
from feather.routines.utils import _unpack_e4m3_to_fp16

from feather.packers.fp8 import *


@triton.autotune(
    configs=[
        triton.Config({"_tile_sz": 64}, num_warps=8, num_stages=1),
    ],
    key=["_seq_len", "_h_dim"],
)
@triton.jit
def _flash_attention_fp8_e5m2_acc_fp32_kernel(
    _q: torch.Tensor,
    _k: torch.Tensor,
    _v: torch.Tensor,
    _attn_out: torch.Tensor,
    _seq_len: int,
    _h_dim: tl.constexpr,
    _tile_sz: tl.constexpr,
):
    """Internal Kernel!, use flash_attention_fp8_e5m2_acc_fp32_kernel"""
    sqrt_d = tl.sqrt(tl.cast(_h_dim * 4, dtype=tl.float32))
    pid = tl.program_id(axis=0)

    # pointer calculation for current tile
    q_offsets_y = pid * _tile_sz + tl.arange(start=0, end=_tile_sz)
    q_offsets_x = tl.arange(start=0, end=_h_dim)
    q_offsets = q_offsets_x[None, :] + q_offsets_y[:, None] * _h_dim

    # mask for current tile
    mask = q_offsets_y < _seq_len

    # load q
    q_tile_packed = tl.load(pointer=_q + q_offsets, mask=mask[:, None], other=0.0)
    # unpack q
    q_tile_a = (tl.cast((q_tile_packed) & 0xFF, dtype=tl.uint16) << 8).to(
        tl.float16, bitcast=True
    )
    q_tile_b = (tl.cast((q_tile_packed >> 8) & 0xFF, dtype=tl.uint16) << 8).to(
        tl.float16, bitcast=True
    )
    q_tile_c = (tl.cast((q_tile_packed >> 16) & 0xFF, dtype=tl.uint16) << 8).to(
        tl.float16, bitcast=True
    )
    q_tile_d = (tl.cast((q_tile_packed >> 24) & 0xFF, dtype=tl.uint16) << 8).to(
        tl.float16, bitcast=True
    )

    # tiles required for online softmax
    m_tile = tl.full(shape=(_tile_sz,), value=float("-inf"), dtype=tl.float32)
    l_tile = tl.zeros(shape=(_tile_sz,), dtype=tl.float32)
    a_tile_a = tl.zeros(shape=(_tile_sz, _h_dim), dtype=tl.float32)
    a_tile_b = tl.zeros(shape=(_tile_sz, _h_dim), dtype=tl.float32)
    a_tile_c = tl.zeros(shape=(_tile_sz, _h_dim), dtype=tl.float32)
    a_tile_d = tl.zeros(shape=(_tile_sz, _h_dim), dtype=tl.float32)

    for tile_k_idx in tl.range(arg1=0, arg2=_seq_len, step=_tile_sz):
        # pointer calculation for current tile
        k_offsets_y = tile_k_idx + tl.arange(start=0, end=_tile_sz)
        k_offsets_x = tl.arange(start=0, end=_h_dim)
        k_offsets = k_offsets_x[None, :] + k_offsets_y[:, None] * _h_dim
        # mask for current tile
        mask_k = k_offsets_y < _seq_len
        # load k
        k_tile_packed = tl.load(
            pointer=_k + (k_offsets), mask=mask_k[:, None], other=0.0
        )
        k_tile_a = (tl.cast((k_tile_packed) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )
        k_tile_b = (tl.cast((k_tile_packed >> 8) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )
        k_tile_c = (tl.cast((k_tile_packed >> 16) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )
        k_tile_d = (tl.cast((k_tile_packed >> 24) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )
        # load v
        v_tile_packed = tl.load(
            pointer=_v + (k_offsets), mask=mask_k[:, None], other=0.0
        )
        v_tile_a = (tl.cast((v_tile_packed) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )
        v_tile_b = (tl.cast((v_tile_packed >> 8) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )
        v_tile_c = (tl.cast((v_tile_packed >> 16) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )
        v_tile_d = (tl.cast((v_tile_packed >> 24) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )
        # partial attention score
        t_tile_a = tl.dot(input=q_tile_a, other=tl.trans(input=k_tile_a))
        t_tile_b = tl.dot(input=q_tile_b, other=tl.trans(input=k_tile_b))
        t_tile_c = tl.dot(input=q_tile_c, other=tl.trans(input=k_tile_c))
        t_tile_d = tl.dot(input=q_tile_d, other=tl.trans(input=k_tile_d))
        t_tile = (
            t_tile_a.to(tl.float32)
            + t_tile_b.to(tl.float32)
            + t_tile_c.to(tl.float32)
            + t_tile_d.to(tl.float32)
        )
        t_tile /= sqrt_d
        # online softmax
        m_tile_inner_this = tl.max(t_tile, axis=1)
        m_tile_inner_new = tl.maximum(m_tile, m_tile_inner_this)
        alpha = tl.exp(m_tile - m_tile_inner_new)
        beta = tl.exp(t_tile - m_tile_inner_new[:, None])
        l_tile = l_tile * alpha + tl.sum(beta, axis=1)
        m_tile = m_tile_inner_new

        a_tile_a *= alpha[:, None]
        a_tile_b *= alpha[:, None]
        a_tile_c *= alpha[:, None]
        a_tile_d *= alpha[:, None]

        beta = beta.to(tl.float16)
        a_tile_a += tl.dot(input=beta, other=v_tile_a)
        a_tile_b += tl.dot(input=beta, other=v_tile_b)
        a_tile_c += tl.dot(input=beta, other=v_tile_c)
        a_tile_d += tl.dot(input=beta, other=v_tile_d)

    a_tile_a /= l_tile[:, None]
    a_tile_b /= l_tile[:, None]
    a_tile_c /= l_tile[:, None]
    a_tile_d /= l_tile[:, None]

    out_row_base = q_offsets_y[:, None] * (_h_dim * 4)
    out_col_base = q_offsets_x[None, :] * 4

    out_ptr_base = _attn_out + out_row_base + out_col_base
    tl.store(out_ptr_base + 0, a_tile_a, mask=mask[:, None])
    tl.store(out_ptr_base + 1, a_tile_b, mask=mask[:, None])
    tl.store(out_ptr_base + 2, a_tile_c, mask=mask[:, None])
    tl.store(out_ptr_base + 3, a_tile_d, mask=mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({"_tile_sz": 64}, num_warps=8, num_stages=1),
    ],
    key=["_seq_len", "_h_dim"],
)
@triton.jit
def _flash_attention_fp8_e4m3_acc_fp32_kernel(
    _q: torch.Tensor,
    _k: torch.Tensor,
    _v: torch.Tensor,
    _attn_out: torch.Tensor,
    _seq_len: int,
    _h_dim: tl.constexpr,
    _tile_sz: tl.constexpr,
):
    """Internal Kernel!, use flash_attention_fp8_e4m3_acc_fp32_kernel"""
    sqrt_d = tl.sqrt(tl.cast(_h_dim * 4, dtype=tl.float32))
    pid = tl.program_id(axis=0)

    # pointer calculation for current tile
    q_offsets_y = pid * _tile_sz + tl.arange(start=0, end=_tile_sz)
    q_offsets_x = tl.arange(start=0, end=_h_dim)
    q_offsets = q_offsets_x[None, :] + q_offsets_y[:, None] * _h_dim

    # mask for current tile
    mask = q_offsets_y < _seq_len

    # load q
    q_tile_packed = tl.load(pointer=_q + q_offsets, mask=mask[:, None], other=0.0)
    # unpack q
    q_tile_a = _unpack_e4m3_to_fp16(tl.cast((q_tile_packed) & 0xFF, dtype=tl.uint16))
    q_tile_b = _unpack_e4m3_to_fp16(
        tl.cast((q_tile_packed >> 8) & 0xFF, dtype=tl.uint16)
    )
    q_tile_c = _unpack_e4m3_to_fp16(
        tl.cast((q_tile_packed >> 16) & 0xFF, dtype=tl.uint16)
    )
    q_tile_d = _unpack_e4m3_to_fp16(
        tl.cast((q_tile_packed >> 24) & 0xFF, dtype=tl.uint16)
    )

    # tiles required for online softmax
    m_tile = tl.full(shape=(_tile_sz,), value=float("-inf"), dtype=tl.float32)
    l_tile = tl.zeros(shape=(_tile_sz,), dtype=tl.float32)
    a_tile_a = tl.zeros(shape=(_tile_sz, _h_dim), dtype=tl.float32)
    a_tile_b = tl.zeros(shape=(_tile_sz, _h_dim), dtype=tl.float32)
    a_tile_c = tl.zeros(shape=(_tile_sz, _h_dim), dtype=tl.float32)
    a_tile_d = tl.zeros(shape=(_tile_sz, _h_dim), dtype=tl.float32)

    for tile_k_idx in tl.range(arg1=0, arg2=_seq_len, step=_tile_sz):
        # pointer calculation for current tile
        k_offsets_y = tile_k_idx + tl.arange(start=0, end=_tile_sz)
        k_offsets_x = tl.arange(start=0, end=_h_dim)
        k_offsets = k_offsets_x[None, :] + k_offsets_y[:, None] * _h_dim
        # mask for current tile
        mask_k = k_offsets_y < _seq_len
        # load k
        k_tile_packed = tl.load(
            pointer=_k + (k_offsets), mask=mask_k[:, None], other=0.0
        )
        k_tile_a = _unpack_e4m3_to_fp16(
            tl.cast((k_tile_packed) & 0xFF, dtype=tl.uint16)
        )
        k_tile_b = _unpack_e4m3_to_fp16(
            tl.cast((k_tile_packed >> 8) & 0xFF, dtype=tl.uint16)
        )
        k_tile_c = _unpack_e4m3_to_fp16(
            tl.cast((k_tile_packed >> 16) & 0xFF, dtype=tl.uint16)
        )
        k_tile_d = _unpack_e4m3_to_fp16(
            tl.cast((k_tile_packed >> 24) & 0xFF, dtype=tl.uint16)
        )

        # load v
        v_tile_packed = tl.load(
            pointer=_v + (k_offsets), mask=mask_k[:, None], other=0.0
        )
        v_tile_a = _unpack_e4m3_to_fp16(
            tl.cast((v_tile_packed) & 0xFF, dtype=tl.uint16)
        )
        v_tile_b = _unpack_e4m3_to_fp16(
            tl.cast((v_tile_packed >> 8) & 0xFF, dtype=tl.uint16)
        )
        v_tile_c = _unpack_e4m3_to_fp16(
            tl.cast((v_tile_packed >> 16) & 0xFF, dtype=tl.uint16)
        )
        v_tile_d = _unpack_e4m3_to_fp16(
            tl.cast((v_tile_packed >> 24) & 0xFF, dtype=tl.uint16)
        )

        # partial attention score
        t_tile_a = tl.dot(input=q_tile_a, other=tl.trans(input=k_tile_a))
        t_tile_b = tl.dot(input=q_tile_b, other=tl.trans(input=k_tile_b))
        t_tile_c = tl.dot(input=q_tile_c, other=tl.trans(input=k_tile_c))
        t_tile_d = tl.dot(input=q_tile_d, other=tl.trans(input=k_tile_d))
        t_tile = (
            t_tile_a.to(tl.float32)
            + t_tile_b.to(tl.float32)
            + t_tile_c.to(tl.float32)
            + t_tile_d.to(tl.float32)
        )
        t_tile /= sqrt_d
        # online softmax
        m_tile_inner_this = tl.max(t_tile, axis=1)
        m_tile_inner_new = tl.maximum(m_tile, m_tile_inner_this)
        alpha = tl.exp(m_tile - m_tile_inner_new)
        beta = tl.exp(t_tile - m_tile_inner_new[:, None])
        l_tile = l_tile * alpha + tl.sum(beta, axis=1)
        m_tile = m_tile_inner_new

        a_tile_a *= alpha[:, None]
        a_tile_b *= alpha[:, None]
        a_tile_c *= alpha[:, None]
        a_tile_d *= alpha[:, None]

        beta = beta.to(tl.float16)
        a_tile_a += tl.dot(input=beta, other=v_tile_a)
        a_tile_b += tl.dot(input=beta, other=v_tile_b)
        a_tile_c += tl.dot(input=beta, other=v_tile_c)
        a_tile_d += tl.dot(input=beta, other=v_tile_d)

    a_tile_a /= l_tile[:, None]
    a_tile_b /= l_tile[:, None]
    a_tile_c /= l_tile[:, None]
    a_tile_d /= l_tile[:, None]

    out_row_base = q_offsets_y[:, None] * (_h_dim * 4)
    out_col_base = q_offsets_x[None, :] * 4

    out_ptr_base = _attn_out + out_row_base + out_col_base
    tl.store(out_ptr_base + 0, a_tile_a, mask=mask[:, None])
    tl.store(out_ptr_base + 1, a_tile_b, mask=mask[:, None])
    tl.store(out_ptr_base + 2, a_tile_c, mask=mask[:, None])
    tl.store(out_ptr_base + 3, a_tile_d, mask=mask[:, None])


@triton.jit
def _paged_attention_fp8_e5m2_acc_fp32_kernel(
    _q: torch.Tensor,  # [batch_size, num_heads, head_dim]
    _k_cache: torch.Tensor,  # [num_blocks, num_heads, head_dim, block_size]
    _v_cache: torch.Tensor,  # [num_blocks, num_heads, head_dim, block_size]
    _block_table: torch.Tensor,  # [batch_size, max_blocks_per_sequence]
    _context_lens: torch.Tensor,  # [batch_size]
    _attention_out: torch.Tensor,  # [batch_size, num_heads, head_dim]
    _batch_size: int,  # number of sequences in this batch
    _head_dim: tl.constexpr,  # dimension of each head
    _n_heads: tl.constexpr,  # number of heads
    _n_heads_per_chunk: tl.constexpr,  # number of heads to load per chunk
    _max_blocks_per_sequence: tl.constexpr,  # for block table traversal
    _block_size: tl.constexpr,  # for cache traversal
):
    """Internal Kernel!, use paged_attention_fp8_e5m2_acc_fp32_kernel"""

    # x - along batch, y - along heads
    pidx = tl.program_id(axis=0)
    pidy = tl.program_id(axis=1)

    # constants
    sqrt_d = tl.sqrt(tl.cast(_head_dim * 4, dtype=tl.float32))
    chunk_size = _n_heads_per_chunk * _head_dim

    # load a contiguous chunk of q from memory
    # q_offsets.shape = (_n_heads_per_chunk, _head_dim)
    q_offsets_y = (
        (pidx * _n_heads * _head_dim)
        + (pidy * _head_dim)
        + tl.arange(start=0, end=_n_heads_per_chunk)
    )

    q_offsets_x = tl.arange(start=0, end=_head_dim)
    q_offsets = q_offsets_x[None, :] + q_offsets_y[:, None] * _head_dim

    # mask for current tile
    mask = q_offsets_y < _n_heads

    # load q
    q_chunk_packed = tl.load(pointer=_q + q_offsets, mask=mask[:, None], other=0.0)
    # unpack q
    q_chunk_a = (tl.cast((q_chunk_packed) & 0xFF, dtype=tl.uint16) << 8).to(
        tl.float16, bitcast=True
    )
    q_chunk_b = (tl.cast((q_chunk_packed >> 8) & 0xFF, dtype=tl.uint16) << 8).to(
        tl.float16, bitcast=True
    )
    q_chunk_c = (tl.cast((q_chunk_packed >> 16) & 0xFF, dtype=tl.uint16) << 8).to(
        tl.float16, bitcast=True
    )
    q_chunk_d = (tl.cast((q_chunk_packed >> 24) & 0xFF, dtype=tl.uint16) << 8).to(
        tl.float16, bitcast=True
    )

    # online softmax tiles
    m_chunk = tl.full(
        shape=(_n_heads_per_chunk,), value=float("-inf"), dtype=tl.float32
    )
    l_chunk = tl.zeros(shape=(_n_heads_per_chunk,), dtype=tl.float32)

    # attention output blocks
    a_chunk_a = tl.zeros(shape=(_n_heads_per_chunk, _head_dim), dtype=tl.float32)
    a_chunk_b = tl.zeros(shape=(_n_heads_per_chunk, _head_dim), dtype=tl.float32)
    a_chunk_c = tl.zeros(shape=(_n_heads_per_chunk, _head_dim), dtype=tl.float32)
    a_chunk_d = tl.zeros(shape=(_n_heads_per_chunk, _head_dim), dtype=tl.float32)

    context_len = tl.load(pointer=_context_lens + pidx)
    n_iterations = (context_len + (_block_size - 1)) // _block_size

    kv_offset = (tl.arange(start=0, end=_head_dim)[:, None] * _block_size) + tl.arange(
        start=0, end=_block_size
    )[None, :]

    for block_index in tl.range(arg1=0, arg2=n_iterations):

        physical_index_offset = (pidx * _max_blocks_per_sequence) + block_index

        physical_index = tl.load(pointer=_block_table + physical_index_offset)

        kv_base_offset = (physical_index * _n_heads * _head_dim * _block_size) + (
            pidy * _head_dim * _block_size
        )

        mask_kv = (
            block_index * _block_size + tl.arange(start=0, end=_block_size)
        ) < context_len

        # load k
        k_chunk_packed = tl.load(
            pointer=_k_cache + kv_offset + kv_base_offset,
            mask=mask_kv[None, :],
            other=0.0,
        )
        # unpack k
        k_chunk_a = (tl.cast((k_chunk_packed) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )
        k_chunk_b = (tl.cast((k_chunk_packed >> 8) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )
        k_chunk_c = (tl.cast((k_chunk_packed >> 16) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )
        k_chunk_d = (tl.cast((k_chunk_packed >> 24) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )

        # load v
        v_chunk_packed = tl.load(
            pointer=_v_cache + kv_offset + kv_base_offset,
            mask=mask_kv[None, :],
            other=0.0,
        )
        # unpack v
        v_chunk_a = (tl.cast((v_chunk_packed) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )
        v_chunk_b = (tl.cast((v_chunk_packed >> 8) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )
        v_chunk_c = (tl.cast((v_chunk_packed >> 16) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )
        v_chunk_d = (tl.cast((v_chunk_packed >> 24) & 0xFF, dtype=tl.uint16) << 8).to(
            tl.float16, bitcast=True
        )

        # partial attention score
        t_chunk_a = tl.dot(input=q_chunk_a, other=k_chunk_a)
        t_chunk_b = tl.dot(input=q_chunk_b, other=k_chunk_b)
        t_chunk_c = tl.dot(input=q_chunk_c, other=k_chunk_c)
        t_chunk_d = tl.dot(input=q_chunk_d, other=k_chunk_d)

        t_chunk = (
            t_chunk_a.to(tl.float32)
            + t_chunk_b.to(tl.float32)
            + t_chunk_c.to(tl.float32)
            + t_chunk_d.to(tl.float32)
        )

        t_chunk /= sqrt_d

        # online softmax
        m_chunk_inner_this = tl.max(t_chunk, axis=1)
        m_chunk_inner_new = tl.maximum(m_chunk, m_chunk_inner_this)

        alpha = tl.exp(m_chunk - m_chunk_inner_new)
        beta = tl.exp(t_chunk - m_chunk_inner_new[:, None])

        l_chunk = l_chunk * alpha + tl.sum(beta, axis=1)
        m_chunk = m_chunk_inner_new

        a_chunk_a *= alpha[:, None]
        a_chunk_b *= alpha[:, None]
        a_chunk_c *= alpha[:, None]
        a_chunk_d *= alpha[:, None]

        beta = beta.to(tl.float16)
        a_chunk_a += tl.dot(input=beta, other=tl.trans(input=v_chunk_a))
        a_chunk_b += tl.dot(input=beta, other=tl.trans(input=v_chunk_b))
        a_chunk_c += tl.dot(input=beta, other=tl.trans(input=v_chunk_c))
        a_chunk_d += tl.dot(input=beta, other=tl.trans(input=v_chunk_d))

    a_chunk_a /= l_chunk[:, None]
    a_chunk_b /= l_chunk[:, None]
    a_chunk_c /= l_chunk[:, None]
    a_chunk_d /= l_chunk[:, None]

    out_row_base = q_offsets_y[:, None] * (_head_dim * 4)
    out_col_base = q_offsets_x[None, :] * 4

    out_ptr_base = _attention_out + out_row_base + out_col_base
    tl.store(out_ptr_base + 0, a_chunk_a, mask=mask[:, None])
    tl.store(out_ptr_base + 1, a_chunk_b, mask=mask[:, None])
    tl.store(out_ptr_base + 2, a_chunk_c, mask=mask[:, None])
    tl.store(out_ptr_base + 3, a_chunk_d, mask=mask[:, None])


def flash_attention_fp8_e5m2_acc_fp32_gpu(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int, h_dim: int
):
    """
    Flashattention kernel on `FP8_E5M2` packed into `FP32` arrays.
    Computation-wise should be equivalent to `torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(dim=0),
        k.unsqueeze(dim=0),
        v.unsqueeze(dim=0),
    ).squeeze(dim=0).to(torch.float32)`.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor (packed format).
    k : torch.Tensor
        Key tensor (packed format).
    v : torch.Tensor
        Value tensor (packed format).
    seq_len : int
        Sequence Length (N)
    h_dim : int
        Embedding Dimension

    Returns
    -------
    torch.Tensor
        Attention tensor in FP32 format.

    Notes
    -----
    - Parameter h_dim must be h_dim (original, before packing) // 4, the function will not do the division for you.
    - Input tensors must be packed using one of the functions exposed in `feather.packers.fp8` module, else computation is undefined.

    Examples
    --------
    >>> q = torch.randint(low=-3, high=3, size=(4, 4), dtype=torch.float16)
    >>> tensor([[ 1.,  1., -2.,  0.],
        [-3., -3., -2., -3.],
        [ 0.,  1.,  1., -2.],
        [ 2.,  0., -1.,  2.]], dtype=torch.float16)
    >>> k = torch.randint(low=-3, high=3, size=(4, 4), dtype=torch.float16)
    >>> tensor([[ 1.,  1., -2.,  0.],
        [-3., -3., -2., -3.],
        [ 0.,  1.,  1., -2.],
        [ 2.,  0., -1.,  2.]], dtype=torch.float16)
    >>> v = torch.randint(low=-3, high=3, size=(4, 4), dtype=torch.float16)
    >>> tensor([[ 1.,  1., -2.,  0.],
        [-3., -3., -2., -3.],
        [ 0.,  1.,  1., -2.],
        [ 2.,  0., -1.,  2.]], dtype=torch.float16)
    >>> tensor([ 2.,  2., -1.,  2.], dtype=torch.float16)
    >>> q_packed = pack_fp8_tensor(a, mode="E5M2").to("cuda")
    >>> k_packed = pack_fp8_tensor(b, mode="E5M2").to("cuda")
    >>> v_packed = pack_fp8_tensor(b, mode="E5M2").to("cuda")
    >>> attention = flash_attention_fp8_e5m2_acc_fp32_gpu(a_packed, b_packed, a.shape)
    """
    # grid = (triton.cdiv(seq_len, TILE_SIZE), )
    grid = lambda meta: (triton.cdiv(seq_len, meta["_tile_sz"]),)
    out = torch.empty((seq_len, h_dim * 4), dtype=torch.float32, device="cuda")
    _flash_attention_fp8_e5m2_acc_fp32_kernel[grid](q, k, v, out, seq_len, h_dim)
    return out


def flash_attention_fp8_e4m3_acc_fp32_gpu(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int, h_dim: int
):
    """
    Flashattention kernel on `FP8_E4M3` packed into `FP32` arrays.
    Computation-wise should be equivalent to `torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(dim=0),
        k.unsqueeze(dim=0),
        v.unsqueeze(dim=0),
    ).squeeze(dim=0).to(torch.float32)`.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor (packed format).
    k : torch.Tensor
        Key tensor (packed format).
    v : torch.Tensor
        Value tensor (packed format).
    seq_len : int
        Sequence Length (N)
    h_dim : int
        Embedding Dimension

    Returns
    -------
    torch.Tensor
        Attention tensor in FP32 format.

    Notes
    -----
    - Parameter h_dim must be h_dim (original, before packing) // 4, the function will not do the division for you.
    - Input tensors must be packed using one of the functions exposed in `feather.packers.fp8` module, else computation is undefined.

    Examples
    --------
    >>> q = torch.randint(low=-3, high=3, size=(4, 4), dtype=torch.float16)
    >>> tensor([[ 1.,  1., -2.,  0.],
        [-3., -3., -2., -3.],
        [ 0.,  1.,  1., -2.],
        [ 2.,  0., -1.,  2.]], dtype=torch.float16)
    >>> k = torch.randint(low=-3, high=3, size=(4, 4), dtype=torch.float16)
    >>> tensor([[ 1.,  1., -2.,  0.],
        [-3., -3., -2., -3.],
        [ 0.,  1.,  1., -2.],
        [ 2.,  0., -1.,  2.]], dtype=torch.float16)
    >>> v = torch.randint(low=-3, high=3, size=(4, 4), dtype=torch.float16)
    >>> tensor([[ 1.,  1., -2.,  0.],
        [-3., -3., -2., -3.],
        [ 0.,  1.,  1., -2.],
        [ 2.,  0., -1.,  2.]], dtype=torch.float16)
    >>> tensor([ 2.,  2., -1.,  2.], dtype=torch.float16)
    >>> q_packed = pack_fp8_tensor(a, mode="E5M2").to("cuda")
    >>> k_packed = pack_fp8_tensor(b, mode="E5M2").to("cuda")
    >>> v_packed = pack_fp8_tensor(b, mode="E5M2").to("cuda")
    >>> attention = flash_attention_fp8_e5m2_acc_fp32_gpu(a_packed, b_packed, a.shape)
    """
    # grid = (triton.cdiv(seq_len, TILE_SIZE), )
    grid = lambda meta: (triton.cdiv(seq_len, meta["_tile_sz"]),)
    out = torch.empty((seq_len, h_dim * 4), dtype=torch.float32, device="cuda")
    _flash_attention_fp8_e4m3_acc_fp32_kernel[grid](q, k, v, out, seq_len, h_dim)
    return out


def paged_attention_fp8_e5m2_acc_fp32_gpu(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    h_dim: int,
):
    """
    Paged Attention kernel on `FP8_E5M2` packed into `FP32` arrays.
    Designed for the decoding phase (1 query token per sequence).

    Parameters
    ----------
    q : torch.Tensor
        Query tensor. Shape: [Batch_Size, Num_Heads, Head_Dim_Packed].
    k_cache : torch.Tensor
        Key Cache. Shape: [Num_Phys_Blocks, Num_Heads, Head_Dim_Packed, Block_Size].
    v_cache : torch.Tensor
        Value Cache. Shape: [Num_Phys_Blocks, Num_Heads, Head_Dim_Packed, Block_Size].
    block_table : torch.Tensor
        Block Table. Shape: [Batch_Size, Max_Blocks_Per_Seq].
    context_lens : torch.Tensor
        Actual sequence lengths. Shape: [Batch_Size].
    h_dim : int
        Packed Embedding Dimension (Original Dim // 4).

    Returns
    -------
    torch.Tensor
        Attention output. Shape: [Batch_Size, Num_Heads, Head_Dim_Packed * 4].
        (Returned as unpacked FP32).
    """

    batch_size, num_heads, _ = q.shape
    block_size = k_cache.shape[-1]
    max_blocks_per_seq = block_table.shape[1]

    out = torch.empty(
        (batch_size, num_heads, h_dim * 4), dtype=torch.float32, device=q.device
    )

    num_heads_per_chunk = 4
    grid = (batch_size, triton.cdiv(num_heads, num_heads_per_chunk))

    _paged_attention_fp8_e5m2_acc_fp32_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_table,
        context_lens,
        out,
        _batch_size=int(batch_size),
        _head_dim=h_dim,
        _n_heads=num_heads,
        _n_heads_per_chunk=num_heads_per_chunk,
        _max_blocks_per_sequence=max_blocks_per_seq,
        _block_size=block_size,
        num_warps=4,
        num_stages=1,
    )

    return out
