import torch
import triton
import triton.language as tl

from feather.packers.fp8 import *

@triton.autotune(
    configs=[
        triton.Config({'_tile_sz': 64}, num_warps=8, num_stages=1),
    ],
    key=['_seq_len', '_h_dim'],
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
    q_tile_a = (tl.cast((q_tile_packed) & 0xFF, dtype=tl.uint16) << 8).to(tl.float16, bitcast=True)
    q_tile_b = (tl.cast((q_tile_packed >> 8) & 0xFF, dtype=tl.uint16) << 8).to(tl.float16, bitcast=True)
    q_tile_c = (tl.cast((q_tile_packed >> 16) & 0xFF, dtype=tl.uint16) << 8).to(tl.float16, bitcast=True)
    q_tile_d = (tl.cast((q_tile_packed >> 24) & 0xFF, dtype=tl.uint16) << 8).to(tl.float16, bitcast=True)

    # tiles required for online softmax
    m_tile = tl.full(shape=(_tile_sz, ), value=float('-inf'), dtype=tl.float32)
    l_tile = tl.zeros(shape=(_tile_sz, ), dtype=tl.float32)
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
        k_tile_packed = tl.load(pointer=_k + (k_offsets), mask=mask_k[:, None], other=0.0)
        k_tile_a = (tl.cast((k_tile_packed) & 0xFF, dtype=tl.uint16) << 8).to(tl.float16, bitcast=True)
        k_tile_b = (tl.cast((k_tile_packed >> 8) & 0xFF, dtype=tl.uint16) << 8).to(tl.float16, bitcast=True)
        k_tile_c = (tl.cast((k_tile_packed >> 16) & 0xFF, dtype=tl.uint16) << 8).to(tl.float16, bitcast=True)
        k_tile_d = (tl.cast((k_tile_packed >> 24) & 0xFF, dtype=tl.uint16) << 8).to(tl.float16, bitcast=True)
        # load v
        v_tile_packed = tl.load(pointer=_v + (k_offsets), mask=mask_k[:, None], other=0.0)
        v_tile_a = (tl.cast((v_tile_packed) & 0xFF, dtype=tl.uint16) << 8).to(tl.float16, bitcast=True)
        v_tile_b = (tl.cast((v_tile_packed >> 8) & 0xFF, dtype=tl.uint16) << 8).to(tl.float16, bitcast=True)
        v_tile_c = (tl.cast((v_tile_packed >> 16) & 0xFF, dtype=tl.uint16) << 8).to(tl.float16, bitcast=True)
        v_tile_d = (tl.cast((v_tile_packed >> 24) & 0xFF, dtype=tl.uint16) << 8).to(tl.float16, bitcast=True)
        # partial attention score
        t_tile_a = tl.dot(input=q_tile_a, other=tl.trans(input=k_tile_a))
        t_tile_b = tl.dot(input=q_tile_b, other=tl.trans(input=k_tile_b))
        t_tile_c = tl.dot(input=q_tile_c, other=tl.trans(input=k_tile_c))
        t_tile_d = tl.dot(input=q_tile_d, other=tl.trans(input=k_tile_d))
        t_tile = t_tile_a.to(tl.float32) + t_tile_b.to(tl.float32) + t_tile_c.to(tl.float32) + t_tile_d.to(tl.float32)        
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
        
def flash_attention_fp8_e5m2_acc_fp32_gpu(
    q:torch.Tensor, 
    k:torch.Tensor, 
    v:torch.Tensor, 
    seq_len:int, 
    h_dim:int
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
    #grid = (triton.cdiv(seq_len, TILE_SIZE), )
    grid = lambda meta: (triton.cdiv(seq_len, meta['_tile_sz']), )
    out = torch.empty((seq_len, h_dim * 4), dtype=torch.float32, device="cuda")
    _flash_attention_fp8_e5m2_acc_fp32_kernel[grid](
        q, k, v, 
        out, 
        seq_len, 
        h_dim
    )
    return out