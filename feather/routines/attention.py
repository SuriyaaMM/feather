import torch
import triton
import triton.language as tl

from feather.packers.fp8 import *

# @triton.jit
# def _attention_fp8_e5m2_acc_fp32_kernel(
#     _q: torch.Tensor,
#     _k: torch.Tensor,
#     _v: torch.Tensor,
#     _attn_out: torch.Tensor,
#     _seq_len: int,
#     _h_dim: tl.constexpr,
#     _blk_size_seq_len: tl.constexpr,
# ):
#     _q_row_id = tl.program_id(axis=1)

#     _q_ptr = _q.to(tl.pointer_type(tl.int8))
#     _k_ptr = _k.to(tl.pointer_type(tl.int8))
#     _v_ptr = _v.to(tl.pointer_type(tl.int8))
    
#     _q_ptr_rows = (_q_row_id * _blk_size_seq_len + tl.arange(start=0, end=_blk_size_seq_len)) * _h_dim
#     _q_ptr_cols = tl.arange(start=0, end=_h_dim)
#     _q_ptr_mesh = _q_ptr_rows[:, None] + _q_ptr_cols[None, :]

#     _k_ptr_cols = (_q_row_id * _blk_size_seq_len + tl.arange(start=0, end=_blk_size_seq_len)) * _h_dim
#     _k_ptr_rows = tl.arange(start=0, end=_h_dim)
#     _k_ptr_mesh = _k_ptr_rows[:, None] + _k_ptr_cols[None, :]

#     _v_ptr_rows = (_q_row_id * _blk_size_seq_len + tl.arange(start=0, end=_blk_size_seq_len)) * _h_dim
#     _v_ptr_cols = tl.arange(start=0, end=_h_dim)
#     _v_ptr_mesh = _v_ptr_rows[:, None] + _v_ptr_cols[None, :]

#     _mask = (_q_row_id * _blk_size_seq_len + tl.arange(start=0, end=_blk_size_seq_len)) < _seq_len

#     _q_rows_packed = tl.load(pointer=_q_ptr + _q_ptr_mesh, mask=_mask[:, None])
#     _k_rows_packed = tl.load(pointer=_k_ptr + _k_ptr_mesh, mask=_mask[None, :])
#     _v_rows_packed = tl.load(pointer=_v_ptr + _v_ptr_mesh, mask=_mask[:, None])

#     _q_rows_u8 = _q_rows_packed.to(tl.uint8).to(tl.uint32)
#     _q_rows_u32 = ((_q_rows_u8 & 0x80) << 24) | \
#                        ((((_q_rows_u8 & 0x7c) >> 2) + 112) << 23) | \
#                        ((_q_rows_u8 & 0x03) << 21)
#     _q_rows = _q_rows_u32.to(tl.float32, bitcast=True)

#     _k_rows_u8 = _k_rows_packed.to(tl.uint8).to(tl.uint32)
#     _k_rows_u32 = ((_k_rows_u8 & 0x80) << 24) | \
#                        ((((_k_rows_u8 & 0x7c) >> 2) + 112) << 23) | \
#                        ((_k_rows_u8 & 0x03) << 21)
#     _k_rows = _k_rows_u32.to(tl.float32, bitcast=True)

#     _v_rows_u8 = _v_rows_packed.to(tl.uint8).to(tl.uint32)
#     _v_rows_u32 = ((_v_rows_u8 & 0x80) << 24) | \
#                        ((((_v_rows_u8 & 0x7c) >> 2) + 112) << 23) | \
#                        ((_v_rows_u8 & 0x03) << 21)
#     _v_rows = _v_rows_u32.to(tl.float32, bitcast=True)

#     _qkT_temp = tl.zeros(shape=(_blk_size_seq_len, _blk_size_seq_len), dtype=tl.float32)
#     _qkT_dotted = tl.dot(input=_q_rows, other=_k_rows, acc=_qkT_temp)

#     _scale = 1.0 / tl.sqrt(tl.cast(_h_dim, dtype=tl.float32))
#     _qkT_dotted = _qkT_dotted * _scale

#     _max_val = tl.max(_qkT_dotted, axis=1)
#     _numerator = tl.exp(_qkT_dotted - _max_val[:, None])
#     _denominator = tl.sum(_numerator, axis=1)
#     _softmax_out = _numerator / _denominator[:, None]

#     _attn_out_temp = tl.zeros(shape=(_blk_size_seq_len, _h_dim), dtype=tl.float32)
#     _attn_final = tl.dot(input=_softmax_out, other=_v_rows, acc=_attn_out_temp)

#     _attn_out_ptr_rows = (_q_row_id * _blk_size_seq_len + tl.arange(start=0, end=_blk_size_seq_len))
#     _attn_out_ptr_cols = tl.arange(start=0, end=_h_dim)
#     _attn_out_ptr_mesh = _attn_out_ptr_rows[:, None] * _h_dim + _attn_out_ptr_cols[None, :]

#     tl.store(pointer=_attn_out + _attn_out_ptr_mesh, value=_attn_final, mask=_mask[:, None])

# @triton.jit
# def _attention_fp8_e5m2_acc_fp32_kernel(
#     _q: torch.Tensor,
#     _k: torch.Tensor,
#     _v: torch.Tensor,
#     _attn_out: torch.Tensor,
#     _seq_len: int,
#     _h_dim: tl.constexpr,
#     _blk_size_seq_len: tl.constexpr,
# ):
#     _q_row_id = tl.program_id(axis=1)

#     _q_ptr = _q.to(tl.pointer_type(tl.int8))
#     _k_ptr = _k.to(tl.pointer_type(tl.int8))
#     _v_ptr = _v.to(tl.pointer_type(tl.int8))
    
#     # pointer calculations
#     _q_ptr_rows = (_q_row_id * _blk_size_seq_len + tl.arange(start=0, end=_blk_size_seq_len)) * _h_dim
#     _q_ptr_cols = tl.arange(start=0, end=_h_dim)
#     _q_ptr_mesh = _q_ptr_rows[:, None] + _q_ptr_cols[None, :]

#     _k_ptr_cols = (_q_row_id * _blk_size_seq_len + tl.arange(start=0, end=_blk_size_seq_len)) * _h_dim
#     _k_ptr_rows = tl.arange(start=0, end=_h_dim)
#     _k_ptr_mesh = _k_ptr_rows[:, None] + _k_ptr_cols[None, :]

#     _v_ptr_rows = (_q_row_id * _blk_size_seq_len + tl.arange(start=0, end=_blk_size_seq_len)) * _h_dim
#     _v_ptr_cols = tl.arange(start=0, end=_h_dim)
#     _v_ptr_mesh = _v_ptr_rows[:, None] + _v_ptr_cols[None, :]

#     _mask = (_q_row_id * _blk_size_seq_len + tl.arange(start=0, end=_blk_size_seq_len)) < _seq_len

#     # unpacking 
#     _q_rows_packed = tl.load(pointer=_q_ptr + _q_ptr_mesh, mask=_mask[:, None])
#     _k_rows_packed = tl.load(pointer=_k_ptr + _k_ptr_mesh, mask=_mask[None, :])
#     _v_rows_packed = tl.load(pointer=_v_ptr + _v_ptr_mesh, mask=_mask[:, None])

#     _q_rows = (_q_rows_packed.to(tl.uint16) << 8).to(tl.float16, bitcast=True).to(tl.float32)
#     _k_rows = (_k_rows_packed.to(tl.uint16) << 8).to(tl.float16, bitcast=True).to(tl.float32)
#     _v_rows = (_v_rows_packed.to(tl.uint16) << 8).to(tl.float16, bitcast=True).to(tl.float32)

#     # attention 
#     _qkT_temp = tl.zeros(shape=(_blk_size_seq_len, _blk_size_seq_len), dtype=tl.float32)
#     _qkT_dotted = tl.dot(input=_q_rows, other=_k_rows, acc=_qkT_temp)

#     _scale = 1.0 / tl.sqrt(tl.cast(_h_dim, dtype=tl.float32))
#     _qkT_dotted = _qkT_dotted * _scale

#     _max_val = tl.max(_qkT_dotted, axis=1)
#     _numerator = tl.exp(_qkT_dotted - _max_val[:, None])
#     _denominator = tl.sum(_numerator, axis=1)
#     _softmax_out = _numerator / _denominator[:, None]

#     _attn_out_temp = tl.zeros(shape=(_blk_size_seq_len, _h_dim), dtype=tl.float32)
#     _attn_final = tl.dot(input=_softmax_out, other=_v_rows, acc=_attn_out_temp)

#     _attn_out_ptr_rows = (_q_row_id * _blk_size_seq_len + tl.arange(start=0, end=_blk_size_seq_len))
#     _attn_out_ptr_cols = tl.arange(start=0, end=_h_dim)
#     _attn_out_ptr_mesh = _attn_out_ptr_rows[:, None] * _h_dim + _attn_out_ptr_cols[None, :]

#     tl.store(pointer=_attn_out + _attn_out_ptr_mesh, value=_attn_final, mask=_mask[:, None])

# def attention_fp8_e5m2_acc_fp32_gpu(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int, h_dim: int):
#     """
#     NOTE: currently only works for BLOCK_SIZE >= seq_len, yet to work on, else
#     it would become local attention.
#     """
#     out = torch.empty((seq_len, h_dim), dtype=torch.float32, device="cuda")

#     BLK_SIZE_SEQ_LEN = 512
#     grid = (1, triton.cdiv(seq_len, BLK_SIZE_SEQ_LEN))

#     _attention_fp8_e5m2_acc_fp32_kernel[grid](
#         q, k, v, 
#         out, 
#         seq_len, 
#         h_dim, 
#         BLK_SIZE_SEQ_LEN
#     )
#     return out

@triton.jit
def _attention_fp8_e5m2_acc_fp32_kernel(
    _q: torch.Tensor,
    _k: torch.Tensor,
    _v: torch.Tensor,
    _attn_out: torch.Tensor,
    _seq_len: int,
    _h_dim: tl.constexpr,
    _blk_size_seq_len: tl.constexpr,
):
    _q_row_id = tl.program_id(axis=1)

    _q_ptr = _q.to(tl.pointer_type(tl.int8))
    _k_ptr = _k.to(tl.pointer_type(tl.int8))
    _v_ptr = _v.to(tl.pointer_type(tl.int8))
    
    # q pointer math & load
    _q_ptr_rows = (_q_row_id * _blk_size_seq_len + tl.arange(start=0, end=_blk_size_seq_len)) * _h_dim
    _q_ptr_cols = tl.arange(start=0, end=_h_dim)
    _q_ptr_mesh = _q_ptr_rows[:, None] + _q_ptr_cols[None, :]

    _mask_q = (_q_row_id * _blk_size_seq_len + tl.arange(start=0, end=_blk_size_seq_len)) < _seq_len

    _q_rows_packed = tl.load(pointer=_q_ptr + _q_ptr_mesh, mask=_mask_q[:, None])
    _q_rows = (_q_rows_packed.to(tl.uint16) << 8).to(tl.float16, bitcast=True).to(tl.float32)

    _m_prev = tl.zeros([_blk_size_seq_len], dtype=tl.float32) - float("inf")
    _d_prev = tl.zeros([_blk_size_seq_len], dtype=tl.float32)
    _acc = tl.zeros([_blk_size_seq_len, _h_dim], dtype=tl.float32)

    for _start_k in range(0, _seq_len, _blk_size_seq_len):
        # k pointer math & load
        _k_ptr_cols = (_start_k + tl.arange(start=0, end=_blk_size_seq_len)) * _h_dim
        _k_ptr_rows = tl.arange(start=0, end=_h_dim)
        _k_ptr_mesh = _k_ptr_rows[:, None] + _k_ptr_cols[None, :]

        # v pointer math & load
        _v_ptr_rows = (_start_k + tl.arange(start=0, end=_blk_size_seq_len)) * _h_dim
        _v_ptr_cols = tl.arange(start=0, end=_h_dim)
        _v_ptr_mesh = _v_ptr_rows[:, None] + _v_ptr_cols[None, :]

        _mask_k = (_start_k + tl.arange(start=0, end=_blk_size_seq_len)) < _seq_len

        _k_rows_packed = tl.load(pointer=_k_ptr + _k_ptr_mesh, mask=_mask_k[None, :])
        _v_rows_packed = tl.load(pointer=_v_ptr + _v_ptr_mesh, mask=_mask_k[:, None])

        _k_rows = (_k_rows_packed.to(tl.uint16) << 8).to(tl.float16, bitcast=True).to(tl.float32)
        _v_rows = (_v_rows_packed.to(tl.uint16) << 8).to(tl.float16, bitcast=True).to(tl.float32)

        # attention scores
        _qkT_temp = tl.zeros(shape=(_blk_size_seq_len, _blk_size_seq_len), dtype=tl.float32)
        _qkT_dotted = tl.dot(input=_q_rows, other=_k_rows, acc=_qkT_temp)

        _scale = 1.0 / tl.sqrt(tl.cast(_h_dim, dtype=tl.float32))
        _qkT_dotted = _qkT_dotted * _scale

        # online softmax
        _m_curr = tl.max(_qkT_dotted, axis=1)
        _m_new = tl.maximum(_m_prev, _m_curr)

        _alpha = tl.exp(_m_prev - _m_new)
        _acc = _acc * _alpha[:, None]
        _d_prev = _d_prev * _alpha

        _numerator = tl.exp(_qkT_dotted - _m_new[:, None])
        _acc += tl.dot(_numerator, _v_rows)
        _d_prev += tl.sum(_numerator, axis=1)

        _m_prev = _m_new

    _attn_final = _acc / _d_prev[:, None]
    # attention 
    _attn_out_ptr_rows = (_q_row_id * _blk_size_seq_len + tl.arange(start=0, end=_blk_size_seq_len))
    _attn_out_ptr_cols = tl.arange(start=0, end=_h_dim)
    _attn_out_ptr_mesh = _attn_out_ptr_rows[:, None] * _h_dim + _attn_out_ptr_cols[None, :]

    tl.store(pointer=_attn_out + _attn_out_ptr_mesh, value=_attn_final, mask=_mask_q[:, None])

def attention_fp8_e5m2_acc_fp32_gpu(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int, h_dim: int):
    out = torch.empty((seq_len, h_dim), dtype=torch.float32, device="cuda")

    BLK_SIZE_SEQ_LEN = 16
    grid = (1, triton.cdiv(seq_len, BLK_SIZE_SEQ_LEN))

    _attention_fp8_e5m2_acc_fp32_kernel[grid](
        q, k, v, 
        out, 
        seq_len, 
        h_dim, 
        BLK_SIZE_SEQ_LEN
    )
    return out