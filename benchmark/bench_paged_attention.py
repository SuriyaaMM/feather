from typing import List, Tuple

import pytest
import torch
import torch.testing as tt

from feather.packers.fp8 import *
from feather.routines.attention import *

SEQ_LEN_PARAMETERS = [128]
H_DIM_PARAMETERS = [64]
BATCH_SIZE_PARAMETERS = [1]
NUM_HEADS_PARAMETERS = [4]
BLOCK_SIZE = 16


# pytorch implementation
def bench_paged_attention_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    batch_size: torch.Tensor,
    num_heads: torch.Tensor,
    head_dim: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(q)
    scale = 1.0 / (head_dim**0.5)

    for i in range(batch_size):
        seq_len = context_lens[i].item()
        num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        indices = block_table[i, :num_blocks].long()
        k_blocks = k_cache[indices]
        v_blocks = v_cache[indices]

        k_seq = k_blocks.permute(1, 0, 3, 2).reshape(num_heads, -1, head_dim)
        v_seq = v_blocks.permute(1, 0, 3, 2).reshape(num_heads, -1, head_dim)

        k_seq = k_seq.contiguous()
        v_seq = v_seq.contiguous()

        k_seq = k_seq[:, :seq_len, :]
        v_seq = v_seq[:, :seq_len, :]
        q_curr = q[i].unsqueeze(1)
        out[i] = torch.nn.functional.scaled_dot_product_attention(
            q_curr, k_seq, v_seq, scale=scale
        ).squeeze(1)

    return out


# ----- feather implementations
def bench_paged_attention_fp8_e5m2_feather_gpu(
    q, k_cache, v_cache, block_table, context_lens, h_dim_packed, num_heads_per_chunk
):
    return paged_attention_fp8_e5m2_acc_fp32_gpu(
        q,
        k_cache,
        v_cache,
        block_table,
        context_lens,
        h_dim_packed,
        num_heads_per_chunk,
    )


def bench_paged_attention_fp8_e4m3_feather_gpu(
    q, k_cache, v_cache, block_table, context_lens, h_dim_packed, num_heads_per_chunk
):
    return paged_attention_fp8_e4m3_acc_fp32_gpu(
        q,
        k_cache,
        v_cache,
        block_table,
        context_lens,
        h_dim_packed,
        num_heads_per_chunk,
    )


# ----- generators
@pytest.fixture
def generate_paged_input_tensors(
    batch_size: int, h_dim: int, num_heads: int, seq_len: int
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:

    max_blocks_per_seq = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = batch_size * max_blocks_per_seq

    q_ref = (
        torch.randn(size=(batch_size, num_heads, h_dim)).to(torch.float16).to("cuda")
    )
    k_ref = (
        torch.randn(size=(total_blocks, num_heads, h_dim, BLOCK_SIZE))
        .to(torch.float16)
        .to("cuda")
    )
    v_ref = (
        torch.randn(size=(total_blocks, num_heads, h_dim, BLOCK_SIZE))
        .to(torch.float16)
        .to("cuda")
    )

    context_lens = torch.randint(
        low=BLOCK_SIZE,
        high=seq_len + 1,
        size=(batch_size,),
        dtype=torch.int32,
        device="cuda",
    )
    block_table = torch.full(
        (batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device="cuda"
    )

    curr_block = 0
    for i in range(batch_size):
        n_blocks = (context_lens[i].item() + BLOCK_SIZE - 1) // BLOCK_SIZE
        indices = torch.arange(curr_block, curr_block + n_blocks, device="cuda")
        block_table[i, :n_blocks] = indices
        curr_block += n_blocks

    return (
        q_ref.to("cuda"),
        k_ref.to("cuda"),
        v_ref.to("cuda"),
        block_table.to("cuda"),
        context_lens.to("cuda"),
        total_blocks,
    )


# ----- tests
@pytest.mark.parametrize("batch_size", BATCH_SIZE_PARAMETERS)
@pytest.mark.parametrize("h_dim", H_DIM_PARAMETERS)
@pytest.mark.parametrize("num_heads", NUM_HEADS_PARAMETERS)
@pytest.mark.parametrize("seq_len", SEQ_LEN_PARAMETERS)
def test_paged_attention_fp32_torch(
    benchmark, batch_size, h_dim, num_heads, seq_len, generate_paged_input_tensors
):

    q, k, v, block_table, context_lens, _ = generate_paged_input_tensors

    benchmark(
        bench_paged_attention_torch,
        q,
        k,
        v,
        batch_size,
        num_heads,
        h_dim,
        block_table,
        context_lens,
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZE_PARAMETERS)
@pytest.mark.parametrize("h_dim", H_DIM_PARAMETERS)
@pytest.mark.parametrize("num_heads", NUM_HEADS_PARAMETERS)
@pytest.mark.parametrize("seq_len", SEQ_LEN_PARAMETERS)
def test_paged_attention_fp8_e5m2_feather_gpu(
    benchmark, batch_size, h_dim, num_heads, seq_len, generate_paged_input_tensors
):
    q, k, v, block_table, context_lens, total_blocks = generate_paged_input_tensors

    q_packed_flat = pack_fp8_tensor(q, mode="E5M2").view(torch.uint32)
    q_packed = q_packed_flat.view(batch_size, num_heads, h_dim // 4).contiguous()
    k_perm = k.permute(0, 1, 3, 2)
    k_packed_flat = pack_fp8_tensor(k_perm, mode="E5M2").view(torch.uint32)
    k_packed = k_packed_flat.view(total_blocks, num_heads, BLOCK_SIZE, h_dim // 4)
    k_cache_packed = k_packed.permute(0, 1, 3, 2).contiguous()
    v_perm = v.permute(0, 1, 3, 2)
    v_packed_flat = pack_fp8_tensor(v_perm, mode="E5M2").view(torch.uint32)
    v_packed = v_packed_flat.view(total_blocks, num_heads, BLOCK_SIZE, h_dim // 4)
    v_cache_packed = v_packed.permute(0, 1, 3, 2).contiguous()

    attn_out = benchmark(
        bench_paged_attention_fp8_e5m2_feather_gpu,
        q_packed.to("cuda"),
        k_cache_packed.to("cuda"),
        v_cache_packed.to("cuda"),
        block_table,
        context_lens,
        h_dim // 4,
        num_heads,
    )

    attn_torch = bench_paged_attention_torch(
        q, k, v, batch_size, num_heads, h_dim, block_table, context_lens
    ).to(torch.float32)

    tt.assert_close(attn_torch, attn_out, rtol=0.40, atol=2.0)


@pytest.mark.parametrize("batch_size", BATCH_SIZE_PARAMETERS)
@pytest.mark.parametrize("h_dim", H_DIM_PARAMETERS)
@pytest.mark.parametrize("num_heads", NUM_HEADS_PARAMETERS)
@pytest.mark.parametrize("seq_len", SEQ_LEN_PARAMETERS)
def test_paged_attention_fp8_e4m3_feather_gpu(
    benchmark, batch_size, h_dim, num_heads, seq_len, generate_paged_input_tensors
):
    q, k, v, block_table, context_lens, total_blocks = generate_paged_input_tensors

    q_packed_flat = pack_fp8_tensor(q, mode="E4M3").view(torch.uint32)
    q_packed = q_packed_flat.view(batch_size, num_heads, h_dim // 4).contiguous()
    k_perm = k.permute(0, 1, 3, 2)
    k_packed_flat = pack_fp8_tensor(k_perm, mode="E4M3").view(torch.uint32)
    k_packed = k_packed_flat.view(total_blocks, num_heads, BLOCK_SIZE, h_dim // 4)
    k_cache_packed = k_packed.permute(0, 1, 3, 2).contiguous()
    v_perm = v.permute(0, 1, 3, 2)
    v_packed_flat = pack_fp8_tensor(v_perm, mode="E4M3").view(torch.uint32)
    v_packed = v_packed_flat.view(total_blocks, num_heads, BLOCK_SIZE, h_dim // 4)
    v_cache_packed = v_packed.permute(0, 1, 3, 2).contiguous()

    attn_out = benchmark(
        bench_paged_attention_fp8_e4m3_feather_gpu,
        q_packed.to("cuda"),
        k_cache_packed.to("cuda"),
        v_cache_packed.to("cuda"),
        block_table,
        context_lens,
        h_dim // 4,
        num_heads,
    )

    attn_torch = bench_paged_attention_torch(
        q, k, v, batch_size, num_heads, h_dim, block_table, context_lens
    ).to(torch.float32)

    tt.assert_close(attn_torch, attn_out, rtol=0.30, atol=2.0)
