from typing import List, Tuple

import pytest
import torch
import torch.testing as tt

from feather.packers.fp8 import *
from feather.routines.attention import *

SEQ_LEN_PARAMETERS = [128, 256, 512, 1024, 4096]
H_DIM_PARAMETERS = [64, 256, 512]
BATCH_SIZE_PARAMETERS = [1, 8, 32]
NUM_HEADS_PARAMETERS = [4, 8, 16]
BLOCK_SIZE = 16


# pytorch implementation
def bench_attention_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    return torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(dim=0),
        k.unsqueeze(dim=0),
        v.unsqueeze(dim=0),
    ).squeeze(dim=0)


def bench_paged_attention_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_heads, head_dim = q.shape
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

        k_seq = k_seq[:, :seq_len, :]
        v_seq = v_seq[:, :seq_len, :]
        q_curr = q[i].unsqueeze(1)
        out[i] = torch.nn.functional.scaled_dot_product_attention(
            q_curr, k_seq, v_seq, scale=scale
        ).squeeze(1)

    return out


# ----- feather implementations
def bench_flash_attention_fp8_e5m2_acc_fp32_feather_gpu(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int, h_dim: int
):
    return flash_attention_fp8_e5m2_acc_fp32_gpu(q, k, v, seq_len, h_dim)


def bench_flash_attention_fp8_e4m3_acc_fp32_feather_gpu(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int, h_dim: int
):
    return flash_attention_fp8_e4m3_acc_fp32_gpu(q, k, v, seq_len, h_dim)


def bench_paged_attention_fp8_e5m2_feather_gpu(
    q, k_cache, v_cache, block_table, context_lens, h_dim_packed
):
    return paged_attention_fp8_e5m2_acc_fp32_gpu(
        q, k_cache, v_cache, block_table, context_lens, h_dim_packed
    )


# ----- generators
@pytest.fixture
def generate_input_tensors_fp32(
    seq_len: int, h_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q: torch.Tensor = (
        torch.normal(mean=0, std=1, size=(seq_len, h_dim)).to(torch.float32).to("cuda")
    )
    k: torch.Tensor = (
        torch.normal(mean=0, std=1, size=(seq_len, h_dim)).to(torch.float32).to("cuda")
    )
    v: torch.Tensor = (
        torch.normal(mean=0, std=1, size=(seq_len, h_dim)).to(torch.float32).to("cuda")
    )
    return q, k, v


@pytest.fixture
def generate_input_tensors_fp16(
    seq_len: int, h_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    q: torch.Tensor = (
        torch.randint(low=-2, high=2, size=(seq_len, h_dim))
        .to(torch.float16)
        .to("cuda")
    )
    k: torch.Tensor = (
        torch.randint(low=-2, high=2, size=(seq_len, h_dim))
        .to(torch.float16)
        .to("cuda")
    )
    v: torch.Tensor = (
        torch.randint(low=-2, high=2, size=(seq_len, h_dim))
        .to(torch.float16)
        .to("cuda")
    )
    return q, k, v


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
]:

    max_blocks_per_seq = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = batch_size * max_blocks_per_seq

    q_ref = (
        torch.randint(low=-2, high=2, size=(batch_size, num_heads, h_dim))
        .to(torch.float16)
        .to("cuda")
    )
    k_ref = (
        torch.randint(low=-2, high=2, size=(total_blocks, num_heads, h_dim, BLOCK_SIZE))
        .to(torch.float16)
        .to("cuda")
    )
    v_ref = (
        torch.randint(low=-2, high=2, size=(total_blocks, num_heads, h_dim, BLOCK_SIZE))
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

    q_packed_flat = pack_fp8_tensor(q_ref, mode="E5M2").view(torch.uint32)
    q_packed = q_packed_flat.view(batch_size, num_heads, h_dim // 4).contiguous()
    k_perm = k_ref.permute(0, 1, 3, 2)
    k_packed_flat = pack_fp8_tensor(k_perm, mode="E5M2").view(torch.uint32)
    k_packed = k_packed_flat.view(total_blocks, num_heads, BLOCK_SIZE, h_dim // 4)
    k_cache_packed = k_packed.permute(0, 1, 3, 2).contiguous()
    v_perm = v_ref.permute(0, 1, 3, 2)
    v_packed_flat = pack_fp8_tensor(v_perm, mode="E5M2").view(torch.uint32)
    v_packed = v_packed_flat.view(total_blocks, num_heads, BLOCK_SIZE, h_dim // 4)
    v_cache_packed = v_packed.permute(0, 1, 3, 2).contiguous()

    return (
        q_ref.to("cuda"),
        k_ref.to("cuda"),
        v_ref.to("cuda"),
        q_packed.to("cuda"),
        k_cache_packed.to("cuda"),
        v_cache_packed.to("cuda"),
        block_table.to("cuda"),
        context_lens.to("cuda"),
    )


# ----- tests
@pytest.mark.parametrize("seq_len", SEQ_LEN_PARAMETERS)
@pytest.mark.parametrize("h_dim", H_DIM_PARAMETERS)
def test_attention_fp32_torch(benchmark, generate_input_tensors_fp32):
    q, k, v = generate_input_tensors_fp32
    attn_out = benchmark(bench_attention_torch, q, k, v)


@pytest.mark.parametrize("seq_len", SEQ_LEN_PARAMETERS)
@pytest.mark.parametrize("h_dim", H_DIM_PARAMETERS)
def test_attention_fp16_torch(benchmark, generate_input_tensors_fp16):
    q, k, v = generate_input_tensors_fp16
    attn_out = benchmark(bench_attention_torch, q, k, v)


@pytest.mark.parametrize("batch_size", BATCH_SIZE_PARAMETERS)
@pytest.mark.parametrize("seq_len", SEQ_LEN_PARAMETERS)
@pytest.mark.parametrize("h_dim", H_DIM_PARAMETERS)
@pytest.mark.parametrize("num_heads", NUM_HEADS_PARAMETERS)
def test_paged_attention_fp32_torch(benchmark, batch_size, seq_len, h_dim, num_heads):
    (q_ref, k_ref, v_ref, _, _, _, block_table, context_lens) = (
        generate_paged_input_tensors(batch_size, h_dim, num_heads, seq_len)
    )

    benchmark(
        bench_paged_attention_torch, q_ref, k_ref, v_ref, block_table, context_lens
    )


@pytest.mark.parametrize("seq_len", SEQ_LEN_PARAMETERS)
@pytest.mark.parametrize("h_dim", H_DIM_PARAMETERS)
def test_flash_attention_fp8_e5m2_acc_fp32_feather_gpu(
    benchmark, generate_input_tensors_fp16
):
    q, k, v = generate_input_tensors_fp16

    q_packed = pack_fp8_tensor(q, mode="E5M2").view(torch.uint32).to("cuda")
    k_packed = pack_fp8_tensor(k, mode="E5M2").view(torch.uint32).to("cuda")
    v_packed = pack_fp8_tensor(v, mode="E5M2").view(torch.uint32).to("cuda")

    attn_out = benchmark(
        bench_flash_attention_fp8_e5m2_acc_fp32_feather_gpu,
        q_packed,
        k_packed,
        v_packed,
        q.shape[0],
        q.shape[1] // 4,
    )

    # torch attention
    attn_torch = (
        torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(dim=0),
            k.unsqueeze(dim=0),
            v.unsqueeze(dim=0),
        )
        .squeeze(dim=0)
        .to(torch.float32)
    )

    tt.assert_close(attn_torch, attn_out, rtol=10, atol=50)


@pytest.mark.parametrize("seq_len", SEQ_LEN_PARAMETERS)
@pytest.mark.parametrize("h_dim", H_DIM_PARAMETERS)
def test_flash_attention_fp8_e4m3_acc_fp32_feather_gpu(
    benchmark, generate_input_tensors_fp16
):
    q, k, v = generate_input_tensors_fp16

    q_packed = pack_fp8_tensor(q, mode="E4M3").view(torch.uint32).to("cuda")
    k_packed = pack_fp8_tensor(k, mode="E4M3").view(torch.uint32).to("cuda")
    v_packed = pack_fp8_tensor(v, mode="E4M3").view(torch.uint32).to("cuda")

    attn_out = benchmark(
        bench_flash_attention_fp8_e4m3_acc_fp32_feather_gpu,
        q_packed,
        k_packed,
        v_packed,
        q.shape[0],
        q.shape[1] // 4,
    )

    # torch attention
    attn_torch = (
        torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(dim=0),
            k.unsqueeze(dim=0),
            v.unsqueeze(dim=0),
        )
        .squeeze(dim=0)
        .to(torch.float32)
    )

    tt.assert_close(attn_torch, attn_out, rtol=10, atol=50)


@pytest.mark.parametrize("batch_size", BATCH_SIZE_PARAMETERS)
@pytest.mark.parametrize("seq_len", SEQ_LEN_PARAMETERS)
@pytest.mark.parametrize("h_dim", H_DIM_PARAMETERS)
@pytest.mark.parametrize("num_heads", NUM_HEADS_PARAMETERS)
def test_paged_attention_fp8_e5m2_feather_gpu(
    benchmark, batch_size, seq_len, h_dim, num_heads, generate_paged_input_tensors
):
    (q_ref, k_ref, v_ref, q_packed, k_packed, v_packed, block_table, context_lens) = (
        generate_paged_input_tensors
    )

    attn_out = benchmark(
        bench_paged_attention_fp8_e5m2_feather_gpu,
        q_packed,
        k_packed,
        v_packed,
        block_table,
        context_lens,
        h_dim // 4,
    )

    attn_torch = bench_paged_attention_torch(
        q_ref, k_ref, v_ref, block_table, context_lens
    ).to(torch.float32)

    tt.assert_close(attn_torch, attn_out, rtol=10, atol=50)
