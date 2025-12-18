import random
from typing import List, Tuple
import math

import pytest
import torch
import torch.nn.functional as F
import torch.testing as tt

from feather.packers.fp8 import *
from feather.routines.attention import *

SEQ_LEN_PARAMETERS = [128, 256, 512, 1024, 4096, 8192]
H_DIM_PARAMETERS = [64, 256, 512]

# pytorch implementation
def bench_attention_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    return torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(dim=0),
        k.unsqueeze(dim=0),
        v.unsqueeze(dim=0),
    ).squeeze(dim=0)

# ----- feather implementations
def bench_flash_attention_fp8_e5m2_acc_fp32_feather_gpu(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int, h_dim: int
):
    return flash_attention_fp8_e5m2_acc_fp32_gpu(q, k, v, seq_len, h_dim)


# ----- generators
@pytest.fixture
def generate_input_tensors_fp32(seq_len: int, h_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        torch.randint(low=-2, high=2, size=(seq_len, h_dim)).to(torch.float16).to("cuda")
    )
    k: torch.Tensor = (
        torch.randint(low=-2, high=2, size=(seq_len, h_dim)).to(torch.float16).to("cuda")
    )
    v: torch.Tensor = (
        torch.randint(low=-2, high=2, size=(seq_len, h_dim)).to(torch.float16).to("cuda")
    )
    return q, k, v


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
        q_packed, k_packed, v_packed, 
        q.shape[0], q.shape[1] // 4
    )

    # torch attention
    attn_torch = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(dim=0),
        k.unsqueeze(dim=0),
        v.unsqueeze(dim=0),
    ).squeeze(dim=0).to(torch.float32)

    tt.assert_close(attn_torch, attn_out, rtol=10, atol=50)