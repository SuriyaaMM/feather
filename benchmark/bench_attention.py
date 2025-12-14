import random
from typing import List, Tuple
import math

import pytest
import torch
import torch.nn.functional as F
import torch.testing as tt

from feather.packers.fp8 import *
from feather.routines.attention import *

SEQ_LEN_PARAMETERS = [128, 256, 512, 1024, 4096]
H_DIM_PARAMETERS = [64, 128, 512]

# pytorch implementation
def bench_attention_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    scale = 1.0 / math.sqrt(q.size(-1))
    attn_weight = q @ k.transpose(-2, -1) * scale
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ v


# ----- feather implementations
def bench_attention_fp8_e5m2_acc_fp32_feather_gpu(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int, h_dim: int
):
    return attention_fp8_e5m2_acc_fp32_gpu(q, k, v, seq_len, h_dim)


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
def generate_attn_input_tensors_packed_fp8_fp32(
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

    scale = 1.0 / math.sqrt(q.size(-1))
    ref_weights = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
    ref_out = ref_weights @ v
    
    tt.assert_close(attn_out, ref_out, rtol=1e-3, atol=1e-3)

fast_attention = torch.compile(bench_attention_torch)
@pytest.mark.parametrize("seq_len", SEQ_LEN_PARAMETERS)
@pytest.mark.parametrize("h_dim", H_DIM_PARAMETERS)
def test_attention_fp32_torch_compiled(benchmark, generate_input_tensors_fp32):
    q, k, v = generate_input_tensors_fp32
    fast_attention(q, k, v) 
    attn_out = benchmark(fast_attention, q, k, v)

    scale = 1.0 / math.sqrt(q.size(-1))
    ref_weights = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
    ref_out = ref_weights @ v
    
    tt.assert_close(attn_out, ref_out, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("seq_len", SEQ_LEN_PARAMETERS)
@pytest.mark.parametrize("h_dim", H_DIM_PARAMETERS)
def test_attention_fp8_e5m2_acc_fp32_feather_gpu(
    benchmark, generate_attn_input_tensors_packed_fp8_fp32
):
    q, k, v = generate_attn_input_tensors_packed_fp8_fp32

    q_packed = pack_fp8_tensor(q, mode="E5M2").view(torch.uint32).to("cuda")
    k_packed = pack_fp8_tensor(k, mode="E5M2").view(torch.uint32).to("cuda")
    v_packed = pack_fp8_tensor(v, mode="E5M2").view(torch.uint32).to("cuda")

    attn_out = benchmark(
        bench_attention_fp8_e5m2_acc_fp32_feather_gpu, 
        q_packed, k_packed, v_packed, 
        q.shape[0], q.shape[1]
    )

    # torch attention
    scale = 1.0 / math.sqrt(q.size(-1))
    q_f32, k_f32, v_f32 = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    ref_weights = torch.softmax((q_f32 @ k_f32.transpose(-2, -1)) * scale, dim=-1)
    ref_out = ref_weights @ v_f32

    tt.assert_close(ref_out, attn_out, rtol=10, atol=50)