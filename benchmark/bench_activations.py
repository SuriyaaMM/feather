from typing import List, Tuple
import math

import pytest
import torch
import torch.testing as tt

from feather.packers.fp8 import *
from feather.routines.activations import *

N_PARAMETERS = [1_500_000, 7_500_000, 12_500_000, 50_000_000]

# pytorch implementation
def bench_relu1d_torch(x:torch.Tensor):
    out = torch.nn.functional.relu(x)
    torch.cuda.synchronize()
    return out

# ----- feather implementations
def bench_relu1d_fp8_ret_fp32_feather_gpu(x:torch.Tensor, n:int):
    out = relu1d_fp8_ret_fp32_feather_gpu(x, n)
    torch.cuda.synchronize()
    return out

# ----- generators
@pytest.fixture
def generate_input_tensors_1d(n:int) -> torch.Tensor:
    return torch.randint(low=-4, high=5, size=(n,)).to(torch.float16).cuda()

# ----- tests
@pytest.mark.parametrize("n", N_PARAMETERS)
def test_relu1d_torch(benchmark, generate_input_tensors_1d):
    x = generate_input_tensors_1d
    relu_out = benchmark(bench_relu1d_torch, x)

@pytest.mark.parametrize("n", N_PARAMETERS)
def test_relu1d_fp8_ret_fp32_feather_gpu(benchmark, generate_input_tensors_1d):
    x = generate_input_tensors_1d
    x_packed = pack_fp8_tensor(x).to(torch.int32).cuda()
    relu_out = benchmark(bench_relu1d_fp8_ret_fp32_feather_gpu, x_packed, x.shape[0])
    relu_torch = torch.nn.functional.relu(x.cuda()).to(torch.float32).cuda()
    tt.assert_close(relu_torch, relu_out)
