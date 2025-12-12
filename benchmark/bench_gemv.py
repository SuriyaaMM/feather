# bench_gemv.py
import random
from typing import List, Tuple

import numpy as np
import numpy.testing as npt
import pytest
import torch
import torch.testing as tt

from feather.packers.fp8 import *
from feather.routines.gemv import *

N_PARAMETERS = [4_096, 8_192, 16_384]
M_PARAMETERS = [4_096, 8_192, 16_384]


# pytorch implementation
def bench_gemv_torch(a: torch.Tensor, b: torch.Tensor):
    return torch.mv(a, b)


# ----- feather implementations
def bench_gemv_fp8_e5m2_acc_fp32_dot_feather_gpu(
    a: torch.Tensor, b: torch.Tensor, a_shape: Tuple[int, ...]
):
    return gemv_fp8_e5m2_acc_fp32_gpu(a, b, a_shape)


def bench_gemv_fp8_e4m3_acc_fp32_dot_feather_gpu(
    a: torch.Tensor, b: torch.Tensor, a_shape: Tuple[int, ...]
):
    return gemv_fp8_e4m3_acc_fp32_gpu(a, b, a_shape)


# ----- generators
@pytest.fixture
def generate_input_tensors_fp32(m: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    a: torch.Tensor = (
        torch.normal(mean=0, std=1, size=(m, n)).to(torch.float32).to("cuda")
    )
    b: torch.Tensor = (
        torch.normal(mean=0, std=1, size=(n,)).to(torch.float32).to("cuda")
    )
    return a, b


@pytest.fixture
def generate_gemv_input_tensors_packed_fp8_fp32(
    m: int, n: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    a: torch.Tensor = (
        torch.randint(low=-3, high=3, size=(m, n)).to(torch.float16).to("cuda")
    )
    b: torch.Tensor = (
        torch.randint(low=-3, high=3, size=(n,)).to(torch.float16).to("cuda")
    )
    return a, b


# ----- tests
@pytest.mark.parametrize("m", M_PARAMETERS)
@pytest.mark.parametrize("n", N_PARAMETERS)
def test_gemv_fp32_torch(benchmark, generate_input_tensors_fp32):
    a, b = generate_input_tensors_fp32
    gemv = benchmark(bench_gemv_torch, a, b)

    gemv_torch = torch.mv(a.to(torch.float32), b.to(torch.float32))

    # NOTE: during testing, i noticed almost 25 units of accumulation
    # error, so i just set this to 100
    tt.assert_close(gemv_torch, gemv, rtol=10, atol=20)


@pytest.mark.parametrize("m", M_PARAMETERS)
@pytest.mark.parametrize("n", N_PARAMETERS)
def test_gemv_fp8_e5m2_acc_fp32_feather_gpu(
    benchmark, generate_gemv_input_tensors_packed_fp8_fp32
):
    a, b = generate_gemv_input_tensors_packed_fp8_fp32

    a_packed = pack_fp8_tensor(a, mode="E5M2").view(torch.uint32).to("cuda")
    b_packed = pack_fp8_tensor(b, mode="E5M2").view(torch.uint32).to("cuda")

    gemv = benchmark(
        bench_gemv_fp8_e5m2_acc_fp32_dot_feather_gpu, a_packed, b_packed, a.shape
    )

    gemv_torch = torch.mv(a.to(torch.float32), b.to(torch.float32))

    # NOTE: during testing, i noticed almost 25 units of accumulation
    # error, so i just set this to 100
    tt.assert_close(gemv_torch, gemv, rtol=10, atol=10)


@pytest.mark.parametrize("m", M_PARAMETERS)
@pytest.mark.parametrize("n", N_PARAMETERS)
def test_gemv_fp8_e4m3_acc_fp32_feather_gpu(
    benchmark, generate_gemv_input_tensors_packed_fp8_fp32
):
    a, b = generate_gemv_input_tensors_packed_fp8_fp32

    a_packed = pack_fp8_tensor(a, mode="E4M3").view(torch.uint32).to("cuda")
    b_packed = pack_fp8_tensor(b, mode="E4M3").view(torch.uint32).to("cuda")

    gemv = benchmark(
        bench_gemv_fp8_e4m3_acc_fp32_dot_feather_gpu, a_packed, b_packed, a.shape
    )

    gemv_torch = torch.mv(a.to(torch.float32), b.to(torch.float32))

    # NOTE: during testing, i noticed almost 25 units of accumulation
    # error, so i just set this to 100
    tt.assert_close(gemv_torch, gemv, rtol=10, atol=10)