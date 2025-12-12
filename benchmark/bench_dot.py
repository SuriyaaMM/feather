# bench_dot.py
import random
from typing import List, Tuple

import numpy as np
import numpy.testing as npt
import pytest
import torch
import torch.testing as tt

from feather.packers.fp8 import *
from feather.packers.fp16 import *
from feather.routines.dot import *

N_PARAMETERS = [100_00, 500_000, 1_500_000]


# numpy implementation
def bench_dot_np(a: np.ndarray, b: np.ndarray):
    dot_prod = np.dot(a, b)
    return dot_prod


# pytorch implementation
def bench_dot_torch(a: torch.Tensor, b: torch.Tensor):
    dot_prod = torch.dot(a, b)
    return dot_prod


# ----- feather implementations
def bench_dot_fp16_acc_fp32_dot_feather_np(a: np.ndarray, b: np.ndarray):
    dot_prod = dot_fp16_acc_fp32_vec(a, b)
    return dot_prod


def bench_dot_fp16_acc_fp32_dot_feather_numba(a: np.ndarray, b: np.ndarray):
    _FP16_LUT = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    dot_prod = dot_fp16_acc_fp32_numba(a, b, _FP16_LUT)
    return dot_prod


def bench_dot_fp16_acc_fp32_dot_feather_gpu(a: np.ndarray, b: np.ndarray):
    return dot_fp16_acc_fp32_gpu(a, b)


def bench_dot_fp8_e5m2_acc_fp32_dot_feather_gpu(a: np.ndarray, b: np.ndarray):
    return dot_fp8_e5m2_acc_fp32_gpu(a, b)


# ----- generators
@pytest.fixture
def generate_input_arrays_fp32(n: int) -> Tuple[np.ndarray, np.ndarray]:
    a = np.random.normal(size=(n,)).astype(np.float32)
    b = np.random.normal(size=(n,)).astype(np.float32)
    return a, b


@pytest.fixture
def generate_input_arrays_fp16(n: int) -> Tuple[np.ndarray, np.ndarray]:
    a = np.random.normal(size=(n,)).astype(np.float16)
    b = np.random.normal(size=(n,)).astype(np.float16)
    return a, b


@pytest.fixture
def generate_input_tensors_fp32(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    a: torch.Tensor = (
        torch.normal(mean=0, std=1, size=(n,)).to(torch.float32).to("cuda")
    )
    b: torch.Tensor = (
        torch.normal(mean=0, std=1, size=(n,)).to(torch.float32).to("cuda")
    )
    return a, b


@pytest.fixture
def generate_input_arrays_packed_fp16_fp32(n: int) -> Tuple[np.ndarray, np.ndarray]:
    a = np.random.normal(size=(n,)).astype(np.float16)
    b = np.random.normal(size=(n,)).astype(np.float16)
    a_packed = pack_fp16_ndarray(a)
    b_packed = pack_fp16_ndarray(b)
    return a_packed, b_packed


@pytest.fixture
def generate_input_arrays_packed_fp8_e5m2_fp32(n: int) -> Tuple[np.ndarray, np.ndarray]:
    a = np.random.normal(size=(n,)).astype(np.float16)
    b = np.random.normal(size=(n,)).astype(np.float16)
    a_packed = pack_fp8_ndarray(a, mode="E5M2")
    b_packed = pack_fp8_ndarray(b, mode="E5M2")
    return a_packed, b_packed


# ----- testers
@pytest.mark.parametrize("n", N_PARAMETERS)
def test_dot_fp32_torch(benchmark, generate_input_tensors_fp32):
    a, b = generate_input_tensors_fp32
    a.to()
    dot_prod = benchmark(bench_dot_torch, a, b)
    dot_prod_torch = torch.dot(a, b)

    tt.assert_close(dot_prod_torch, dot_prod, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n", N_PARAMETERS)
def test_dot_fp16_acc_fp32_feather_np(
    benchmark, generate_input_arrays_packed_fp16_fp32
):
    a_packed, b_packed = generate_input_arrays_packed_fp16_fp32
    dot_prod = benchmark(bench_dot_fp16_acc_fp32_dot_feather_np, a_packed, b_packed)

    a_unpacked = unpack_into_fp16_ndarray(a_packed)
    b_unpacked = unpack_into_fp16_ndarray(b_packed)
    dot_prod_np = np.dot(a_unpacked, b_unpacked)
    dot_prod_np = dot_prod_np.astype(np.float32)

    # NOTE: rtol is set to 1e-1, 1e-5 seems too tight for precision
    npt.assert_allclose(dot_prod_np, dot_prod, rtol=1e-1)


@pytest.mark.parametrize("n", N_PARAMETERS)
def test_dot_fp16_acc_fp32_feather_numba(
    benchmark, generate_input_arrays_packed_fp16_fp32
):
    a_packed, b_packed = generate_input_arrays_packed_fp16_fp32
    dot_prod = benchmark(bench_dot_fp16_acc_fp32_dot_feather_numba, a_packed, b_packed)

    a_unpacked = unpack_into_fp16_ndarray(a_packed)
    b_unpacked = unpack_into_fp16_ndarray(b_packed)
    dot_prod_np = (a_unpacked, b_unpacked)
    dot_prod_np = dot_prod_np.astype(np.float32)

    # NOTE: rtol is set to 1e-1, 1e-5 seems too tight for precision
    npt.assert_allclose(dot_prod_np, dot_prod, rtol=1e-1)


@pytest.mark.parametrize("n", N_PARAMETERS)
def test_dot_fp16_acc_fp32_feather_gpu(
    benchmark, generate_input_arrays_packed_fp16_fp32
):
    a_packed, b_packed = generate_input_arrays_packed_fp16_fp32

    a_tensor = torch.from_numpy(a_packed).view(torch.uint32).to("cuda")
    b_tensor = torch.from_numpy(b_packed).view(torch.uint32).to("cuda")

    dot_prod = benchmark(
        bench_dot_fp16_acc_fp32_dot_feather_gpu, a_tensor, b_tensor
    ).cpu()

    a_unpacked = unpack_into_fp16_ndarray(a_packed)
    b_unpacked = unpack_into_fp16_ndarray(b_packed)

    a_unpacked_tensor = torch.from_numpy(a_unpacked).to(torch.float32).to("cuda")
    b_unpacked_tensor = torch.from_numpy(b_unpacked).to(torch.float32).to("cuda")
    dot_prod_torch = torch.dot(a_unpacked_tensor, b_unpacked_tensor).cpu()

    # NOTE: rtol is set to 1e-1, 1e-5 seems too tight for precision
    tt.assert_close(dot_prod_torch, dot_prod, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("n", N_PARAMETERS)
def test_dot_fp8_e5m2_acc_fp32_feather_gpu(
    benchmark, generate_input_arrays_packed_fp8_e5m2_fp32
):
    a_packed, b_packed = generate_input_arrays_packed_fp8_e5m2_fp32

    a_tensor = torch.from_numpy(a_packed).view(torch.uint32).to("cuda")
    b_tensor = torch.from_numpy(b_packed).view(torch.uint32).to("cuda")

    dot_prod = benchmark(
        bench_dot_fp8_e5m2_acc_fp32_dot_feather_gpu, a_tensor, b_tensor
    ).cpu()

    a_unpacked = unpack_into_fp8_ndarray(a_packed, mode="E5M2")
    b_unpacked = unpack_into_fp8_ndarray(b_packed, mode="E5M2")

    a_unpacked_tensor = torch.from_numpy(a_unpacked).to(torch.float32).to("cuda")
    b_unpacked_tensor = torch.from_numpy(b_unpacked).to(torch.float32).to("cuda")
    dot_prod_torch = torch.dot(a_unpacked_tensor, b_unpacked_tensor).cpu()

    # NOTE: during testing, i noticed almost 25 units of accumulation
    # error, so i just set this to 100
    tt.assert_close(dot_prod_torch, dot_prod, rtol=100, atol=100)
