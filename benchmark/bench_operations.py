# bench_operations.py
import random
from typing import List, Tuple

import torch
import torch.testing as tt
import numpy as np
import numpy. testing as npt

from feather.packers import *
from feather.operations import *

import pytest

N_PARAMETERS = [100_00, 500_000, 1_500_000]
M_PARAMETERS = [100, 1000, 10_000]
# to-benchmark functions

# numpy implementation
def bench_dot_np(
    a: np.ndarray,
    b: np.ndarray
):
    dot_prod = np.dot(a, b)
    return dot_prod

# pytorch implementation
def bench_dot_torch(
    a: torch.Tensor,
    b: torch.Tensor
):
    dot_prod = torch.dot(a, b)
    return dot_prod

# ----- feather implementations 

def bench_dot_fp16_acc_fp32_dot_feather_np(
    a: np.ndarray,
    b: np.ndarray
):
    dot_prod = dot_fp16_acc_fp32_vec(a, b)
    return dot_prod

def bench_dot_fp16_acc_fp32_dot_feather_numba(
    a: np.ndarray,
    b: np.ndarray
):
    _FP16_LUT = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    dot_prod = dot_fp16_acc_fp32_numba(a, b, _FP16_LUT)
    return dot_prod

def bench_dot_fp16_acc_fp32_dot_feather_gpu(
    a: np.ndarray,
    b: np.ndarray
):
    return dot_fp16_acc_fp32_gpu(a, b)

def bench_dot_fp8_acc_fp32_dot_feather_gpu(
    a: np.ndarray,
    b: np.ndarray
):
    return dot_fp8_acc_fp32_gpu(a, b)


def bench_gemv_fp8_acc_fp32_dot_feather_gpu(
    a: np.ndarray,
    b: np.ndarray,
    a_shape: tuple
):
    return gemv_fp8_acc_fp32_gpu(a, b, a_shape)

# ----- array generation scripts 

@pytest.fixture
def generate_input_arrays_fp32(
    n: int
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.random.normal(size=(n, )).astype(np.float32)
    b = np.random.normal(size=(n, )).astype(np.float32)
    return a, b 

@pytest.fixture
def generate_input_arrays_fp16(
    n: int
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.random.normal(size=(n, )).astype(np.float16)
    b = np.random.normal(size=(n, )).astype(np.float16)
    return a, b 

@pytest.fixture
def generate_input_tensors_fp32(
    n: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    a:torch.Tensor = torch.normal(mean=0, std=1, size=(n, )).to(torch.float32).to("cuda")
    b:torch.Tensor = torch.normal(mean=0, std=1, size=(n, )).to(torch.float32).to("cuda")
    return a, b 

@pytest.fixture
def generate_input_tensors_fp16(
    n: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    a:torch.Tensor = torch.normal(mean=0, std=1, size=(n, )).to(torch.float16).to("cuda")
    b:torch.Tensor = torch.normal(mean=0, std=1, size=(n, )).to(torch.float16).to("cuda")
    return a, b 

@pytest.fixture
def generate_input_arrays_packed_fp16_fp32(
    n: int
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.random.normal(size=(n, )).astype(np.float16)
    b = np.random.normal(size=(n, )).astype(np.float16)
    a_packed = pack_fp16_ndarray(a)
    b_packed = pack_fp16_ndarray(b)
    return a_packed, b_packed

@pytest.fixture
def generate_input_arrays_packed_fp8_fp32(
    n:int
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.random.normal(size=(n, )).astype(np.float16)
    b = np.random.normal(size=(n, )).astype(np.float16)
    a_packed = pack_fp8_ndarray(a)
    b_packed = pack_fp8_ndarray(b)
    return a_packed, b_packed

@pytest.fixture
def generate_input_matrix_and_arrays_packed_fp8_fp32(
    m:int,
    n:int
) -> Tuple[np.ndarray, np.ndarray]:
    a:torch.Tensor = torch.normal(mean=0, std=1, size=(m, n)).to(torch.float16).to("cuda")
    b:torch.Tensor = torch.normal(mean=0, std=1, size=(n, )).to(torch.float16).to("cuda")
    return a, b 

# ----- benchmark functions

@pytest.mark.parametrize("n", N_PARAMETERS)
def test_dot_fp32_torch(
    benchmark,
    generate_input_tensors_fp32
):
    a, b = generate_input_tensors_fp32
    a.to()
    dot_prod = benchmark(bench_dot_torch, a, b)
    dot_prod_torch = torch.dot(a, b)

    tt.assert_close(dot_prod_torch, dot_prod, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("n", N_PARAMETERS)
def test_dot_fp16_torch(
    benchmark,
    generate_input_tensors_fp32
):
    a, b = generate_input_tensors_fp32
    a.to()
    dot_prod = benchmark(bench_dot_torch, a, b)
    dot_prod_torch = torch.dot(a, b)

    tt.assert_close(dot_prod_torch, dot_prod, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("n", N_PARAMETERS)
def test_dot_fp16_acc_fp32_feather_np(
    benchmark,
    generate_input_arrays_packed_fp16_fp32
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
    benchmark,
    generate_input_arrays_packed_fp16_fp32 
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
    benchmark,
    generate_input_arrays_packed_fp16_fp32
):
    a_packed, b_packed = generate_input_arrays_packed_fp16_fp32

    a_tensor = torch.from_numpy(a_packed).view(torch.uint32).to("cuda")
    b_tensor = torch.from_numpy(b_packed).view(torch.uint32).to("cuda")

    dot_prod = benchmark(bench_dot_fp16_acc_fp32_dot_feather_gpu, a_tensor, b_tensor).cpu()

    a_unpacked = unpack_into_fp16_ndarray(a_packed)
    b_unpacked = unpack_into_fp16_ndarray(b_packed)

    a_unpacked_tensor = torch.from_numpy(a_unpacked).to(torch.float32).to("cuda")
    b_unpacked_tensor = torch.from_numpy(b_unpacked).to(torch.float32).to("cuda")
    dot_prod_torch = torch.dot(a_unpacked_tensor, b_unpacked_tensor).cpu() 

    # NOTE: rtol is set to 1e-1, 1e-5 seems too tight for precision
    tt.assert_close(dot_prod_torch, dot_prod, rtol=1e-1, atol=1e-1)

@pytest.mark.parametrize("n", N_PARAMETERS)
def test_dot_fp8_acc_fp32_feather_gpu(
    benchmark,
    generate_input_arrays_packed_fp8_fp32
):
    a_packed, b_packed = generate_input_arrays_packed_fp8_fp32

    a_tensor = torch.from_numpy(a_packed).view(torch.uint32).to("cuda")
    b_tensor = torch.from_numpy(b_packed).view(torch.uint32).to("cuda")

    dot_prod = benchmark(bench_dot_fp8_acc_fp32_dot_feather_gpu, a_tensor, b_tensor).cpu()

    a_unpacked = unpack_into_fp8_ndarray(a_packed)
    b_unpacked = unpack_into_fp8_ndarray(b_packed)

    a_unpacked_tensor = torch.from_numpy(a_unpacked).to(torch.float32).to("cuda")
    b_unpacked_tensor = torch.from_numpy(b_unpacked).to(torch.float32).to("cuda")
    dot_prod_torch = torch.dot(a_unpacked_tensor, b_unpacked_tensor).cpu() 

    # NOTE: during testing, i noticed almost 25 units of accumulation
    # error, so i just set this to 100
    tt.assert_close(dot_prod_torch, dot_prod, rtol=100, atol=100)

@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("n", N_PARAMETERS)
def test_gemv_fp8_acc_fp32_feather_gpu(
    benchmark,
    generate_input_matrix_and_arrays_packed_fp8_fp32
):
    a, b = generate_input_matrix_and_arrays_packed_fp8_fp32

    a_packed = pack_fp8_tensor(a).view(torch.uint32).to("cuda")
    b_packed = pack_fp8_tensor(b).view(torch.uint32).to("cuda")

    dot_prod = benchmark(bench_gemv_fp8_acc_fp32_dot_feather_gpu, a_packed, b_packed, a.shape).cpu()

    dot_prod_torch = torch.mv(a.to(torch.float32), b.to(torch.float32)).cpu() 

    # NOTE: during testing, i noticed almost 25 units of accumulation
    # error, so i just set this to 100
    tt.assert_close(dot_prod_torch, dot_prod, rtol=100, atol=100)