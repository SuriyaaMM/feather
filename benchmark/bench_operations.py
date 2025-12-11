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

PARAMETERIZE_LIST = [100_000, 500_000, 1_500_000]

# to-benchmark functions

def bench_dot_python_eager(
    a: List[float],
    b: List[float]
):
    dot_prod = 0.0
    for a_i, b_i in zip(a, b):
        dot_prod += (a_i * b_i)

    return dot_prod

def bench_dot_np(
    a: np.ndarray,
    b: np.ndarray
):
    dot_prod = np.dot(a, b)
    return dot_prod

def bench_dot_torch(
    a: torch.Tensor,
    b: torch.Tensor
):
    dot_prod = torch.dot(a, b)
    return dot_prod

def bench_dot_feather_np(
    a: np.ndarray,
    b: np.ndarray
):
    dot_prod = dot_fp16_acc_fp32_vec(a, b)
    return dot_prod

def bench_dot_feather_numba(
    a: np.ndarray,
    b: np.ndarray
):
    _FP16_LUT = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    dot_prod = dot_fp16_acc_fp32_numba(a, b, _FP16_LUT)
    return dot_prod

def bench_dot_feather_gpu(
    a: np.ndarray,
    b: np.ndarray
):
    return dot_fp16_acc_fp32_gpu(a, b)

# array generation scripts

@pytest.fixture
def generate_input_lists(
    n: int
) -> Tuple[List, List]:
    a = [random.gauss() for i in range(n)]
    b = [random.gauss() for i in range(n)]
    return a, b

@pytest.fixture
def generate_input_arrays(
    n: int
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.random.normal(size=(n, )).astype(np.float32)
    b = np.random.normal(size=(n, )).astype(np.float32)
    return a, b 

@pytest.fixture
def generate_input_tensors(
    n: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    a:torch.Tensor = torch.normal(mean=0, std=1, size=(n, )).to(torch.float32).to("cuda")
    b:torch.Tensor = torch.normal(mean=0, std=1, size=(n, )).to(torch.float32).to("cuda")
    return a, b 

@pytest.fixture
def generate_input_arrays_packed(
    n: int
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.random.normal(size=(n, )).astype(np.float16)
    b = np.random.normal(size=(n, )).astype(np.float16)
    a_packed = pack_fp16_ndarray(a)
    b_packed = pack_fp16_ndarray(b)
    return a_packed, b_packed

# ----- benchmark functions

@pytest.mark.parametrize("n", PARAMETERIZE_LIST)
def test_dot_python_eager(
    benchmark,
    generate_input_lists
):
    a, b = generate_input_lists
    dot_prod = benchmark(bench_dot_python_eager, a, b)
    dot_prod_np = np.dot(a, b)

    npt.assert_allclose(dot_prod_np, dot_prod, rtol=1e-5)

@pytest.mark.parametrize("n", PARAMETERIZE_LIST)
def test_dot_np(
    benchmark,
    generate_input_arrays
):
    a, b = generate_input_arrays
    dot_prod = benchmark(bench_dot_np, a, b)
    dot_prod_np = np.dot(a, b)

    npt.assert_allclose(dot_prod_np, dot_prod, rtol=1e-5)

@pytest.mark.parametrize("n", PARAMETERIZE_LIST)
def test_dot_torch(
    benchmark,
    generate_input_tensors
):
    a, b = generate_input_tensors
    a.to()
    dot_prod = benchmark(bench_dot_torch, a, b)
    dot_prod_torch = torch.dot(a, b)

    tt.assert_close(dot_prod_torch, dot_prod, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("n", PARAMETERIZE_LIST)
def test_dot_feather_np(
    benchmark,
    generate_input_arrays_packed 
):
    a_packed, b_packed = generate_input_arrays_packed
    dot_prod = benchmark(bench_dot_feather_np, a_packed, b_packed)

    a_unpacked = unpack_fp32_ndarray(a_packed)
    b_unpacked = unpack_fp32_ndarray(b_packed)
    dot_prod_np = np.dot(a_unpacked, b_unpacked)
    dot_prod_np = dot_prod_np.astype(np.float32)

    # NOTE: rtol is set to 1e-1, 1e-5 seems too tight for precision
    npt.assert_allclose(dot_prod_np, dot_prod, rtol=1e-1)

@pytest.mark.parametrize("n", PARAMETERIZE_LIST)
def test_dot_feather_np(
    benchmark,
    generate_input_arrays_packed 
):
    a_packed, b_packed = generate_input_arrays_packed
    a_tensor = torch.from_numpy(a_packed).view(torch.uint32).to("cuda")
    b_tensor = torch.from_numpy(b_packed).view(torch.uint32).to("cuda")

    # compiles the numba kernel
    bench_dot_feather_gpu(a_tensor, b_tensor)

    dot_prod = benchmark(bench_dot_feather_gpu, a_tensor, b_tensor).cpu()

    a_unpacked = unpack_fp32_ndarray(a_packed)
    b_unpacked = unpack_fp32_ndarray(b_packed)
    dot_prod_np = (a_unpacked, b_unpacked)
    dot_prod_np = dot_prod_np.astype(np.float32)

    # NOTE: rtol is set to 1e-1, 1e-5 seems too tight for precision
    npt.assert_allclose(dot_prod_np, dot_prod, rtol=1e-1)

@pytest.mark.parametrize("n", PARAMETERIZE_LIST)
def test_dot_feather_gpu(
    benchmark,
    generate_input_arrays_packed
):
    a_packed, b_packed = generate_input_arrays_packed

    a_tensor = torch.from_numpy(a_packed).view(torch.uint32).to("cuda")
    b_tensor = torch.from_numpy(b_packed).view(torch.uint32).to("cuda")

    dot_prod = benchmark(bench_dot_feather_gpu, a_tensor, b_tensor).cpu()

    a_unpacked = unpack_fp32_ndarray(a_packed)
    b_unpacked = unpack_fp32_ndarray(b_packed)

    a_unpacked_tensor = torch.from_numpy(a_unpacked).to(torch.float32).to("cuda")
    b_unpacked_tensor = torch.from_numpy(b_unpacked).to(torch.float32).to("cuda")
    dot_prod_torch = torch.dot(a_unpacked_tensor, b_unpacked_tensor).cpu() 

    # NOTE: rtol is set to 1e-1, 1e-5 seems too tight for precision
    tt.assert_close(dot_prod_torch, dot_prod, rtol=1e-1, atol=1e-1)