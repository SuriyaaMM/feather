import torch
import feather
from feather.packers.fp8 import pack_fp8_tensor
from feather.routines.gemv import *

def profile_gemv_torch():
    M, N = 16384, 16384

    a = torch.randn(M, N, dtype=torch.float16, device="cuda")
    b = torch.randn(N, dtype=torch.float16, device="cuda")

    torch.mv(a, b)
    torch.cuda.synchronize()

    torch.mv(a, b)
    torch.cuda.synchronize()

def profile_gemv_fp8_e5m2_acc_fp32_gpu():
    M, N = 16384, 16384

    a = torch.randn(M, N, dtype=torch.float16, device="cuda")
    b = torch.randn(N, dtype=torch.float16, device="cuda")

    a_packed = pack_fp8_tensor(a, mode="E5M2").view(torch.uint32).cuda()
    b_packed = pack_fp8_tensor(b, mode="E5M2").view(torch.uint32).cuda()

    gemv_fp8_e5m2_acc_fp32_gpu(a_packed, b_packed, (M, N))
    torch.cuda.synchronize()

    gemv_fp8_e5m2_acc_fp32_gpu(a_packed, b_packed, (M, N))
    torch.cuda.synchronize()

def profile_gemv_fp8_e4m3_acc_fp32_gpu():
    M, N = 16384, 16384

    a = torch.randn(M, N, dtype=torch.float16, device="cuda")
    b = torch.randn(N, dtype=torch.float16, device="cuda")

    a_packed = pack_fp8_tensor(a, mode="E5M2").view(torch.uint32).cuda()
    b_packed = pack_fp8_tensor(b, mode="E5M2").view(torch.uint32).cuda()

    gemv_fp8_e4m3_acc_fp32_gpu(a_packed, b_packed, (M, N))
    torch.cuda.synchronize()

    gemv_fp8_e4m3_acc_fp32_gpu(a_packed, b_packed, (M, N))
    torch.cuda.synchronize()

if __name__ == "__main__":
    profile_gemv_torch()
    profile_gemv_fp8_e4m3_acc_fp32_gpu()
    profile_gemv_fp8_e5m2_acc_fp32_gpu()