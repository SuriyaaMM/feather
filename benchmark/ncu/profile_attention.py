import torch
import feather
import math
from feather.packers.fp8 import pack_fp8_tensor
from feather.routines.attention import *

def profile_flash_attention_fp8_e5m2_acc_fp32_gpu():
    M, N = 8192, 512

    q = torch.randn(size=(M, N), dtype=torch.float16, device="cuda")
    k = torch.randn(size=(M, N), dtype=torch.float16, device="cuda")
    v = torch.randn(size=(M, N), dtype=torch.float16, device="cuda")

    q_packed = pack_fp8_tensor(q, mode="E5M2").view(torch.uint32).cuda()
    k_packed = pack_fp8_tensor(k, mode="E5M2").view(torch.uint32).cuda()
    v_packed = pack_fp8_tensor(v, mode="E5M2").view(torch.uint32).cuda()

    flash_attention_fp8_e5m2_acc_fp32_gpu(q_packed, k_packed, v_packed, M, N // 4)
    torch.cuda.synchronize()
    flash_attention_fp8_e5m2_acc_fp32_gpu(q_packed, k_packed, v_packed, M, N // 4)
    torch.cuda.synchronize()

def profile_attention_torch():
    M, N = 8192, 512

    q = torch.randn(size=(M, N), dtype=torch.float32, device="cuda")
    k = torch.randn(size=(M, N), dtype=torch.float32, device="cuda")
    v = torch.randn(size=(M, N), dtype=torch.float32, device="cuda")

    out = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(dim=0),
        k.unsqueeze(dim=0),
        v.unsqueeze(dim=0),
    ).squeeze(dim=0)
    torch.cuda.synchronize()
    out = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(dim=0),
        k.unsqueeze(dim=0),
        v.unsqueeze(dim=0),
    ).squeeze(dim=0)
    torch.cuda.synchronize()

    return out

if __name__ == "__main__":
    profile_flash_attention_fp8_e5m2_acc_fp32_gpu()
    profile_attention_torch()