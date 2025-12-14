import torch
import feather
import math
from feather.packers.fp8 import pack_fp8_tensor
from feather.routines.attention import *

def profile_attention_fp8_e5m2_acc_fp32_gpu():
    M, N = 8192, 512

    q = torch.randn(size=(M, N), dtype=torch.float16, device="cuda")
    k = torch.randn(size=(M, N), dtype=torch.float16, device="cuda")
    v = torch.randn(size=(M, N), dtype=torch.float16, device="cuda")

    q_packed = pack_fp8_tensor(q, mode="E5M2").view(torch.uint32).cuda()
    k_packed = pack_fp8_tensor(k, mode="E5M2").view(torch.uint32).cuda()
    v_packed = pack_fp8_tensor(v, mode="E5M2").view(torch.uint32).cuda()

    attention_fp8_e5m2_acc_fp32_gpu(q_packed, k_packed, v_packed, M, N)
    torch.cuda.synchronize()
    attention_fp8_e5m2_acc_fp32_gpu(q_packed, k_packed, v_packed, M, N)
    torch.cuda.synchronize()

def profile_attention_fp8_e5m2_acc_fp32_tiled_gpu():
    M, N = 8192, 512

    q = torch.randn(size=(M, N), dtype=torch.float16, device="cuda")
    k = torch.randn(size=(M, N), dtype=torch.float16, device="cuda")
    v = torch.randn(size=(M, N), dtype=torch.float16, device="cuda")

    q_packed = pack_fp8_tensor(q, mode="E5M2").view(torch.uint32).cuda()
    k_packed = pack_fp8_tensor(k, mode="E5M2").view(torch.uint32).cuda()
    v_packed = pack_fp8_tensor(v, mode="E5M2").view(torch.uint32).cuda()

    attention_fp8_e5m2_acc_fp32_tiled_gpu(q_packed, k_packed, v_packed, M, N)
    torch.cuda.synchronize()
    attention_fp8_e5m2_acc_fp32_tiled_gpu(q_packed, k_packed, v_packed, M, N)
    torch.cuda.synchronize()

def profile_attention_torch():
    M, N = 8192, 512

    q = torch.randn(size=(M, N), dtype=torch.float32, device="cuda")
    k = torch.randn(size=(M, N), dtype=torch.float32, device="cuda")
    v = torch.randn(size=(M, N), dtype=torch.float32, device="cuda")

    scale = 1.0 / math.sqrt(q.size(-1))
    
    attn_weight = q @ k.transpose(-2, -1) * scale
    attn_weight = torch.softmax(attn_weight, dim=-1)
    out = attn_weight @ v
    
    torch.cuda.synchronize()

    scale = 1.0 / math.sqrt(q.size(-1))
    
    attn_weight = q @ k.transpose(-2, -1) * scale
    attn_weight = torch.softmax(attn_weight, dim=-1)
    out = attn_weight @ v
    torch.cuda.synchronize()

    return out

if __name__ == "__main__":
    profile_attention_fp8_e5m2_acc_fp32_tiled_gpu()
    profile_attention_fp8_e5m2_acc_fp32_gpu()
    profile_attention_torch()