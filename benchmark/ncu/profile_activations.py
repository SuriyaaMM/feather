import torch
import math
from feather.packers.fp8 import pack_fp8_tensor
from feather.routines.activations import *

def profile_relu1d_fp8_ret_fp32():
    N = 16_384
    x = torch.randint(low=-4, high=4, size=(N, ), dtype=torch.float16)

    x_packed = pack_fp8_tensor(x, mode="E5M2").view(torch.uint32).cuda()
    relu1d_fp8_ret_fp32_feather_gpu(x_packed)
    torch.cuda.synchronize()

def profile_relu1d_torch():
    N = 16_384
    x = torch.randint(low=-4, high=4, size=(N, ), dtype=torch.float16)

    torch.nn.functional.relu(x)
    torch.cuda.synchronize()
    torch.cuda.synchronize()


if __name__ == "__main__":
    profile_relu1d_fp8_ret_fp32()
    profile_relu1d_torch()