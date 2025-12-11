import numpy as np
from packers import *
from operations import *

xx = np.random.normal(size=(15000,)).astype(np.float16)
xxx2 = np.ones(shape=(15000, )).astype(np.float16)
xxx = np.ones(shape=(15000,)).astype(np.float16)

xxxp = pack_fp16_ndarray(xxx)
xxx2p = pack_fp16_ndarray(xxx2)

xxdot = dot_fp16_acc_fp32(xxxp, xxx2p)

print(f"xx size = {xx.nbytes}")
print(f"xx dot standard  = {np.dot(xxx, xxx2)}")
print(f"xx dot compressed = {xxdot}")