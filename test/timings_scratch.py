#%%
from nqgl.bmask_bmm.test.time_gpu import TimedFunc
import torch

#%%
transposeF = TimedFunc(
    lambda A: A.transpose(-1, -2), 
    "transpose"
)


A = torch.rand(4096, 4096, device='cuda', dtype=torch.float16)
B = torch.rand(4096, 4096, device='cuda', dtype=torch.float16)
transposeF(A)


matmulTF = TimedFunc( "transpose"(
    lambda A, B: torch.matmul(A, B),
    "torch matmul"
)

matmulTF(A, B)


# %%

from nqgl.bmask_bmm.tri_matmul_grouped import matmul as matmulG1
from nqgl.bmask_bmm.cust_tri_matmuls.tri_matmul_2group import matmul as matmulG2
from nqgl.bmask_bmm.cust_tri_matmuls.tri_matmul_2group_thread_inefficient import matmul as matmulG2TI

TFmatmulG1 = TimedFunc(matmulG1, "custom tri 1grouped")
TFmatmulG2 = TimedFunc(matmulG2, "custom tri 2grouped")
TFmatmulG2TI = TimedFunc(matmulG2TI, "custom tri 2g--TI")

TFmatmulG1(A, B)
TFmatmulG2(A, B)
TFmatmulG2TI(A, B)
# %%
