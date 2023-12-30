from numba import cuda
import torch
from torch.autograd import Function
from nqgl.sae.hsae.spbmbmm import baseline
@cuda.jit
def batchmasked_matmul_kernel(output, a, bt, mask, maskmods, a1d, bt1d):
    """
    a   (*B', N, M)
    bt: (*B', K, M)
    """

        mmb = tid % maskmods[1:]
        mbids = mmb // maskmods[:-1]
        N_id = mbids[-2]
        K_id = mbids[-1]
        aid = mbids[:-2] * a1d
        btid = mbids[:-2] * bt1d



class BatchMaskedBMM(Function):
    @staticmethod
    def forward(ctx, a, bt, mask):
        """
        a   (*B', N, M)
        bt: (*B', K, M)
        mask: (*B', N, K)
        """
        threadsperblock = 32
        blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock



def main():
    ex = baseline.generate_example(8, 10, 12, 16, xlist=False)

    x = ex.x
    M = ex.W_enc
    gate = ex.gate
