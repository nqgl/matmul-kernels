from numba import cuda
import torch
from torch.autograd import Function
from nqgl.sae.hsae.spbmbmm import baseline




@cuda.jit
def matmul_basic_kernel(output, a, bt):
    """
    # a   (*B', N, M)
    # b (*B', N, M)
    a: (N, M)
    bt: (K, M)
    """
    N = a.shape[0]
    M = a.shape[1]
    K = bt.shape[0]

    tw_s = 32
    w_s = cuda.blockDim.x // tw_s
    b_s = cuda.gridDim.x

    twid = cuda.threadIdx.x % tw_s
    wid = cuda.threadIdx.x // tw_s
    bid = cuda.blockIdx.x
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    awid = tid // tw_s

    # btid = cuda.blockIdx.x
    # wtid = cuda.threadIdx.x
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    bwid = bid * w_s + wid
    while awid < a.shape[0] * bt.shape[0]:
        n = awid % N
        k = (awid // N) % K
        m = twid
        # m = i % tw_s
        # n = (i // tw_s) % N
        # k = i // (tw_s * N)
        acc = 0
        while m < M:
            acc += a[n, m] * bt[k, m]
            m += tw_s
        acc = sum_warp(acc, 1)
        if twid == 0:
            cuda.atomic.add(output, (n, k), acc)
        awid += b_s * w_s
        # i += cuda.blockDim.x * cuda.gridDim.x
        # i += cuda.blockDim.x * cuda.gridDim.x

def matmul(a, b):
    """
    a   (*B', N, M)
    b: (*B', N, M)
    """
    output = torch.zeros(a.shape[0], b.shape[1], device='cuda')
    bt = b.transpose(0, 1)
    threadsperblock = 32
    threadsperblock = 32 * 8 
    blockspergrid = (min(max(a.shape[0], bt.shape[0]) * a.shape[1], 6000 // 10 ) + (threadsperblock - 1)) // threadsperblock
    blockspergrid = (min(max(a.shape[0], bt.shape[0]) * a.shape[1], threadsperblock * 128) + (threadsperblock - 1)) // threadsperblock

    matmul_basic_kernel[blockspergrid, threadsperblock](output, a, bt)
    return output


@cuda.jit(device=True)
def sum_warp(acc, mult = 1):
    off = 32
    while off > 1:
        off = off // 2
        acc += cuda.shfl_down_sync(0xffffffff, acc, off * mult)
        # acc += acc
        # cuda.syncthreads()
    # off = 0
    # acc += cuda.shfl_down_sync(0xffffffff, acc, off * mult)

    return acc

def main():
    i = 0

    k = 32
    m = 32
    n = 32

    a = torch.rand(n, m, device='cuda')
    b = torch.randn(m, k, device='cuda')
    c = matmul(a, b)
    print(torch.allclose(c, a @ b, rtol=1e-4, atol=1e-4))
    while not torch.allclose(c, a @ b, rtol=1e-4, atol=1e-4):
        a = torch.rand(n, m, device='cuda')
        b = torch.randn(m, k, device='cuda')
        c = matmul(a, b)
        i += 1
        print(i)
    


    k = 6699
    m = 2007
    n = 6503


    # k = 32 * 128 - 1
    # m = 32 * 128 - 111
    # n = 32 * 128// 2 + 1

    a = torch.rand(n, m, device='cuda')
    b = torch.randn(m, k, device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = a @ b
    end.record()
    torch.cuda.synchronize()
    print("torch:", start.elapsed_time(end))

    a = torch.rand(n, m, device='cuda')
    b = torch.randn(m, k, device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    
    start.record()
    c = matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    print("custom:", start.elapsed_time(end))

if __name__ == "__main__":
    main()