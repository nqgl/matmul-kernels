from numba import cuda
import torch
from torch.autograd import Function
from nqgl.sae.hsae.spbmbmm import baseline
import test_matmul
from custom_matmul_basic import matmul as matmul_basic


@cuda.jit
def matmul_boxed_kernel(output, a, bt):
    """
    # a   (*B', N, M)
    # b (*B', N, M)
    a: (N, M)
    bt: (K, M)
    """
    N = a.shape[0]
    M = a.shape[1]
    K = bt.shape[0]

    # Format:
    # xzid means z index within x
    # w = warp
    # b = block
    # t = thread

    wt_s = 32 # warp.thread size
    bw_s = cuda.blockDim.x // wt_s # block.warp size
    b_s = cuda.gridDim.x # block size

    wtid = cuda.threadIdx.x % wt_s # warp.thread id
    bwid = cuda.threadIdx.x // wt_s # block.warp id
    btid = cuda.threadIdx.x     # block.thread id
    
    bid = cuda.blockIdx.x           # block id
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
                                    # absolute thread id
    awid = bid * bw_s + bwid        # absolute warp id

    # btid = cuda.blockIdx.x
    # wtid = cuda.threadIdx.x
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    boxdn = 6
    boxdm = 1024
    boxdk = 6
    box_a = cuda.shared.array(shape=(boxdn, boxdm), dtype="float32")
    box_bt = cuda.shared.array(shape=(boxdk, boxdm), dtype="float32")
    Nb = (N + boxdn - 1) // boxdn
    Mb = (M + boxdm - 1) // boxdm
    Kb = (K + boxdk - 1) // boxdk



    # bwid = bid * bw_s + bwid


    i_nk = bid

    loop_btid = btid

    while i_nk < Nb * Mb * Kb:

        # box coordinates
        nb = i_nk % Nb
        kb = (i_nk // Nb) % Kb
        mb = (i_nk // (Nb * Kb)) # future think about n,k,m order to make more cache hit and also 
                                # manually pred and use the hit
        # box starting index
        n0 = nb * boxdn
        k0 = kb * boxdk
        m0 = mb * boxdm

        # fill the boxes
        i_b_nmk = btid
        # cuda.syncthreads()
        while i_b_nmk < max(boxdn * boxdm, boxdk * boxdm):
            mi = i_b_nmk % boxdm
            ni = i_b_nmk // boxdm
            ki = i_b_nmk // boxdm

            if mi < boxdm and mi + m0 < M:
                if ni < boxdn and ni + n0 < N:
                    box_a[ni, mi] = a[n0 + ni, m0 + mi]
                if ki < boxdk and ki + k0 < K:
                    box_bt[ki, mi] = bt[k0 + ki, m0 + mi]
            i_b_nmk += cuda.blockDim.x
        cuda.syncthreads()
        # m = i % wt_s
        # n = (i // wt_s) % N
        # k = i // (wt_s * N)
        i_b_nk = bwid # each warp handles an n, k
                        # thread inside warp handles a m
        while i_b_nk < boxdn * boxdk:
            ni = i_b_nk % boxdn
            ki = i_b_nk // boxdn
            if  ni + n0 < N and ki + k0 < K: #dont need ni < boxdn and ki < boxdk
                acc = 0
                mi = wtid
                while mi < min(boxdm, M - m0):
                    # PICK UP HERE
                    #
                    #
                    #
                    acc += box_a[ni, mi] * box_bt[ki, mi]
                    mi += wt_s
                acc = sum_warp(acc, 1)
                if wtid == 0:
                    cuda.atomic.add(output, (n0 + ni, k0 + ki), acc)
            i_b_nk += bw_s # add num warps in block to hit all n, k
        i_nk += b_s
        cuda.syncthreads()





        # acc = 0
        # while m < M:
        #     acc += a[n, m] * bt[k, m]
        #     m += wt_s
        # acc = sum_warp(acc, 1)
        # if wtid == 0:
        #     cuda.atomic.add(output, (n, k), acc)
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

    matmul_boxed_kernel[blockspergrid, threadsperblock](output, a, bt)
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
    # n = 1024
    # m = 1024
    # k = 1024
    # k = 32 * 5
    # m = 128
    # n = 32 * 8

    k = 169
    m = 207
    n = 153


    test_matmul.runtests(matmul, n, m, k)

    # exit()
    print(matmul(torch.eye(33, device='cuda'), torch.eye(33, device='cuda')))


    a = torch.ones(n, m, device='cuda')
    b = torch.ones(m, k, device='cuda')
    print(matmul(a, b))
    print(a @ b)
    # exit()
    # input()




    i = 0

    k = 32 * 5
    m = 32 * 7
    n = 32 * 8

    a = torch.rand(n, m, device='cuda')
    b = torch.randn(m, k, device='cuda')
    c = matmul(a, b)
    print(torch.allclose(c, a @ b, rtol=1e-3, atol=1e-3))
    while not torch.allclose(c, a @ b, rtol=1e-3, atol=1e-3):
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
    print("boxed:", start.elapsed_time(end))



    a = torch.rand(n, m, device='cuda')
    b = torch.randn(m, k, device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = matmul_basic(a, b)
    end.record()
    torch.cuda.synchronize()
    print("basic:", start.elapsed_time(end))

if __name__ == "__main__":
    main()