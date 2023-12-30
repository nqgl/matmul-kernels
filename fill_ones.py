from numba import cuda
import torch
from torch.autograd import Function
from nqgl.sae.hsae.spbmbmm import baseline


# @cuda.jit 

@cuda.jit
def xor_matrix_kernel(output, a):
    """
    a   (*B', N, M)
    b: (*B', N, M)
    """
    btid = cuda.blockIdx.x
    wtid = cuda.threadIdx.x
    x = cuda.shared.array(shape=(32, 32), dtype="float32")
    ai = btid * 32 + wtid
    while ai < a.shape[0] * a.shape[1]:
        ai0 = ai % a.shape[0]
        ai1 = ai // a.shape[0]
        x[ai % 32, (ai // 32) % 32] = a[ai0, ai1]
        ai += 32 * 32
    cuda.atomic.add(output, (wtid, btid), x[wtid, btid])


    
    # if tid < output.size:
    #     output[tid] = a[tid] ^ b[tid]
@cuda.jit(device=True)
def shuffle_reduce_sum(o, v, tid):
    pass





@cuda.jit
def parr_sum_all_kernel(out, v):
    """
    v: (N)
    """
    tw_s = cuda.blockDim.x
    b_s = cuda.gridDim.x

    twid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    tid = bid * tw_s + twid
    s = cuda.shared.array(shape=(32), dtype="float32")
    i = tid
    acc = 0
    while i < v.shape[0]:
        acc += v[i]
        i += cuda.blockDim.x * cuda.gridDim.x
    # cuda.syncthreads()
    acc = sum_warp(acc, 1)
    cuda.syncthreads()

    if twid == 0:
        s[bid] = acc
        cuda.atomic.add(out, (bid), acc)
    cuda.syncthreads()
    if bid == 0:
        # s[twid] = 1.
        acc = s[twid]
        # cuda.syncthreads()
        acc = sum_warp(acc, 1)

        # if tid == 0:
        # out[0] = acc

    # cuda.atomic.add(out, (3,), cuda.threadIdx.x)
    # while tid < v.shape[0]:
    #     cuda.atomic.add(out, (0,), v[tid])
    #     tid += cuda.blockDim.x * cuda.gridDim.x
        


def parr_sum_all(v):
    out = torch.zeros(32, device='cuda')
    threadsperblock = 32
    blockspergrid = (v.shape[0] + (threadsperblock - 1)) // threadsperblock
    parr_sum_all_kernel[blockspergrid, threadsperblock](out, v)
    return out



# an_array = torch.eye(80, device='cuda')

# threadsperblock = 32
# blockspergrid = (80 * 79 + (threadsperblock - 1)) // threadsperblock
# out = torch.zeros(32, 32, device='cuda')
# print(xor_matrix_kernel[blockspergrid, threadsperblock](out, an_array))
# print(out)
# print(torch.sum(out))




def testvecs(n = 32 * 32):
    d = {}
    d["arange"] = torch.arange(0, 32 * 32,  device="cuda")
    d["arange_desc"] = torch.arange(32 * 32, 0, -1,  device="cuda")

    v11 = torch.ones(32 * 32, device="cuda")
    v10 = torch.ones(32 * 32, device="cuda")
    for i in range(32 * 32):
        if i % 2 == 0:
            v10[i] = 0
        else:
            v11[i] = 0

    v1 = torch.ones(32 * 32, device="cuda")
    d["ones"] = v1
    d["ones_skip%2=0"] = v10
    d["ones_skip%2=1"] = v11
    return d    

def colprint(l, spacing = 4):
    l = [[str(e) for e in ll] for ll in l] # convert to strings

    num_cols = max([len(ll) for ll in l]) # longest row determines number of columns
    
    colwidths = [ # colwidths = longest string in each column + spacing
        max([
            len(l[i][j]) 
            for i in range(num_cols) 
            if j < len(l[i])]) + spacing
        for j in range(num_cols)
    ]

    for line in l: # 4. for each line
        print("".join([ # 3. join the columns, then print them as a line
            line[i].ljust(colwidths[i]) # 1. left justify each string in 
                                            # the column with the width for that column
            for i in range(len(line))]))  # 2. for each column in the line

def runtests():
    d = testvecs()
    l = []
    for k, v in d.items():
        l += [[k, torch.sum(v).item(), parr_sum_all(v),]]
    colprint(l)


def main():
    # a = torch.randn(39, 39, device='cuda')
    # b = torch.randn(39, 39, device='cuda')
    # # a = torch.eye(8, device='cuda')
    # # b = torch.eye(8, device='cuda')
    # c = matmul(a, b)
    # print(c - a @ b)
    # print(c)
    # print(torch.allclose(c, a @ b, atol=1e-5))

    runtests()  
    i = 0
    a = torch.rand(6503, 2007, device='cuda')
    b = torch.randn(2007, 6699, device='cuda')
    c = matmul(a, b)
    print(torch.allclose(c, a @ b, rtol=1e-4, atol=1e-4))
    while not torch.allclose(c, a @ b, rtol=1e-4, atol=1e-4):
        a = torch.rand(6503, 2007, device='cuda')
        b = torch.randn(2007, 6699, device='cuda')
        c = matmul(a, b)
        i += 1
    


    k = 6699
    m = 2007
    n = 6503


    k = 32 * 16 - 1
    m = 32 * 16 - 111
    n = 32 * 16// 2 + 1

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