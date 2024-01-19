from numba import cuda
import torch
from torch.autograd import Function
from nqgl.sae.hsae.spbmbmm import baseline
import triton
import triton.language as tl
import test_matmul

@triton.jit
def modulate(i, mod, n, mod2):
    if i > n - n % mod2:
        return i
    else:
        return i // mod + i % mod * mod + (mod2 - mod) * (i // mod2)
@triton.jit
def modulate(i, mod, n, a):
    return i // mod * mod + i % mod

@triton.jit
def itermod(it, mod):
    return it % mod, it // mod

@triton.jit
def itermodgrouped(it, mod, groupsize, pid_per_group):
    group = it % pid_per_group

    return it % mod, it // mod


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B' : 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_N' : 8, 'GROUP_SIZE_M' : 8}, num_stages=2,num_warps=4),
        triton.Config({'BLOCK_SIZE_B' : 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_N' : 8, 'GROUP_SIZE_M' : 8}, num_stages=1,num_warps=4),
        triton.Config({'BLOCK_SIZE_B' : 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_N' : 8, 'GROUP_SIZE_M' : 8}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_B' : 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_N' : 8, 'GROUP_SIZE_M' : 8}, num_stages=2,num_warps=4),
        triton.Config({'BLOCK_SIZE_B' : 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_N' : 8, 'GROUP_SIZE_M' : 8}, num_stages=2,num_warps=4),
        triton.Config({'BLOCK_SIZE_B' : 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_N' : 8, 'GROUP_SIZE_M' : 8}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_B' : 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_N' : 8, 'GROUP_SIZE_M' : 8}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_B' : 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_N' : 8, 'GROUP_SIZE_M' : 8}, num_stages=2,num_warps=4),
        triton.Config({'BLOCK_SIZE_B' : 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_N' : 8, 'GROUP_SIZE_M' : 8}, num_stages=2,num_warps=4),
        triton.Config({'BLOCK_SIZE_B' : 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_N' : 8, 'GROUP_SIZE_M' : 8}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_B' : 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_N' : 8, 'GROUP_SIZE_M' : 8}, num_stages=5,num_warps=4),
        triton.Config({'BLOCK_SIZE_B' : 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_N' : 8, 'GROUP_SIZE_M' : 8}, num_stages=3,num_warps=4),

        # triton.Config({'BLOCK_SIZE_B' : 4, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M' : 8}, num_stages=2,num_warps=4),
        # triton.Config({'BLOCK_SIZE_B' : 4, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M' : 8}, num_stages=1,num_warps=4),
        # triton.Config({'BLOCK_SIZE_B' : 4, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M' : 8}, num_stages=4,num_warps=4),
        # triton.Config({'BLOCK_SIZE_B' : 4, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M' : 8}, num_stages=2,num_warps=4),
        # triton.Config({'BLOCK_SIZE_B' : 4, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M' : 8}, num_stages=2,num_warps=4),
        # triton.Config({'BLOCK_SIZE_B' : 4, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M' : 8}, num_stages=4,num_warps=4),
        # triton.Config({'BLOCK_SIZE_B' : 4, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M' : 8}, num_stages=4,num_warps=4),
        # triton.Config({'BLOCK_SIZE_B' : 4, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M' : 8}, num_stages=2,num_warps=4),
        # triton.Config({'BLOCK_SIZE_B' : 4, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M' : 8}, num_stages=2,num_warps=4),
        # triton.Config({'BLOCK_SIZE_B' : 4, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M' : 8}, num_stages=4,num_warps=4),
        # triton.Config({'BLOCK_SIZE_B' : 4, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M' : 8}, num_stages=5,num_warps=4),
        # triton.Config({'BLOCK_SIZE_B' : 4, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M' : 8}, num_stages=3,num_warps=4),


    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def batchmatmul_basic_kernel(
        out_ptr, p_ptr, q_ptr,
        B :tl.constexpr,
        M :tl.constexpr, 
        N :tl.constexpr, 
        K :tl.constexpr,
        stride_pb, stride_pm, stride_pk, 
        stride_qb, stride_qk, stride_qn,
        stride_outb, stride_outm, stride_outn,
        BLOCK_SIZE_B :tl.constexpr,
        BLOCK_SIZE_M :tl.constexpr,
        BLOCK_SIZE_N :tl.constexpr,
        BLOCK_SIZE_K :tl.constexpr,
        GROUP_SIZE_M :tl.constexpr,
        GROUP_SIZE_N :tl.constexpr,
        DTYPE_RET :tl.constexpr,
        DTYPE_ACC :tl.constexpr,
        DTYPE_AB :tl.constexpr,
        TRANS :tl.constexpr,
        
    ):
    """
    a: (b', M, K) b' is b or 1
    bt: (b, N, K)
    """

    n_Bblocks = tl.cdiv(B, BLOCK_SIZE_B)
    n_Mblocks = tl.cdiv(M, BLOCK_SIZE_M)
    n_Nblocks = tl.cdiv(N, BLOCK_SIZE_N)
    n_Kblocks = tl.cdiv(K, BLOCK_SIZE_K)
    pid = tl.program_id(axis=0)
    pid_per_group = n_Nblocks * GROUP_SIZE_M * n_Bblocks
    re = pid
    # re = pid % pid_per_group
    # ig = pid // pid_per_group
    j, re = itermod(re, GROUP_SIZE_N) # future can group by changing
    i, re = itermod(re, GROUP_SIZE_M)
    jg, re = itermod(re, tl.cdiv(n_Nblocks, GROUP_SIZE_N))
    ig, re = itermod(re, tl.cdiv(n_Mblocks, GROUP_SIZE_M))
    b, re = itermod(re, n_Bblocks)

    i = i + ig * GROUP_SIZE_M
    j = j + jg * GROUP_SIZE_N
    assert re == 0

    mus = tl.arange(0, BLOCK_SIZE_M) + i * BLOCK_SIZE_M
    nus = tl.arange(0, BLOCK_SIZE_N) + j * BLOCK_SIZE_N

    
    kus = tl.arange(0, BLOCK_SIZE_K)
    bu = b * BLOCK_SIZE_B
    for bi in range(BLOCK_SIZE_B):
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=DTYPE_ACC)
        for ki in range(n_Kblocks):
            p_ptrs = (
                p_ptr 
                + stride_pb * bu
                + stride_pm * mus[:, None] 
                + stride_pk * kus[None, :]
            )
            q_ptrs = (
                q_ptr 
                + stride_qb * bu
                + stride_qn * nus[None, :]
                + stride_qk * kus[:, None] 
            )
            pi = tl.load(
                p_ptrs, 
                mask=(  
                    (mus[:, None] < M) 
                    & (kus[None, :] < K)
                    ), 
                other=0.0
            )
            qi = tl.load(
                q_ptrs, 
                mask=(  
                    (nus[None, :] < N)
                    & (kus[:, None] < K) 
                    ), 
                other=0.0
            )
            # bi = tl.trans(bi)

            if DTYPE_AB is not None:
                pi = pi.to(DTYPE_AB)
                qi = qi.to(DTYPE_AB)
            acc += tl.dot(pi, qi).to(DTYPE_ACC)
            kus += BLOCK_SIZE_K
        out_ptrs = (
            out_ptr 
            + stride_outb * bu
            + stride_outm * mus[:, None]
            + stride_outn * nus[None, :]
        )
        if DTYPE_RET is not None:
            tl.store(
                out_ptrs, 
                acc.to(DTYPE_RET), 
                mask=(
                    (mus[:, None] < M)
                    & (nus[None, :] < N)
                )
            )
        else:
            tl.store(
                out_ptrs, 
                acc, 
                mask=(
                    (mus[:, None] < M)
                    & (nus[None, :] < N)
                )
            )
        bu += 1



def matmul(a, b, outtype=torch.float16, DTYPE_AB = torch.float16):
    """
    a   (*b', M, K)
    b: (*b', K, N)
    """
    b1a, M, K = a.shape
    b1b, K, N = b.shape
    didflip=False
    assert b1a == b1b 
    # if b1b == 1:
    #     a,b = b,a
    #     M,N = N,M
    #     didflip = True
    #     # TODO need to deal with the pre-rearranged cases
    #     # and flip strides
    B = b1a

    output = torch.zeros(B, M, N, device='cuda', dtype=outtype)

    if DTYPE_AB is not None:
        a = a.to(DTYPE_AB)
        b = b.to(DTYPE_AB)

    grid = lambda META: (
        triton.cdiv(M, META['GROUP_SIZE_M'] * META['BLOCK_SIZE_M']) * META['GROUP_SIZE_M']
        * triton.cdiv(N, META['BLOCK_SIZE_N'])
        * triton.cdiv(B, META['BLOCK_SIZE_B']), 
    )
    bt = b.transpose(0, 1)
    tltype = lambda T: tl.float16 if T == torch.float16 else tl.float32

    batchmatmul_basic_kernel[grid](
        output, a, bt,
        B, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        # BLOCK_SIZE_M=128, 
        # BLOCK_SIZE_N=64,
        # BLOCK_SIZE_K=128
        DTYPE_ACC = tltype(torch.float32),
        DTYPE_RET = tltype(outtype),
        DTYPE_AB = None if DTYPE_AB is torch.float16 else tltype(a.dtype),
        TRANS = True
    )

    return output.to(outtype)


def main():

    import test_mm_speeds
    i = 0

    import random
    b = random.randint(1,16)
    k = random.randint(1, 1024)
    m = random.randint(1, 1024)
    n = random.randint(1, 1024)

    # b = 4
    # k = 128
    # m = 51
    # n = 64
    b, m, n, k = 50, 193, 1018, 215
    print(b,m,n,k)
    test_mm_speeds.timetest(m, k, n, matmul, "cust bmm ungrouped", b=[b], dtype=torch.float16, iterations=10)
    test_mm_speeds.timetest(m, k, n, torch.bmm, "torch.bmm", b=[b], dtype=torch.float16)
    exit()
    a = torch.rand(b, n, k, device='cuda', dtype=torch.float16)
    b = torch.randn(b, k, m, device='cuda', dtype=torch.float16)
    print(a.shape)
    print(b.shape)
    c = matmul(a, b)
    print(c)
    cc = a.clone() @ b.clone()
    print(c - cc)
    print(torch.sum(c-cc))
    print("done")
    exit()


    test_mm_speeds.timetest(n, m, k, lambda *a, **k : matmul(*a, outtype=torch.float32, **k), "matmul fp32", dtype=torch.float32)
    print()
    test_mm_speeds.timetest(n, m, k, matmul, "matmul fp16", dtype=torch.float16)

    test_matmul.runtests(matmul, n, m, k)
    # input()
    # a[7, 37] = -77
    c = matmul(a.clone(), b.clone())
    print(torch.allclose(c, torch.matmul(a,b).to(c.dtype), rtol=1e-4, atol=1e-2))
    # while not torch.allclose(c, torch.matmul(a, b).to(c.dtype), rtol=1e-4, atol=1e-2):
    #     a = torch.rand(n, m, device='cuda')
    #     b = torch.randn(m, k, device='cuda')
    #     c = matmul(a, b)
    #     i += 1
    #     print(i)
    


    k = 4096
    m = 4096
    n = 4096


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



    # a = torch.rand(n, m, device='cuda')
    # b = torch.randn(m, k, device='cuda')

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    
    # start.record()
    # c = matmul(a, b)
    # end.record()
    # torch.cuda.synchronize()
    # print("custom:", start.elapsed_time(end))



    a = torch.rand(n, m, device='cuda')
    b = torch.randn(m, k, device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    
    start.record()
    c = matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    print("custom:", start.elapsed_time(end))


    t = 0
    for i in range(100):
        a = torch.rand(n, m, device='cuda')
        b = torch.randn(m, k, device='cuda')
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        c = matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        t0 = start.elapsed_time(end)
        print("custom:", t0)
        t += t0
    print("time for 100 iters", t)

if __name__ == "__main__":
    main()