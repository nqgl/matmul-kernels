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

from nqgl.bmask_bmm.cust_tri_matmuls.configurate import pows2, defconfigspace

batchmm = defconfigspace(
    BLOCK_SIZE_B = 1    ,
    GROUP_SIZE_B1 = pows2(1, 4)   ,
    GROUP_SIZE_B2 = pows2(1, 4)   ,
    BLOCK_SIZE_M = pows2(32, 128)  ,
    BLOCK_SIZE_N = pows2(32, 128)  ,
    BLOCK_SIZE_K = pows2(64, 256)  ,
    GROUP_SIZE_N = pows2(1, 16)    ,
    GROUP_SIZE_M = pows2(1, 64)    ,
    num_stages=range(1,2),
    num_warps=pows2(1, 1)
)

from nqgl.bmask_bmm.cust_tri_matmuls import configurate
@triton.autotune(
    configs=configurate.to_configs(batchmm, 1),
    key=['M', 'N', 'K'],
)
@triton.jit
def batchmatmul_basic_kernel(
        out_ptr, p_ptr, q_ptr, mask_ptr,
        B0 :tl.constexpr,
        B1 :tl.constexpr,
        B2 :tl.constexpr,

        M :tl.constexpr, 
        N :tl.constexpr, 
        K :tl.constexpr,
        stride_pb0, stride_pb1, stride_pb2, stride_pm, stride_pk, 
        stride_qb0, stride_qb1, stride_qb2, stride_qk, stride_qn,
        stride_outb0, stride_outb1, stride_outb2, stride_outm, stride_outn,
        stride_maskb0, stride_maskb1, stride_maskb2,

        BLOCK_SIZE_B :tl.constexpr,
        BLOCK_SIZE_M :tl.constexpr,
        BLOCK_SIZE_N :tl.constexpr,
        BLOCK_SIZE_K :tl.constexpr,
        GROUP_SIZE_M :tl.constexpr,
        GROUP_SIZE_N :tl.constexpr,
        GROUP_SIZE_B1 :tl.constexpr,
        GROUP_SIZE_B2 :tl.constexpr,
        DTYPE_RET :tl.constexpr,
        DTYPE_ACC :tl.constexpr,
        DTYPE_AB :tl.constexpr,
        TRANS :tl.constexpr,
        
    ):
    """
    a: (B0, B1, 1, M, K)
    bt: (B0, 1, B2, N, K)
    """

    n_B0blocks = B0 # tl.cdiv(B0, BLOCK_SIZE_B)
    n_B1blocks = B1 # tl.cdiv(B1, BLOCK_SIZE_B)
    n_B2blocks = B2 # tl.cdiv(B2, BLOCK_SIZE_B)

    n_Mblocks = tl.cdiv(M, BLOCK_SIZE_M)
    n_Nblocks = tl.cdiv(N, BLOCK_SIZE_N)
    n_Kblocks = tl.cdiv(K, BLOCK_SIZE_K)
    pid = tl.program_id(axis=0)

    re = pid

    # GROUP_SIZE_B2 = 8
    # GROUP_SIZE_B1 = 8

    # b2, re = itermod(re, n_B2blocks)
    # b1, re = itermod(re, n_B1blocks)
    i, re = itermod(re, GROUP_SIZE_M)
    j, re = itermod(re, GROUP_SIZE_N) # future can group by changing
    ig, re = itermod(re, tl.cdiv(n_Mblocks, GROUP_SIZE_M))
    jg, re = itermod(re, tl.cdiv(n_Nblocks, GROUP_SIZE_N))
    b2, re = itermod(re, GROUP_SIZE_B2)
    b1, re = itermod(re, GROUP_SIZE_B1)
    b2g, re = itermod(re, tl.cdiv(n_B2blocks, GROUP_SIZE_B2))
    b1g, re = itermod(re, tl.cdiv(n_B1blocks, GROUP_SIZE_B1))
    b0, re = itermod(re, n_B0blocks)

    i = i + GROUP_SIZE_M * ig
    j = j + GROUP_SIZE_N * jg
    b1 = b1 + GROUP_SIZE_B1 * b1g
    b2 = b2 + GROUP_SIZE_B2 * b2g

    assert re == 0

    mus = tl.arange(0, BLOCK_SIZE_M) + i * BLOCK_SIZE_M
    nus = tl.arange(0, BLOCK_SIZE_N) + j * BLOCK_SIZE_N

    
    kus = tl.arange(0, BLOCK_SIZE_K)
    bu0 = b0 * BLOCK_SIZE_B
    bu1 = b1 * BLOCK_SIZE_B
    bu2 = b2 * BLOCK_SIZE_B
    
    # for bi in range(BLOCK_SIZE_B):
    if (bu0 < B0) and (bu1 < B1 and bu2 < B2):
        mask_value = tl.load(
            mask_ptr 
            + stride_maskb0 * bu0
            + stride_maskb1 * bu1
            + stride_maskb2 * bu2
        )
        if mask_value != 0:
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=DTYPE_ACC)
            for ki in range(n_Kblocks):
                p_ptrs = (
                    p_ptr 
                    + stride_pb0 * bu0
                    + stride_pb1 * bu1
                    # + stride_pb2 * bu2
                    + stride_pm * mus[:, None] 
                    + stride_pk * kus[None, :]
                )
                q_ptrs = (
                    q_ptr 
                    + stride_qb0 * bu0
                    # + stride_qb1 * bu1
                    + stride_qb2 * bu2
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
                acc += tl.dot(pi, qi).to(DTYPE_ACC) * mask_value
                kus += BLOCK_SIZE_K
            out_ptrs = (
                out_ptr 
                + stride_outb0 * bu0
                + stride_outb1 * bu1
                + stride_outb2 * bu2
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
            bu0 += 1
        # else:
        #     print("skipped")


def masked_matmul(p, q, mask, outtype=torch.float16, DTYPE_AB = torch.float16):
    """
    a   (*b', M, K)
    b: (*b', K, N)
    """
    B0p, B1p, B2p, M, K = p.shape
    B0q, B1q, B2q, K, N = q.shape
    B0mask, B1mask, B2mask = mask.shape
    didflip=False
    assert B0p == B0q == B0mask
    assert B2p == 1
    assert B1q == 1
    # if b1b == 1:
    #     a,b = b,a
    #     M,N = N,M
    #     didflip = True
    #     # TODO need to deal with the pre-rearranged cases
    #     # and flip strides
    B0 = B0p
    B1 = B1p
    B2 = B2q
    assert B1mask == B1
    assert B2mask == B2



    output = torch.zeros(B0, B1, B2, M, N, device='cuda', dtype=outtype)

    if DTYPE_AB is not None:
        p = p.to(DTYPE_AB)
        q = q.to(DTYPE_AB)

    grid = lambda META: (
          triton.cdiv(M,  META['GROUP_SIZE_M'] * META['BLOCK_SIZE_M'])  * META['GROUP_SIZE_M']
        * triton.cdiv(N,  META['GROUP_SIZE_N'] * META['BLOCK_SIZE_N'])  * META['GROUP_SIZE_N']
        * triton.cdiv(B1, META['BLOCK_SIZE_B'] * META['GROUP_SIZE_B1']) * META['GROUP_SIZE_B1']
        * triton.cdiv(B2, META['BLOCK_SIZE_B'] * META['GROUP_SIZE_B2']) * META['GROUP_SIZE_B2'] 
        * triton.cdiv(B0, META['BLOCK_SIZE_B'])
    ,)
    bt = q.transpose(0, 1)
    tltype = lambda T: tl.float16 if T == torch.float16 else tl.float32

    batchmatmul_basic_kernel[grid](
        output, p, q, mask,
        B0, B1, B2, M, N, K,
        p.stride(0), p.stride(1), p.stride(2), p.stride(3), p.stride(4),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
        mask.stride(0), mask.stride(1), mask.stride(2),
        # BLOCK_SIZE_M=128, 
        # BLOCK_SIZE_N=64,
        # BLOCK_SIZE_K=128
        DTYPE_ACC = tltype(torch.float32),
        DTYPE_RET = tltype(outtype),
        DTYPE_AB = None if DTYPE_AB is torch.float16 else tltype(p.dtype),
        TRANS = True
    )

    return output.to(outtype)


def main():
    import test_mm_speeds
    i = 0

    import random
    b0 = random.randint(1,16 // 2)
    b1 = random.randint(4,64)
    b2 = random.randint(4,64)
    k = random.randint(1, 128)
    m = random.randint(1, 128)
    n = random.randint(1, 128)


    # b = 4
    # k = 128
    # m = 51
    # n = 64
    # b, m, n, k = 50, 193, 1018, 215
    # b0, b1, b2, k, m, n = 4, 5, 6, 171, 311, 180
    b0, b1, b2, k, m, n = 8, 8, 8, 128, 128, 128
    # print(b0, b1, b2, m,n,k)


    # p = torch.rand(b0, b1, 1, m, k, device='cuda', dtype=torch.float16)
    # q = torch.randn(b0, 1, b2, k, n, device='cuda', dtype=torch.float16)
    # mask = torch.randint(0, 2, (b0, b1, b2), device='cuda', dtype=torch.float16)
    # out = masked_matmul(p, q, mask, outtype=torch.float16)
    # comp = (p.clone() @ q.clone()) * mask.clone().unsqueeze(-1).unsqueeze(-1)
    # print(torch.allclose(out, comp, rtol=1e-4, atol=1e-1))
    # print((out - comp).abs().sum())
    # print(torch.count_nonzero(out - comp))
    sparsity = 0.99
    # sparsity = 0.05

    print(
        sum(
            test_mm_speeds.spbm_timetest(
                m, 
                k, 
                n, 
                masked_matmul, 
                "cust spbm_mm ungrouped", 
                bp=[b0, b1, 1], 
                bq=[b0, 1, b2], 
                dtype=torch.float16, 
                iterations=100,
                return_times=True,
                sparsity=sparsity
            )
        [10:])/90
    )
    print(
        sum(
            test_mm_speeds.spbm_timetest(
                m, 
                k, 
                n, 
                lambda p,q,m: torch.matmul(p,q) * m.unsqueeze(-1).unsqueeze(-1), 
                "torch.mm * mask", 
                bp=[b0, b1, 1], 
                bq=[b0, 1, b2], 
                dtype=torch.float16,
                iterations=100,
                return_times=True,
                sparsity=sparsity
            )[10:]
        ) / 90
    )

    print(batchmatmul_basic_kernel.best_config)
    exit()
    # a = torch.rand(b, n, k, device='cuda', dtype=torch.float16)
    # b = torch.randn(b, k, m, device='cuda', dtype=torch.float16)
    # print(a.shape)
    # print(b.shape)
    # c = matmul(a, b)
    # print(c)
    # cc = a.clone() @ b.clone()
    # print(c - cc)
    # print(torch.sum(c-cc))
    # print("done")
    # exit()


    # test_mm_speeds.timetest(n, m, k, lambda *a, **k : matmul(*a, outtype=torch.float32, **k), "matmul fp32", dtype=torch.float32)
    # print()
    # test_mm_speeds.timetest(n, m, k, matmul, "matmul fp16", dtype=torch.float16)

    # test_matmul.runtests(matmul, n, m, k)
    # # input()
    # # a[7, 37] = -77
    # c = matmul(a.clone(), b.clone())
    # print(torch.allclose(c, torch.matmul(a,b).to(c.dtype), rtol=1e-4, atol=1e-2))
    # # while not torch.allclose(c, torch.matmul(a, b).to(c.dtype), rtol=1e-4, atol=1e-2):
    # #     a = torch.rand(n, m, device='cuda')
    # #     b = torch.randn(m, k, device='cuda')
    # #     c = matmul(a, b)
    # #     i += 1
    # #     print(i)
    


    # k = 4096
    # m = 4096
    # n = 4096


    # # k = 32 * 128 - 1
    # # m = 32 * 128 - 111
    # # n = 32 * 128// 2 + 1

    # a = torch.rand(n, m, device='cuda')
    # b = torch.randn(m, k, device='cuda')

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    # start.record()
    # c = a @ b
    # end.record()
    # torch.cuda.synchronize()
    # print("torch:", start.elapsed_time(end))



    # # a = torch.rand(n, m, device='cuda')
    # # b = torch.randn(m, k, device='cuda')

    # # start = torch.cuda.Event(enable_timing=True)
    # # end = torch.cuda.Event(enable_timing=True)

    
    # # start.record()
    # # c = matmul(a, b)
    # # end.record()
    # # torch.cuda.synchronize()
    # # print("custom:", start.elapsed_time(end))



    # a = torch.rand(n, m, device='cuda')
    # b = torch.randn(m, k, device='cuda')

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    
    # start.record()
    # c = matmul(a, b)
    # end.record()
    # torch.cuda.synchronize()
    # print("custom:", start.elapsed_time(end))


    # t = 0
    # for i in range(100):
    #     a = torch.rand(n, m, device='cuda')
    #     b = torch.randn(m, k, device='cuda')
    #     start = torch.cuda.Event(enable_timing=True)
    #     end = torch.cuda.Event(enable_timing=True)
    #     start.record()
    #     c = matmul(a, b)
    #     end.record()
    #     torch.cuda.synchronize()
    #     t0 = start.elapsed_time(end)
    #     print("custom:", t0)
    #     t += t0
    # print("time for 100 iters", t)

if __name__ == "__main__":
    main()