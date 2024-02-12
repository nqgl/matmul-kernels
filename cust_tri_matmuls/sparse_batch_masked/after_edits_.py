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
    GROUP_SIZE_B = pows2(1, 4)   ,
    # GROUP_SIZE_B2 = pows2(1, 4)   ,
    BLOCK_SIZE_M = pows2(16, 128)  ,
    BLOCK_SIZE_N = pows2(16, 128)  ,
    BLOCK_SIZE_K = pows2(16, 128)  ,
    GROUP_SIZE_N = pows2(1, 32)    ,
    GROUP_SIZE_M = pows2(1, 16)    ,
    num_stages=range(2,6),
    num_warps=pows2(2, 8)
)

good1 = triton.Config(
    {
        'BLOCK_SIZE_B': 1,
        'GROUP_SIZE_B': 2,
        'BLOCK_SIZE_M': 32,
        'BLOCK_SIZE_N': 32,
        'BLOCK_SIZE_K': 32,
        'GROUP_SIZE_N': 1,
        'GROUP_SIZE_M': 2
    },
    num_warps =  4,
    num_stages = 4
)


@triton.jit
def store(
    out_ptr, acc, 
    mus, nus, 
    M, N, 
    bu0, bu1, bu2,
    stride_outb0, stride_outb1, stride_outb2, stride_outm, stride_outn,
    DTYPE_RET=None
):
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


from nqgl.bmask_bmm.cust_tri_matmuls import configurate
@triton.autotune(
    configs=[good1] + configurate.to_configs(batchmm, 16),
    key=['M', 'N', 'K'],
)
@triton.jit
def batchmatmul_basic_kernel(
        out_ptr, p_ptr, q_ptr, mask_ptr, bitters,
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
        stride_bitters,

        BLOCK_SIZE_B :tl.constexpr,
        BLOCK_SIZE_M :tl.constexpr,
        BLOCK_SIZE_N :tl.constexpr,
        BLOCK_SIZE_K :tl.constexpr,
        GROUP_SIZE_M :tl.constexpr,
        GROUP_SIZE_N :tl.constexpr,
        GROUP_SIZE_B :tl.constexpr,
        # GROUP_SIZE_B2 :tl.constexpr,
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
    # b2, re = itermod(re, GROUP_SIZE_B2)
    # b1, re = itermod(re, GROUP_SIZE_B1)
    # b2g, re = itermod(re, tl.cdiv(n_B2blocks, GROUP_SIZE_B2))
    # b1g, re = itermod(re, tl.cdiv(n_B1blocks, GROUP_SIZE_B1))
    mbi, re = itermod(re, n_B0blocks * n_B1blocks * n_B2blocks)
    bre = tl.load(bitters + stride_bitters * mbi)
    b2, bre = itermod(bre, B2)
    b1, bre = itermod(bre, B1)
    b0, bre = itermod(bre, B0)
    # assert bre == 0
    # print(bis[range(1), range(1)])


    i = i + GROUP_SIZE_M * ig
    j = j + GROUP_SIZE_N * jg

    # assert re == 0

    mus = tl.arange(0, BLOCK_SIZE_M) + i * BLOCK_SIZE_M
    nus = tl.arange(0, BLOCK_SIZE_N) + j * BLOCK_SIZE_N

    kus = tl.arange(0, BLOCK_SIZE_K)
    bu0 = b0
    bu1 = b1
    bu2 = b2    
    
    # for bi in range(BLOCK_SIZE_B):
    if (bu0 < B0) and (bu1 < B1 and bu2 < B2):

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

            # if DTYPE_AB is not None:
            #     pi = pi.to(DTYPE_AB)
            #     qi = qi.to(DTYPE_AB)
            acc += tl.dot(pi, qi)
            kus += BLOCK_SIZE_K
        # out_ptrs = (
        #     out_ptr 
        #     + stride_outb0 * bu0
        #     + stride_outb1 * bu1
        #     + stride_outb2 * bu2
        #     + stride_outm * mus[:, None]
        #     + stride_outn * nus[None, :]
        # )

        store(
            out_ptr, acc, 
            mus, nus, 
            M, N, 
            bu0, bu1, bu2,
            stride_outb0, stride_outb1, stride_outb2, stride_outm, stride_outn,
            tl.float32
        )
        # if DTYPE_RET is not None:
        #     tl.store(
        #         out_ptrs, 
        #         acc.to(DTYPE_RET), 
        #         mask=(
        #             (mus[:, None] < M)
        #             & (nus[None, :] < N)
        #         )
        #     )
        # else:
        #     tl.store(
        #         out_ptrs, 
        #         acc, 
        #         mask=(
        #             (mus[:, None] < M)
        #             & (nus[None, :] < N)
        #         )
        #     )
        # bu0 += 1
    # else:
    #     print("skipped")


def masked_matmul(p, q, mask :torch.Tensor, outtype=torch.float32, DTYPE_AB = torch.float16):
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
    mask_indices = mask.to_sparse().indices() # shape (3, nnz)
    mask_values = mask.to_sparse().values()
    bitters = (
        mask_indices[2] 
        + mask_indices[1] * B2 
        + mask_indices[0] * B2 * B1
    )
    # bitters = bitters[torch.randperm(bitters.shape[0])]
    # print("nnz", bitters.shape[0], mask.sum())
    output = torch.zeros(B0, B1, B2, M, N, device='cuda', dtype=outtype)

    if DTYPE_AB is not None:
        p = p.to(DTYPE_AB)
        q = q.to(DTYPE_AB)

    grid = lambda META: (
          triton.cdiv(M,  META['GROUP_SIZE_M'] * META['BLOCK_SIZE_M'])  * META['GROUP_SIZE_M']
        * triton.cdiv(N,  META['GROUP_SIZE_N'] * META['BLOCK_SIZE_N'])  * META['GROUP_SIZE_N']
        * bitters.shape[0]
        # * triton.cdiv(B1, META['BLOCK_SIZE_B'] * META['GROUP_SIZE_B1']) * META['GROUP_SIZE_B1']
        # * triton.cdiv(B2, META['BLOCK_SIZE_B'] * META['GROUP_SIZE_B2']) * META['GROUP_SIZE_B2'] 
        # * triton.cdiv(B0, META['BLOCK_SIZE_B'])
    ,)
    bt = q.transpose(0, 1)
    tltype = lambda T: tl.float16 if T == torch.float16 else tl.float32
    printmatcall = True
    # if printmatcall:
    #     print("output", output)
    #     print("p", p)
    #     print("q", q)
    #     print("mask", mask)
    #     print("bitters", bitters)
    #     print("B0", B0)
    #     print("B1", B1)
    #     print("B2", B2)
    #     print("M", M)
    #     print("N", N)
    #     print("K", K)
    #     print("p.stride(0)", p.stride(0))
    #     print("p.stride(1)", p.stride(1))
    #     print("p.stride(2)", p.stride(2))
    #     print("p.stride(3)", p.stride(3))
    #     print("p.stride(4)", p.stride(4))
    #     print("q.stride(0)", q.stride(0))
    #     print("q.stride(1)", q.stride(1))
    #     print("q.stride(2)", q.stride(2))
    #     print("q.stride(3)", q.stride(3))
    #     print("q.stride(4)", q.stride(4))
    #     print("output.stride(0)", output.stride(0))
    #     print("output.stride(1)", output.stride(1))
    #     print("output.stride(2)", output.stride(2))
    #     print("output.stride(3)", output.stride(3))
    #     print("output.stride(4)", output.stride(4))
    #     print("mask.stride(0)", mask.stride(0))
    #     print("mask.stride(1)", mask.stride(1))
    #     print("mask.stride(2)", mask.stride(2))
    #     print("bitters.stride(0)", bitters.stride(0))
    #     print("DTYPE_ACC", tltype(torch.float32),)
    #     print("DTYPE_RET", tltype(outtype),)
    #     print("DTYPE_AB", None if DTYPE_AB is torch.float16 else tltype(p.dtype),)
    #     print("TRANS", True)

    

    batchmatmul_basic_kernel[grid](
        output, p, q, mask, bitters,
        B0, B1, B2, M, N, K,
        p.stride(0), p.stride(1), p.stride(2), p.stride(3), p.stride(4),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
        mask.stride(0), mask.stride(1), mask.stride(2),
        bitters.stride(0),
        DTYPE_ACC = tltype(torch.float32),
        DTYPE_RET = tltype(outtype),
        DTYPE_AB = tl.float16 if DTYPE_AB is torch.float16 else tltype(p.dtype),
        TRANS = True
    )

    return output


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
    b0, b1, b2, m, k, n = 8, 16, 16, 16, 1024, 256
    # print(b0, b1, b2, m,n,k)
    sparsity = 0.98
    # sparsity = 0.05
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