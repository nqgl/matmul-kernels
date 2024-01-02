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

@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M' : 8}, num_stages=2,num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M' : 8}, num_stages=1,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M' : 2,  'GROUP_SIZE_N' : 4}, num_stages=4,num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M' : 2,  'GROUP_SIZE_N' : 4}, num_stages=2,num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M' : 2,  'GROUP_SIZE_N' : 4}, num_stages=2,num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M' : 2,  'GROUP_SIZE_N' : 4}, num_stages=4,num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M' : 2,  'GROUP_SIZE_N' : 4}, num_stages=4,num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M' : 2,  'GROUP_SIZE_N' : 4}, num_stages=2,num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M' : 8,  'GROUP_SIZE_N' : 4}, num_stages=2,num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M' : 4,  'GROUP_SIZE_N' : 8}, num_stages=4,num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M' : 8}, num_stages=5,num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M' : 8}, num_stages=3,num_warps=4),
        
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=3 + 4,num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=5 + 4,num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=4 + 4,num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=1,    num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=1,    num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4 + 4,num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=5 + 4,num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3 + 4,num_warps=4),

        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256}, num_stages=3,num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=3,num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4,num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5,num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5,num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3,num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4,num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4,num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4,num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4,num_warps=4),
    

    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_basic_kernel(
        out_ptr, a_ptr, bt_ptr,
        M :tl.constexpr, 
        N :tl.constexpr, 
        K :tl.constexpr,
        stride_am, stride_ak, 
        stride_bk, stride_bn,
        stride_outm, stride_outn,
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
    a: (M, K)
    bt: (N, K)
    """
    # pid_m = pid_block % tl.cdiv(M, BLOCK_SIZE)
    # pid_n = (pid_block // tl.cdiv(M, BLOCK_SIZE)) % tl.cdiv(N, BLOCK_SIZE)
    # pid_block
    # offs_k = 
    # offs_am = 
    n_Nblocks = tl.cdiv(N, BLOCK_SIZE_N)
    n_Kblocks = tl.cdiv(K, BLOCK_SIZE_K)
    n_Mblocks = tl.cdiv(M, BLOCK_SIZE_M)

    pid = tl.program_id(axis=0)
    pid_per_group = GROUP_SIZE_M * GROUP_SIZE_N # so each group does 
                                                # M // group_size_m of M
                                                # and 1 of N ?
    group_id = pid // pid_per_group
    nb_per_group = tl.cdiv(n_Nblocks, GROUP_SIZE_N)
    mb_per_group = tl.cdiv(n_Mblocks, GROUP_SIZE_M)
    group_n0 = group_id % nb_per_group * GROUP_SIZE_N
    group_m0 = group_id // nb_per_group * GROUP_SIZE_M
    
    group_size_m = min(n_Mblocks - group_m0, GROUP_SIZE_M)
    group_size_n = min(n_Nblocks - group_n0, GROUP_SIZE_N)

    id_mb = group_m0 + pid % group_size_m
    id_nb = group_n0 + (pid // group_size_m) % group_size_n
    assert id_mb < n_Mblocks
    n0 = id_nb * BLOCK_SIZE_N 
    m0 = id_mb * BLOCK_SIZE_M
    offs_n = (tl.arange(0, BLOCK_SIZE_N) + n0) % N
    offs_m = (tl.arange(0, BLOCK_SIZE_M) + m0) % M
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=DTYPE_ACC)
    
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    for ki in range(n_Kblocks):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = bt_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

        ai = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        bi = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        # bi = tl.trans(bi)
        if TRANS:
            bi = tl.trans(bi)
        if DTYPE_AB is not None:
            ai = ai.to(DTYPE_AB)
            bi = bi.to(DTYPE_AB)
        acc += tl.dot(ai, bi).to(DTYPE_ACC)
        offs_k = offs_k + BLOCK_SIZE_K
    # if acc[0][0] != 0:
    #     print("acc")
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn

    if DTYPE_RET is not None:
        tl.store(out_ptrs, acc.to(DTYPE_RET), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    else:
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))



def matmul(a, b, outtype=torch.float32, DTYPE_AB = torch.float16):
    """
    a   (*B', M, K)
    b: (*B', K, N)
    """
    M, K = a.shape
    K, N = b.shape
    output = torch.zeros(M, N, device='cuda', dtype=outtype)

    if DTYPE_AB is not None:
        a = a.to(DTYPE_AB)
        b = b.to(DTYPE_AB)
    grid = lambda META: (
          triton.cdiv(M, META['BLOCK_SIZE_M']) 
        * triton.cdiv(N, META['BLOCK_SIZE_N']), 
    )
    bt = b.transpose(0, 1)
    tltype = lambda T: tl.float16 if T == torch.float16 else tl.float32

    matmul_basic_kernel[grid](
        output, a, bt,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0), output.stride(1),
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


    k = 4096
    m = 4096
    n = 4096


    test_mm_speeds.timetest(n, m, k, lambda *a, **k : matmul(*a, outtype=torch.float32, **k), "matmul fp32", dtype=torch.float32)
    print()
    test_mm_speeds.timetest(n, m, k, matmul, "matmul fp16", dtype=torch.float16)
    exit()

    test_matmul.runtests(matmul, n, m, k)
    # input()
    a = torch.rand(n, m, device='cuda', dtype=torch.float16)
    b = torch.randn(m, k, device='cuda', dtype=torch.float16)
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