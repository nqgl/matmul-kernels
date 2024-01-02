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
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=3,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=5,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=2,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=2,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=5,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3,num_warps=4),
        
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
    mod = tl.cdiv(n_Nblocks, 2)
    imod = tl.cdiv(n_Nblocks * n_Mblocks, 2)
    # imod = 32
    # mod = 2
    # imod = 2
    # mod = 8
    
    pid = modulate(pid, mod, n_Mblocks * n_Nblocks, mod * mod)
    pid = modulate(pid, imod, n_Mblocks * n_Nblocks, imod * imod)
    id_nb = (pid) % n_Nblocks
    id_mb = (pid) // n_Nblocks
    assert id_mb < n_Mblocks
    n0 = id_nb * BLOCK_SIZE_N 
    m0 = id_mb * BLOCK_SIZE_M
    offs_n = tl.arange(0, BLOCK_SIZE_N) + n0
    offs_m = tl.arange(0, BLOCK_SIZE_M) + m0
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    for ki in range(n_Kblocks):
        offs_k = tl.arange(0, BLOCK_SIZE_K) + ki * BLOCK_SIZE_K

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = bt_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

        ai = tl.load(a_ptrs, mask=(offs_k[None, :] < K) * (offs_m[:, None] < M), other=0.0)
        bi = tl.load(b_ptrs, mask=(offs_k[None, :] < K) * (offs_n[:, None] < N), other=0.0)
        bit = tl.trans(bi)
        acc += tl.dot(ai.to(tl.float16), bit.to(tl.float16)).to(tl.float16)
    # if acc[0][0] != 0:
    #     print("acc")
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    tl.store(out_ptrs, acc.to(tl.float32), mask=(offs_m[:, None] < M) * (offs_n[None, :] < N))



def matmul(a, b, dtype=torch.float32):
    """
    a   (*B', M, K)
    b: (*B', K, N)
    """
    M, K = a.shape
    K, N = b.shape
    output = torch.zeros(M, N, device='cuda', dtype=dtype)

    grid = lambda META: (
          triton.cdiv(M, META['BLOCK_SIZE_M']) 
        * triton.cdiv(N, META['BLOCK_SIZE_N']), 
    )
    bt = b.transpose(0, 1)

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
    )

    return output


def main():

    i = 0


    k = 512
    m = 512
    n = 512
    test_matmul.runtests(matmul, n, m, k)
    # input()
    a = torch.rand(n, m, device='cuda', dtype=torch.float16)
    b = torch.randn(m, k, device='cuda', dtype=torch.float16)
    # a[7, 37] = -77
    c = matmul(a.clone(), b.clone())
    print(torch.allclose(c, torch.matmul(a,b), rtol=1e-4, atol=1e-2))
    while not torch.allclose(c, torch.matmul(a, b), rtol=1e-4, atol=1e-2):
        a = torch.rand(n, m, device='cuda')
        b = torch.randn(m, k, device='cuda')
        c = matmul(a, b)
        i += 1
        print(i)
    


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