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
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M' : 8,  'GROUP_SIZE_N' : 4}, num_stages=4,num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M' : 2,  'GROUP_SIZE_N' : 2}, num_stages=2,num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M' : 2,  'GROUP_SIZE_N' : 2}, num_stages=2,num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M' : 2,  'GROUP_SIZE_N' : 2}, num_stages=4,num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M' : 2,  'GROUP_SIZE_N' : 2}, num_stages=2,num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M' : 2,  'GROUP_SIZE_N' : 2}, num_stages=4,num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M' : 2,  'GROUP_SIZE_N' : 2}, num_stages=4,num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M' : 2,  'GROUP_SIZE_N' : 2}, num_stages=2,num_warps=8),
        
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
    n_Mblocks = tl.cdiv(M, BLOCK_SIZE_M) #-
    pid = tl.program_id(axis=0)
    # Mblocks_empty_per_last_group_m = GROUP_SIZE_M - n_Mblocks % GROUP_SIZE_M
    Mblocks_in_last_group_m = n_Mblocks % GROUP_SIZE_M
    Nblocks_in_last_group_n = n_Nblocks % GROUP_SIZE_N
                                                # M // group_size_m of M
                                                # and 1 of N ?
    pid_per_Medge_group = Mblocks_in_last_group_m * GROUP_SIZE_N
    pid_per_Nedge_group = GROUP_SIZE_M * Nblocks_in_last_group_n
    pid_per_corner_group = Mblocks_in_last_group_m * Nblocks_in_last_group_n

    pid_per_group = GROUP_SIZE_M * GROUP_SIZE_N # so each group does 


    # pid_per_M_direction = (n_Mblocks // GROUP_SIZE_M * pid_per_group + pid_per_Medge_group)
    core_group_id = pid // pid_per_group

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_core_groups_M = n_Mblocks // GROUP_SIZE_M
    num_core_groups_N = n_Nblocks // GROUP_SIZE_N
    if core_group_id < num_core_groups_M * num_core_groups_N:
        group_id_m = core_group_id % num_core_groups_M
        group_id_n = core_group_id // num_core_groups_M
        group_pid = pid % pid_per_group
    else:


        edge_pid = pid - num_core_groups_M * num_core_groups_N * pid_per_group
        if pid_per_Medge_group == 0:
            Medge_group_id = num_core_groups_N
        else:
            Medge_group_id = edge_pid // pid_per_Medge_group
        # tl.device_print("mp", pid_per_Medge_group, pid_per_Nedge_group)
        if Medge_group_id < num_core_groups_N:
            Medge_group_pid = edge_pid % pid_per_Medge_group
            # block is in a group along the M edge
            # tl.device_print("Medge")
            group_id_m = num_core_groups_M
            group_id_n = Medge_group_id
            # group_pid = edge_group_pid % pid_per_Medge_group WRONG
            group_pid = (
                   Medge_group_pid % Mblocks_in_last_group_m 
                + (Medge_group_pid // Mblocks_in_last_group_m) * GROUP_SIZE_M
            )
        else:

            Nedge_pid = edge_pid - pid_per_Medge_group * num_core_groups_N
            Nedge_group_pid = Nedge_pid % pid_per_Nedge_group
            Nedge_group_id = Nedge_pid // pid_per_Nedge_group
            if Nedge_group_id < num_core_groups_M:
                # along the N edge
                # tl.device_print("Nedge")
                group_id_m = Nedge_group_id
                group_id_n = num_core_groups_N
                group_pid = Nedge_group_pid % pid_per_Nedge_group # WRONG if there were more than 2dim but ok here

            else:
                # the "corner case"
                corner_group_pid = (
                    edge_pid 
                    - pid_per_Medge_group * num_core_groups_N 
                    - pid_per_Nedge_group * num_core_groups_M
                )
                group_id_m = num_core_groups_M
                group_id_n = num_core_groups_N
                group_pid = (
                    corner_group_pid % Mblocks_in_last_group_m
                    + (corner_group_pid // Mblocks_in_last_group_m) * GROUP_SIZE_M

                )
                # tl.device_print("corner")


        


    # old
    # pid_per_group = GROUP_SIZE_M * GROUP_SIZE_N # so each group does 
    # group_id = pid // pid_per_group
    # num_groups_M = tl.cdiv(n_Mblocks, GROUP_SIZE_M)
    # num_groups_N = tl.cdiv(n_Nblocks, GROUP_SIZE_N)
    # group_id_m = (group_id % num_groups_M)
    # group_id_n = group_id // num_groups_M
    # assert group_id_n < num_groups_N
    # assert group_id_m < num_groups_M #-

    # group_id = group_id // num_groups_M * Mblocks_empty_per_last_group_m # skip empty blocks in M direction
    group_n0 = group_id_n * GROUP_SIZE_N
    group_m0 = group_id_m * GROUP_SIZE_M #-
    group_size_m = min(num_pid_m - group_m0, GROUP_SIZE_M)
    group_size_n = min(num_pid_n - group_n0, GROUP_SIZE_N) #-
    group_size_m = GROUP_SIZE_M
    group_size_n = GROUP_SIZE_N

    # group_pid = 
    





    id_mb = group_m0 + group_pid % group_size_m
    id_nb = group_n0 + (group_pid // group_size_m) % group_size_n


    
    # tl.device_print(
    #     "\tpid:", pid, tl.num_programs(axis=0))
    # tl.device_print("pid_per_group:", pid_per_group)
    # tl.device_print("n_Nblocks:", n_Nblocks)
    # tl.device_print("n_Kblocks:", n_Kblocks)
    # tl.device_print("n_Mblocks:", n_Mblocks)
    # tl.device_print("group_id:", group_id)
    # tl.device_print("num_groups_N:", num_groups_N)
    # tl.device_print("num_groups_M:", num_groups_M)
    # tl.device_print("group_id_m:", group_id_m)
    # tl.device_print("group_id_n:", group_id_n)
    # tl.device_print("group_n0:", group_n0)
    # tl.device_print("group_m0:", group_m0)
    # tl.device_print("group_size_m:", group_size_m)
    # tl.device_print("group_size_n:", group_size_n)
    # tl.device_print("group_pid:", group_pid)
    # tl.device_print("id_mb:", id_mb)
    # tl.device_print("id_nb:", id_nb)
    

    # id_mb = group_m0 + group_pid % GROUP_SIZE_M
    # id_nb = group_n0 + (group_pid // GROUP_SIZE_M) % GROUP_SIZE_N

    assert id_mb < n_Mblocks
    m0 = id_mb * BLOCK_SIZE_M
    n0 = id_nb * BLOCK_SIZE_N 
    offs_m = tl.arange(0, BLOCK_SIZE_M) + m0
    offs_n = tl.arange(0, BLOCK_SIZE_N) + n0
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=DTYPE_ACC)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    for ki in range(n_Kblocks):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = bt_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

        ai = tl.load(a_ptrs, mask=(offs_m[:, None] < M) * (offs_k[None, :] < K), other=0.0)
        bi = tl.load(b_ptrs, mask=(offs_n[:, None] < N) * (offs_k[None, :] < K), other=0.0)
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
        tl.store(out_ptrs, acc.to(DTYPE_RET), mask=(offs_m[:, None] < M) * (offs_n[None, :] < N))
    else:
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) * (offs_n[None, :] < N))



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
          triton.cdiv(M, META['GROUP_SIZE_M'] * META['BLOCK_SIZE_M']) * META['GROUP_SIZE_M'] * 
          triton.cdiv(N, META['GROUP_SIZE_N'] * META['BLOCK_SIZE_N']) * META['GROUP_SIZE_N'], 
    )
    grid = lambda META: (
          triton.cdiv(M, META['BLOCK_SIZE_M']) * 
          triton.cdiv(N, META['BLOCK_SIZE_N']), 
    )
    bt = b.transpose(0, 1)
    tltype = lambda T: tl.float16 if T == torch.float16 else tl.float32

    matmul_basic_kernel[grid](
        out_ptr = output,
        a_ptr = a,
        bt_ptr = bt,
        M=M, N=N, K=K,
        stride_am = a.stride(0), stride_ak = a.stride(1),
        stride_bk = b.stride(0), stride_bn = b.stride(1),
        stride_outm = output.stride(0), stride_outn = output.stride(1),
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


    m = 64 * 2 * 4
    k = 2* 1024
    n = 256

    test_mm_speeds.timetest(m, k, n, lambda *a, **k : matmul(*a, outtype=torch.float32, **k), "matmul fp32", dtype=torch.float32)
    # print()
    test_mm_speeds.timetest(m, k, n, matmul, "matmul fp16", dtype=torch.float16)

    # input()
    m = 1 * 128 + 79 + 1 * 32 + 1024
    k = 1 * 512 + 79 - 3
    n = 128 + 64 + 256 - 5
    print(m, k, n)
    mb, nb = triton.cdiv(m, 32), triton.cdiv(n, 32)
    print(mb, nb, mb % 4, nb % 4)
    test_mm_speeds.timetest(m, k, n, lambda *a, **k : matmul(*a, outtype=torch.float32, **k), "matmul fp32", dtype=torch.float32)
    test_mm_speeds.timetest(m, k, n, matmul, "matmul fp16", dtype=torch.float16)
    def onesab(m, k, n):
        onesa = torch.ones((m, k), device='cuda', dtype=torch.float16)
        onesb = torch.ones((k, n), device='cuda', dtype=torch.float16)
        onesa[-1, 0] = -1
        onesa[0, -1] = -2
        onesb[-1, 0] = -3
        onesb[0, -1] = -4
        onesa[-1, -1] = 0
        onesb[-1, -1] = 0
        return onesa, onesb


    a,b = onesab(m, k, n)
    test_matmul.test(matmul, a, b, atol=1)
    input()



    for i in range(100):
        m = random.randint(1, 2048)
        k = random.randint(1, 2048)
        n = random.randint(1, 2048)
        print(i, m, k, n)
        a,b = onesab(m, k, n)
        if not test_matmul.test(matmul, a, b, print_error_radius=6, atol=1e-0):
            print("error", m, k, n)






    # test_matmul.test(matmul, onesa, onesb, print_error_radius=3)
    errors = []
    
    for i in range(100):
        m = random.randint(1, 4096)
        k = random.randint(1, 4096)
        n = random.randint(1, 4096)
        a = torch.randn(m, k, device='cuda', dtype=torch.float16)
        b = torch.randn(k, n, device='cuda', dtype=torch.float16)

        if not test_matmul.test(matmul, a, b, print_error_radius=3, atol=1e-0):
            errors.append((m, k, n))
            print("error", m, k, n)
        print(i, m, k, n)
        m = random.randint(1, 512)
        k = random.randint(1, 512)
        n = random.randint(1, 512)
        a = torch.randn(m, k, device='cuda', dtype=torch.float16)
        b = torch.randn(k, n, device='cuda', dtype=torch.float16)

        if not test_matmul.test(matmul, a, b, print_error_radius=3, atol=1e-0):
            errors.append((m, k, n))
            print("error", m, k, n)
        print(i, m, k, n)


    exit()
    test_matmul.runtests(matmul, m, k, n, print_error_radius=3)

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