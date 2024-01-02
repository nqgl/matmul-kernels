import torch
# from triton_examples import matmul_tri_transposed
# from test_matmul import colprint

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



def timetest(m, k, n, matmul, name, iterations = 3, check=True, dtype = torch.float32):
    al = [torch.rand(m, k, device='cuda', dtype=dtype) for _ in range(iterations)]
    bl = [torch.randn(k, n, device='cuda', dtype=dtype) for _ in range(iterations)]

    
    
    cl = []
    ts = []
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for a, b, start, end in zip(al, bl, starts, ends):
        a_ = a.clone()  
        b_ = b.clone()
        start.record()
        cl += [matmul(a_, b_)]
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end)
        ts.append(t)
        # print(f"{name}: {t}")
    l = [f"{name}: "]
    for a, b, c, t in zip(al, bl, cl, ts):
        l.append(t)
        if check:
            val = torch.matmul(a.clone(),b.clone()).to(c.dtype)
            # print(f"val: {val.dtype}, c: {c.dtype}")
            valid = torch.allclose(c, val, rtol=0, atol=1e-0)
            l.append(valid)
    print(*l)
    return l
    



def test():
    import custom_matmul_basic
    import custom_matmul_boxed
    from triton_examples import matmul_tri_ex
    import tri_matmul_grouped
    import tri_matmul_basic_port
    from cust_tri_matmuls import tri_matmul_2group

    k = 4096
    m = 4096
    n = 4096

    mmd = {
        "torch": lambda a, b: a @ b,
        "triton tutorial": matmul_tri_ex.matmul,
        "custom tri grouped" : tri_matmul_grouped.matmul,
        "custom tri 2grouped" : tri_matmul_2group.matmul,
        "custom tri basic": tri_matmul_basic_port.matmul,
        # "custom boxed": custom_matmul_boxed.matmul,
        # "custom basic": custom_matmul_basic.matmul,
    
    }
    l = []
    for name, matmul in mmd.items():
        l += [
            timetest(n, m, k, matmul, name, dtype=torch.float32)
        ]
    # colprint(l)
    l2 = []
    print("fp16")
    for name, matmul in mmd.items():
        l2 += [
            timetest(n, m, k, matmul, name, dtype=torch.float16)
        ]


    print("\n\n")
    exit()

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
    c = matmul_tri_ex.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    print("triton tutorial:", start.elapsed_time(end))

    a = torch.rand(n, m, device='cuda')
    b = torch.randn(m, k, device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = matmul_tri_ex.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    print("triton tutorial 2nd run:", start.elapsed_time(end))


    ### Custom
    print("# CUSTOM")

    a = torch.rand(n, m, device='cuda')
    b = torch.randn(m, k, device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    
    start.record()
    c = custom_matmul_boxed.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    print("boxed:", start.elapsed_time(end))


    a = torch.rand(n, m, device='cuda')
    b = torch.randn(m, k, device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    
    start.record()
    c = custom_matmul_boxed.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    print("boxed:", start.elapsed_time(end))


    a = torch.rand(n, m, device='cuda')
    b = torch.randn(m, k, device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    
    start.record()
    c = custom_matmul_boxed.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    print("boxed:", start.elapsed_time(end))


    a = torch.rand(n, m, device='cuda')
    b = torch.randn(m, k, device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = custom_matmul_basic.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    print("basic:", start.elapsed_time(end))



    a = torch.rand(n, m, device='cuda')
    b = torch.randn(m, k, device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = custom_matmul_basic.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    print("basic:", start.elapsed_time(end))

    a = torch.rand(n, m, device='cuda')
    b = torch.randn(m, k, device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = custom_matmul_basic.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    print("basic:", start.elapsed_time(end))
def main():
    test()

if __name__ == "__main__":
    main()