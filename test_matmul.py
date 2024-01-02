from numba import cuda
import torch
from nqgl.sae.hsae.spbmbmm import baseline


def testmats(n, m, k, skipfreq=2):
    d = {}

    d["eye_n"] = (
        torch.eye(n, device="cuda"),
        torch.eye(n, device="cuda")
    )
    d["eye_m"] = (
        torch.eye(m, device="cuda"),
        torch.eye(m, device="cuda")
    )
    d["eye_k"] = (
        torch.eye(k, device="cuda"),
        torch.eye(k, device="cuda")
    )
    d["ones"] = (
        torch.ones(n, m, device="cuda"),
        torch.ones(m, k, device="cuda")
    )
    a = torch.ones(n, m, device="cuda")
    b = torch.ones(m, k, device="cuda")
    # a[n//2:, m//2:] = -1
    # a[0,0] = -2
    # a[0, m-1] = -3
    # a[n-1, 0] = -5
    b[m//2:, k//2:] = -1
    # b[0,0] = -2
    # b[0, k-1] = -3
    # b[m-1, 0] = -5


    d["ones_corner_zero_a"] = (
        a,b
    )

    # d["arange"] = (
    #     torch.arange(0, n * m, device="cuda", dtype="float32").reshape(n, m),
    #     torch.arange(0, k * m, device="cuda", dtype="float32").reshape(m, k)
    # )
    d["rand"] = (
        torch.rand(n, m, device="cuda"),
        torch.rand(m, k, device="cuda")
    )
    d["rand1n"] = (
        torch.randn(n, m, device="cuda"),
        torch.rand(m, k, device="cuda")
    )
    # for skip0 in [0, 1]:
    #     for skip1 in [0, 1]:
    #         d[f"ones_skip{skip0}{skip1}even"] = (
    #             torch.ones(n, m, device="cuda"),
    #             torch.ones(m, k, device="cuda")
    #         )
    #         d[f"ones_skip{skip0}{skip1}odd"] = (
    #             torch.ones(n, m, device="cuda"),
    #             torch.ones(m, k, device="cuda")
    #         )
    #         mats_even = d[f"ones_skip{skip0}{skip1}even"]
    #         mats_odd = d[f"ones_skip{skip0}{skip1}odd"]
            
    #         for i in range(n):
    #             for j in range(m):
    #                 if skip0 == 0:
    #                     if j % skipfreq == 0:
    #                         mats_even[0][:, j] = 0
    #                     else:
    #                         mats_odd[0][:, j] = 0
    #                 else:
    #                     if i % skipfreq == 0:
    #                         mats_even[0][:, j] = 0
    #                     else:
    #                         mats_odd[0][:, j] = 0
                    
    #         for i in range(m):
    #             for j in range(k):
    #                 if skip1 == 0:
    #                     if j % skipfreq == 0:
    #                         mats_even[1][:, j] = 0
    #                     else:
    #                         mats_odd[1][:, j] = 0
    #                 else:
    #                     if i % skipfreq == 0:
    #                         mats_even[1][:, j] = 0
    #                     else:
    #                         mats_odd[1][:, j] = 0
    return d    

def colprint(l, spacing = 4, linesep = False):
    num_cols = max([len(ll) for ll in l])
    l = [[str(e).split("\n") for e in ll] for ll in l]

    maxlines_per_row = [max([len(col) for col in row]) for row in l]
    expanded = [
        [
            [   
                col[cline] if cline < len(col) else ""
                for j, col in enumerate(row)
            ]
            for cline in range(maxlines_per_row[i])
            
        ] + [[""] * (num_cols)] * linesep
        for i, row in enumerate(l)
    ]
    

    # l = [
    #     [
    #         row[coln][i]
    #         for coln in range(num_cols)
    #     ]
    #     for j, row in enumerate(expanded)
    #     for i in range(maxlines_per_col[j])
    # ]
    # l = [
    #     [
    #         expanded[i][j][k]
    #         for j in range(len(expanded[i]))
    #     ]
    #     for i in range(len(expanded))
    #     for k in range(maxlines_per_row[i])
    # ]
    # l = expanded
    colwidths = [
    
        max([
            max([
                len(colrow) + spacing
                for colrow in row[j]
            ]) if j < len(row) else 0
            for row in expanded
        ])
        for j in range(num_cols)
    ]
    for l in expanded:

        for line in l:
            print("".join([ 
                line[i].ljust(colwidths[i])
                                            
                for i in range(len(line))])) 



def tzoom(t, coords, rad):
    return t[coords[0] - rad: coords[0] + rad + 1, coords[1] - rad: coords[1] + rad + 1]

def zoomer(t, coords, rad, step=1):
    i = 0
    while i != "q":
        print(tzoom(t, coords, rad))
        i = input()
        if i == "w":
            coords[0] -= step
        elif i == "s":
            coords[0] += step
        elif i == "a":
            coords[1] -= step
        elif i == "d":
            coords[1] += step
        elif i == "e":
            rad += 1
        elif i == "r":
            rad -= 1
        elif i == "q":
            break
        else:
            print("invalid input")


def print_error_region(res, res2, radius, coords):
    res = res.clone()
    res2 = res2.clone()
    region = (
        (
            max(coords[0] - radius, 0), 
            max(coords[1] - radius, 0)
        ), (
            min(coords[0] + radius + 1, res.shape[0]), 
            min(coords[1] + radius + 1, res.shape[1])
        )
    )
    r = res[region[0][0]: region[1][0], region[0][1]: region[1][1]]
    r2 = res2[region[0][0]: region[1][0], region[0][1]: region[1][1]]
    print(f"looking inside a {res.shape} at {coords} with radius {radius}")
    print(f"region is ({region[0][0]}:{region[1][0]}, {region[0][1]}:{region[1][1]})")
    print(f"\nresult:\n", r)
    print(f"\nactual:\n", r2)
    print(f"\ndiff:\n", r - r2)


def findwrong(res, res2, print_error_radius = 3, interact=False, atol=1e-2, rtol=1e-2):
    success = torch.allclose(res, res2, atol=atol, rtol=rtol)
    if not success:
        absdiff = torch.abs(res - res2)
        print(res.shape)
        max_diff = torch.max(absdiff, dim=1)
        maxx = torch.argmax(max_diff.values)
        # max_diff_coords_x = max_diff.indices[]
        maxy = max_diff.indices[torch.argmax(max_diff.values)]
        max_diff_coords = (maxx.item(), maxy.item())

        if print_error_radius:
            print_error_region(res, res2, print_error_radius, max_diff_coords)
        pass
    return success


def test(matmul, a, b, name=None, **kwargs):
    print(name) if name is not None else None
    res = matmul(a.clone(), b.clone())
    res2 = a.clone() @ b.clone()
    if res.dtype != res2.dtype:
        print("dtype mismatch")
        res2 = res2.to(res.dtype)
    success = findwrong(res, res2, **kwargs)
    # line = [name, success]
    return success





def runtests(matmul, n, m, k, atol=1e-2, rtol=1e-2, print_error_radius=0):
    d = testmats(n=n, k=k, m=m)
    l = []
    failed = []
    for k, v in d.items():
        print(k)
        a, b = v
        res = matmul(a.clone(),b.clone())
        res2 = a @ b
        success = torch.allclose(res, res2, atol=atol, rtol=rtol)
        line = [k, success]
        if not success:
            failed += [[k, res]]
            absdiff = torch.abs(res - res2)
            print(res.shape)
            max_diff = torch.max(absdiff, dim=1)
            maxx = torch.argmax(max_diff.values)
            # max_diff_coords_x = max_diff.indices[]
            maxy = max_diff.indices[torch.argmax(max_diff.values)]
            max_diff_coords = (maxx.item(), maxy.item())
            line += [max_diff_coords, torch.max(absdiff).item()]

            if print_error_radius:
                print_error_region(res, res2, print_error_radius, max_diff_coords)
            pass
        l += [line]
        print("done")
    if failed:
        colprint(failed)
    colprint(l)


def main():
    s = "aaaa.bbb.c\nc.aaaa.b\nb.c\nccc.aaaa.b.c"
    l = s.split(".")
    l = [l[0:3], l[3:6], l[6:9]]
    print(l)
    colprint(l, spacing=4)

if __name__ == "__main__":
    main()
