import random
from unpythonic import box
import math
import triton




def pows2(a,b):
    a2 = math.log2(a)
    b2 = math.log2(b)
    assert a2.is_integer() and b2.is_integer()
    l = []
    for i in range(int(a2), int(b2)+1):
        l.append(2**i)
    return l


def select(element, mode="random"):
    if mode == "count":
        if isinstance(element, int):
            return 1
        else:
            return len(element)
    if isinstance(element, int):
        return element
    elif mode == "random":
        return random.choice(element)
    elif isinstance(mode, box):
        i = mode.x % len(element)
        mode.x = mode.x // len(element)
        return element[i]


def defconfigspace(num_stages = range(1, 5), num_warps = pows2(1, 8), **kwargs):
    d = {
        "num_stages" : num_stages,
        "num_warps" : num_warps,
        **kwargs
    }
    return d


def to_config(configspace, mode="random"):
    if isinstance(mode, int):
        mode = box(mode)
    d = {
        k : select(v, mode)
        for k, v in configspace.items()
    }
    num_stages = d["num_stages"]
    num_warps = d["num_warps"]
    del d["num_stages"]
    del d["num_warps"]
    return triton.Config(d, num_stages=num_stages, num_warps=num_warps)

def to_configs(configspace, n, end=None, mode="random"):
    if end is None:
        n = range(n)
    else:
        n = range(n, end)
        mode = "iter"
    if mode == "random":
        return [
            to_config(configspace, mode)
            for _ in n
        ]
    elif mode == "iter":
        return (
            to_config(configspace, i)
            for i in n
        )
    
def count_configspace(configspace):
    return math.prod([
        select(v, "count")
        for k, v in configspace.items()
    ])

batchmm = defconfigspace(
    BLOCK_SIZE_B = 1    ,
    GROUP_SIZE_B1 = pows2(1, 32)   ,
    GROUP_SIZE_B2 = pows2(1, 32)   ,
    BLOCK_SIZE_M = pows2(32, 128)  ,
    BLOCK_SIZE_N = pows2(32, 128)  ,
    BLOCK_SIZE_K = pows2(32, 256)  ,
    GROUP_SIZE_N = pows2(1, 64)    ,
    GROUP_SIZE_M = pows2(1, 64)    ,
)

print(to_configs(batchmm, 10))

print(count_configspace(batchmm))