#%%
def modulate(i, mod, n):
    if i > n - n % mod ** 2:
        return i
    else:
        return i // mod + i % mod * mod + (mod ** 2 - mod) * (i // mod**2)
#%%
def modulate2(i, mod, n):
    # mod2 = mod * mod
    # if i > n - n % mod2:
    #     return i
    # else:
    return i // mod * mod + i % mod #+ (mod2 - mod) * (i // mod2)
    
#%%
    
def check(n, mod):
    c = [
        modulate(i, mod, n)
        for i in range(n)
    ]
    a = list(range(n))

    assert set(a) == set(c)
def check2(n, mod):
    c = [
        modulate2(i, mod, n)
        for i in range(n)
    ]
    a = list(range(n))
    # print(set(a) - set(c), n, mod)
    # print(c)
    assert set(a) == set(c)


#%%
for n in range(1, 1000):
    print(n)
    for mod in range(2, 1000):
        check2(n, mod)

# %%
set(a) == set(b)
# %%
set(a) == set(c)
# %%
set(a) - set(c)

# %%
set(c) - set(a)
# %%
