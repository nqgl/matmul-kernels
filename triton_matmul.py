import torch
import triton
import triton.language as tl



@triton.jit
def matmul_basic_triton_kernel(output, a, bt):
    bid = tl.program_id(axis=0)
    tid = tl.id




def matmul(a, b):

    output = torch.empty((a.shape[0], b.shape[1]), device='cuda')

    n_elements = output.shape[0]

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    bt = b.transpose(0, 1)

    matmul_basic_triton_kernel[grid](output, a, bt)

    return output