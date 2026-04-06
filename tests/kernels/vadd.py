import torch

from vit_quant import kernels

num = 1000
a = torch.randn((num,), device="cuda")
b = torch.randn((num,), device="cuda")
c = torch.empty_like(a)
kernels.vadd(a, b, c)

res = torch.equal(c, a + b)
print(f"vadd (num={num}) => test {'passed' if res else 'failed'}")
