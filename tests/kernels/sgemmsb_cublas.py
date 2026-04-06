import torch

from vit_quant import kernels

B, M, N, K = 1, 197, 1152, 384
i = torch.randn((B, M, K), device="cuda")
w = torch.randn((B, K, N), device="cuda")
o = torch.empty((B, M, N), device="cuda")

kernels.sgemm_cublas(i, w, o, alpha=1.0, beta=0.0)
res = torch.equal(o, i @ w)
print(f"gemmsb (B={B}, M={M}, N={N}, K={K}) => {'passed' if res else 'failed'}")
