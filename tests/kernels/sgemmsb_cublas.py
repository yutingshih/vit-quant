import torch

from vit_quant import kernels

# ViT-S-16-224
B, N, D, H = 10, 197, 384, 6


def verify(out: torch.Tensor, ref: torch.Tensor, name: str = "gemmsb") -> None:
    res = torch.allclose(out, ref, atol=1e-3, rtol=1e-4)
    diff = torch.abs(out - ref).max().item()
    status = "passed" if res else "failed"
    print(f"{name} => {status} (max diff: {diff:.2e})")


def qkv_proj() -> None:
    i = torch.randn((B, N, D), device="cuda")
    w = torch.randn((D, 3 * D), device="cuda")
    o = torch.empty((B, N, 3 * D), device="cuda")
    kernels.sgemm_cublas(i, w, o, alpha=1.0, beta=0.0, strb=0)
    verify(o, i @ w, "qkv_proj")


def out_proj() -> None:
    i = torch.randn((B, N, D), device="cuda")
    w = torch.randn((D, D), device="cuda")
    o = torch.empty((B, N, D), device="cuda")
    kernels.sgemm_cublas(i, w, o, alpha=1.0, beta=0.0, strb=0)
    verify(o, i @ w, "out_proj")


def qk_matmul() -> None:
    i = torch.randn((B * H, N, D // H), device="cuda")
    w = torch.randn((B * H, D // H, N), device="cuda")
    o = torch.empty((B * H, N, N), device="cuda")
    kernels.sgemm_cublas(i, w, o, alpha=1.0, beta=0.0)
    verify(o, i @ w, "qk_matmul")


def qkv_matmul() -> None:
    i = torch.randn((B * H, N, N), device="cuda")
    w = torch.randn((B * H, N, D // H), device="cuda")
    o = torch.empty((B * H, N, D // H), device="cuda")
    kernels.sgemm_cublas(i, w, o, alpha=1.0, beta=0.0)
    verify(o, i @ w, "qkv_matmul")


def fc1() -> None:
    i = torch.randn((B, N, D), device="cuda")
    w = torch.randn((D, 4 * D), device="cuda")
    o = torch.empty((B, N, 4 * D), device="cuda")
    kernels.sgemm_cublas(i, w, o, alpha=1.0, beta=0.0, strb=0)
    verify(o, i @ w, "fc1")


def fc2() -> None:
    i = torch.randn((B, N, 4 * D), device="cuda")
    w = torch.randn((4 * D, D), device="cuda")
    o = torch.empty((B, N, D), device="cuda")
    kernels.sgemm_cublas(i, w, o, alpha=1.0, beta=0.0, strb=0)
    verify(o, i @ w, "fc2")


if __name__ == "__main__":
    qkv_proj()
    out_proj()
    qk_matmul()
    qkv_matmul()
    fc1()
    fc2()
