from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_histogram(
    name: str, data: torch.Tensor, output_dir: str | Path, kind: str = "activation"
) -> None:
    output_dir = Path(output_dir) / "histogram" / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    flat = data.flatten().cpu().numpy()
    plt.figure(figsize=(6, 4))
    # sns.histplot(flat, bins=100, kde=True)
    sns.histplot(flat, bins=100)
    plt.title(f"{kind.capitalize()} Histogram - {name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name.replace('.', '_')}.png")
    plt.close()


def plot_heatmap(
    name: str, data: torch.Tensor, output_dir: str | Path, kind: str = "activation"
) -> None:
    if data.ndim == 4:  # (B, C, H, W)
        flat = data.squeeze(0).view(data.size(1), -1).cpu()  # (C, H*W)
    elif data.ndim == 3:  # (B, N, D)
        flat = data.squeeze(0).cpu()  # (N, D)
    elif data.ndim == 2:  # (B, D)
        flat = data.squeeze(0).unsqueeze(0).cpu()  # (1, D)
    else:
        raise ValueError(f"Unsupported tensor shape: {data.shape}")

    output_dir = Path(output_dir) / "heatmap" / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(flat, cmap="coolwarm", cbar=True)
    plt.title(f"{kind.capitalize()} Heatmap - {name}")
    plt.xlabel("Channel")
    plt.ylabel("Token")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name.replace('.', '_')}.png")
    plt.close()
