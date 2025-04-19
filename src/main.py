#!/usr/bin/env python3

from time import time
from pathlib import Path

import typer
import timm
import torch
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
import numpy as np
from rich.traceback import install

from src.dataset.imagenet import ImageNetLoaderGenerator as ImageNet
from src.analysis.hook import get_activations, get_weights
from src.utils.plot import plot_histogram, plot_heatmap

app = typer.Typer()
install(show_locals=False)

@app.command(name="eval", help="Evaluate a model on ImageNet-1K dataset.")
def evaluate(
    model: str | None = None,
    dataset: str | None = None,
    batch_size: int = 256,
    device: str = "cuda",
) -> None:
    model = model or "vit_small_patch16_224"
    model = timm.create_model(model, pretrained=True).eval().to(device)

    dataset_path = Path(dataset or "./datasets/imagenet/image_dir").absolute()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    print(f"Dataset path: {dataset_path}")

    imagenet = ImageNet(root=dataset_path, model=model)
    test_loader = imagenet.test_loader(batch_size=batch_size)

    print(f"number of images: {len(test_loader.dataset)}")
    print(f"batch size: {batch_size}")
    print(f"number of batches: {len(test_loader)}")

    t = time()
    top1_acc = MulticlassAccuracy(num_classes=1000, top_k=1).to(device)
    top5_acc = MulticlassAccuracy(num_classes=1000, top_k=5).to(device)
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        top1_acc(logits, labels)
        top5_acc(logits, labels)

    print(f"evaluation time: {time() - t:.2f} seconds")
    print(f"top-1 accuracy: {top1_acc.compute(): .4f}")
    print(f"top-5 accuracy: {top5_acc.compute(): .4f}")


@app.command(help="Infer a model on a single image from ImageNet-1K.")
def infer(
    model: str = "vit_small_patch16_224",
    dataset: str = "./datasets/imagenet/image_dir",
    index: int = typer.Option(0, "-i", "--index", help="Index of the image to infer."),
    device: str = "cuda",
) -> None:
    print(f"Inferencing image at index {index} ...")

    model = timm.create_model(model, pretrained=True).eval().to(device)
    dataset_path = Path(dataset).absolute()
    imagenet = ImageNet(root=dataset_path, model=model)
    test_loader = imagenet.test_loader(batch_size=1)

    for i, (image, label) in enumerate(test_loader):
        if i == index:
            break

    image, label = image.to(device), label.to(device)
    output = model(image).argmax(dim=1)
    print(f"Output: {output}")
    print(f"Label: {label}")


@app.command(name="dump-arch", help="Dump the model information.")
def dump_model_info(
    model: str = typer.Argument("vit_small_patch16_224", help="Name of a timm model."),
    output: str = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file for model information. Default: logs/model_info/{model}.txt",
    ),
) -> None:
    output = Path(output or f"./logs/model_info/{model}.txt").absolute()
    output.parent.mkdir(exist_ok=True, parents=True)
    model = timm.create_model(model, pretrained=True)
    print(model, file=open(output, "w"))
    print(f"Model information dumped to {output}")


@app.command(help="Dump input tensors, output tensors, and weights of a model.")
def dump_tensors(
    model: str = "vit_small_patch16_224",
    dataset: str = "./datasets/imagenet/image_dir",
    output_dir: str = "./tensors",
    device: str = "cuda",
    format: str = typer.Option(
        "pt", "-f", "--format", help="Output format for tensors. Options: 'pt' or 'npz'."
    ),
) -> None:
    if format not in ["pt", "npz"]:
        raise ValueError("Format must be either 'pt' or 'npz'.")

    output_path = Path(output_dir) / model
    output_path.mkdir(parents=True, exist_ok=True)

    model = timm.create_model(model, pretrained=True).eval().to(device)
    dataset_path = Path(dataset).absolute()
    imagenet = ImageNet(root=dataset_path, model=model)
    test_loader = imagenet.test_loader(batch_size=1)
    image = next(iter(test_loader))[0].to(device)

    print("[*] Extracting activations and weights...")
    inputs, outputs = get_activations(model, image)
    weights = get_weights(model)

    print("[*] Dumping tensors...")
    match format:
        case "pt":
            torch.save(inputs, output_path / "inputs.pt")
            torch.save(outputs, output_path / "outputs.pt")
            torch.save(weights, output_path / "weights.pt")
        case "npz":
            np.savez(
                output_path / "inputs.npz",
                **{k: v.cpu().numpy() for k, v in inputs.items()},
            )
            np.savez(
                output_path / "outputs.npz",
                **{k: v.cpu().numpy() for k, v in outputs.items()},
            )
            np.savez(
                output_path / "weights.npz",
                **{k: v.cpu().numpy() for k, v in weights.items()},
            )
    print(f"[*] Tensors dumped to {output_path}")


@app.command(help="Plot histograms and heatmaps of activations and weights.")
def plot(
    model: str = "vit_small_patch16_224",
    dataset: str = "./datasets/imagenet/image_dir",
    output: str = "./logs/analysis",
    device: str = "cuda",
) -> None:
    outdir = Path(output)
    outdir.mkdir(parents=True, exist_ok=True)

    model = timm.create_model(model, pretrained=True).eval().to(device)
    dataset_path = Path(dataset).absolute()
    imagenet = ImageNet(root=dataset_path, model=model)
    test_loader = imagenet.test_loader(batch_size=1)

    images, _ = next(iter(test_loader))
    images = images.to(device)

    print("[*] Extracting activations and weights...")
    inputs, outputs = get_activations(model, images)
    weights = get_weights(model)

    print("[*] Plotting activations...")
    for name, act in outputs.items():
        plot_histogram(name, act, outdir , kind="activation")
        plot_heatmap(name, act, outdir , kind="activation")

    print("[*] Plotting weights...")
    for name, weight in weights.items():
        plot_histogram(name, weight, outdir , kind="weight")

    print(f"[*] Analysis completed. Plots saved to {outdir}")


if __name__ == "__main__":
    app()
