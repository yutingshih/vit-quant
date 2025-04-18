#!/usr/bin/env python3

from pathlib import Path

import typer
import timm
import torch
import numpy as np

from src.dataset.imagenet import ImageNetLoaderGenerator as ImageNet
from src.analysis.hook import get_activations, get_weights
from src.utils.plot import plot_histogram, plot_heatmap
from src.utils.evaluate import evaluate


app = typer.Typer()


@app.command(name="eval", help="Evaluate a model on ImageNet-1K dataset.")
def evaluate_model(
    model: str = "vit_small_patch16_224",
    dataset: str = "./data/imagenet/image_dir",
    batch_size: int = 256,
    device: str = "cuda",
) -> None:
    print(f"[*] Evaluating model {model} ...")
    model = timm.create_model(model, pretrained=True).eval().to(device)

    dataset_path = Path(dataset).absolute()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    print(f"Dataset path: {dataset_path}")

    imagenet = ImageNet(root=dataset_path, model=model)
    test_loader = imagenet.test_loader(batch_size=batch_size)

    print(f"number of images: {len(test_loader.dataset)}")
    print(f"batch size: {batch_size}")
    print(f"number of batches: {len(test_loader)}")

    res = evaluate(model=model, loader=test_loader, device=device)
    print(f"evaluation time: {res['time']:.2f} seconds")
    print(f"top-1 accuracy: {res['top1']:.4f}")
    print(f"top-5 accuracy: {res['top5']:.4f}")


@app.command(help="Infer a model on a single image from ImageNet-1K.")
def infer(
    model_name: str = typer.Option(
        "vit_small_patch16_224", "-m", "--model", help="Name of a timm model."
    ),
    dataset: str = "./data/imagenet/image_dir",
    index: int = typer.Option(0, "-i", "--index", help="Index of the image to infer."),
    device: str = "cuda",
) -> None:
    print(f"Inferencing image at index {index} ...")

    model = timm.create_model(model_name, pretrained=True).eval().to(device)
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
    output_dir: str = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file for model information. Default: logs/model_info/{model}.txt",
    ),
) -> None:
    output = Path(output_dir or f"./logs/model_info/{model}.txt").absolute()
    output.parent.mkdir(exist_ok=True, parents=True)
    model = timm.create_model(model, pretrained=True)
    print(model, file=open(output, "w"))
    print(f"Model information dumped to {output}")


@app.command(help="Dump input tensors, output tensors, and weights of a model.")
def dump_tensors(
    model: str = "vit_small_patch16_224",
    dataset: str = "./data/imagenet/image_dir",
    output_dir: str = "./tensors",
    device: str = "cuda",
    format: str = typer.Option(
        "pt",
        "-f",
        "--format",
        help="Output format for tensors. Options: 'pt' or 'npz'.",
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
    dataset: str = "./data/imagenet/image_dir",
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
        plot_histogram(name, act, outdir, kind="activation")
        plot_heatmap(name, act, outdir, kind="activation")

    print("[*] Plotting weights...")
    for name, weight in weights.items():
        plot_histogram(name, weight, outdir, kind="weight")

    print(f"[*] Analysis completed. Plots saved to {outdir}")


if __name__ == "__main__":
    app()
