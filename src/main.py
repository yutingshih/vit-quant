#!/usr/bin/env python3

from time import time
from pathlib import Path

import typer
import timm
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from src.dataset.imagenet import ImageNetLoaderGenerator as ImageNet


app = typer.Typer()


@app.command(name="eval", short_help="Evaluate a model on ImageNet-1K dataset.")
def evaluate(
    model: str | None = None,
    dataset: str | None = None,
    batch_size: int = 256,
    device: str = "cuda",
):
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


if __name__ == "__main__":
    app()
