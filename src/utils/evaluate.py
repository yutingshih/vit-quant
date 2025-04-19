from time import time

from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm


def evaluate(model: str, loader: str, device: str = "cuda") -> None:
    t = time()
    top1_acc = MulticlassAccuracy(num_classes=1000, top_k=1).to(device)
    top5_acc = MulticlassAccuracy(num_classes=1000, top_k=5).to(device)
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        top1_acc(logits, labels)
        top5_acc(logits, labels)

    return {
        "top1": top1_acc.compute(),
        "top5": top5_acc.compute(),
        "time": time() - t,
    }
