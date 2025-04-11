import os
from typing import Any

from torchvision import datasets, transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from src.dataset.base import LoaderGenerator


class ImageNetLoaderGenerator(LoaderGenerator):
    def __init__(
        self,
        root: str,
        *,
        model: str | None = None,
        num_workers: int = 16,
        pin_memory: bool = True,
        **common_args,
    ):
        super().__init__(
            root, num_workers=num_workers, pin_memory=pin_memory, **common_args
        )
        self.root = root
        self.model = model

    def train_set(self):
        return datasets.ImageFolder(
            root=os.path.join(self.root, "ILSVRC2012_img_train"),
            transform=get_imagenet_transform(
                is_training=True, model=self.model
            ),
        )

    def test_set(self):
        return datasets.ImageFolder(
            root=os.path.join(self.root, "val"),
            transform=get_imagenet_transform(
                is_training=False, model=self.model
            ),
        )


def get_imagenet_transform(is_training: bool = False, model: Any | None = None):
    # Use timm's data config if available
    if model:
        config = resolve_data_config(model=model)
        return create_transform(**config, is_training=is_training)

    # Fallback to torchvision transforms
    if is_training:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
