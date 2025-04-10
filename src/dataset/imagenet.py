import os

from torchvision import transforms, datasets
from torch.utils.data import DataLoader


class ImageNet:
    NORMALOZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    IMAGE_SIZE = 224

    def __init__(
        self,
        root: str,
        use_gpu: bool = True,
    ) -> None:
        self.root = root
        self.use_gpu = use_gpu
        self._train_loader = None
        self._val_loader = None

    def get_train_loader(self, batch_size: int = 256) -> DataLoader:
        if self._train_loader is None:
            self._train_loader = DataLoader(
                datasets.ImageFolder(
                    os.path.join(self.root, "ILSVRC2012_img_train"),
                    transforms.Compose(
                        [
                            transforms.RandomResizedCrop(self.IMAGE_SIZE),
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize(self.IMAGE_SIZE),
                            transforms.ToTensor(),
                            self.NORMALOZE,
                        ]
                    ),
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=self.use_gpu,
            )
        return self._train_loader

    def get_val_loader(self, batch_size: int = 1000) -> DataLoader:
        if self._val_loader is None:
            self._val_loader = DataLoader(
                datasets.ImageFolder(
                    os.path.join(self.root, "val"),
                    transforms.Compose(
                        [
                            transforms.Resize(256),
                            transforms.CenterCrop(self.IMAGE_SIZE),
                            transforms.Resize(self.IMAGE_SIZE),
                            transforms.ToTensor(),
                            self.NORMALOZE,
                        ]
                    ),
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=self.use_gpu,
            )
        return self._val_loader
