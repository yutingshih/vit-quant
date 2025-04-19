from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import Dataset, DataLoader


class LoaderGenerator(ABC):
    def __init__(
        self,
        root: str | Path,
        *,
        num_workers: int = 16,
        pin_memory: bool = True,
        **common_args,
    ):
        self.root = root
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.common_args = common_args

    @abstractmethod
    def train_set(self) -> Dataset:
        raise NotImplementedError("train_set() must be implemented by subclasses")

    @abstractmethod
    def test_set(self) -> Dataset:
        raise NotImplementedError("test_set() must be implemented by subclasses")

    def train_loader(self, batch_size: int = 64):
        return DataLoader(
            self.train_set(),
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **self.common_args,
        )

    def test_loader(self, batch_size: int = 256):
        return DataLoader(
            self.test_set(),
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **self.common_args,
        )
