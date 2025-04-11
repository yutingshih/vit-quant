import pytest
import timm

from src.dataset.imagenet import ImageNetLoaderGenerator as ImageNet


@pytest.mark.slow
def test_imagenet_train_loader():
    NUM_IMAGES = 1_281_167  # Total number of training samples in ImageNet
    BATCH_SIZE = 256

    # Use torchvision-style data transform
    imagenet = ImageNet(root="./datasets/imagenet/image_dir")

    loader = imagenet.train_loader(batch_size=BATCH_SIZE)
    assert len(loader) == (NUM_IMAGES - 1) // BATCH_SIZE + 1

    images, labels = next(iter(loader))
    assert images.shape == (BATCH_SIZE, 3, 224, 224)
    assert labels.shape == (BATCH_SIZE,)


@pytest.mark.slow
def test_imagenet_test_loader():
    NUM_IMAGES = 50_000  # Total number of validation samples in ImageNet
    BATCH_SIZE = 1000

    # Use timm-style data transform
    imagenet = ImageNet(
        root="./datasets/imagenet/image_dir",
        model=timm.create_model("vit_tiny_patch16_224", pretrained=True),
    )

    loader = imagenet.test_loader(batch_size=BATCH_SIZE)
    assert len(loader) == (NUM_IMAGES - 1) // BATCH_SIZE + 1

    images, labels = next(iter(loader))
    assert images.shape == (BATCH_SIZE, 3, 224, 224)
    assert labels.shape == (BATCH_SIZE,)
