import pytest

from src.dataset.imagenet import ImageNet


@pytest.fixture
def imagenet_instance():
    return ImageNet(root="./datasets/imagenet/image_dir", use_gpu=True)


@pytest.mark.slow
def test_imagenet_train_loader(imagenet_instance: ImageNet):
    NUM_IMAGES = 1_281_167  # Total number of training samples in ImageNet
    BATCH_SIZE = 256

    try:
        loader = imagenet_instance.get_train_loader(batch_size=BATCH_SIZE)
    except Exception:
        pytest.fail("get_train_loader() raised an exception unexpectedly!")

    assert len(loader) == (NUM_IMAGES - 1) // BATCH_SIZE + 1

    images, labels = next(iter(loader))
    assert images.shape == (BATCH_SIZE, 3, 224, 224)
    assert labels.shape == (BATCH_SIZE,)


@pytest.mark.slow
def test_imagenet_val_loader(imagenet_instance: ImageNet):
    NUM_IMAGES = 50_000  # Total number of validation samples in ImageNet
    BATCH_SIZE = 1000

    try:
        loader = imagenet_instance.get_val_loader(batch_size=BATCH_SIZE)
    except Exception:
        pytest.fail("get_val_loader() raised an exception unexpectedly!")


    assert len(loader) == (NUM_IMAGES - 1) // BATCH_SIZE + 1

    images, labels = next(iter(loader))
    assert images.shape == (BATCH_SIZE, 3, 224, 224)
    assert labels.shape == (BATCH_SIZE,)
