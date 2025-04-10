# vit-quant

## Environment Setup

Clone the repository.

```shell
git clone git@github.com:yutingshih/vit-quant.git
```

Activate conda environment and install dependencies.

```shell
source ./scripts/conda.sh
pip install -r requirements.txt
```

Create sybolic links to datasets and model weights.

```shell
./scripts/setup_links.sh
```

## Dataset Preparation

### ImageNet-1K

Download the dataset from [ImageNet](http://www.image-net.org/) and place it in the `datasets/imagenet` directory. The directory structure should look like this:

```
datasets/imagenet/image_dir
├── ILSVRC2012_img_train
└── val
```

Run smoke test to verify the dataset. (about 30 seconds)

```shell
pytest tests/dataset/imagenet.py -m slow
```
