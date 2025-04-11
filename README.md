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
pytest tests/dataset -m slow
```

## Getting Started

### Main Script

Under the project root, run the main script with the following command to see the help message:

```
PYTHONPATH=. python3 src/main.py --help
```

To run an evaluation on ImageNet-1K with a pretrained model, use the following command:

```shell
PYTHONPATH=. python3 src/main.py \
    --model vit_small_patch16_224 \
    --dataset datasets/imagenet/image_dir \
    --batch-size 256 \
    --device cuda:0
```
