#!/usr/bin/env python3

import argparse
import os
from pprint import pprint

from datasets import load_dataset, load_dataset_builder


parser = argparse.ArgumentParser(
    description="Download ImageNet dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("path", type=str, default="datasets/imagenet", help="path to save the dataset")
args = parser.parse_args()

os.environ.setdefault("IMAGENET_USERNAME", input("Enter your ImageNet username: "))
os.environ.setdefault("IMAGENET_PASSWORD", input("Enter your ImageNet password: "))
os.environ.setdefault("HUGGINGFACE_TOKEN", input("Enter your HuggingFace access token: "))

ds_builder = load_dataset_builder("imagenet-1k")
info = ds_builder.info

print(f'{info.dataset_name = }')
print(f'{info.version = }')
print(f'{info.homepage = }')
print(f'{info.dataset_size = }')
pprint(info.splits)

print(f"Downloading ImageNet dataset to {args.path}...")
res = input("Continue? y/[n]: ")

if res.lower() == "y":
    imagenet = load_dataset("imagenet-1k", split="validation", cache_dir=args.path)
    print(imagenet)
else:
    print("Aborted")
