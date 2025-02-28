#!/bin/bash

PY_VER=3.11
ENV_NAME=${1:-vit-$PY_VER}
ROOT_DIR=$(realpath $(dirname $0)/..)

if ! conda env list | grep -q $ENV_NAME; then
    conda create -n $ENV_NAME python=$PY_VER -y
fi

conda activate $ENV_NAME
