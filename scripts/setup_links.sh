#!/bin/bash

PROJECT=$(dirname $(dirname $(realpath $0)))

DATASETS=/storage/master112/nn6124030/datasets
WEIGHTS=/storage/master112/nn6124030/weights

ln -s $DATASETS $PROJECT/datasets
ln -s $WEIGHTS $PROJECT/weights
