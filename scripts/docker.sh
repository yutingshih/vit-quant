PROJ_DIR=$(realpath $(dirname $0)/..)
WORK_DIR=/root/vit-quant
IMAGE=vit-quant:$USER

build() {
    docker build -t $IMAGE $PROJ_DIR
}

run() {
    docker run \
        -it \
        --rm \
        --gpus all \
        --name vit-quant \
        --hostname docker \
        --workdir ${WORK_DIR} \
        --mount type=bind,src=${PROJ_DIR},dst=${WORK_DIR} \
        $IMAGE
}

if [[ $# -lt 1 ]]; then
    echo "Usage $0 [build | run]"
else
    $@
fi
