PROJ_DIR=$(realpath $(dirname $0)/..)
WORK_DIR=/root/vit-quant
IMAGE=vit-quant:$USER
NAME=vit-quant

build() {
    docker build -t $IMAGE $PROJ_DIR
}

run() {
    docker run \
        -it \
        --rm \
        --gpus all \
        --name $NAME \
        --hostname docker \
        --workdir $WORK_DIR \
        --mount type=bind,src=$PROJ_DIR,dst=$WORK_DIR \
        $IMAGE
}

attach() {
    docker attach $NAME
}

if [[ $# -ne 1 ]]; then
    echo "Usage $0 [build | run | attach]"
else
    $@
fi
