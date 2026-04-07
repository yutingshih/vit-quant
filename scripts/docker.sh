PROJ_DIR=$(realpath $(dirname $0)/..)
WORK_DIR=/root/vit-quant
IMAGE=vit-quant
NAME=vit-quant

build() {
    docker build -t $IMAGE $PROJ_DIR
}

run() {
    local status=$(docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null)

    case "$status" in
        "")
            docker run \
                -it \
                --gpus all \
                --name $NAME \
                --hostname docker \
                --workdir $WORK_DIR \
                --mount type=bind,src=$PROJ_DIR,dst=$WORK_DIR \
                $IMAGE
            ;;
        "running")
            docker attach $NAME
            ;;
        "created" | "exited" | "pause")
            docker start -ai $NAME
            ;;
        *)
            echo "Invalid status: $status"
            ;;
    esac
}

if [[ $# -ne 1 ]]; then
    echo "Usage $0 [build | run]"
else
    $@
fi
