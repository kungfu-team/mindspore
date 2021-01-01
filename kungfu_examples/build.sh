#!/bin/sh
set -e

cd $(dirname $0)/..

datatime() {
    date '+%Y-%m-%d %H:%M:%S'
}

notify() {
    if [ -f ~/.slack/notify ]; then
        ~/.slack/notify "$@"
    fi
}

CUDA_HOME=/usr/local/cuda
export CUDACXX=$CUDA_HOME/bin/nvcc

build_flags() {
    # echo -e cpu
    echo -e gpu
}

main() {
    env \
        OPENSSL_ROOT_DIR=$HOME/local/openssl \
        ENABLE_KUNGFU=ON \
        ./build.sh $(build_flags)
}

notify "start building mindspore at $(datatime)"
main
notify "finish building mindspore at $(datatime)"
