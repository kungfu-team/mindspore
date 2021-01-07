#!/bin/sh
set -e

cd $(dirname $0)
ROOT=$PWD/../..

KUNGFU_LIB_PATH=$ROOT/third_party/kungfu/lib

export LD_LIBRARY_PATH=$KUNGFU_LIB_PATH:$ROOT/mindspore/lib:$ROOT/build/mindspore/_deps/ompi-src/ompi/.libs:$ROOT/build/mindspore/_deps/nccl-src/build/lib

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np 1

    local enable_elastic=1
    if [ $enable_elastic -eq 1 ]; then
        echo -w
        echo -builtin-config-port 9100
        echo -config-server http://127.0.0.1:9100/config
    fi
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

all_flags() {
    echo --net=resnet50
    echo --dataset=cifar10
    echo --dataset_path=$HOME/var/data/cifar/cifar-10-batches-bin
    echo --device_num=4
    echo --device_target="GPU"
    echo --run_kungfu=True
    echo --elastic=True
}

train() {
    rm -fr logs
    rm -fr resnet-graph.meta
    rm -fr ckpt_*
    rm -fr cuda_meta_*
    kungfu_run \
        /usr/bin/python3.7 train.py $(all_flags)
}

train
