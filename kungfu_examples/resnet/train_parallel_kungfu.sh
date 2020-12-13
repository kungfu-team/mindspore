#!/bin/sh
set -e

cd $(dirname $0)
ROOT=$PWD/../..

KUNGFU_LIB_PATH=$ROOT/third_party/kungfu/lib

export LD_LIBRARY_PATH=$KUNGFU_LIB_PATH:$ROOT/mindspore/lib:$ROOT/build/mindspore/_deps/ompi-src/ompi/.libs

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np 4
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
    # echo --run_distribute=True
    echo --run_kungfu=True
}

train() {
    rm -fr resnet-graph.meta
    rm -fr ckpt_*
    rm -fr cuda_meta_*
    kungfu_run \
        /usr/bin/python3.7 train.py $(all_flags)
}

train
