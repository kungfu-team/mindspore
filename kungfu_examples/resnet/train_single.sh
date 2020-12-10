#!/bin/sh
set -e

cd $(dirname $0)
ROOT=$PWD/../..

KUNGFU_LIB_PATH=$ROOT/third_party/kungfu/lib

export LD_LIBRARY_PATH=$KUNGFU_LIB_PATH:$ROOT/mindspore/lib:$ROOT/build/mindspore/_deps/ompi-src/ompi/.libs

# export LD_LIBRARY_PATH=$PWD/third_party/kungfu/lib:$PWD/mindspore/lib:$PWD/build/mindspore/_deps/ompi-src/ompi/.libs:$PWD/build/mindspore/_deps/nccl-src/build/lib

# cd model_zoo/official/cv/resnet/scripts
# cd kungfu_examples/resnet
# ./run_standalone_train_gpu.sh resnet50 cifar10 $HOME/var/data/cifar/cifar-10-batches-bin

train() {
    rm -fr resnet-graph.meta
    /usr/bin/python3.7 train.py --net=$1 --dataset=$2 --device_target="GPU" --dataset_path=$3
}

train resnet50 cifar10 $HOME/var/data/cifar/cifar-10-batches-bin
