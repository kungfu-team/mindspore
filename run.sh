#!/bin/sh
set -e

export LD_LIBRARY_PATH=$PWD/third_party/kungfu/lib:$PWD/mindspore/lib:$PWD/build/mindspore/_deps/ompi-src/ompi/.libs:$PWD/build/mindspore/_deps/nccl-src/build/lib

# cd model_zoo/official/cv/resnet/scripts
cd kungfu_examples/resnet
# ./run_standalone_train_gpu.sh resnet50 cifar10 $HOME/var/data/cifar/cifar-10-batches-bin

train() {
    /usr/bin/python3.7 train.py --net=$1 --dataset=$2 --device_target="GPU" --dataset_path=$3
}

train resnet50 cifar10 $HOME/var/data/cifar/cifar-10-batches-bin
