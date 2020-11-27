#!/bin/sh
set -e

export LD_LIBRARY_PATH=$PWD/mindspore/lib:$PWD/build/mindspore/_deps/ompi-src/ompi/.libs

data_dir=$HOME/var/data/cifar/cifar-10-batches-bin

train() {
    net=resnet50
    dataset=cifar10

    python3.7 train.py --net=$net --dataset=$dataset --dataset_path=$data_dir
}

cd model_zoo/official/cv/resnet
# train

alias python=python3.7
cd scripts
./run_standalone_train_gpu.sh resnet50 cifar10 $data_dir
