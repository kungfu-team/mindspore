#!/bin/sh
set -e

cd $(dirname $0)
ROOT=$PWD/../..

KUNGFU_LIB_PATH=$ROOT/third_party/kungfu/lib

export LD_LIBRARY_PATH=$KUNGFU_LIB_PATH:$ROOT/mindspore/lib:$ROOT/build/mindspore/_deps/ompi-src/ompi/.libs

mpi_flags() {
    echo --allow-run-as-root
    echo -np 4
    echo --output-filename log_output
    echo --merge-stderr-to-stdout
}

train() {
    rm -fr resnet-graph.meta

    mpirun $(mpi_flags) \
        /usr/bin/python3.7 train.py --net=$1 --dataset=$2 --run_distribute=True \
        --device_num=4 --device_target="GPU" --dataset_path=$3
}

train resnet50 cifar10 $HOME/var/data/cifar/cifar-10-batches-bin
