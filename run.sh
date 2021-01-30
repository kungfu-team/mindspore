#!/bin/sh
set -e

# export LD_LIBRARY_PATH=$PWD/mindspore/lib:$PWD/build/mindspore/_deps/ompi-src/ompi/.libs:$PWD/build/mindspore/_deps/nccl-src/build/lib

mkdir -p bin
echo "$(which python3.7) \$@" >bin/python
chmod +x bin/python
export PATH=$PWD/bin:$PATH

cd model_zoo/official/cv/resnet/scripts
./run_standalone_train_gpu.sh resnet50 cifar10 $HOME/var/data/cifar/cifar-10-batches-bin
