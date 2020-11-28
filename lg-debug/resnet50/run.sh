#!/bin/sh
set -e

cd $(dirname $0)
pwd

mkdir -p bin
echo "$(which python3.7) \$@" >bin/python
chmod +x bin/python
export PATH=$PWD/bin:$PATH

ROOT=$PWD/../..
export PYTHONPATH=$ROOT
net=resnet50
dataset=cifar10
data_dir=$HOME/var/data/cifar/cifar-10-batches-bin

export GLOG_v=3 # ERROR
python3.7 train.py --net=$net --dataset=$dataset --dataset_path=$data_dir --device_target GPU
