#!/bin/sh
set -e

cd $(dirname $0)
pwd

net=resnet50
dataset=cifar10
data_dir=$HOME/var/data/cifar/cifar-10-batches-bin
python3.7 train.py --net=$net --dataset=$dataset --dataset_path=$data_dir --device_target GPU
