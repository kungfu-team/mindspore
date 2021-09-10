#!/bin/sh
set -e

cd $(dirname $0)/..

. ./lg_scripts/measure.sh

CUDA_HOME=/usr/local/cuda
export CUDACXX=$CUDA_HOME/bin/nvcc

if [ `uname` = "Darwin" ]; then
    measure ./build.sh -e cpu
else
    measure ./build.sh -e gpu
fi
