#!/bin/sh
set -e

cd $(dirname $0)/..

. ./lg_scripts/measure.sh

CUDA_HOME=/usr/local/cuda
export CUDACXX=$CUDA_HOME/bin/nvcc

measure ./build.sh -e gpu
