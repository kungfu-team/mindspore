#!/bin/sh
set -e

cd $(dirname $0)
ROOT=$PWD/../..

KUNGFU_LIB_PATH=$ROOT/third_party/kungfu/lib
# CUDA_HOME=/usr/local/cuda

# export LD_LIBRARY_PATH=$KUNGFU_LIB_PATH
export LD_LIBRARY_PATH=$KUNGFU_LIB_PATH:$ROOT/mindspore/lib:$ROOT/build/mindspore/_deps/ompi-src/ompi/.libs
# export LD_LIBRARY_PATH=$PWD/mindspore/lib:$ROOT/build/mindspore/_deps/ompi-src/ompi/.libs

# export PATH=$PATH:$CUDA_HOME/bin

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np $np
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

mpi_run_flags() {
    echo --allow-run-as-root
    echo -np $np
    echo -x LD_LIBRARY_PATH
}

mpi_run() {
    mpirun $(mpi_run_flags) $@
}

app_flags() {
    # echo --device CPU
    echo --device GPU

    echo --warmup-steps 2
    echo --steps 100

    # echo --model vgg16
    echo --model resnet50
    # echo --collective mindspore
    # echo --collective kungfu
}

trace() {
    echo "BEGIN $@"
    $@
    echo "END $@"
    echo
    echo
}

main() {
    # kungfu_run python3.7 ./hello_world.py $(app_flags)
    # for np in $(seq  4); do
    np=4
    trace kungfu_run python3.7 ./benchmark_all_reduce.py $(app_flags)
    # mpi_run python3.7 ./benchmark_all_reduce.py $(app_flags)
}

main
