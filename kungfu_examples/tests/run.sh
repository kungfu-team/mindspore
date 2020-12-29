#!/bin/sh
set -e

cd $(dirname $0)
ROOT=$PWD/../..

KUNGFU_LIB_PATH=$ROOT/third_party/kungfu/lib

export LD_LIBRARY_PATH=$KUNGFU_LIB_PATH:$ROOT/mindspore/lib:$ROOT/build/mindspore/_deps/ompi-src/ompi/.libs:$ROOT/build/mindspore/_deps/nccl-src/build/lib

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np 4
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

test_broadcast_op() {
    kungfu_run python3.7 test_broadcast_op.py --device CPU --dtype i32
    kungfu_run python3.7 test_broadcast_op.py --device CPU --dtype f32

    # FIXME:
    # mindspore/KungFu/srcs/cpp/src/nccl/gpu_collective.cpp::141: unhandled cuda error(1) in broadcast

    # kungfu_run python3.7 test_broadcast_op.py --device GPU --dtype i32
    # kungfu_run python3.7 test_broadcast_op.py --device GPU --dtype f32
}

test_allreduce_op() {
    kungfu_run python3.7 test_allreduce_op.py --device CPU --dtype i32
    kungfu_run python3.7 test_allreduce_op.py --device CPU --dtype f32

    # FIXME:
    # mindspore/KungFu/srcs/cpp/src/nccl/gpu_collective.cpp::163: unhandled cuda error(1) in all_reduce
    # kungfu_run python3.7 test_allreduce_op.py --device GPU --dtype i32
    # kungfu_run python3.7 test_allreduce_op.py --device GPU --dtype f32
}

test_broadcast_op
test_allreduce_op
