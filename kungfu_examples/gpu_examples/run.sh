#!/bin/sh
set -e

cd $(dirname $0)
ROOT=$PWD/../..
KUNGFU_LIB_PATH=$ROOT/third_party/kungfu/lib

export LD_LIBRARY_PATH=$KUNGFU_LIB_PATH

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np 4
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

app_flags() {
    # echo --device CPU
    echo --device GPU
}

main() {
    # kungfu_run python3.7 ./hello_world.py $(app_flags)
    kungfu_run python3.7 ./benchmark_all_reduce.py $(app_flags)
}

main
