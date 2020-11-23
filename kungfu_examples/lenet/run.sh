#!/bin/sh
set -e

cd $(dirname $0)

KUNGFU_LIB_PATH=$HOME/code/repos/github.com/lgarithm/mindspore/third_party/kungfu/lib
export LD_LIBRARY_PATH=$KUNGFU_LIB_PATH

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np 4
}

kungfu_run() {
    # export KUNGFU_MINDSPORE_DEBUG=1
    # KUNGFU_MINDSPORE_DEBUG=$KUNGFU_MINDSPORE_DEBUG \
    env \
        LD_LIBRARY_PATH=$KUNGFU_LIB_PATH \
        kungfu-run $(kungfu_run_flags) $@
}

train_flags() {
    local data_dir=$HOME/var/data/mindspore/mnist
    echo --data-dir $data_dir

    # echo --device CPU
    echo --device GPU

    # echo --epoch-size 1
    # echo --repeat-size 1
}

single_train() {
    rm -f *.meta
    python3.7 train.py $(train_flags)
}

kungfu_train() {
    rm -f *.meta
    kungfu_run python3.7 train.py $(train_flags) --use-kungfu
}

main() {
    if [ $(hostname) = "platypus2" ]; then
        kungfu_train
    else
        single_train
    fi
}

main
