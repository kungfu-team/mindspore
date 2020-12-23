#!/bin/sh
set -e

cd $(dirname $0)

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np 1

    local enable_elastic=1
    if [ $enable_elastic -eq 1 ]; then
        echo -w
        echo -builtin-config-port 9100
        echo -config-server http://127.0.0.1:9100/config
    fi
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

train_flags() {
    local data_dir=$HOME/var/data/mindspore/mnist

    echo --device CPU
    # echo --device GPU
    echo --data-dir $data_dir
    echo --batch-size 200
    # echo --epoch-size 1
    # echo --repeat-size 1
    # echo --run-test
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
    rm -f *.meta
    kungfu_run python3.7 train.py $(train_flags) --use-kungfu --use-kungfu-elastic
}

main
