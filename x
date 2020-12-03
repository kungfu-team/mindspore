#!/bin/sh
set -e

# ./b
# ./U

# export KUNGFU_MINDSPORE_DEBUG=1
export LD_LIBRARY_PATH=$PWD/third_party/kungfu/lib

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    # echo -np 2
    echo -np 1

    echo -w
    echo -builtin-config-port 9100
    echo -config-server http://127.0.0.1:9100/config
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

run_single() {
    python3.7 ./kungfu.py
}

run_parallel() {
    kungfu_run python3.7 ./kungfu.py
}

run_elastic() {
    kungfu_run python3.7 ./kungfu_elastic_example.py \
        --max-step 100
}

run_train_mnist_slp() {
    python3.7 ./kungfu_examples/train_mnist_slp.py
}

run_train_mnist_lenet() {
    ./kungfu_examples/lenet/run.sh
}

run_train_mnist_lenet_elastic() {
    ./kungfu_examples/lenet-elastic/run.sh
}

trace() {
    echo "BEGIN $@"
    $@
    echo "END $@"
    echo
    echo
}

main() {
    # trace run_single
    # trace run_parallel
    # trace run_elastic
    # trace run_train_mnist_slp
    # trace run_train_mnist_lenet
    # trace run_train_mnist_lenet_elastic
    # trace ./kungfu_examples/gpu_examples/run.sh
    trace ./kungfu_examples/gpu_examples/run_elastic_nccl_all_reduce.sh
}

# export GLOG_v=2
# export GLOG_v=1
main
