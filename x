#!/bin/sh
set -e

# ./b
# ./U

# export KUNGFU_MINDSPORE_DEBUG=1
export LD_LIBRARY_PATH=$PWD/third_party/kungfu/lib

trace() {
    echo "BEGIN $@"
    $@
    echo "END $@"
    echo
    echo
}

main() {
    # trace ./kungfu_examples/lenet-elastic/run-elastic.sh
    # trace ./kungfu_examples/gpu_examples/run.sh
    # trace ./kungfu_examples/gpu_examples/run_elastic_nccl_all_reduce.sh
    # trace ./kungfu_examples/resnet-elastic/train_single.sh
    # trace ./kungfu_examples/resnet-elastic/train_parallel_mpi.sh
    # trace ./kungfu_examples/resnet-elastic/train_parallel_kungfu.sh
    trace ./kungfu_examples/resnet-elastic/train_parallel_kungfu_elastic.sh
}

export GLOG_v=3 # ERROR
# export GLOG_v=2 # WARNING
main
