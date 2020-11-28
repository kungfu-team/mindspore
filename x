#!/bin/sh
set -e

export LD_LIBRARY_PATH=$PWD/mindspore/lib:$PWD/build/mindspore/_deps/ompi-src/ompi/.libs

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
    echo --device $DEVICE
    echo --warmup-steps 2
    echo --steps 8
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
    # kungfu_run python3.7 ./benchmark_all_reduce.py $(app_flags)
    DEVICE=GPU
    # np=4
    for np in $(seq 4 4); do
        trace mpi_run python3.7 ./lg-debug/benchmark_all_reduce.py $(app_flags)
    done
}

# main
#
./lg-debug/resnet50/run.sh
