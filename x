#!/bin/sh
set -e

# ./b
# ./U

export LD_LIBRARY_PATH=$PWD/third_party/kungfu/lib

kungfu_run_flags() {
    echo -q
    echo -np 4
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

main() {
    run_single
    run_parallel
}

main
