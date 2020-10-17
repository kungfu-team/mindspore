#!/bin/sh
set -e

# ./b
# ./U

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

trace() {
    echo "BEGIN $@"
    $@
    echo "END $@"
    echo
    echo
}

main() {
    trace run_single
    trace run_parallel
    trace run_elastic
}

main
