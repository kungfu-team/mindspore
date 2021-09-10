#!/bin/sh
set -e

REPEAT=1
# REPEAT=100

list_tf_records() {
    for i in $(seq $REPEAT); do
        echo /data/squad1/train.tf_record
    done
}

export STD_TRACER_PATIENT=1

./build/mindspore/ms-elastic-build-tf-index $(list_tf_records)

index_file=tf-index-${REPEAT}.idx.txt

PYTHON=$(which python3.7)
echo "Using $PYTHON"

reload=1

runner_flags() {
    echo -logdir logs
    echo -q

    echo -w

    if [ "$reload" -eq 1 ]; then
        echo -elastic-mode reload
    fi
    echo -builtin-config-port 9100
    echo -config-server http://127.0.0.1:9100/config
}

elastic_run_n() {
    local init_np=$1
    shift
    $PYTHON -m kungfu.cmd $(runner_flags) -np $init_np $@
}

global_batch_size=32
elastic_run_n 4 ./build/mindspore/ms-elastic-create-tf-records $index_file $global_batch_size

echo "built tf_records"

echo "original"
./build/mindspore/ms-read-tf-records $(list_tf_records)

echo "sharded"
./build/mindspore/ms-read-tf-records *.tf_record
