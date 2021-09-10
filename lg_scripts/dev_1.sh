#!/bin/sh
set -e

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

app() {
    echo $PYTHON ./lg_scripts/elastic_dateset_dev_1.py
    if [ "$reload" -eq 1 ]; then
        echo --reload
    fi
    echo --global-batch-size 24

    echo --max-progress 88641
    # echo --max-progress $((24 * 10))

    echo --run
}

rm -fr progress-*
elastic_run_n 1 $(app)
