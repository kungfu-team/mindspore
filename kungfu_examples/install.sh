#!/bin/sh
set -e

cd $(dirname $0)/..

reinstall() {
    python3.7 -m pip install -U output/mindspore_gpu-1.1.0-cp37-cp37m-linux_x86_64.whl
}

reinstall
