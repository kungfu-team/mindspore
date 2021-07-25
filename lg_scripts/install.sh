#!/bin/sh
set -e

# pip3 config set global.extra-index-url ''
python3.7 -m pip install -U ./output/mindspore_gpu-1.3.0-cp37-cp37m-linux_x86_64.whl --force-reinstall
# python3.9 -m pip install -U ./output/mindspore_gpu-1.3.0-cp39-cp39-linux_x86_64.whl
