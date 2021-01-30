#!/bin/sh
set -e

python3.7 -m pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.0.1/MindSpore/gpu/ubuntu_x86/cuda-10.1/mindspore_gpu-1.0.1-cp37-cp37m-linux_x86_64.whl \
    --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple \
    -U
