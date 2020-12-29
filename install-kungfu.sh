#!/bin/sh
set -e

cd $(dirname $0)
PREFIX=$PWD/third_party/kungfu

if [ ! -d KungFu ]; then
    git clone https://github.com/lsds/KungFu.git
fi

cd KungFu
git checkout master
git pull

config_flags() {
    echo --prefix=$PREFIX
    echo --enable-nccl
    echo --with-nccl=$PWD/build/mindspore/_deps/nccl-src/build
}

./configure $(config_flags)
make -j 8

if [ -d $PREFIX ]; then
    rm -fr $PREFIX
fi

make install
