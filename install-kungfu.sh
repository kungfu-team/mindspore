#!/bin/sh
set -e

# PREFIX=$HOME/local/kungfu
PREFIX=$PWD/third_party/kungfu

if [ ! -d KungFu ]; then
    git clone https://github.com/lsds/KungFu.git
fi

cd KungFu
git checkout master
git pull

./configure --prefix=$PREFIX
make -j 8

if [ -d $PREFIX ]; then
    rm -fr $PREFIX
fi

make install
