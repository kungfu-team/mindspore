#!/bin/sh
set -e

PYTHON=`which python3.7`

with_pwd() {
    local CD=$PWD
    $@
    cd $CD
}

cd `dirname $0`

get_kungfu() {
    if [ ! -d KungFu ]; then
        git clone https://github.com/lsds/KungFu
    fi
    cd KungFu
    git checkout ms-support
}

install_kungfu() {
    echo "Using $PYTHON"
    cd KungFu
    $PYTHON -m pip install --no-index -U .
}

with_pwd get_kungfu
with_pwd install_kungfu
# $PYTHON -m pip list
