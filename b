#!/bin/sh
set -e

datatime() {
    date '+%Y-%m-%d %H:%M:%S'
}

notify() {
    if [ -f ~/.slack/notify ]; then
        ~/.slack/notify "$@"
    fi
}

main() {
    env \
        OPENSSL_ROOT_DIR=$HOME/local/openssl \
        ./build.sh -e cpu
}

notify "start building mindspore at $(datatime)"
main
notify "finish building mindspore at $(datatime)"
