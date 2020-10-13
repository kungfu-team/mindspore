#!/bin/sh
set -e

datatime() {
    date '+%Y-%M-%d %H:%I:%S'
}

notify() {
    if [ -f ~/.slack/notify ]; then
        ~/.slack/notify "$@"
    fi
}

main() {
    ./build.sh -e cpu
}

notify "start building mindspore at $(datatime)"
main
notify "finish building mindspore at $(datatime)"
