#!/bin/sh
set -e

if [ -z "$@" ]; then
    git diff v1.3.0-baseline --name-only
else
    git diff v1.3.0-baseline $@
fi
