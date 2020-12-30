#!/bin/sh
set -e

export KUNGFU_MINDSPORE_DEBUG=1

export GLOG_v=2 # WARNING

kungfu_examples/tests/run.sh
