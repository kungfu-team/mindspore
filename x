#!/bin/sh
set -e

#export STDML_COLLECTIVE_ENABLE_LOG=1
#export KUNGFU_CONFIG_LOG_LEVEL=debug

cd $(dirname $0)
. ./lg_scripts/launcher.sh

# ./lg_scripts/build.sh

#python3.7 ./lg_scripts/main.py

# erun 1 python3.7 ./lg_scripts/main_elastic.py
erun 1 python3.7 ./lg_scripts/fake_model.py

#./lg_scripts/run_squad_origin.sh
#./lg_scripts/run_squad_debug.sh
#
#./lg_scripts/run_squad_elastic.sh
