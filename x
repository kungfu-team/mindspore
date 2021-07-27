#!/bin/sh
set -e

cd $(dirname $0)
. ./lg_scripts/launcher.sh

# ./lg_scripts/build.sh

# python3.7 ./lg_scripts/main.py

# erun 1 python3.7 ./lg_scripts/main_elastic.py

# ./lg_scripts/run_squad_origin.sh
./lg_scripts/run_squad_elastic.sh
