#!/bin/sh
set -e

cd $(dirname $0)/..
ROOT=$PWD

cd model_zoo/official/nlp/bert
# cd scripts
pwd

export CUDA_VISIBLE_DEVICES=0

mkdir -p ms_log
CUR_DIR=$(pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log

# KUNGFU_MINDSPORE=$HOME/code/repos/github.com/kungfu-ml/kungfu-mindspore
# . $KUNGFU_MINDSPORE/ld_library_path.sh
# export LD_LIBRARY_PATH=$(ld_library_path $KUNGFU_MINDSPORE/mindspore)

SCHEMA_DIR="/data/squad1/squad_schema.json"

python3.7 run_squad.py \
    --device_target="GPU" \
    --do_train="true" \
    --do_eval="false" \
    --device_id=0 \
    --epoch_num=1 \
    --num_class=2 \
    --train_data_shuffle="false" \
    --eval_data_shuffle="false" \
    --train_batch_size=16 \
    --eval_batch_size=1 \
    --vocab_file_path="/data/bert/vocab.txt" \
    --save_finetune_checkpoint_path="$ROOT/checkpoint" \
    --load_pretrain_checkpoint_path="/data/bert/bert_base.ckpt" \
    --train_data_file_path="/data/squad1/train.tf_record" \
    --eval_json_path="/data/squad1/dev-v1.1.json" \
    --schema_file_path=${SCHEMA_DIR} # >$ROOT/squad.log 2>&1
