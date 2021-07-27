#!/bin/sh
set -e

cd $(dirname $0)/..
ROOT=$PWD

cd model_zoo/official/nlp/bert
# cd scripts
pwd

export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p ms_log
CUR_DIR=$(pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log

# KUNGFU_MINDSPORE=$HOME/code/repos/github.com/kungfu-ml/kungfu-mindspore
# . $KUNGFU_MINDSPORE/ld_library_path.sh
# export LD_LIBRARY_PATH=$(ld_library_path $KUNGFU_MINDSPORE/mindspore)

init_cluster_size=1
EPOCH_SIZE=1
DATA_DIR="/data/squad1/train.tf_record"
SCHEMA_DIR="/data/squad1/squad_schema.json"

kungfu_run_flags() {
    echo -np $init_cluster_size
    echo -logfile kungfu-run.log
    echo -logdir ./log
    echo -port-range 40000-41000

    echo -w
    local config_port=9999
    echo -builtin-config-port $config_port
    echo -config-server http://127.0.0.1:$config_port/config
}

#DO_EVAL=true
DO_EVAL=false

kungfu-run $(kungfu_run_flags) \
    python3.7 run_squad_elastic.py \
    --device_target="GPU" \
    --distribute="true" \
    --do_train="true" \
    --do_eval="$DO_EVAL" \
    --device_id=0 \
    --epoch_num=${EPOCH_SIZE} \
    --num_class=2 \
    --train_data_shuffle="false" \
    --eval_data_shuffle="false" \
    --train_batch_size=8 \
    --eval_batch_size=1 \
    --vocab_file_path="/data/bert/vocab.txt" \
    --save_finetune_checkpoint_path="$ROOT/checkpoint" \
    --load_pretrain_checkpoint_path="/data/bert/bert_base.ckpt" \
    --train_data_file_path=${DATA_DIR} \
    --eval_json_path="/data/squad1/dev-v1.1.json" \
    --schema_file_path=${SCHEMA_DIR} >$ROOT/squad_elastic.log 2>&1

echo "$0 done"
