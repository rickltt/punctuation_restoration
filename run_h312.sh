#!/bin/bash
DATA_DIR='./data/cmrpt'
MODEL_NAME_OR_PATH='./distill/bert/h312'
OUTPUT_DIR='/disc1/models_output/pr_outputs/output_h312'
CACHE='/disc1/models_output/pr_outputs/output_h312/cache'
for seed in {1..10}
do
    CUDA_VISIBLE_DEVICES='2' python classifier_run.py \
    --data_dir $DATA_DIR \
    --seed $seed \
    --cache_dir $CACHE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 256 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-5 \
    --weight_decay 1e-5 \
    --num_train_epochs 20 
done