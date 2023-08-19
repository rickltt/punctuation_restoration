#!/bin/bash
DATA_DIR='./data/cmrpt'
MODEL_NAME_OR_PATH='/disc1/models/rbt4'
OUTPUT_DIR='/disc1/models_output/pr_outputs/output_rbt4'
CACHE='/disc1/models_output/pr_outputs/output_rbt4/cache'
for seed in {1..10}
do
    CUDA_VISIBLE_DEVICES='1' python token_classification.py \
    --data_dir $DATA_DIR \
    --seed $seed \
    --cache_dir $CACHE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 256 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --learning_rate 3e-5 \
    --num_train_epochs 10 
done