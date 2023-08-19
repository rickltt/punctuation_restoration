#!/bin/bash
TRAIN='./train.txt'
TEACHER='/disc1/models/chinese-roberta-wwm-ext'
OUTPUT_DIR='/disc1/models_output/pr_outputs/output_distill_312'
CACHE='/disc1/models_output/pr_outputs/output_distill_312/cache'
CONFIG='./distill_configs/h312.json'
CUDA_VISIBLE_DEVICES='2' python distill.py \
    --train_file $TRAIN \
    --cache_dir $CACHE \
    --teacher_name_or_path $TEACHER \
    --student_config $CONFIG \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 512 \
    --train_batch_size 256 \
    --learning_rate 4e-4 \
    --ckpt_steps 20000 \
    --num_train_steps 100000

