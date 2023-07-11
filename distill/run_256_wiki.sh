#!/bin/bash
TEACHER='/disc1/models/bert-base-uncased'
OUTPUT_DIR='/disc1/models_output/pr_outputs/output_distill_256_wiki'
CACHE='/disc1/models_output/pr_outputs/output_distill_256_wiki/cache'
CONFIG='./distill_configs/h256_bert.json'
CUDA_VISIBLE_DEVICES='1' python distill.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --cache_dir $CACHE \
    --teacher_name_or_path $TEACHER \
    --student_config $CONFIG \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 512 \
    --train_batch_size 256 \
    --learning_rate 4e-4 \
    --ckpt_steps 20000 \
    --num_train_steps 100000

