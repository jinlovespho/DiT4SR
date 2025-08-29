#!/bin/bash

CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --config_file multi-gpu.yaml train/train_dit4sr.py \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --output_dir="./train_result/dit4sr_test" \
    --root_folders="preset/datasets/train_datasets/NKUSR8K/PAIRS" \
    --mixed_precision="fp16" \
    --learning_rate=5e-5 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --null_text_ratio=0.2 \
    --dataloader_num_workers=0 \
    --checkpointing_steps=10000 \
    --tracker_project_name 'iclr26_tair_vlm' \
    --tracker_run_name 'dit4sr_test' \
    --report_to wandb \
