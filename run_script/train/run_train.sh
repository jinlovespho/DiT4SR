#!/bin/bash

CUDA="0"
NUM_GPU=1

INIT_ARGS="
    --seed 42 \
    --pretrained_model_name_or_path preset/models/stable-diffusion-3.5-medium \
    --transformer_model_name_or_path preset/models/stable-diffusion-3.5-medium \
    --vae_model_name_or_path preset/models/stable-diffusion-3.5-medium \
"

DATA_ARGS="
    --data_name satext \
    --data_path /media/dataset2/text_restoration/100K \
    --num_workers 0 \
    --root_folders preset/datasets/train_datasets/NKUSR8K/PAIRS \
"


MODEL_ARGS="
    --resolution 512 \
    --null_text_ratio 0.2 \
"

TRAIN_ARGS="
    --finetune dit4sr_lr_branch \
    --train_batch_size 4 \
"

VAL_ARGS="
    --val_batch_size 4 \
"

OPTIM_ARGS="
    --mixed_precision no \
    --lr_scheduler constant \
    --lr_warmup_steps 500 \
    --learning_rate 5e-5 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
"





SAVE_ARGS="
    --output_dir ./train_result/dit4sr_test \
    --checkpointing_steps 10000 \
"



LOG_ARGS="
    --report_to wandb \
    --tracker_project_name iclr26_tair_vlm \
    --tracker_run_name dit4sr_test \
    --logging_dir logs \

"


CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch --num_processes ${NUM_GPU} train/train_dit4sr.py ${INIT_ARGS} ${DATA_ARGS} ${MODEL_ARGS} ${TRAIN_ARGS} ${VAL_ARGS} ${OPTIM_ARGS} ${SAVE_ARGS} ${LOG_ARGS}
    

