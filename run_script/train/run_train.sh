#!/bin/bash

CUDA="1"
NUM_GPU=1

INIT_ARGS="
    --seed 42 \
    --pretrained_model_name_or_path preset/models/stable-diffusion-3.5-medium \
    --transformer_model_name_or_path preset/models/dit4sr_q \
    --vae_model_name_or_path preset/models/stable-diffusion-3.5-medium \
"

DATA_ARGS="
    --data_name satext \
    --data_path /media/dataset2/text_restoration/100K \
    --hq_prompt_path /media/dataset2/text_restoration/100K/dit4sr/data/train/llava13b_hq_prompt \
    --lq_prompt_path /media/dataset1/jinlovespho/iclr26/DiT4SR/results/satext/lv3/dit4sr_q_llavaprompt/txt \
    --num_workers 0 \
    --root_folders preset/datasets/train_datasets/NKUSR8K/PAIRS \
"


MODEL_ARGS="
    --resume_from_checkpoint latest \
    --resolution 512 \
    --null_text_ratio 0.0 \
    --load_precomputed_caption \
"


TRAIN_ARGS="
    --finetune dit4sr_lr_branch \
    --train_batch_size 2 \
    --num_train_epochs 10 \
"

VAL_ARGS="
    --val_every_step 100
"


OPTIM_ARGS="
    --mixed_precision no \
    --lr_scheduler constant \
    --lr_warmup_steps 500 \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
"


SAVE_ARGS="
    --output_dir ./train_result/dit4sr \
    --checkpointing_steps 100 \
"


LOG_ARGS="
    --report_to wandb \
    --tracker_project_name iclr26_tair_vlm \
    --tracker_run_name TRAIN_server12_gpu${CUDA}_satext_dit4sr_lr1e-5_bs2_gradaccum4_llava13bprompt \
    --logging_dir logs \
"



CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch --num_processes ${NUM_GPU} train/train_dit4sr.py ${INIT_ARGS} ${DATA_ARGS} ${MODEL_ARGS} ${TRAIN_ARGS} ${VAL_ARGS} ${OPTIM_ARGS} ${SAVE_ARGS} ${LOG_ARGS}
    
