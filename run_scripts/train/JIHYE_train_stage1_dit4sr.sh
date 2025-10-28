#!/bin/bash

CUDA="3"
NUM_GPU=1

CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch  --num_processes ${NUM_GPU} train/train_dit4sr.py \
                                                --config run_configs/train/JIHYE_train_stage1_dit4sr.yaml

