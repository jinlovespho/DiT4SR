#!/bin/bash

CUDA="0"
NUM_GPU=1

CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch --num_processes ${NUM_GPU} test/test.py \
    --config run_configs/val/val_satext_lv2_dit4sr_baseline.yaml


