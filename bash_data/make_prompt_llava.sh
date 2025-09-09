
# llava - make prompts for training, the prompt is extracted from hq(gt) image
CUDA_VISIBLE_DEVICES=6 python utils_data/make_prompt.py \
    --img_dir /media/dataset2/text_restoration/100K/train  \
    --save_dir /media/dataset2/text_restoration/100K/dit4sr/data/train/llava13b_hq_prompt \
    --stop_num -1 \
    --start_num 0 \
    --captioner llava \
