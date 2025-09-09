

# 
CUDA_VISIBLE_DEVICES=0 python utils_data/make_embedding.py \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --root_path preset/datasets/train_datasets/NKUSR8K/PAIRS/ \
    --start_num 0 \
    --end_num -1


# tair
CUDA_VISIBLE_DEVICES=0 python utils_data/make_embedding.py \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --root_path preset/datasets/train_datasets/NKUSR8K/PAIRS/ \
    --start_num 0 \
    --end_num -1