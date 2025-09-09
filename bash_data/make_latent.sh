# python utils_data/make_latents.py \
#     --root_path preset/datasets/train_datasets/NKUSR8K/PAIRS/gt \
#     --save_path preset/datasets/train_datasets/NKUSR8K/PAIRS/latent_hr \
#     --start_num 0 \
#     --end_num -1


# tair training data
CUDA_VISIBLE_DEVICES=0 python utils_data/make_latents.py \
    --root_path preset/datasets/train_datasets/NKUSR8K/PAIRS/lr \
    --save_path /media/dataset2/text_restoration/100K/dit4sr/data/train/latent_lr \
    --start_num 0 \
    --end_num -1