# CUDA_VISIBLE_DEVICES=0 python utils_data/make_paired_data.py \
#     --gt_path 'preset/datasets/train_datasets/NKUSR8K/HR' \
#     --save_dir 'preset/datasets/train_datasets/NKUSR8K/LR' \
#     --epoch 1


# tair make LR from HR 
CUDA_VISIBLE_DEVICES=1 python utils_data/make_paired_data.py \
    --gt_path /media/dataset2/text_restoration/100K/test \
    --save_dir /media/dataset2/text_restoration/100K/dit4sr/data/train/LR \
    --epoch 5