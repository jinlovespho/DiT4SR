
# # satext lv3 - use gt prompt 
# CUDA_VISIBLE_DEVICES=1 python test/test_wllava.py \
#     --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
#     --transformer_model_name_or_path="preset/models/dit4sr_q" \
#     --image_path /media/dataset2/text_restoration/SAMText_test_degradation/lv3 \
#     --satext_ann_path /media/dataset2/text_restoration/100K/test/dataset.json \
#     --output_dir results/satext/lv3/dit4sr_q_wllava_gtprompt_lremb \
#     --use_satext_gt_prompt \
#     --save_prompts \
#     --start_point lr


# # satext lv3 - null prompt 
# CUDA_VISIBLE_DEVICES=0 python test/test_wllava.py \
#     --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
#     --transformer_model_name_or_path="preset/models/dit4sr_q" \
#     --image_path /media/dataset2/text_restoration/SAMText_test_degradation/lv3 \
#     --satext_ann_path /media/dataset2/text_restoration/100K/test/dataset.json \
#     --output_dir results/satext/lv3/dit4sr_q_wllava_nullprompt_lremb \
#     --use_null_prompt \
#     --save_prompts \
#     --start_point lr


# # satext lv3 - llava prompt 
# CUDA_VISIBLE_DEVICES=0 python test/test_wllava.py \
#     --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
#     --transformer_model_name_or_path="preset/models/dit4sr_q" \
#     --image_path /media/dataset2/text_restoration/SAMText_test_degradation/lv3 \
#     --satext_ann_path /media/dataset2/text_restoration/100K/test/dataset.json \
#     --output_dir results/satext/lv3/dit4sr_q_wllava_llavaprompt_lremb \
#     --save_prompts \
#     --start_point lr 


# 72 32 7 3
for size in 7; do
    # satext lv3 - qwen prompt 
    CUDA_VISIBLE_DEVICES=7 python test/test_wllava.py \
        --pretrained_model_name_or_path preset/models/stable-diffusion-3.5-medium \
        --transformer_model_name_or_path train_result/dit4sr/TRAIN_server12_gpu1_satext_dit4sr_lr5e-6_bs2_gradaccum4_qwen7bprompt/checkpoint-26000 \
        --image_path /media/dataset2/text_restoration/SAMText_test_degradation/lv3 \
        --satext_ann_path /media/dataset2/text_restoration/100K/test/dataset.json \
        --output_dir results/satext/lv3/dit4sr_q_qwen${size}prompt_ckpt26000 \
        --save_prompts \
        --captioner qwen \
        --captioner_size $size \
        --saved_caption_path results/satext/lv3/vlm_ocr_result/qwen
done


