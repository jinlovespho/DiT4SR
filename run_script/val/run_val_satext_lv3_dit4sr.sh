
# satext lv3 - use gt prompt 
CUDA_VISIBLE_DEVICES=1 python test/test_wllava.py \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --transformer_model_name_or_path="preset/models/dit4sr_q" \
    --image_path /media/dataset2/text_restoration/SAMText_test_degradation/lv3 \
    --satext_ann_path /media/dataset2/text_restoration/100K/test/dataset.json \
    --output_dir results/satext/lv3/dit4sr_q_wllava_gtprompt_lremb \
    --use_satext_gt_prompt \
    --save_prompts \
    --start_point lr


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




