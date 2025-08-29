
# realtext
CUDA_VISIBLE_DEVICES=1 python test/test_wllava.py \
    --pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
    --transformer_model_name_or_path="preset/models/dit4sr_q" \
    --image_path /media/dataset2/text_restoration/tair_published/real_text/LQ \
    --output_dir results/realtext/dit4sr_q_wllava \
    --save_prompts

