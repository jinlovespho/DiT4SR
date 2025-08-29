mkdir preset
mkdir preset/models
mkdir llava_ckpt

# download dit4sr
huggingface-cli download acceptee/DiT4SR --local-dir preset/dit4sr

# download Stable Diffusion 3.5 medium
huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir preset/models/stable-diffusion-3.5-medium

# download CLIP ViT-L/14-336
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir llava_ckpt/clip-vit-large-patch14-336

# download LLaVA-1.5 13B
huggingface-cli download liuhaotian/llava-v1.5-13b --local-dir llava_ckpt/llava-v1.5-13b
