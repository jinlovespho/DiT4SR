import torch
from PIL import Image
import os
from tqdm import tqdm
import re
import sys
sys.path.append(os.getcwd())
from llava.llm_agent import LLavaAgent
from CKPT_PTH import LLAVA_MODEL_PATH
from qwen_vl_utils import process_vision_info
import glob

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, default='preset/datasets/train_datasets/training_for', help='the dataset you want to tag.') # 
parser.add_argument("--save_dir", type=str, default='preset/datasets/train_datasets/training_for', help='the dataset you want to tag.') # 
parser.add_argument("--stop_num", type=int, default=-1)
parser.add_argument("--start_num", type=int, default=0)
parser.add_argument('--captioner', type=str, default='llava')
parser.add_argument('--captioner_size', type=int)
args = parser.parse_args()

def remove_focus_sentences(text):
    prohibited_words = ['focus', 'focal', 'prominent', 'close-up', 'black and white', 'blur', 'depth', 'dense', 'locate', 'position']
    parts = re.split(r'([.?!])', text)
    filtered_sentences = []
    i = 0
    while i < len(parts):
        sentence = parts[i]
        punctuation = parts[i+1] if (i+1 < len(parts)) else ''
        full_sentence = sentence + punctuation
        full_sentence_lower = full_sentence.lower()
        skip = False
        for word in prohibited_words:
            if word.lower() in full_sentence_lower:
                skip = True
                break
        if not skip:
            filtered_sentences.append(full_sentence)
        i += 2
    return "".join(filtered_sentences).strip()

@torch.no_grad()
def process_llava(
    input_image):
    llama_prompt = llava_agent.gen_image_caption([input_image])[0]
    llama_prompt = remove_focus_sentences(llama_prompt)
    return llama_prompt

@torch.no_grad()
def process_qwen(model, processor, input_image_path):
    question = 'Please describe the actual objects and the texts in the image in a very detailed manner. Please do not include descriptions related to the focus and bokeh of this image. Please do not include descriptions like the background is blurred.'
    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{input_image_path}",
                    },
                    {"type": "text", "text": f"{question}"},
                ],
            }
        ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    clean_text = output_text[0].replace('\n', '')
    clean_text = remove_focus_sentences(clean_text)
    return clean_text

def PrintInfo(x):
    if not isinstance(x,list):
        x=[x]
    for i in x:
        print('shape : {} ; dtype : {} ; max : {} ; min : {}'.format(i.shape,i.dtype,i.max(),i.min())  )

img_ext=f'jpg'
img_folder = args.img_dir
prompt_save_folder = args.save_dir
os.makedirs(prompt_save_folder, exist_ok=True)
imgs = sorted(glob.glob(f'{img_folder}/*.{img_ext}'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load vlm captioner
if args.captioner == 'llava':
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device, load_8bit=True, load_4bit=False)
elif args.captioner =='qwen':
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    captioner_size=args.captioner_size
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(f"Qwen/Qwen2.5-VL-{captioner_size}B-Instruct", torch_dtype="auto", device_map="auto")
    vlm_processor = AutoProcessor.from_pretrained(f"Qwen/Qwen2.5-VL-{captioner_size}B-Instruct")


for img in tqdm(imgs[args.start_num:args.stop_num]):
    print('img: ', img)
    img_id = img.split('/')[-1].split('.')[0]
    if args.captioner == 'llava':
        img = Image.open(img).convert('RGB')
        prompt = process_llava(img)
    elif args.captioner == 'qwen':
        prompt = process_qwen(vlm_model, vlm_processor, img)
    with open(f'{prompt_save_folder}/{img_id}.txt', 'w', encoding="utf-8") as f:
        f.write(prompt)