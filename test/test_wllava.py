import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import numpy as np
import re
from PIL import Image
import json

import torch

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from pipelines.pipeline_dit4sr import StableDiffusion3ControlNetPipeline
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from torchvision import transforms
import time


from qwen_vl_utils import process_vision_info


logger = get_logger(__name__, log_level="INFO")

# unicode conversion: char <-> int
# use chr() and ord()
# char_table = [chr(i) for i in range(32,127)]
# valid_voc = list(string.printable[:-6])
# invalid_voc=['□', '∫', 'æ', '⬏', 'Σ', '■', 'Å', 'Ḏ', '£', 'ń', '⌀', 'Ù', '│', 'Ⅶ', 'Â', 'ς', 'Ⅻ', '⁴', 'ъ', '∁', 'Æ', 'α', 'Ç', 'ˣ', '・', '⤤', 'Đ', 'ı', '≡', '⋄', 'Å', 'ᴴ', 'ᵗ', 'Ȃ', 'δ', 'Ì', 'Ρ', '⟷', 'ï', '«', 'ȯ', 'Ǒ', '⇩', 'ζ', '✰', '⁹', 'м', 'Ộ', '❘', '₄', '²', 'φ', '⌴', '⇨', 'ƌ', 'σ', 'Ⅸ', '∞', 'ţ', 'ů', '◁', '½', '¾', 'ᴾ', '�', 'ê', 'Ⅵ', 'ˢ', '°', 'ɮ', '⇪', 'ᵈ', 'Ė', 'Ǐ', '⊲', '·', 'û', '˅', '⊤', '↰', 'Ī', 'ȍ', '×', '⊝', '‟', '√', '➀', 'î', '↹', '➞', '↑', 'ü', '⋏', '℃', 'Û', 'Ȅ', '›', '⟶', '○', 'Ⓡ', 'Ȋ', '➜', 'ᴺ', 'å', '►', '˂', 'ι', 'ā', 'Ś', '∇', '•', '¥', '★', '⋅', 'ₖ', 'ũ', '⁼', 'İ', '∓', '⊂', '➯', '₅', 'Ồ', '»', 'Ž', 'ì', 'Ⅴ', '„', 'Ň', 'ú', '‑', 'Ä', '⊣', '˄', '˙', 'Ó', '±', '╳', 'ⁿ', 'ū', 'ş', 'л', 'Ṡ', 'ᴵ', 'Ȏ', 'ñ', 'λ', '✓', 'ø', '✞', '≤', 'Õ', '⎯', '⬌', 'ʳ', 'Š', '◉', '➨', 'ᶜ', 'ź', 'ġ', 'ÿ', '◦', 'ḻ', '➮', 'ᴸ', 'Ú', '─', '⇧', '⤶', 'ð', 'ë', 'Ξ', 'ȑ', '⇦', '↻', 'ă', 'Ě', 'Ω', 'Á', '₃', 'к', 'Ⅰ', '▬', '—', '∈', 'Ạ', '☐', '⁸', 'Ŕ', 'ù', 'â', 'п', 'ᴭ', '÷', '↲', '‘', 'Ȇ', 'ᵀ', '¿', 'Ț', '▎', 'ě', 'ⱽ', 'Λ', '∷', '△', 'ç', 'ǫ', 'Ầ', '➩', 'и', 'Ū', 'ý', '―', '⇵', 'Í', 'ꝋ', '↓', '©', '³', 'Ɔ', 'è', '🠈', 'ğ', 'Ⓐ', 'я', 'Φ', 'Ấ', 'ᵖ', '︽', '˚', 'œ', '∥', 'β', 'й', 'Ⓒ', '⬍', '∨', '℮', '¼', 'ć', '␣', 'Ã', '🡨', 'Ą', 'ǵ', '™', 'Ế', 'ᵐ', '◄', 'Ń', '✱', 'ô', '¢', '₁', 'Ⅱ', '¹', 'π', 'µ', 'Ĺ', '⍙', 'р', 'Ï', 'ε', '⟵', '∆', 'ы', '⧫', 'ã', 'ė', '⁰', '⬉', '−', '⬋', '◯', 'о', 'À', 'ρ', '☰', 'τ', 'ŗ', '⸬', 'Ö', 'é', 'ə', 'Ǫ', 'Ē', '⎵', '𝔀', 'ⓒ', 'ȏ', '“', 'Č', 'č', 'Î', '∙', 'ṣ', '\u200b', '✚', 'ō', '”', 'ö', 'ᴹ', '▢', 'ν', '⌣', '：', '︾', '﹘', 'а', '∖', '⌄', 'в', '︿', 'ᵃ', 'ớ', '↺', '▲', '▽', '…', 'Ë', '⌫', '⤷', '€', '⊘', 'Ŏ', '₂', '⤺', '⁵', 'Ȧ', '∧', 'ω', '卐', 'Ⅳ', '⁻', '↵', 'ĩ', 'Ⅲ', 'Ă', '⬸', 'ʃ', 'ȇ', '←', '⅓', '⮌', '⇥', 'η', '➦', 'Ô', '⬊', '℉', '⊥', 'á', 'ŉ', '⊚', '–', 'Ā', '∅', 'Ć', '∎', '⤸', '⦁', 'ē', 'ί', 'õ', 'ᴱ', 'υ', 'ß', '◡', 'È', '∣', 'Δ', 'ᴙ', 'ò', '⊢', 'κ', '☓', 'Ề', 'Θ', 'ä', '﹀', '☆', 'Ò', '˃', 'à', 'Ê', 'ʰ', 'Ğ', '’', '→', '®', '●', '⁺', 'Ţ', 'Ż', '̓', '▼', 'Ể', 'ᵒ', 'Ý', 'б', '➔', 'г', '∴', '⅔', '⬈', 'Ō', '∊', 'Π', 'Ⅷ', 'Ñ', '➝', 'É', 'Ł', 'ó', '∉', 'Ø', 'Ü', '⋮', 'ĺ', '≣', '∼', '↱', 'í', 'Ⅹ', 'ę', '⋯', 'с', '╎', '⤦', '⊼', 'ȧ', '∝', '⤻', 'ξ', 'š', '▾', 'γ', '¡', '⊳', 'д', '⁷', 'ж', '➧', 'ᴰ', '‧', '∘', 'ž', 'Ȯ', 'Ⅺ']
CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']


def decode(idxs):
    s = ''
    for idx in idxs:
        if idx < len(CTLABELS):
            s += CTLABELS[idx] 
        else:
            return s
    return s


def encode(word):
    s = []
    max_word_len = 25
    for i in range(max_word_len):
        if i < len(word):
            char=word[i]
            idx = CTLABELS.index(char)
            s.append(idx)
        else:
            s.append(96)
    return s


tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

if torch.cuda.device_count() >= 2:
    LLaVA_device = 'cuda:1'
    dit4sr_device = 'cuda:0'
elif torch.cuda.device_count() == 1:
    LLaVA_device = 'cuda:0'
    dit4sr_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')


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

# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, class_three, args):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def load_dit4sr_pipeline(args, accelerator):
    
    from model_dit4sr.transformer_sd3 import SD3Transformer2DModel
 
    # Load scheduler, tokenizer and models.
    
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.transformer_model_name_or_path, subfolder="transformer"
    )
    # controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, subfolder='controlnet')
    # Load the tokenizer
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # import correct text encoder class
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
            text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, args
        )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    transformer.requires_grad_(False)

    # Get the validation pipeline
    validation_pipeline = StableDiffusion3ControlNetPipeline(
        vae=vae, text_encoder=text_encoder_one, text_encoder_2=text_encoder_two, text_encoder_3=text_encoder_three, 
        tokenizer=tokenizer_one, tokenizer_2=tokenizer_two, tokenizer_3=tokenizer_three, 
        transformer=transformer, scheduler=scheduler,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

@torch.no_grad()
def process_llava(llava_agent, input_image):
    llama_prompt = llava_agent.gen_image_caption([input_image])[0]
    llama_prompt = remove_focus_sentences(llama_prompt)
    return llama_prompt

@torch.no_grad()
def process_qwen(model, processor, input_image_path):

    question = "OCR this image."
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
    clean_text = output_text[0].replace("\n", "")

    return clean_text


def process_saved_qwen(saved_caption_path, captioner_size, img_id):
    caption_json_path = f'{saved_caption_path}/{captioner_size}b.json'
    with open(caption_json_path, 'r', encoding='utf-8') as file:
        anns = json.load(file)
    ann = anns[img_id]
    gt_text = ann['gt_text']
    vlm_output_text = ann['vlm_output']
    return gt_text, vlm_output_text


def main(args):
    txt_path = os.path.join(args.output_dir, 'txt')
    os.makedirs(txt_path, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dit4sr")

    pipeline = load_dit4sr_pipeline(args, accelerator)


    # load vlm
    if args.captioner =='llava' and args.saved_caption_path is None:
        from llava.llm_agent import LLavaAgent
        from CKPT_PTH import LLAVA_MODEL_PATH
        cap_agent = LLavaAgent(LLAVA_MODEL_PATH, LLaVA_device, load_8bit=True, load_4bit=False)

    elif args.captioner == 'qwen' and args.saved_caption_path is None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model_size=args.captioner_size
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(f"Qwen/Qwen2.5-VL-{model_size}B-Instruct", torch_dtype="auto", device_map="auto")
        vlm_processor = AutoProcessor.from_pretrained(f"Qwen/Qwen2.5-VL-{model_size}B-Instruct")



    # load SAText annotation for gt prompting 
    if args.satext_ann_path is not None:
        model_H, model_W = 512, 512 
        # load json 
        json_path = args.satext_ann_path
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            json_data = sorted(json_data.items())
        val_gt_json = {}
        for idx, (img_id, img_anns) in enumerate(json_data):
            anns = img_anns['0']['text_instances']
            boxes=[]
            texts=[]
            text_encs=[]
            polys=[]
            prompts=[]
            for ann in anns:
                # process text 
                text = ann['text']
                count=0
                for char in text:
                    # only allow OCR english vocab: range(32,127)
                    if 32 <= ord(char) and ord(char) < 127:
                        count+=1
                        # print(char, ord(char))
                if count == len(text) and count < 26:
                    texts.append(text)
                    text_encs.append(encode(text))
                    assert text == decode(encode(text)), 'check text encoding !'
                else:
                    continue
                # process box
                box_xyxy = ann['bbox']
                x1,y1,x2,y2 = box_xyxy
                box_xywh = [ x1, y1, x2-x1, y2-y1 ]
                box_xyxy_scaled = list(map(lambda x: x/model_H, box_xyxy))  # scale box coord to [0,1]
                x1,y1,x2,y2 = box_xyxy_scaled 
                box_cxcywh = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]   # xyxy -> cxcywh
                # # select box format
                # if cfg.dataset.data_args['bbox_format'] == 'xywh_unscaled':
                #     processed_box = box_xywh
                #     processed_box = list(map(lambda x: int(x), processed_box))
                # elif cfg.dataset.data_args['bbox_format'] == 'xyxy_scaled':
                #     processed_box = box_xyxy_scaled
                #     processed_box = list(map(lambda x: round(x,4), processed_box))
                # elif cfg.dataset.data_args['bbox_format'] == 'cxcywh_scaled':
                #     processed_box = box_cxcywh
                #     processed_box = list(map(lambda x: round(x,4), processed_box))
                processed_box = box_cxcywh
                processed_box = list(map(lambda x: round(x,4), processed_box))
                boxes.append(processed_box)
                # process polygon
                poly = np.array(ann['polygon']).astype(np.int32)    # 16 2
                # scale poly
                poly_scaled = poly / np.array([model_W, model_H])
                polys.append(poly_scaled)
            # check is anns are properly processed
            assert len(boxes) == len(texts) == len(text_encs) == len(polys), f" Check len"
            if len(boxes) == 0 or len(polys) == 0:
                    continue
            # process prompt
            caption = [f'"{txt}"' for txt in texts]
            # prompt = f"A high-quality photo containing the word {', '.join(caption) }."
            # if cfg.prompter_args.prompt_style == 'CAPTION':
            #     prompt = f"A realistic scene where the texts {', '.join(caption) } appear clearly on signs, boards, buildings, or other objects."
            # elif cfg.prompter_args.prompt_style == 'TAG':
            #     prompt = f"{', '.join(caption)}"
            # if cfg.prompter_args.use_llava_prompt:
            #     prompt = llava_dic[img_id]
            prompt = f"A realistic scene where the texts {', '.join(caption) } appear clearly on signs, boards, buildings, or other objects."
            prompts.append(prompt)
            val_gt_json[img_id] = {
                'boxes': boxes,
                'texts': texts,
                'text_encs': text_encs,
                'polys': polys,
                'gtprompts': prompts
            }


    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        if os.path.isdir(args.image_path):
            image_names = sorted(glob.glob(f'{args.image_path}/*.*'))
        else:
            image_names = [args.image_path]
        image_names = sorted(image_names)
        print(f'Number of testing images: {len(image_names)}')
        for image_idx, image_name in enumerate(image_names[:]):
            
            # img id 
            img_id = image_name.split('/')[-1].split('.')[0]
            img_ann = val_gt_json[img_id]
            gt_prompt = img_ann['gtprompts'][0]
            print(f'================== processing {image_idx} img: {img_id} ===================')

            # read img
            validation_image = Image.open(image_name).convert("RGB")

            # process prompt 
            if args.captioner == 'llava':
                validation_prompt = process_llava(cap_agent, validation_image)
            elif args.captioner == 'qwen':
                if args.saved_caption_path is not None:
                    gt_text, validation_prompt = process_saved_qwen(args.saved_caption_path, args.captioner_size, img_id)
                else:
                    validation_prompt = process_qwen(vlm_model, vlm_processor, image_name)
            if args.use_satext_gt_prompt:
                print('Using SAText GT prompt ...')
                validation_prompt = gt_prompt
            validation_prompt += ' ' + args.added_prompt # clean, extremely detailed, best quality, sharp, clean
            if args.use_null_prompt:
                validation_prompt = ''
            negative_prompt = args.negative_prompt #dirty, messy, low quality, frames, deformed, 

            # save prompt
            if args.save_prompts:
                txt_save_path = f"{txt_path}/{os.path.basename(image_name).split('.')[0]}.txt"
                file = open(txt_save_path, "w")
                file.write(validation_prompt)
                file.close()
            print(f'{validation_prompt}')

            ori_width, ori_height = validation_image.size
            resize_flag = False
            rscale = args.upscale
            if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale: # f
                scale = (args.process_size//rscale)/min(ori_width, ori_height)
                tmp_image = validation_image.resize((int(scale*ori_width), int(scale*ori_height)),Image.BICUBIC)

                validation_image = tmp_image
                resize_flag = True

            validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale), Image.BICUBIC)
            validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8), Image.BICUBIC)
            width, height = validation_image.size
            resize_flag = True #

            print(f'input size: {height}x{width}')

            for sample_idx in range(args.sample_times):
                os.makedirs(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/', exist_ok=True)

            for sample_idx in range(args.sample_times):  
                with torch.autocast("cuda"):
                    start_time = time.time()
                    image = pipeline(
                            prompt=validation_prompt, control_image=validation_image, num_inference_steps=args.num_inference_steps, generator=generator, height=height, width=width,
                            guidance_scale=args.guidance_scale, negative_prompt=negative_prompt,
                            start_point=args.start_point, latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap,
                            args=args,
                        ).images[0]
                    end_time = time.time()
                    print(f'inference time: {end_time-start_time:.2f}s')
                
                if args.align_method == 'nofix':
                    image = image
                else:
                    if args.align_method == 'wavelet':
                        image = wavelet_color_fix(image, validation_image)
                    elif args.align_method == 'adain':
                        image = adain_color_fix(image, validation_image)

                if resize_flag: 
                    image = image.resize((ori_width*rscale, ori_height*rscale), Image.BICUBIC)
                    
                name, ext = os.path.splitext(os.path.basename(image_name))
                
                image.save(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/{name}.png')
                # image.save(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}_{name}.png')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='preset/models/stable-diffusion-3.5-medium')
    parser.add_argument("--transformer_model_name_or_path", type=str, default='dit4sr_q')
    parser.add_argument("--prompt", type=str, default="") # user can add self-prompt to improve the results
    parser.add_argument("--added_prompt", type=str, default='Cinematic, hyper sharpness, highly detailed, perfect without deformations, '
                            'camera, hyper detailed photo - realistic maximum detail, 32k, Color '
                            'Grading, ultra HD, extreme meticulous detailing, skin pore detailing. ')
    parser.add_argument("--negative_prompt", type=str, default='motion blur, noisy, dotted, bokeh, pointed, '
                            'CG Style, 3D render, unreal engine, blurring, dirty, messy, '
                            'worst quality, low quality, frames, watermark, signature, jpeg artifacts, '
                            'deformed, lowres, chaotic')
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # latent size, for 24G
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # image size, for 13G
    parser.add_argument("--latent_tiled_size", type=int, default=64) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=24) 
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='noise') # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point. 
    parser.add_argument("--save_prompts", action='store_true')
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    # pho
    parser.add_argument("--satext_ann_path", type=str)
    parser.add_argument("--use_satext_gt_prompt", action='store_true')
    parser.add_argument("--use_null_prompt", action='store_true')
    parser.add_argument('--captioner', type=str, default='llava')
    parser.add_argument('--captioner_size', type=int)
    parser.add_argument('--saved_caption_path', type=str)

    args = parser.parse_args()
    main(args)



