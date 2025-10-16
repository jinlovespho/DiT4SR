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

from omegaconf import OmegaConf
from torchvision.utils import save_image 

import pyiqa
import cv2 
import wandb


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
def load_text_encoders(class_one, class_two, class_three, cfg):
    text_encoder_one = class_one.from_pretrained(
        cfg.ckpt.init_path.text_encoder, subfolder="text_encoder", revision=None, variant=None
    )
    text_encoder_two = class_two.from_pretrained(
        cfg.ckpt.init_path.text_encoder, subfolder="text_encoder_2", revision=None, variant=None
    )
    text_encoder_three = class_three.from_pretrained(
        cfg.ckpt.init_path.text_encoder, subfolder="text_encoder_3", revision=None, variant=None
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def load_dit4sr_pipeline(cfg, accelerator):
    
    from model_dit4sr.transformer_sd3 import SD3Transformer2DModel
 
    # Load scheduler, tokenizer and models.
    
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.ckpt.init_path.noise_scheduler, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        cfg.ckpt.init_path.vae,
        subfolder="vae",
    )
    if cfg.ckpt.resume_path.dit is not None:
        transformer = SD3Transformer2DModel.from_pretrained(
            cfg.ckpt.resume_path.dit, subfolder="transformer"
        )
        print(f"Loaded Trained DiT checkpoint: {cfg.ckpt.resume_path.dit}")
    else:
        transformer = SD3Transformer2DModel.from_pretrained(
            cfg.ckpt.init_path.dit, subfolder="transformer"
        )
    # controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, subfolder='controlnet')
    # Load the tokenizer
    tokenizer_one = CLIPTokenizer.from_pretrained(
        cfg.ckpt.init_path.tokenizer,
        subfolder="tokenizer",
        revision=None,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        cfg.ckpt.init_path.tokenizer,
        subfolder="tokenizer_2",
        revision=None,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        cfg.ckpt.init_path.tokenizer,
        subfolder="tokenizer_3",
        revision=None,
    )

    # import correct text encoder class
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        cfg.ckpt.init_path.text_encoder, None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        cfg.ckpt.init_path.text_encoder, None, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        cfg.ckpt.init_path.text_encoder, None, subfolder="text_encoder_3"
    )

    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
            text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, cfg
        )


    
    ts_module=None
    # load ts module 
    if 'testr' in cfg.train.model:
        from testr.adet.modeling.transformer_detector import TransformerDetector
        from testr.adet.config import get_cfg

        # get testr config
        config_testr = get_cfg()
        config_testr.merge_from_file('./testr/configs/TESTR/TESTR_R_50_Polygon.yaml')
        config_testr.freeze()

        # load testr model
        ts_module = TransformerDetector(config_testr)

        # load trained ckpt
        if cfg.ckpt.resume_path.dit is not None:
            ckpt_iter = int(cfg.ckpt.resume_path.dit.split('/')[-1].split('-')[-1])
            ts_ckpt_path = f'{cfg.ckpt.resume_path.dit}/ts_module{ckpt_iter:07d}.pt'
            ckpt = torch.load(ts_ckpt_path, map_location="cpu")
            load_result = ts_module.load_state_dict(ckpt['ts_module'], strict=False)
            print("Loaded TESTR checkpoint keys:")
            print(" - Missing keys:", load_result.missing_keys)
            print(" - Unexpected keys:", load_result.unexpected_keys)


        ts_module.requires_grad_(False)
        ts_module.eval()


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
        transformer=transformer, scheduler=scheduler, ts_module = ts_module, cfg=cfg
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


def process_saved_qwen(saved_caption_path, captioner_size, lq_id):
    caption_json_path = f'{saved_caption_path}/{captioner_size}b.json'
    with open(caption_json_path, 'r', encoding='utf-8') as file:
        anns = json.load(file)
    ann = anns[lq_id]
    gt_text = ann['gt_text']
    vlm_output_text = ann['vlm_output']
    return gt_text, vlm_output_text


def main(cfg):
    val_data_name = cfg.data.val.name
    assert val_data_name in ['realtext', 'satext_lv3', 'satext_lv2', 'satext_lv1']
    exp_name = cfg.ckpt.resume_path.dit.split('/')[-2]
    num_ckpt = cfg.ckpt.resume_path.dit.split('/')[-1]
    exp_name = f'{exp_name}_{num_ckpt}'
    cfg.exp_name = exp_name

    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
    )

    # If passed along, set the training seed now.
    if cfg.init.seed is not None:
        set_seed(cfg.init.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(cfg.save.output_dir, exist_ok=True)

    # set tracker
    if accelerator.is_main_process:
        wandb.login(key=cfg.log.tracker.key)
        wandb.init(
            project=cfg.log.tracker.project_name,
            name=f'VAL_{val_data_name}_{exp_name}',
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # load pipeline
    pipeline = load_dit4sr_pipeline(cfg, accelerator)


    # prompt selection
    if cfg.data.val.use_precomputed_prompts is not None:
        precom_prompts = sorted(glob.glob(f"{cfg.data.val.use_precomputed_prompts}/*.txt"))
    else:
        if cfg.data.val.captioner is not None:
            # load vlm
            if cfg.data.val.captioner =='llava' and cfg.data.val.saved_caption_path is None:
                from llava.llm_agent import LLavaAgent
                from CKPT_PTH import LLAVA_MODEL_PATH
                cap_agent = LLavaAgent(LLAVA_MODEL_PATH, LLaVA_device, load_8bit=True, load_4bit=False)
            elif cfg.data.val.captioner == 'qwen' and cfg.data.val.saved_caption_path is None:
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                model_size=cfg.data.val.captioner_size
                vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(f"Qwen/Qwen2.5-VL-{model_size}B-Instruct", torch_dtype="auto", device_map="auto")
                vlm_processor = AutoProcessor.from_pretrained(f"Qwen/Qwen2.5-VL-{model_size}B-Instruct")
    
    # load SAText annotation for gt prompting 
    if cfg.data.val.name in ['realtext', 'satext_lv3', 'satext_lv2', 'satext_lv1']:
        model_H, model_W = 512, 512 
        # load json 
        json_path = cfg.data.val.ann_path
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            json_data = sorted(json_data.items())
        val_gt_json = {}
        for idx, (lq_id, img_anns) in enumerate(json_data):
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
            #     prompt = llava_dic[lq_id]
            prompt = f"A realistic scene where the texts {', '.join(caption) } appear clearly on signs, boards, buildings, or other objects."
            prompts.append(prompt)
            val_gt_json[lq_id] = {
                'boxes': boxes,
                'texts': texts,
                'text_encs': text_encs,
                'polys': polys,
                'gtprompts': prompts
            }


    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if cfg.init.seed is not None:
            generator.manual_seed(cfg.init.seed)

        if os.path.isdir(cfg.data.val.lq_image_path):
            lq_imgs = sorted(glob.glob(f'{cfg.data.val.lq_image_path}/*.jpg'))
        gt_exist=False
        if os.path.isdir(cfg.data.val.gt_image_path):
            gt_exist=True
            gt_imgs = sorted(glob.glob(f'{cfg.data.val.gt_image_path}/*.jpg'))
            assert len(lq_imgs) == len(gt_imgs)
        print(f'Number of LQ testing images: {len(lq_imgs)}')


        # SR metrics
        metric_psnr = pyiqa.create_metric('psnr', device=accelerator.device)
        metric_ssim = pyiqa.create_metric('ssimc', device=accelerator.device)
        metric_lpips = pyiqa.create_metric('lpips', device=accelerator.device)
        metric_dists = pyiqa.create_metric('dists', device=accelerator.device)
        # metric_fid = pyiqa.create_metric('fid', device=device)
        metric_niqe = pyiqa.create_metric('niqe', device=accelerator.device)
        metric_musiq = pyiqa.create_metric('musiq', device=accelerator.device)
        metric_maniqa = pyiqa.create_metric('maniqa', device=accelerator.device)
        metric_clipiqa = pyiqa.create_metric('clipiqa', device=accelerator.device)


        tot_val_psnr=[]
        tot_val_ssim=[]
        tot_val_lpips=[]
        tot_val_dists=[]
        tot_val_niqe=[]
        tot_val_musiq=[]
        tot_val_maniqa=[]
        tot_val_clipiqa=[]


        for image_idx in range(len(lq_imgs)):

            # img id 
            lq_name = lq_imgs[image_idx]
            lq_id = lq_name.split('/')[-1].split('.')[0]
            img_ann = val_gt_json[lq_id]
            gt_prompt = img_ann['gtprompts'][0]

            val_text = img_ann['texts']
            val_polys = img_ann['polys']

            if gt_exist:
                gt_name = gt_imgs[image_idx]
                gt_id = gt_name.split('/')[-1].split('.')[0]
                assert lq_id == gt_id 
                gt_image = Image.open(gt_name).convert("RGB")


            print(f'================== processing {image_idx} img: {lq_id} ===================')

            # read img
            lq_image = Image.open(lq_name).convert("RGB")

            validation_prompt = cfg.data.val.added_prompt # clean, extremely detailed, best quality, sharp, clean
            negative_prompt = cfg.data.val.negative_prompt #dirty, messy, low quality, frames, deformed, 

            if cfg.data.val.use_precomputed_prompts is not None:
                precom_prompt = precom_prompts[image_idx]
                prompt_id = precom_prompt.split('/')[-1].split('.')[0]
                assert lq_id == prompt_id
                with open(f'{precom_prompt}', 'r') as f:
                    validation_prompt = f.read().strip()
            else:
                if cfg.data.val.captioner is not None:
                    # process prompt 
                    if cfg.data.val.captioner == 'llava':
                        validation_prompt = process_llava(cap_agent, lq_image)
                    elif cfg.data.val.captioner == 'qwen':
                        if cfg.data.val.saved_caption_path is not None:
                            gt_text, validation_prompt = process_saved_qwen(cfg.data.val.saved_caption_path, cfg.data.val.captioner_size, lq_id)
                        else:
                            validation_prompt = process_qwen(vlm_model, vlm_processor, lq_name)

            if cfg.data.val.use_satext_gtprompt:
                print('Using SAText GT prompt ...')
                validation_prompt = gt_prompt

            if cfg.data.val.use_nullprompt:
                validation_prompt = validation_prompt
            
            # # save prompt
            # if cfg.data.val.save_prompts:
            #     txt_save_path = f"{cfg.save.output_dir}/{cfg.log.tracker.run_name}/txt"
            #     os.makedirs(txt_save_path, exist_ok=True)
            #     file = open(f'{txt_save_path}/{lq_id}.txt', "w")
            #     file.write(validation_prompt)
            #     file.close()
            # print(f'{validation_prompt}')

            ori_width, ori_height = lq_image.size
            resize_flag = False
            rscale = cfg.data.val.upscale
            if ori_width < cfg.data.val.process_size//rscale or ori_height < cfg.data.val.process_size//rscale: # f
                scale = (cfg.data.val.process_size//rscale)/min(ori_width, ori_height)
                tmp_image = lq_image.resize((int(scale*ori_width), int(scale*ori_height)),Image.BICUBIC)

                lq_image = tmp_image
                resize_flag = True

            lq_image = lq_image.resize((lq_image.size[0]*rscale, lq_image.size[1]*rscale), Image.BICUBIC)
            lq_image = lq_image.resize((lq_image.size[0]//8*8, lq_image.size[1]//8*8), Image.BICUBIC)
            width, height = lq_image.size
            resize_flag = True #

            print(f'input size: {height}x{width}')

            # for sample_idx in range(cfg.data.val.sample_times):
            #     os.makedirs(f'{cfg.save.output_dir}/sample{str(sample_idx).zfill(2)}/', exist_ok=True)

            for sample_idx in range(cfg.data.val.sample_times):  
                with torch.autocast("cuda"):
                    start_time = time.time()
                    val_out = pipeline(
                            prompt=validation_prompt, control_image=lq_image, num_inference_steps=cfg.data.val.num_inference_steps, generator=generator, height=height, width=width,
                            guidance_scale=cfg.data.val.guidance_scale, negative_prompt=negative_prompt,
                            start_point=cfg.data.val.start_point, latent_tiled_size=cfg.data.val.latent_tiled_size, latent_tiled_overlap=cfg.data.val.latent_tiled_overlap,
                            output_type = 'pt', return_dict=False, lq_id=lq_id, val_data_name=val_data_name, cfg=cfg, mode='val'
                        )
                    val_restored_img = val_out[0]
                    if len(val_out) > 1:
                        val_ocr_result = val_out[1]
                    end_time = time.time()
                    print(f'inference time: {end_time-start_time:.2f}s')
                
                # Save only the restored image
                res_save_path = f'{cfg.save.output_dir}/{val_data_name}/{exp_name}/final_restored_img'
                os.makedirs(res_save_path, exist_ok=True)
                save_image(val_restored_img, f'{res_save_path}/{lq_id}.png')
                

                # lq 
                val_lq_img = transforms.ToTensor()((lq_image)).unsqueeze(dim=0).cuda()  # [0,1]
                # val_lq_img = (val_lq_img + 1.0) / 2.0
                # gt 
                val_gt_img = transforms.ToTensor()((gt_image)).unsqueeze(dim=0).cuda()  # [0,1]
                # val_gt_img = (val_gt_img + 1.0) / 2.0
                # restored
                val_res_img = val_restored_img.squeeze(dim=0).permute(1,2,0).cpu().detach().numpy()*255.0
                val_res_img = val_res_img.astype(np.uint8)
                # val_img = torch.concat([val_lq_img, val_restored_img, val_gt_img], dim=3)
                # save_image(val_img, f'{val_save_path}/step{global_step}_valimg{val_step}_{val_lq_id[val_idx]}.png')
                # save_image(val_img, f'{val_save_path}/restored_{lq_id}.png')
                # save_image(val_lq_img, f'{val_save_path}/step{global_step}_{val_lq_id[val_idx]}.png', normalize=True)
                # save_image(val_res_img, f'{val_save_path}/step{global_step}_{val_lq_id[val_idx]}.png', normalize=True)
                
                # log total psnr, ssim, lpips for val
                tot_val_psnr.append(torch.mean(metric_psnr(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())
                tot_val_ssim.append(torch.mean(metric_ssim(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())
                tot_val_lpips.append(torch.mean(metric_lpips(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())
                tot_val_dists.append(torch.mean(metric_dists(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())
                tot_val_niqe.append(torch.mean(metric_niqe(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())
                tot_val_musiq.append(torch.mean(metric_musiq(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())
                tot_val_maniqa.append(torch.mean(metric_maniqa(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())
                tot_val_clipiqa.append(torch.mean(metric_clipiqa(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())

                # log sampling val imgs to wandb
                if accelerator.is_main_process and cfg.log.tracker.report_to == 'wandb':
                    wandb.log({f'val_metric/val_psnr': torch.mean(metric_psnr(
                                                                                    val_restored_img.to(torch.float32), 
                                                                                    val_gt_img.to(torch.float32))).item(),
                            f'val_metric/val_ssim': torch.mean(metric_ssim(
                                                                                    val_restored_img.to(torch.float32), 
                                                                                    val_gt_img.to(torch.float32))).item(),
                            f'val_metric/val_lpips': torch.mean(metric_lpips(
                                                                                    val_restored_img.to(torch.float32), 
                                                                                    val_gt_img.to(torch.float32))).item(),
                            f'val_metric/val_dists': torch.mean(metric_dists(
                                                                                    val_restored_img.to(torch.float32), 
                                                                                    val_gt_img.to(torch.float32))).item(),
                            f'val_metric/val_niqe': torch.mean(metric_niqe(
                                                                                    val_restored_img.to(torch.float32), 
                                                                                    val_gt_img.to(torch.float32))).item(),
                            f'val_metric/val_musiq': torch.mean(metric_musiq(
                                                                                    val_restored_img.to(torch.float32), 
                                                                                    val_gt_img.to(torch.float32))).item(),
                            f'val_metric/val_maniqa': torch.mean(metric_maniqa(
                                                                                    val_restored_img.to(torch.float32), 
                                                                                    val_gt_img.to(torch.float32))).item(),
                            f'val_metric/val_clipiqa': torch.mean(metric_clipiqa(
                                                                                    val_restored_img.to(torch.float32), 
                                                                                    val_gt_img.to(torch.float32))).item(),
                            })


                if 'testr' in cfg.train.model:
                    ## -------------------- vis val ocr results -------------------- 
                    print('== logging val ocr results ===')
                    val_save_path = f'{cfg.save.output_dir}/{val_data_name}/{exp_name}/final_result'
                    os.makedirs(val_save_path, exist_ok=True)

                    img_lq = val_lq_img.detach().permute(0,2,3,1).cpu().numpy() # b h w c
                    img_gt = val_gt_img.detach().permute(0,2,3,1).cpu().numpy()

                    for vis_batch_idx in range(len(img_gt)):
                        vis_lq = img_lq[vis_batch_idx] # h w c 
                        # vis_lq = (vis_lq + 1.0)/2.0 * 255.0
                        vis_lq = vis_lq * 255.0
                        vis_lq = vis_lq.astype(np.uint8)
                        vis_lq = vis_lq.copy()

                        vis_gt = img_gt[vis_batch_idx] # h w c
                        # vis_gt = (vis_gt + 1.0)/2.0 * 255.0
                        vis_gt = vis_gt * 255.0
                        vis_gt = vis_gt.astype(np.uint8)
                        vis_pred = vis_lq.copy()
                        vis_gt = vis_gt.copy()
                        vis_gt2 = vis_gt.copy()

                        ocr_res = val_ocr_result[vis_batch_idx]
                        vis_polys = ocr_res.polygons.view(-1,16,2)  # b 16 2
                        vis_recs = ocr_res.recs                     # b 25
                        for vis_img_idx in range(len(vis_polys)):
                            pred_poly = vis_polys[vis_img_idx]   # 16 2
                            pred_poly = np.array(pred_poly.detach().cpu()).astype(np.int32)         
                            pred_rec = vis_recs[vis_img_idx]     # 25
                            pred_txt = decode(pred_rec.tolist())
                            cv2.polylines(vis_pred, [pred_poly], isClosed=True, color=(0,255,0), thickness=2)
                            cv2.putText(vis_pred, pred_txt, (pred_poly[0][0], pred_poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        # cv2.imwrite(f'{val_ocr_save_path}/ocr_pred_{lq_id}.jpg.jpg', vis_pred[:,:,::-1])

                        if gt_exist:
                            gt_polys = val_polys
                            gt_texts = val_text
                            for vis_img_idx in range(len(gt_polys)):
                                gt_poly = gt_polys[vis_img_idx]*512.0   # 16 2
                                gt_poly = np.array(gt_poly).astype(np.int32)
                                gt_txt = gt_texts[vis_img_idx]
                                cv2.polylines(vis_gt, [gt_poly], isClosed=True, color=(0,255,0), thickness=2)
                                cv2.putText(vis_gt, gt_txt, (gt_poly[0][0], gt_poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                            vis_result = cv2.hconcat([vis_lq, val_res_img, vis_gt2, vis_pred, vis_gt])
                            cv2.imwrite(f'{val_save_path}/{lq_id}.jpg', vis_result[:,:,::-1])
                    ## -------------------- vis val ocr results -------------------- 
        
        # average using numpy
        tot_val_psnr = np.array(tot_val_psnr).mean()
        tot_val_ssim = np.array(tot_val_ssim).mean()
        tot_val_lpips = np.array(tot_val_lpips).mean()
        tot_val_dists = np.array(tot_val_dists).mean()
        tot_val_niqe = np.array(tot_val_niqe).mean()
        tot_val_musiq = np.array(tot_val_musiq).mean()
        tot_val_maniqa = np.array(tot_val_maniqa).mean()
        tot_val_clipiqa = np.array(tot_val_clipiqa).mean()

        # log total val metrics 
        if accelerator.is_main_process and cfg.log.tracker.report_to == 'wandb':
            wandb.log({
                f'val_metric/tot_val_psnr': tot_val_psnr,
                f'val_metric/tot_val_ssim': tot_val_ssim,
                f'val_metric/tot_val_lpips': tot_val_lpips,
                f'val_metric/tot_val_dists': tot_val_dists,
                f'val_metric/tot_val_niqe': tot_val_niqe,
                f'val_metric/tot_val_musiq': tot_val_musiq,
                f'val_metric/tot_val_maniqa': tot_val_maniqa,
                f'val_metric/tot_val_clipiqa': tot_val_clipiqa,
            })
                
                # if cfg.data.val.align_method == 'nofix':
                #     image = val_restored_img
                # else:
                #     if cfg.data.val.align_method == 'wavelet':
                #         image = wavelet_color_fix(image, lq_image)
                #     elif cfg.data.val.align_method == 'adain':
                #         image = adain_color_fix(image, lq_image)

                # if resize_flag: 
                #     image = image.resize((ori_width*rscale, ori_height*rscale), Image.BICUBIC)
                    
                # name, ext = os.path.splitext(os.path.basename(image_name))
                
                # image.save(f'{cfg.save.output_dir}/sample{str(sample_idx).zfill(2)}/{name}.png')
                # # image.save(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}_{name}.png')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)



