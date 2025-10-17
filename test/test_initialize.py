import json 
import torch 

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from pipelines.pipeline_dit4sr import StableDiffusion3ControlNetPipeline
from qwen_vl_utils import process_vision_info
from test_utils import remove_focus_sentences
from test_utils import encode, decode


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


# load annotation for gt prompting 
def load_json_anns(cfg):
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