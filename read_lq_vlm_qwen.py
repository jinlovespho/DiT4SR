import json
import glob
import torch
from PIL import Image 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


ann_path = '/media/dataset2/text_restoration/100K/test/dataset.json'
lq_path = '/media/dataset2/text_restoration/SAMText_test_degradation/lv3'
hq_path = '/media/dataset2/text_restoration/100K/test'


anns = json.load(open(ann_path))
lq_imgs = sorted(glob.glob(f'{lq_path}/*.jpg'))
hq_imgs = sorted(glob.glob(f'{hq_path}/*.jpg'))



# load vlm
model_size=72
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(f"Qwen/Qwen2.5-VL-{model_size}B-Instruct", torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(f"Qwen/Qwen2.5-VL-{model_size}B-Instruct")

txt_save_path = f"results/satext/lv3/vlm_ocr_result/qwen/{model_size}b.txt"
with open(txt_save_path, 'a') as file:
        
    imgs = zip(lq_imgs, hq_imgs)
    question = "OCR this image."
    file.write(f'{question}\n\n\n')
    for i, (lq, hq) in enumerate(imgs):

        lq_img_id = lq.split('/')[-1].split('.')[0]
        hq_img_id = hq.split('/')[-1].split('.')[0]
        assert lq_img_id == hq_img_id

        file.write(f'{i} {lq_img_id}\n')

        lq_img = Image.open(lq).convert("RGB")
        img_anns = anns[lq_img_id]['0']['text_instances']

        gt_texts=[]
        for img_ann in img_anns:
            gt_texts.append(img_ann['text'])
        gt_prompt = ' '.join(gt_texts)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{lq}",
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
        print('GT TEXT: ', gt_prompt)
        print('VLM OUTPUT: ', output_text[0])

        file.write(f'GT TEXT: {gt_prompt}\n')
        clean_text = output_text[0].replace('\n', " ")
        file.write(f"VLM OUTPUT: {clean_text}\n\n")

        # breakpoint()