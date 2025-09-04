import json
import glob
import torch
from PIL import Image 
from transformers import AutoProcessor, LlavaForConditionalGeneration


ann_path = '/media/dataset2/text_restoration/100K/test/dataset.json'
lq_path = '/media/dataset2/text_restoration/SAMText_test_degradation/lv3'
hq_path = '/media/dataset2/text_restoration/100K/test'


anns = json.load(open(ann_path))
lq_imgs = sorted(glob.glob(f'{lq_path}/*.jpg'))
hq_imgs = sorted(glob.glob(f'{hq_path}/*.jpg'))


# load vlm
model_size=7
model = LlavaForConditionalGeneration.from_pretrained(f"llava-hf/llava-1.5-{model_size}b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained(f"llava-hf/llava-1.5-{model_size}b-hf")

txt_save_path = f"results/satext/lv3/vlm_ocr_result/llava/{model_size}b.txt"
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


        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": f"{lq}"},
                    {"type": "text", "text": question},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, torch.float16)


        # Add the actual image tensor
        inputs["pixel_values"] = processor.image_processor([lq_img], return_tensors="pt").pixel_values.to(model.device, torch.float16)


        # Generate
        generate_ids = model.generate(**inputs, max_new_tokens=30)
        output = processor.batch_decode(generate_ids, skip_special_tokens=True)
        print('GT TEXT: ', gt_prompt)
        print(output[0])

        file.write(f'GT TEXT: {gt_prompt}\n')
        file.write(f"VLM OUTPUT: {output[0].split('ASSISTANT')[-1][2:]}\n\n")


print('done!')