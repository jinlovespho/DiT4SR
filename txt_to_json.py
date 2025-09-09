import json
import re

model_size=13

# input_file = f"results/satext/lv3/vlm_ocr_result/qwen/{model_size}b.txt"
# output_file = f"results/satext/lv3/vlm_ocr_result/qwen/{model_size}b.json"


input_file = f"results/satext/lv3/vlm_ocr_result/llava/{model_size}b.txt"
output_file = f"results/satext/lv3/vlm_ocr_result/llava/{model_size}b.json"

data = {}

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Split into blocks (skip the first "OCR this image." line if present)
blocks = re.split(r"\n(?=\d+ )", content.strip())
blocks = [b for b in blocks if b and not b.startswith("OCR this image.")]

for block in blocks:
    lines = block.strip().split("\n")
    idx, image_id = lines[0].split(" ", 1)
    gt_text = lines[1].replace("GT TEXT: ", "").strip()
    vlm_output = lines[2].replace("VLM OUTPUT: ", "").strip()
    
    # Use image name as key
    data[image_id] = {
        "gt_text": gt_text,
        "vlm_output": vlm_output
    }

# Save as JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Saved {len(data)} entries to {output_file}")


with open(output_file, 'r', encoding='utf-8') as file:
    anns = json.load(file)

breakpoint()