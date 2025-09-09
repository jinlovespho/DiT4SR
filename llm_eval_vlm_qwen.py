import os 
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-14B-AWQ"



# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

model_sizes = [72, 32, 7, 3]

for model_size in model_sizes:

    count_one=0
    count_two=0
    count_three=0

    txt_path = f'./results/satext/lv3/vlm_ocr_result/qwen/{model_size}b.txt'
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    vlm_result = []

    # Parse the file
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and line[0].isdigit():  # Detect new entry (starts with an index)
            entry = {}
            entry['id'] = line.split()[1]  # e.g., sa_827563_crop_0
            # Next line: GT TEXT
            i += 1
            entry['gt_text'] = lines[i].strip().replace("GT TEXT: ", "")
            # Next line: VLM OUTPUT
            i += 1
            entry['vlm_output'] = lines[i].strip().replace("VLM OUTPUT: ", "")
            vlm_result.append(entry)
        i += 1


    print(f'-- EVALUATING FOR {len(vlm_result)} IMGS -- ')

    for idx, res in enumerate(vlm_result):
        img_id = res['id']
        gt_text = res['gt_text']
        vlm_output = res['vlm_output']


        # prepare the model input
        prompt = f"""
        Ground truth text: "{gt_text}"
        VLM OCR output: "{vlm_output}"

        Step 1: Analyze the VLM OCR output and identify the text it extracted from the entire VLM output.
        Step 2: Compare the extracted text with the ground truth.
        Step 3: If you cannot identify the extracted text from the VLM OCR output, classify it as Incorrect.

        Categories:
        1 — Correct: the OCR output exactly matches the ground truth.
        2 — Slightly correct: minor differences (typos, extra/missing spaces) but mostly correct.
        3 — Incorrect: largely wrong, does not match, is empty, or the text cannot be identified.

        Answer with only the category number (1, 2, or 3).
        """
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        print("thinking content:", thinking_content)
        print("content:", content)

        try:
            decision = int(content)
        except ValueError:
            decision = 3  # default to Incorrect if output isn't a number
        if decision == 1:
            count_one += 1 
        elif decision == 2:
            count_two +=1 
        elif decision == 3:
            count_three +=1 

        print(count_one)
        print(count_two)
        print(count_three)

        statistic_path = f'results/satext/lv3/vlm_ocr_result/statistic/qwen_{model_size}b'
        os.makedirs(statistic_path, exist_ok=True)
        with open(f'{statistic_path}/{img_id}.txt', 'w') as file:
            file.write(f'{idx} img id: {img_id}\n\n')
            file.write(f'gt_text: {gt_text}\n')
            file.write(f'vlm_output: {vlm_output}\n\n')
            file.write(f'----------------------------\n\n')
            file.write(f'{thinking_content}\n')
            file.write(f'LLM classification result: {content}\n')
            file.write(f'1. Correct Num: {count_one}\n')
            file.write(f'2. Slightly Correct Num: {count_two}\n')
            file.write(f'3. Incorrect Num: {count_three}\n')

        # breakpoint()



    total = len(vlm_result)
    p1 = count_one / total * 100
    p2 = count_two / total * 100
    p3 = count_three / total * 100

    final_stat_path = f'{statistic_path}/final_statistic.txt'
    with open(f'{statistic_path}/final_stat_qwen_{model_size}b.txt', 'w') as file:
        file.write("=== VLM LQ OCR Evaluation Statistics ===\n\n")
        file.write(f'Qwen2.5 VL ({model_size}b) LQ OCR Result using Qwen3(14b)\n')
        file.write(f"Total images processed: {total}\n\n")

        file.write(f"1. Correct Num: {count_one} ({p1:.2f}%)\n")
        file.write(f"2. Slightly Correct Num: {count_two} ({p2:.2f}%)\n")
        file.write(f"3. Incorrect Num: {count_three} ({p3:.2f}%)\n\n")

        file.write("=== Accuracy Metrics ===\n")
        file.write(f"Exact Accuracy: {p1:.2f}%\n")
        file.write(f"Lenient Accuracy (Correct + Slightly Correct): {(p1+p2):.2f}%\n")
        file.write(f"Incorrect Accuracy: {p3:.2f}%\n")


    print(f'VLM LQ OCR RESULT for Qwen2.5VL({model_size}b)')
    print(count_one)
    print(count_two)
    print(count_three)


print(f'ALL DONE!')
breakpoint()
