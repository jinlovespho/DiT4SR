import os
import re

# Folder containing your .txt files
input_folder = "result_val/satext_lv3/orig_dit4sr_baseline_noise/txt"
output_folder = os.path.join(input_folder, "cleaned")

# Create output folder if not exist
os.makedirs(output_folder, exist_ok=True)

# Keywords indicating where the descriptive sentence ends
cut_keywords = ["Cinematic", "hyper sharpness", "highly detailed", "ultra HD", "Color Grading"]

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        in_path = os.path.join(input_folder, filename)
        out_path = os.path.join(output_folder, filename)

        with open(in_path, "r") as f:
            text = f.read().strip()

        # Split at the first stylistic keyword
        pattern = r'\b(' + '|'.join(map(re.escape, cut_keywords)) + r')\b'
        clean_text = re.split(pattern, text, maxsplit=1)[0].strip()

        # Ensure clean ending
        if not clean_text.endswith('.'):
            clean_text += '.'

        # Save cleaned text
        with open(out_path, "w") as f:
            f.write(clean_text)

        print(f"✅ Cleaned: {filename}")

print("🎯 All text files cleaned and saved to:", output_folder)
