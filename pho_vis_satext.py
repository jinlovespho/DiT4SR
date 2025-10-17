import glob
import os
import cv2

dataset = 'satext_lv2'

def load_images(path_pattern):
    files = sorted(glob.glob(path_pattern))
    return {os.path.splitext(os.path.basename(f))[0]: f for f in files}

# Load as dictionaries keyed by image ID
dit_baseline_testr_1e4 = load_images(
    f'result_train/stage2/fp16_stage2_testr_1e-04__ocrloss0.01_descriptive_DiTfeat24_dit4sr_baseline_testr_pretrained/{dataset}/final_result/*.jpg')
dit_baseline_testr_1e5 = load_images(
    f'result_train/stage2/fp16_stage2_testr_1e-05__ocrloss0.01_descriptive_DiTfeat24_dit4sr_baseline_testr_pretrained/{dataset}/final_result/*.jpg')
dit_trained_testr_1e4 = load_images(
    f'result_train/stage2/fp16_stage2_testr_1e-04__ocrloss0.01_descriptive_DiTfeat24_dit4sr_lr1e5_ckpt18k_testr_pretrained/{dataset}/final_result/*.jpg')
dit_trained_testr_1e5 = load_images(
    f'result_train/stage2/fp16_stage2_testr_1e-05__ocrloss0.01_descriptive_DiTfeat24_dit4sr_lr1e5_ckpt18k_testr_pretrained/{dataset}/final_result/*.jpg')


# Common IDs across all 4 sets
common_ids = sorted(set(dit_baseline_testr_1e4.keys())
                    & set(dit_baseline_testr_1e5.keys())
                    & set(dit_trained_testr_1e4.keys())
                    & set(dit_trained_testr_1e5.keys()))

save_dir = f'./vis/{dataset}/stage2_ocr_comparison'
os.makedirs(save_dir, exist_ok=True)

for img_id in common_ids:
    tmp1 = cv2.imread(dit_baseline_testr_1e4[img_id])
    tmp2 = cv2.imread(dit_baseline_testr_1e5[img_id])
    tmp3 = cv2.imread(dit_trained_testr_1e4[img_id])
    tmp4 = cv2.imread(dit_trained_testr_1e5[img_id])
    
    img = cv2.vconcat([tmp1, tmp2, tmp3, tmp4])
    cv2.imwrite(f'{save_dir}/{img_id}.jpg', img)

print('FINISH!')



dataset = 'satext_lv1'

def load_images(path_pattern):
    files = sorted(glob.glob(path_pattern))
    return {os.path.splitext(os.path.basename(f))[0]: f for f in files}

# Load as dictionaries keyed by image ID
dit_baseline_testr_1e4 = load_images(
    f'result_train/stage2/fp16_stage2_testr_1e-04__ocrloss0.01_descriptive_DiTfeat24_dit4sr_baseline_testr_pretrained/{dataset}/final_result/*.jpg')
dit_baseline_testr_1e5 = load_images(
    f'result_train/stage2/fp16_stage2_testr_1e-05__ocrloss0.01_descriptive_DiTfeat24_dit4sr_baseline_testr_pretrained/{dataset}/final_result/*.jpg')
dit_trained_testr_1e4 = load_images(
    f'result_train/stage2/fp16_stage2_testr_1e-04__ocrloss0.01_descriptive_DiTfeat24_dit4sr_lr1e5_ckpt18k_testr_pretrained/{dataset}/final_result/*.jpg')
dit_trained_testr_1e5 = load_images(
    f'result_train/stage2/fp16_stage2_testr_1e-05__ocrloss0.01_descriptive_DiTfeat24_dit4sr_lr1e5_ckpt18k_testr_pretrained/{dataset}/final_result/*.jpg')


# Common IDs across all 4 sets
common_ids = sorted(set(dit_baseline_testr_1e4.keys())
                    & set(dit_baseline_testr_1e5.keys())
                    & set(dit_trained_testr_1e4.keys())
                    & set(dit_trained_testr_1e5.keys()))

save_dir = f'./vis/{dataset}/stage2_ocr_comparison'
os.makedirs(save_dir, exist_ok=True)

for img_id in common_ids:
    tmp1 = cv2.imread(dit_baseline_testr_1e4[img_id])
    tmp2 = cv2.imread(dit_baseline_testr_1e5[img_id])
    tmp3 = cv2.imread(dit_trained_testr_1e4[img_id])
    tmp4 = cv2.imread(dit_trained_testr_1e5[img_id])
    
    img = cv2.vconcat([tmp1, tmp2, tmp3, tmp4])
    cv2.imwrite(f'{save_dir}/{img_id}.jpg', img)

print('FINISH!')
