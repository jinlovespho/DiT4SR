

##  ⚒️ Training Preparation

### 1. Environment
```
conda create -n dit4sr python=3.9 -y
conda activate dit4sr
```

### 2. Installation
```
# Install the libraries in the order listed below.
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install numpy==1.26.3 --no-deps
pip install pyiqa==0.1.14.1 --no-deps 
cd detectron2 
pip install -e . --no-deps --no-build-isolation --config-settings editable_mode=compat
cd testr 
pip install -e . --no-deps --no-build-isolation --config-settings editable_mode=compat
pip install cloudpickle --no-deps
```

### 3. Download Weights and Dataset

#### 3-1. Training Dataset 
- Download the **train/** folder from this [google drive link](https://drive.google.com/drive/folders/15W9OFbUX_rTzVOQLNnyhNUZwDdCsE0Hy?usp=drive_link).

#### 3-2. Pretrained Weights

- SD3 weights
```
# Enter your HF access token to download SD3 weights
huggingface-cli login
bash download_bash/download_sd3.sh 
```

- DiT4SR weights
```
bash download_bash/download_dit4sr.sh 
```

- TESTR weights
```
bash download_bash/download_testr.sh 
```


## 🔥 Training Recipe  

### Training Script  

- Training bash script: [`run_scripts/train/run_train_dit4sr_testr_JIHYE.sh`](run_scripts/train/run_train_dit4sr_testr_JIHYE.sh)  
- Training config file: [`run_configs/train/train_dit4sr_testr_JIHYE.yaml`](run_configs/train/train_dit4sr_testr_JIHYE.yaml)  

#### Example (4 GPUs)  
```bash
# Example script for distributed training on 4 GPUs
CUDA="0,1,2,3"
NUM_GPU=4

CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch --num_processes ${NUM_GPU} train/train_dit4sr.py     --config run_configs/train/train_dit4sr_testr.yaml
```

---

### Example Training Config  

```yaml
data:
  train:
    hq_img_path: /SET/PATH/TO/100K/train
    ann_path: /SET/PATH/TO/100K/train/dataset.json

  val:
    eval_list: []
    realtext:
      lq_img_path:
      hq_img_path:
      ann_path:
      val_num_img:
    satext_lv3:
      lq_img_path:
      hq_img_path:
      ann_path:
      val_num_img:
    satext_lv2:
      lq_img_path:
      hq_img_path:
      ann_path:
      val_num_img:
    satext_lv1:
      lq_img_path:
      hq_img_path:
      ann_path:
      val_num_img:

ckpt:
  init_path:
    vae: preset/models/stable-diffusion-3.5-medium
    noise_scheduler: preset/models/stable-diffusion-3.5-medium
    tokenizer: preset/models/stable-diffusion-3.5-medium
    text_encoder: preset/models/stable-diffusion-3.5-medium
    dit: preset/models/dit4sr/dit4sr_q
    ts_module: preset/models/testr/totaltext_testr_R_50_polygon.pth
  resume_path:
    dit: latest

train:
  batch_size: 1
  num_workers: 4
  num_train_epochs: 300
  mixed_precision: "fp16"
  gradient_checkpointing: True
  gradient_accumulation_steps: 8
  ocr_loss_weight: 0.01

val:
  val_every_step: 500

save:
  output_dir: ./result_train
  checkpointing_steps: 2000

log:
  tracker:
    report_to:
    key:
    project_name:
    server:
    gpu:
    msg: JIHYE_LOG_ADDITIONAL_MSG
  log_dir: logs
```

---

### Run Training  
```bash
bash run_scripts/train/run_train_dit4sr_testr.sh
```
