

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
Before training, modify the training script and training config.
- training script -> for setting cuda and GPU
- training config -> for setting data_path, training_batch_size, etc

### Stage1 Training

 - Stage1 training script: [JIHYE_train_stage1_dit4sr.sh](run_scripts/train/JIHYE_train_stage1_dit4sr.sh)
 - Stage1 training config: [JIHYE_train_stage1_dit4sr.yaml](run_configs/train/JIHYE_train_stage1_dit4sr.yaml)

```bash
# stage1 training 
bash run_scripts/train/JIHYE_train_stage1_dit4sr.sh
```

### Stage2 Training
- Stage2 training script: [JIHYE_train_stage2_testr.sh](run_scripts/train/JIHYE_train_stage2_testr.sh)
- Stage2 training config: [JIHYE_train_stage2_testr.yaml](run_configs/train/JIHYE_train_stage2_testr.yaml)

```bash
# stage2 training 
bash run_scripts/train/JIHYE_train_stage2_testr.sh
```

 ### Stage3 Training
 - Stage3 training script: [JIHYE_train_stage3_dit4sr_testr.sh](run_scripts/train/JIHYE_train_stage3_dit4sr_testr.sh)
 - Stage3 training config: [JIHYE_train_stage3_dit4sr_testr.yaml](run_configs/train/JIHYE_train_stage3_dit4sr_testr.yaml)

```bash
# stage3 training 
bash run_scripts/train/JIHYE_train_stage3_dit4sr_testr.sh
```
