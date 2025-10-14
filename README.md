

##  ⚒️ Training Preparation

### Environment
```
conda create -n dit4sr python=3.10 -y
conda activate dit4sr
```

### Installation
```
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cd detectron2 
pip install -e .
cd testr 
pip install -e .
```

### Download Pretrained Weights and Dataset

1. Run the bash script `download_weights.sh` to download the pretrained weights for the image restoration module.  
   Additionally, download the pretrained text spotting module from [this link](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xiz102_ucsd_edu/ESwSFxppsplEiEaUphJB0TABkIKoRvIljkVIazPUNEXI7g?e=Q8zJ0Q) and place it in the `./weights` directory.

2. Download the SA-Text dataset using the **Google Drive** link provided above.
   Once downloaded, unzip the contents and place the folder in your working directory.
---

## 🔥 Training Recipe
Our text-aware restoration model, **TeReDiff**, comprises two main modules: an image restoration module and a text spotting module. 
Training is conducted in three stages:
- **Stage 1**: Train only the image restoration module.
- **Stage 2**: Train only the text spotting module.
- **Stage 3**: Jointly train both modules.


### Training Script

- Run the following bash script for **Stage1** training. Its configuration file can be found [here](configs/train/train_stage1_terediff.yaml). Refer to the comments within the configuration file for a detailed explanation of each setting.

```
bash run_script/train_script/run_train_stage1_terediff.sh
```
- Run the following bash script for **Stage2** training. Its configuration file can be found [here](configs/train/train_stage2_terediff.yaml)

```
bash run_script/train_script/run_train_stage2_terediff.sh
```
- Run the following bash script for **Stage3** training. Its configuration file can be found [here](configs/train/train_stage3_terediff.yaml)

```
bash run_script/train_script/run_train_stage3_terediff.sh
```

## 🚀 Text-Aware Image Restoration (TAIR) Demo


### Demo Script

Download the released checkpoint of our model (**TeReDiff**) from [here](https://drive.google.com/drive/folders/1Xn0DaL-3ViXpl1pWHPvcmSejTDoIjAQn?usp=drive_link), and set the appropriate parameters in the demo configuration file [here](configs/val/val_terediff.yaml). Then, run the script below to perform a demo on low-quality images and generate high-quality, text-aware restored outputs. The results will be saved in **val_demo_result/** by default.

```
bash run_script/val_script/run_val_terediff.sh
```

### TAIR Demo Results 
Running the demo script above will generate the following restoration results. The visualized images are shown in the order: **Low-Quality (LQ) image / Restored image / High-Quality (HQ) Ground Truth image**. Note that when the text in the LQ images is severely degraded, the model may fail to accurately restore the textual content due to insufficient visual information.


<p align="center">
  <img src="assets/demo_imgs/restored/restored_sa_922529_crop_0_concat.png" width="800">
</p>
<p align="center">
  <img src="assets/demo_imgs/restored/restored_sa_924654_crop_0_concat.png" width="800">
</p>
<p align="center">
  <img src="assets/demo_imgs/restored/restored_sa_965829_crop_1_concat.png" width="800">
</p>
<p align="center">
  <img src="assets/demo_imgs/restored/restored_sa_991053_crop_0_concat.png" width="800">
</p>


## Citation

If you find our work useful for your research, please consider citing it :)

```
@article{min2025text,
  title={Text-Aware Image Restoration with Diffusion Models},
  author={Min, Jaewon and Kim, Jin Hyeon and Cho, Paul Hyunbin and Lee, Jaeeun and Park, Jihye and Park, Minkyu and Kim, Sangpil and Park, Hyunhee and Kim, Seungryong},
  journal={arXiv preprint arXiv:2506.09993},
  year={2025}
}
```