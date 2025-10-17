import re
import os
import json
import sys
sys.path.append(os.getcwd())
import pyiqa
import time
import glob
import wandb
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

import cv2 
import torch
from torchvision import transforms
from torchvision.utils import save_image 

import test_initialize
from test_utils import encode, decode


logger = get_logger(__name__, log_level="INFO")


# if torch.cuda.device_count() >= 2:
#     LLaVA_device = 'cuda:1'
#     dit4sr_device = 'cuda:0'
# elif torch.cuda.device_count() == 1:
#     LLaVA_device = 'cuda:0'
#     dit4sr_device = 'cuda:0'
# else:
#     raise ValueError('Currently support CUDA only.')


def main(cfg):
    
    
    # set validation experiment name 
    val_data_name = cfg.data.val.name
    assert val_data_name in ['realtext', 'satext_lv3', 'satext_lv2', 'satext_lv1']
    exp_name = cfg.ckpt.resume_path.dit.split('/')[-2]
    num_ckpt = cfg.ckpt.resume_path.dit.split('/')[-1]
    exp_name = f'{exp_name}_{num_ckpt}'
    cfg.exp_name = exp_name

    
    # set accelerator
    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)


    # set val save folder
    if accelerator.is_main_process:
        os.makedirs(cfg.save.output_dir, exist_ok=True)


    # set tracker
    if accelerator.is_main_process:
        wandb.login(key=cfg.log.tracker.key)
        wandb.init(
            project=cfg.log.tracker.project_name,
            name=f'VAL_{val_data_name}_{exp_name}',
            config=OmegaConf.to_container(cfg, resolve=True)
        )


    # load pipeline
    pipeline = test_initialize.load_dit4sr_pipeline(cfg, accelerator)


    breakpoint()

    # # set text condition configuration
    # if cfg.data.val.text_cond_prompt.prompt_type == 'pred':

        
    #     # use vlm captioner
    #     if cfg.data.val.text_cond_prompt.captioner.model == 'vlm':
    #         # use llava
    #         if cfg.data.val.text_cond_prompt.captioner.vlm_model == 'llava':

    #         # use qwen-vl                
    #         elif cfg.data.val.text_cond_prompt.captioner.vlm_model == 'qwenvl':
            
    #         # use intern-vl
    #         elif cfg.data.val.text_cond_prompt.captioner.vlm_model == 'internvl':
        
        
    #     # use tsm captioner
    #     else:
    
    
    # elif cfg.data.val.text_cond_prompt.prompt_type == 'gt':
        
        
    # elif cfg.data.val.text_cond_prompt.prompt_type == 'null':
        
        

    # # prompt selection
    # if cfg.data.val.use_precomputed_prompts is not None:
    #     precom_prompts = sorted(glob.glob(f"{cfg.data.val.use_precomputed_prompts}/*.txt"))
    # else:
    #     if cfg.data.val.captioner is not None:
    #         # load vlm
    #         if cfg.data.val.captioner =='llava' and cfg.data.val.saved_caption_path is None:
    #             from llava.llm_agent import LLavaAgent
    #             from CKPT_PTH import LLAVA_MODEL_PATH
    #             cap_agent = LLavaAgent(LLAVA_MODEL_PATH, LLaVA_device, load_8bit=True, load_4bit=False)
    #         elif cfg.data.val.captioner == 'qwen' and cfg.data.val.saved_caption_path is None:
    #             from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    #             model_size=cfg.data.val.captioner_size
    #             vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(f"Qwen/Qwen2.5-VL-{model_size}B-Instruct", torch_dtype="auto", device_map="auto")
    #             vlm_processor = AutoProcessor.from_pretrained(f"Qwen/Qwen2.5-VL-{model_size}B-Instruct")
    
    
    # load annotations
    val_gt_json = test_initialize.load_json_anns(cfg,)
    
    
    # set seed 
    generator = torch.Generator(device=accelerator.device)
    if cfg.init.seed is not None:
        set_seed(cfg.init.seed)
        generator.manual_seed(cfg.init.seed)


    if os.path.isdir(cfg.data.val.lq_image_path):
        lq_imgs = sorted(glob.glob(f'{cfg.data.val.lq_image_path}/*.jpg'))
    gt_exist=False
    if os.path.isdir(cfg.data.val.gt_image_path):
        gt_exist=True
        gt_imgs = sorted(glob.glob(f'{cfg.data.val.gt_image_path}/*.jpg'))
        assert len(lq_imgs) == len(gt_imgs)
    print(f'Number of LQ testing images: {len(lq_imgs)}')


    # SR metrics
    metric_psnr = pyiqa.create_metric('psnr', device=accelerator.device)
    metric_ssim = pyiqa.create_metric('ssimc', device=accelerator.device)
    metric_lpips = pyiqa.create_metric('lpips', device=accelerator.device)
    metric_dists = pyiqa.create_metric('dists', device=accelerator.device)
    # metric_fid = pyiqa.create_metric('fid', device=device)
    metric_niqe = pyiqa.create_metric('niqe', device=accelerator.device)
    metric_musiq = pyiqa.create_metric('musiq', device=accelerator.device)
    metric_maniqa = pyiqa.create_metric('maniqa', device=accelerator.device)
    metric_clipiqa = pyiqa.create_metric('clipiqa', device=accelerator.device)


    tot_val_psnr=[]
    tot_val_ssim=[]
    tot_val_lpips=[]
    tot_val_dists=[]
    tot_val_niqe=[]
    tot_val_musiq=[]
    tot_val_maniqa=[]
    tot_val_clipiqa=[]


    for image_idx in range(len(lq_imgs)):

        # img id 
        lq_name = lq_imgs[image_idx]
        lq_id = lq_name.split('/')[-1].split('.')[0]
        img_ann = val_gt_json[lq_id]
        gt_prompt = img_ann['gtprompts'][0]

        val_text = img_ann['texts']
        val_polys = img_ann['polys']

        if gt_exist:
            gt_name = gt_imgs[image_idx]
            gt_id = gt_name.split('/')[-1].split('.')[0]
            assert lq_id == gt_id 
            gt_image = Image.open(gt_name).convert("RGB")


        print(f'================== processing {image_idx} img: {lq_id} ===================')

        # read img
        lq_image = Image.open(lq_name).convert("RGB")

        validation_prompt = cfg.data.val.added_prompt # clean, extremely detailed, best quality, sharp, clean
        negative_prompt = cfg.data.val.negative_prompt #dirty, messy, low quality, frames, deformed, 

        if cfg.data.val.use_precomputed_prompts is not None:
            precom_prompt = precom_prompts[image_idx]
            prompt_id = precom_prompt.split('/')[-1].split('.')[0]
            assert lq_id == prompt_id
            with open(f'{precom_prompt}', 'r') as f:
                validation_prompt = f.read().strip()
        else:
            if cfg.data.val.captioner is not None:
                # process prompt 
                if cfg.data.val.captioner == 'llava':
                    validation_prompt = test_initialize.process_llava(cap_agent, lq_image)
                elif cfg.data.val.captioner == 'qwen':
                    if cfg.data.val.saved_caption_path is not None:
                        gt_text, validation_prompt = test_initialize.process_saved_qwen(cfg.data.val.saved_caption_path, cfg.data.val.captioner_size, lq_id)
                    else:
                        validation_prompt = test_initialize.process_qwen(vlm_model, vlm_processor, lq_name)

        if cfg.data.val.use_satext_gtprompt:
            print('Using SAText GT prompt ...')
            validation_prompt = gt_prompt

        if cfg.data.val.use_nullprompt:
            validation_prompt = validation_prompt
        
        # # save prompt
        # if cfg.data.val.save_prompts:
        #     txt_save_path = f"{cfg.save.output_dir}/{cfg.log.tracker.run_name}/txt"
        #     os.makedirs(txt_save_path, exist_ok=True)
        #     file = open(f'{txt_save_path}/{lq_id}.txt', "w")
        #     file.write(validation_prompt)
        #     file.close()
        # print(f'{validation_prompt}')

        ori_width, ori_height = lq_image.size
        resize_flag = False
        rscale = cfg.data.val.upscale
        if ori_width < cfg.data.val.process_size//rscale or ori_height < cfg.data.val.process_size//rscale: # f
            scale = (cfg.data.val.process_size//rscale)/min(ori_width, ori_height)
            tmp_image = lq_image.resize((int(scale*ori_width), int(scale*ori_height)),Image.BICUBIC)

            lq_image = tmp_image
            resize_flag = True

        lq_image = lq_image.resize((lq_image.size[0]*rscale, lq_image.size[1]*rscale), Image.BICUBIC)
        lq_image = lq_image.resize((lq_image.size[0]//8*8, lq_image.size[1]//8*8), Image.BICUBIC)
        width, height = lq_image.size
        resize_flag = True #

        print(f'input size: {height}x{width}')

        # for sample_idx in range(cfg.data.val.sample_times):
        #     os.makedirs(f'{cfg.save.output_dir}/sample{str(sample_idx).zfill(2)}/', exist_ok=True)

        for sample_idx in range(cfg.data.val.sample_times):  
            with torch.autocast("cuda"):
                start_time = time.time()
                val_out = pipeline(
                        prompt=validation_prompt, control_image=lq_image, num_inference_steps=cfg.data.val.num_inference_steps, generator=generator, height=height, width=width,
                        guidance_scale=cfg.data.val.guidance_scale, negative_prompt=negative_prompt,
                        start_point=cfg.data.val.start_point, latent_tiled_size=cfg.data.val.latent_tiled_size, latent_tiled_overlap=cfg.data.val.latent_tiled_overlap,
                        output_type = 'pt', return_dict=False, lq_id=lq_id, val_data_name=val_data_name, cfg=cfg, mode='val'
                    )
                val_restored_img = val_out[0]
                if len(val_out) > 1:
                    val_ocr_result = val_out[1]
                end_time = time.time()
                print(f'inference time: {end_time-start_time:.2f}s')
            
            # Save only the restored image
            res_save_path = f'{cfg.save.output_dir}/{val_data_name}/{exp_name}/final_restored_img'
            os.makedirs(res_save_path, exist_ok=True)
            save_image(val_restored_img, f'{res_save_path}/{lq_id}.png')
            

            # lq 
            val_lq_img = transforms.ToTensor()((lq_image)).unsqueeze(dim=0).cuda()  # [0,1]
            # val_lq_img = (val_lq_img + 1.0) / 2.0
            # gt 
            val_gt_img = transforms.ToTensor()((gt_image)).unsqueeze(dim=0).cuda()  # [0,1]
            # val_gt_img = (val_gt_img + 1.0) / 2.0
            # restored
            val_res_img = val_restored_img.squeeze(dim=0).permute(1,2,0).cpu().detach().numpy()*255.0
            val_res_img = val_res_img.astype(np.uint8)
            # val_img = torch.concat([val_lq_img, val_restored_img, val_gt_img], dim=3)
            # save_image(val_img, f'{val_save_path}/step{global_step}_valimg{val_step}_{val_lq_id[val_idx]}.png')
            # save_image(val_img, f'{val_save_path}/restored_{lq_id}.png')
            # save_image(val_lq_img, f'{val_save_path}/step{global_step}_{val_lq_id[val_idx]}.png', normalize=True)
            # save_image(val_res_img, f'{val_save_path}/step{global_step}_{val_lq_id[val_idx]}.png', normalize=True)
            
            # log total psnr, ssim, lpips for val
            tot_val_psnr.append(torch.mean(metric_psnr(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())
            tot_val_ssim.append(torch.mean(metric_ssim(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())
            tot_val_lpips.append(torch.mean(metric_lpips(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())
            tot_val_dists.append(torch.mean(metric_dists(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())
            tot_val_niqe.append(torch.mean(metric_niqe(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())
            tot_val_musiq.append(torch.mean(metric_musiq(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())
            tot_val_maniqa.append(torch.mean(metric_maniqa(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())
            tot_val_clipiqa.append(torch.mean(metric_clipiqa(val_restored_img.to(torch.float32), val_gt_img.to(torch.float32))).item())

            # log sampling val imgs to wandb
            if accelerator.is_main_process and cfg.log.tracker.report_to == 'wandb':
                wandb.log({f'val_metric/val_psnr': torch.mean(metric_psnr(
                                                                                val_restored_img.to(torch.float32), 
                                                                                val_gt_img.to(torch.float32))).item(),
                        f'val_metric/val_ssim': torch.mean(metric_ssim(
                                                                                val_restored_img.to(torch.float32), 
                                                                                val_gt_img.to(torch.float32))).item(),
                        f'val_metric/val_lpips': torch.mean(metric_lpips(
                                                                                val_restored_img.to(torch.float32), 
                                                                                val_gt_img.to(torch.float32))).item(),
                        f'val_metric/val_dists': torch.mean(metric_dists(
                                                                                val_restored_img.to(torch.float32), 
                                                                                val_gt_img.to(torch.float32))).item(),
                        f'val_metric/val_niqe': torch.mean(metric_niqe(
                                                                                val_restored_img.to(torch.float32), 
                                                                                val_gt_img.to(torch.float32))).item(),
                        f'val_metric/val_musiq': torch.mean(metric_musiq(
                                                                                val_restored_img.to(torch.float32), 
                                                                                val_gt_img.to(torch.float32))).item(),
                        f'val_metric/val_maniqa': torch.mean(metric_maniqa(
                                                                                val_restored_img.to(torch.float32), 
                                                                                val_gt_img.to(torch.float32))).item(),
                        f'val_metric/val_clipiqa': torch.mean(metric_clipiqa(
                                                                                val_restored_img.to(torch.float32), 
                                                                                val_gt_img.to(torch.float32))).item(),
                        })


            if 'testr' in cfg.train.model:
                ## -------------------- vis val ocr results -------------------- 
                print('== logging val ocr results ===')
                val_save_path = f'{cfg.save.output_dir}/{val_data_name}/{exp_name}/final_result'
                os.makedirs(val_save_path, exist_ok=True)

                img_lq = val_lq_img.detach().permute(0,2,3,1).cpu().numpy() # b h w c
                img_gt = val_gt_img.detach().permute(0,2,3,1).cpu().numpy()

                for vis_batch_idx in range(len(img_gt)):
                    vis_lq = img_lq[vis_batch_idx] # h w c 
                    # vis_lq = (vis_lq + 1.0)/2.0 * 255.0
                    vis_lq = vis_lq * 255.0
                    vis_lq = vis_lq.astype(np.uint8)
                    vis_lq = vis_lq.copy()

                    vis_gt = img_gt[vis_batch_idx] # h w c
                    # vis_gt = (vis_gt + 1.0)/2.0 * 255.0
                    vis_gt = vis_gt * 255.0
                    vis_gt = vis_gt.astype(np.uint8)
                    vis_pred = vis_lq.copy()
                    vis_gt = vis_gt.copy()
                    vis_gt2 = vis_gt.copy()

                    ocr_res = val_ocr_result[vis_batch_idx]
                    vis_polys = ocr_res.polygons.view(-1,16,2)  # b 16 2
                    vis_recs = ocr_res.recs                     # b 25
                    for vis_img_idx in range(len(vis_polys)):
                        pred_poly = vis_polys[vis_img_idx]   # 16 2
                        pred_poly = np.array(pred_poly.detach().cpu()).astype(np.int32)         
                        pred_rec = vis_recs[vis_img_idx]     # 25
                        pred_txt = decode(pred_rec.tolist())
                        cv2.polylines(vis_pred, [pred_poly], isClosed=True, color=(0,255,0), thickness=2)
                        cv2.putText(vis_pred, pred_txt, (pred_poly[0][0], pred_poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    # cv2.imwrite(f'{val_ocr_save_path}/ocr_pred_{lq_id}.jpg.jpg', vis_pred[:,:,::-1])

                    if gt_exist:
                        gt_polys = val_polys
                        gt_texts = val_text
                        for vis_img_idx in range(len(gt_polys)):
                            gt_poly = gt_polys[vis_img_idx]*512.0   # 16 2
                            gt_poly = np.array(gt_poly).astype(np.int32)
                            gt_txt = gt_texts[vis_img_idx]
                            cv2.polylines(vis_gt, [gt_poly], isClosed=True, color=(0,255,0), thickness=2)
                            cv2.putText(vis_gt, gt_txt, (gt_poly[0][0], gt_poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        vis_result = cv2.hconcat([vis_lq, val_res_img, vis_gt2, vis_pred, vis_gt])
                        cv2.imwrite(f'{val_save_path}/{lq_id}.jpg', vis_result[:,:,::-1])
                ## -------------------- vis val ocr results -------------------- 
    
    # average using numpy
    tot_val_psnr = np.array(tot_val_psnr).mean()
    tot_val_ssim = np.array(tot_val_ssim).mean()
    tot_val_lpips = np.array(tot_val_lpips).mean()
    tot_val_dists = np.array(tot_val_dists).mean()
    tot_val_niqe = np.array(tot_val_niqe).mean()
    tot_val_musiq = np.array(tot_val_musiq).mean()
    tot_val_maniqa = np.array(tot_val_maniqa).mean()
    tot_val_clipiqa = np.array(tot_val_clipiqa).mean()

    # log total val metrics 
    if accelerator.is_main_process and cfg.log.tracker.report_to == 'wandb':
        wandb.log({
            f'val_metric/tot_val_psnr': tot_val_psnr,
            f'val_metric/tot_val_ssim': tot_val_ssim,
            f'val_metric/tot_val_lpips': tot_val_lpips,
            f'val_metric/tot_val_dists': tot_val_dists,
            f'val_metric/tot_val_niqe': tot_val_niqe,
            f'val_metric/tot_val_musiq': tot_val_musiq,
            f'val_metric/tot_val_maniqa': tot_val_maniqa,
            f'val_metric/tot_val_clipiqa': tot_val_clipiqa,
        })
            
            # if cfg.data.val.align_method == 'nofix':
            #     image = val_restored_img
            # else:
            #     if cfg.data.val.align_method == 'wavelet':
            #         image = wavelet_color_fix(image, lq_image)
            #     elif cfg.data.val.align_method == 'adain':
            #         image = adain_color_fix(image, lq_image)

            # if resize_flag: 
            #     image = image.resize((ori_width*rscale, ori_height*rscale), Image.BICUBIC)
                
            # name, ext = os.path.splitext(os.path.basename(image_name))
            
            # image.save(f'{cfg.save.output_dir}/sample{str(sample_idx).zfill(2)}/{name}.png')
            # # image.save(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}_{name}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)



