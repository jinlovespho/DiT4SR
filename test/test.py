
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.getcwd())

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

import torch
import torchvision.transforms as T 
import torch.nn.functional as F 
from torchvision.utils import save_image

import cv2 
import wandb
import pyiqa
import argparse
import numpy as np 
from PIL import Image 
from omegaconf import OmegaConf

import initialize 
from dataloaders.utils import encode, decode
from pipelines.pipeline_dit4sr import StableDiffusion3ControlNetPipeline

from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix


logger = get_logger(__name__)


def main(cfg):
    
    
    # global reproducability
    if cfg.init.seed is not None:
        set_seed(cfg.init.seed)  # seeds Python, NumPy, and PyTorch globally
    
    
    # safety check 
    val_data_name = cfg.data.val.eval_list[0]
    assert val_data_name in ['realtext', 'satext_lv3', 'satext_lv2', 'satext_lv1']
    
    
    
    # set experiment name
    if cfg.ckpt.resume_path.dit is not None:
        exp_name = cfg.ckpt.resume_path.dit.split('/')[-2]        
    else:
        exp_name = f'dit4sr_baseline'
        
        
    exp_name = f'{exp_name}__startpoint-{cfg.data.val.start_point}__alignmethod-{cfg.data.val.align_method}__cfg-{int(cfg.data.val.guidance_scale)}'
    
    
    if cfg.data.val.text_cond_prompt == 'pred_vlm':
        
        if cfg.data.val.eval_list[0] == 'realtext':
            vlm_captioner = cfg.data.val.realtext.vlm_captioner
            vlm_input_ques = cfg.data.val.realtext.vlm_input_ques
        elif cfg.data.val.eval_list[0] == 'satext_lv3':
            vlm_captioner = cfg.data.val.satext_lv3.vlm_captioner
            vlm_input_ques = cfg.data.val.satext_lv3.vlm_input_ques
        elif cfg.data.val.eval_list[0] == 'satext_lv2':
            vlm_captioner = cfg.data.val.satext_lv2.vlm_captioner
            vlm_input_ques = cfg.data.val.satext_lv2.vlm_input_ques
        elif cfg.data.val.eval_list[0] == 'satext_lv1':
            vlm_captioner = cfg.data.val.satext_lv1.vlm_captioner
            vlm_input_ques = cfg.data.val.satext_lv1.vlm_input_ques
            
        cfg.vlm_captioner = vlm_captioner
        cfg.vlm_input_ques_num = vlm_input_ques
        
        # english focused input prompt
        question_list = [
            'Please describe the actual objects in the image in a very detailed manner. Please do not include descriptions related to the focus and bokeh of this image. Please do not include descriptions like the background is blurred.',
            "OCR this image and transcribe only the English text.",
            "Read and transcribe all English text visible in this low-resolution image.",
            "Describe the contents of this blurry image, focusing only on any visible English text or characters.",
            "Extract all visible English words and letters from this low-quality image, even if they appear unclear.",
        ]
        
        cfg.vlm_input_ques = question_list[vlm_input_ques]
        exp_name = f'{exp_name}__{cfg.data.val.text_cond_prompt}-prompt__{vlm_captioner}__ques-{str(vlm_input_ques)}'
    
    else:
        exp_name = f'{exp_name}__{cfg.data.val.text_cond_prompt}-prompt'
    
    
    exp_name = f'{exp_name}__{cfg.log.tracker.msg}'
    cfg.exp_name = exp_name
    print('- EXP NAME: ', exp_name)

    
    # set val save directory 
    os.makedirs(f'{cfg.save.output_dir}/{val_data_name}/{exp_name}', exist_ok=True)
    
    
    # set accelerator and wandb 
    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)
    if accelerator.is_main_process and cfg.log.tracker.report_to == 'wandb':
        wandb.login(key=cfg.log.tracker.key)
        wandb.init(
            project=cfg.log.tracker.project_name,
            name=f'VAL__{cfg.save.output_dir.split("/")[-1]}__{val_data_name}__serv{str(cfg.log.tracker.server)}gpu{str(cfg.log.tracker.gpu)}__{exp_name}',
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    

    # load val data
    _, val_datasets = initialize.load_data(cfg)

    
    # load models 
    models = initialize.load_model(cfg, accelerator)
    

    # place models on cuda and proper weight dtype(float32, float16)
    weight_dtype = initialize.set_model_device(cfg, accelerator, models)


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


    # load tsm
    if 'testr' in cfg.train.model:
        ts_module = models['testr'] 
    else:
        ts_module = None 
        
    # load validation pipeline
    val_pipeline = StableDiffusion3ControlNetPipeline(
        vae=models['vae'], text_encoder=models['text_encoders'][0], text_encoder_2=models['text_encoders'][1], text_encoder_3=models['text_encoders'][2], 
        tokenizer=models['tokenizers'][0], tokenizer_2=models['tokenizers'][1], tokenizer_3=models['tokenizers'][2], 
        transformer=models['transformer'], scheduler=models['noise_scheduler'], ts_module=ts_module,
    )


    metrics={}


    # val loop
    for val_data_name, val_data in val_datasets.items():

        metrics[f'{val_data_name}_psnr'] = []
        metrics[f'{val_data_name}_ssim'] = []
        metrics[f'{val_data_name}_lpips'] = []
        metrics[f'{val_data_name}_dists'] = []
        metrics[f'{val_data_name}_niqe'] = []
        metrics[f'{val_data_name}_musiq'] = []
        metrics[f'{val_data_name}_maniqa'] = []
        metrics[f'{val_data_name}_clipiqa'] = []
        
        # min max normalized results 
        metrics[f'{val_data_name}_psnr_norm'] = []
        metrics[f'{val_data_name}_ssim_norm'] = []
        metrics[f'{val_data_name}_lpips_norm'] = []
        metrics[f'{val_data_name}_dists_norm'] = []
        metrics[f'{val_data_name}_niqe_norm'] = []
        metrics[f'{val_data_name}_musiq_norm'] = []
        metrics[f'{val_data_name}_maniqa_norm'] = []
        metrics[f'{val_data_name}_clipiqa_norm'] = []


        for sample_idx, val_sample in enumerate(val_data):
            
            print('-------------------------------------------------')
            print(f'{val_data_name} - {sample_idx+1}/{len(val_data)}')
            
            generator = None
            if accelerator.is_main_process and cfg.init.seed is not None:
                generator = torch.Generator(device=accelerator.device)
                generator.manual_seed(cfg.init.seed)


            # load val anns
            val_lq_path = val_sample['lq_path']
            val_hq_path = val_sample['hq_path']
            val_gt_text = val_sample['text']
            val_bbox = val_sample['bbox']
            val_polys = val_sample['poly']
            val_img_id = val_sample['img_id']
            val_vlm_cap = val_sample['vlm_cap']
            
            
            # process lq image 
            val_lq_pil = Image.open(val_lq_path).convert("RGB") # 128 128 
            ori_width, ori_height = val_lq_pil.size
            rscale = 4  # upscale x4
            # for shortest side smaller than 128, resize
            if ori_width < 512//rscale or ori_height < 512//rscale:
                scale = (512//rscale)/min(ori_width, ori_height)
                tmp_image = val_lq_pil.resize((int(scale*ori_width), int(scale*ori_height)),Image.BICUBIC)
                val_lq_pil = tmp_image
            val_lq_pil = val_lq_pil.resize((val_lq_pil.size[0]*rscale, val_lq_pil.size[1]*rscale), Image.BICUBIC)
            val_lq_pil = val_lq_pil.resize((val_lq_pil.size[0]//8*8, val_lq_pil.size[1]//8*8), Image.BICUBIC)
            

            
            # -------------------------------------------------
            #       input prompt to diffusion model
            # -------------------------------------------------
            
            # gt prompt 
            if cfg.data.val.text_cond_prompt == 'gt':
                texts = [f'"{t}"' for t in val_gt_text]
                if cfg.model.dit.text_condition.caption_style == 'descriptive':
                    val_init_prompt = [f'The image features the texts {", ".join(texts)} that appear clearly on signs, boards, buildings, or other objects.']
                elif cfg.model.dit.text_condition.caption_style == 'tag':
                    val_init_prompt = [f"{', '.join(texts)}"]

            # text spotting module prompt
            elif cfg.data.val.text_cond_prompt == 'pred_tsm':
                val_init_prompt = ['']
                
            # vlm prompt 
            elif cfg.data.val.text_cond_prompt == 'pred_vlm':
                val_init_prompt = [val_vlm_cap]
                
            # null prompt 
            elif cfg.data.val.text_cond_prompt == 'null':
                val_init_prompt = ['']
            
            # added prompt             
            if cfg.data.val.added_prompt is not None:
                val_init_prompt = [f'{val_init_prompt[0]} {cfg.data.val.added_prompt}']

            # cfg negative prompt 
            if cfg.data.val.negative_prompt is not None:
                neg_prompt = cfg.data.val.negative_prompt
            else:
                neg_prompt = None 
                
                
                
                
        
            # -------------------------------------------------
            #       validation pipeline forward pass 
            # -------------------------------------------------
            with torch.no_grad():
                val_out = val_pipeline(
                    prompt=val_init_prompt[0], control_image=val_lq_pil, num_inference_steps=cfg.data.val.num_inference_steps, generator=generator, height=512, width=512,
                    guidance_scale=cfg.data.val.guidance_scale, negative_prompt=neg_prompt,
                    start_point=cfg.data.val.start_point, latent_tiled_size=cfg.data.val.latent_tiled_size, latent_tiled_overlap=cfg.data.val.latent_tiled_overlap,
                    output_type = 'pil', return_dict=False, lq_id=val_img_id, val_data_name=val_data_name, cfg=cfg, mode='val'
                )
            
            
            
            
            # retrieve restored image 
            val_res_pil = val_out[0][0]   # 1 3 512 512 [0,1]
            
            
            # post processing
            if cfg.data.val.align_method is not None:                
                if cfg.data.val.align_method == 'wavelet':
                    val_res_pil = wavelet_color_fix(val_res_pil, val_lq_pil)
                elif cfg.data.val.align_method == 'adain':
                    val_res_pil = adain_color_fix(val_res_pil, val_lq_pil)
            

            # Save only the restored image
            val_res_save_path = f'{cfg.save.output_dir}/{val_data_name}/{exp_name}/final_restored_img'
            os.makedirs(val_res_save_path, exist_ok=True)
            val_res_pil.save(f'{val_res_save_path}/{val_img_id}.png')
            

            
            
            
            # ---------------------------------------
            #       image metric calculation
            # ---------------------------------------
            
            # restored pil img -> tensor 
            val_res_pt = T.ToTensor()(val_res_pil)
            val_res_pt = val_res_pt.to(device=accelerator.device, dtype=torch.float32).unsqueeze(dim=0).clamp(0.0, 1.0)   # 1 3 512 512 
            
            # gt pil img -> tensor 
            val_hq_pil = Image.open(val_hq_path).convert("RGB") 
            val_hq_pt = T.ToTensor()(val_hq_pil)
            val_hq_pt = val_hq_pt.to(device=accelerator.device, dtype=torch.float32).unsqueeze(dim=0).clamp(0.0, 1.0)   # 1 3 512 512 
            
            # lq pil img -> tensor 
            val_lq_pt = T.ToTensor()(val_lq_pil)
            val_lq_pt = val_lq_pt.to(device=accelerator.device, dtype=torch.float32).unsqueeze(dim=0).clamp(0.0, 1.0)   # 1 3 512 512 
            
            
            # append metric result
            metrics[f'{val_data_name}_psnr'].append(torch.mean(metric_psnr(val_res_pt, val_hq_pt)).item())
            metrics[f'{val_data_name}_ssim'].append(torch.mean(metric_ssim(val_res_pt, val_hq_pt)).item())
            metrics[f'{val_data_name}_lpips'].append(torch.mean(metric_lpips(val_res_pt, val_hq_pt)).item())
            metrics[f'{val_data_name}_dists'].append(torch.mean(metric_dists(val_res_pt, val_hq_pt)).item())
            metrics[f'{val_data_name}_niqe'].append(torch.mean(metric_niqe(val_res_pt, val_hq_pt)).item())
            metrics[f'{val_data_name}_musiq'].append(torch.mean(metric_musiq(val_res_pt, val_hq_pt)).item())
            metrics[f'{val_data_name}_maniqa'].append(torch.mean(metric_maniqa(val_res_pt, val_hq_pt)).item())
            metrics[f'{val_data_name}_clipiqa'].append(torch.mean(metric_clipiqa(val_res_pt, val_hq_pt)).item())
            
            
            # min max normalization 
            val_res_pt_norm = (val_res_pt-val_res_pt.min())/(val_res_pt.max()-val_res_pt.min())
            val_hq_pt_norm = (val_hq_pt-val_hq_pt.min())/(val_hq_pt.max()-val_hq_pt.min())
            val_lq_pt_norm = (val_lq_pt-val_lq_pt.min())/(val_lq_pt.max()-val_lq_pt.min())
            
            
            # append metric result
            metrics[f'{val_data_name}_psnr_norm'].append(torch.mean(metric_psnr(val_res_pt_norm, val_hq_pt_norm)).item())
            metrics[f'{val_data_name}_ssim_norm'].append(torch.mean(metric_ssim(val_res_pt_norm, val_hq_pt_norm)).item())
            metrics[f'{val_data_name}_lpips_norm'].append(torch.mean(metric_lpips(val_res_pt_norm, val_hq_pt_norm)).item())
            metrics[f'{val_data_name}_dists_norm'].append(torch.mean(metric_dists(val_res_pt_norm, val_hq_pt_norm)).item())
            metrics[f'{val_data_name}_niqe_norm'].append(torch.mean(metric_niqe(val_res_pt_norm, val_hq_pt_norm)).item())
            metrics[f'{val_data_name}_musiq_norm'].append(torch.mean(metric_musiq(val_res_pt_norm, val_hq_pt_norm)).item())
            metrics[f'{val_data_name}_maniqa_norm'].append(torch.mean(metric_maniqa(val_res_pt_norm, val_hq_pt_norm)).item())
            metrics[f'{val_data_name}_clipiqa_norm'].append(torch.mean(metric_clipiqa(val_res_pt_norm, val_hq_pt_norm)).item())
            


            # --------------------------------------------------
            #       restoration and OCR visualization 
            # --------------------------------------------------

            # save path
            val_res_ocr_save_path = f'{cfg.save.output_dir}/{val_data_name}/{exp_name}/final_result'
            os.makedirs(val_res_ocr_save_path, exist_ok=True)
            
            
            # hq img: pt -> np
            vis_hq = val_hq_pt.squeeze(dim=0).permute(1,2,0).detach().cpu().numpy() * 255.0
            vis_hq = vis_hq.astype(np.uint8)
            vis_hq1 = vis_hq.copy()
            vis_hq2 = vis_hq.copy()
            
            # restored img: pt -> np
            vis_res = val_res_pt.squeeze(dim=0).permute(1,2,0).detach().cpu().numpy() * 255.0
            vis_res = vis_res.astype(np.uint8)
            
            # lq img: pt -> np
            vis_lq = val_lq_pt.squeeze(dim=0).permute(1,2,0).detach().cpu().numpy() * 255.0
            vis_lq = vis_lq.astype(np.uint8)
            vis_lq = vis_lq.copy()
            
                        
                        
            # vis ocr result
            if ('testr' in cfg.train.model) and (cfg.data.val.ocr.vis_ocr):
                
                # prepare ocr visualization
                val_ocr_save_path = f'{cfg.save.output_dir}/{val_data_name}/{exp_name}/final_ocr_result'
                os.makedirs(val_ocr_save_path, exist_ok=True)
                
                
                print('== logging val ocr results ===')
                print(f'- Evaluating {val_data_name} - {val_img_id}')
                
                
                val_ocr_result = val_out[1]
                
                
                # ------------------ overlay gt ocr ------------------
                vis_hq1 = vis_hq1.copy()
                gt_polys = val_polys           # b 16 2
                gt_texts = val_gt_text
                for vis_img_idx in range(len(gt_polys)):
                    gt_poly = gt_polys[vis_img_idx]   # 16 2
                    gt_poly = gt_poly.astype(np.int32)
                    gt_txt = gt_texts[vis_img_idx]
                    cv2.polylines(vis_hq1, [gt_poly], isClosed=True, color=(0,255,0), thickness=2)
                    cv2.putText(vis_hq1, gt_txt, (gt_poly[0][0], gt_poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                # ------------------ overlay gt ocr ------------------
                
                
                vis_result = cv2.hconcat([vis_hq1, vis_lq])
                # visualize ocr results per denoising timestep
                for ocr_res in val_ocr_result:
                    timeiter, ocr_res = next(iter(ocr_res.items()))
                    timeiter_int = int(timeiter.split('_')[-1])
                    
                    # ------------------ overlay pred ocr ------------------
                    vis_pred = vis_lq.copy()
                    vis_polys = ocr_res.polygons.view(-1,16,2)  # b 16 2
                    vis_recs = ocr_res.recs                     # b 25
                    for vis_img_idx in range(len(vis_polys)):
                        pred_poly = vis_polys[vis_img_idx]   # 16 2
                        pred_poly = np.array(pred_poly.detach().cpu()).astype(np.int32)         
                        pred_rec = vis_recs[vis_img_idx]     # 25
                        pred_txt = decode(pred_rec.tolist())
                        cv2.polylines(vis_pred, [pred_poly], isClosed=True, color=(0,255,0), thickness=2)
                        cv2.putText(vis_pred, pred_txt, (pred_poly[0][0], pred_poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    h, w, _ = vis_pred.shape
                    cv2.putText(vis_pred, str(timeiter_int), (w-50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    # ------------------ overlay pred ocr ------------------
                    
                    vis_result = cv2.hconcat([vis_result, vis_pred])
                cv2.imwrite(f'{val_ocr_save_path}/{val_img_id}.jpg', vis_result[:,:,::-1])
                
                # save w/ restored results
                vis_result = cv2.hconcat([vis_lq, vis_res, vis_hq2, vis_pred, vis_hq1])
                cv2.imwrite(f'{val_res_ocr_save_path}/{val_img_id}.jpg', vis_result[:,:,::-1])
                                
            else:
                ## -------------------- visualize only restored results -------------------- 
                vis_result = cv2.hconcat([vis_lq, vis_res, vis_hq2])
                cv2.imwrite(f'{val_res_ocr_save_path}/{val_img_id}.jpg', vis_result[:,:,::-1])



        # calculate total val metric
        tot_val_psnr = np.array(metrics[f'{val_data_name}_psnr']).mean()
        tot_val_ssim = np.array(metrics[f'{val_data_name}_ssim']).mean()
        tot_val_lpips = np.array(metrics[f'{val_data_name}_lpips']).mean()
        tot_val_dists = np.array(metrics[f'{val_data_name}_dists']).mean()
        tot_val_niqe = np.array(metrics[f'{val_data_name}_niqe']).mean()
        tot_val_musiq = np.array(metrics[f'{val_data_name}_musiq']).mean()
        tot_val_maniqa = np.array(metrics[f'{val_data_name}_maniqa']).mean()
        tot_val_clipiqa = np.array(metrics[f'{val_data_name}_clipiqa']).mean()


        # calculate total val metric for min-max normalized images
        tot_val_psnr_norm = np.array(metrics[f'{val_data_name}_psnr_norm']).mean()
        tot_val_ssim_norm = np.array(metrics[f'{val_data_name}_ssim_norm']).mean()
        tot_val_lpips_norm = np.array(metrics[f'{val_data_name}_lpips_norm']).mean()
        tot_val_dists_norm = np.array(metrics[f'{val_data_name}_dists_norm']).mean()
        tot_val_niqe_norm = np.array(metrics[f'{val_data_name}_niqe_norm']).mean()
        tot_val_musiq_norm = np.array(metrics[f'{val_data_name}_musiq_norm']).mean()
        tot_val_maniqa_norm = np.array(metrics[f'{val_data_name}_maniqa_norm']).mean()
        tot_val_clipiqa_norm = np.array(metrics[f'{val_data_name}_clipiqa_norm']).mean()


        # log total val metrics
        if accelerator.is_main_process:
            # neatly print results for both normal and normalized versions
            print("\n" + "="*80)
            print(f"Validation Results for '{val_data_name}' Dataset")
            print("="*80)
            print(f"{'Metric':<12} | {'Original':>12} | {'Min–Max Normalized':>20}")
            print("-"*80)
            print(f"{'PSNR':<12} | {tot_val_psnr:>12.4f} | {tot_val_psnr_norm:>20.4f}")
            print(f"{'SSIM':<12} | {tot_val_ssim:>12.4f} | {tot_val_ssim_norm:>20.4f}")
            print(f"{'LPIPS':<12} | {tot_val_lpips:>12.4f} | {tot_val_lpips_norm:>20.4f}")
            print(f"{'DISTS':<12} | {tot_val_dists:>12.4f} | {tot_val_dists_norm:>20.4f}")
            print(f"{'NIQE':<12} | {tot_val_niqe:>12.4f} | {tot_val_niqe_norm:>20.4f}")
            print(f"{'MUSIQ':<12} | {tot_val_musiq:>12.4f} | {tot_val_musiq_norm:>20.4f}")
            print(f"{'MANIQA':<12} | {tot_val_maniqa:>12.4f} | {tot_val_maniqa_norm:>20.4f}")
            print(f"{'CLIPIQA':<12} | {tot_val_clipiqa:>12.4f} | {tot_val_clipiqa_norm:>20.4f}")
            print("="*80 + "\n")


            # log to wandb if enabled
            if cfg.log.tracker.report_to == 'wandb':
                wandb.log({
                    f'val_metric_orig_imgs/{val_data_name}_val_psnr': tot_val_psnr,
                    f'val_metric_orig_imgs/{val_data_name}_val_ssim': tot_val_ssim,
                    f'val_metric_orig_imgs/{val_data_name}_val_lpips': tot_val_lpips,
                    f'val_metric_orig_imgs/{val_data_name}_val_dists': tot_val_dists,
                    f'val_metric_orig_imgs/{val_data_name}_val_niqe': tot_val_niqe,
                    f'val_metric_orig_imgs/{val_data_name}_val_musiq': tot_val_musiq,
                    f'val_metric_orig_imgs/{val_data_name}_val_maniqa': tot_val_maniqa,
                    f'val_metric_orig_imgs/{val_data_name}_val_clipiqa': tot_val_clipiqa,

                    f'val_metric_normalized_imgs/{val_data_name}_val_psnr_norm': tot_val_psnr_norm,
                    f'val_metric_normalized_imgs/{val_data_name}_val_ssim_norm': tot_val_ssim_norm,
                    f'val_metric_normalized_imgs/{val_data_name}_val_lpips_norm': tot_val_lpips_norm,
                    f'val_metric_normalized_imgs/{val_data_name}_val_dists_norm': tot_val_dists_norm,
                    f'val_metric_normalized_imgs/{val_data_name}_val_niqe_norm': tot_val_niqe_norm,
                    f'val_metric_normalized_imgs/{val_data_name}_val_musiq_norm': tot_val_musiq_norm,
                    f'val_metric_normalized_imgs/{val_data_name}_val_maniqa_norm': tot_val_maniqa_norm,
                    f'val_metric_normalized_imgs/{val_data_name}_val_clipiqa_norm': tot_val_clipiqa_norm,
                })



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
