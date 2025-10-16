
from diffusers.optimization import get_scheduler
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.getcwd())
from accelerate.logging import get_logger

import cv2 
import wandb
import pyiqa
import math
import argparse
import numpy as np 
from PIL import Image 
from tqdm.auto import tqdm
from einops import rearrange 
from omegaconf import OmegaConf

from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory

import initialize 
import train_utils
from dataloaders.utils import encode, decode
from train_utils import encode_prompt, get_sigmas
from dataloaders.utils import realesrgan_degradation
from pipelines.pipeline_dit4sr import StableDiffusion3ControlNetPipeline

import torch
import torchvision.transforms as T 
import torch.nn.functional as F 
from torchvision.utils import save_image 


logger = get_logger(__name__)


def main(cfg):
    

    # set experiment name
    exp_name = f'{cfg.train.mixed_precision}_{"-".join(cfg.train.model)}_{"-".join(f"{lr:.0e}" for lr in cfg.train.lr)}_{"-".join(cfg.train.finetune)}_ocrloss{cfg.train.ocr_loss_weight}_{cfg.model.dit.text_condition.caption_style}_msg{cfg.log.tracker.msg}'
    cfg.exp_name = exp_name


    # set accelerator and basic settings (seed, logging, dir_path)
    accelerator = initialize.load_experiment_setting(cfg, logger, exp_name)


    # load data
    train_dataloader, val_datasets = initialize.load_data(cfg)

    
    # load models 
    models = initialize.load_model(cfg, accelerator)


    # load model parameters (total_params, trainable_params, frozen_params)
    model_params = initialize.load_model_params(cfg, models)


    # load optimizer 
    optimizer = initialize.load_optim(cfg, accelerator, models)


    # place models on cuda and proper weight dtype(float32, float16)
    weight_dtype = initialize.set_model_device(cfg, accelerator, models)


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.train.gradient_accumulation_steps)
    if cfg.train.max_train_steps is None:
        cfg.train.max_train_steps = cfg.train.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.train.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.train.max_train_steps * accelerator.num_processes,
        num_cycles=cfg.train.lr_num_cycles,
        power=cfg.train.lr_power,
    )

    # Prepare everything with our `accelerator`.
    # transformer, ts_module, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(models['transformer'], models['testr'], optimizer, train_dataloader, lr_scheduler)
    # transformer, models['testr'], optimizer, train_dataloader, lr_scheduler = accelerator.prepare(models['transformer'], models['testr'], optimizer, train_dataloader, lr_scheduler)
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(models['transformer'], optimizer, train_dataloader, lr_scheduler)
    vae_img_processor = VaeImageProcessor(vae_scale_factor=8)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.train.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.train.max_train_steps = cfg.train.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.train.num_train_epochs = math.ceil(cfg.train.max_train_steps / num_update_steps_per_epoch)
    
    initialize.load_trackers(cfg, accelerator, exp_name)


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


    tot_train_epochs = cfg.train.num_train_epochs
    tot_train_steps = cfg.train.max_train_steps


    # Train!
    total_batch_size = cfg.train.batch_size * accelerator.num_processes * cfg.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info("=== Model Parameters ===")
    logger.info(f"  Total Params    : {model_params['tot_param']:,} ({model_params['tot_param']/1e6:.2f}M)")
    logger.info(f"  Trainable Params: {model_params['train_param']:,} ({model_params['train_param']/1e6:.2f}M)")
    logger.info(f"  Frozen Params   : {model_params['frozen_param']:,} ({model_params['frozen_param']/1e6:.2f}M)")

    logger.info("=== Training Setup ===")
    logger.info(f"  Num training samples = {len(train_dataloader.dataset)}")
    # logger.info(f"  Num validation samples = {len(val_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {tot_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {tot_train_steps}")

    logger.info("=== Parameter Names ===")
    logger.info(f"  Frozen Params ({len(model_params['frozen_param_names'])}):")
    for name in model_params['frozen_param_names']:
        logger.info(f" FROZEN - {name}")
    logger.info(f"  Trainable Params ({len(model_params['train_param_names'])}):")
    for name in model_params['train_param_names']:
        logger.info(f" TRAINING - {name}")


    global_step = 0
    first_epoch = 0

    ocr_loss = 0.0 
    ocr_losses={}  


    # Load previous trained ckpts
    if cfg.ckpt.resume_path.dit is not None:
        resume_path = cfg.ckpt.resume_path.dit
        num_ckpt = resume_path.split('/')[-1].split('-')[-1]
        accelerator.print(f"Resuming from checkpoint {resume_path}")
        accelerator.load_state(resume_path)
        # load ts_module ckpt
        if 'testr' in cfg.train.model:
            ts_ckpt_path = os.path.join(resume_path, f"ts_module{int(num_ckpt):07d}.pt")
            ckpt = torch.load(ts_ckpt_path, map_location="cpu")
            load_result = models['testr'].load_state_dict(ckpt['ts_module'], strict=False)
            print("Loaded TESTR checkpoint keys:")
            print(" - Missing keys:", load_result.missing_keys)
            print(" - Unexpected keys:", load_result.unexpected_keys)
        # resume global step
        global_step = int(num_ckpt)
        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
    else:
        accelerator.print(f"Checkpoint '{cfg.ckpt.resume_path.dit}' does not exist. Starting a new training run.")
        cfg.ckpt.resume_path.dit = None
        initial_global_step = 0

        


    #     if cfg.ckpt.resume_path.dit != "latest":
    #         path = os.path.basename(cfg.ckpt.resume_path.dit)
    #     else:
    #         # Get the most recent checkpoint
    #         dirs = os.listdir(f'{cfg.save.output_dir}/{exp_name}')
    #         dirs = [d for d in dirs if d.startswith("checkpoint")]
    #         dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    #         path = dirs[-1] if len(dirs) > 0 else None

    #     if path is None:
    #         accelerator.print(
    #             f"Checkpoint '{cfg.ckpt.resume_path.dit}' does not exist. Starting a new training run."
    #         )
    #         cfg.ckpt.resume_path.dit = None
    #         initial_global_step = 0
    #     else:
    #         breakpoint()
    #         # load transformer ckpt
    #         accelerator.print(f"Resuming from checkpoint {path}")
    #         resume_path = os.path.join(cfg.save.output_dir, exp_name, path)
    #         accelerator.load_state(resume_path)
            
    #         # load ts_module ckpt
    #         if 'testr' in cfg.train.model:
    #             ts_ckpt_path = os.path.join(resume_path, f"ts_module{int(path.split('-')[-1]):07d}.pt")
    #             ckpt = torch.load(ts_ckpt_path, map_location="cpu")
    #             load_result = models['testr'].load_state_dict(ckpt['ts_module'], strict=False)
    #             print("Loaded TESTR checkpoint keys:")
    #             print(" - Missing keys:", load_result.missing_keys)
    #             print(" - Unexpected keys:", load_result.unexpected_keys)

    #         global_step = int(path.split("-")[1])

    #         initial_global_step = global_step
    #         first_epoch = global_step // num_update_steps_per_epoch
    # else:
    #     initial_global_step = 0


    progress_bar = tqdm(range(0, tot_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process,)
    free_memory()
    for epoch in range(first_epoch, tot_train_epochs):
        for step, batch in enumerate(train_dataloader):

            if cfg.data.train.name == 'satext':
                batch = realesrgan_degradation(batch)

                gt = batch['gt']
                lq = batch['lq']
                text = batch['text']
                text_encs = batch['text_enc']
                hq_prompt = batch['hq_prompt']
                # lq_prompt = batch['lq_prompt']
                boxes = batch['bbox']    # len(bbox) = b
                polys = batch['poly']    # len(poly) = b
                img_id = batch['img_id']
            

            with accelerator.accumulate([transformer]):

                if cfg.data.train.name == 'satext':
                    with torch.no_grad():
                        # hq vae encoding
                        gt = gt.to(device=accelerator.device, dtype=weight_dtype) * 2.0 - 1.0   # b 3 512 512 
                        hq_latents = models['vae'].encode(gt).latent_dist.sample()  # b 16 64 64
                        model_input = (hq_latents - models['vae'].config.shift_factor) * models['vae'].config.scaling_factor    # b 16 64 64 
                        model_input = model_input.to(dtype=weight_dtype)
                        # lq vae encoding
                        lq = lq.to(device=accelerator.device, dtype=weight_dtype) * 2.0 - 1.0   # b 3 512 512 
                        lq_latents = models['vae'].encode(lq).latent_dist.sample()  # b 16 64 64 
                        controlnet_image = (lq_latents - models['vae'].config.shift_factor) * models['vae'].config.scaling_factor   # b 16 64 64 
                        controlnet_image = controlnet_image.to(dtype=weight_dtype)
                        # load caption
                        if cfg.model.dit.load_precomputed_caption:
                            hq_prompt = hq_prompt 
                        else:
                            # vlm captioner
                            lq_tmp = F.interpolate(lq, size=(336, 336), mode="bilinear", align_corners=False)
                            hq_prompt = models['vlm_agent'].gen_image_caption(lq_tmp)
                            hq_prompt = [train_utils.remove_focus_sentences(p) for p in hq_prompt]

                        # set prompt style
                        if cfg.model.dit.use_gtprompt:
                            if cfg.model.dit.text_condition.caption_style == 'descriptive':
                                texts = [[f'"{t}"' for t in txt] for txt in text]
                                hq_prompt = [f'The image features the texts {", ".join(txt)} that appear clearly on signs, boards, buildings, or other objects.' for txt in texts]
                            elif cfg.model.dit.text_condition.caption_style == 'tag':
                                texts = [[f'"{t}"' for t in txt] for txt in text]
                                hq_prompt=[', '.join(words) for words in texts]  

                        # encode prompt 
                        prompt_embeds, pooled_prompt_embeds = encode_prompt(models['text_encoders'], models['tokenizers'], hq_prompt, 77)
                        prompt_embeds = prompt_embeds.to(model_input.dtype)                 # b 154 4096
                        pooled_prompt_embeds = pooled_prompt_embeds.to(model_input.dtype)   # b 2048
                else:
                    model_input = batch["pixel_values"].to(dtype=weight_dtype)
                    controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)   # b 16 64 64 
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=cfg.model.noise_scheduler.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=cfg.model.noise_scheduler.logit_mean,
                    logit_std=cfg.model.noise_scheduler.logit_std,
                    mode_scale=cfg.model.noise_scheduler.mode_scale,
                )

                indices = (u * models['noise_scheduler_copy'].config.num_train_timesteps).long()
                timesteps = models['noise_scheduler_copy'].timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching. b
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, accelerator, models['noise_scheduler_copy'], n_dim=model_input.ndim, dtype=model_input.dtype)    # b 1 1 1
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise   # b 16 64 64 
                with torch.cuda.amp.autocast(enabled=False):
                    # Predict the noise residual
                    trans_out = transformer(                       
                        hidden_states=noisy_model_input,            # b 16 64 64 
                        controlnet_image=controlnet_image,          # b 16 16 16  
                        timestep=timesteps,                         # b
                        encoder_hidden_states=prompt_embeds,        # b 154 4096
                        pooled_projections=pooled_prompt_embeds,    # b 2048
                        return_dict=False,
                    )
                model_pred = trans_out[0]   # b 16 64 64

                if len(trans_out) > 1:
                    etc_out = trans_out[1]
                    # unpatchify
                    patch_size = models['transformer'].config.patch_size  # 2
                    hidden_dim = models['transformer'].config.num_attention_heads * models['transformer'].config.attention_head_dim     # 1536
                    height = 64 // patch_size       # 32
                    width = 64 // patch_size        # 32
                    extracted_feats = [ rearrange(feat['extract_feat'], 'b (H W) (pH pW d) -> b d (H pH) (W pW)', H=height, W=width, pH=patch_size, pW=patch_size) for feat in etc_out ]    # b 384 64 64 

                '''
                (Pdb) extracted_feats[0].shape  torch.Size([2, 1280, 16, 16])
                (Pdb) extracted_feats[1].shape  torch.Size([2, 1280, 32, 32])
                (Pdb) extracted_feats[2].shape  torch.Size([2, 640, 64, 64])
                (Pdb) extracted_feats[3].shape  torch.Size([2, 320, 64, 64])

                (Pdb) trans_out[0]['extract_feat'].shape    torch.Size([2, 1024, 1536])
                (Pdb) trans_out[1]['extract_feat'].shape    torch.Size([2, 1024, 1536])
                (Pdb) trans_out[2]['extract_feat'].shape    torch.Size([2, 1024, 1536])
                (Pdb) trans_out[3]['extract_feat'].shape    torch.Size([2, 1024, 1536])
                '''


                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                if cfg.model.noise_scheduler.precondition_outputs:   # t
                    model_pred = model_pred * (-sigmas) + noisy_model_input # b 16 64 64

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=cfg.model.noise_scheduler.weighting_scheme, sigmas=sigmas)   # b 1 1 1

                # flow matching loss
                if cfg.model.noise_scheduler.precondition_outputs:   # t
                    target = model_input
                else:
                    target = noise - model_input

                # Compute regular loss.
                diff_loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                diff_loss = diff_loss.mean()


                # ts module loss 
                if 'testr' in cfg.train.model:
                    # process annotations for OCR training loss
                    train_targets=[]
                    for i in range(bsz):
                        num_box=len(boxes[i])
                        tmp_dict={}
                        tmp_dict['labels'] = torch.tensor([0]*num_box).to(accelerator.device)  # 0 for text
                        tmp_dict['boxes'] = torch.tensor(boxes[i]).to(accelerator.device)   # xyxy format, absolute coord, [num_box, 4]
                        tmp_dict['texts'] = text_encs[i]
                        tmp_dict['ctrl_points'] = polys[i]
                        train_targets.append(tmp_dict)
                    with torch.cuda.amp.autocast(enabled=False):
                        # OCR model forward pass
                        ocr_loss_dict, ocr_result = models['testr'](extracted_feats, train_targets, MODE='TRAIN')
                    # OCR total_loss
                    ocr_tot_loss = sum(v for v in ocr_loss_dict.values())
                    # OCR losses
                    for ocr_key, ocr_val in ocr_loss_dict.items():
                        if ocr_key in ocr_losses.keys():
                            ocr_losses[ocr_key].append(ocr_val.item())
                        else:
                            ocr_losses[ocr_key]=[ocr_val.item()]
                    total_loss = diff_loss + cfg.train.ocr_loss_weight * ocr_tot_loss      
                else:
                    total_loss = diff_loss
                    ocr_tot_loss=torch.tensor(0).cuda()

                # backpropagate
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    # params_to_clip = controlnet.parameters()
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, cfg.train.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=cfg.train.set_grads_to_none)


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % cfg.save.checkpointing_steps == 0:
                        # save transformer
                        save_path = os.path.join(cfg.save.output_dir, exp_name, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        if 'testr' in cfg.train.model:
                            # save ts_module
                            ts_ckpt = {}
                            ts_ckpt['ts_module'] = models['testr'].state_dict()
                            ckpt_path = f"{save_path}/ts_module{global_step:07d}.pt"
                            torch.save(ts_ckpt, ckpt_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step==1 or global_step % cfg.val.val_every_step == 0:

                        if 'testr' in cfg.train.model:
                            ts_module = models['testr'] 
                        else:
                            ts_module = None 
                        # Get the validation pipeline
                        val_pipeline = StableDiffusion3ControlNetPipeline(
                            vae=models['vae'], text_encoder=models['text_encoders'][0], text_encoder_2=models['text_encoders'][1], text_encoder_3=models['text_encoders'][2], 
                            tokenizer=models['tokenizers'][0], tokenizer_2=models['tokenizers'][1], tokenizer_3=models['tokenizers'][2], 
                            transformer=models['transformer'], scheduler=models['noise_scheduler'], ts_module=ts_module, cfg=cfg
                        )

                        metrics={}

                        for val_data_name, val_data in val_datasets.items():

                            metrics[f'{val_data_name}_psnr'] = []
                            metrics[f'{val_data_name}_ssim'] = []
                            metrics[f'{val_data_name}_lpips'] = []
                            metrics[f'{val_data_name}_dists'] = []
                            metrics[f'{val_data_name}_niqe'] = []
                            metrics[f'{val_data_name}_musiq'] = []
                            metrics[f'{val_data_name}_maniqa'] = []
                            metrics[f'{val_data_name}_clipiqa'] = []


                            for val_sample in val_data:

                                # set seed
                                if accelerator.is_main_process:
                                    generator = torch.Generator(device=accelerator.device)
                                    if cfg.init.seed is not None:
                                        generator.manual_seed(cfg.init.seed)

                                # get anns
                                val_lq_path = val_sample['lq_path']
                                val_hq_path = val_sample['hq_path']
                                val_text = val_sample['text']
                                val_bbox = val_sample['bbox']
                                val_polys = val_sample['poly']
                                val_img_id = val_sample['img_id']

                                # place lq on cuda
                                val_lq = T.ToTensor()(Image.open(val_lq_path)).to(device=accelerator.device, dtype=weight_dtype).unsqueeze(dim=0)   # 1 3 128 128 
                                val_lq = F.interpolate(val_lq, (512,512), mode='bilinear', align_corners=False) # 1 3 512 512 
                                val_lq = (val_lq - val_lq.min()) / (val_lq.max() - val_lq.min() + 1e-8) # [0,1]
                                val_lq = val_lq * 2. - 1.   # [-1,1]
                                
                                # place gt to cuda
                                val_gt = T.ToTensor()(Image.open(val_hq_path)).to(device=accelerator.device, dtype=weight_dtype).unsqueeze(dim=0)   # 1 3 128 128 
                                val_gt = (val_gt - val_gt.min()) / (val_gt.max() - val_gt.min() + 1e-8)
                                val_gt = val_gt * 2. - 1.

                                val_prompt = ['']
                                neg_prompt = None 

                                # validation forward pass
                                with torch.no_grad():
                                    val_out = val_pipeline(
                                        prompt=val_prompt, control_image=val_lq, num_inference_steps=cfg.data.val.num_inference_steps, generator=generator, height=height, width=width,
                                        guidance_scale=cfg.data.val.guidance_scale, negative_prompt=neg_prompt,
                                        start_point=cfg.data.val.start_point, latent_tiled_size=cfg.data.val.latent_tiled_size, latent_tiled_overlap=cfg.data.val.latent_tiled_overlap,
                                        output_type = 'pt', return_dict=False, lq_id=val_img_id, val_data_name=val_data_name, global_step=global_step, cfg=cfg, mode='train'
                                    )
                                
                                # retrive validation results
                                val_restored_img = val_out[0]   # 1 3 512 512 [0,1]
                                if 'testr' in cfg.train.model:
                                    val_ocr_result = val_out[1]

                                # prepare visualization
                                val_save_path = f'{cfg.save.output_dir}/{exp_name}/{val_data_name}/final_result'
                                os.makedirs(val_save_path, exist_ok=True)
                                
                                # lq 
                                val_lq_img = val_lq
                                val_lq_img = (val_lq_img + 1.0) / 2.0
                                # gt 
                                val_gt_img = val_gt
                                val_gt_img = (val_gt_img + 1.0) / 2.0
                                # restored
                                val_res_img = val_restored_img.squeeze(dim=0).permute(1,2,0).cpu().detach().numpy()*255.0
                                val_res_img = val_res_img.astype(np.uint8)

                                # preprocess
                                img_lq = val_lq.detach().permute(0,2,3,1).cpu().numpy() # b h w c
                                img_gt = val_gt.detach().permute(0,2,3,1).cpu().numpy()

                                vis_lq = img_lq[0] # h w c 
                                vis_lq = (vis_lq + 1.0)/2.0 * 255.0
                                vis_lq = vis_lq.astype(np.uint8)
                                vis_lq = vis_lq.copy()
                                vis_pred = vis_lq.copy()

                                vis_gt = img_gt[0] # h w c
                                vis_gt = (vis_gt + 1.0)/2.0 * 255.0
                                vis_gt = vis_gt.astype(np.uint8)
                                vis_gt = vis_gt.copy()
                                vis_gt2 = vis_gt.copy()


                                if 'testr' in cfg.train.model:
                                    ## -------------------- visualize restored + OCR results -------------------- 
                                    print('== logging val ocr results ===')
                                    print(f'- Evaluating {val_data_name} - {val_img_id}')
                                    
                                    ocr_res = val_ocr_result[0]
                                    vis_polys = ocr_res.polygons.view(-1,16,2)  # b 16 2
                                    vis_recs = ocr_res.recs                     # b 25
                                    for vis_img_idx in range(len(vis_polys)):
                                        pred_poly = vis_polys[vis_img_idx]   # 16 2
                                        pred_poly = np.array(pred_poly.detach().cpu()).astype(np.int32)         
                                        pred_rec = vis_recs[vis_img_idx]     # 25
                                        pred_txt = decode(pred_rec.tolist())
                                        cv2.polylines(vis_pred, [pred_poly], isClosed=True, color=(0,255,0), thickness=2)
                                        cv2.putText(vis_pred, pred_txt, (pred_poly[0][0], pred_poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                                    gt_polys = val_polys           # b 16 2
                                    gt_texts = val_text
                                    for vis_img_idx in range(len(gt_polys)):
                                        gt_poly = gt_polys[vis_img_idx]   # 16 2
                                        # gt_poly = np.array(gt_poly.detach().cpu()).astype(np.int32)
                                        gt_poly = gt_poly.astype(np.int32)
                                        gt_txt = gt_texts[vis_img_idx]
                                        cv2.polylines(vis_gt, [gt_poly], isClosed=True, color=(0,255,0), thickness=2)
                                        cv2.putText(vis_gt, gt_txt, (gt_poly[0][0], gt_poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                                    vis_result = cv2.hconcat([vis_lq, val_res_img, vis_gt2, vis_pred, vis_gt])
                                    cv2.imwrite(f'{val_save_path}/{val_img_id}_step{global_step:09d}.jpg', vis_result[:,:,::-1])
                                else:
                                    ## -------------------- visualize only restored results -------------------- 
                                    vis_result = cv2.hconcat([vis_lq, val_res_img, vis_gt2])
                                    cv2.imwrite(f'{val_save_path}/{val_img_id}_step{global_step:09d}.jpg', vis_result[:,:,::-1])

                                # append val metrics
                                metrics[f'{val_data_name}_psnr'].append(torch.mean(metric_psnr(val_restored_img.to(torch.float32), torch.clamp((val_gt_img.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                                metrics[f'{val_data_name}_ssim'].append(torch.mean(metric_ssim(val_restored_img.to(torch.float32), torch.clamp((val_gt_img.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                                metrics[f'{val_data_name}_lpips'].append(torch.mean(metric_lpips(val_restored_img.to(torch.float32), torch.clamp((val_gt_img.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                                metrics[f'{val_data_name}_dists'].append(torch.mean(metric_dists(val_restored_img.to(torch.float32), torch.clamp((val_gt_img.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                                metrics[f'{val_data_name}_niqe'].append(torch.mean(metric_niqe(val_restored_img.to(torch.float32), torch.clamp((val_gt_img.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                                metrics[f'{val_data_name}_musiq'].append(torch.mean(metric_musiq(val_restored_img.to(torch.float32), torch.clamp((val_gt_img.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                                metrics[f'{val_data_name}_maniqa'].append(torch.mean(metric_maniqa(val_restored_img.to(torch.float32), torch.clamp((val_gt_img.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                                metrics[f'{val_data_name}_clipiqa'].append(torch.mean(metric_clipiqa(val_restored_img.to(torch.float32), torch.clamp((val_gt_img.to(torch.float32) + 1) / 2, min=0, max=1))).item())

                            # calculate total val metric
                            tot_val_psnr = np.array(metrics[f'{val_data_name}_psnr']).mean()
                            tot_val_ssim = np.array(metrics[f'{val_data_name}_ssim']).mean()
                            tot_val_lpips = np.array(metrics[f'{val_data_name}_lpips']).mean()
                            tot_val_dists = np.array(metrics[f'{val_data_name}_dists']).mean()
                            tot_val_niqe = np.array(metrics[f'{val_data_name}_niqe']).mean()
                            tot_val_musiq = np.array(metrics[f'{val_data_name}_musiq']).mean()
                            tot_val_maniqa = np.array(metrics[f'{val_data_name}_maniqa']).mean()
                            tot_val_clipiqa = np.array(metrics[f'{val_data_name}_clipiqa']).mean()

                            # log total val metrics 
                            if accelerator.is_main_process and cfg.log.tracker.report_to == 'wandb':
                                wandb.log({
                                    f'val_metric/{val_data_name}_val_psnr': tot_val_psnr,
                                    f'val_metric/{val_data_name}_val_ssim': tot_val_ssim,
                                    f'val_metric/{val_data_name}_val_lpips': tot_val_lpips,
                                    f'val_metric/{val_data_name}_val_dists': tot_val_dists,
                                    f'val_metric/{val_data_name}_val_niqe': tot_val_niqe,
                                    f'val_metric/{val_data_name}_val_musiq': tot_val_musiq,
                                    f'val_metric/{val_data_name}_val_maniqa': tot_val_maniqa,
                                    f'val_metric/{val_data_name}_val_clipiqa': tot_val_clipiqa,
                                }, step=global_step)

            # log 
            logs = {"loss/total_loss": total_loss.detach().item(), 
                    'loss/diff_loss': diff_loss.detach().item(),
                    "optim/lr": lr_scheduler.get_last_lr()[0],
                    }
            
            # ocr log
            if 'testr' in cfg.train.model:
                logs["loss/ocr_tot_loss"] = ocr_tot_loss.detach().item()
                logs['optim/ts_module_lr'] = cfg.train.lr[1]
                for ocr_key, ocr_val in ocr_loss_dict.items():
                    logs[f"loss/ocr_{ocr_key}"] = ocr_val.detach().item()


            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= cfg.train.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if cfg.model.dit.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )
    main(cfg)
