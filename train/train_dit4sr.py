import argparse
import math
from diffusers.optimization import get_scheduler
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutil
import sys
sys.path.append(os.getcwd())
import torch
from accelerate.logging import get_logger
from tqdm.auto import tqdm

from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory, cast_training_params

import torch.nn.functional as F
from omegaconf import OmegaConf

import initialize 
from train_utils import _encode_prompt_with_t5, _encode_prompt_with_clip, encode_prompt, unwrap_model, compute_text_embeddings, get_sigmas
from dataloaders.utils import realesrgan_degradation
import train_utils 

import wandb
from torchvision.utils import save_image 
import pyiqa
from einops import rearrange 
import cv2 
import numpy as np 

from dataloaders.utils import encode, decode 
from pipelines.pipeline_dit4sr import StableDiffusion3ControlNetPipeline

logger = get_logger(__name__)



def main(cfg):
    

    # set accelerator and basic settings (seed, logging, dir_path)
    accelerator = initialize.load_experiment_setting(cfg, logger)


    # load data
    train_dataloader, val_dataloader = initialize.load_data(cfg)

    
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
    
    initialize.load_trackers(cfg, accelerator)


    # Get the validation pipeline
    val_pipeline = StableDiffusion3ControlNetPipeline(
        vae=models['vae'], text_encoder=models['text_encoders'][0], text_encoder_2=models['text_encoders'][1], text_encoder_3=models['text_encoders'][2], 
        tokenizer=models['tokenizers'][0], tokenizer_2=models['tokenizers'][1], tokenizer_3=models['tokenizers'][2], 
        transformer=transformer, scheduler=models['noise_scheduler'], ts_module = models['testr']
    )


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
    logger.info(f"  Num validation samples = {len(val_dataloader.dataset)}")
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


    # Potentially load in the weights and states from a previous save
    if cfg.ckpt.resume_path.dit is not None:
        if cfg.ckpt.resume_path.dit != "latest":
            path = os.path.basename(cfg.ckpt.resume_path.dit)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(f'{cfg.save.output_dir}/{cfg.log.tracker.run_name}')
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.ckpt.resume_path.dit}' does not exist. Starting a new training run."
            )
            cfg.ckpt.resume_path.dit = None
            initial_global_step = 0
        else:
            # load transformer ckpt
            accelerator.print(f"Resuming from checkpoint {path}")
            resume_path = os.path.join(cfg.save.output_dir, cfg.log.tracker.run_name, path)
            accelerator.load_state(resume_path)
            # accelerator.load_state(os.path.join(cfg.save.output_dir, cfg.log.tracker.run_name, path))
            # load ts_module ckpt
            ts_ckpt_path = os.path.join(resume_path, f"ts_module{int(path.split('-')[-1]):07d}.pt")
            ckpt = torch.load(ts_ckpt_path, map_location="cpu")
            load_result = models['testr'].load_state_dict(ckpt['ts_module'], strict=False)
            print("Loaded TESTR checkpoint keys:")
            print(" - Missing keys:", load_result.missing_keys)
            print(" - Unexpected keys:", load_result.unexpected_keys)

            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0


    progress_bar = tqdm(range(0, tot_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process,)
    free_memory()
    image_logs = None
    for epoch in range(first_epoch, tot_train_epochs):
        for step, batch in enumerate(train_dataloader):

            if cfg.data.name == 'satext':
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

                if cfg.data.name == 'satext':
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

                        if cfg.model.dit.use_gtprompt:
                            hq_prompt=[', '.join(words) for words in text]  # gt words using tag style

                        # encode prompt 
                        prompt_embeds, pooled_prompt_embeds = encode_prompt(models['text_encoders'], models['tokenizers'], hq_prompt, 77)
                        prompt_embeds = prompt_embeds.to(model_input.dtype)                 # b 154 4096
                        pooled_prompt_embeds = pooled_prompt_embeds.to(model_input.dtype)   # b 2048
                else:
                    model_input = batch["pixel_values"].to(dtype=weight_dtype)
                    controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                # image_embedding = controlnet_image.view(controlnet_image.shape[0], 16, -1)
                # pad_tensor = torch.zeros(controlnet_image.shape[0], 77 - image_embedding.shape[1], 4096).to(image_embedding.device, dtype=weight_dtype)
                # image_embedding = torch.cat([image_embedding, pad_tensor], dim=1)


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
                    patch_size = transformer.config.patch_size  # 2
                    hidden_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim     # 1536
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
                if cfg.train.finetune_model == 'dit4sr_testr':
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


                ## -------------------- vis training ocr results -------------------- 
                if cfg.val.log_train_ocr_result_every != -1 and global_step % cfg.val.log_train_ocr_result_every == 0:
                    print('== logging training ocr results ===')
                    train_ocr_save_path = f'{cfg.save.output_dir}/{cfg.log.tracker.run_name}/ocr_result_train'
                    os.makedirs(train_ocr_save_path, exist_ok=True)

                    img_lq = lq.detach().permute(0,2,3,1).cpu().numpy() # b h w c
                    img_gt = gt.detach().permute(0,2,3,1).cpu().numpy()

                    for vis_batch_idx in range(len(ocr_result)):
                        # vis_lq = img_lq[vis_batch_idx] # h w c 
                        # vis_lq = (vis_lq + 1.0)/2.0 * 255.0
                        # vis_lq = vis_lq.astype(np.uint8)
                        # vis_lq = vis_lq.copy()

                        vis_gt = img_gt[vis_batch_idx] # h w c
                        vis_gt = (vis_gt + 1.0)/2.0 * 255.0
                        vis_gt = vis_gt.astype(np.uint8)
                        vis_pred = vis_gt.copy()
                        vis_gt = vis_gt.copy()

                        ocr_res = ocr_result[vis_batch_idx]
                        vis_polys = ocr_res.polygons.view(-1,16,2)  # b 16 2
                        vis_recs = ocr_res.recs                     # b 25
                        for vis_img_idx in range(len(vis_polys)):
                            pred_poly = vis_polys[vis_img_idx]   # 16 2
                            pred_poly = np.array(pred_poly.detach().cpu()).astype(np.int32)         
                            pred_rec = vis_recs[vis_img_idx]     # 25
                            pred_txt = decode(pred_rec.tolist())
                            cv2.polylines(vis_pred, [pred_poly], isClosed=True, color=(0,255,0), thickness=2)
                            cv2.putText(vis_pred, pred_txt, (pred_poly[0][0], pred_poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        cv2.imwrite(f'{train_ocr_save_path}/step{global_step}_{img_id[vis_batch_idx]}_ocr_pred.jpg.jpg', vis_pred[:,:,::-1])

                        gt_polys = polys[vis_batch_idx]             # b 16 2
                        gt_texts = text[vis_batch_idx]
                        for vis_img_idx in range(len(gt_polys)):
                            gt_poly = gt_polys[vis_img_idx]*512.0   # 16 2
                            gt_poly = np.array(gt_poly.detach().cpu()).astype(np.int32)
                            gt_txt = gt_texts[vis_img_idx]
                            cv2.polylines(vis_gt, [gt_poly], isClosed=True, color=(0,255,0), thickness=2)
                            cv2.putText(vis_gt, gt_txt, (gt_poly[0][0], gt_poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        cv2.imwrite(f'{train_ocr_save_path}/step{global_step}_{img_id[vis_batch_idx]}_ocr_gt.jpg', vis_gt[:,:,::-1])
                ## -------------------- vis training ocr results -------------------- 



            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % cfg.save.checkpointing_steps == 0:
                        # save transformer
                        save_path = os.path.join(cfg.save.output_dir, cfg.log.tracker.run_name, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        # save ts_module
                        ts_ckpt = {}
                        ts_ckpt['ts_module'] = models['testr'].state_dict()
                        ckpt_path = f"{save_path}/ts_module{global_step:07d}.pt"
                        torch.save(ts_ckpt, ckpt_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step==1 or global_step % cfg.val.val_every_step == 0:

                        tot_val_psnr=[]
                        tot_val_ssim=[]
                        tot_val_lpips=[]
                        tot_val_dists=[]
                        tot_val_niqe=[]
                        tot_val_musiq=[]
                        tot_val_maniqa=[]
                        tot_val_clipiqa=[]


                        # val_pipeline = val_pipeline.to(torch.float16)

                        for val_step, val_batch in enumerate(val_dataloader):

                            if cfg.data.name == 'satext':
                                val_batch = realesrgan_degradation(val_batch)

                                val_gt = val_batch['gt']
                                val_lq = val_batch['lq']
                                val_text = val_batch['text']
                                val_text_encs = val_batch['text_enc']
                                val_hq_prompt = val_batch['hq_prompt']
                                # lq_prompt = val_batch['lq_prompt']
                                val_boxes = val_batch['bbox']    # len(bbox) = b
                                val_polys = val_batch['poly']    # len(poly) = b
                                val_img_id = val_batch['img_id']
                                val_bs = val_gt.shape[0]


                            if accelerator.is_main_process:
                                generator = torch.Generator(device=accelerator.device)
                                if cfg.init.seed is not None:
                                    generator.manual_seed(cfg.init.seed)


                            # val_lq = val_lq * 2.0 - 1.0 
                            # val_gt = val_gt * 2.0 - 1.0
                            val_lq = val_lq.to(device=accelerator.device, dtype=weight_dtype) * 2.0 - 1.0   # b 3 512 512 
                            val_gt = val_gt.to(device=accelerator.device, dtype=weight_dtype) * 2.0 - 1.0   # b 3 512 512 
                            num_inf_step=40
                            guidance_scale = 1.0
                            start_point = 'noise'
                            val_prompt = ['' for _ in range(val_bs)]
                            # neg_prompt = 'motion blur, noisy, dotted, bokeh, pointed, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, chaotic'
                            neg_prompt = None 
                            
                            with torch.no_grad():
                                val_out = val_pipeline(
                                                    prompt=val_prompt, 
                                                    control_image=val_lq, 
                                                    num_inference_steps=num_inf_step, 
                                                    generator=generator, 
                                                    height=512, 
                                                    width=512,
                                                    guidance_scale=guidance_scale, 
                                                    negative_prompt=neg_prompt,
                                                    start_point=start_point, 
                                                    latent_tiled_size=64, 
                                                    latent_tiled_overlap=24,
                                                    output_type = 'pt',
                                                    return_dict=False
                                                )
                            val_restored_img = val_out[0]
                            val_ocr_result = val_out[1]

                            # Save as PNG
                            val_save_path = f'{cfg.save.output_dir}/{cfg.log.tracker.run_name}/restored_val'
                            os.makedirs(val_save_path, exist_ok=True)
                            for val_idx in range(val_bs):
                                # lq 
                                val_lq_img = val_lq[val_idx]
                                val_lq_img = (val_lq_img + 1.0) / 2.0
                                # gt 
                                val_gt_img = val_gt[val_idx]
                                val_gt_img = (val_gt_img + 1.0) / 2.0
                                # restored
                                val_res_img = val_restored_img[val_idx]
                                val_img = torch.concat([val_lq_img, val_res_img, val_gt_img], dim=2)
                                # save_image(val_img, f'{val_save_path}/step{global_step}_valimg{val_step}_{val_img_id[val_idx]}.png')
                                save_image(val_img, f'{val_save_path}/restored_{val_img_id[val_idx]}_step{global_step:09d}.png')
                                # save_image(val_lq_img, f'{val_save_path}/step{global_step}_{val_img_id[val_idx]}.png', normalize=True)
                                # save_image(val_res_img, f'{val_save_path}/step{global_step}_{val_img_id[val_idx]}.png', normalize=True)
                            

                            # log total psnr, ssim, lpips for val
                            tot_val_psnr.append(torch.mean(metric_psnr(val_restored_img.to(torch.float32), torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                            tot_val_ssim.append(torch.mean(metric_ssim(val_restored_img.to(torch.float32), torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                            tot_val_lpips.append(torch.mean(metric_lpips(val_restored_img.to(torch.float32), torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                            tot_val_dists.append(torch.mean(metric_dists(val_restored_img.to(torch.float32), torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                            tot_val_niqe.append(torch.mean(metric_niqe(val_restored_img.to(torch.float32), torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                            tot_val_musiq.append(torch.mean(metric_musiq(val_restored_img.to(torch.float32), torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                            tot_val_maniqa.append(torch.mean(metric_maniqa(val_restored_img.to(torch.float32), torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                            tot_val_clipiqa.append(torch.mean(metric_clipiqa(val_restored_img.to(torch.float32), torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item())
                        
                            # log sampling val imgs to wandb
                            if accelerator.is_main_process and cfg.log.tracker.report_to == 'wandb':

                                # log sampling val metrics 
                                wandb.log({f'val_metric/val_psnr': torch.mean(metric_psnr(
                                                                                                val_restored_img.to(torch.float32), 
                                                                                                torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item(),
                                        f'val_metric/val_ssim': torch.mean(metric_ssim(
                                                                                                val_restored_img.to(torch.float32), 
                                                                                                torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item(),
                                        f'val_metric/val_lpips': torch.mean(metric_lpips(
                                                                                                val_restored_img.to(torch.float32), 
                                                                                                torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item(),
                                        f'val_metric/val_dists': torch.mean(metric_dists(
                                                                                                val_restored_img.to(torch.float32), 
                                                                                                torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item(),
                                        f'val_metric/val_niqe': torch.mean(metric_niqe(
                                                                                                val_restored_img.to(torch.float32), 
                                                                                                torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item(),
                                        f'val_metric/val_musiq': torch.mean(metric_musiq(
                                                                                                val_restored_img.to(torch.float32), 
                                                                                                torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item(),
                                        f'val_metric/val_maniqa': torch.mean(metric_maniqa(
                                                                                                val_restored_img.to(torch.float32), 
                                                                                                torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item(),
                                        f'val_metric/val_clipiqa': torch.mean(metric_clipiqa(
                                                                                                val_restored_img.to(torch.float32), 
                                                                                                torch.clamp((val_gt.to(torch.float32) + 1) / 2, min=0, max=1))).item(),
                                        }, step=global_step)


                            ## -------------------- vis val ocr results -------------------- 
                            print('== logging val ocr results ===')
                            val_ocr_save_path = f'{cfg.save.output_dir}/{cfg.log.tracker.run_name}/ocr_result_val'
                            os.makedirs(val_ocr_save_path, exist_ok=True)

                            img_lq = val_lq.detach().permute(0,2,3,1).cpu().numpy() # b h w c
                            img_gt = val_gt.detach().permute(0,2,3,1).cpu().numpy()

                            for vis_batch_idx in range(len(val_gt)):
                                # vis_lq = img_lq[vis_batch_idx] # h w c 
                                # vis_lq = (vis_lq + 1.0)/2.0 * 255.0
                                # vis_lq = vis_lq.astype(np.uint8)
                                # vis_lq = vis_lq.copy()

                                vis_gt = img_gt[vis_batch_idx] # h w c
                                vis_gt = (vis_gt + 1.0)/2.0 * 255.0
                                vis_gt = vis_gt.astype(np.uint8)
                                vis_pred = vis_gt.copy()
                                vis_gt = vis_gt.copy()

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
                                # cv2.imwrite(f'{val_ocr_save_path}/ocr_pred_valimg{val_step}_{val_img_id[vis_batch_idx]}_step{global_step:09d}.jpg.jpg', vis_pred[:,:,::-1])
                                cv2.imwrite(f'{val_ocr_save_path}/ocr_pred_{val_img_id[vis_batch_idx]}_step{global_step:09d}.jpg.jpg', vis_pred[:,:,::-1])

                                gt_polys = val_polys[vis_batch_idx]             # b 16 2
                                gt_texts = val_text[vis_batch_idx]
                                for vis_img_idx in range(len(gt_polys)):
                                    gt_poly = gt_polys[vis_img_idx]*512.0   # 16 2
                                    gt_poly = np.array(gt_poly.detach().cpu()).astype(np.int32)
                                    gt_txt = gt_texts[vis_img_idx]
                                    cv2.polylines(vis_gt, [gt_poly], isClosed=True, color=(0,255,0), thickness=2)
                                    cv2.putText(vis_gt, gt_txt, (gt_poly[0][0], gt_poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                                # cv2.imwrite(f'{val_ocr_save_path}/ocr_gt_valimg{val_step}_{val_img_id[vis_batch_idx]}.jpg', vis_gt[:,:,::-1])
                                cv2.imwrite(f'{val_ocr_save_path}/ocr_gt_{val_img_id[vis_batch_idx]}.jpg', vis_gt[:,:,::-1])
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
                            }, step=global_step)

    
            # TRYING TO LOG PROMPT TEXT CONDITION
            # if accelerator.is_main_process and cfg.log.tracker.report_to == 'wandb':
            #     wandb.log({"text/text_condition": wandb.Text("\n".join(hq_prompt))}, step=global_step)


            # log 
            logs = {"loss/total_loss": total_loss.detach().item(), 
                    'loss/diff_loss': diff_loss.detach().item(),
                    "optim/lr": lr_scheduler.get_last_lr()[0],
                    }
            
            # ocr log
            if cfg.train.finetune_model in ['dit4sr_testr']:
                logs["loss/ocr_tot_loss"] = ocr_tot_loss.detach().item()
                logs['optim/ts_module_lr'] = cfg.train.learning_rate.ts_module
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
    # parser.add_argument(
    #     "--pretrained_model_name_or_path",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Path to pretrained model or model identifier from huggingface.co/models.",
    # )
    # parser.add_argument(
    #     "--transformer_model_name_or_path",
    #     type=str,
    #     default=None,
    #     help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
    #     " If not specified controlnet weights are initialized from unet.",
    # )
    # parser.add_argument(
    #     "--revision",
    #     type=str,
    #     default=None,
    #     required=False,
    #     help="Revision of pretrained model identifier from huggingface.co/models.",
    # )
    # parser.add_argument(
    #     "--variant",
    #     type=str,
    #     default=None,
    #     help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="controlnet-model",
    #     help="The output directory where the model predictions and checkpoints will be written.",
    # )
    # parser.add_argument(
    #     "--cache_dir",
    #     type=str,
    #     default=None,
    #     help="The directory where the downloaded models and datasets will be stored.",
    # )
    # parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # parser.add_argument(
    #     "--resolution",
    #     type=int,
    #     default=512,
    #     help=(
    #         "The resolution for input images, all the images in the train/validation dataset will be resized to this"
    #         " resolution"
    #     ),
    # )
    # parser.add_argument(
    #     "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    # )
    # parser.add_argument("--num_train_epochs", type=int, default=1000)
    # parser.add_argument(
    #     "--max_train_steps",
    #     type=int,
    #     default=None,
    #     help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    # )
    # parser.add_argument(
    #     "--checkpointing_steps",
    #     type=int,
    #     default=500,
    #     help=(
    #         "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
    #         "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
    #         "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
    #         "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
    #         "instructions."
    #     ),
    # )
    # parser.add_argument(
    #     "--checkpoints_total_limit",
    #     type=int,
    #     default=None,
    #     help=("Max number of checkpoints to store."),
    # )
    # parser.add_argument(
    #     "--resume_from_checkpoint",
    #     type=str,
    #     default=None,
    #     help=(
    #         "Whether training should be resumed from a previous checkpoint. Use a path saved by"
    #         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
    #     ),
    # )
    # parser.add_argument(
    #     "--gradient_accumulation_steps",
    #     type=int,
    #     default=1,
    #     help="Number of updates steps to accumulate before performing a backward/update pass.",
    # )
    # parser.add_argument(
    #     "--gradient_checkpointing",
    #     action="store_true",
    #     help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    # )
    # parser.add_argument(
    #     "--learning_rate",
    #     type=float,
    #     default=5e-6,
    #     help="Initial learning rate (after the potential warmup period) to use.",
    # )
    # parser.add_argument(
    #     "--scale_lr",
    #     action="store_true",
    #     default=False,
    #     help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    # )
    # parser.add_argument(
    #     "--lr_scheduler",
    #     type=str,
    #     default="constant",
    #     help=(
    #         'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
    #         ' "constant", "constant_with_warmup"]'
    #     ),
    # )
    # parser.add_argument(
    #     "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    # )
    # parser.add_argument(
    #     "--lr_num_cycles",
    #     type=int,
    #     default=1,
    #     help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    # )
    # parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    # parser.add_argument(
    #     "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    # )
    # parser.add_argument(
    #     "--num_workers",
    #     type=int,
    #     default=0,
    #     help=(
    #         "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
    #     ),
    # )
    # parser.add_argument(
    #     "--weighting_scheme",
    #     type=str,
    #     default="logit_normal",
    #     choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    # )
    # parser.add_argument(
    #     "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    # )
    # parser.add_argument(
    #     "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    # )
    # parser.add_argument(
    #     "--mode_scale",
    #     type=float,
    #     default=1.29,
    #     help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    # )
    # parser.add_argument(
    #     "--precondition_outputs",
    #     type=int,
    #     default=1,
    #     help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
    #     "model `target` is calculated.",
    # )
    # parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    # parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    # parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    # parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    # parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    # parser.add_argument(
    #     "--hub_model_id",
    #     type=str,
    #     default=None,
    #     help="The name of the repository to keep in sync with the local `output_dir`.",
    # )
    # parser.add_argument(
    #     "--logging_dir",
    #     type=str,
    #     default="logs",
    #     help=(
    #         "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
    #         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
    #     ),
    # )
    # parser.add_argument(
    #     "--allow_tf32",
    #     action="store_true",
    #     help=(
    #         "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
    #         " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
    #     ),
    # )
    # parser.add_argument(
    #     "--report_to",
    #     type=str,
    #     default="tensorboard",
    #     help=(
    #         'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
    #         ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
    #     ),
    # )
    # parser.add_argument(
    #     "--mixed_precision",
    #     type=str,
    #     default=None,
    #     choices=["no", "fp16", "bf16"],
    #     help=(
    #         "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
    #         " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
    #         " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    #     ),
    # )
    # parser.add_argument(
    #     "--set_grads_to_none",
    #     action="store_true",
    #     help=(
    #         "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
    #         " behaviors, so disable this argument if it causes any problems. More info:"
    #         " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
    #     ),
    # )
    # parser.add_argument(
    #     "--dataset_name",
    #     type=str,
    #     default='NOTHING',
    #     help=(
    #         "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
    #         " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
    #         " or to a folder containing files that 🤗 Datasets can understand."
    #     ),
    # )
    # parser.add_argument(
    #     "--dataset_config_name",
    #     type=str,
    #     default=None,
    #     help="The config of the Dataset, leave as None if there's only one config.",
    # )
    # parser.add_argument(
    #     "--train_data_dir",
    #     type=str,
    #     default=None,
    #     help=(
    #         "A folder containing the training data. Folder contents must follow the structure described in"
    #         " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
    #         " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
    #     ),
    # )
    # parser.add_argument(
    #     "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    # )
    # parser.add_argument(
    #     "--conditioning_image_column",
    #     type=str,
    #     default="conditioning_image",
    #     help="The column of the dataset containing the controlnet conditioning image.",
    # )
    # parser.add_argument(
    #     "--caption_column",
    #     type=str,
    #     default="text",
    #     help="The column of the dataset containing a caption or a list of captions.",
    # )
    # parser.add_argument(
    #     "--max_train_samples",
    #     type=int,
    #     default=None,
    #     help=(
    #         "For debugging purposes or quicker training, truncate the number of training examples to this "
    #         "value if set."
    #     ),
    # )
    # parser.add_argument(
    #     "--proportion_empty_prompts",
    #     type=float,
    #     default=0,
    #     help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    # )
    # parser.add_argument(
    #     "--max_sequence_length",
    #     type=int,
    #     default=77,
    #     help="Maximum sequence length to use with with the T5 text encoder",
    # )
    # parser.add_argument(
    #     "--validation_prompt",
    #     type=str,
    #     default=None,
    #     nargs="+",
    #     help=(
    #         "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
    #         " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
    #         " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
    #     ),
    # )
    # parser.add_argument(
    #     "--validation_image",
    #     type=str,
    #     default=None,
    #     nargs="+",
    #     help=(
    #         "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
    #         " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
    #         " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
    #         " `--validation_image` that will be used with all `--validation_prompt`s."
    #     ),
    # )
    # parser.add_argument(
    #     "--num_validation_images",
    #     type=int,
    #     default=4,
    #     help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    # )
    # parser.add_argument(
    #     "--validation_steps",
    #     type=int,
    #     default=100,
    #     help=(
    #         "Run validation every X steps. Validation consists of running the prompt"
    #         " `args.validation_prompt` multiple times: `args.num_validation_images`"
    #         " and logging the images."
    #     ),
    # )
    # parser.add_argument(
    #     "--tracker_project_name",
    #     type=str,
    #     default="train_controlnet",
    #     help=(
    #         "The `project_name` argument passed to Accelerator.init_trackers for"
    #         " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
    #     ),
    # )

    parser.add_argument("--root_folders",  type=str , default='' )
    parser.add_argument("--null_text_ratio", type=float, default=0.5)
    parser.add_argument('--trainable_modules', nargs='*', type=str, default=["control"])

    # # pho
    # parser.add_argument("--tracker_run_name", type=str)
    # parser.add_argument('--data_name', type=str)
    # parser.add_argument('--data_path', type=str)
    # parser.add_argument('--hq_prompt_path', type=str)
    # parser.add_argument('--lq_prompt_path', type=str)
    # parser.add_argument('--vae_model_name_or_path', type=str)
    # parser.add_argument('--finetune', type=str)
    # parser.add_argument('--val_batch_size', type=int)
    # parser.add_argument('--load_precomputed_caption', action='store_true')
    # parser.add_argument('--val_every_step', type=int)
    # parser.add_argument('--model_name', type=str)
    # parser.add_argument('--ts_module_lr', type=float)

    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)


    # breakpoint()

    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    # if args.dataset_name is not None and args.train_data_dir is not None:
    #     raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    # if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
    #     raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    # if args.validation_prompt is not None and args.validation_image is None:
    #     raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    # if args.validation_prompt is None and args.validation_image is not None:
    #     raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    # if (
    #     args.validation_image is not None
    #     and args.validation_prompt is not None
    #     and len(args.validation_image) != 1
    #     and len(args.validation_prompt) != 1
    #     and len(args.validation_image) != len(args.validation_prompt)
    # ):
    #     raise ValueError(
    #         "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
    #         " or the same number of `--validation_prompt`s and `--validation_image`s"
    #     )

    if cfg.model.dit.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    main(cfg)
