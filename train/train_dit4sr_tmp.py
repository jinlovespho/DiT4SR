#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import math
from diffusers.optimization import get_scheduler
import os
import shutil
import sys
sys.path.append(os.getcwd())
import accelerate
import torch
from accelerate.logging import get_logger
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Pipeline,
)

from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory, cast_training_params



import torch.nn.functional as F
from omegaconf import OmegaConf
import wandb 

import initialize 
from train_utils import _encode_prompt_with_t5, _encode_prompt_with_clip, encode_prompt, unwrap_model, compute_text_embeddings, get_sigmas

from dataloaders.utils import realesrgan_degradation

import train_utils 

from torchvision.utils import save_image 
import pyiqa

logger = get_logger(__name__)




from transformers import TrainingArguments, Trainer 





def main(args):

    # set accelerator and basic settings (seed, logging, dir_path)
    accelerator = initialize.load_experiment_setting(args, logger)

    # set training args 
    train_args = TrainingArguments(
        output_dir="pho_dit4sr",
        learning_rate=5e-6,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01,
        # eval_strategy="epoch",
        # save_strategy="epoch",
        # load_best_model_at_end=True,
        push_to_hub=False,
        report_to='none'
    )


    # set model
    models = initialize.load_model(args, accelerator)
    from pho_model import PhoModel
    model = PhoModel(args, models)

    # set data 
    if args.data_name == 'satext':
        from basicsr.data.pho_realesrgan_dataset import PhoRealESRGANDataset
        from basicsr.data.pho_realesrgan_dataset import collate_fn_real
        collate_fn = collate_fn_real

        train_ds = PhoRealESRGANDataset(args, mode='train')
        # val_ds = PhoRealESRGANDataset(args, mode='val')


    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        # eval_dataset=val_ds,
        data_collator=collate_fn,
        # compute_metrics=None,
    )

    trainer.train()

    breakpoint()



    # load model parameters (total_params, trainable_params, frozen_params)
    model_params = initialize.load_model_params(args, models)


    # load optimizer 
    optimizer = initialize.load_optim(args, accelerator, models)


    # place models on cuda and proper weight dtype(float32, float16)
    weight_dtype = initialize.set_model_device(args, accelerator, models)


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(models['transformer'], optimizer, train_dataloader, lr_scheduler)
    vae_img_processor = VaeImageProcessor(vae_scale_factor=8)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    initialize.load_trackers(args, accelerator)


    # # SR metrics
    # metric_psnr = pyiqa.create_metric('psnr', device=accelerator.device)
    # metric_ssim = pyiqa.create_metric('ssimc', device=accelerator.device)
    # metric_lpips = pyiqa.create_metric('lpips', device=accelerator.device)
    # metric_dists = pyiqa.create_metric('dists', device=accelerator.device)
    # # metric_fid = pyiqa.create_metric('fid', device=device)
    # metric_niqe = pyiqa.create_metric('niqe', device=accelerator.device)
    # metric_musiq = pyiqa.create_metric('musiq', device=accelerator.device)
    # metric_maniqa = pyiqa.create_metric('maniqa', device=accelerator.device)
    # metric_clipiqa = pyiqa.create_metric('clipiqa', device=accelerator.device)


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info("=== Model Parameters ===")
    logger.info(f"  Total Params    : {model_params['tot_param']:,} ({model_params['tot_param']/1e6:.2f}M)")
    logger.info(f"  Trainable Params: {model_params['train_param']:,} ({model_params['train_param']/1e6:.2f}M)")
    logger.info(f"  Frozen Params   : {model_params['frozen_param']:,} ({model_params['frozen_param']/1e6:.2f}M)")

    logger.info("=== Training Setup ===")
    logger.info(f"  Num training samples = {len(train_dataloader.dataset)}")
    # logger.info(f"  Num validation samples = {len(val_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    logger.info("=== Parameter Names ===")
    logger.info(f"  Frozen Params ({len(model_params['frozen_param_names'])}):")
    for name in model_params['frozen_param_names']:
        logger.info(f" FROZEN - {name}")
    logger.info(f"  Trainable Params ({len(model_params['train_param_names'])}):")
    for name in model_params['train_param_names']:
        logger.info(f" TRAINING - {name}")


    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(f'{args.output_dir}/{args.tracker_run_name}')
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, args.tracker_run_name, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0


    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process,)
    free_memory()
    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):

            if args.data_name == 'satext':
                batch = realesrgan_degradation(batch)

                gt = batch['gt']
                lq = batch['lq']
                text = batch['text']
                text_enc = batch['text_enc']
                hq_prompt = batch['hq_prompt']
                # lq_prompt = batch['lq_prompt']
                bbox = batch['bbox']
                poly = batch['poly']
                img_id = batch['img_id']
                # save_image(gt, './img_gt.jpg')
                # save_image(lq, './img_lq.jpg')

            with accelerator.accumulate(transformer):

                if args.data_name == 'satext':
                    with torch.no_grad():
                        # hq vae encoding
                        gt = gt * 2.0 - 1.0 # b 3 512 512 
                        hq_latents = models['vae'].encode(gt).latent_dist.sample()  # b 16 64 64
                        model_input = (hq_latents - models['vae'].config.shift_factor) * models['vae'].config.scaling_factor    # b 16 64 64 
                        # lq vae encoding
                        lq = lq * 2.0 - 1.0 # b 3 512 512 
                        lq_latents = models['vae'].encode(lq).latent_dist.sample()  # b 16 64 64 
                        controlnet_image = (lq_latents - models['vae'].config.shift_factor) * models['vae'].config.scaling_factor   # b 16 64 64 
                        controlnet_image = controlnet_image.to(dtype=weight_dtype)
                        # load caption
                        if args.load_precomputed_caption:
                            hq_prompt = hq_prompt 
                        else:
                            # vlm captioner
                            lq_tmp = F.interpolate(lq, size=(336, 336), mode="bilinear", align_corners=False)
                            hq_prompt = models['vlm_agent'].gen_image_caption(lq_tmp)
                            hq_prompt = [train_utils.remove_focus_sentences(p) for p in hq_prompt]
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
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * models['noise_scheduler_copy'].config.num_train_timesteps).long()
                timesteps = models['noise_scheduler_copy'].timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, accelerator, models['noise_scheduler_copy'], n_dim=model_input.ndim, dtype=model_input.dtype)    # b 1 1 1
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise   # b 16 64 64 
                # Predict the noise residual
                model_pred = transformer(                       # b 16 64 64
                    hidden_states=noisy_model_input,            # b 16 64 64 
                    controlnet_image=controlnet_image,          # b 16 16 16  
                    timestep=timesteps,                         # b
                    encoder_hidden_states=prompt_embeds,        # b 154 4096
                    pooled_projections=pooled_prompt_embeds,    # b 2048
                    return_dict=False,
                )[0]

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                if args.precondition_outputs:   # t
                    model_pred = model_pred * (-sigmas) + noisy_model_input # b 16 64 64

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)   # b 1 1 1

                # flow matching loss
                if args.precondition_outputs:   # t
                    target = model_input
                else:
                    target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # params_to_clip = controlnet.parameters()
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        ckpt_save_path = f'{args.output_dir}/{args.tracker_run_name}'
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(ckpt_save_path)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(ckpt_save_path, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, args.tracker_run_name, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step==1 or global_step % args.val_every_step == 0:
                        # visualize the restored image from the transformer output
                        with torch.no_grad():
                            # Option A: precondition_outputs=True → model_pred ≈ denoised latents
                            restored_latents = model_pred

                            # Option B: precondition_outputs=False → need to reconstruct denoised latents
                            if not args.precondition_outputs:
                                restored_latents = noisy_model_input - sigmas * model_pred

                            # Decode latents into images
                            restored_imgs = models['vae'].decode(restored_latents).sample
                            restored_imgs = vae_img_processor.postprocess(restored_imgs, output_type="pil")  

                            # Save as PNG
                            val_save_path = f'{args.output_dir}/{args.tracker_run_name}/restored_img'
                            os.makedirs(val_save_path, exist_ok=True)
                            val_save_imgs = zip(restored_imgs, img_id)
                            for res_img, id in val_save_imgs:
                                res_img.save(f'{val_save_path}/step{global_step}_{id}.png')              


            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--transformer_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='NOTHING',
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument("--root_folders",  type=str , default='' )
    parser.add_argument("--null_text_ratio", type=float, default=0.5)
    parser.add_argument('--trainable_modules', nargs='*', type=str, default=["control"])

    # pho
    parser.add_argument("--tracker_run_name", type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--hq_prompt_path', type=str)
    parser.add_argument('--lq_prompt_path', type=str)
    parser.add_argument('--vae_model_name_or_path', type=str)
    parser.add_argument('--finetune', type=str)
    parser.add_argument('--val_batch_size', type=int)
    parser.add_argument('--load_precomputed_caption', action='store_true')
    parser.add_argument('--val_every_step', type=int)
    
    
    args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    main(args)
