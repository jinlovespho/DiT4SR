
import os 
import math
import copy
import logging
import argparse
import diffusers
import accelerate
import transformers
from pathlib import Path
from packaging import version
from omegaconf import OmegaConf
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
import torch
import torch.utils 
from dataloaders.paired_dataset_sd3_latent import PairedCaptionDataset

import wandb

import bitsandbytes as bnb
from model_dit4sr.transformer_sd3 import SD3Transformer2DModel
from diffusers.training_utils import cast_training_params
from transformers import PretrainedConfig


# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, class_three, cfg):
    text_encoder_one = class_one.from_pretrained(
        cfg.ckpt.init_path.text_encoder, subfolder="text_encoder", revision=None, variant=None
    )
    text_encoder_two = class_two.from_pretrained(
        cfg.ckpt.init_path.text_encoder, subfolder="text_encoder_2", revision=None, variant=None
    )
    text_encoder_three = class_three.from_pretrained(
        cfg.ckpt.init_path.text_encoder, subfolder="text_encoder_3", revision=None, variant=None
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def load_experiment_setting(cfg, logger):
    logging_dir = Path(cfg.save.output_dir, cfg.log.log_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=cfg.save.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision=cfg.train.mixed_precision,
        log_with=cfg.log.tracker.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # If passed along, set the training seed now.
    if cfg.init.seed is not None:
        set_seed(cfg.init.seed)
    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.save.output_dir is not None:
            os.makedirs(f'{cfg.save.output_dir}/{cfg.log.tracker.run_name}', exist_ok=True)
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1
                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    if hasattr(model, "save_pretrained"):  # Hugging Face
                        model.save_pretrained(os.path.join(output_dir, "transformer"))
                    else:  # Bit of a hacky solution to save non-Hugging Face models - e.g., text spotting module
                        ckpt_dict={}
                        unwrapped_model = accelerator.unwrap_model(model)
                        ckpt_dict['ts_module'] = unwrapped_model.state_dict()
                        ckpt_path = f"{output_dir}/ts_module.pt"
                        torch.save(ckpt_dict, ckpt_path)
                    i -= 1
        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()
                if hasattr(model, "register_to_config"):
                    load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer") # , low_cpu_mem_usage=False, ignore_mismatched_sizes=True
                    # load diffusers style into model
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                else:
                    ts_ckpt = torch.load(f'{input_dir}/ts_module.pt', map_location="cpu")
                    load_result = model.load_state_dict(ts_ckpt['ts_module'], strict=False)
                    print('- TS MODULE load result: ', load_result)
                    breakpoint()
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    return accelerator



def load_data(cfg):

    # breakpoint()
    if cfg.data.name == 'satext':
        from basicsr.data.pho_realesrgan_dataset import PhoRealESRGANDataset
        from basicsr.data.pho_realesrgan_dataset import collate_fn_real
        collate_fn = collate_fn_real

        train_ds = PhoRealESRGANDataset(cfg.data, mode='train')
        val_ds = PhoRealESRGANDataset(cfg.data, mode='val')
    
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, batch_size=cfg.val.batch_size, num_workers=cfg.val.num_workers, collate_fn=collate_fn)
    # val_loader = None

    return train_loader, val_loader



def load_model(cfg, accelerator):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from model_dit4sr.transformer_sd3 import SD3Transformer2DModel
    from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

    models = {}

    # load vae 
    vae = AutoencoderKL.from_pretrained(cfg.ckpt.init_path.vae, subfolder="vae", revision=None)
    vae.requires_grad_(False)
    models['vae'] = vae

    # load scheduler 
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(cfg.ckpt.init_path.noise_scheduler, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    models['noise_scheduler'] = noise_scheduler
    models['noise_scheduler_copy'] = noise_scheduler_copy

    # load transformer 
    transformer = SD3Transformer2DModel.from_pretrained_local(cfg.ckpt.init_path.dit, subfolder="transformer", revision=None, variant=None)
    transformer.requires_grad_(False)
    models['transformer'] = transformer

    # load tokenizer 
    tokenizer_one = CLIPTokenizer.from_pretrained(
        cfg.ckpt.init_path.tokenizer,
        subfolder="tokenizer",
        revision=None,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        cfg.ckpt.init_path.tokenizer,
        subfolder="tokenizer_2",
        revision=None,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        cfg.ckpt.init_path.tokenizer,
        subfolder="tokenizer_3",
        revision=None
    )
    # import correct text encoder class
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        cfg.ckpt.init_path.text_encoder, None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        cfg.ckpt.init_path.text_encoder, None, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        cfg.ckpt.init_path.text_encoder, None, subfolder="text_encoder_3"
    )
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, cfg)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    text_encoder_one.eval()
    text_encoder_two.eval()
    text_encoder_three.eval()

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    models['tokenizers'] = tokenizers 
    models['text_encoders'] = text_encoders 

    # load ts module 
    if cfg.train.finetune_model == 'dit4sr_testr':
        from testr.adet.modeling.transformer_detector import TransformerDetector
        from testr.adet.config import get_cfg

        # get testr config
        config_testr = get_cfg()
        config_testr.merge_from_file('./testr/configs/TESTR/TESTR_R_50_Polygon.yaml')
        config_testr.freeze()

        # load testr model
        detector = TransformerDetector(config_testr)

        # load testr pretrained weights
        ckpt = torch.load(cfg.ckpt.init_path.ts_module, map_location="cpu")
        load_result = detector.load_state_dict(ckpt['model'], strict=False)
        
        if accelerator.is_main_process:
            print("Loaded Initial TESTR checkpoint keys:")
            print(" - Initial Missing keys:", load_result.missing_keys)

        models['testr'] = detector.train()


    # load vlm captioner 
    if not cfg.model.dit.load_precomputed_caption:
        from llava.llm_agent import LLavaAgent
        from CKPT_PTH import LLAVA_MODEL_PATH
        llava_agent = LLavaAgent(LLAVA_MODEL_PATH, accelerator.device, load_8bit=True, load_4bit=False)
        models['vlm_agent'] = llava_agent

    return models 


def load_model_params(cfg, models):

    tot_param_names=[]
    train_param_names=[]
    frozen_param_names=[]

    tot_param_count=0
    train_param_count=0
    frozen_param_count=0

    for name, param in models['transformer'].named_parameters():
        numel = param.numel()
        tot_param_count += numel
        tot_param_names.append(name)

        # dit4sr baseline (training only the lr branch)
        if cfg.train.finetune_method == 'dit4sr_lr_branch':
            if 'control' in name:
                param.requires_grad = True
                train_param_count += numel
                train_param_names.append(name)
            else:
                param.requires_grad = False
                frozen_param_count += numel
                frozen_param_names.append(name)
        

        # other method
        elif cfg.train.finetune_method == 'add other':
            pass
    
    if cfg.train.finetune_model == 'dit4sr_testr':
        for name, param in models['testr'].named_parameters():
            # Count total parameters
            numel = param.numel()
            tot_param_count += numel
            tot_param_names.append(f"testr.{name}")

            # Enable training
            param.requires_grad = True
            train_param_count += numel
            train_param_names.append(f"testr.{name}")
            


    model_params = {}
    model_params['tot_param_names'] = tot_param_names
    model_params['train_param_names'] = train_param_names
    model_params['frozen_param_names'] = frozen_param_names
    model_params['tot_param'] = tot_param_count
    model_params['train_param'] = train_param_count
    model_params['frozen_param'] = frozen_param_count

    return model_params




def load_optim(cfg, accelerator, models):

    # scale lr
    if cfg.train.scale_lr:
        cfg.train.learning_rate.dit = (cfg.train.learning_rate.dit * cfg.train.gradient_accumulation_steps * cfg.train.batch_size * accelerator.num_processes)

    # setup optimizer class
    optimizer_class = bnb.optim.AdamW8bit if cfg.train.use_8bit_adam else torch.optim.AdamW


    # separate trainable parameters
    transformer_params = list(filter(lambda p: p.requires_grad, models['transformer'].parameters()))
    testr_params = list(filter(lambda p: p.requires_grad, models['testr'].parameters()))

    # define parameter groups with different LRs
    param_groups = [
        {"params": transformer_params, "lr": cfg.train.learning_rate.dit},
        {"params": testr_params, "lr": cfg.train.learning_rate.ts_module},
    ]

    # # define parameter groups with different LRs
    # param_groups = [
    #     {"params": testr_params, "lr": cfg.train.learning_rate.ts_module},
    # ]

    # optimizer
    optimizer = optimizer_class(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-08,
    )

    # # load trainable params
    # params_to_optimize = list(filter(lambda p: p.requires_grad, models['transformer'].parameters()))

    # # optimizer
    # optimizer = optimizer_class(
    #     params_to_optimize,
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )

    return optimizer


def load_trackers(cfg, accelerator):
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        wandb.login(key=cfg.log.tracker.key)
        accelerator.init_trackers(
            project_name=cfg.log.tracker.project_name,
            config=argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True)),
            init_kwargs={
                    'wandb':{
                        'name': cfg.log.tracker.run_name,}
                }
        )



# def set_model_device(cfg, accelerator, models):

#     # For mixed precision training we cast the text_encoder and vae weights to half-precision
#     # as these models are only used for inference, keeping weights in full precision is not required.
#     weight_dtype = torch.float32
#     if accelerator.mixed_precision == "fp16":
#         weight_dtype = torch.float16
#     elif accelerator.mixed_precision == "bf16":
#         weight_dtype = torch.bfloat16

#     # place models on cuda 
#     for name, model in models.items():
#         if isinstance(model, torch.nn.Module):
#             model = model.to(accelerator.device, dtype=weight_dtype)
#         elif isinstance(model, list):
#             if name == 'text_encoders':
#                 for mod in model:
#                     mod = mod.to(accelerator.device, dtype=torch.float16)
#         else:
#             # leave schedulers, tokenizers, etc. as they are
#             pass

#     # Make sure the trainable params are in float32.
#     if cfg.train.mixed_precision == "fp16":
#         tmp_models = [models['transformer']]
#         # only upcast trainable parameters (LoRA) into fp32
#         cast_training_params(tmp_models, dtype=torch.float32)
    
#     return weight_dtype


def set_model_device(cfg, accelerator, models):

    # Choose dtype for inference-only models
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to device
    for name, model in models.items():
        if isinstance(model, torch.nn.Module):
            models[name] = model.to(accelerator.device, dtype=weight_dtype)
        elif isinstance(model, list) and name == 'text_encoders':
            models[name] = [mod.to(accelerator.device, dtype=weight_dtype) for mod in model]
        else:
            # leave schedulers, tokenizers, etc. as they are
            pass

    # Ensure trainable params are in fp32 (LoRA or finetuning)
    if cfg.train.mixed_precision == "fp16":
        fp32_models = [models['transformer'], models['testr']]
        cast_training_params(fp32_models, dtype=torch.float32)

    return weight_dtype
