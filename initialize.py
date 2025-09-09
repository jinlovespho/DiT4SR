
import os 
import math
import copy
import logging
import diffusers
import accelerate
import transformers
from pathlib import Path
from packaging import version
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
def load_text_encoders(class_one, class_two, class_three, args):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def load_experiment_setting(args, logger):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
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
    if args.seed is not None:
        set_seed(args.seed)
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(f'{args.output_dir}/{args.tracker_run_name}', exist_ok=True)
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1
                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    sub_dir = "transformer"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))
                    i -= 1
        def load_model_hook(models, input_dir):
            # while len(models) > 0:
                # pop models so that they are not loaded again
            model = models.pop()
            load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer") # , low_cpu_mem_usage=False, ignore_mismatched_sizes=True
            # load diffusers style into model
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    return accelerator



def load_data(args):

    # breakpoint()
    if args.data_name == 'satext':
        from basicsr.data.pho_realesrgan_dataset import PhoRealESRGANDataset
        from basicsr.data.pho_realesrgan_dataset import collate_fn_real
        collate_fn = collate_fn_real

        train_ds = PhoRealESRGANDataset(args, mode='train')
        # val_ds = PhoRealESRGANDataset(args, mode='val')
    
    else:
        train_ds = PairedCaptionDataset(root_folder=args.root_folders, null_text_ratio=args.null_text_ratio)
        # val_ds = None 
        collate_fn=None

    
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=args.train_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    # val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, batch_size=args.val_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader=None

    return train_loader, val_loader



def load_model(args, accelerator):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from model_dit4sr.transformer_sd3 import SD3Transformer2DModel
    from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

    models = {}

    # load vae 
    vae = AutoencoderKL.from_pretrained(args.vae_model_name_or_path, subfolder="vae", revision=None)
    vae.requires_grad_(False)
    models['vae'] = vae

    # load scheduler 
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    models['noise_scheduler_copy'] = noise_scheduler_copy

    # load transformer 
    transformer = SD3Transformer2DModel.from_pretrained_local(args.transformer_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant)
    transformer.requires_grad_(False)
    models['transformer'] = transformer

    # load tokenizer 
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )
    # import correct text encoder class
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, args)
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


    # load vlm captioner 
    if not args.load_precomputed_caption:
        from llava.llm_agent import LLavaAgent
        from CKPT_PTH import LLAVA_MODEL_PATH
        llava_agent = LLavaAgent(LLAVA_MODEL_PATH, accelerator.device, load_8bit=True, load_4bit=False)
        models['vlm_agent'] = llava_agent


    return models 


def load_model_params(args, models):

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
        if args.finetune == 'dit4sr_lr_branch':
            if 'control' in name:
                param.requires_grad = True
                train_param_count += numel
                train_param_names.append(name)
            else:
                param.requires_grad = False
                frozen_param_count += numel
                frozen_param_names.append(name)
        

        # other method
        elif args.finetune == 'add other':
            pass

    model_params = {}
    model_params['tot_param_names'] = tot_param_names
    model_params['train_param_names'] = train_param_names
    model_params['frozen_param_names'] = frozen_param_names
    model_params['tot_param'] = tot_param_count
    model_params['train_param'] = train_param_count
    model_params['frozen_param'] = frozen_param_count

    return model_params




def load_optim(args, accelerator, models):

    # scale lr
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    # setup 8bit adam
    if args.use_8bit_adam:
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # load trainable params
    params_to_optimize = list(filter(lambda p: p.requires_grad, models['transformer'].parameters()))

    # optimizer
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    return optimizer


def load_trackers(args, accelerator):
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")
        wandb.login(key='e32eed0c2509bf898b850b0065ab62345005fb73')
        accelerator.init_trackers(
            project_name=args.tracker_project_name,
            config=tracker_config,
            init_kwargs={
                    'wandb':{
                        'name': args.tracker_run_name,}
                }
        )



def set_model_device(args, accelerator, models):

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # place models on cuda 
    for name, model in models.items():
        if isinstance(model, torch.nn.Module):
            model = model.to(accelerator.device, dtype=weight_dtype)
        elif isinstance(model, list):
            if name == 'text_encoders':
                for mod in model:
                    mod = mod.to(accelerator.device, dtype=torch.float16)
        else:
            # leave schedulers, tokenizers, etc. as they are
            pass

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        tmp_models = [models['transformer']]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(tmp_models, dtype=torch.float32)
    
    return weight_dtype