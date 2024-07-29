from collections import OrderedDict
# import cv2

# Bootstrapped from:
from utils import generate_mask_ml,generate_spatial_rendering_ml
from utils import visualize_box,numpy_to_pil
import numpy as np
import sys
sys.path.insert(0, './packages')
import itertools
import math
import os
import inspect
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import pdb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    UNet2DModel
)

from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, AutoProcessor
from lora_diffusion import (
    extract_lora_ups_down,
    inject_trainable_lora,
    inject_trainable_lora_extended,
    safetensors_available,
    save_lora_weight,
    save_safeloras_with_embeds,
)
from lora_diffusion.xformers_utils import set_use_memory_efficient_attention_xformers
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import random
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
from lora_diffusion import tune_lora_scale, patch_pipe
import torchvision.transforms as T
import inspect
from azureml.core import Run
run = Run.get_context()
import socket
from data_utils import cycle, create_wbd

hostname = socket.gethostname()

from datasets.ocr_dataset_ml_zeroshot import OCRDatasetMLZeroShot, mask_transforms
from config import parse_args
import cv2
from torch import nn
import torchvision.ops.roi_align as roi_align



image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = get_logger(__name__)


def save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
            .get_input_embeddings()
            .weight[placeholder_token_id]
    )
    print("Current Learned Embeddings: ", learned_embeds[:4])
    print("saved to ", save_path)
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)

def freeze_params(params):
    for param in params:
        param.requires_grad = False
def load_mario_batch(mario_loader,tokenizer):
    mario_data=next(mario_loader)
    return mario_data


def unfreeze_params(params):
    for param in params:
        param.requires_grad = True
def load_laion_batch(laion_loader,tokenizer):
    laion_batch={}
    laion_data = next(laion_loader)
    # print(type(laion_data),'laion_data',type(laion_loader),'type(laion_loader)')
    laion_images = laion_data['image_tensor'].cuda()
    bsz=len(laion_images)
    laion_ids = laion_data['text_tokens'].squeeze(1).cuda()
    # masks
    masks=[]
    for _ in range(bsz):
        mask = np.zeros((512, 512)).astype(np.uint8)
        mask = Image.fromarray(mask)
        # mask=mask_transforms(mask)
        mask=mask_transforms(mask).float()
        masks.append(mask)
    masks = torch.stack(masks)
    
    # rendering
    spatial_renderings_rgb=[image_transforms(Image.new("RGB", (512, 512), (255, 255, 255)))]*bsz
    spatial_renderings_rgb = torch.stack(spatial_renderings_rgb)
    text_boxes=[]
    char_gts=[]
    text_idxs=[False]*bsz
    synth_idxs=[False]*bsz

    laion_batch = {
            "pixel_values": laion_images,
            "input_ids": laion_ids,
            "masks": masks,
            "spatial_renderings_rgb": spatial_renderings_rgb,
            "text_boxes": text_boxes,
            "char_gts": char_gts,
            "text_idxs": text_idxs,
            "synth_idxs": synth_idxs,
            }
    return laion_batch

    

import json
def main(args):
    moving_avg=0
    sample_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
            args.train_text_encoder
            and args.gradient_accumulation_steps > 1
            and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )
    # Local Machine Logging
    if accelerator.is_main_process:
        if (hostname == 'ubuntu' or hostname.startswith('Qlab')):
            viz_dir = os.path.join(args.output_dir,'viz')
            os.makedirs(viz_dir, exist_ok=True)
            codepath=os.path.join(args.output_dir,'src')
            if 'tmp' not in args.output_dir:
                assert not os.path.exists(codepath)
            os.makedirs(codepath,exist_ok=True)
            os.system('cp *.py {}'.format(codepath))
            os.system('cp datasets {} -R'.format(codepath))
            os.system('cp packages {} -R'.format(codepath))
            os.system('cp rendering {} -R'.format(codepath))
            # 1. command
            command_path=os.path.join(codepath,'command.txt')
            command_file=open(command_path,'w')
            command_file.write('cwd\t{}\n'.format(os.getcwd()))
            print(command_path,'command_path')
            idx=0
            while idx<len(sys.argv):
                item=sys.argv[idx]
                print(item,'item')
                command_file.write('{}\n'.format(item))
                idx+=1
            command_file.close()

    if args.seed is not None:
        set_seed(args.seed)
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    """CLIP initialization"""

    # """original text_encoder"""
    # force_down=False
    # if not (hostname == 'ubuntu' or hostname.startswith('Qlab') or hostname.startswith('qlab')):
    #     force_down=True
    # else:
    #     force_down=False
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        # "stabilityai/stable-diffusion-2-1",
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        # "stabilityai/stable-diffusion-2-1",
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )




    """VAE Initialization"""
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        # "stabilityai/stable-diffusion-2-1",
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )
    """UNet Initialization"""
    print(inspect.getsourcefile(UNet2DConditionModel.from_pretrained), 'inspect')
    unet, load_info = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        # 'stabilityai/stable-diffusion-2-1',
        subfolder="unet",
        revision=args.revision,
        output_loading_info=True,
        in_channels=4,
        device_map=None,
        low_cpu_mem_usage=False
    )
    for param in unet.parameters():
        param.requires_grad = False

    unet_new_params = []
    unet_new_params_name = []

    unet.requires_grad_(False)
    unet_lora_params, unet_lora_params_name = inject_trainable_lora_extended(unet, r=args.lora_rank, verbose=True)

    init_scale = 0.01
    for k, v in unet.named_parameters():
        if 'conv_in2' in k:
            if 'weight' in k:
                v.data.normal_(0, init_scale)
            elif 'bias' in k:
                v.data.zero_()
            print(k, 'New params')
            v.requires_grad = True
            unet_new_params.append(v)
            unet_new_params_name.append(k)

    print('New added parameters:')
    print(unet_new_params_name)
    print('New lora parameters:')
    print(unet_lora_params_name)
    vae.requires_grad_(False)
    if args.mlt_config=='latin':
        from datasets.script_info_latin import scriptInfo
    elif args.mlt_config=='no_russian':
        from datasets.script_info_no_russian import scriptInfo
    elif args.mlt_config=='all':
        from datasets.script_info_all import scriptInfo
    elif args.mlt_config=='no_greek':
        from datasets.script_info_no_greek import scriptInfo
    else:
        print(args.mlt_config,'mlt_config')
        assert False
    script_info=scriptInfo(charset_path=args.charset_path)

    from textrecnet_ml import TextRecNetML
    recnet = TextRecNetML(in_channels=4, out_dim=len(script_info.char2idx), max_char=args.max_char_seq_length)    
    
    if args.use_xformers:
        set_use_memory_efficient_attention_xformers(unet, True)
        set_use_memory_efficient_attention_xformers(vae, True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        # if args.train_text_encoder:

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate
                * args.gradient_accumulation_steps
                * args.train_batch_size
                * accelerator.num_processes
        )
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    params_to_optimize_unet = [
        # {"params": unet_new_params + fusion_module_params + list(clip_vision_proj.parameters()), "lr": args.learning_rate},
        # {"params": unet_new_params , "lr": args.learning_rate},
        # {"params": unet_new_params + list(clip_model.parameters()) , "lr": args.learning_rate},
        {"params": itertools.chain(*unet_lora_params), "lr": args.learning_rate},
        {"params": unet_new_params, "lr": args.learning_rate},
        {"params": recnet.parameters(), "lr":  args.learning_rate_text},
        # {"params": clip_model.parameters() , "lr": args.learning_rate/1.0},
        # {"params": unet_new_params , "lr": args.learning_rate/10.0},
        # {"params": fusion_module_params, "lr": args.learning_rate},
    ]
    

    optimizer = optimizer_class(
        params_to_optimize_unet,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(
        # "stabilityai/stable-diffusion-2-1", 
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )
    
    if not (hostname == 'ubuntu' or hostname.startswith('Qlab')):
        mario_db=['mario_laion']
    else:
        mario_db=['mario_laion500K_ml']
    train_dataset = OCRDatasetMLZeroShot(
        instance_data_root=args.instance_data_root,
        tokenizer=tokenizer,
        mask_all_ratio=args.mask_all_ratio,
        max_word_len=25,#output_gt will be in (max_word_len+1) dim
        charset_path=args.charset_path,#output_gt will be in (max_word_len+1) dim
        script_info=script_info,#output_gt will be in (max_word_len+1) dim
        synth_ratio=args.synth_ratio,#output_gt will be in (max_word_len+1) dim
        mario_db=mario_db[0],#output_gt will be in (max_word_len+1) dim
        mario_prob=args.mario_prob
    )
    
    if args.mario_batch_size:
        from datasets.ocr_dataset_mario import OCRDataset
        train_dataset_mario = OCRDataset(
            instance_data_list=mario_db,
            instance_data_root=args.instance_data_root,
            tokenizer=tokenizer,
            max_word_len=25,#output_gt will be in (max_word_len+1) dim
            script_info=script_info,
        )


    def collate_fn(examples):
        nonlatin_idxs=[]
        for example in examples:
            nonlatin_idxs +=example["instance_nonlatin_idxs"] 
            # print(len(example["instance_nonlatin_idxs"]),'instance_nonlatin_idxs')
        synth_idxs=[]
        for example in examples:
            synth_idxs +=example["instance_synth_idx"] 

        input_ids = [example["instance_prompt_ids"] for example in examples]
        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        pixel_values = [example["instance_images"] for example in examples]
        spatial_renderings_rgb = [example["instance_spatial_rendering_rgb"] for example in examples]
        masks = [example["instance_mask"] for example in examples]
        pixel_values = torch.stack(pixel_values)
        spatial_renderings_rgb = torch.stack(spatial_renderings_rgb)
        masks = torch.stack(masks)
        char_gts=[]
        for eidx,example in enumerate(examples):
            char_gts+=example["instance_char_gts"]#returns: T,26
        char_gts = torch.stack(char_gts) #shape N,T,26
        text_idxs=[]
        text_boxes = []
        for example in examples:
            text_idxs.append(len(example["instance_text_boxes"])>0)
            if len(example["instance_text_boxes"]):
                # NOTE!!!!!!!
                text_boxes.append(example["instance_text_boxes"])
                
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        is_latins=[]
        for example in examples:
            is_latins+=example['is_latin']
        # load laion data
        batch = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "masks": masks,
            "spatial_renderings_rgb": spatial_renderings_rgb,
            "text_boxes": text_boxes,
            "char_gts": char_gts,
            "text_idxs": text_idxs,
            "synth_idxs": synth_idxs,
            "nonlatin_idxs": nonlatin_idxs,
            "is_latins": is_latins,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    if args.mario_batch_size:
        mario_loader = torch.utils.data.DataLoader(
            train_dataset_mario,
            batch_size=args.mario_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
        )
        mario_loader = cycle(mario_loader)

    args.data_list = ['laion-aes']
    args.img_size = args.resolution
    args.batch_size = args.laion_batch_size
    # args.root_path = args.instance_data_root
    if args.laion_batch_size:
        laion_loader = create_wbd(args, load_txt=True, tokenizer=tokenizer)
        laion_loader = cycle(laion_loader)


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    (
        unet,
        train_dataloader,
        recnet,
    ) = accelerator.prepare(unet, 
                            train_dataloader,
                            recnet
                            )
    
    if args.resume_unet_ckpt_path and args.resume_unet_ckpt_path!='None':
        state_dict = torch.load(args.resume_unet_ckpt_path, map_location=torch.device('cpu'))
        if not isinstance(state_dict,OrderedDict):
            state_dict=state_dict()
        unet.load_state_dict(state_dict)
        print('unet parameters loaded')
        del state_dict
    if args.resume_recnet_ckpt_path and args.resume_recnet_ckpt_path!='None':
        saved_state_dict = torch.load(args.resume_recnet_ckpt_path, map_location=torch.device('cpu'))
        defined_state_dict=recnet.state_dict()
        defined_keys=list(defined_state_dict.keys())
        print(len(defined_keys),'len(defined_keys)')
        for key in defined_keys:
            print(key,'defined_key')
        new_state_dict={}
        exist_count=0
        for saved_key in saved_state_dict:
            print(saved_key,'recnet_saved_key')
            if saved_key in defined_keys:
                exist_count+=1
                print('saved_key EXISTS',saved_key in defined_keys)
            else:
                print('saved_key NOT IN',saved_key in defined_keys)
            if not args.rec_strict_load:
                if 'final' in saved_key:
                    continue
            new_state_dict[saved_key]=saved_state_dict[saved_key]
        print((exist_count/len(saved_state_dict)),'exist_ratio')
        recnet.load_state_dict(new_state_dict,strict=args.rec_strict_load)
        if args.rec_strict_load:
            print('recnet strictly loaded')
            assert (exist_count/len(saved_state_dict))==1

        else:
            print('recnet NOTSTRICTLY loaded')
            assert (exist_count/len(saved_state_dict))>0.9


        del saved_state_dict

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = (
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
    )
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0
    accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                            inspect.signature(
                                accelerator.unwrap_model
                            ).parameters.keys()
                        )
    extra_args = (
        {"keep_fp32_wrapper": True}
        if accepts_keep_fp32_wrapper
        else {}
    )
    std_pipeline = StableDiffusionPipeline.from_pretrained( 
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet, **extra_args),
                            tokenizer=accelerator.unwrap_model(tokenizer, **extra_args))
    std_scheduler = std_pipeline.scheduler
    std_fe_extractor = std_pipeline.feature_extractor
    del std_pipeline
    if not (hostname == 'ubuntu' or hostname.startswith('Qlab')):
        display_steps = 100
    else:
        display_steps = 1
    print('Start Training')
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            unet.train()
            
            input_images=batch["pixel_values"].to(accelerator.device,dtype=weight_dtype)
            # print(input_images.shape,'input_images.shape')
            masks = batch["masks"].to(accelerator.device,dtype=weight_dtype)
            spatial_renderings_rgb=batch["spatial_renderings_rgb"].to(accelerator.device,dtype=weight_dtype)
            input_ids = batch["input_ids"].to(accelerator.device)
            text_idxs = batch["text_idxs"]
            if args.laion_batch_size:
                # Laion data
                laion_batch=load_laion_batch(laion_loader,tokenizer)
                input_images=torch.cat((input_images,laion_batch["pixel_values"].to(accelerator.device,dtype=weight_dtype)),0)
                masks=torch.cat((batch["masks"].to(accelerator.device,dtype=weight_dtype),laion_batch["masks"].to(accelerator.device,dtype=weight_dtype)),0)
                spatial_renderings_rgb=torch.cat((batch["spatial_renderings_rgb"].to(accelerator.device,dtype=weight_dtype),laion_batch["spatial_renderings_rgb"].to(accelerator.device,dtype=weight_dtype)),0)
                input_ids=torch.cat((batch["input_ids"].to(accelerator.device),
                                     laion_batch["input_ids"].to(accelerator.device)),0).to(accelerator.device)
                text_idxs=text_idxs+laion_batch["text_idxs"]
            if args.mario_batch_size:
                # Mario data
                mario_batch=load_mario_batch(mario_loader,tokenizer)
                input_images=torch.cat((input_images,mario_batch["pixel_values"].to(accelerator.device,dtype=weight_dtype)),0)
                masks = torch.cat((masks,mario_batch["masks"].to(accelerator.device,dtype=weight_dtype)),0)
                spatial_renderings_rgb=torch.cat((spatial_renderings_rgb,mario_batch["spatial_renderings_rgb"].to(accelerator.device,dtype=weight_dtype)),0)
                input_ids = torch.cat((input_ids,mario_batch["input_ids"].to(accelerator.device)),0)
                text_idxs=text_idxs+mario_batch["text_idxs"]
            
            masks_64 = torch.nn.functional.interpolate(masks, size=(64, 64))
            with torch.no_grad():
                # vae
                latents = vae.encode(input_images.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215
                rendering_emb = vae.encode(spatial_renderings_rgb.to(dtype=weight_dtype)).latent_dist.sample()
                rendering_emb = rendering_emb * 0.18215
            noise = torch.randn_like(latents)  # a normal distribution with mean 0 and variance 1
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  # L'=a(t)L+b(t) #([1, 4, 64, 113])
            unet_input = torch.cat([noisy_latents, masks_64, rendering_emb], dim=1)

            clip_text_embedding = text_encoder(input_ids)[0].to(accelerator.device, dtype=weight_dtype)
            drop_mask = (torch.rand(bsz)>0.1)*1.0
            drop_mask = drop_mask.view(-1, 1, 1, 1).to(accelerator.device)
            unet_input[:, 4:, :, :] *= drop_mask
            import time
            import numpy as np
            model_pred = unet(unet_input, timesteps, clip_text_embedding).sample
            if noise_scheduler.config.prediction_type == "epsilon":  # CHECK
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )
            """ignore this for now : with_prior_preservation"""
            synth_idxs=batch["synth_idxs"]
            if args.laion_batch_size:
                synth_idxs+=([False]*args.laion_batch_size)
            if args.mario_batch_size:
                synth_idxs+=([False]*args.mario_batch_size)
            synth_idxs=torch.BoolTensor(synth_idxs)
            real_idxs=torch.logical_not(synth_idxs)
            if args.synth_unet_loss_weight==0:
                # Totally exclude from the unet_loss
                # only include real image for unet mse loss
                loss_unet = F.mse_loss(model_pred[real_idxs].float(), target[real_idxs].float(), reduction="none")
                loss_unet=torch.mean(loss_unet)
            else:
                loss_unet = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss_unet[synth_idxs]*=args.synth_unet_loss_weight
                loss_unet=torch.mean(loss_unet)
                    

            if noise_scheduler.config.prediction_type == "epsilon":  # CHECK
                pred_x0 = noise_scheduler.get_x0_from_noise(model_pred, timesteps, noisy_latents) # pred_x0 corresponds to (latents*0.18215)

            elif noise_scheduler.config.prediction_type == "v_prediction":
                pred_x0 = noise_scheduler.get_x0_from_velocity(model_pred, timesteps, noisy_latents) # pred_x0 corresponds to (latents*0.18215)
            else:
                assert False

            
            text_rec_loss_recnet=None
            text_boxes=batch["text_boxes"]
            nonlatin_idxs=batch["nonlatin_idxs"]
            if args.mario_batch_size:
                text_boxes=text_boxes+mario_batch["text_boxes"]
                nonlatin_idxs+=nonlatin_idxs+mario_batch["nonlatin_idxs"]
                # print(len(mario_batch["nonlatin_idxs"]),'mario_batch["nonlatin_idxs"]')
            if moving_avg==0:
                moving_avg=np.sum(nonlatin_idxs)/len(nonlatin_idxs)
            else:
                moving_avg=moving_avg*0.9+0.1*np.sum(nonlatin_idxs)/len(nonlatin_idxs)
            if len(text_boxes):
                for idx in range(len(text_boxes)):
                    text_boxes[idx]=text_boxes[idx].cuda()
                assert np.sum(text_idxs)!=0
                assert len(latents[text_idxs])==len(text_boxes)
                assert len(pred_x0[text_idxs])==len(text_boxes)
                text_gts=batch["char_gts"].long().to(accelerator.device)
                if args.mario_batch_size:
                    text_gts=torch.cat((text_gts, mario_batch["char_gts"].long().to(accelerator.device)),0)
                    
                ce_criterion = torch.nn.CrossEntropyLoss()
                # NOTE!!!!!!!
                roi_features = roi_align((pred_x0[text_idxs]).to(device=accelerator.device),text_boxes,output_size=(32,128),spatial_scale=64/512).to(device=accelerator.device,dtype=weight_dtype)
                # roi_features_original = roi_align((latents[text_idxs]).to(device=accelerator.device),text_boxes,output_size=(32,128),spatial_scale=64/512).to(device=accelerator.device,dtype=weight_dtype)
                # text_preds_original = recnet(roi_features_original)
                text_preds_x0 = recnet(roi_features)
                text_rec_loss = ce_criterion(text_preds_x0.view(-1,text_preds_x0.shape[-1]), text_gts.contiguous().view(-1))#*0.05
                loss_unet = loss_unet + text_rec_loss*args.text_rec_loss_weight
            if accelerator.sync_gradients:  # True when distributed
                params_to_clip = (
                    itertools.chain(unet.parameters())
                )
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            accelerator.backward(loss_unet)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            if global_step % display_steps == 0:
                progress_bar.update(display_steps)
                if (not (hostname == 'ubuntu' or hostname.startswith('Qlab'))) and accelerator.is_main_process and global_step % display_steps == 0:
                    run.log('loss_unet', loss_unet.detach().item())
                    if text_rec_loss_recnet is not None:
                        run.log('text_rec_loss_recnet', text_rec_loss_recnet.detach().item()*args.text_rec_loss_weight)
                    if text_rec_loss is not None:
                        run.log('text_rec_loss', text_rec_loss.detach().item()*args.text_rec_loss_weight)
                    run.log('lr', lr_scheduler.get_last_lr()[0])
                    run.log('lr_text', lr_scheduler.get_last_lr()[-1])
                    run.log('nonlatin_ratio', moving_avg)

            # Visualization
            timesteps=timesteps.detach().cpu().numpy()[text_idxs]
            maxstep=0
            viz_idx=0
            for step_idx,step in enumerate(timesteps):
                if step>maxstep:
                    maxstep=step
                    viz_idx=step_idx
            if (global_step%args.visualize_steps)==0 and accelerator.is_main_process and (hostname == 'ubuntu' or hostname.startswith('Qlab')):
                
                input_images=input_images[text_idxs]
                spatial_renderings_rgb=spatial_renderings_rgb[text_idxs]
                masks=masks[text_idxs]
                pred_x0=pred_x0[text_idxs]
                # 1. visualize x0
                pred_x0 = (1 / vae.config.scaling_factor) * pred_x0
                pred_x0=pred_x0[viz_idx].unsqueeze(0).to(vae.device,dtype=weight_dtype)
                image_x0 = vae.decode(pred_x0).sample
                image_x0 = (image_x0 / 2 + 0.5).clamp(0, 1)
                image_x0 = image_x0.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                image_x0=numpy_to_pil(image_x0)[0]
                image_x0=visualize_box(image_x0,text_boxes[viz_idx])
                image_x0.save(os.path.join(viz_dir,'{:05d}_tstep{:04d}_image_x0_idx{}.jpg'.format(global_step,timesteps[viz_idx],viz_idx))) #good

                # 2. visualize input
                input_images = (input_images / 2 + 0.5).clamp(0, 1)
                input_np = input_images.detach().cpu().permute(0, 2, 3, 1).float().numpy()[viz_idx]
                input_pil=numpy_to_pil(input_np)[0]
                input_pil=visualize_box(input_pil,text_boxes[viz_idx])

                # 3. visualize mask
                image_mask=masks.detach().cpu().permute(0, 2, 3, 1).float().numpy()[viz_idx] # 0 ~ 1 range
                image_mask=cv2.resize(image_mask,dsize=(512,512),interpolation=cv2.INTER_NEAREST)
                Image.fromarray(image_mask*255).convert('RGB').save(os.path.join(viz_dir,'{:05d}_tstep{:04d}_mask_idx{}.jpg'.format(global_step,timesteps[viz_idx],viz_idx)))
                box_viz_idx=0
                for box_iter_idx,boxes in enumerate(text_boxes):
                    if box_iter_idx==viz_idx:
                        break
                    box_viz_idx+=len(boxes)
                input_pil.save(os.path.join(viz_dir,'{:05d}_tstep{:04d}_image_input_idx{}.jpg'.format(global_step,timesteps[viz_idx],viz_idx)))
                # 4. spatial rendering
                rendered_whole_np = spatial_renderings_rgb.detach().cpu().permute(0, 2, 3, 1).float().numpy()[viz_idx]
                rendered_whole_np=(rendered_whole_np*0.5)+0.5
                rendered_whole_pil=Image.fromarray((rendered_whole_np*255).astype(np.uint8)).convert('RGB')
                rendered_whole_pil=visualize_box(rendered_whole_pil,text_boxes[viz_idx])
                rendered_whole_pil.save(os.path.join(viz_dir,'{:05d}_tstep{:04d}_image_whole_rendered_idx{}.jpg'.format(global_step,timesteps[viz_idx],viz_idx)))

            if (global_step%args.visualize_steps)==0 and accelerator.is_main_process:    
                # # # # # # # # # # # # # # # # 
                # 5. Print Recognition results
                # # # # # # # # # # # # # # # #
                start_idx=0
                end_idx=0
                """Get word indexes corresponding to the instance saved for X0 visualiation"""
                for idx in range(len(text_boxes)):
                    if idx==viz_idx:
                        end_idx+=len(text_boxes[idx])
                        break
                    start_idx+=len(text_boxes[idx])
                    end_idx+=len(text_boxes[idx])
                text_preds_x0=text_preds_x0.detach().cpu().numpy()# (B,L,96)
                # (B,L, 96) text_preds.shape
                text_preds_x0=np.argmax(text_preds_x0,2) # (B,L,96)->(B,L)
                word_preds=[]
                for char_idxs in text_preds_x0:
                    word_chars=[]
                    for char_idx in char_idxs:
                        char_idx=int(char_idx)
                        char=script_info.idx2char[char_idx]
                        word_chars.append(char)
                    word=''.join(word_chars)
                    word_preds.append(word[:word.find('[s]')])
                word_gts=[]
                # (B,L) text_gts.shape
                text_gts=text_gts.detach().cpu().numpy().astype(np.int32)# (B,L)
                for char_idxs in text_gts:
                    word=[]
                    for char_idx in char_idxs:
                        char=script_info.idx2char[char_idx]
                        word.append(char)
                    word=''.join(word)
                    word_gts.append(word[:word.find('[s]')])
                dashed_line= '-' * 80
                print()
                print(dashed_line)
                print(dashed_line)
                print('Results at {:05d}'.format(global_step))
                print(dashed_line)
                print('{:25s} | {:25s} | {:25s}'.format("Ground Truth","Prediction","T/F"))
                print(dashed_line)
                for gt, pred, in zip(word_gts[start_idx:end_idx], word_preds[start_idx:end_idx]):
                    print('{:25s} | {:25s} | {:20s}'.format(gt[:25],pred[:25],str(gt==pred)))
                print(dashed_line)
                print(dashed_line)
                print()
                # # # # # # # # # # # # # # # # 
                # Print Recognition results
                # # # # # # # # # # # # # # # # 

            global_step += 1
            if accelerator.sync_gradients:
                # if (args.save_steps and global_step - last_save >= args.save_steps) or (global_step<=1):
                # print(global_step-1,'global_step',(global_step - 1) % args.inference_steps,'HERE')
                if ((global_step - 1) % args.inference_steps) == 0:    
                    unet.eval()
                    if accelerator.is_main_process:
                        accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                            inspect.signature(
                                accelerator.unwrap_model
                            ).parameters.keys()
                        )
                        extra_args = (
                            {"keep_fp32_wrapper": True}
                            if accepts_keep_fp32_wrapper
                            else {}
                        )
                        pipeline = StableDiffusionPipeline(
                            vae=accelerator.unwrap_model(vae, **extra_args),
                            unet=accelerator.unwrap_model(unet, **extra_args),
                            text_encoder=accelerator.unwrap_model(text_encoder, **extra_args),
                            tokenizer=accelerator.unwrap_model(tokenizer, **extra_args),
                            scheduler=accelerator.unwrap_model(std_scheduler, **extra_args),
                            feature_extractor=accelerator.unwrap_model(std_fe_extractor, **extra_args),
                            safety_checker=None,
                            requires_safety_checker=False,
                        )

                        # glyph_texts = ["بعقد.","éçlameùr","तेवरमों","äußert"]
                        # inference_lang_list=['arabic','french','hindi','german']
                        # glyph_texts = ["διακπές","äußert","πύκνωσε","einöni"]
                        # inference_lang_list=['greek','german','greek','german']
                        glyph_texts_batch = [["пдвржен"],["éçlameùr"],["বৃষস্কন্ধ"],["περνούτ"]]
                        inference_lang_list_batch=[['russian'],['french'],['bengali'],['greek']]
                        # glyph_texts_batch = [["περλφύτ"],["بعقد."],["ธนานุบาล"],["äußert"]]
                        # inference_lang_list_batch=[['greek'],["arabic"],["thai"],["german"]]
                        # glyph_texts_batch = [["пдвржен"],["Хорошо"],["Извините"],["Кошка"],['περλφύτ']]
                        # inference_lang_list_batch=[["russian"],["russian"],["russian"],["russian"],["greek"]]
                        assert len(glyph_texts_batch)==len(inference_lang_list_batch)
                        # Inference
                        viz_rendering_list=[]
                        viz_mask_list=[]
                        rendering_batch=[]
                        mask_batch=[]
                        # glyph_texts_batch=glyph_texts_batch
                        prompt_batch=["A dog holding a paper saying a word"]*len(glyph_texts_batch)

                        for gen_idx in range(len(glyph_texts_batch)):
                            glyph_texts=glyph_texts_batch[gen_idx]
                            inference_lang_list=inference_lang_list_batch[gen_idx]
                            inference_scripts_list=[script_info.lang2script[lang]for lang in inference_lang_list]
                            coords=np.array([[100, 145,390, 265]])
                            coords=coords.astype(np.int32).tolist()
                            random_mask_np=generate_mask_ml(height=512,width=512,coords=coords,mask_type='rectangle')
                            random_mask=Image.fromarray(random_mask_np)
                            masks_tensor = mask_transforms(random_mask).float().unsqueeze(0).to(accelerator.device,dtype=weight_dtype)
                            masks_64 = torch.nn.functional.interpolate(masks_tensor, size=(64, 64), mode='nearest')
                            viz_mask=Image.fromarray(np.array(random_mask_np*255, np.uint8))
                            spatial_renderings=generate_spatial_rendering_ml(width=512,height=512,words=glyph_texts,dst_coords=coords,lang_list=inference_lang_list)
                            rendering_drawn=visualize_box(spatial_renderings.convert("RGB"),coords)
                            spatial_renderings_rgb=image_transforms(spatial_renderings.convert("RGB"))
                            spatial_renderings_rgb=spatial_renderings_rgb.unsqueeze(0).to(accelerator.device,dtype=weight_dtype)
                            rendering_batch.append(spatial_renderings_rgb)
                            mask_batch.append(masks_64)
                            viz_rendering_list.append(rendering_drawn)
                            viz_mask_list.append(viz_mask)
                        with torch.no_grad():
                            if args.debug:
                                num_inference_steps=2
                            else:
                                num_inference_steps=args.num_inference_steps
                            mask_batch=torch.cat(mask_batch)
                            rendering_batch=torch.cat(rendering_batch)
                            images = pipeline(prompt=prompt_batch, 
                            spatial_render=rendering_batch, masks_tensor=mask_batch, #not inverted
                            num_inference_steps=num_inference_steps, 
                            guidance_scale=7, width=512, height=512).images
                        
                        
                        merged_viz = Image.new('RGB', (512*2*4, 512*1), (255, 255, 255)) # SAME
                        for idx, (image,rend,mask) in enumerate(zip(images,viz_rendering_list,viz_mask_list)):
                            row_idx=idx//4
                            col_idx=(idx-(row_idx*4))
                            merged_viz.paste(image.convert('RGB'),(col_idx*512*2,row_idx*512))
                            merged_viz.paste(rend.convert('RGB'),(col_idx*512*2+512,row_idx*512))
                        merged_viz.save(os.path.join(sample_dir, "img_{:06d}_result.jpg".format(global_step)))
                        if not (hostname == 'ubuntu' or hostname.startswith('Qlab')):
                            run.log_image(name=str(os.path.join("img_{:06d}_result.jpg".format(global_step))), path=str(os.path.join(sample_dir, "img_{:06d}_result.jpg".format(global_step))), \
                                plot=None, description=str(os.path.join(sample_dir, "img_{:06d}_result.jpg".format(global_step))))
                        # Inference
                        if (global_step - 1) % (args.save_steps) == 0:
                            filename_unet = (
                                f"{args.output_dir}/unet_weight_s{global_step}.pt"
                            )
                            filename_recnet = (
                                f"{args.output_dir}/recnet_weight_s{global_step}.pt"
                            )
                            filename_optimizer = (
                                f"{args.output_dir}/optimizer_weight_s{global_step}.pt"
                            )
                            if global_step>-1 and (not args.debug):
                                print(f"save weights {filename_unet}")
                                torch.save(unet.state_dict(), filename_unet)
                                print(f"save weights {filename_recnet}")
                                torch.save(recnet.state_dict(), filename_recnet)
                                print(f"save weights {filename_optimizer}")
                                torch.save(optimizer.state_dict(), filename_optimizer)

                if global_step % display_steps == 0:
                    logs = {
                            "loss_unet": loss_unet.detach().item(),
                            "lr": lr_scheduler.get_last_lr()[0],
                            "nonlatin_ratio": moving_avg,
                        }
                    if text_rec_loss is not None:
                        logs['text_rec_loss']=text_rec_loss.detach().item()*args.text_rec_loss_weight
                    if text_rec_loss_recnet is not None:
                        logs['text_rec_loss_recnet']=text_rec_loss_recnet.detach().item()*args.text_rec_loss_weight_recnet
                    

                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                    if global_step >= args.max_train_steps:
                        break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    exit()
