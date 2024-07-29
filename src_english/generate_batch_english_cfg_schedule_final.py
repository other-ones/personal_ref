import copy
from torch.utils.data.distributed import DistributedSampler as DistributedSampler
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import re

# import cv2
from collections import OrderedDict
# Bootstrapped from:
# from utils import random_crop_image_mask,create_random_mask,create_mask_from_coords
# from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPProcessor
import numpy as np
import sys
sys.path.insert(0, './packages')
import argparse
import hashlib
import itertools
import math
import os
import inspect
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import pdb
from accelerate import Accelerator
# from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipelineScheduledCFG2,
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
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision import transforms
from pathlib import Path
import random
from PIL import Image
from lora_diffusion import tune_lora_scale, patch_pipe
import torchvision.transforms as T
import inspect

# from azureml.core import Run
# run = Run.get_context()
import socket
hostname = socket.gethostname()
hostname=hostname.lower()
print(hostname,'hostname')
from datasets.test_dataset import TestDataset
from config import parse_args
from torch import nn

def to_img(x, clip=True):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, x.size(3))
    if clip:
        x = torch.clip(x, 0, 1)
    return x


def _randomset(lis):
    ret = []
    for i in range(len(lis)):
        if random.random() < 0.5:
            ret.append(lis[i])
    return ret


def _shuffle(lis):
    return random.sample(lis, len(lis))

image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


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


def unfreeze_params(params):
    for param in params:
        param.requires_grad = True


def main(args):
    model_name = 'stabilityai/stable-diffusion-2-1'
    





    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    if (
            args.train_text_encoder
            and args.gradient_accumulation_steps > 1
            and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)
    if (hostname == 'ubuntu' or hostname.startswith('qlab')):
        # Output directory
        trained_model_name=args.resume_unet_ckpt_path.split('/')[-2]
        trained_model_name=trained_model_name.split('Text-')[0]
        ckpt_name=args.resume_unet_ckpt_path.split('/')[-1].split('.')[0]
        sample_dir=os.path.join(args.output_dir,'samples')
        rendering_dir=os.path.join(args.output_dir,'renderings')
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(rendering_dir, exist_ok=True)
        codepath=os.path.join(args.output_dir,'src')
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
        
    else:
        sample_dir=os.path.join(args.output_dir,'samples')
        rendering_dir=os.path.join(args.output_dir,'renderings')
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(rendering_dir, exist_ok=True)

    meta_path=os.path.join(args.output_dir,'meta.txt')
    meta_file=open(meta_path,'a+')
    
    # """original text_encoder"""
    
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_name,
        subfolder="text_encoder",
        revision=args.revision,
    )
    
    # collate_fn
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        masks = [example["instance_mask"] for example in examples]
        layout_files = [example["layout_files"] for example in examples]
        masks = torch.stack(masks)
        spatial_renderings_rgb = [example["instance_spatial_rendering_rgb"] for example in examples]
        spatial_renderings_rgb = torch.stack(spatial_renderings_rgb)

        raw_captions = [example["instance_raw_captions"] for example in examples]
        
        # text_idxs=[]
        # text_boxes = []
        # for example in examples:
        #     text_idxs.append(len(example["instance_text_boxes"])>0)
        #     if len(example["instance_text_boxes"]):
        #         text_boxes.append(example["instance_text_boxes"])
                
        batch = {
            "input_ids": input_ids,
            "raw_captions": raw_captions,
            "masks": masks,
            "layout_files": layout_files,
            "spatial_renderings_rgb": spatial_renderings_rgb,
            # "text_boxes": text_boxes,
        }
        return batch
    # collate_fn

    # ocrdataset
    """
    self,
    # char_tokenizer,
    dbname,
    instance_data_root,
    tokenizer=None,
    include_suffix=True,
    diversify_font=False,
    roi_weight=1,
    charset_path=None,
    script_info=None,
    """
    
    test_dataset = TestDataset(
        dbname=args.dbname,
        instance_data_root=args.instance_data_root,
        tokenizer=tokenizer,
        include_suffix= args.include_suffix, #not (args.no_suffix),
        # charset_path=args.charset_path,
        diversify_font=False,
        target_subset=args.target_subset,
        uniform_layout=args.uniform_layout
    )

    local_rank = accelerator.process_index
    len_dataset = len(test_dataset)
    index_list = [i for i in range(len_dataset) if (i%accelerator.state.num_processes)==local_rank]
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=SubsetRandomSampler(index_list)
    )    
    # print(len(index_list),'len(index_list)',local_rank,'local_rank')
    # print(args.eval_batch_size,'args.eval_batch_size')
    
    """VAE Initialization"""
    vae = AutoencoderKL.from_pretrained(
        model_name,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )
    """UNet Initialization"""
    print(inspect.getsourcefile(UNet2DConditionModel.from_pretrained), 'inspect')
    unet, load_info = UNet2DConditionModel.from_pretrained(
        # args.pretrained_model_name_or_path,
        model_name,
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

    
    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate
                * args.gradient_accumulation_steps
                * args.train_batch_size
                * accelerator.num_processes
        )
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)


    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

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

    # Load UNet!!
    saved_state_dict = torch.load(args.resume_unet_ckpt_path, map_location=torch.device('cpu'))
    # saved_state_dict = torch.load(args.resume_unet_ckpt_path)
    defined_state_dict=unet.state_dict()
    if not isinstance(saved_state_dict,OrderedDict):
        saved_state_dict=saved_state_dict()
    new_state_dict={}
    for saved_key in saved_state_dict:
        new_key=saved_key
        if saved_key.startswith('module.'):
            new_key=saved_key[7:]
        assert new_key in defined_state_dict
        new_state_dict[new_key]=saved_state_dict[saved_key]
    unet.load_state_dict(new_state_dict,strict=True)
    if accelerator.is_main_process:
        print('unet param loaded')
    print(len(test_dataloader),'len(test_dataloader)2') # 4
    # exit()
    (unet
    #  test_dataloader
     ) = accelerator.prepare(
                    unet
                    # test_dataloader
                    )
    print(len(test_dataloader),'len(test_dataloader)3') # 4->1
    # exit()




    std_pipeline = StableDiffusionPipelineScheduledCFG2.from_pretrained( model_name,
                            unet=accelerator.unwrap_model(unet, **extra_args),
                            tokenizer=accelerator.unwrap_model(tokenizer, **extra_args))
    
    std_scheduler = std_pipeline.scheduler
    std_fe_extractor = std_pipeline.feature_extractor
    del std_pipeline
    unet.eval()
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
    
    pipeline = StableDiffusionPipelineScheduledCFG2(
        vae=accelerator.unwrap_model(vae, **extra_args),
        unet=accelerator.unwrap_model(unet, **extra_args),
        text_encoder=accelerator.unwrap_model(text_encoder, **extra_args),
        tokenizer=accelerator.unwrap_model(tokenizer, **extra_args),
        scheduler=accelerator.unwrap_model(std_scheduler, **extra_args),
        feature_extractor=accelerator.unwrap_model(std_fe_extractor, **extra_args),
        safety_checker=None,
        requires_safety_checker=False,
    )


    # New Implementation
    print(f"Num total batch = {len(test_dataloader)}")
    unet.eval()
    guidance_scale=6
    moving_avg=0
    import time
    st=time.time()
    count=0
    # distributed_state = PartialState()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            masks = batch["masks"].to(accelerator.device,dtype=weight_dtype)
            spatial_renderings_rgb=batch["spatial_renderings_rgb"].to(accelerator.device,dtype=weight_dtype)
            # input_ids = batch["input_ids"].to(accelerator.device)
            prompt=batch["raw_captions"]
            layout_files=batch["layout_files"]
            # layout_files=np.array(layout_files)
            masks_tensor = torch.nn.functional.interpolate(masks, size=(64, 64))
            # print(spatial_renderings_rgb.shape,'spatial_renderings_rgb.shape')
            # print(masks_tensor.shape,'masks_tensor.shape')


                       


            # g_schedule1: ref  - decreasing
            # g_schedule2: text - increasing
            # g_schedule3: both - constant

            g_schedule1=(np.linspace(1,0,100)**(args.decay_rate))*(args.base_scale) # ref - decreasing base_scale->0 : w_r in paper
            g_schedule2=(args.base_scale)-g_schedule1 # prompt - increasing 0->base_scale : w_c in paper
            if args.switch_schedule:
                g_schedule1_copy=copy.deepcopy(g_schedule1)
                g_schedule2_copy=copy.deepcopy(g_schedule2)
                g_schedule2=g_schedule1_copy
                g_schedule1=g_schedule2_copy

            g_schedule3=np.ones_like(g_schedule2)*args.cfg_const
            image_list = pipeline(prompt=prompt,
                                    g_schedule1=g_schedule1, # ref / decreasing
                                    g_schedule2=g_schedule2, # prompt / increasing
                                    g_schedule3=g_schedule3, # all / constant
                                    spatial_render=spatial_renderings_rgb, 
                                    masks_tensor=masks_tensor,
                                    num_inference_steps=100,
                                    width=512, height=512, 
                                    num_images_per_prompt=1,
                                    verbose=True,
                                    ).images
            for img_idx,img in enumerate(image_list):
                fname=layout_files[img_idx].split('.')[0]
                # sample
                img.save(os.path.join(sample_dir,'{}.png'.format(fname)))
                meta_file.write('{}\t{}\n'.format(fname,prompt[img_idx]))
                meta_file.flush()
                rendered_whole_np = spatial_renderings_rgb.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                rendered_whole_np=(rendered_whole_np*0.5)+0.5
                rendered_whole_pil=Image.fromarray((rendered_whole_np[img_idx]*255).astype(np.uint8)).convert('RGB')
                rendered_whole_pil.save(os.path.join(rendering_dir,'{}_rendering.png'.format(fname)))
                # if not (hostname == 'ubuntu' or hostname.startswith('qlab')):
                #     run.log_image(name=str(os.path.join(sample_dir,"{}.png".format(fname))), 
                #                 path=str(os.path.join(sample_dir, "{}.png".format(fname))), \
                #                 description=str(os.path.join(sample_dir, "{}.png".format(fname))),
                #                 plot=None)
                #     run.log_image(name=str(os.path.join(rendering_dir,"{}_rendering.png".format(fname))), 
                #                 path=str(os.path.join(rendering_dir, "{}_rendering.png".format(fname))), \
                #                 description=str(os.path.join(rendering_dir, "{}_rendering.png".format(fname))),
                #                 plot=None)
            if accelerator.is_main_process:
                delay=time.time()-st
                if moving_avg==0:
                    moving_avg=delay
                else:
                    moving_avg=moving_avg*0.9+delay*0.1
                num_remaining=len(test_dataloader)-step-1
                print(accelerator.num_processes ,'num_processes')
                print('saved at', sample_dir)
                print('STEP: {}/{} ETA:{:.4f}'.format(step+1,len(test_dataloader),(moving_avg*num_remaining)/60))
                st=time.time()
                count+=1
                st=time.time()
    if accelerator.is_main_process:   
        print(count,'ended')
    # meta_file.close()
    # os.system('cp {} {}'.format())
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    exit()
