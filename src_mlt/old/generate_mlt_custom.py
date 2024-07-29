from torch.utils.data.distributed import DistributedSampler as DistributedSampler
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import re

# import cv2
from collections import OrderedDict
# Bootstrapped from:
# from utils import random_crop_image_mask,create_random_mask,create_mask_from_coords
from utils import generate_mask_ml,generate_spatial_rendering_ml
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
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision import transforms
from pathlib import Path
import random
from PIL import Image
from lora_diffusion import tune_lora_scale, patch_pipe
import torchvision.transforms as T
import inspect

from azureml.core import Run
run = Run.get_context()
import socket
hostname = socket.gethostname()

from datasets.test_dataset import TestDataset
from config import parse_args
from torch import nn


mask_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),

    ]
)

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
    if (hostname == 'ubuntu' or hostname.startswith('Qlab')):
        # Output directory
        trained_model_name=args.resume_unet_ckpt_path.split('/')[-2]
        trained_model_name=trained_model_name.split('Text-')[0]
        ckpt_name=args.resume_unet_ckpt_path.split('/')[-1].split('.')[0]
        stepname=ckpt_name.split('_')[-1]
        # sample_dir = os.path.join(args.output_dir,'{}_{}'.format(trained_model_name,stepname),'samples')
        # rendering_dir = os.path.join(args.output_dir,'{}_{}'.format(trained_model_name,stepname),'renderings')
        sample_dir=os.path.join(args.output_dir,'samples')
        rendering_dir=os.path.join(args.output_dir,'renderings')
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(rendering_dir, exist_ok=True)
    else:
        sample_dir=os.path.join(args.output_dir,'samples')
        rendering_dir=os.path.join(args.output_dir,'renderings')
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(rendering_dir, exist_ok=True)

    meta_path=os.path.join(args.output_dir,'meta.txt')
    meta_file=open(meta_path,'a+',encoding='utf-8')
    
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
        # blank_idxs = [example["instance_blank_idxs"] for example in examples]
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
        raw_words = [example["instance_raw_words"] for example in examples]
                
        batch = {
            "input_ids": input_ids,
            "raw_captions": raw_captions,
            "raw_words": raw_words,
            "masks": masks,
            "layout_files": layout_files,
            "spatial_renderings_rgb": spatial_renderings_rgb,
            # "blank_idxs": blank_idxs,
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
    diversify_font=False,
    roi_weight=1,
    charset_path=None,
    script_info=None,
    """
    

    local_rank = accelerator.process_index
    
    
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
    # exit()
    (unet
    #  test_dataloader
     ) = accelerator.prepare(
                    unet
                    # test_dataloader
                    )
    std_pipeline = StableDiffusionPipeline.from_pretrained( model_name,
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


    # New Implementation
    unet.eval()
    guidance_scale=7
    moving_avg=0
    import time
    st=time.time()
    count=0
    # distributed_state = PartialState()

    # glyph_texts_batch = [["перепись"],["เซนต์บริด"],["ธนานุบาล"],["περλφύτ"],["तेवरमों"],["অইছিল"],["пдвржен"],["dormìpò"],["défià"],["äußert"]]
    # inference_lang_list_batch=[["russian"],["thai"],['thai'],['greek'],["hindi"],["bengali"],["russian"],["italian"],["french"],["german"]]
    # glyph_texts_batch = [["կոշիկներ"],["բրդյա"],["գոտի"],["գլխարկ"]]
    # glyph_texts_batch = [["ໂຮໂຣນາຢິມ"],["ຮຳຣານ"],["ຮ້າໂນ້ຍ"],["ຮາງນໍ້າ"]]
    # glyph_texts_batch = [["перепись"],["περλφύτ"],["пдвржен"],["défià"]]
    # inference_lang_list_batch=[["russian"],["greek"],['russian'],['french']]
    # glyph_texts_batch = [["перепись"],["περλφύτ"],["пдвржен"],["défià"]]
    # inference_lang_list_batch=[["armenian"],["armenian"],['armenian'],['armenian']]
    # glyph_texts_batch = [["كتاب"],["سماء"],["بيت"],["شمس"]]
    # inference_lang_list_batch=[["arabic"],["arabic"],['arabic'],['arabic']]
    # glyph_texts_batch = [["மனி"],["ஐக்கி"],["நாடு"],["யேயா"]]
    # inference_lang_list_batch=[["tamil"],["tamil"],['tamil'],['tamil']]
    glyph_texts_batch = [["ఆంతరం"],["కుటుంబ"],["గృహ"],["చెందిన"]]
    inference_lang_list_batch=[['telugu']]*len(glyph_texts_batch)
    rendering_batch=[]
    mask_batch=[]
    prompt_batch=[
            "a street sign with words",
            "a mug with texts",
            "a word printed on a cap",
            "a plant pot with a tag",
            # "a label on a bottle",
            # "a frames photograph with the caption",




            # # "little squirrel holding a sign saying words",
            # "a pig holding a sign that saying words",
            # # "a hat with the words printed on it",
            # # "a dog holding a paper saying words",
            # # "little bee holding a sign saying words",
            # # "little deer holding a sign saying words'",
            # "a squirrel holding a sign saying words'",
            # "a cat holding a sign saying words'",
            # # "a turtle holding a sign saying words'",
            # "a backpack with the words printed on it",
            # "cosmetic bottle saying wors",
            # # "slogan printed on school bus",
            # "Photo of the restaurant with words",
    ]
    glyph_texts_batch=glyph_texts_batch[:len(prompt_batch)]
    inference_lang_list_batch=inference_lang_list_batch[:len(prompt_batch)]
    # from datasets.script_info import scriptInfo
    # script_info=scriptInfo(
    #     charset_path=args.charset_path,
    #     target_languages=args.target_languages
    #     # instance_data_list_real=instance_data_list_real,
    #     # instance_data_list_synth=instance_data_list_synth,
    #     )

    for gen_idx in range(len(glyph_texts_batch)):
        glyph_texts=glyph_texts_batch[gen_idx]
        inference_lang_list=inference_lang_list_batch[gen_idx]
        # inference_scripts_list=[script_info.lang2script[lang]for lang in inference_lang_list]
        coords=np.array([[100, 145,390, 265]])
        coords=coords.astype(np.int32).tolist()
        random_mask_np=generate_mask_ml(height=512,width=512,coords=coords,mask_type='rectangle')
        random_mask=Image.fromarray(random_mask_np)
        masks_tensor = mask_transforms(random_mask).float().unsqueeze(0).to(accelerator.device,dtype=weight_dtype)
        masks_64 = torch.nn.functional.interpolate(masks_tensor, size=(64, 64), mode='nearest')
        spatial_renderings=generate_spatial_rendering_ml(width=512,height=512,
                                                        words=glyph_texts,
                                                        dst_coords=coords,
                                                        lang_list=inference_lang_list)
        spatial_renderings_rgb=image_transforms(spatial_renderings.convert("RGB"))
        spatial_renderings_rgb=spatial_renderings_rgb.unsqueeze(0).to(accelerator.device,dtype=weight_dtype)
        rendering_batch.append(spatial_renderings_rgb)
        mask_batch.append(masks_64)
    mask_batch=torch.cat(mask_batch)
    rendering_batch=torch.cat(rendering_batch)
    print(rendering_batch.shape,'rendering_batch.shape')
    print(mask_batch.shape,'mask_batch.shape')
    print(len(prompt_batch),'prompt_batch')
    image_list = pipeline(prompt=prompt_batch, 
                        spatial_render=rendering_batch, 
                        masks_tensor=mask_batch,
                        num_inference_steps=100, guidance_scale=guidance_scale, 
                        width=512, height=512, 
                         num_images_per_prompt=1).images
    count=0
    rendering_batch = rendering_batch.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    for img_idx,img in enumerate(image_list):
        randnum=np.random.randint(low=1,high=999999)
        word_splits=prompt_batch[img_idx].split()
        # print(word_splits,'word_splits')
        # print(prompt_batch[img_idx],'prompt_batch')
        fname='{}_{:06}'.format('_'.join(word_splits),randnum)
        img.save(os.path.join(sample_dir,'{}.png'.format(fname)))
        words_cat=','.join(glyph_texts_batch[img_idx])
        meta_file.write('{}\t{}\t{}\n'.format(fname,prompt_batch[img_idx],words_cat))
        meta_file.flush()
        rendered_whole_np=rendering_batch[img_idx]
        rendered_whole_np=(rendered_whole_np*0.5)+0.5
        print(rendered_whole_np.shape,'rendered_whole_np.shape')
        rendered_whole_pil=Image.fromarray((rendered_whole_np*255).astype(np.uint8)).convert('RGB')
        rendered_whole_pil.save(os.path.join(rendering_dir,'{}_rendering.png'.format(fname)))
    if accelerator.is_main_process:
        delay=time.time()-st
        st=time.time()
        if moving_avg==0:
            moving_avg=delay
        else:
            moving_avg=moving_avg*0.9+delay*0.1
        print('saved at', sample_dir)
        count+=1
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
