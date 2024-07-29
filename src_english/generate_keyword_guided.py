import cv2
import torchvision.ops.roi_align as roi_align
import json
from torch.utils.data.distributed import DistributedSampler as DistributedSampler
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import re
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

# from azureml.core import Run
# run = Run.get_context()
import socket
hostname = socket.gethostname()

from datasets.test_dataset_keyword import TestDataset
from config import parse_args
from torch import nn
from pipeline_keyword import StableDiffusionPipelineKeygenRD


image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = get_logger(__name__)






def main(args):
    with open('ckpt/chartokenizer/char_vocab_english.json') as f:
        letters=json.load(f)
    p2idx = {p: idx for idx, p in enumerate(letters)}
    idx2p = {idx: p for idx, p in enumerate(letters)}
    model_name = args.pretrained_model_name_or_path#'stabilityai/stable-diffusion-2-1'
    # if (hostname == 'ubuntu' or hostname.startswith('Qlab')):
    #     verbose=True
    # else:
    #     verbose=False
    verbose=True





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
        sample_dir = os.path.join(args.output_dir,'samples')
        src_dir = os.path.join(args.output_dir,'src')
        attn_dir = os.path.join(args.output_dir,'attns')
        pmask_dir = os.path.join(args.output_dir,'pmasks')
        nmask_dir = os.path.join(args.output_dir,'nmasks')
        rendering_dir = os.path.join(args.output_dir,'renderings')
        # if os.path.exists(sample_dir):
        #     os.system('rm {} -R'.format(sample_dir))
        if os.path.exists(attn_dir) and 'tmp' in attn_dir:
            os.system('rm {} -R'.format(attn_dir))
        if 'tmp' in sample_dir and os.path.exists(sample_dir):
            os.system('rm {} -R'.format(sample_dir))
        if 'tmp' in rendering_dir and os.path.exists(rendering_dir):
            os.system('rm {} -R'.format(rendering_dir))
        if 'tmp' in rendering_dir and os.path.exists(attn_dir):
            os.system('rm {} -R'.format(attn_dir))
        if 'tmp' in rendering_dir and os.path.exists(pmask_dir):
            os.system('rm {} -R'.format(pmask_dir))
        if 'tmp' in rendering_dir and os.path.exists(nmask_dir):
            os.system('rm {} -R'.format(nmask_dir))
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(rendering_dir, exist_ok=True)
        os.makedirs(attn_dir, exist_ok=True)
        os.makedirs(pmask_dir, exist_ok=True)
        os.makedirs(nmask_dir, exist_ok=True)
        os.makedirs(src_dir, exist_ok=True)
        codepath=os.path.join(args.output_dir,'src')
        os.makedirs(codepath,exist_ok=True)
        os.system('cp *.py {}'.format(codepath))
        os.system('cp packages {} -R'.format(codepath))
        os.system('cp datasets {} -R'.format(codepath))
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
        attn_dir=os.path.join(args.output_dir,'attns')
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
    ).to(device)
    
    # collate_fn
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        eot_idxs_batch = [example["instance_eot_idxs"] for example in examples]


        is_keyword_tokens = [example["instance_is_keyword_tokens"] for example in examples] #[n,77]
        pos_masks = [example["instance_pos_masks"] for example in examples]
        neg_masks = [example["instance_neg_masks"] for example in examples]
        pos_masks=torch.stack(pos_masks)#[n,77,512,512]
        neg_masks=torch.stack(neg_masks)#[n,77,512,512]


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
        


        char_gts=[]
        for eidx,example in enumerate(examples):
            char_gts+=example["instance_char_gts"]#returns: T,26
        if len(char_gts):
            char_gts = torch.stack(char_gts) #shape N,T,26
        text_boxes = []
        nonzero_idxs=[]
        for example in examples:
            nonzero_idxs.append(len(example["instance_text_boxes"])>0)
            if len(example["instance_text_boxes"]):
                text_boxes.append(example["instance_text_boxes"])
                
        batch = {
            "input_ids": input_ids,
            "raw_captions": raw_captions,
            "masks": masks,
            "layout_files": layout_files,
            "spatial_renderings_rgb": spatial_renderings_rgb,
            "nonzero_idxs": nonzero_idxs,
            "text_boxes": text_boxes,
            "char_gts": char_gts,
            "is_keyword_tokens": is_keyword_tokens,
            "pos_masks": pos_masks,
            "neg_masks": neg_masks,
            "eot_idxs_batch": eot_idxs_batch,
        }
        return batch
    # collate_fn
    test_dataset = TestDataset(
        dbname=args.dbname,
        instance_data_root=args.instance_data_root,
        tokenizer=tokenizer,
        include_suffix= args.include_suffix, #not (args.no_suffix),
        # charset_path=args.charset_path,
        diversify_font=False,
        max_word_len=25,#output_gt will be in (max_word_len+1) dim
        debug=args.debug,
        sample_dir=sample_dir,
        mask_bg=args.mask_bg,
        mask_fg=args.mask_fg,
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

    
    unet_lora_params, unet_lora_params_name = inject_trainable_lora_extended(unet, r=args.lora_rank, verbose=False)
    print('New added parameters:')
    print(unet_new_params_name)
    print('New lora parameters:')
    print(unet_lora_params_name)
    vae.requires_grad_(False)
    
    
    
    
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

    # # Load UNet!!
    if args.resume_unet_ckpt_path and args.resume_unet_ckpt_path !='None':
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
        del saved_state_dict, new_state_dict
    else:
        assert False
   


    


    if accelerator.is_main_process:
        print('unet param loaded')
    print(len(test_dataloader),'len(test_dataloader) HERE2') # 4
    # exit()
    (
        unet
    #  test_dataloader
     ) = accelerator.prepare(
                    unet
                    # test_dataloader
                    )
    print(len(test_dataloader),'len(test_dataloader) HERE3') # 4->1


    keygen_pipeline = StableDiffusionPipelineKeygenRD.from_pretrained( model_name,
                            unet=accelerator.unwrap_model(unet, **extra_args),
                            tokenizer=accelerator.unwrap_model(tokenizer, **extra_args),
                            text_encoder=accelerator.unwrap_model(text_encoder, **extra_args),
                            vae=accelerator.unwrap_model(vae, **extra_args),
                            )
    scheduler = keygen_pipeline.scheduler
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
    
    

    # New Implementation'
    # unet.requires_grad_(True)
    # unet.train()

    moving_avg_delay=0
    import time
    count=0
    unet=unet.to(accelerator.device)
    # unet.eval()
    # vae.eval()
    if 'stabilityai' in args.pretrained_model_name_or_path:
        extra_step_kwargs={'eta':0.0,'generator':None}
    else:
        extra_step_kwargs={}
    print(f"Num total batch = {len(test_dataloader)}")
    count=0
    st=time.time()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            prompt_batch=batch["raw_captions"]
            layout_files=batch["layout_files"]
            text_boxes=batch["text_boxes"]
            spatial_renderings_rgb=batch["spatial_renderings_rgb"].to(accelerator.device,dtype=weight_dtype)
            batch_size=len(prompt_batch)
            masks = batch["masks"].to(accelerator.device,dtype=weight_dtype)
            pos_masks_batch = batch["pos_masks"].to(accelerator.device,dtype=weight_dtype)
            neg_masks_batch = batch["neg_masks"].to(accelerator.device,dtype=weight_dtype)
            # token_maps_batch=F.interpolate(token_maps_batch,size=(16, 16))
            is_keyword_tokens = batch["is_keyword_tokens"]
            eot_idxs_batch = batch["eot_idxs_batch"]


            # 3. Construct Initial noise
            # print('vae encodes')
            
            render_embs = vae.encode(spatial_renderings_rgb).latent_dist.sample().to(dtype=weight_dtype, device=device)
            # print('vae encodes done')
            # exit()

            render_embs = render_embs * 0.18215
            mask_tensors = torch.nn.functional.interpolate(masks, size=(64, 64))


            attn_res=256
            attn_mod_params={
                "is_keyword_tokens":is_keyword_tokens,
                "pos_masks_batch":pos_masks_batch,
                "neg_masks_batch":neg_masks_batch,
                "do_classifier_free_guidance":args.guidance_scale>1,
                "treg_pos":args.treg_pos,
                "treg_neg":args.treg_neg,
            }
            print(render_embs.shape,'render_embs.shape')
            print(mask_tensors.shape,'mask_tensors.shape')
            print(len(prompt_batch),'len(prompt_batch)')
            num_copy=1
            image_list,attention_maps_list=keygen_pipeline(prompt_batch*num_copy,
                            height=512,width=512,
                            num_inference_steps=args.num_inference_steps,
                            attn_res=(int(math.sqrt(attn_res)), int(math.sqrt(attn_res))),
                            render_embs=render_embs.repeat(num_copy,1,1,1),
                            mask_tensors=mask_tensors.repeat(num_copy,1,1,1),
                            attn_mod_params=attn_mod_params,
                            guidance_scale=args.guidance_scale,
                            verbose=verbose
                            )
            image_list=image_list.images
            # attention_maps: bsz,77,16,16
            assert len(image_list)==len(attention_maps_list)
            
            
            
            

            # Decode results
            # is_keyword_tokens: bsz,77
            rendered_whole_np = spatial_renderings_rgb.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            pos_masks_batch = pos_masks_batch.detach().cpu().float().numpy()
            neg_masks_batch = neg_masks_batch.detach().cpu().float().numpy()
            rendered_whole_np=(rendered_whole_np*0.5)+0.5
            for img_idx,img in enumerate(image_list):
                randnum='{:06d}'.format(np.random.randint(0,1e6))
                # attention map visualization
                pos_masks=pos_masks_batch[img_idx] #(3,77,16,16)->(77,16,16) (list of list)
                neg_masks=neg_masks_batch[img_idx] #(3,77,16,16)->(77,16,16) (list of list)
                attention_maps=attention_maps_list[img_idx] #(3,77,16,16)->(77,16,16) (list of list)
                is_keywords=is_keyword_tokens[img_idx]
                fname=layout_files[img_idx].split('.')[0]
                eot_idx=eot_idxs_batch[img_idx]
                prompt=prompt_batch[img_idx]
                ids = keygen_pipeline.tokenizer(prompt).input_ids
                ids=tokenizer.pad(
                    {"input_ids": ids},
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    # return_tensors="pt",
                ).input_ids
                indices = {i: tok 
                    for tok, i in zip(keygen_pipeline.tokenizer.convert_ids_to_tokens(ids), range(len(ids)))
                }

                if (hostname == 'ubuntu' or hostname.startswith('Qlab')):
                    for kidx in range(len(is_keywords)):
                        if kidx>(eot_idx+1):
                            break
                        ktok=indices[kidx].replace('</w>','')
                        ktok=ktok.replace('<|startoftext|>','[SOS]')
                        ktok=ktok.replace('<|endoftext|>','[EOS]')

                        attn_map=attention_maps[kidx].detach().cpu().numpy()#*args.vis_strength
                        pmask=pos_masks[kidx]
                        nmask=neg_masks[kidx]
                        attn_map=(255 * (attn_map / np.max(attn_map)))

                        pmask=pmask*255
                        nmask=nmask*255
                        attn_map=cv2.resize(attn_map,(512,512))
                        dst_attn_dir=os.path.join(attn_dir,fname+randnum)
                        dst_pmask_dir=os.path.join(pmask_dir,fname+randnum)
                        dst_nmask_dir=os.path.join(nmask_dir,fname+randnum)
                        os.makedirs(dst_attn_dir,exist_ok=True)
                        os.makedirs(dst_pmask_dir,exist_ok=True)
                        os.makedirs(dst_nmask_dir,exist_ok=True)
                        

                        if is_keywords[kidx]:
                            cv2.imwrite(os.path.join(dst_attn_dir,'{}_{:02d}_key_{}.png'.format(fname+randnum,kidx,ktok)),attn_map)
                            cv2.imwrite(os.path.join(dst_pmask_dir,'{}_{:02d}_key_{}_pmask.png'.format(fname+randnum,kidx,ktok)),pmask)
                            cv2.imwrite(os.path.join(dst_nmask_dir,'{}_{:02d}_key_{}_nmask.png'.format(fname+randnum,kidx,ktok)),nmask)
                        else:
                            cv2.imwrite(os.path.join(dst_attn_dir,'{}_{:02d}_nonkey_{}.png'.format(fname+randnum,kidx,ktok)),attn_map)
                            cv2.imwrite(os.path.join(dst_pmask_dir,'{}_{:02d}_nonkey_{}_pmask.png'.format(fname+randnum,kidx,ktok)),pmask)
                            cv2.imwrite(os.path.join(dst_nmask_dir,'{}_{:02d}_nonkey_{}_nmask.png'.format(fname+randnum,kidx,ktok)),nmask)


                
                img.save(os.path.join(sample_dir,'{}.png'.format(fname+randnum)))
                meta_file.write('{}\t{}\n'.format(fname+randnum,prompt_batch[img_idx]))
                meta_file.flush()
                rendered_whole_pil=Image.fromarray((rendered_whole_np[img_idx]*255).astype(np.uint8)).convert('RGB')
                rendered_whole_pil.save(os.path.join(rendering_dir,'{}_rendering.png'.format(fname+randnum)))
                # if not (hostname == 'ubuntu' or hostname.startswith('Qlab')):
                #     run.log_image(name=str(os.path.join(sample_dir,"{}.png".format(fname))), 
                #             path=str(os.path.join(sample_dir, "{}.png".format(fname))), \
                #             description=str(os.path.join(sample_dir, "{}.png".format(fname))),
                #             plot=None)
            
            if accelerator.is_main_process:
                delay_seconds=time.time()-st
                st=time.time() # returns time in secends
                if moving_avg_delay==0:
                    moving_avg_delay=delay_seconds
                else:
                    moving_avg_delay=moving_avg_delay*0.9+delay_seconds*0.1
                num_remaining=len(test_dataloader)-(step+1)
                print('saved at', sample_dir)
                print('STEP: {}/{} ETA:{:.4f} (minutes)'.format(step+1,len(test_dataloader),(moving_avg_delay*num_remaining)/60))
                if args.debug:
                    break

    if accelerator.is_main_process:   
        print(count,'ended')
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    exit()
