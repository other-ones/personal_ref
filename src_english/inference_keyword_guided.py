from utils import generate_mask,generate_pos_neg_masks
from utils import generate_spatial_rendering
from utils import visualize_box
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
hostname = socket.gethostname().lower()

from config import parse_args
from torch import nn
from pipeline_keyword import StableDiffusionPipelineKeygenRD

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
logger = get_logger(__name__)






def main(args):
    with open('ckpt/chartokenizer/char_vocab_english.json') as f:
        letters=json.load(f)
    p2idx = {p: idx for idx, p in enumerate(letters)}
    idx2p = {idx: p for idx, p in enumerate(letters)}
    model_name = args.pretrained_model_name_or_path#'stabilityai/stable-diffusion-2-1'
    # if (hostname == 'ubuntu' or hostname.startswith('qlab')):
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
    if (hostname == 'ubuntu' or hostname.startswith('qlab')):
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
        # if os.path.exists(attn_dir) and 'tmp' in attn_dir:
        #     os.system('rm {} -R'.format(attn_dir))
        # if 'tmp' in sample_dir and os.path.exists(sample_dir):
        #     os.system('rm {} -R'.format(sample_dir))
        # if 'tmp' in rendering_dir and os.path.exists(rendering_dir):
        #     os.system('rm {} -R'.format(rendering_dir))
        # if 'tmp' in rendering_dir and os.path.exists(attn_dir):
        #     os.system('rm {} -R'.format(attn_dir))
        # if 'tmp' in rendering_dir and os.path.exists(pmask_dir):
        #     os.system('rm {} -R'.format(pmask_dir))
        # if 'tmp' in rendering_dir and os.path.exists(nmask_dir):
        #     os.system('rm {} -R'.format(nmask_dir))
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
    # exit()
    (
        unet
     ) = accelerator.prepare(
                    unet
                    )


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
    count=0
    st=time.time()
    with torch.no_grad():
        keywords_list=['red','apple']
        # caption="a dog holding a paper saying {}".format(' '.join(keywords_list))
        caption="a dog holding a tree engraved with {}".format(' '.join(keywords_list))
        # caption="words saying {} printed on the package".format(keywords_list[0],keywords_list[1])
        prompt_batch=[caption]
        text_boxes=np.array([[100,200,220,260],
                             [200, 220, 370, 270]])
        coords_list=[
            [
                [100,200],
                [220,200],
                [220,260],
                [100,260]
            ],
            [
                [200,220],
                [370,220],
                [370,270],
                [200,270],
             ]
                     ]

        # 1. rendering
        spatial_rendering_rgb=generate_spatial_rendering(width=512,height=512,words=keywords_list,dst_coords=text_boxes)
        spatial_rendering_rgb=image_transforms(spatial_rendering_rgb.convert("RGB"))
        spatial_rendering_rgb=spatial_rendering_rgb.unsqueeze(0).to(accelerator.device,dtype=weight_dtype)# torch.Size([1, 3, 512, 512])
        # 2. mask
        mask_np=generate_mask(height=512,width=512,coords=coords_list,mask_type='rectangle')
        mask_pil=Image.fromarray(mask_np)
        masks_tensor = mask_transforms(mask_pil).float().unsqueeze(0).to(accelerator.device,dtype=weight_dtype)
        masks_64 = torch.nn.functional.interpolate(masks_tensor, size=(64, 64), mode='nearest') # torch.Size([1, 1, 64, 64])
        masks_64 = masks_64.to(accelerator.device,dtype=weight_dtype)


        #### Parse word index to coordinate index ####
        widx_to_crd_idx={}
        caption_splits=caption.split()
        # First parameter is the replacement, second parameter is your input string
        hits=[False]*len(keywords_list) # index of keyword list, all keywords should be visited 
        # print()
        for widx in range(len(caption_splits)):
            cap_word=caption_splits[widx]
            cap_word_nopunc = re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039])", "", cap_word) 
            # print(cap_word_nopunc,'cap_word_nopunc',hits,'hits')
            if cap_word_nopunc in keywords_list: #keyword
                for crd_idx in range(len(coords_list)):
                    keyword=keywords_list[crd_idx]
                    if (not hits[crd_idx]) and (cap_word_nopunc == keyword):
                        widx_to_crd_idx[widx]=crd_idx
                        hits[crd_idx]=True
                        break
        if np.sum(hits)!=len(hits):
            print('Assertion HERE')
            print(np.array(coords_list)[hits],'hit coords')
            print(np.array(keywords_list)[hits],'hit keywords')
            print(caption,'caption')
            print(hits,'hits')
            print(keywords_list,'keywords_list')
            print(coords_list,'coords')
            print(np.sum(hits),'np.sum(hits)')
            print(len(hits),'len(hits)')
        assert np.sum(hits)==len(hits),'keyword parsing error'
        #### Parse token index to coordinate index ####




        # FINAL GOAL!!!! : tidx_to_crd_idx
        tidx_to_crd_idx={} #token index to keyword coordinate index
        # FINAL GOAL!!!!
        is_keyword_tokens=[False]*tokenizer.model_max_length #first token is special token
        # index for enumerating token idxs of the tokenized prompt
        tidx=1 #starts from 1 since first token is for the special token
        eot_idx=1 # 1 for start token
        for widx,cap_word in enumerate(caption.split()):
            # convert word to token
            word_token_ids=tokenizer.encode(cap_word,add_special_tokens=False)
            eot_idx+=len(word_token_ids)
            # add_special_tokens: if False do not include "<|startoftext|>" and "<|endoftext|>"
            # e.g., neurips -> [3855,30182] which corresponds to ['neu', 'rips</w>'] tokens
            # this can be confirmed by tokenizer.convert_ids_to_tokens(word_token_ids)
            # the token ids here is not used, but we need to count the number of tokens for each word
            word_token_idxs=[]
            # iterate over token ids of a word
            # e.g., neurips -> [3855,30182]
            num_tokens=len(word_token_ids)
            for _ in range(num_tokens):
                word_token_idxs.append(tidx)
                # token_index to word_index mapping
                # e.g., "word saying 'neurips'" -> token_idx for neu<\w>: 2
                # per_token_word_idxs[2]->2: 2 for word index of neurips
                if widx in widx_to_crd_idx:
                    is_keyword_tokens[tidx]=True
                    tidx_to_crd_idx[tidx]=widx_to_crd_idx[widx]
                tidx+=1
                if tidx==(len(is_keyword_tokens)-2):
                    break
            if tidx==(len(is_keyword_tokens)-2):
                    break
            # word_index to token_index_list mapping
            # e.g., "word saying 'Neurips'" -> word_idx for neurips: 2
            # NOTE: len(word_token_idxs)==(len(input_ids)-2): two for special tokens
        assert not is_keyword_tokens[-1],'last element of is_keyword_tokens should be False'
        #### Parse token index to coordinate index ####
        

        # HERE
        pos_masks,neg_masks = generate_pos_neg_masks(
            512, 
            512, 
            coords_list,
            is_keyword_tokens,
            tidx_to_crd_idx,
            fg_mask=mask_np,
            eot_idx=eot_idx,
            mask_type='rectangle'
            )
        # 3. attn mod mask
        pos_masks_batch=torch.Tensor(pos_masks).float().unsqueeze(0).to(accelerator.device,dtype=weight_dtype)
        neg_masks_batch=torch.Tensor(neg_masks).float().unsqueeze(0).to(accelerator.device,dtype=weight_dtype)
        eot_idxs_batch=[eot_idx]
        is_keyword_tokens = [is_keyword_tokens]
        
        
        
        
        
        
        
        
        
        
        



        # 3. Construct Initial noise
        # print('vae encodes')
        render_embs = vae.encode(spatial_rendering_rgb).latent_dist.sample().to(dtype=weight_dtype, device=device)
        render_embs = render_embs * 0.18215
        # mask_tensors = torch.nn.functional.interpolate(masks, size=(64, 64))


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
        print(masks_64.shape,'mask_tensors.shape')
        num_copy=1
        image_list,attention_maps_list=keygen_pipeline(prompt_batch*num_copy,
                        height=512,width=512,
                        num_inference_steps=args.num_inference_steps,
                        attn_res=(int(math.sqrt(attn_res)), int(math.sqrt(attn_res))),
                        render_embs=render_embs.repeat(num_copy,1,1,1),
                        mask_tensors=masks_64.repeat(num_copy,1,1,1),
                        attn_mod_params=attn_mod_params,
                        guidance_scale=args.guidance_scale,
                        verbose=verbose
                        )
        image_list=image_list.images
        # attention_maps: bsz,77,16,16
        assert len(image_list)==len(attention_maps_list)
        
        
        
        

        # Decode results
        # is_keyword_tokens: bsz,77
        rendered_whole_np = spatial_rendering_rgb.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        pos_masks_batch = pos_masks_batch.detach().cpu().float().numpy()
        neg_masks_batch = neg_masks_batch.detach().cpu().float().numpy()
        rendered_whole_np=(rendered_whole_np*0.5)+0.5
        for img_idx,img in enumerate(image_list):
            # randnum='{:06d}'.format(np.random.randint(0,1e6))
            # attention map visualization
            pos_masks=pos_masks_batch[img_idx] #(3,77,16,16)->(77,16,16) (list of list)
            neg_masks=neg_masks_batch[img_idx] #(3,77,16,16)->(77,16,16) (list of list)
            attention_maps=attention_maps_list[img_idx] #(3,77,16,16)->(77,16,16) (list of list)
            is_keywords=is_keyword_tokens[img_idx]
            cap_words=caption.split()
            fname='{}_pos{}_neg{}'.format('_'.join(cap_words),int(args.treg_pos*10),int(args.treg_neg*10))
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

            if (hostname == 'ubuntu' or hostname.startswith('qlab')):
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
                    dst_attn_dir=os.path.join(attn_dir,fname)
                    dst_pmask_dir=os.path.join(pmask_dir,fname)
                    dst_nmask_dir=os.path.join(nmask_dir,fname)
                    os.makedirs(dst_attn_dir,exist_ok=True)
                    os.makedirs(dst_pmask_dir,exist_ok=True)
                    os.makedirs(dst_nmask_dir,exist_ok=True)
                    

                    if is_keywords[kidx]:
                        cv2.imwrite(os.path.join(dst_attn_dir,'{}_{:02d}_key_{}.png'.format(fname,kidx,ktok)),attn_map)
                        cv2.imwrite(os.path.join(dst_pmask_dir,'{}_{:02d}_key_{}_pmask.png'.format(fname,kidx,ktok)),pmask)
                        cv2.imwrite(os.path.join(dst_nmask_dir,'{}_{:02d}_key_{}_nmask.png'.format(fname,kidx,ktok)),nmask)
                    else:
                        cv2.imwrite(os.path.join(dst_attn_dir,'{}_{:02d}_nonkey_{}.png'.format(fname,kidx,ktok)),attn_map)
                        cv2.imwrite(os.path.join(dst_pmask_dir,'{}_{:02d}_nonkey_{}_pmask.png'.format(fname,kidx,ktok)),pmask)
                        cv2.imwrite(os.path.join(dst_nmask_dir,'{}_{:02d}_nonkey_{}_nmask.png'.format(fname,kidx,ktok)),nmask)


            img.save(os.path.join(sample_dir,'{}.png'.format(fname)))
            meta_file.write('{}\t{}\n'.format(fname,prompt_batch[img_idx]))
            meta_file.flush()
            rendered_whole_pil=Image.fromarray((rendered_whole_np[img_idx]*255).astype(np.uint8)).convert('RGB')
            rendered_whole_pil.save(os.path.join(rendering_dir,'{}_rendering.png'.format(fname)))
            # if not (hostname == 'ubuntu' or hostname.startswith('qlab')):
            #     run.log_image(name=str(os.path.join(sample_dir,"{}.png".format(fname))), 
            #             path=str(os.path.join(sample_dir, "{}.png".format(fname))), \
            #             description=str(os.path.join(sample_dir, "{}.png".format(fname))),
            #             plot=None)
        

    if accelerator.is_main_process:   
        print(count,'ended')
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    exit()
