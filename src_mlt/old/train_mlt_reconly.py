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

from datasets.script_info_latin import scriptInfo
from datasets.ocr_dataset_ml import OCRDatasetML, mask_transforms
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

    laion_images = laion_data['image_tensor'].cuda()
    bsz=len(laion_images)
    # print(type(laion_data),'laion_data',type(laion_loader),'type(laion_loader)')
    laion_ids = laion_data['text_tokens'].squeeze(1).cuda()
    # rendering
    text_boxes=[]
    char_gts=[]
    text_idxs=[False]*bsz
    synth_idxs=[False]*bsz
    laion_batch = {
            "text_boxes": text_boxes,
            "char_gts": char_gts,
            "text_idxs": text_idxs,
            "synth_idxs": synth_idxs,
            }
    return laion_batch

    

import json
def main(args):
    moving_avg=0
    args.target_languages=args.target_languages.split('-')
    with open(args.charset_path) as f:
        letters=json.load(f)
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



    """VAE Initialization"""
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        # "stabilityai/stable-diffusion-2-1",
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )
    vae.requires_grad_(False)
    instance_data_list=args.instance_data_list.split('-')
    script_info=scriptInfo(
        charset_path=args.charset_path,
        target_languages=args.target_languages
        )
    print(script_info.lang_set,'script_info.lang_set')
    print(script_info.script_set,'script_info.script_set')
    print(script_info.script2idx,'script_info.script2idx')
    print(script_info.idx2script,'script_info.idx2script')
    from textrecnet_ml import TextRecNetML
    recnet = TextRecNetML(in_channels=4, out_dim=len(script_info.char2idx), max_char=args.max_char_seq_length)    
    
    if args.use_xformers:
        set_use_memory_efficient_attention_xformers(vae, True)
        # set_use_memory_efficient_attention_xformers(fusion_module, True)


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
    params_to_optimize = [
        {"params": recnet.parameters(), "lr":  args.learning_rate_text},
        # {"params": clip_model.parameters() , "lr": args.learning_rate/1.0},
        # {"params": unet_new_params , "lr": args.learning_rate/10.0},
        # {"params": fusion_module_params, "lr": args.learning_rate},
    ]
    

    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )
    
    
    train_dataset = OCRDatasetML(
        instance_data_list=instance_data_list,
        instance_data_root=args.instance_data_root,
        tokenizer=None,
        mask_all_ratio=args.mask_all_ratio,
        max_word_len=25,#output_gt will be in (max_word_len+1) dim
        charset_path=args.charset_path,#output_gt will be in (max_word_len+1) dim
        script_info=script_info,#output_gt will be in (max_word_len+1) dim
        synth_ratio=args.synth_ratio,#output_gt will be in (max_word_len+1) dim
    )
    
    if not (hostname == 'ubuntu' or hostname.startswith('Qlab')):
        mario_db=['mario_laion']
    else:
        mario_db=['mario_laion500K_ml']
    if args.mario_batch_size:
        train_dataset_mario = OCRDatasetML(
            instance_data_list=mario_db,
            instance_data_root=args.instance_data_root,
            tokenizer=None,
            mask_all_ratio=args.mask_all_ratio,
            max_word_len=25,#output_gt will be in (max_word_len+1) dim
            random_crop=args.random_crop,#output_gt will be in (max_word_len+1) dim
            charset_path=args.charset_path,#output_gt will be in (max_word_len+1) dim
            script_info=script_info,
            synth_ratio=args.synth_ratio,#output_gt will be in (max_word_len+1) dim
        )

    def collate_fn(examples):
        pixel_values = [example["instance_images"] for example in examples]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        nonlatin_idxs=[]
        for example in examples:
            nonlatin_idxs +=example["instance_nonlatin_idxs"] 
        synth_idxs=[]
        for example in examples:
            synth_idxs +=example["instance_synth_idx"] 
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
                
        # load laion data
        batch = {
            "pixel_values": pixel_values,
            "text_boxes": text_boxes,
            "char_gts": char_gts,
            "text_idxs": text_idxs,
            "synth_idxs": synth_idxs,
            "nonlatin_idxs": nonlatin_idxs,
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
        train_dataloader,
        recnet,
    ) = accelerator.prepare(
                            train_dataloader,
                            recnet
                            )
    
    if args.resume_recnet_ckpt_path and args.resume_recnet_ckpt_path!='None':
        saved_state_dict = torch.load(args.resume_recnet_ckpt_path, map_location=torch.device('cpu'))
        defined_state_dict=recnet.state_dict()
        defined_keys=list(defined_state_dict.keys())
        new_state_dict={}
        exist_count=0
        for saved_key in saved_state_dict:
            print(saved_key,'recnet_saved_key')
            if saved_key in defined_keys:
                exist_count+=1
            if 'final' in saved_key:
                continue
            new_state_dict[saved_key]=saved_state_dict[saved_key]
        assert (exist_count/len(saved_state_dict))>0.9
        print((exist_count/len(saved_state_dict)),'exist_ratio')
        recnet.load_state_dict(new_state_dict,strict=False)
        print('recnet parameters loaded')
        del saved_state_dict

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
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
    
    if not (hostname == 'ubuntu' or hostname.startswith('Qlab')):
        display_steps = 100
    else:
        display_steps = 1
    print('Start Training')
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            input_images=batch["pixel_values"].to(accelerator.device,dtype=weight_dtype)
            text_idxs = batch["text_idxs"]
            if args.laion_batch_size:
                # Laion data
                laion_batch=load_laion_batch(laion_loader,tokenizer)
                input_images=torch.cat((input_images,laion_batch["pixel_values"].to(accelerator.device,dtype=weight_dtype)),0)
                text_idxs=text_idxs+laion_batch["text_idxs"]
            if args.mario_batch_size:
                # Mario data
                mario_batch=load_mario_batch(mario_loader,tokenizer)
                input_images=torch.cat((input_images,mario_batch["pixel_values"].to(accelerator.device,dtype=weight_dtype)),0)
                text_idxs=text_idxs+mario_batch["text_idxs"]
            with torch.no_grad():
                # vae
                latents = vae.encode(input_images.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215
            bsz = latents.shape[0]
            import time
            import numpy as np
            text_boxes=batch["text_boxes"]
            nonlatin_idxs=batch["nonlatin_idxs"]
            if args.mario_batch_size:
                text_boxes=text_boxes+mario_batch["text_boxes"]
                nonlatin_idxs+=nonlatin_idxs+mario_batch["nonlatin_idxs"]
            if moving_avg==0:
                moving_avg=np.sum(nonlatin_idxs)/len(nonlatin_idxs)
            else:
                moving_avg=moving_avg*0.9+0.1*np.sum(nonlatin_idxs)/len(nonlatin_idxs)
            if len(text_boxes):
                for idx in range(len(text_boxes)):
                    text_boxes[idx]=text_boxes[idx].cuda()
                assert np.sum(text_idxs)!=0
                assert len(latents[text_idxs])==len(text_boxes)
                text_gts=batch["char_gts"].long().to(accelerator.device)
                if args.mario_batch_size:
                    text_gts=torch.cat((text_gts, mario_batch["char_gts"].long().to(accelerator.device)),0)

                ce_criterion = torch.nn.CrossEntropyLoss()
                roi_features = roi_align((latents[text_idxs]).to(device=accelerator.device),text_boxes,output_size=(32,128),spatial_scale=64/512).to(device=accelerator.device,dtype=weight_dtype)
                text_preds_x0 = recnet(roi_features)
                text_rec_loss = ce_criterion(text_preds_x0.view(-1,text_preds_x0.shape[-1]), text_gts.contiguous().view(-1))#*0.05
            accelerator.backward(text_rec_loss)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            if global_step % display_steps == 0:
                progress_bar.update(display_steps)
                if (not (hostname == 'ubuntu' or hostname.startswith('Qlab'))) and accelerator.is_main_process and global_step % display_steps == 0:
                    if text_rec_loss is not None:
                        run.log('text_rec_loss', text_rec_loss.detach().item()*args.text_rec_loss_weight)
                    run.log('lr', lr_scheduler.get_last_lr()[0])
                    run.log('nonlatin_ratio', moving_avg)

            # Visualization
            if (global_step%args.visualize_steps)==0 and accelerator.is_main_process:    
                # # # # # # # # # # # # # # # # 
                # 5. Print Recognition results
                # # # # # # # # # # # # # # # #
                start_idx=0
                end_idx=len(text_boxes[0])
                """Get word indexes corresponding to the instance saved for X0 visualiation"""
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
                if (global_step - 1) % args.inference_steps == 0:    
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
                        # Inference
                        if (global_step - 1) % (args.save_steps) == 0:
                            filename_recnet = (
                                f"{args.output_dir}/recnet_weight_s{global_step}.pt"
                            )
                            if global_step>-1 and (not args.debug):
                                print(f"save weights {filename_recnet}")
                                torch.save(recnet.state_dict(), filename_recnet)

                if global_step % display_steps == 0:
                    logs = {
                            "lr": lr_scheduler.get_last_lr()[0],
                            "nonlatin_ratio": moving_avg,
                        }
                    if text_rec_loss is not None:
                        logs['text_rec_loss']=text_rec_loss.detach().item()*args.text_rec_loss_weight

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
