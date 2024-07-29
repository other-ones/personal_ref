# import cv2

# Bootstrapped from:
from utils import visualize_box,numpy_to_pil,generate_mask
# from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPProcessor
import numpy as np
import sys
sys.path.insert(0, './packages')
# import argparse
# import hashlib
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
# from custom_transformer import CausalTransformer
# import clip
# from rendering.official_rendering2 import font_devise_64, font_devise_512, render_text_image
import torchvision.transforms as T
# import open_clip
import inspect
from utils import generate_spatial_rendering

from azureml.core import Run
run = Run.get_context()
import socket
from data_utils import cycle, create_wbd

hostname = socket.gethostname()

# import resnet
from datasets.ocr_dataset import OCRDataset, mask_transforms, glyph_transforms
from config import parse_args
import cv2
from torch import nn
import torchvision.ops.roi_align as roi_align
# from shapely.geometry import Polygon

tmp_count=0
def to_img(x, clip=True):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, x.size(3))
    if clip:
        x = torch.clip(x, 0, 1)
    return x



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
    rendered_whole_images_rgb=[image_transforms(Image.new("RGB", (512, 512), (255, 255, 255)))]*bsz
    rendered_whole_images_rgb = torch.stack(rendered_whole_images_rgb)
    # rendered_whole_images_rgb=image_transforms(rendered_whole_images_rgb)
    text_boxes=[]
    char_gts=[]
    nonzero_idxs=[False]*bsz
    laion_batch = {
            "pixel_values": laion_images,
            "input_ids": laion_ids,
            "masks": masks,
            "rendered_whole_images_rgb": rendered_whole_images_rgb,
            "text_boxes": text_boxes,
            "char_gts": char_gts,
            "nonzero_idxs": nonzero_idxs}
    return laion_batch
def load_mario_batch(mario_loader,tokenizer):
    mario_data=next(mario_loader)
    return mario_data

    
class DiceCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCELoss, self).__init__()
        # self.num_classes = num_classes
        
    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        bce_loss = torch.nn.CrossEntropyLoss()(inputs, targets)
        inputs = torch.sigmoid(inputs)       
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        targets = torch.nn.functional.one_hot(targets.long(), num_classes=inputs.shape[1]).permute(0,3,1,2).float()
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        Dice_BCE = 1.0 * bce_loss + 0.25 * dice_loss
        # Dice_BCE = dice_loss
        
        return Dice_BCE
import json
def main(args):
    with open('ckpt/chartokenizer/char_vocab_english.json') as f:
        letters=json.load(f)
    p2idx = {p: idx for idx, p in enumerate(letters)}
    idx2p = {idx: p for idx, p in enumerate(letters)}
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
        if (hostname == 'ubuntu' or hostname.startswith('Qlab') or hostname.startswith('qlab')):
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
    tokenizer = CLIPTokenizer.from_pretrained(
        # "stabilityai/stable-diffusion-2-1",
        args.pretrained_model_name_or_path,
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
    conv_in2_params=[]
    unet_params=[]
    for k, v in unet.named_parameters():
        unet_params.append(np.prod(v.shape))
        if 'conv_in2' in k:
            conv_in2_params.append(np.prod(v.shape))

            if 'weight' in k:
                v.data.normal_(0, init_scale)
            elif 'bias' in k:
                v.data.zero_()
            print(k, 'New params')
            v.requires_grad = True
            unet_new_params.append(v)
            unet_new_params_name.append(k)
    lora_params=[]
    for param in unet_lora_params:
        for item in param:
            print(item.shape,'item')
            lora_params.append(np.prod(item.shape))
        # lora_params.append(np.prod(p.shape))
    
    print('New added parameters:')
    print(unet_new_params_name)
    print('New lora parameters:')
    print(unet_lora_params_name)
    vae.requires_grad_(False)

    
    from textrecnet import TextRecNet
    recnet = TextRecNet(in_channels=4, out_channels=len(p2idx), max_char=args.max_char_seq_length)
    recnet_params=[]
    for key,val in recnet.named_parameters():
        recnet_params.append(np.prod(val.shape))

    print('unet\t{}'.format(np.sum(unet_params)))
    print('lora\t{}'.format(np.sum(lora_params)))
    print('recnet\t{}'.format(np.sum(recnet_params)))
    print('conv_in2\t{}'.format(np.sum(conv_in2_params)))

    if args.use_xformers:
        set_use_memory_efficient_attention_xformers(unet, True)
        set_use_memory_efficient_attention_xformers(vae, True)
        # set_use_memory_efficient_attention_xformers(fusion_module, True)

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
        {"params": recnet.parameters(), "lr": args.learning_rate},
        # {"params": clip_model.parameters() , "lr": args.learning_rate/1.0},
        # {"params": unet_new_params , "lr": args.learning_rate/10.0},
        # {"params": fusion_module_params, "lr": args.learning_rate},
    ]
    params_to_optimize_recnet = [
        {"params": recnet.parameters(), "lr": args.learning_rate},
    ]

    optimizer = optimizer_class(
        params_to_optimize_unet,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # optimizer_recnet = optimizer_class(
    #     params_to_optimize_recnet,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )

    noise_scheduler = DDPMScheduler.from_config(
        # "stabilityai/stable-diffusion-2-1", 
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )

    if not (hostname == 'ubuntu' or hostname.startswith('Qlab') or hostname.startswith('qlab')):
        # instance_data_list = ['ArT','COCO','icdar2013','TextOCR','totaltext2', 'mario_laion_sampled5M']
        # instance_data_list = ['mario_laion_sampled5M']
        # instance_data_list = ['mario_laion']
        instance_data_list=args.instance_data_list.split('-')
        # instance_data_list = ['ArT','COCO','icdar2013','TextOCR','totaltext2']

    else:
        # instance_data_list = ['ArT','COCO','icdar2013','TextOCR','totaltext2', 'mario_laion100K']
        # instance_data_list = ['mario_laion100K']
        instance_data_list=args.instance_data_list.split('-')



    train_dataset = OCRDataset(
        # char_tokenizer=char_tokenizer,
        instance_data_list=instance_data_list,
        instance_data_root=args.instance_data_root,
        placeholder_token=args.placeholder_token,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        size=args.resolution,
        tokenizer=tokenizer,
        unet_training=False,
        include_suffix= args.include_suffix,
        suffix_drop_rate=args.suffix_drop_rate,
        max_char_seq_length=args.max_char_seq_length,
        max_word_len=25,#output_gt will be in (max_word_len+1) dim
        roi_weight=args.roi_weight,#output_gt will be in (max_word_len+1) dim
    )
    
    if not (hostname == 'ubuntu' or hostname.startswith('Qlab') or hostname.startswith('qlab')):
        mario_db=['mario_laion']
    else:
        mario_db=['mario_laion500K_ml']
    if args.mario_batch_size:
        train_dataset_mario = OCRDataset(
            instance_data_list=mario_db,
            instance_data_root=args.instance_data_root,
            placeholder_token=args.placeholder_token,
            class_data_root=args.class_data_dir if args.with_prior_preservation else None,
            size=args.resolution,
            tokenizer=tokenizer,
            unet_training=False,
            include_suffix= args.include_suffix,
            suffix_drop_rate=args.suffix_drop_rate,
            max_char_seq_length=args.max_char_seq_length,
            max_word_len=25,#output_gt will be in (max_word_len+1) dim
            roi_weight=args.roi_weight,#output_gt will be in (max_word_len+1) dim
        )


    def collate_fn(examples):
        if examples[0]["instance_roi_weight_mask"] is not None:
            roi_weight_mask = torch.cat([torch.from_numpy(example["instance_roi_weight_mask"]).unsqueeze(0).unsqueeze(0) for example in examples], dim=0)
        else:
            roi_weight_mask=None
        input_ids = [example["instance_prompt_ids"] for example in examples]
        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        pixel_values = [example["instance_images"] for example in examples]

        rendered_whole_images = [example["instance_rendered_whole_image"] for example in examples]
        rendered_whole_images_rgb = [example["instance_rendered_whole_image_rgb"] for example in examples]
        masks = [example["instance_mask"] for example in examples]
        pixel_values = torch.stack(pixel_values)
        rendered_whole_images = torch.stack(rendered_whole_images)
        rendered_whole_images_rgb = torch.stack(rendered_whole_images_rgb)
        masks = torch.stack(masks)
        char_gts=[]
        for eidx,example in enumerate(examples):
            char_gts+=example["instance_char_gts"]#returns: T,26
        char_gts = torch.stack(char_gts) #shape N,T,26
        nonzero_idxs=[]
        text_boxes = []
        for example in examples:
            nonzero_idxs.append(len(example["instance_text_boxes"])>0)

            if len(example["instance_text_boxes"]):
                text_boxes.append(example["instance_text_boxes"])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        # load laion data
        batch = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "masks": masks,
            "rendered_whole_images_rgb": rendered_whole_images_rgb,
            "text_boxes": text_boxes,
            "char_gts": char_gts,
            "nonzero_idxs": nonzero_idxs,
            "roi_weight_mask": roi_weight_mask,
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
        # segnet,
        recnet,
    ) = accelerator.prepare(unet, 
                            train_dataloader,
                            # segnet,
                            recnet
                            )
    
    if args.resume_unet_ckpt_path and args.resume_unet_ckpt_path!='None':
        state_dict = torch.load(args.resume_unet_ckpt_path, map_location=torch.device('cpu'))
        unet.load_state_dict(state_dict)
        print('unet parameters loaded')
        del state_dict
    if args.resume_recnet_ckpt_path and args.resume_recnet_ckpt_path!='None':
        state_dict = torch.load(args.resume_recnet_ckpt_path, map_location=torch.device('cpu'))
        new_state_dict={}
        for key in recnet.state_dict():
            print(key,'recnet_defined_key')
        for key in state_dict:
            print(key,'recnet_saved_key')
            if (hostname == 'ubuntu' or hostname.startswith('Qlab') or hostname.startswith('qlab')) and 'final' in key:
                continue
            new_state_dict[key]=state_dict[key]
        if (hostname == 'ubuntu' or hostname.startswith('Qlab') or hostname.startswith('qlab')):
            recnet.load_state_dict(new_state_dict,strict=True)
        else:
            recnet.load_state_dict(new_state_dict,strict=True) #azure
        print('recnet parameters loaded')
        del state_dict

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
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))
    # Train!
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
    if not (hostname == 'ubuntu' or hostname.startswith('Qlab') or hostname.startswith('qlab')):
        display_steps = 100
    else:
        display_steps = 1
    if args.criterion=='ce_loss':
        ce_criterion = torch.nn.CrossEntropyLoss()
    # elif args.criterion=='focal_loss':
    #     from focal_loss import FocalLoss
    #     ce_criterion = FocalLoss(gamma=args.gamma)
    #     print('focal_loss')
    else:
        assert False
    
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            unet.train()
            input_images=batch["pixel_values"].to(accelerator.device,dtype=weight_dtype)
            masks = batch["masks"].to(accelerator.device,dtype=weight_dtype)
            rendered_whole_images_rgb=batch["rendered_whole_images_rgb"].to(accelerator.device,dtype=weight_dtype)
            input_ids = batch["input_ids"].to(accelerator.device)
            nonzero_idxs = batch["nonzero_idxs"]

            # Load laion batch and merge
            if args.laion_batch_size:
                # Laion data
                laion_batch=load_laion_batch(laion_loader,tokenizer)
                input_images=torch.cat((input_images,laion_batch["pixel_values"].to(accelerator.device,dtype=weight_dtype)),0)
                masks=torch.cat((batch["masks"].to(accelerator.device,dtype=weight_dtype),laion_batch["masks"].to(accelerator.device,dtype=weight_dtype)),0)
                rendered_whole_images_rgb=torch.cat((batch["rendered_whole_images_rgb"].to(accelerator.device,dtype=weight_dtype),laion_batch["rendered_whole_images_rgb"].to(accelerator.device,dtype=weight_dtype)),0)
                input_ids=torch.cat((batch["input_ids"].to(accelerator.device),
                                     laion_batch["input_ids"].to(accelerator.device)),0).to(accelerator.device)
                nonzero_idxs=nonzero_idxs+laion_batch["nonzero_idxs"]
            if args.mario_batch_size:
                # Mario data
                mario_batch=load_mario_batch(mario_loader,tokenizer)
                input_images=torch.cat((input_images,mario_batch["pixel_values"].to(accelerator.device,dtype=weight_dtype)),0)
                masks = torch.cat((masks,mario_batch["masks"].to(accelerator.device,dtype=weight_dtype)),0)
                rendered_whole_images_rgb=torch.cat((rendered_whole_images_rgb,mario_batch["rendered_whole_images_rgb"].to(accelerator.device,dtype=weight_dtype)),0)
                input_ids = torch.cat((input_ids,mario_batch["input_ids"].to(accelerator.device)),0)
                nonzero_idxs=nonzero_idxs+mario_batch["nonzero_idxs"]
            masks_64 = torch.nn.functional.interpolate(masks, size=(64, 64))
            with torch.no_grad():
                latents = vae.encode(input_images.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215
                rendering_emb = vae.encode(rendered_whole_images_rgb.to(dtype=weight_dtype)).latent_dist.sample()
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
            # print(latents.shape,'latents.shape',masks_64.shape,'masks_64.shape',rendering_emb.shape,'rendering_emb.shape')
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  # L'=a(t)L+b(t) #([1, 4, 64, 113])
            unet_input = torch.cat([noisy_latents, masks_64, rendering_emb], dim=1)

            clip_text_embedding = text_encoder(input_ids)[0].to(accelerator.device, dtype=weight_dtype)
            drop_mask = (torch.rand(bsz)>0.1)*1.0
            drop_mask = drop_mask.view(-1, 1, 1, 1).to(accelerator.device)
            unet_input[:, 4:, :, :] *= drop_mask
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
            if args.with_prior_preservation:  # DONT CARE

                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)
                loss_unet = (
                    F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        .mean([1, 2, 3])
                        .mean()
                )
                prior_loss = F.mse_loss(
                    model_pred_prior.float(), target_prior.float(), reduction="mean"
                )
                loss_unet = loss_unet + args.prior_loss_weight * prior_loss  #
            else:
                roi_weight_mask= F.interpolate(batch["roi_weight_mask"].float(), size=(64, 64), mode="nearest").long().to(accelerator.device)
                n,c,h,w=roi_weight_mask.shape
                mario_idxs=[False]*len(batch["roi_weight_mask"])
                if args.laion_batch_size:
                    roi_weight_mask_laion=torch.ones((args.laion_batch_size,c,h,w)).long().to(accelerator.device)
                    roi_weight_mask=torch.cat((roi_weight_mask,roi_weight_mask_laion),0)
                    mario_idxs+=[False]*len(roi_weight_mask_laion)
                if args.mario_batch_size:
                    roi_weight_mask_mario= F.interpolate(mario_batch["roi_weight_mask"].float(), size=(64, 64), mode="nearest").long().to(accelerator.device)
                    roi_weight_mask=torch.cat((roi_weight_mask,roi_weight_mask_mario),0)
                    mario_idxs+=[True]*len(roi_weight_mask_mario)
                mario_idxs=torch.BoolTensor(mario_idxs)
                non_mario_idxs=torch.logical_not(mario_idxs)
                if args.mario_unet_loss_weight==0:
                    # Totally exclude from the unet_loss
                     # only include pexe/laion image for unet mse loss
                    loss_unet = F.mse_loss(model_pred[non_mario_idxs].float(), target[non_mario_idxs].float(), reduction="none")
                    loss_unet=torch.mean(loss_unet*roi_weight_mask[non_mario_idxs])
                else:
                    loss_unet = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss_unet[mario_idxs]*=args.mario_unet_loss_weight
                    loss_unet=torch.mean(loss_unet*roi_weight_mask)
                    # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            if noise_scheduler.config.prediction_type == "epsilon":  # CHECK
                pred_x0 = noise_scheduler.get_x0_from_noise(model_pred, timesteps, noisy_latents) # pred_x0 corresponds to (latents*0.18215)

            elif noise_scheduler.config.prediction_type == "v_prediction":
                pred_x0 = noise_scheduler.get_x0_from_velocity(model_pred, timesteps, noisy_latents) # pred_x0 corresponds to (latents*0.18215)
            else:
                assert False
            
            text_boxes=batch["text_boxes"]
            if args.mario_batch_size:
                text_boxes=text_boxes+mario_batch["text_boxes"]
            rec_loss=None
            if len(text_boxes):
                for idx in range(len(text_boxes)):
                    text_boxes[idx]=text_boxes[idx].cuda()
                assert len(latents[nonzero_idxs])==len(text_boxes)
                assert len(pred_x0[nonzero_idxs])==len(text_boxes)
                text_gts=batch["char_gts"].to(accelerator.device).long()
                if args.mario_batch_size:
                    text_gts=torch.cat((text_gts, mario_batch["char_gts"].to(accelerator.device).long()),0)
                roi_features = roi_align((pred_x0[nonzero_idxs]).to(device=accelerator.device),text_boxes,output_size=(32,128),spatial_scale=64/512).to(device=accelerator.device,dtype=weight_dtype)
                text_preds_x0 = recnet(roi_features)
                
                rec_loss = ce_criterion(text_preds_x0.view(-1,text_preds_x0.shape[-1]), text_gts.contiguous().view(-1))
                loss_unet = loss_unet + args.rec_loss_weight * rec_loss
                


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
                if (not (hostname == 'ubuntu' or hostname.startswith('Qlab') or hostname.startswith('qlab'))) and accelerator.is_main_process and global_step % display_steps == 0:
                    run.log('loss_unet', loss_unet.detach().item())
                    run.log('rec_loss', rec_loss.detach().item()*args.rec_loss_weight)

            # Visualization
            # timesteps=timesteps.detach().cpu().numpy()[nonzero_idxs]
            maxstep=0
            viz_idx=0

            for step_idx,step in enumerate(timesteps):
                if step>maxstep:
                    maxstep=step
                    viz_idx=step_idx
            if (global_step%args.visualize_steps)==0 and accelerator.is_main_process and (hostname == 'ubuntu' or hostname.startswith('Qlab') or hostname.startswith('qlab')):
                # input_images=input_images[nonzero_idxs]
                # rendered_whole_images_rgb=rendered_whole_images_rgb[nonzero_idxs]
                # masks=masks[nonzero_idxs]
                # 1. visualize x0
                pred_x0 = (1 / vae.config.scaling_factor) * pred_x0
                pred_x0=pred_x0[viz_idx].unsqueeze(0).to(vae.device,dtype=weight_dtype)
                image_x0 = vae.decode(pred_x0).sample

                image_x0 = (image_x0 / 2 + 0.5).clamp(0, 1)
                image_x0 = image_x0.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                image_x0=numpy_to_pil(image_x0)[0]
                image_x0.save(os.path.join(viz_dir,'{:05d}_tstep{:04d}_image_x0_idx{}.jpg'.format(global_step,timesteps[viz_idx],viz_idx))) #good

                # 2. visualize input
                input_images = (input_images / 2 + 0.5).clamp(0, 1)
                input_np = input_images.detach().cpu().permute(0, 2, 3, 1).float().numpy()[viz_idx]
                input_pil=numpy_to_pil(input_np)[0]
                # input_pil=visualize_box(input_pil,text_boxes[viz_idx])

                # 3. visualize mask
                image_mask=masks.detach().cpu().permute(0, 2, 3, 1).float().numpy()[viz_idx] # 0 ~ 1 range
                image_mask=cv2.resize(image_mask,dsize=(512,512),interpolation=cv2.INTER_NEAREST)
                Image.fromarray(image_mask*255).convert('RGB').save(os.path.join(viz_dir,'{:05d}_tstep{:04d}_mask_idx{}.jpg'.format(global_step,timesteps[viz_idx],viz_idx)))
                input_pil.save(os.path.join(viz_dir,'{:05d}_tstep{:04d}_image_input_idx{}.jpg'.format(global_step,timesteps[viz_idx],viz_idx)))
                # 4. spatial rendering
                rendered_whole_np = rendered_whole_images_rgb.detach().cpu().permute(0, 2, 3, 1).float().numpy()[viz_idx]
                rendered_whole_np=(rendered_whole_np*0.5)+0.5
                rendered_whole_pil=Image.fromarray((rendered_whole_np*255).astype(np.uint8)).convert('RGB')
                # rendered_whole_pil=visualize_box(rendered_whole_pil,text_boxes[viz_idx])
                rendered_whole_pil.save(os.path.join(viz_dir,'{:05d}_tstep{:04d}_image_whole_rendered_idx{}.jpg'.format(global_step,timesteps[viz_idx],viz_idx)))
                # for idx in range(len(text_boxes)):
                #     if idx==viz_idx:
                #         end_idx+=len(text_boxes[idx])
                #         break
                #     start_idx+=len(text_boxes[idx])
                #     end_idx+=len(text_boxes[idx])
            # Visualization


            # index_no_updates = torch.arange(len(tokenizer)) != placeholder_token_id
            # with torch.no_grad():
            #     text_encoder.get_input_embeddings().weight[
            #         index_no_updates
            #     ] = orig_embeds_params[index_no_updates]

            global_step += 1
            if accelerator.sync_gradients:
                # if (args.save_steps and global_step - last_save >= args.save_steps) or (global_step<=1):
                if (global_step - 1) % args.save_steps == 0:    
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
                        # pipeline = StableDiffusionPipeline.from_pretrained(
                        pipeline = StableDiffusionPipeline(
                            # "stabilityai/stable-diffusion-2-1",
                            vae=accelerator.unwrap_model(vae, **extra_args),
                            unet=accelerator.unwrap_model(unet, **extra_args),
                            text_encoder=accelerator.unwrap_model(text_encoder, **extra_args),
                            tokenizer=accelerator.unwrap_model(tokenizer, **extra_args),
                            scheduler=accelerator.unwrap_model(std_scheduler, **extra_args),
                            feature_extractor=accelerator.unwrap_model(std_fe_extractor, **extra_args),
                            safety_checker=None,
                            requires_safety_checker=False,
                        )
                        
                        glyph_texts_batch = [
                                                ["Midnight"],
                                                ["Galactic"],
                                                ["Nostalgia"],
                                                # ["Revolution"],
                                                [],
                                                ["Organic","Bliss"]
                                             ]
                        inference_lang_list_batch=[['english']]*len(glyph_texts_batch)
                        viz_rendering_list=[]
                        viz_result_list=[]
                        coords_batch=[[[80, 160, 430, 300]],
                                         [[80, 160, 430, 300]],
                                         [[80, 160, 430, 300]],
                                        #  [[80, 160, 430, 300]],
                                         [[]],
                                         [[80, 120, 420, 240],[80, 250, 420, 370]]]
                        prompt=["pop art text"]*len(coords_batch)
                        mask_batch=[]
                        rendering_batch=[]
                        for gen_idx in range(len(glyph_texts_batch)):
                            coords=np.array(coords_batch[gen_idx])
                            coords=coords.astype(np.int32).tolist()
                            glyph_texts=glyph_texts_batch[gen_idx]
                            random_mask_np=generate_mask(height=512,width=512,coords=coords,mask_type='rectangle')
                            random_mask=Image.fromarray(random_mask_np)
                            masks_tensor = mask_transforms(random_mask).float().unsqueeze(0).to(accelerator.device,dtype=weight_dtype)
                            # torch.Size([1, 1, 64, 64]) masks_64.shape
                            masks_64 = torch.nn.functional.interpolate(masks_tensor, size=(64, 64), mode='nearest') # torch.Size([1, 1, 64, 64])
                            mask_batch.append(masks_64)
                            assert np.all(np.unique(random_mask_np)==np.array([0,1])) or np.all(np.unique(random_mask_np)==np.array([0]))
                            rendered_whole_images=generate_spatial_rendering(width=512,height=512,
                                                                                    words=glyph_texts,
                                                                                    dst_coords=coords)
                            rendering_drawn=visualize_box(rendered_whole_images.convert("RGB"),coords)
                            rendered_whole_images_rgb=image_transforms(rendered_whole_images.convert("RGB"))
                            rendered_whole_images_rgb=rendered_whole_images_rgb.unsqueeze(0).to(accelerator.device,dtype=weight_dtype)# torch.Size([1, 3, 512, 512])

                            rendering_batch.append(rendered_whole_images_rgb)
                            viz_rendering_list.append(rendering_drawn)


                        mask_batch=torch.cat(mask_batch,0)            
                        rendering_batch=torch.cat(rendering_batch,0)            
                        print('Start pipeline')
                        print(prompt[0],'prompt')
                        with torch.no_grad():
                            if args.debug:
                                num_inference_steps=2
                            else:
                                num_inference_steps=50
                            images = pipeline(prompt=prompt, 
                                            spatial_render=rendering_batch, 
                                            masks_tensor=mask_batch, #not inverted
                                            num_inference_steps=num_inference_steps, 
                                            guidance_scale=7.5, width=512, height=512).images
                        merged_viz = Image.new('RGB', (512*2, 512*len(coords_batch)), (255, 255, 255))
                        for idx, (res,rend) in enumerate(zip(images,viz_rendering_list)):
                            merged_viz.paste(res.convert('RGB'),(0,idx*512))
                            merged_viz.paste(rend.convert('RGB'),(512,idx*512))
                        merged_viz.save(os.path.join(sample_dir, "img_{:06d}_result.jpg".format(global_step)))
                        # image.save(os.path.join(sample_dir, "img_{:06d}_guided.jpg".format(global_step)))
                        if not (hostname == 'ubuntu' or hostname.startswith('Qlab') or hostname.startswith('qlab')):
                            run.log_image(name=str(os.path.join("img_{:06d}_result.jpg".format(global_step))), path=str(os.path.join(sample_dir, "img_{:06d}_result.jpg".format(global_step))), \
                                plot=None, description=str(os.path.join(sample_dir, "img_{:06d}_result.jpg".format(global_step))))

                        
                        if (global_step - 1) % (args.save_steps*5) == 0:
                            filename_unet = (f"{args.output_dir}/unet_weight_s{global_step}.pt")
                            filename_recnet = (f"{args.output_dir}/recnet_weight_s{global_step}.pt")
                            filename_optimizer = (f"{args.output_dir}/optimizer_weight_s{global_step}.pt")
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
                        }
                    logs ['rec_loss']=rec_loss.detach().item()*args.rec_loss_weight
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                    if global_step >= args.max_train_steps:
                        break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
