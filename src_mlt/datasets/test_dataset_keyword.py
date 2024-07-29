import re
from utils import generate_spatial_rendering_ml
from utils import generate_mask_ml
from utils import get_uniform_layout_word
import random
import time
import cv2
count=0
from shapely.geometry import Polygon
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
import pdb
from torch.utils.data import Dataset
import os
from PIL import Image,ImageDraw
import string
# from utils import random_crop_with_roi
from utils import generate_pos_neg_masks_ml
Image.MAX_IMAGE_PIXELS = 1000000000

alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' ' # len(aphabet) = 95
alphabet_dic = {}
for index, c in enumerate(alphabet):
    alphabet_dic[c] = index + 1 # the index 0 stands for non-character

mask_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),

    ]
)
roi_mask_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),

    ]
)

appending_templates_single=[
    "is in the image.",
]
appending_templates_mult=[
    "are in the image",
]
prepending_templates_single = [
    "a text{}",
]
prepending_templates_mult = [
    "texts{}",
]
class TestDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
            self,
            # char_tokenizer,
            dbname,
            instance_data_root,
            tokenizer=None,
            diversify_font=False,
            roi_weight=1,
            uniform_layout=False,
            max_word_len=25,
            debug=False,
            sample_dir=None,
            attnres=16,
            mask_bg=0,
            mask_fg=0,
            coords_jitter=0,
            # chunk_id=0
    ):
        self.coords_jitter = coords_jitter
        self.mask_bg = mask_bg
        self.mask_fg = mask_fg
        self.attnres = attnres
        self.sample_dir = sample_dir
        self.debug = debug
        self.max_word_len = max_word_len
        self.uniform_layout = uniform_layout
        self.dbname = dbname
        self.leng2ar={}
        ar_lines=open('ar_list.txt').readlines()
        for line in ar_lines:
            line=line.strip()
            splits=line.split('\t')
            length,med_ar,mean_ar=splits
            length=int(length)
            aspect_ratio=float(mean_ar)
            self.leng2ar[length]=aspect_ratio
        self.roi_weight=roi_weight
        self.diversify_font=diversify_font
        self.tokenizer=tokenizer
        self.instance_data_root=instance_data_root
        self.num_instance_images = 0




        caption_path = os.path.join(instance_data_root, dbname, '{}_blip_caption.txt'.format(dbname))
        with open(caption_path, 'r',encoding='utf-8', errors='ignore') as f:
            caption_lines=f.readlines()
            np.random.shuffle(caption_lines)
            self.caption_lines=[]
            for line in caption_lines[:]:
                line_split=line.split('\t')
                layout_file,caption=line_split
                fname=layout_file.split('.')[0]
                saved_path=os.path.join(sample_dir,'{}.png'.format(fname))
                if os.path.exists(saved_path):
                    continue
                if 'spanish_' in line  and '.txt' in line:
                    continue
                else:
                    self.caption_lines.append(line)
            self.num_instance_images+=len(self.caption_lines)




        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    def __len__(self):
        return self.num_instance_images#len(self.db_list)
    def __getitem__(self, index):
        global count
        example = {}


        # New implementation
        # 1. load caption/keyword
        caption_line=self.caption_lines[index]
        caption_line = caption_line.strip()
        layout_file, caption = caption_line.split('\t')
        caption_words=[]
        for caption_word in caption.split():
            caption_word=re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039])", "", caption_word)
            if len(caption_word):
                caption_words.append(caption_word)
        caption=' '.join(caption_words)
        if caption.endswith('.'):
            caption=caption[:-1]
        layout_path = os.path.join(self.instance_data_root, self.dbname,'layouts', layout_file)
        example["layout_files"]=layout_file
        layout_lines=open(layout_path,encoding='utf-8-sig').readlines()

        # 2. get layouts
        coords_list=[]
        keywords_list=[]
        lang_list=[]
        text_boxes=[]
        if len(layout_lines):
            for layout_line in layout_lines:
                layout_line = layout_line.strip()
                layout_splits = layout_line.split('\t')
                coords,lang,keyword=layout_splits
                coords=np.array(coords.split(',')).astype(np.int32)
                x1,y1,x2,y2,x3,y3,x4,y4=coords
                # adjust coords
                xs,ys=coords.reshape(-1,2)[:,0],coords.reshape(-1,2)[:,1]
                minx,miny=np.min(xs),np.min(ys)
                maxx,maxy=np.max(xs),np.max(ys)
                offset=maxx-511
                if offset>0:
                    maxx=maxx-offset
                    minx=minx-offset
                height=maxy-miny
                width=maxx-minx
                word_length=len(keyword)
                tb=[minx,miny,maxx,maxy]
                text_boxes.append(tb)
                keywords_list.append(keyword)
                lang_list.append(lang)
                coords_list.append(coords)
                lang_list.append(lang)
        if self.uniform_layout and len(coords_list):
            print('uniform layout')
            # assert False
            text_boxes=[]
            ar_list=[self.leng2ar[len(key_word)] for key_word in keywords_list]
            coords_list=get_uniform_layout_word(keywords_list,ar_list=ar_list)
            text_boxes=[]
            for coords in coords_list:
                xs,ys=coords.reshape(-1,2)[:,0],coords.reshape(-1,2)[:,1]
                minx,miny,maxx,maxy=np.min(xs),np.min(ys),np.max(xs),np.max(ys)
                tb=[minx,miny,maxx,maxy]
                text_boxes.append(tb)     
        assert len(coords_list)==len(keywords_list),'num_words!=num_coords'
        # 2. get layouts


        # 3. Binary Mask
        instance_mask_np = generate_mask_ml(512, 512, coords_list,mask_type='polygon')#fg:1 bg:0
        instance_mask=Image.fromarray(instance_mask_np)
        instance_mask=mask_transforms(instance_mask).float().unsqueeze(0)
        example["instance_mask"] = instance_mask[0]
        # 3. Binary Mask


        #### 4. Parse word index to coordinate index ####
        widx_to_crd_idx={}
        caption_splits=caption.split()
        hits=[False]*len(keywords_list) # index of keyword list, all keywords should be visited 
        for widx in range(len(caption_splits)):
            cap_word=caption_splits[widx]
            # if cap_word in keywords_list: #keyword
            if cap_word=='words':
                widx_to_crd_idx[widx]=[]
                for crd_idx in range(len(coords_list)):
                    if (not hits[crd_idx]):
                        widx_to_crd_idx[widx].append(crd_idx)
                        hits[crd_idx]=True
        if np.sum(hits)!=len(hits):
            print()
            print()
            print('Assertion HERE')
            print(np.array(coords_list)[hits],'corods')
            print(np.array(keywords_list)[hits],'keywords_list')
            print(caption,'caption')
            print(hits,'hits')
            print(keywords_list,'keywords_list')
            print(layout_file ,'layout_file')
            print(coords_list,'coords')
            print(np.sum(hits),'np.sum(hits)')
            print(len(hits),'len(hits)')
            print()
            print()
        assert np.sum(hits)==len(hits),'keyword parsing error'
        #### 5. Parse word index to coordinate index ####
        
        




        #### 5. Parse token index to coordinate index ####
        # FINAL GOAL!!!!
        tidx_to_crd_idx_list={} #token index to keyword coordinate index
        # FINAL GOAL!!!!
        is_keyword_tokens=[False]*self.tokenizer.model_max_length #first token is special token
        # index for enumerating token idxs of the tokenized prompt
        tidx=1 #starts from 1 since first token is for the special token
        eot_idx=1 # 1 for start token
        for widx,cap_word in enumerate(caption.split()):
            # convert word to token
            word_token_ids=self.tokenizer.encode(cap_word,add_special_tokens=False)
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
                    assert num_tokens==1
                    is_keyword_tokens[tidx]=True
                    tidx_to_crd_idx_list[tidx]=widx_to_crd_idx[widx]
                tidx+=1
                if tidx==(len(is_keyword_tokens)-2):
                    break
            if tidx==(len(is_keyword_tokens)-2):
                    break
            # word_index to token_index_list mapping
            # e.g., "word saying 'Neurips'" -> word_idx for neurips: 2
            # NOTE: len(word_token_idxs)==(len(input_ids)-2): two for special tokens
        assert not is_keyword_tokens[-1],'last element of is_keyword_tokens should be False'
        #### 5. Parse token index to coordinate index ####








        
        
            
        
        # 2. get per token positive/negative attention target
        # should be (77,512,512)
        from scipy.ndimage.filters import gaussian_filter
        pos_masks,neg_masks = generate_pos_neg_masks_ml(
            512, 
            512, 
            text_boxes,
            is_keyword_tokens,
            tidx_to_crd_idx_list,
            fg_mask=instance_mask_np,
            eot_idx=eot_idx,
            mask_type='rectangle'
            )
        example["instance_pos_masks"] = torch.Tensor(pos_masks).float() # binary mask
        example["instance_neg_masks"] = torch.Tensor(neg_masks).float() # binary mask
        example["instance_is_keyword_tokens"] = torch.Tensor(is_keyword_tokens).bool()
        example["instance_eot_idxs"] = eot_idx



        
            
        spatial_rendering=generate_spatial_rendering_ml(width=512,height=512,words=keywords_list,dst_coords=text_boxes,lang_list=lang_list)
        spatial_rendering_rgb=spatial_rendering.convert('RGB')
        spatial_rendering_rgb=self.image_transforms(spatial_rendering_rgb)
        example["instance_spatial_rendering_rgb"] = spatial_rendering_rgb
        # example["instance_text_boxes"] = text_boxes
        if len(text_boxes):
            example["instance_text_boxes"] = torch.Tensor(np.array(text_boxes).astype(np.int32))
        else:
            example["instance_text_boxes"] = []
            


        
        example["instance_raw_captions"]=caption
        example["instance_prompt_ids"] = self.tokenizer(
            caption,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        # New implementation


       
        
        return example
 