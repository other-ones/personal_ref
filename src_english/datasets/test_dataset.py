from utils import get_uniform_layout
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
from utils import generate_mask,generate_spatial_rendering
import string
from utils import random_crop_with_roi
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
            include_suffix=True,
            diversify_font=False,
            roi_weight=1,
            charset_path=None,
            uniform_layout=False,
            target_subset=None,
            blank_mask=False,
            blank_ref=False,
            output_dir=None,
            # chunk_id=0
    ):
        self.blank_ref = blank_ref
        self.blank_mask = blank_mask
        self.target_subset = target_subset
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
        self.include_suffix=include_suffix
        # self.char_tokenizer=char_tokenizer
        self.tokenizer=tokenizer
        # self.db_list=[]
        self.instance_data_root=instance_data_root
        self.num_instance_images = 0
        import json


        caption_path = os.path.join(instance_data_root, dbname, '{}_blip_caption.txt'.format(dbname))
        with open(caption_path, 'r',encoding='utf-8', errors='ignore') as f:
            caption_lines=f.readlines()
            self.caption_lines=[]
            for line in caption_lines:
                if 'spanish_' in line  and '.txt' in line:
                    continue
                splits=line.split('\t')
                layout_file, caption = line.split('\t')
                output_path=os.path.join(output_dir,'samples',layout_file.replace('.txt','.png'))
                if os.path.exists(output_path):
                    print(output_path,'exists')
                    continue
                # if not 'ChineseDrawText_91.txt' in line:
                #     continue
                splits=line.strip().split('\t')
                fname=splits[0]
                # if (self.target_subset is not None) and (self.target_subset !='None') and (self.target_subset not in fname):
                if self.target_subset is not None and self.target_subset!='None':
                    if (self.target_subset not in fname):
                        continue
                self.caption_lines.append(line)
            exit()
            print(len(self.caption_lines),'len(self.caption_lines)')
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
        captoin_line=self.caption_lines[index]
        captoin_line = captoin_line.strip()
        layout_file, caption = captoin_line.split('\t')
        caption=caption.replace("\'","")
        layout_path = os.path.join(self.instance_data_root, self.dbname,'layouts', layout_file)
        example["layout_files"]=layout_file
        layout_lines=open(layout_path,encoding='utf-8-sig').readlines()
        # 2. get layouts
        coords_list=[]
        words_list=[]
        lang_list=[]
        text_boxes=[]
        if len(layout_lines):
            for layout_line in layout_lines:
                layout_line = layout_line.strip()
                layout_splits = layout_line.split('\t')
                coords,lang,word=layout_splits
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
                # word=splits[-1]
                word_length=len(word)
                aspect_ratio=self.leng2ar[word_length]
                aspect_ratio=min(2.5,aspect_ratio)
                adjusted_width=int(height*aspect_ratio)
                adjusted_box=[minx,miny,minx+adjusted_width,miny+height]
                adjusted_coords=[minx,miny,
                                minx+adjusted_width,miny,
                                minx+adjusted_width,miny+height,
                                minx,miny+height]
                text_boxes.append(adjusted_box)
                coords_list.append(adjusted_coords)
                # adjust coords


                # tb=[x1,y1,x3,y3]
                # text_boxes.append(tb)
                # coords_list.append(coords)
                words_list.append(word)
                lang_list.append(lang)
        
        if len(coords_list)>=8:
            if self.uniform_layout:
                print('uniform layout')
                ar_list=[]
                for word in words_list:
                    ar=self.leng2ar[len(word)]
                    ar_list.append(ar)
                coords_list=get_uniform_layout(words_list=words_list,ar_list=ar_list)
                # for coords in coords_list:
                text_boxes=[]
                for coords in coords_list:
                    x1,y1,x2,y2,x3,y3,x4,y4=coords
                    tb=[x1,y1,x3,y3]
                    text_boxes.append(tb)
            # else:
            #     coords_list=[]
            #     words_list=[]
            #     lang_list=[]
            #     text_boxes=[]
        if self.blank_ref:
            spatial_rendering = Image.new('RGB', (512, 512), (255, 255, 255))
        else:                                            
            spatial_rendering=generate_spatial_rendering(width=512,height=512,words=words_list,dst_coords=text_boxes)
        
        spatial_rendering_rgb=spatial_rendering.convert('RGB')
        spatial_rendering_rgb=self.image_transforms(spatial_rendering_rgb)
        example["instance_spatial_rendering_rgb"] = spatial_rendering_rgb
        example["instance_text_boxes"] = text_boxes
        # if not len(text_boxes):
        #     example["instance_blank_idxs"]=True
        # else:
        #     example["instance_blank_idxs"]=False


        # 4. Binary Mask
        if self.blank_mask:
            instance_mask_np=np.zeros((512, 512)).astype(np.int32)
        else:
            instance_mask_np = generate_mask(512, 512, coords_list,mask_type='polygon')#
        instance_mask=Image.fromarray(instance_mask_np)
        instance_mask=mask_transforms(instance_mask).float().unsqueeze(0)
        example["instance_mask"] = instance_mask[0]

         # 5. Input Ids
        # prompt input_ids
        if self.include_suffix and len(words_list):
            if len(words_list)>1:
                p_template=prepending_templates_mult
                s_template=appending_templates_mult
            else:
                p_template=prepending_templates_single
                s_template=appending_templates_single
            prepending=p_template[0]
            appending=s_template[0]
            glyph_listing=''
            for word in words_list[:-1]:
                if not word:
                    continue
                glyph_listing+=" \"{}\",".format(word)
            glyph_listing+=" \"{}\"".format(words_list[-1])
            suffix=prepending.format(glyph_listing)+' '+appending
            caption=caption+' '+suffix
        example["instance_raw_captions"]=caption
        example["instance_prompt_ids"] = self.tokenizer(
            caption,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        # New implementation

       

        return example
 