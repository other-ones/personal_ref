from utils import get_uniform_layout_word,generate_spatial_rendering_ml
count=0
import numpy as np
from torchvision import transforms
from pathlib import Path
import pdb
from torch.utils.data import Dataset
import os
from PIL import Image,ImageDraw
from utils import generate_mask_ml
import string
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
            diversify_reference=False,
            target_subset=None,
            target_language=None,
            coords_jitter=0,
            # chunk_id=0
    ):
        self.target_language = target_language
        self.coords_jitter = coords_jitter
        self.target_subset = target_subset
        self.diversify_reference = diversify_reference
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
                line=line.strip()
                splits=line.split('\t')
                fname,caption=splits
                lang=fname.split('_')[0]
                if 'spanish_' in fname  and '.txt' in line:
                    continue
                # if self.target_language is not None and target_language not in fname:
                #     continue
                if self.target_subset is not None and lang not in self.target_subset:
                    continue
                else:
                    self.caption_lines.append(line)
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
        if self.coords_jitter:
            global_offset_x=np.random.randint(20,60)
            global_offset_y=np.random.randint(20,60)
            plus_x=np.random.rand()<0.5
            plus_y=np.random.rand()<0.5
            if not plus_x:
                global_offset_x=-1*global_offset_x
            if not plus_y:
                global_offset_y=-1*global_offset_y
        if len(layout_lines)<8:
            for lidx,layout_line in enumerate(layout_lines):
                layout_line = layout_line.strip()
                layout_splits = layout_line.split('\t')
                coords,lang,word=layout_splits
                coords=np.array(coords.split(',')).astype(np.int32)
                if self.coords_jitter:
                    x1,y1,x2,y2,x3,y3,x4,y4=coords
                    deviation=50 # deviation from border
                    max_offset=10
                    min_offset=5
                    # random offset for x
                    if np.random.rand()<0.5: # add offset
                        if (511-x3-deviation>0):
                            # off_x=np.random.randint(0,min(max_offset,511-x3-deviation))
                            off_x=np.random.randint(min_offset,max_offset)
                        else:
                            off_x=0
                        x1=x1+off_x
                        x3=x3+off_x
                    else:
                        if (x1>deviation): # subtract offset
                            off_x=np.random.randint(min_offset,max_offset)
                        else:
                            off_x=0
                        x1=x1-off_x
                        x3=x3-off_x

                    # random offset for y
                    if np.random.rand()<0.5: # add offset
                        if  len(layout_lines)>1:
                            min_offset=2
                            max_offset=4
                        if (511-y3-deviation>0):
                            off_y=np.random.randint(min_offset,max_offset)
                        else:
                            off_y=0
                        if lidx==(len(layout_lines)-1):
                            y1=y1+off_y
                            y3=y3+off_y
                    else:
                        if (y1>deviation):
                            off_y=np.random.randint(min_offset,max_offset)
                        else:
                            off_y=0
                        if lidx==0:
                            y1=y1-off_y
                            y3=y3-off_y
                    x1+=global_offset_x
                    x3+=global_offset_x
                    y1+=global_offset_y
                    y3+=global_offset_y
                    coords=[
                            x1,y1,
                            x3,y1,
                            x3,y3,
                            x1,y3]
                    coords=np.array(coords)

                

                xs,ys=coords.reshape(-1,2)[:,0],coords.reshape(-1,2)[:,1]
                minx,miny=np.min(xs),np.min(ys)
                maxx,maxy=np.max(xs),np.max(ys)
                offset=maxx-511
                if offset>0:
                    maxx=maxx-offset
                    minx=minx-offset
                height=maxy-miny
                width=maxx-minx
                tb=[minx,miny,maxx,maxy]
                text_boxes.append(tb)
                words_list.append(word)
                lang_list.append(lang)
                coords_list.append(coords)
                

        
            
        # (2-1) Uniform layout based on template logo size
        if self.uniform_layout and len(coords_list):
            ar_list=[self.leng2ar[len(word)] for word in words_list]
            coords_list=get_uniform_layout_word(words_list,ar_list=ar_list)
            text_boxes=[]
            for coords in coords_list:
                xs,ys=coords.reshape(-1,2)[:,0],coords.reshape(-1,2)[:,1]
                minx,miny,maxx,maxy=np.min(xs),np.min(ys),np.max(xs),np.max(ys)
                tb=[minx,miny,maxx,maxy]
                text_boxes.append(tb)                                                       
        # 3. spatial rendering
        spatial_rendering=generate_spatial_rendering_ml(width=512,height=512,words=words_list,dst_coords=text_boxes,lang_list=lang_list)
        spatial_rendering_rgb=spatial_rendering.convert('RGB')
        spatial_rendering_rgb=self.image_transforms(spatial_rendering_rgb)
        example["instance_spatial_rendering_rgb"] = spatial_rendering_rgb
        example["instance_text_boxes"] = text_boxes


        # 4. Binary Mask
        instance_mask_np = generate_mask_ml(512, 512, coords_list,mask_type='polygon')#range: 0~script_idx
        instance_mask=Image.fromarray(instance_mask_np)
        instance_mask=mask_transforms(instance_mask).float().unsqueeze(0)
        example["instance_mask"] = instance_mask[0]

         # 5. Input Ids
        # prompt input_ids
        example["instance_raw_captions"]=caption
        example["instance_raw_words"]=words_list
        example["instance_prompt_ids"] = self.tokenizer(
            caption,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
       

        return example
 