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

coords_word1=np.array([
    [170, 200,350, 270],
                 ])
coords_word2=np.array([
    [170, 190,350, 250],
    [170, 260,350, 320],
    ]
    )
coords_word3=np.array([
    [170, 175,335, 230],
    [170, 235,335, 290],
    [170, 295,335, 355],
    ])






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
            coord_jitter=0,
            # chunk_id=0
    ):
        self.coord_jitter = coord_jitter
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
        caption_path = os.path.join('special_set', 'caption_set.txt')
        with open(caption_path, 'r',encoding='utf-8', errors='ignore') as f:
            caption_lines=f.readlines()
            self.caption_lines=[]
            for line in caption_lines:
                caption=line.strip()
                caption_splits=caption.split()
                word_cat='_'.join(caption_splits)
                layout_file=word_cat
                caption_line='{}\t{}'.format(layout_file,caption)
                self.caption_lines.append(caption_line)
            self.caption_lines=np.array(self.caption_lines).reshape(-1,1)
            self.caption_lines=np.repeat(self.caption_lines,repeats=20,axis=1)
            self.caption_lines=self.caption_lines.reshape(-1).tolist()
            print(len(self.caption_lines),'len(self.caption_lines)')
            self.num_instance_images+=len(self.caption_lines)
        vocab_path=os.path.join('special_set','{}_special.txt'.format(self.target_subset))


        self.word_pool=[]
        with open(os.path.join(vocab_path),encoding='utf-8', errors='ignore') as f:
            lines=f.readlines()
            for line in lines:
                word=line.strip()
                self.word_pool.append(word)

            



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
        example["layout_files"]=layout_file








        # 2. get layouts
        num_words=np.random.choice([1,2,3],p=[0.7,0.2,0.1])
        if num_words==3:
            coords_list_raw=coords_word3
        elif num_words==2:
            coords_list_raw=coords_word2
        else:
            coords_list_raw=coords_word1

            
            
            
        coords_list=[]
        lang_list=[]
        text_boxes=[]
        words_list=np.random.choice(self.word_pool,size=num_words,replace=False)
        if self.coord_jitter:
            global_offset_x=np.random.randint(20,60)
            global_offset_y=np.random.randint(20,60)
            plus_x=np.random.rand()<0.5
            plus_y=np.random.rand()<0.5
            if not plus_x:
                global_offset_x=-1*global_offset_x
            if not plus_y:
                global_offset_y=-1*global_offset_y

        for widx in range(num_words):
            coords=np.array(coords_list_raw[widx]).astype(np.int32)
            if self.coord_jitter:
                x1,y1,x3,y3=coords
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
                    if  num_words>1: #middle
                        min_offset=2
                        max_offset=4
                    if (511-y3-deviation>0):
                        off_y=np.random.randint(min_offset,max_offset)
                    else:
                        off_y=0
                    if widx==(num_words-1): #last word
                        y1=y1+off_y
                        y3=y3+off_y
                else:
                    if (y1>deviation):
                        off_y=np.random.randint(min_offset,max_offset)
                    else:
                        off_y=0
                    if widx==0:
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

            
            # cut off-valued coords
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
            lang_list.append(self.target_subset)
            coords_list.append(coords)
                

        # print(lang_list,len(lang_list),'lang_list')
        # print(len(coords_list),'coords_list')
        # print(len(text_boxes),'text_boxes')
        # print(len(words_list),'words_list')
        # # (2-1) Uniform layout based on template logo size
        # if self.uniform_layout and len(coords_list):
        #     ar_list=[self.leng2ar[len(word)] for word in words_list]
        #     coords_list=get_uniform_layout_word(words_list,ar_list=ar_list)
        #     text_boxes=[]
        #     for coords in coords_list:
        #         xs,ys=coords.reshape(-1,2)[:,0],coords.reshape(-1,2)[:,1]
        #         minx,miny,maxx,maxy=np.min(xs),np.min(ys),np.max(xs),np.max(ys)
        #         tb=[minx,miny,maxx,maxy]
        #         text_boxes.append(tb)                                                       

        
        
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
 