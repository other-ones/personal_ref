import random
import time
import torchvision.transforms as TF
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
Image.MAX_IMAGE_PIXELS = 1000000000
# from utils import random_crop_image_mask,create_mask_from_coords,generate_spatial_rendering,generate_random_rectangles
# from utils import generate_mask,generate_spatial_rendering_ml
from utils import generate_mask_ml,generate_spatial_rendering_ml
import string
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

appending_templates_single=[
    "is in the image.",
    "is in the photo.",
    "is in the drawing.",
    "is in the generated result.",

    "is shown in the image.",
    "is shown in the photo.",
    "is shown in the drawing.",
    "is shown in the rendition.",
    "is shown in the generated result.",

    "exists.",
    "exists in the image.",
    "exists in the drawing.",
    "exists in the photo.",
    "exists in the generation.",
    "exists in the generated result.",

    "is shown.", 
    "is shown in the image.", 
    "is shown in the drawing.", 
    "is shown in the photo.", 
    "is shown in the generation.", 
    "is shown in the generated result.", 

    "is displayed.",
    "is displayed in the image.",
    "is displayed in the drawing.",
    "is displayed in the photo.",
    "is displayed in the generation.",
    "is displayed in the generated result.",

    "can be found in the image.",
    "can be found in the drawing.",
    "can be found in the photo.",
    "can be found in the generated result.",
    "can be found in the generation.",

    "should be shown in the image.",
    "should be shown in the drawing.",
    "should be shown in the photo.",
    "should be shown in the generation.",
    "should be shown in the generated result.",
]
appending_templates_mult=[
    "are in the image.",
    "are in the photo.",
    "are in the drawing.",
    "are in the generated result.",

    "are shown in the image.",
    "are shown in the photo.",
    "are shown in the drawing.",
    "are shown in the rendition.",
    "are shown in the generated result.",

    "exist in the image.",
    "exist in the drawing.",
    "exist in the photo.",
    "exist in the generation.",
    "exist in the generated result.",

    "are shown.", 
    "are shown in the image.", 
    "are shown in the drawing.", 
    "are shown in the photo.", 
    "are shown in the generation.", 
    "are shown in the generated result.", 

    "are displayed.",
    "are displayed in the image.",
    "are displayed in the drawing.",
    "are displayed in the photo.",
    "are displayed in the generation.",
    "are displayed in the generated result.",

    "can be found in the image.",
    "can be found in the drawing.",
    "can be found in the photo.",
    "can be found in the generated result.",
    "can be found in the generation.",

    "should be shown in the image.",
    "should be shown in the drawing.",
    "should be shown in the photo.",
    "should be shown in the generation.",
    "should be shown in the generated result.",
]
prepending_templates_single = [
    "a text{}",
    "a rendering of{}",
    "a rendition of{}",
    "a text writing of{}",
    "a writing of{}",
    "a word saying{}",
    "a text rendering of{}",
    "a printed text of{}",
    "a printed rendering of{}",
    "a printed word of{}",
    "a glyph text of{}",
    "a glyph of{}",
    "a text saying{}",
    "a rendered text saying{}",
    "a rendered word in{}",
    "a rendered writing in{}",
    "a spelling of{}",
    "a word spelled in{}",
    "a written word saying{}",
    "a written word meaning{}",

]
prepending_templates_mult = [
    "texts{}",
    "renderings of{}",
    "renditions of{}",
    "renditions of{}",
    "text writings of{}",
    "writings of{}",
    "words saying{}",
    "text renderings of{}",
    "printed texts of{}",
    "printed text words of{}",
    "printed renderings of{}",
    "printed words of{}",
    "a glyph text of{}",
    "glyph texts of{}",
    "glyphs of{}",
    "texts saying{}",
    "rendered texts saying{}",
    "rendered words in{}",
    "rendered writings of{}",
    "a spelling of{}",
    "spellings of{}",
    "words spelled in{}",
    "a written word saying{}",
    "a written word meaning{}",
]


glyph_transforms =  transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomApply([color_jitter], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466,0.4578275,0.40821073), std=(0.26862954,0.26130258,0.27577711))
        ])

class OCRDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
            self,
            # char_tokenizer,
            instance_data_list,
            instance_data_root,
            script_info,
            tokenizer=None,
            suffix_drop_rate=0.2,
            include_suffix=False,
            max_word_len=25,
            roi_weight=1,
    ):
        self.script_info=script_info
        self.roi_weight=roi_weight
        self.max_word_len=max_word_len
        self.suffix_drop_rate=suffix_drop_rate
        self.include_suffix=include_suffix
        self.instance_data_list = instance_data_list
        self.glyph_transforms=glyph_transforms
        self.tokenizer=tokenizer
        self.caption_lines=[]
        self.db_list=[]
        self.instance_data_root=instance_data_root
        self.num_instance_images = 0
        import json
        with open(self.script_info.charset_path) as f:
            self.letters=json.load(f)
        self.EOS=0
        for db in self.instance_data_list:
            caption_path = os.path.join(instance_data_root, db, '{}_blip_caption.txt'.format(db))
            with open(caption_path, 'r',encoding='utf-8', errors='ignore') as f:
                caption_lines=f.readlines()
                print(db,'num_samples:{}'.format(len(caption_lines)))
            self.caption_lines+=caption_lines
            self.db_list+=[db]*(len(caption_lines))
            self.num_instance_images+=len(caption_lines)
        assert len(self.db_list)==len(self.caption_lines)
        

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    def __len__(self):
        return self.num_instance_images#len(self.db_list)
    def random_augment(self,image):
        if np.random.random()<0.2:
            image=TF.GaussianBlur(kernel_size=(5,5),sigma=(0.1,2))(image)
        if np.random.random()<0.1:
            jitter = TF.ColorJitter(brightness=.5, hue=.1)
            image=jitter(image)
        if np.random.random()<0.1:
            posterizer = TF.RandomPosterize(bits=2)
            image=posterizer(image)
        if np.random.random()<0.1:
            equalizer = TF.RandomEqualize()
            image=equalizer(image)
        return image
    def __getitem__(self, index):
        global count
        example = {}
        scale=64/512
        whilecount=0
        while True:
            whilecount+=1
            if whilecount>=100000:
                print('Infinite loop stuck at getitem')
                assert False
            line=self.caption_lines[index%len(self.caption_lines)]
            db_name=self.db_list[index%len(self.caption_lines)]
            line = line.strip()
            img_file, caption = line.split('\t')
            img_path = os.path.join(self.instance_data_root, db_name,'images/train', img_file)
            text_label_path = os.path.join(self.instance_data_root,db_name, 'labels/train', img_file.split('.')[0]+'.txt')
            coords_list=[]
            words_list=[]
            if not os.path.exists(text_label_path):
                index+=1
                continue
            text_label_lines= open(text_label_path,encoding='utf-8-sig').readlines()
            if not len(text_label_lines):
                index+=1
                continue
            for text_label_line in text_label_lines:
                text_label_line = text_label_line.strip()
                label_splits = text_label_line.split('\t')
                coord_splits=np.array(label_splits[0].split(',')).astype(np.int32)
                if (len(coord_splits)%2)!=0 or len(coord_splits)<8:
                    continue
                word=label_splits[-1]
                if not word:
                    continue
                coords=np.array(coord_splits).astype(np.int32).reshape(-1,2)
                xs,ys=coords[:,0],coords[:,1] 
                max_x,max_y=np.max(xs),np.max(ys)
                min_x,min_y=np.min(xs),np.min(ys)
                if not (max_x>(min_x) and max_y>(min_y)):
                    continue
                coords_list.append(coords.reshape(-1).tolist())
                words_list.append(word)
                assert len(word)<64 and len(word)>0
            if len(words_list) and len(coords_list) and (len(coords_list)==len(words_list)):
                input_img=Image.open(img_path).convert('RGB')
                break
            index+=1
        # original_num_words=len(words_list)
        """1. input image"""
        assert len(words_list)
        # words=words_list
        # coords_sampled=coords
        img_w,img_h=input_img.size
        # Text boxes for ROI pooling
        scale=64/512
        w_ratio=512/img_w
        h_ratio=512/img_h
        text_boxes=[]
        valid_words=[]
        valid_coords=[]
        for idx in range(len(coords_list)):
            crd=coords_list[idx]
            crd=np.array(crd).reshape(-1,2)
            xs,ys=crd[:,0],crd[:,1] 
            if 'mario_laion' not in db_name: # 'mario_laion' already provides label in (512,512) scale
                xs=(xs*w_ratio)
                ys=(ys*h_ratio)
                crd[:,0]=xs
                crd[:,1]=ys
            max_x,max_y=np.max(xs),np.max(ys)
            min_x,min_y=np.min(xs),np.min(ys)
            if not (max_x>(min_x) and max_y>(min_y)):
                print(max_x,min_x,'xs')
                print(max_y,min_y,'ys')
                continue
            
            text_box=[min_x,min_y,max_x,max_y] #4
            text_boxes.append(text_box)
            valid_words.append(words_list[idx]) 
            valid_coords.append(crd.reshape(-1)) 
        
        words_list=valid_words
        coords_list=valid_coords
        # rendered_whole_image=generate_spatial_rendering(width=512,height=512,words=words_list,
        #                                                              dst_coords=text_boxes,
        #                                                              font_path=self.font_path,
        #                                                              )
        lang_list=['english']*len(words_list)
        rendered_whole_image=generate_spatial_rendering_ml(width=512,height=512,words=words_list,dst_coords=text_boxes,lang_list=lang_list)
        if len(text_boxes):
            example["instance_text_boxes"] = torch.Tensor(np.array(text_boxes).astype(np.int32))
        else:
            example["instance_text_boxes"] = []
        # if 'mario_laion' in db_name:
        #     # roi (masked): 1 | non_roi: 0
        #     instance_mask_np = generate_mask(512, 512, coords_list,mask_type='polygon')#range: 0~1
        # else:
        #     instance_mask_np = generate_mask(512, 512, coords_list,mask_type='polygon')#range: 0~1
        instance_mask_np = generate_mask_ml(512, 512, coords_list,mask_type='polygon')#range: 0~script_idx
        # roi_weight_mask
        roi_weight_mask=np.copy(instance_mask_np)
        bg_idxs=np.where(roi_weight_mask==0)
        fg_idxs=np.where(roi_weight_mask>0)
        roi_weight_mask[bg_idxs]=1.
        roi_weight_mask[fg_idxs]=self.roi_weight
        example["instance_roi_weight_mask"] = roi_weight_mask 
        input_img  = input_img.resize((512, 512))
        count+=1
        instance_mask=Image.fromarray(instance_mask_np)
        input_img=self.image_transforms(input_img)
        instance_mask=mask_transforms(instance_mask).float().unsqueeze(0)
        rendered_whole_image_224=rendered_whole_image.resize((224,224),Image.Resampling.LANCZOS)
        rendered_whole_image_rgb=rendered_whole_image.convert('RGB')
        glyph_clip=self.glyph_transforms(rendered_whole_image_224)


        if self.include_suffix:
            if np.random.rand()>self.suffix_drop_rate:
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
        if np.random.rand()<0.1:
            caption=""
        if self.tokenizer is None:
            example["instance_prompt_ids"]=None
        else:
            example["instance_prompt_ids"] = self.tokenizer(
                caption,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        char_gts=[]
        nonlatin_idxs=[]
        for w_idx,word in enumerate(words_list):
            token = torch.zeros(self.max_word_len+1)#zero for [s] token. shape (26,)
            nonlatin_idxs+=([False]*(self.max_word_len+1))
            for i in range(min(len(word), self.max_word_len)):
                token[i] = self.script_info.char2idx[word[i]]
            if len(word) >= self.max_word_len:
                token[-1] = self.EOS
            else:
                token[len(word)] = self.EOS
            char_gts.append(token)
        # print(len(nonlatin_idxs),'len(nonlatin_idxs) mario')
        example["instance_nonlatin_idxs"]=nonlatin_idxs
        example["instance_char_gts"] = char_gts #(num_words_in_image,26)
        rendered_whole_image=self.image_transforms(rendered_whole_image.convert("L"))
        rendered_whole_image_rgb=self.image_transforms(rendered_whole_image_rgb)
        example["instance_spatial_rendering"] = rendered_whole_image
        example["instance_spatial_rendering_rgb"] = rendered_whole_image_rgb
        example["instance_images"] = input_img
        example["instance_mask"] = instance_mask[0]
        example["instance_glyph_clip"] = glyph_clip # to clip_vision
        example['is_latin']=[True]
        example['instance_synth_idx']=[False]

        return example
