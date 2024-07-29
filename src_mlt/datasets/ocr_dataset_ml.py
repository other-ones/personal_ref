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
from utils import generate_mask_ml,generate_spatial_rendering_ml
import string
Image.MAX_IMAGE_PIXELS = 1000000000

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




class OCRDatasetML(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
            self,
            instance_data_list,
            instance_data_root,
            tokenizer=None,
            mask_all_ratio=0.5,
            max_word_len=25,
            diversify_reference=False,
            roi_weight=1,
            synth_ratio=0.5,
            charset_path=None,
            script_info=None,
            
    ):
        self.script_info = script_info
        self.synth_ratio = synth_ratio
        self.instance_data_list = instance_data_list
        self.max_word_len=max_word_len
        self.mask_all_ratio=mask_all_ratio
        self.tokenizer=tokenizer
        self.instance_image_paths = []
        self.instance_mask_paths = []
        self.instance_words = []
        self.instance_coords = []
        self.per_db_count={}
        self.instance_data_root=instance_data_root
        self.num_instance_images = 0
        import json
        with open(charset_path) as f:
            self.letters=json.load(f)
        self.EOS=0
        self.caption_lines={}
        for db in self.instance_data_list:
            caption_path = os.path.join(instance_data_root, db, '{}_blip_caption.txt'.format(db))
            with open(caption_path, 'r',encoding='utf-8', errors='ignore') as f:
                caption_lines=f.readlines()
                print(len(caption_lines),'len(caption_lines)')
            self.caption_lines[db]=caption_lines
            self.num_instance_images+=len(caption_lines)
        for db in self.instance_data_list:
            if 'icdar' in db:
                self.make_icdar_per_script_indices(db)
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        if 'mario' in self.instance_data_list[0] and len(self.instance_data_list)==1:
            prob_map={
                'mario':1.0,
            }
        else:
            prob_map={
                    'icdar':0.1,
                    'synth_pexel_german':0.05,
                    'synth_pexel_italian':0.05,
                    'synth_pexel_french':0.05,
                    'synth_pexel_hindi':0.05,
                    'synth_pexel_bengali':0.05,
                    'synth_pexel_thai':0.15,
                    'synth_pexel_greek':0.15,
                    'synth_pexel_russian':0.15,
                }
            # prob_map={
            #         'icdar':0.3,
            #         'synth_pexel_hindi':0.7,
            #         # 'synth_pexel_russian':0.15,
            #     }
            
        # make sure sum=1
        print(prob_map,'prob_map')
        print(self.instance_data_list,'')
        assert len(prob_map)==len(self.instance_data_list)
        self.db_ratios={}
        for db_prefix in prob_map:
            for db_name in self.instance_data_list:
                if db_prefix in db_name:
                    self.db_ratios[db_name]=prob_map[db_prefix]

    def make_icdar_per_script_indices(self,db_name,threshold=8):
        icdar_caption_lines=self.caption_lines[db_name]
        self.per_lang_caption_indices={}
        """
        00001 - 01000:  arabic
        01001 - 02000:  english
        02001 - 03000:  french
        03001 - 04000:  chinese
        04001 - 05000:  german
        05001 - 06000:  korean
        06001 - 07000:  japanese
        07001 - 08000:  italian
        08001 - 09000:  bangla
        09001 - 10000:  hindi
        """
        lang2range={
            'english':list(range(1001,2001)),
            'french':list(range(2001,3001)),
            'german':list(range(4001,5001)),
            'italian':list(range(7001,8001)),
            'bengali':list(range(8001,9001)),
            'hindi':list(range(9001,10001))
            }
        for lang in lang2range:
            self.per_lang_caption_indices[lang]=[]
            assert len(lang2range[lang])
        for caption_line_index in range(len(icdar_caption_lines)):
            caption_line=icdar_caption_lines[caption_line_index]
            line = caption_line.strip()
            img_file, caption = line.split('\t')
            filename_idx=int(img_file.split('.')[0].split('_')[-1])
            text_label_path = os.path.join(self.instance_data_root,db_name, 'labels/train', img_file.split('.')[0]+'.txt')
            text_label_lines=open(text_label_path,encoding='utf-8-sig').readlines()
            if len(text_label_lines)>threshold or len(text_label_lines)<1:
                continue
            for lang in lang2range:
                cur_lang_range=lang2range[lang]
                if filename_idx in cur_lang_range:
                    self.per_lang_caption_indices[lang].append(caption_line_index)
                    break

    def __len__(self):
        return self.num_instance_images#len(self.db_list)
    def get_icdar_idx(self,index,caption_lines,db_name,threshold=16):
        """
        00001 - 01000:  arabic
        01001 - 02000:  english
        02001 - 03000:  french
        03001 - 04000:  chinese
        04001 - 05000:  german
        05001 - 06000:  korean
        06001 - 07000:  japanese
        07001 - 08000:  italian
        08001 - 09000:  bangla
        09001 - 10000:  hindi
        """
        
        lang2script={
            'english':'latin',
            'french':'latin',
            'german':'latin',
            'italian':'latin',
            'bengali':'bengali',
            'hindi':'hindi',
        }
        sampled_lang=np.random.choice(['english','french','german','italian','bengali','hindi'],
                                      p=[1/21,2/21,2/21,2/21,1/3,1/3])
        lang_caption_indices=self.per_lang_caption_indices[sampled_lang]
        # print(len(lang_caption_indices),'lang_caption_indices',sampled_lang)
        caption_index=np.random.choice(lang_caption_indices)
        sampled_script=lang2script[sampled_lang]
        return caption_index, sampled_script, sampled_lang
    
    def __getitem__(self, index):
        global count
        example = {}
        scale=64/512
        whilecount1=0
        while True:
            db_name=np.random.choice(list(self.db_ratios.keys()),p=list(self.db_ratios.values()))
            caption_lines=self.caption_lines[db_name]
            if 'icdar' in db_name:
                index,range_script,range_lang=self.get_icdar_idx(index=index,db_name=db_name,caption_lines=caption_lines)
            # 2. from selected dataset,
            # get coords/words/script/langs
            line=caption_lines[index%len(caption_lines)]
            line = line.strip()
            img_file, caption = line.split('\t')
            img_path = os.path.join(self.instance_data_root, db_name,'images/train', img_file)
            whilecount1+=1
            if whilecount1>=10000:
                assert False
            text_label_path = os.path.join(self.instance_data_root,db_name, 'labels/train', img_file.split('.')[0]+'.txt')
            coords_list=[]
            words_list=[]
            script_idx_list=[]
            scripts_list=[]
            lang_list=[]
            if not os.path.exists(text_label_path):
                index+=1
                # print('cont1',text_label_path,'text_label_path')
                continue
            text_label_lines=open(text_label_path,encoding='utf-8-sig').readlines()
            if not len(text_label_lines):
                index+=1
                # print('cont2_text_label_lines')
                continue
            for text_label_line in open(text_label_path,encoding='utf-8-sig').readlines():
                text_label_line = text_label_line.strip()
                label_splits = text_label_line.split('\t')
                # (1). Parse Script and Language
                #NOTE: mario_laion dataset do not have language information, hence len(splits)==2
                if db_name=='mario_laion': 
                    script='latin'
                    lang='english'
                else:
                    if len(label_splits)!=3: # STUCK HERE when mario_laion
                        continue
                    if db_name=='mario_laion500K_ml':
                        script='latin'
                        lang='english'
                    elif 'icdar' in db_name:
                        lang=range_lang
                        if lang=='bangla':
                            lang='bengali'
                        script=self.script_info.lang2script[lang]
                    else: # synth
                        lang=label_splits[-2]
                        script=self.script_info.lang2script[lang]
                if not(lang!='bangla' and script!='bangla'):
                    print(lang,'lang',script,'script',db_name,'db_name',text_label_path,'text_label_path')
                assert lang!='bangla' and script!='bangla'

                # (2). Parse Coord
                coord_splits=np.array(label_splits[0].split(',')).astype(np.int32)
                if (len(coord_splits)%2)!=0 or len(coord_splits)<8:
                    continue
                coords=np.array(coord_splits).astype(np.int32).reshape(-1,2)
                xs,ys=coords[:,0],coords[:,1] 
                max_x,max_y=np.max(xs),np.max(ys)
                min_x,min_y=np.min(xs),np.min(ys)
                if not (max_x>(min_x) and max_y>(min_y)):
                    # print('cont4')
                    continue
                # (3). Parse Word
                word=label_splits[-1]
                if not word:
                    # print('cont5')
                    continue
                if len(word)>26:
                    # print('cont6')
                    continue
                if script not in self.script_info.script2idx:
                    # print('cont7',script,self.script_info.script2idx)
                    continue
                # (4). Append extracted information
                script_idx=self.script_info.script2idx[script]
                coords_list.append(coords.reshape(-1).tolist())
                script_idx_list.append(script_idx)
                scripts_list.append(script)
                lang_list.append(lang)
                words_list.append(word)
                assert len(word)<64 and len(word)>0
            if len(words_list):
                input_img=Image.open(img_path).convert('RGB')
                break
            index+=1 
            
            
        assert len(words_list)==len(coords_list) and len(coords_list)==len(lang_list)
        
        img_w,img_h=input_img.size


        # 1. Scale Text boxes
        # Scale the Coordinates to be in 512-scale
        scale=64/512
        w_ratio=512/img_w
        h_ratio=512/img_h
        text_boxes=[]
        valid_words=[]
        valid_coords=[]
        valid_script_idx_list=[]
        valid_scripts_list=[]
        valid_lang_list=[]
        for idx in range(len(coords_list)):
            script=scripts_list[idx]
            lang=lang_list[idx]
            script_idx=script_idx_list[idx]
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
            valid_script_idx_list.append(script_idx)
            valid_scripts_list.append(script)
            valid_lang_list.append(lang)
            text_boxes.append(text_box)
            valid_words.append(words_list[idx]) 
            valid_coords.append(crd.reshape(-1)) 
        
        words_list=valid_words
        valid_num_words=len(valid_words)
        coords_list=valid_coords
        script_idx_list=valid_script_idx_list
        scripts_list=valid_scripts_list
        lang_list=valid_lang_list
        if len(text_boxes):
            # NOTE HERE!!!!!!!!!!!!
            # Text Boxes is only for text recognition
            text_boxes_wo_logos=[]
            for tb,lang in zip(text_boxes,lang_list):
                assert lang!='logo'
                text_boxes_wo_logos.append(tb)

            if len(text_boxes_wo_logos):
                example["instance_text_boxes"] = torch.Tensor(np.array(text_boxes_wo_logos).astype(np.int32))
            else:
                example["instance_text_boxes"] = []
            



        # 2. Spatial Rendering
        # !!!!!!!!!!!!!!!!!!!!!!!!!
        # NOTE: words_list contains 
        # 1) characters or words or 
        # 2) name of logos
        # !!!!!!!!!!!!!!!!!!!!!!!!!
        assert len(words_list)==len(text_boxes) and len(words_list)==len(lang_list)
        # generate_spatial_rendering_ml(width, height, words,dst_coords=None,lang_list=None,font_root=None):
        spatial_rendering=generate_spatial_rendering_ml(width=512,height=512,words=words_list,dst_coords=text_boxes,lang_list=lang_list)
        
            
        # 3. Spatial Mask
        instance_mask_np = generate_mask_ml(512, 512, coords_list,mask_type='polygon')#range: 0~script_idx
        instance_mask=Image.fromarray(instance_mask_np)
        instance_mask=mask_transforms(instance_mask).float().unsqueeze(0)


                

        # 5. Input Ids
        # prompt input_ids
        
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

        # 6. Char-gts for text recognition
        # max_word_len:25+1
        char_gts=[]
        nonlatin_idxs=[]
        for w_idx,word in enumerate(words_list):
            token = torch.zeros(self.max_word_len+1)#zero for [s] token. shape (26,)
            w_script_idx=script_idx_list[w_idx]
            w_script=scripts_list[w_idx]
            # NOTE HERE!!!!!!!!!!!!
            if w_script!='latin':
                nonlatin_idxs+=([True]*(self.max_word_len+1))
            else:
                nonlatin_idxs+=([False]*(self.max_word_len+1))
            for i in range(min(len(word), self.max_word_len)):
                token[i] = self.script_info.char2idx[word[i]]
                
            if len(word) >= self.max_word_len:
                token[-1] = self.EOS
            else:
                token[len(word)] = self.EOS
            char_gts.append(token)
        example["instance_nonlatin_idxs"]=nonlatin_idxs
        assert len(char_gts)==len(text_boxes_wo_logos)
        example["instance_char_gts"] = char_gts # (num_words_in_image,26)











        spatial_rendering_rgb=spatial_rendering.convert('RGB')
        spatial_rendering_rgb=self.image_transforms(spatial_rendering_rgb)
        example["instance_spatial_rendering_rgb"] = spatial_rendering_rgb

        input_img  = input_img.resize((512, 512))
        input_img=self.image_transforms(input_img)
        example["instance_images"] = input_img
        example["instance_mask"] = instance_mask[0]
        if 'synth_pexel' in db_name:
            example["instance_synth_idx"] = [True]
        else:
            example["instance_synth_idx"] = [False]
        return example
 