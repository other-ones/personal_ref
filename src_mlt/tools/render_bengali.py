import pickle
import os

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import numpy as np
import random
from PIL import Image,ImageDraw,ImageFont
import random
import cv2
import string
import os
import shutil
import numpy as np
dst_root='bengali_rendered'

shutil.rmtree(dst_root)
os.makedirs(dst_root,exist_ok=True)
lang='bengali'
root=    '/data/twkim/diffusion/ocr-dataset/synth_pexel_{}3/labels/train/'.format(lang)
img_root='/data/twkim/diffusion/ocr-dataset/synth_pexel_{}3/images/train/'.format(lang)
flist=os.listdir(root)

def generate_spatial_rendering_ml(width, height, words,dst_coords=None,lang_list=None):
    image = Image.new('RGB', (width, height), (255, 255, 255)) # SAME
    draw = ImageDraw.Draw(image) # SAME
    for idx in range(len(words)):
        lang=lang_list[idx]
        coords=dst_coords[idx] # SAME
        word = words[idx] # SAME
        # Word labeled as Text
        if lang in ['italian','french','spanish','german','english']:
            font_root=os.path.join('../ml_fonts','english') 
        else:
            font_root=os.path.join('../ml_fonts',lang_list[idx])             
        script_color=0
        available_fonts=os.listdir(font_root)
        font_path=os.path.join(font_root,available_fonts[0])
        x1, y1, x2, y2 = np.array(coords).astype(np.int32) # np.min(xs),np.min(ys),np.max(xs),np.max(ys) # SAME
        region_width = x2 - x1
        region_height = y2 - y1
        min_side=min(region_width,region_height) # SAME
        if region_height>(region_width*2): # Vertical Text
            font_size = int(min(region_width, region_height) / (len(word)))
            if lang_list[idx] in ['korean','chinese']:
                scaler=0.7
            elif lang_list[idx] in ['arabic']:
                scaler=1.5
            elif lang_list[idx] in ['russian','german']:
                scaler=1.1
            elif lang_list[idx] in ['french','greek','thai']:
                scaler=1.3
            elif lang_list[idx] in ['hindi']:
                scaler=1.3

            else:
                scaler=0.9
            font_size=font_size*scaler
            font_size=int(font_size)
            font_size=max(1,font_size)
            font_size=min(min_side,font_size)
            font_size=max(5,font_size)
            font = ImageFont.truetype(font_path, font_size)
            text_width, text_height = draw.textsize(word, font=font)
            text_x = x1 + (region_width - text_width) // 2
            text_y = y1 + (region_height - text_height) // 2
            draw.text((text_x, text_y), word, font=font, fill=script_color)
        else: # Horizontal Text
            divider=(len(word))
            divider=max(1,divider)
            # if lang_list[idx] in ['chinese']:
            #     scaler=0.8
            # elif lang_list[idx] in ['russian','german']:
            #     scaler=1.4
            # elif lang_list[idx] in ['french','greek','thai']:
            #     scaler=1.6
            # elif lang_list[idx] in ['hindi']:
            #     scaler=1.6
            # else:
            #     scaler=1.2 #english

            if lang_list[idx] in ['french','german','english','spanish','italian']:
                scaler=1.4
            elif lang_list[idx] in ['russian']:
                scaler=1.5
            elif lang_list[idx] in ['thai']:
                scaler=1.4
            elif lang_list[idx] in ['greek']:
                scaler=1.5
            elif lang_list[idx] in ['hindi']:
                scaler=3
            elif lang_list[idx] in ['bengali']:
                scaler=1.3
            elif lang_list[idx] in ['korean']:
                print('korean')
                scaler=1.2
            else:
                scaler=1.5
            font_size = int(max(region_width, region_height)*scaler / divider)
            # font_size = int(max(region_width, region_height)*1.4 / len(word))
            font_size=int(font_size)
            font_size=min(min_side,font_size)
            font_size=min(56,font_size)
            if lang_list[idx] in ['korean']:
                font_size=max(28,font_size)
            else:
                font_size=max(34,font_size)


            # font_size=int(font_size)
            # font_size=max(1,font_size)
            # font_size=min(min_side,font_size)
            # font_size=min(56,font_size)
            font = ImageFont.truetype(font_path, font_size)
            text_width, text_height = draw.textsize(word, font=font)
            text_x = x1 + (region_width - text_width) // 2
            text_y = y1 + (region_height - text_height) // 2
            draw.text((text_x, text_y), word, font=font, fill=script_color)
    return image
for f in flist[:10]:
    fpath=os.path.join(root,f)
    lines=open(fpath).readlines()
    coords_list=[]
    glyph_texts=[]
    inference_lang_list=[]
    image_path=os.path.join(img_root,f.split('.')[0]+'.jpg')
    input_img=Image.open(image_path)
    img_w,img_h=input_img.size
    w_ratio=512/img_w
    h_ratio=512/img_h
    for line in lines:
        line=line.strip()
        coords,lang,word=line.split('\t')
        coords=np.array(coords.split(',')).astype(np.float32)
        crd=np.array(coords).reshape(-1,2)
        xs,ys=crd[:,0],crd[:,1] 
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
        glyph_texts.append(word)
        print(word)
        inference_lang_list.append(lang)
        coords_list.append(text_box)
    spatial_renderings=generate_spatial_rendering_ml(width=512,height=512,words=glyph_texts,
                                                        dst_coords=coords_list,lang_list=inference_lang_list)
    dst_path=os.path.join(dst_root,f.split('.')[0]+'_render.jpg')
    dst_path_image=os.path.join(dst_root,f.split('.')[0]+'_input.jpg')
    
    input_img=input_img.resize((512,512))
    spatial_renderings.save(dst_path)
    input_img.save(dst_path_image)