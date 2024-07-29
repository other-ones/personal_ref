import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import sys
from utils import generate_spatial_rendering_ml
os.environ['PYTHONPATH']='/home/twkim/project/azure/refdiffuser/src_mlt'
language='german'
font_path='/home/twkim/project/azure/refdiffuser/src_mlt/ml_fonts/english/GoNotoCurrent.ttf'
glyph_texts_batch = [["dormìpò"],["défià"],["äußert"],["Purdue"]]
inference_lang_list_batch=[["italian"],["french"],["german"],["english"]]
def visualize_box(image,boxes,chars=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    image=np.array(image)
    if chars is not None:
        for box,char in zip(boxes,chars):
            box=(box.detach().cpu().numpy()*2).astype(np.int32)
            x0,y0,x1,y1=box
            box=(box).reshape(-1,2).astype(np.int32)
            point1=box[0]
            point2=box[1]
            image=cv2.rectangle(image,tuple(point1),tuple(point2),color=(0,255,0),thickness=1)   
            image = cv2.putText(image, char, (x0,y1+5), font, fontScale, (255,0,0), 1, cv2.LINE_AA)
    else:
        for box in boxes:
            if torch.is_tensor(box):
                box=(box.detach().cpu().numpy().astype(np.int32)).reshape(-1,2)
            else:
                box=np.array(box).reshape(-1,2)
            point1=box[0]
            point2=box[1]
            image=cv2.rectangle(image,tuple(point1),tuple(point2),color=(0,255,0),thickness=1)
    image=Image.fromarray(image)
    return image


word_root='/home/twkim/project/azure/refdiffuser/evaluation/mlt_benchmark/raw/words'
targets=['italian','spanish','german','french']
dst_root='latin_renderings'
if os.path.exists(dst_root):
    os.system('rm {} -R'.format(dst_root))
os.makedirs(dst_root,exist_ok=True)

for target in targets:
    # target='italian'
    # /home/twkim/project/azure/refdiffuser/evaluation/mlt_benchmark/seen_word_sets/french5K.txt
    fpath='/home/twkim/project/azure/refdiffuser/evaluation/mlt_benchmark/seen_word_sets/{}5K.txt'.format(target)
    lines=open(fpath,encoding='utf-8-sig').readlines()
    target_list=[]
    for line in lines:
        word=line.strip()
        if not word.isascii():
            # print(word)
            target_list.append(word)

    np.random.shuffle(target_list)
    words=target_list[:20]
    for gen_idx in range(len(words)):
        # glyph_texts=glyph_texts_batch[gen_idx]
        glyph_texts=[words[gen_idx]]
        inference_lang_list=[target]
        # inference_lang_list=inference_lang_list_batch[gen_idx]
        coords=np.array([[100, 145,390, 265]])
        coords=coords.astype(np.int32).tolist()
        spatial_renderings=generate_spatial_rendering_ml(width=512,height=512,words=glyph_texts,
                                                        dst_coords=coords,lang_list=inference_lang_list,
                                                        font_root='/home/twkim/project/azure/refdiffuser/src_mlt')
        spatial_renderings=spatial_renderings.convert('RGB')
        spatial_renderings_drawn=visualize_box(spatial_renderings,coords)
        lang=inference_lang_list[0]
        spatial_renderings_drawn.save('latin_renderings/{}_{}.png'.format(target,gen_idx))