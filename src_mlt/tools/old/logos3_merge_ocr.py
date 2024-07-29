import shutil
import re
import string
import os
import numpy as np
import json
from PIL import Image
pattern = r'[^a-zA-Z0-9' +r'{}]'.format(string.punctuation)
pattern=re.compile(pattern)
    
            


    
dst_root='/data/twkim/diffusion/ocr-dataset/merged_logos3_ocr'
dst_img_root=os.path.join(dst_root,'images/train')
dst_label_root=os.path.join(dst_root,'labels/train')
os.makedirs(dst_label_root,exist_ok=True)
os.makedirs(dst_img_root,exist_ok=True)


src_root='/data/twkim/diffusion/ocr-dataset/merged_logos2'
dst_root='/data/twkim/diffusion/ocr-dataset/merged_logos3_ocr'
src_img_root=os.path.join(src_root,'images/train')
src_label_root=os.path.join(src_root,'labels/train')
src_ocr_root=os.path.join(src_root,'ocr_labels/train')
labels=os.listdir(src_label_root)
for label in labels:
    label_path=os.path.join(src_label_root,label)
    img_path=os.path.join(src_img_root,label.split('.')[0]+'.jpg')
    img=Image.open(img_path)#.convert('RGB')
    imgw,imgh=img.size
    minside=min(imgw,imgh)
    maxside=max(imgw,imgh)
    if minside<100:
        continue
    dst_img_path=os.path.join(dst_img_root,label.split('.')[0]+'.jpg')
    shutil.copy(img_path,dst_img_path)
    # 1. Parse Logo Label
    lines=open(label_path,'r').readlines()
    logo_coords_list=[]
    logo_lang_list=[]
    logo_words_list=[]
    for line in lines:
        line=line.strip()
        coords,lang,word=line.split('\t')
        coords=np.array(coords.split(',')).astype(np.int32)
        logo_coords_list.append(coords)
        logo_lang_list.append(lang)
        logo_words_list.append(word)
    # write logo label
    if minside>=4000:
        scaler=0.5
    else:
        scaler=1
    dst_label_path=os.path.join(dst_label_root,label)
    dst_label_file=open(dst_label_path,'w')
    for coords,lang,word in zip(logo_coords_list,logo_lang_list,logo_words_list):
        x1,y1,x2,y2,x3,y3,x4,y4=(coords*scaler).astype(np.int32)
        dst_label_file.write('{},{},{},{},{},{},{},{}\t{}\t{}\n'.format(x1,y1,x2,y2,x3,y3,x4,y4,lang,word))
    dst_label_file.close()



    # 2. Parse OCR Label
    ocr_coords_list=[]
    ocr_lang_list=[]
    ocr_words_list=[]
    json_path=os.path.join(src_ocr_root,label.replace('.txt','.json'))
    if not os.path.exists(json_path):
        print('no ocr1')
        continue
    json_file=open(json_path)
    lines=json_file.readlines()
    if len(lines)==0:
        print('no ocr2')
        continue
    json_file.seek(0)
    data=json.load(open(json_path))
    page_data_list=data['analyze_result']['read_results']
    assert len(page_data_list)==1
    img_h=page_data_list[0]['height']
    img_w=page_data_list[0]['width']
    line_data_list=data['analyze_result']['read_results'][0]['lines']
    if len(line_data_list)==0:
        continue
    for line_data in line_data_list:
        for word_data in line_data['words']:
            bbox=word_data['bounding_box']
            word=word_data['text']
            x1,y1,x2,y2,x3,y3,x4,y4=np.array(bbox).astype(np.int32)
            max_x,min_x=max([x1,x2,x3,x4]),min([x1,x2,x3,x4])
            max_y,min_y=max([y1,y2,y3,y4]),min([y1,y2,y3,y4])
            text_width=max_x-min_x
            text_height=max_y-min_y
            if max(text_width,text_height)<0.015*max(img_h,img_w):
                continue
            coords=np.array([x1,y1,x2,y2,x3,y3,x4,y4]).astype(np.int32)
            word=str(word.encode('utf-8').decode())
            out=re.findall(pattern,word)
            if len(out):
                continue
            ocr_coords_list.append(coords)
            ocr_words_list.append(word)
            ocr_lang_list.append('english')
    # Write ocr label
    print('write label')
    dst_label_path=os.path.join(dst_label_root,label)
    dst_label_file=open(dst_label_path,'a')
    for coords,lang,word in zip(ocr_coords_list,ocr_lang_list,ocr_words_list):
        x1,y1,x2,y2,x3,y3,x4,y4=(coords*scaler).astype(np.int32)
        dst_label_file.write('{},{},{},{},{},{},{},{}\t{}\t{}\n'.format(x1,y1,x2,y2,x3,y3,x4,y4,lang,word))
    dst_label_file.close()
            





