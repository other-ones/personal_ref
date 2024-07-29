import copy
import os
import numpy as np
import shutil
from PIL import Image
import cv2


def visualize_polygon(image,coords):
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale = 0.6
    image=np.array(image)
    color=(0,255,0)
    thickness=1
    coords=coords.reshape(-1,2)
    cv2.line(image, coords[0], coords[1], color, thickness)
    cv2.line(image, coords[1], coords[2], color, thickness)
    cv2.line(image, coords[2], coords[3], color, thickness)
    cv2.line(image, coords[3], coords[0], color, thickness)
    image=Image.fromarray(image)
    return image



src_root='/data/twkim/diffusion/ocr-dataset/merged_logos2'
src_imgroot=os.path.join(src_root,'images/train')
src_labelroot=os.path.join(src_root,'labels/train')
imgfiles=os.listdir(src_imgroot)
dst_root='/data/twkim/diffusion/ocr-dataset/merged_logos3'
dst_imgroot=os.path.join(dst_root,'images/train')
dst_labelroot=os.path.join(dst_root,'labels/train')
os.makedirs(dst_imgroot,exist_ok=True)
os.makedirs(dst_labelroot,exist_ok=True)

error_file=open('errors.txt','w')
for f in imgfiles:
    try:
        fpath=os.path.join(src_imgroot,f)
        labelpath=os.path.join(src_labelroot,f.split('.')[0]+'.txt')
        label_lines=open(labelpath,'r').readlines()
        dst_label_path=os.path.join(dst_labelroot,f.split('.')[0]+'.txt')
        dst_img_path=os.path.join(dst_imgroot,f)
        print(fpath)
        img=Image.open(fpath)#.convert('RGB')
        imgw,imgh=img.size
        minside=min(imgw,imgh)
        maxside=max(imgw,imgh)
        if minside<100:
            continue
        if minside>=4000:
            neww=int(imgw*0.5)
            newh=int(imgh*0.5)
            img_resized=img.resize((neww,newh))
            coords_list=[]
            words_list=[]
            for line in label_lines:
                splits=line.strip().split('\t')
                coords,_,logo=splits
                coords=np.array(coords.split(',')).astype(np.int32)
                if minside>=3000:
                    coords=(coords*0.5).astype(np.int32)
                coords_list.append(coords)
                words_list.append(logo)
            dstfile=open(dst_label_path,'w')
            for coords, word in zip(coords_list,words_list):
                x1,y1,x2,y2,x3,y3,x4,y4=coords
                dstfile.write('{},{},{},{},{},{},{},{}\tlogo\t{}\n'.format(x1,y1,x2,y2,x3,y3,x4,y4,word))
            dstfile.close()
            img_resized.save(dst_img_path)
                

        else:
            img.save(dst_img_path)
            shutil.copy(labelpath,dst_label_path)
    except Exception as e:
        error_file.write('{}\t{}\n'.format(f,e))
        
    
    
    
        