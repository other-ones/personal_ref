import numpy as np
from PIL import Image
import os
img_root='/data/twkim/diffusion/ocr-dataset/merged_logos2/images/train'
files=os.listdir(img_root)
min_sides=[]
max_sides=[]
for file in files:
    filepath=os.path.join(img_root,file)
    print(filepath)
    img=Image.open(filepath)
    imgw,imgh=img.size
    min_side=min(imgw,imgh)
    max_side=max(imgw,imgh)
    max_sides.append(max_side)
    min_sides.append(min_side)
min_sides=np.array(min_sides)
max_sides=np.array(max_sides)
minside_files=np.array(files)
maxside_files=np.array(files)


idxs_minsides=np.argsort(min_sides)
idxs_maxsides=np.argsort(max_sides)

min_sides_sorted=min_sides[idxs_minsides]
minside_files_sorted=maxside_files[idxs_minsides]

max_sides_sorted=max_sides[idxs_maxsides]
maxside_files_sorted=maxside_files[idxs_maxsides]
large_files=[]
for item1,item2 in zip(min_sides_sorted[-20:],minside_files_sorted[-20:]):
    large_files.append(item2)
    print(item1,os.path.join(img_root,item2),'min')
    
for item1,item2 in zip(max_sides_sorted[-20:],maxside_files_sorted[-20:]):
    large_files.append(item2)
    print(item1,os.path.join(img_root,item2),'max')
    
caption_lines=open('/data/twkim/diffusion/ocr-dataset/merged_logos2/merged_logos2_blip_caption.txt','r').readlines()
f2caption={}
for line in caption_lines:
    line=line.strip()
    splits=line.split('\t')
    f2caption[splits[0]]=splits[1]
dst_caption_path=os.path.join('../tmp_large_file_captions.txt')    
dst_file=open(dst_caption_path,'w')
large_files=list(set(large_files))
for item in large_files:
    dst_file.write('{}\t{}\n'.format(item,f2caption[item]))
dst_file.close()
    