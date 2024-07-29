from utils import random_crop_with_roi
import numpy as np
from PIL import Image
import os
image_root='/data/twkim/diffusion/ocr-dataset/icdar2019/images/train'
label_root='/data/twkim/diffusion/ocr-dataset/icdar2019/labels/train'
images=os.listdir(image_root)
np.random.shuffle(images)
for image in images:
    image_path=os.path.join(image_root,image)
    label_path=os.path.join(label_root,image.split('.')[0]+'.txt')
    label_lines=open(label_path).readlines()
    word_coords=[]
    img=Image.open(image_path).convert('RGB')
    for line in label_lines:
        line=line.strip()
        splits=line.split('\t')
        if len(splits)!=3:
            continue
        coords=np.array(splits[0].split(',')).astype(np.int32)
        word_coords.append(coords)
    print(image_path)
    if len(word_coords):
        cropped,_=random_crop_with_roi(img,word_coords)
    else:
        print('EMPTY',image_path)
        
    # cropped.save('cropped.jpg')
    # img.save('input.jpg')
    # exit(0)
        
