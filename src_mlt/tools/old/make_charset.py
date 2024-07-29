import os
import numpy as np
import json
label_root='/data/twkim/diffusion/ocr-dataset/pexel3/labels/train'
files=os.listdir(label_root)
chars=[]
for file in files:
    lines=open(os.path.join(label_root,file)).readlines()
    for line in lines:
        line=line.strip()
        splits=line.split('\t')
        coords,lang,word=splits
        for char in word:
            if char not in chars:
                chars.append(char)
chars=sorted(chars)
chars=['[s]']+chars+['[UNK]']

dst_path='../ckpt/chartokenizer/char_vocab4.json'
dstfile=open(dst_path,'w',encoding='utf-8')
json.dump(chars,dstfile)