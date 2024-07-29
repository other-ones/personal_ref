import json
import os
import shutil
src_root='/data/twkim/diffusion/ocr-dataset/merged_logos2'

caption_lines=open(src_root+'/merged_logos2_blip_caption.txt','r').readlines()
f2caption={}
for line in caption_lines:
    line=line.strip()
    splits=line.split('\t')
    f,caption=splits
    f2caption[f]=caption

dst_db='merged_logos3_ocr'
dst_root='/data/twkim/diffusion/ocr-dataset/{}'.format(dst_db)
dst_caption_path=os.path.join(dst_root,'{}_blip_caption.txt'.format(dst_db))
dst_file=open(dst_caption_path,'w')
img_files=os.listdir(os.path.join(dst_root,'images/train'))
for f in img_files:
    dst_caption=f2caption[f]
    dst_file.write('{}\t{}\n'.format(f,dst_caption))