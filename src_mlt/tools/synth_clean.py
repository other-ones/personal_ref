import os
import numpy as np
import shutil
root='/data/twkim/diffusion/ocr-dataset/'
lang_folders=os.listdir(root)
for lf in sorted(lang_folders):
    if not ('synth_pexel' in lf and lf.endswith('2')):
        continue
    lf_path=os.path.join(root,lf)
    new_lf=lf.replace('2','3')
    dst_lf_path=os.path.join(root,new_lf)
    print(lf_path)
    img_root=os.path.join(lf_path,'images/train')
    label_root=os.path.join(lf_path,'labels/train')

    dst_img_root=os.path.join(dst_lf_path,'images/train')
    dst_label_root=os.path.join(dst_lf_path,'labels/train')
    os.makedirs(dst_img_root,exist_ok=True)
    os.makedirs(dst_label_root,exist_ok=True)

    caption_lines=open(os.path.join(lf_path,'{}_blip_caption.txt'.format(lf))).readlines()
    f2caption_map={}
    count=0
    for line in caption_lines:
        line=line.strip()
        splits=line.split('\t')
        f,caption=splits
        f2caption_map[f]=caption
    ifiles=os.listdir(img_root)


        
    valid_ifiles=[]
    for ifile in ifiles:
        ipath =os.path.join(img_root,ifile)
        lpath=os.path.join(label_root,ifile.split('.')[0]+'.txt')
        if not(os.path.exists(ipath) and os.path.exists(lpath) and ifile in f2caption_map):
            count+=1
        else:
            valid_ifiles.append(ifile)
    dst_caption_file=open(os.path.join(dst_lf_path,'{}_blip_caption.txt'.format(new_lf)),'w')
    for vifile in valid_ifiles:
        src_vipath =os.path.join(img_root,vifile)
        src_vlpath=os.path.join(label_root,vifile.split('.')[0]+'.txt')
        caption=f2caption_map[vifile]
        dst_caption_file.write('{}\t{}\n'.format(vifile,caption))
        dst_vipath=os.path.join(dst_img_root,vifile)
        dst_vlpath=os.path.join(dst_label_root,vifile.split('.')[0]+'.txt')
        # print(os.path.exists(src_vipath),os.path.exists(src_vlpath))
        os.rename(src_vipath,dst_vipath)
        os.rename(src_vlpath,dst_vlpath)
    print(count,'count')
