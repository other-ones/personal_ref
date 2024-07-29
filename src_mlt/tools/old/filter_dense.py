import numpy as np
import os
label_root='/data/twkim/diffusion/ocr-dataset/pexel/labels/train'
caption_path='/data/twkim/diffusion/ocr-dataset/pexel/pexel_blip_caption_dense.txt'
dst_caption_path='/data/twkim/diffusion/ocr-dataset/pexel/pexel_blip_caption.txt'
dst_file=open(dst_caption_path,'w',encoding='utf-8')
with open(caption_path, 'r',encoding='utf-8', errors='ignore') as f:
    lines=f.readlines()
    for line in lines:
        line=line.strip()
        file,caption=line.split('\t')
        name=file.split('.')[0]
        label_path=os.path.join(label_root,name+'.txt')
        label_lines=open(label_path).readlines()
        box_counts=0
        for line in label_lines:
            line=line.strip()
            splits=line.split('\t')
            coords,_,word=splits
            coords=np.array(coords.split(',')).astype(np.int32).reshape(-1,2)
            xs,ys=coords[:,0],coords[:,1] 
            max_x,max_y=np.max(xs),np.max(ys)
            min_x,min_y=np.min(xs),np.min(ys)
            if not (max_x>(min_x) and max_y>(min_y)) :
                continue
            box_counts+=1
        if box_counts<=10:
            dst_file.write('{}\t{}\n'.format(file,caption))
dst_file.close()
