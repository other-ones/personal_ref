import pdb
import numpy as np
import os
dblist=['icdar2019']
imgroot='/data/twkim/diffusion/ocr-dataset/'
for dbname in dblist:
    root='/data/twkim/diffusion/ocr-dataset/{}'.format(dbname)
    caption_path='/data/twkim/diffusion/ocr-dataset/{}/{}_blip_caption.txt'.format(dbname,dbname)
    with open(caption_path, 'r',encoding='utf-8', errors='ignore') as f:
        lines=f.readlines()
    label_root=os.path.join(root,'labels/train')
    # 1. check label existence
    for line in lines:
        line=line.strip()
        splits=line.split('\t')
        file,caption=splits
        name=file.split('.')[0]
        # if not os.path.exists(os.path.join(label_root,name+'.txt')):
        #     print(line)
    # 2. count boxes
    labels=os.listdir(label_root)
    count_list=[]
    file_list=[]
    for line in lines:
        line=line.strip()
        splits=line.split('\t')
        file,caption=splits
        name=file.split('.')[0]
        label_path=os.path.join(label_root,name+'.txt')
        label_lines=open(label_path).readlines()
        box_counts=0
        for line in label_lines:
            line=line.strip()
            splits=line.split('\t')
            if len(splits)!=3:
                continue
            coords,_,word=splits

            coords=np.array(coords.split(',')).astype(np.int32).reshape(-1,2)
            xs,ys=coords[:,0],coords[:,1] 
            max_x,max_y=np.max(xs),np.max(ys)
            min_x,min_y=np.min(xs),np.min(ys)
            if not (max_x>(min_x) and max_y>(min_y)) :
                # print(label_path,line,'diff',max_x-min_x,max_y-min_y)
                continue
            box_counts+=1
        count_list.append(box_counts)
        file_list.append(file)
    # count_list=sorted(count_list)
    count_list=np.array(count_list)
    file_list=np.array(file_list)
    idxs=np.argsort(count_list)
    count_list=count_list[idxs]
    file_list=file_list[idxs]
    print('here',np.sum(count_list>8),len(count_list))
    idxs=np.where(count_list>100)
    count_list=count_list[idxs]
    file_list=file_list[idxs]
    # print(dbname,np.mean(count_list),count_list[:50],count_list[-50:])
    # print(dbname,file_list[:20],file_list[-20:])
    for item,count in zip(file_list[-20:],count_list[-20:]):
        print(os.path.join(imgroot,dbname,'images/train',item),count)
    # pdb.set_trace()