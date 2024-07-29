import os
import numpy as np
root='/data/twkim/diffusion/ocr-dataset/icdar2019/labels/train'
files=os.listdir(root)
min_idx=np.infty
max_idx=0
# 06001 - 07000:  Japanese
idxs=[]
for file in files:
    filename=file.split('.')[0]
    filename_splits=filename.split('_')
    if len(filename_splits)!=3:
        continue
    # print(filename_splits,file)
    fileidx=int(filename_splits[-1])
    lines=open(os.path.join(root,file)).readlines()
    for line in lines:
        line=line.strip()
        splits=line.split('\t')
        # if len(splits)!=3:
        #     print(file,line)
        script=splits[1]
        if script=='japanese':
            if max_idx<fileidx:
                max_idx=fileidx
            if min_idx>fileidx:
                min_idx=fileidx
            idxs.append(fileidx)
print(max_idx,min_idx)
print(sorted(idxs))
        