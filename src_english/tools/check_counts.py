import os
import numpy as np
import shutil
root='/data/twkim/diffusion/ocr-dataset//mario_eval2/layouts'
flist=os.listdir(root)
num_lines=[]
for ff in flist:
    fpath=os.path.join(root,ff)
    lines=open(fpath).readlines()
    num_lines.append(len(lines))
print(np.max(num_lines),np.min(num_lines))