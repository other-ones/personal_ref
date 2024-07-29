import os
root='/data/twkim/diffusion/ocr-dataset/mario_eval2/layouts'
flist=os.listdir(root)
num_words_list=[]
for ff in flist:
    fpath=os.path.join(root,ff)
    lines=open(fpath).readlines()
    num_words=len(lines)
    if num_words>=8:
        print(ff)
    num_words_list.append(num_words)

import numpy as np
num_words_list=np.array(num_words_list)
print(np.sum(num_words_list>7))
print(np.max(num_words_list))
print(np.min(num_words_list))