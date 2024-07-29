import os
import numpy as np

dblist=[
        'synth_pexel_latin3',
        'synth_pexel_hindi3',
        'synth_pexel_bengali3',
        'synth_pexel_thai3',
        'synth_pexel_greek3',
        'synth_pexel_russian3',
        'mario_laion',
]

root='/data/twkim/diffusion/ocr-dataset'
for db in dblist:
    dbpath=os.path.join(root,db)
    label_root=os.path.join(dbpath,'labels/train')
    flist=os.listdir(label_root)
    for f in flist:
        fpath=os.path.join(label_root,f)
        lines=open(fpath).readlines()
        for line in lines:
            line=line.strip()
            splits=line.split('\t')
            if len(splits)!=3:
                print(fpath,len(splits))
