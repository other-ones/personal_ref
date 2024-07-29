import os
import numpy as np

langs=['italian','german','french', 'arabic', 'hindi','hebrew','greek','thai','russian']
root='/data/twkim/diffusion/ocr-dataset/synth_pexel_{}/labels/train'
for lang in langs:
    lang_path=root.format(lang)
    files=os.listdir(lang_path)
    for f in files:
        fpath=os.path.join(lang_path,f)
        lines=open(fpath).readlines()
        for line in lines:
            line=line.strip()
            splits=line.split('\t')
            word=splits[-1]
            if len(word)>30:
                print(word,fpath)