import os
import json
no_rus_set=json.load(open('/home/twkim/project/azure/refdiffuser/src_mlt/ckpt/chartokenizer/char_vocab_mlt_no_russian_azure.json'))
dst_char_set=[]
src_root='/data/twkim/ocr-dataset'
db='synth_pexel_russian3'
db_root=os.path.join(src_root,db)
label_root=os.path.join(db_root,'labels/train')
flist=os.listdir(label_root)
for f in flist:
    fpath=os.path.join(label_root,f)
    lines=open(fpath,encoding='utf-8-sig').readlines()
    for line in lines:
        line=line.strip()
        splits=line.split('\t')
        word=splits[-1]
        for char in word:
            char_code=ord(char)
            if not char in no_rus_set:
                print(char)