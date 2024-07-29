import os
import numpy as np
import json
# charset1=json.load(open('/home/twkim/project/azure/text_enh/src_english/ckpt/chartokenizer/char_vocab2.json'))[1:-1]
# charset2=json.load(open('/home/twkim/project/azure/text_enh/src_english/ckpt/chartokenizer/char_vocab4.json'))[1:-1]
# chars=charset1+charset2
# chars=list(set(chars))
# chars=sorted(chars)

chars_english=json.load(open('/home/jacobwang/code/azure/text_enh/src_mlt2/ckpt/chartokenizer/char_vocab_english.json'))[1:-1]
# Pexel synthesis
langs=['arabic','german','hindi','italian','french']
mlt_chars_pexel=[]
for lang in langs:
    lang_path=os.path.join('/data/twkim/diffusion/ocr-dataset/synth_pexel_{}2/labels/train'.format(lang))
    labels=os.listdir(lang_path)
    for label in labels:
        label_path=os.path.join(lang_path,label)
        lines=open(label_path,encoding='utf-8').readlines()
        for line in lines:
            line=line.strip()
            splits=line.split('\t')
            word=splits[-1]
            for char in word:
                char_code=ord(char)
                if char_code>=12356 and char_code<=55141:
                # if char_code>=44032:
                    print(lang,char)
                if not char in mlt_chars_pexel:
                    mlt_chars_pexel.append(char)
mlt_chars_pexel=list(set(mlt_chars_pexel))



# ICDAR2019
mlt_chars_ic19=[]
ic_root='/data/twkim/diffusion/ocr-dataset/icdar2019/labels/train'
files=os.listdir(ic_root)
for file in files:
    filepath=os.path.join(ic_root,file)
    lines=open(filepath).readlines()
    file_idx=int(file.split('.')[0].split('_')[-1])
    if (file_idx<=1000): # Arabic
        range_script='arabic'
        range_lang='arabic'
    elif (file_idx>=1001 and file_idx<=2000): # English
        range_script='latin'
        range_lang='english'
    elif (file_idx>=2001 and file_idx<=3000): # French
        range_script='latin'
        range_lang='french'
    elif (file_idx>=4001 and file_idx<=5000): # German
        range_script='latin'
        range_lang='german'
    elif (file_idx>=7001 and file_idx<=8000): # Italian
        range_script='latin'
        range_lang='italian'
    elif (file_idx>=9001): # Hindi
        range_script='hindi'
        range_lang='hindi'
    else:
        continue
    for line in lines:
        line=line.strip()
        splits=line.split('\t')
        lang=splits[1]
        """
        00001 - 01000:  Arabic
        01001 - 02000:  English
        02001 - 03000:  French
        03001 - 04000:  Chinese
        04001 - 05000:  German
        05001 - 06000:  Korean
        06001 - 07000:  Japanese
        07001 - 08000:  Italian
        08001 - 09000:  Bangla
        09001 - 10000:  Hindi
        """
        # if lang=='chinese' or lang=='bangla' or lang=='japanese' or lang=='korean':
        #     continue
        
        word=splits[-1]
        for char in word:
            char_code=ord(char)
            # 47560 54620
            if char_code>=44032 and char_code<=55141:
            # if char_code>=44032:
                print(lang,char)
            if not char in mlt_chars_ic19:
                mlt_chars_ic19.append(char)
mlt_chars_ic19=list(set(mlt_chars_ic19))

chars=chars_english+mlt_chars_pexel+mlt_chars_ic19
chars=list(set(chars))
chars=sorted(chars)
assert '[s]' not in chars
assert '[UNK]' not in chars
chars=['[s]']+chars+['[UNK]']

dst_path='/home/jacobwang/code/azure/text_enh/src_mlt2/ckpt/chartokenizer/char_vocab_mlt_ic19.json'
dstfile=open(dst_path,'w',encoding='utf-8')
print(len(chars),'len(chars)')
json.dump(chars,dstfile)