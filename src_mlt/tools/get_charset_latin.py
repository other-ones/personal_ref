import os
import json
dst_path='ckpt/chartokenizer/char_vocab_mlt_latin.json'
src_root='/data/twkim/ocr-dataset'
latins=['italian','german','french']
english_chars=json.load(open('../ckpt/chartokenizer/char_vocab_english.json'))[1:-1]
print(english_chars,'english_chars')

db_list=['synth_pexel_latin3']
dst_char_set=[]
dst_char_set+=english_chars
for db in db_list:
    print(db,'db')
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
                # if char_code>=12356 and char_code<=55141:
                #     print(char)
                if not char in dst_char_set:
                    dst_char_set.append(char)


# ICDAR2019
ic_root='/data/twkim/diffusion/ocr-dataset/icdar2019_matched/labels/train'
files=os.listdir(ic_root)
ic_chars=[]
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
    if range_script!='latin':
        continue
    for line in lines:
        line=line.strip()
        splits=line.split('\t')
        lang=splits[1]
        word=splits[-1]
        for char in word:
            char_code=ord(char)
            # print(char_code,char,'char',lang)
            if char_code>=1000 and char_code<=55141:
                print(char_code,range_lang,char,file)
            if not char in ic_chars:
                ic_chars.append(char)
print(len(ic_chars),'ic_chars')
print(len(dst_char_set),'dst_char_set')
dst_char_set+=ic_chars
dst_char_set=list(set(dst_char_set))
dst_char_set=sorted(dst_char_set)
assert '[s]' not in dst_char_set
assert '[UNK]' not in dst_char_set
dst_char_set=['[s]']+dst_char_set+['[UNK]']
# dst_path='/home/jacobwang/code/azure/text_enh/src_mlt2/ckpt/chartokenizer/char_vocab_mlt.json'
# dstfile=open(dst_path,'w',encoding='utf-8')
dst_file=open('../ckpt/chartokenizer/char_vocab_mlt_latin.json','w',encoding='utf-8')
print(len(dst_char_set),'len(chars)')
print(dst_char_set,'dst_char_set')
json.dump(dst_char_set,dst_file)
