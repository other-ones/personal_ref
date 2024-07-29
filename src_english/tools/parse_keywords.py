import re
import os
import numpy as np
import shutil
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, AutoProcessor
caption_path='/data/twkim/ocr-dataset/mario_eval2/mario_eval2_blip_caption.txt'
lroot='/data/twkim/ocr-dataset/mario_eval2/layouts'
model_name="stabilityai/stable-diffusion-2-1"
tokenizer = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer",
        revision=None,
    )
lines=open(caption_path).readlines()
for line in lines:
    line=line.strip()
    splits=line.split('\t')
    lfile=splits[0]
    lpath=os.path.join(lroot,lfile)
    llines=open(lpath).readlines()
    keywords_list=[]
    for ll in llines:
        ll=ll.strip()
        ll_splits=ll.split('\t')
        keyword=ll_splits[-1]
        keyword=re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039])", "", keyword) 
        keywords_list.append(keyword)



    caption=splits[-1]
    widx_to_crd_idx={}
    caption_splits=caption.split()
    # caption=re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039])", "", caption) 
    hits=[False]*len(keywords_list)
    for widx in range(len(caption_splits)):
        cap_word=caption_splits[widx]
        cap_word_nopunc = re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039])", "", cap_word) 
        if cap_word_nopunc in keywords_list: #keyword
            for crd_idx in range(len(keywords_list)):
                keyword=keywords_list[crd_idx]
                if (not hits[crd_idx]) and (cap_word_nopunc == keyword):
                    widx_to_crd_idx[widx]=crd_idx
                    hits[crd_idx]=True
                    break
    if np.sum(hits)!=len(hits):
        print('Assertion HERE')
        print(keywords_list,'keywords_list')
        print(np.array(keywords_list)[hits],'keywords_list')
        print(caption,'caption')
        print(hits,'hits')
        print(keywords_list,'keywords_list')
        print(lfile ,'layout_file')
        print(np.sum(hits),'np.sum(hits)')
        print(len(hits),'len(hits)')
    assert np.sum(hits)==len(hits),'keyword parsing error'


    #### 5. Parse token index to coordinate index ####

    # FINAL GOAL!!!!
    tidx_to_crd_idx={} #token index to keyword coordinate index
    # FINAL GOAL!!!!

    is_keyword_tokens=[False]*tokenizer.model_max_length #first token is special token
    # index for enumerating token idxs of the tokenized prompt
    tidx=1 #starts from 1 since first token is for the special token
    eot_idx=1 # 1 for start token
    for widx,cap_word in enumerate(caption.split()):
        # convert word to token
        word_token_ids=tokenizer.encode(cap_word,add_special_tokens=False)
        eot_idx+=len(word_token_ids)
        # add_special_tokens: if False do not include "<|startoftext|>" and "<|endoftext|>"
        # e.g., neurips -> [3855,30182] which corresponds to ['neu', 'rips</w>'] tokens
        # this can be confirmed by tokenizer.convert_ids_to_tokens(word_token_ids)
        # the token ids here is not used, but we need to count the number of tokens for each word
        word_token_idxs=[]
        # iterate over token ids of a word
        # e.g., neurips -> [3855,30182]
        num_tokens=len(word_token_ids)
        for _ in range(num_tokens):
            word_token_idxs.append(tidx)
            # token_index to word_index mapping
            # e.g., "word saying 'neurips'" -> token_idx for neu<\w>: 2
            # per_token_word_idxs[2]->2: 2 for word index of neurips
            if widx in widx_to_crd_idx:
                is_keyword_tokens[tidx]=True
                tidx_to_crd_idx[tidx]=widx_to_crd_idx[widx]
            tidx+=1
            if tidx==(len(is_keyword_tokens)-2):
                break
        if tidx==(len(is_keyword_tokens)-2):
                break
        
        # word_index to token_index_list mapping
        # e.g., "word saying 'Neurips'" -> word_idx for neurips: 2
        # NOTE: len(word_token_idxs)==(len(input_ids)-2): two for special tokens
    #### 5. Parse token index to coordinate index ####
    assert not is_keyword_tokens[-1],'last element'
print('no error')





    


