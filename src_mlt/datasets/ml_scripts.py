import numpy as np
import json
langs=['english','italian','german','french', 'arabic', 'hindi','hebrew','greek','thai','russian']
# langs=['english','italian','german','french', 'arabic', 'hindi']

scripts=['latin', 'arabic','hindi','hebrew','greek','cyrillic','thai']
# scripts=['latin', 'arabic','hindi']
lang2script={'english':'latin',
             'italian':'latin',
             'german':'latin',
             'french':'latin',
             'arabic':'arabic',
             'hindi':'hindi',
             'hebrew':'hebrew',
             'greek':'greek',
             'russian':'cyrillic',
             'latin':'latin',
             'thai':'thai'}
script2idx={}
idx2script={}
for idx,script in enumerate(scripts):
    script2idx[script]=idx
    idx2script[idx]=script

charset=json.load(open('ckpt/chartokenizer/char_vocab_mlt.json'))
# charset=json.load(open('ckpt/chartokenizer/char_vocab_mlt_ic19.json'))
char2idx={}
for char in charset:
    char2idx[char]=len(char2idx)
idx2char={}
for char in char2idx:
    charid=char2idx[char]
    idx2char[charid]=char

    
# if __name__=='__main__':
#     # print(idx2char)