path='/home/jacobwang/code/azure/text_enh/src_mlt2/ckpt/chartokenizer/char_vocab_mlt.json'
import json
import numpy as np
charset=json.load(open(path))[1:-1]
for i in range(len(charset)):
    if i%5==0:
        print(charset[i],ord(charset[i]))
        # print(charset[i])
    else:
        print(charset[i],ord(charset[i]),end=' ')
        # print(charset[i],end=' ')
print(len(charset),'len(charset)')