import re
import os
# langs=['french','german','italian','english']
langs=['english']
fpath='/home/twkim/project/azure/refdiffuser/evaluation/mlt_benchmark/mlt_data/seen_word_sets/raw/english_popular.txt'
dst_path='/home/twkim/project/azure/refdiffuser/evaluation/mlt_benchmark/mlt_data/seen_word_sets/raw/english_special.txt'
dst_file=open(dst_path,'w')
lines=open(fpath).readlines()
for line in lines:
    word=line.strip()
    word_len=len(word)
    if word_len>6 or word_len<3:
        continue
    if re.search(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", word):
        continue
    dst_file.write('{}\n'.format(word))

    
