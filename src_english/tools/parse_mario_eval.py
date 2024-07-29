import os
import re
root='/data/dataset/ocr/mario_eval_final/raw/prompts'

def parse_caption(caption):
    # Find all quoted phrases
    quoted_phrases = re.findall(r'"(.*?)"|\'(.*?)\'', caption)
    # Flatten the list of tuples and remove empty strings
    keywords = [kw for pair in quoted_phrases for kw in pair if kw]
    # Replace each quoted phrase with individual quoted words
    all_words=[]
    for phrase in keywords:
        words = phrase.split()
        all_words+=words
        replacement = " ".join(f"'{word}'" for word in words)
        caption = re.sub(rf'(["\']){re.escape(phrase)}\1', replacement, caption)
    

    return caption,all_words
caption='a picture of "zxcv zxcV zxCV" and \'123 123 123\''
parsed,all_words=parse_caption(caption)
flist=['ChineseDrawText.txt','DrawBenchText.txt','DrawTextCreative.txt','LAIONEval4000.txt','OpenLibraryEval500.txt','TMDBEval500.txt']
for ff in flist:
    fpath=os.path.join(root,ff)
    lines=open(fpath).readlines()
    for lidx,line in enumerate(lines):
        line=line.strip()
        parsed,all_words=parse_caption(line)
        if len(all_words)==8:
            print(fpath,lidx)
            # exit()