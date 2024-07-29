import string
import re
from utils import generate_spatial_rendering_or_logo
from utils import generate_spatial_rendering
from utils import visualize_box
import numpy as np
glyph_texts = ["двржен","éçlameùr","तेवरमों","περνούτ"]
glyph_texts_batch = [["περλφύτ"],["तेवरमों"],["ธนานุบาล"],["äußert"]]
inference_lang_list_batch=[['greek'],["hindi"],["thai"],["german"]]
# Inference
lang2script={
    'english':'latin',
    'italian':'latin',
    'german':'latin',
    'french':'latin',
    'arabic':'arabic',
    'hindi':'hindi',
    'hebrew':'hebrew',
    'greek':'greek',
    'russian':'cyrillic',
    'latin':'latin',
    'thai':'thai',
    'logo':'logo'
}
for gen_idx in range(len(glyph_texts_batch)):
    out=re.sub(r'[a-zA-Z0-9]', '', glyph_texts[0])
    my_punct = list(string.punctuation)
    punct_pattern = re.compile("[" + re.escape("".join(my_punct)) + "]")
    out=re.sub(punct_pattern, '', out)
    # print(out,len(out),glyph_texts[0])    
    glyph_texts=glyph_texts_batch[gen_idx]
    inference_lang_list=inference_lang_list_batch[gen_idx]
    inference_scripts_list=[lang2script[lang]for lang in inference_lang_list]
    coords=np.array([[120, 180,380, 300]])
    coords=coords.astype(np.int32).tolist()
    rendered_whole_images,_,_=generate_spatial_rendering_or_logo(width=512,height=512,words=glyph_texts,dst_coords=coords,lang_list=inference_lang_list)
    rendered_whole_images=visualize_box(rendered_whole_images,coords)
    rendered_whole_images.save('rendered_whole_images{}.jpg'.format(gen_idx))