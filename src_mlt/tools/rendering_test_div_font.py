import os
import string
import re
import numpy as np
from utils import generate_spatial_rendering_or_logo2,FontColor
from PIL import Image, ImageDraw,ImageFont

def generate_spatial_rendering(width, height, words,dst_coords=None,font_path=None,lang_list=None):
    image = Image.new('RGB', (width, height), (255, 255, 255)) # SAME
    draw = ImageDraw.Draw(image) # SAME
    draw.text_kerning = 30.0
    for idx in range(len(words)):
        # print(len(words),'len(words)',len(dst_coords),'len(dst_coords)')
        script_color=0

        coords=dst_coords[idx] # SAME
        word = words[idx] # SAME
        if not word:
            continue
        x1, y1, x2, y2 = np.array(coords).astype(np.int32) # np.min(xs),np.min(ys),np.max(xs),np.max(ys) # SAME
        region_width = x2 - x1
        region_height = y2 - y1
        min_side=min(region_width,region_height) # SAME
        divider=(len(word))
        divider=max(1,divider)
        scaler=1.2
        font_size = int(max(region_width, region_height)*scaler / divider)
        font_size=int(font_size)
        font_size=max(1,font_size)
        font_size=min(min_side,font_size)
        font_size=min(56,font_size)
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(word, font=font)
        text_x = x1 + (region_width - text_width) // 2
        text_y = y1 + (region_height - text_height) // 2
        draw.text((text_x, text_y), word, font=font, fill=script_color)
    return image,np.array(coords),words

coords=np.array([[200, 20,400, 100],
                 [200, 120,400, 200],
                 [200, 220,400, 300],
                 [200, 320,400, 400]
                 ])
glyph_texts = ["brownredfox","~!@#$%^&*","1234567","+_)(*&)"]
font_root='/home/jacobwang/code/azure/text_enh/src_mlt2/ml_fonts/div_fonts'
available_fonts=os.listdir('/home/jacobwang/code/azure/text_enh/src_mlt2/ml_fonts/div_fonts')
dst_root='font_viz2'
os.makedirs(dst_root,exist_ok=True)
for item in available_fonts:
    name=item.split('.')[0]
    fpath=os.path.join(font_root,item)
    lang_list=['english']*len(glyph_texts)
    font_color=FontColor('/home/jacobwang/code/azure/text_enh/src_mlt2/colors_new.cp')
    # width, height, words,dst_coords=None,diversify_font=False,lang_list=None,logo_templates=None,font_color=None
    templates=[None]*len(glyph_texts)
    rend,_,_=generate_spatial_rendering_or_logo2(512,512,glyph_texts,coords,diversify_font=True,lang_list=lang_list,font_color=font_color,logo_templates=templates)
    rend.save(os.path.join(dst_root,name+'.jpg'))