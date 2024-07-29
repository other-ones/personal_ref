import os
import string
import re
import numpy as np
# from utils import generate_spatial_rendering_or_logo,FontColor
from PIL import Image, ImageDraw,ImageFont

def generate_spatial_rendering_or_logo(width, height, words,dst_coords=None,diversify_font=False,lang_list=None,logo_templates=None):
    image = Image.new('RGB', (width, height), (255, 255, 255)) # SAME
    draw = ImageDraw.Draw(image) # SAME
    for idx in range(len(words)):
        lang=lang_list[idx]
        print(lang)
        coords=dst_coords[idx] # SAME
        word = words[idx] # SAME
        if 'logo' in lang:
            # Word labeled as LOGO
            assert logo_templates is not None
            logo_template = logo_templates[idx] # SAME
            assert logo_template is not None
            x1,y1,x3,y3=np.array(coords).astype(np.int32)
            logo_width,logo_height=x3-x1,y3-y1
            logo_width=int(logo_width)
            logo_height=int(logo_height)
            logo_template=logo_template.convert('RGB')
            logo_template=logo_template.resize((logo_width,logo_height))
            image.paste(logo_template,(x1,y1))
        else:
            # Word labeled as Text
            assert logo_templates[idx] is None
            font_root=os.path.join('../ml_fonts',lang_list[idx]) 
            script_color=0
            available_fonts=os.listdir(font_root)
            if diversify_font:
                font_sampled=np.random.choice(available_fonts)
                font_path=os.path.join(font_root,font_sampled)
            else:
                font_path=os.path.join(font_root,available_fonts[0])
            if not word:
                continue
            x1, y1, x2, y2 = np.array(coords).astype(np.int32) # np.min(xs),np.min(ys),np.max(xs),np.max(ys) # SAME
            region_width = x2 - x1
            region_height = y2 - y1
            min_side=min(region_width,region_height) # SAME
            if region_height>(region_width*2): # Vertical Text
                font_size = int(min(region_width, region_height) / (len(word)))
                if lang_list[idx] in ['korean','chinese']:
                    scaler=0.7
                elif lang_list[idx] in ['arabic']:
                    scaler=1.5
                elif lang_list[idx] in ['russian','german']:
                    scaler=1.1
                elif lang_list[idx] in ['french','greek','thai']:
                    scaler=1.3
                elif lang_list[idx] in ['hindi']:
                    scaler=1.3
                else:
                    scaler=0.9
                font_size=font_size*scaler
                font_size=int(font_size)
                font_size=max(1,font_size)
                font_size=min(min_side,font_size)
                font_size=max(5,font_size)
                font = ImageFont.truetype(font_path, font_size)
                text_width, text_height = draw.textsize(word, font=font)
                text_x = x1 + (region_width - text_width) // 2
                text_y = y1 + (region_height - text_height) // 2
                draw.text((text_x, text_y), word, font=font, fill=script_color)
            else: # Horizontal Text
                divider=(len(word))
                divider=max(1,divider)
                if lang_list[idx] in ['korean']:
                    scaler=1.0
                if lang_list[idx] in ['chinese']:
                    scaler=0.8
                elif lang_list[idx] in ['arabic']:
                    scaler=2.0
                elif lang_list[idx] in ['russian','german']:
                    scaler=1.4
                elif lang_list[idx] in ['french','greek','thai']:
                    scaler=1.6
                elif lang_list[idx] in ['hindi']:
                    scaler=1.6
                else:
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

coords=np.array([[100, 220,420, 270],
                #  [200, 120,400, 200],
                #  [200, 220,400, 300],
                #  [200, 320,400, 400]
                 ])
coords_square=np.array([[150, 150,350, 350],
                 ])

glyph_texts = ["CVPR","Midnight","ГоПЬКо","spicy","mcdonalds"]
lang_list=['english','english','russian','english','logo']
logo_templates=[]
for idx,lang in enumerate(lang_list):
    if 'logo' in lang:
        template=Image.open('../logo_templates/{}.jpg'.format(glyph_texts[idx]))
        logo_templates.append(template)
    else:
        logo_templates.append(None)



dst_root='controlnet'
os.makedirs(dst_root,exist_ok=True)
for idx,word in enumerate(glyph_texts):
    lang=lang_list[idx]
    if lang=='logo':
        crd=coords_square
    else:
        crd=coords
    rend,_,_=generate_spatial_rendering_or_logo(512,512,[word],crd,
                                                diversify_font=True,
                                                lang_list=[lang_list[idx]],
                                                logo_templates=[logo_templates[idx]])
    rend.save(os.path.join(dst_root,str(idx)+'.jpg'))