import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import sys
os.environ['PYTHONPATH']='/home/twkim/project/azure/refdiffuser/src_mlt'
language='german'
font_path='/home/twkim/project/azure/refdiffuser/src_mlt/ml_fonts/english/GoNotoCurrent.ttf'

def generate_spatial_rendering_ml(width, height, words,dst_coords=None,lang_list=None):
    image = Image.new('RGB', (width, height), (255, 255, 255)) # SAME
    draw = ImageDraw.Draw(image) # SAME
    for idx in range(len(words)):
        lang=lang_list[idx]
        coords=dst_coords[idx] # SAME
        word = words[idx] # SAME
        # Word labeled as Text
        if lang in ['italian','french','spanish','german','english']:
            font_root=os.path.join('../ml_fonts','english') 
        else:
            font_root=os.path.join('../ml_fonts',lang_list[idx])        
        script_color=0
        available_fonts=os.listdir(font_root)
        font_path=os.path.join(font_root,available_fonts[0])
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
            # if lang_list[idx] in ['chinese']:
            #     scaler=0.8
            # elif lang_list[idx] in ['russian','german']:
            #     scaler=1.4
            # elif lang_list[idx] in ['french','greek','thai']:
            #     scaler=1.6
            # elif lang_list[idx] in ['hindi']:
            #     scaler=1.6
            # else:
            #     scaler=1.2 #english

            if lang_list[idx] in ['french','german','english','spanish','italian']:
                scaler=1.4
            elif lang_list[idx] in ['russian']:
                scaler=1.5
            elif lang_list[idx] in ['thai']:
                scaler=1.4
            elif lang_list[idx] in ['greek']:
                scaler=1.5
            elif lang_list[idx] in ['hindi']:
                scaler=3
            elif lang_list[idx] in ['bengali']:
                scaler=1.3
            elif lang_list[idx] in ['korean']:
                print('korean')
                scaler=1.2
            else:
                scaler=1.5
            font_size = int(max(region_width, region_height)*scaler / divider)
            # font_size = int(max(region_width, region_height)*1.4 / len(word))
            font_size=int(font_size)
            font_size=min(min_side,font_size)
            font_size=min(56,font_size)
            if lang_list[idx] in ['korean']:
                font_size=max(28,font_size)
            else:
                font_size=max(34,font_size)


            # font_size=int(font_size)
            # font_size=max(1,font_size)
            # font_size=min(min_side,font_size)
            # font_size=min(56,font_size)
            font = ImageFont.truetype(font_path, font_size)
            text_width, text_height = draw.textsize(word, font=font)
            text_x = x1 + (region_width - text_width) // 2
            text_y = y1 + (region_height - text_height) // 2
            draw.text((text_x, text_y), word, font=font, fill=script_color)
    return image
def visualize_box(image,boxes,chars=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    image=np.array(image)
    if chars is not None:
        for box,char in zip(boxes,chars):
            box=(box.detach().cpu().numpy()*2).astype(np.int32)
            x0,y0,x1,y1=box
            box=(box).reshape(-1,2).astype(np.int32)
            point1=box[0]
            point2=box[1]
            image=cv2.rectangle(image,tuple(point1),tuple(point2),color=(0,255,0),thickness=1)   
            image = cv2.putText(image, char, (x0,y1+5), font, fontScale, (255,0,0), 1, cv2.LINE_AA)
    else:
        for box in boxes:
            if torch.is_tensor(box):
                box=(box.detach().cpu().numpy().astype(np.int32)).reshape(-1,2)
            else:
                box=np.array(box).reshape(-1,2)
            point1=box[0]
            point2=box[1]
            image=cv2.rectangle(image,tuple(point1),tuple(point2),color=(0,255,0),thickness=1)
    image=Image.fromarray(image)
    return image

# glyph_texts_batch = [["ธนานุบาล"],["περλφύτ"],["तेवरमों"],["অইছিল"],["пдвржен"]]
glyph_texts_batch = [["퍼듀대학교"],["菜农们"],["でないと"],["כיבדך"]]
inference_lang_list_batch=[['korean'],['chinese'],["japanese"],["hebrew"]]
# glyph_texts_batch = [["অইছিল"]]
# inference_lang_list_batch=[["bengali"]]
dst_root='renderings_nonlatin'
if os.path.exists(dst_root):
    os.system('rm {} -R'.format(dst_root))
os.makedirs(dst_root,exist_ok=True)
for gen_idx in range(len(glyph_texts_batch)):
    inference_lang_list=inference_lang_list_batch[gen_idx]
    coords=np.array([[100, 145,390, 265]])
    coords=coords.astype(np.int32).tolist()
    for ii in range(3):
        glyph_texts=[(glyph_texts_batch[gen_idx][0]*(ii+1))]
        # print(len(glyph_texts_batch[gen_idx][0]*(ii+1)),glyph_texts)

        spatial_renderings=generate_spatial_rendering_ml(width=512,height=512,words=glyph_texts,dst_coords=coords,lang_list=inference_lang_list)
        spatial_renderings=spatial_renderings.convert('RGB')
        spatial_renderings_drawn=visualize_box(spatial_renderings,coords)
        lang=inference_lang_list[0]
        spatial_renderings_drawn.save('renderings_nonlatin/{}_{}_{}.png'.format(lang,gen_idx,ii))